"""State store for MA-ribbon auto-execution. JSON file with atomic writes,
corruption detection, and append-only history.

Single source of truth for: enabled / halted / ramp-up day / config /
ledger / pending signals / errors. The scanner reads + writes this on
every tick.
"""
from __future__ import annotations
import json
import os
import typing
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any


_STATE_PATH_DEFAULT   = Path("data/state/ma_ribbon_auto_state.json")
_HISTORY_PATH_DEFAULT = Path("data/state/ma_ribbon_auto_state_history.jsonl")
_DAY_SECONDS = 86_400


class StateCorruptError(Exception):
    """Raised when the state file exists but is unreadable or schema-broken.
    The scanner halts on this — we never silently zero out a financial ledger.
    """


@dataclass
class UniverseFilter:
    min_volume_usd: float = 1_000_000.0
    product_types: list[str] = field(default_factory=lambda: ["USDT-FUTURES"])


@dataclass
class FetchCfg:
    pages_per_symbol: int = 30
    concurrency: int = 12


@dataclass
class AutoStateConfig:
    universe_filter: UniverseFilter = field(default_factory=UniverseFilter)
    tfs: list[str] = field(default_factory=lambda: ["5m", "15m", "1h", "4h"])
    directions: list[str] = field(default_factory=lambda: ["long", "short"])
    ribbon_buffer_pct: dict[str, float] = field(default_factory=lambda: {
        "5m": 0.01, "15m": 0.04, "1h": 0.07, "4h": 0.10,
    })
    layer_risk_pct: dict[str, float] = field(default_factory=lambda: {
        "LV1": 0.001, "LV2": 0.0025, "LV3": 0.005, "LV4": 0.02,
    })
    max_concurrent_orders: int = 25
    per_symbol_risk_cap_pct: float = 0.02
    dd_halt_pct: float = 0.15
    strategy_capital_usd: float = 0.0
    fetch_cfg: FetchCfg = field(default_factory=FetchCfg)


@dataclass
class Ledger:
    trades: list[dict[str, Any]]      = field(default_factory=list)
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    realized_pnl_usd_cumulative: float = 0.0


@dataclass
class AutoState:
    enabled: bool = False
    halted: bool = False
    halt_reason: str | None = None
    locked_until_utc: int | None = None
    first_enabled_at_utc: int | None = None
    config: AutoStateConfig = field(default_factory=AutoStateConfig)
    ledger: Ledger = field(default_factory=Ledger)
    last_processed_bar_ts: dict[str, int] = field(default_factory=dict)
    pending_signals: list[dict[str, Any]] = field(default_factory=list)
    errors_recent: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def default() -> "AutoState":
        return AutoState()


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in obj.__dataclass_fields__.values()}
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Tolerant constructor: missing fields fall back to defaults; extra fields
    raise StateCorruptError to surface schema drift.

    NOTE: with `from __future__ import annotations` active, dataclass field types
    are stored as strings. We resolve them with typing.get_type_hints which
    evaluates the strings against the class's module namespace.
    """
    if not is_dataclass(cls):
        return data
    valid = {f.name: f for f in cls.__dataclass_fields__.values()}
    extra = set(data.keys()) - set(valid.keys())
    if extra:
        raise StateCorruptError(f"unrecognised fields in {cls.__name__}: {sorted(extra)}")
    try:
        resolved_types = typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001 — fallback to no-resolution path
        resolved_types = {}
    kwargs: dict[str, Any] = {}
    for name, fld in valid.items():
        if name not in data:
            continue
        value = data[name]
        ftype = resolved_types.get(name, fld.type)
        # If ftype is a dataclass, recurse; else use value as-is.
        if isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[name] = _from_dict(ftype, value or {})
        else:
            kwargs[name] = value
    return cls(**kwargs)


def save_state(
    state: AutoState,
    path: Path | None = None,
    history_path: Path | None = None,
) -> None:
    p = Path(path or _STATE_PATH_DEFAULT)
    h = Path(history_path or _HISTORY_PATH_DEFAULT)
    p.parent.mkdir(parents=True, exist_ok=True)
    h.parent.mkdir(parents=True, exist_ok=True)

    payload = _to_dict(state)
    blob = json.dumps(payload, indent=2, sort_keys=False).encode("utf-8")

    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)

    with h.open("ab") as f:
        f.write(json.dumps(payload).encode("utf-8") + b"\n")


def load_state(path: Path | None = None) -> AutoState:
    p = Path(path or _STATE_PATH_DEFAULT)
    if not p.exists():
        return AutoState.default()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StateCorruptError(f"json parse failed: {exc}") from exc
    return _from_dict(AutoState, raw)


def current_ramp_cap_pct(state: AutoState, now_utc: int) -> float:
    """Day 1 = 2 %, +1 % per 24 h, capped at 15 %.
    If never enabled, returns 0 (no spawning allowed)."""
    if state.first_enabled_at_utc is None:
        return 0.0
    days = (now_utc - state.first_enabled_at_utc) // _DAY_SECONDS
    cap = 0.02 + 0.01 * days
    return min(cap, 0.15)
