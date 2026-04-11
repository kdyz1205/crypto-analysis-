from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.config import PROJECT_ROOT
from ..data_service import get_ohlcv_with_df
from ..drawings import augment_snapshot_with_manual_signals
from ..history_coverage import build_analysis_history
from ..execution import PaperExecutionConfig, PaperExecutionEngine
from ..execution.live_adapter import LiveExecutionAdapter
from ..execution.live_engine import LiveBridgeConfig, LiveExecutionEngine
from ..execution.types import (
    KillSwitchState,
    OrderIntent,
    PaperAccountSummary,
    PaperExecutionState,
    PaperFill,
    PaperOrder,
    PaperPosition,
    dataclass_to_dict,
    stable_execution_id,
)
from ..strategy import ReplayResult, StrategyConfig, apply_strategy_overrides, build_latest_snapshot
from ..runtime.types import (
    RuntimeInstanceConfig,
    RuntimeInstanceRecord,
    RuntimeInstanceStatus,
    RuntimeMode,
    RuntimeStrategyConfig,
)

DATA_DIR = PROJECT_ROOT / "data"
RUNTIME_STORE_PATH = DATA_DIR / "subaccount_runtime_instances.json"
RUNTIME_EVENT_LOG_PATH = DATA_DIR / "subaccount_runtime_events.jsonl"
VALID_INTERVALS = {"1m", "3m", "5m", "15m", "1h", "4h", "1d", "1w"}


class SubaccountRuntimeManager:
    def __init__(
        self,
        *,
        store_path: Path | None = None,
        event_log_path: Path | None = None,
        adapter_provider=None,
    ) -> None:
        self.store_path = store_path or RUNTIME_STORE_PATH
        self.event_log_path = event_log_path or RUNTIME_EVENT_LOG_PATH
        self.adapter_provider = adapter_provider or (lambda: LiveExecutionAdapter())
        self._records: dict[str, RuntimeInstanceRecord] = {}
        self._paper_engines: dict[str, PaperExecutionEngine] = {}
        self._live_engines: dict[str, LiveExecutionEngine] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._loaded = False

    async def startup(self) -> None:
        self._load_from_disk()
        for instance_id, record in self._records.items():
            if record.status.runtime_state == "running":
                if record.config.auto_restart_on_boot:
                    self._spawn_loop(instance_id)
                else:
                    record.status.runtime_state = "stopped"
                    record.status.last_runtime_note = "recovered_stopped"
        self._persist()

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()
        self._persist()

    def list_instances(self) -> list[RuntimeInstanceRecord]:
        self._ensure_loaded()
        return sorted(self._records.values(), key=lambda record: record.config.label.lower())

    def get_instance(self, instance_id: str) -> RuntimeInstanceRecord:
        self._ensure_loaded()
        record = self._records.get(instance_id)
        if record is None:
            raise KeyError(instance_id)
        return record

    def create_instance(
        self,
        *,
        label: str,
        symbol: str,
        timeframe: str,
        subaccount_label: str = "",
        history_mode: str = "fast_window",
        analysis_bars: int = 500,
        days: int = 365,
        tick_interval_seconds: int = 60,
        auto_restart_on_boot: bool = False,
        live_mode: RuntimeMode = "disabled",
        auto_live_preview: bool = True,
        auto_live_submit: bool = False,
        notes: str = "",
        paper_config: PaperExecutionConfig | None = None,
        strategy_config: RuntimeStrategyConfig | None = None,
    ) -> RuntimeInstanceRecord:
        self._ensure_loaded()
        normalized_symbol = _normalize_symbol(symbol)
        normalized_timeframe = _normalize_interval(timeframe)
        instance_id = stable_execution_id("runtime", normalized_symbol, normalized_timeframe, label, datetime.now(timezone.utc).timestamp())
        strategy_config = _normalize_strategy_config(strategy_config or RuntimeStrategyConfig())
        config = RuntimeInstanceConfig(
            instance_id=instance_id,
            label=label.strip() or instance_id,
            symbol=normalized_symbol,
            timeframe=normalized_timeframe,
            subaccount_label=subaccount_label.strip(),
            history_mode=_normalize_history_mode(history_mode),
            analysis_bars=int(analysis_bars),
            days=int(days),
            tick_interval_seconds=max(5, int(tick_interval_seconds)),
            auto_restart_on_boot=bool(auto_restart_on_boot),
            live_mode=live_mode,
            auto_live_preview=bool(auto_live_preview),
            auto_live_submit=bool(auto_live_submit),
            notes=notes,
            paper_config=paper_config or PaperExecutionConfig(),
            strategy_config=strategy_config,
        )
        paper_engine = PaperExecutionEngine(config=config.paper_config)
        live_engine = LiveExecutionEngine(adapter_provider=self.adapter_provider, config=LiveBridgeConfig.from_env())
        status = RuntimeInstanceStatus(paper_state=paper_engine.get_state())
        record = RuntimeInstanceRecord(config=config, status=status)
        self._records[instance_id] = record
        self._paper_engines[instance_id] = paper_engine
        self._live_engines[instance_id] = live_engine
        self._locks[instance_id] = asyncio.Lock()
        self._persist()
        self._append_event(instance_id, "instance.created", {"label": config.label, "symbol": config.symbol, "timeframe": config.timeframe})
        return record

    def update_instance(self, instance_id: str, **changes) -> RuntimeInstanceRecord:
        self._ensure_loaded()
        record = self.get_instance(instance_id)
        config = record.config

        if "symbol" in changes and changes["symbol"] is not None:
            changes["symbol"] = _normalize_symbol(changes["symbol"])
        if "timeframe" in changes and changes["timeframe"] is not None:
            changes["timeframe"] = _normalize_interval(changes["timeframe"])
        if "history_mode" in changes and changes["history_mode"] is not None:
            changes["history_mode"] = _normalize_history_mode(changes["history_mode"])
        if "tick_interval_seconds" in changes and changes["tick_interval_seconds"] is not None:
            changes["tick_interval_seconds"] = max(5, int(changes["tick_interval_seconds"]))

        paper_cfg_changes = changes.pop("paper_config", None)
        if paper_cfg_changes:
            config = replace(config, paper_config=replace(config.paper_config, **paper_cfg_changes))
        strategy_cfg_changes = changes.pop("strategy_config", None)
        if strategy_cfg_changes:
            config = replace(config, strategy_config=_normalize_strategy_config(replace(config.strategy_config, **strategy_cfg_changes)))

        config = replace(config, **{key: value for key, value in changes.items() if value is not None})
        record.config = config
        self._paper_engines[instance_id].update_config(**dataclass_to_dict(config.paper_config))
        self._persist()
        self._append_event(instance_id, "instance.updated", {"changes": dataclass_to_dict(changes), "paper_config": dataclass_to_dict(paper_cfg_changes or {})})
        return record

    def delete_instance(self, instance_id: str) -> None:
        self._ensure_loaded()
        self.stop_instance(instance_id, reason="deleted")
        self._records.pop(instance_id, None)
        self._paper_engines.pop(instance_id, None)
        self._live_engines.pop(instance_id, None)
        self._locks.pop(instance_id, None)
        self._persist()
        self._append_event(instance_id, "instance.deleted", {})

    def start_instance(self, instance_id: str) -> RuntimeInstanceRecord:
        self._ensure_loaded()
        record = self.get_instance(instance_id)
        record.status.runtime_state = "running"
        record.status.last_error = ""
        record.status.last_runtime_note = "manual_start"
        self._spawn_loop(instance_id)
        self._persist()
        self._append_event(instance_id, "instance.started", {})
        return record

    def stop_instance(self, instance_id: str, *, reason: str = "manual_stop") -> RuntimeInstanceRecord:
        self._ensure_loaded()
        record = self.get_instance(instance_id)
        task = self._tasks.pop(instance_id, None)
        if task is not None:
            task.cancel()
        record.status.runtime_state = "stopped"
        record.status.last_runtime_note = reason
        self._persist()
        self._append_event(instance_id, "instance.stopped", {"reason": reason})
        return record

    def set_instance_kill_switch(self, instance_id: str, blocked: bool, reason: str = "") -> RuntimeInstanceRecord:
        self._ensure_loaded()
        record = self.get_instance(instance_id)
        engine = self._paper_engines[instance_id]
        engine.set_manual_kill_switch(blocked, reason)
        record.status.paper_state = engine.get_state()
        record.status.last_runtime_note = "kill_switch_on" if blocked else "kill_switch_off"
        self._persist()
        self._append_event(instance_id, "instance.kill_switch", {"blocked": blocked, "reason": reason})
        return record

    async def tick_instance(self, instance_id: str, *, bar_index: int | None = None) -> RuntimeInstanceRecord:
        self._ensure_loaded()
        record = self.get_instance(instance_id)
        engine = self._paper_engines[instance_id]
        live_engine = self._live_engines[instance_id]
        lock = self._locks.setdefault(instance_id, asyncio.Lock())

        async with lock:
            try:
                candles_df, strategy_cfg, history = await _load_runtime_inputs(record.config)
                stream_key = f"{record.config.symbol}:{record.config.timeframe}"
                last_processed = engine.last_processed_bar_by_stream.get(stream_key, -1)
                target_bar = (last_processed + 1) if bar_index is None else bar_index
                max_index = len(candles_df) - 1
                if max_index < 0:
                    raise ValueError("runtime_input_empty")
                if target_bar > max_index:
                    target_bar = max_index
                if target_bar >= last_processed + 1:
                    replay_result, snapshot_offset = _build_step_replay_result(
                        candles_df,
                        strategy_cfg,
                        record.config.symbol,
                        record.config.timeframe,
                        start_bar=max(0, last_processed + 1),
                        end_bar=target_bar,
                        enabled_trigger_modes=record.config.strategy_config.enabled_trigger_modes,
                        strategy_window_bars=record.config.strategy_config.window_bars,
                    )
                    engine.step(
                        record.config.symbol,
                        record.config.timeframe,
                        candles_df,
                        replay_result,
                        bar_index=bar_index,
                        snapshot_offset=snapshot_offset,
                    )

                if record.config.live_mode != "disabled":
                    await self._process_live_bridge(record, live_engine)

                record.status.paper_state = engine.get_state()
                record.status.live_engine_state = live_engine.export_state()
                record.status.last_tick_at = _utc_iso_now()
                record.status.last_processed_bar = record.status.paper_state.account.last_processed_bar_by_stream.get(stream_key)
                record.status.last_error = ""
                record.status.last_history = history
                if record.status.runtime_state != "running":
                    record.status.runtime_state = "stopped"
                record.status.paper_current_day = engine.current_day
                self._persist()
                self._append_event(
                    instance_id,
                    "instance.ticked",
                    {
                        "stream": stream_key,
                        "last_processed_bar": record.status.last_processed_bar,
                        "live_mode": record.config.live_mode,
                    },
                )
                return record
            except Exception as exc:
                record.status.runtime_state = "blocked"
                record.status.last_error = str(exc)
                record.status.last_runtime_note = "tick_failed"
                record.status.last_tick_at = _utc_iso_now()
                self._persist()
                self._append_event(instance_id, "instance.error", {"error": str(exc)})
                raise

    def get_events(self, *, instance_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if not self.event_log_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.event_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if instance_id and payload.get("instance_id") != instance_id:
                    continue
                events.append(payload)
        return events[-limit:]

    async def _process_live_bridge(
        self,
        record: RuntimeInstanceRecord,
        live_engine: LiveExecutionEngine,
    ) -> None:
        mode = record.config.live_mode
        if mode == "disabled":
            return

        reconciliation = live_engine.reconciliation_by_mode.get(mode)
        if reconciliation is None:
            reconciliation = await live_engine.reconcile_startup(mode)
            self._append_event(record.config.instance_id, "live.reconcile", {"mode": mode, "ok": reconciliation.get("ok"), "reason": reconciliation.get("reason", "")})

        intents = [
            intent for intent in self._paper_engines[record.config.instance_id].order_manager.get_intents()
            if intent.status in {"approved", "submitted"}
        ]
        intents.sort(key=lambda intent: (intent.created_at_bar, intent.signal_id))
        for intent in intents:
            preview = await live_engine.preview_live_submission(intent, mode=mode)
            record.status.last_live_result = preview
            if record.config.auto_live_preview:
                self._append_event(
                    record.config.instance_id,
                    "live.preview",
                    {
                        "mode": mode,
                        "intent_id": intent.order_intent_id,
                        "ok": preview.get("ok"),
                        "reason": preview.get("reason", ""),
                    },
                )
            if preview.get("ok") and record.config.auto_live_submit:
                submitted = await live_engine.execute_live_submission(intent, mode=mode, confirm=True)
                record.status.last_live_result = submitted
                self._append_event(
                    record.config.instance_id,
                    "live.submit",
                    {
                        "mode": mode,
                        "intent_id": intent.order_intent_id,
                        "ok": submitted.get("ok"),
                        "reason": submitted.get("reason", ""),
                        "exchange_order_id": submitted.get("exchange_order_id", ""),
                    },
                )

    def _spawn_loop(self, instance_id: str) -> None:
        existing = self._tasks.get(instance_id)
        if existing is not None and not existing.done():
            return

        async def _runner() -> None:
            while True:
                record = self.get_instance(instance_id)
                if record.status.runtime_state != "running":
                    return
                await self.tick_instance(instance_id)
                await asyncio.sleep(record.config.tick_interval_seconds)

        self._tasks[instance_id] = asyncio.create_task(_runner())

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        self._records.clear()
        self._paper_engines.clear()
        self._live_engines.clear()
        self._locks.clear()
        if self.store_path.exists():
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
            for item in payload.get("instances", []):
                record = _runtime_record_from_dict(item)
                self._records[record.config.instance_id] = record
                paper_engine = PaperExecutionEngine(config=record.config.paper_config)
                if record.status.paper_state is not None:
                    paper_engine.load_state(record.status.paper_state, current_day=record.status.paper_current_day)
                self._paper_engines[record.config.instance_id] = paper_engine
                live_engine = LiveExecutionEngine(adapter_provider=self.adapter_provider, config=LiveBridgeConfig.from_env())
                live_engine.load_state(record.status.live_engine_state)
                self._live_engines[record.config.instance_id] = live_engine
                self._locks[record.config.instance_id] = asyncio.Lock()
        self._loaded = True

    def _persist(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "instances": [dataclass_to_dict(record) for record in self._records.values()],
        }
        self.store_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _append_event(self, instance_id: str, event_type: str, payload: dict[str, Any]) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": _utc_iso_now(),
            "instance_id": instance_id,
            "event_type": event_type,
            "payload": payload,
        }
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")


async def _load_runtime_inputs(config: RuntimeInstanceConfig) -> tuple[pd.DataFrame, StrategyConfig, dict[str, Any]]:
    polars_df, market_payload = await get_ohlcv_with_df(
        config.symbol,
        config.timeframe,
        None,
        config.days,
        history_mode=config.history_mode,
        include_price_precision=True,
        include_render_payload=False,
    )
    if polars_df is None or polars_df.is_empty():
        raise ValueError(f"No data for {config.symbol} {config.timeframe}")

    candles_df = _standardize_strategy_candles(polars_df)
    if len(candles_df) > config.analysis_bars:
        candles_df = candles_df.iloc[-config.analysis_bars:].reset_index(drop=True)

    price_precision = market_payload.get("pricePrecision") if isinstance(market_payload, dict) else None
    strategy_cfg = _config_with_market_precision(StrategyConfig(), price_precision)
    history = build_analysis_history(market_payload, candles_df)
    return candles_df, apply_strategy_overrides(
        strategy_cfg,
        lookback_bars=config.strategy_config.lookback_bars,
        min_touches=config.strategy_config.min_touches,
        confirm_threshold=config.strategy_config.confirm_threshold,
        score_threshold=config.strategy_config.score_threshold,
        rr_target=config.strategy_config.rr_target,
    ), history


def _standardize_strategy_candles(polars_df) -> pd.DataFrame:
    pdf = polars_df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    pdf["timestamp"] = pdf["timestamp"].map(lambda value: int(pd.Timestamp(value).timestamp()))
    for column in ("open", "high", "low", "close", "volume"):
        pdf[column] = pd.to_numeric(pdf[column], errors="raise")
    return pdf[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _config_with_market_precision(config: StrategyConfig, price_precision: int | None) -> StrategyConfig:
    if price_precision is None:
        return config
    tick_size = 1.0 if int(price_precision) <= 0 else float(10 ** (-int(price_precision)))
    return replace(config, tick_size=tick_size)


def _build_step_replay_result(
    candles_df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    symbol: str,
    interval: str,
    *,
    start_bar: int,
    end_bar: int,
    enabled_trigger_modes: tuple[str, ...] | list[str] | None = None,
    strategy_window_bars: int | None = None,
) -> tuple[ReplayResult, int]:
    if end_bar < start_bar:
        return ReplayResult(symbol=symbol, timeframe=interval, snapshots=tuple()), start_bar
    snapshots = []
    for current_bar in range(start_bar, end_bar + 1):
        prefix_start = max(0, current_bar - strategy_window_bars + 1) if strategy_window_bars else 0
        prefix = candles_df.iloc[prefix_start : current_bar + 1].reset_index(drop=True)
        snapshot = build_latest_snapshot(
            prefix,
            strategy_cfg,
            symbol=symbol,
            timeframe=interval,
            enabled_trigger_modes=enabled_trigger_modes,
        )
        snapshot = augment_snapshot_with_manual_signals(
            snapshot,
            prefix,
            strategy_cfg,
            symbol=symbol,
            timeframe=interval,
            enabled_trigger_modes=enabled_trigger_modes,
        )
        snapshots.append(snapshot)
    return ReplayResult(symbol=symbol, timeframe=interval, snapshots=tuple(snapshots)), start_bar


def _runtime_record_from_dict(payload: dict[str, Any]) -> RuntimeInstanceRecord:
    config_payload = dict(payload.get("config") or {})
    paper_config_payload = dict(config_payload.get("paper_config") or {})
    config = RuntimeInstanceConfig(
        instance_id=config_payload["instance_id"],
        label=config_payload["label"],
        symbol=config_payload["symbol"],
        timeframe=config_payload["timeframe"],
        subaccount_label=config_payload.get("subaccount_label", ""),
        history_mode=config_payload.get("history_mode", "fast_window"),
        analysis_bars=int(config_payload.get("analysis_bars", 500)),
        days=int(config_payload.get("days", 365)),
        tick_interval_seconds=int(config_payload.get("tick_interval_seconds", 60)),
        auto_restart_on_boot=bool(config_payload.get("auto_restart_on_boot", False)),
        live_mode=config_payload.get("live_mode", "disabled"),
        auto_live_preview=bool(config_payload.get("auto_live_preview", True)),
        auto_live_submit=bool(config_payload.get("auto_live_submit", False)),
        notes=config_payload.get("notes", ""),
        paper_config=PaperExecutionConfig(**paper_config_payload),
        strategy_config=_normalize_strategy_config(RuntimeStrategyConfig(**dict(config_payload.get("strategy_config") or {}))),
    )
    status_payload = dict(payload.get("status") or {})
    paper_state_payload = status_payload.get("paper_state")
    status = RuntimeInstanceStatus(
        runtime_state=status_payload.get("runtime_state", "stopped"),
        last_tick_at=status_payload.get("last_tick_at"),
        last_processed_bar=status_payload.get("last_processed_bar"),
        last_error=status_payload.get("last_error", ""),
        last_runtime_note=status_payload.get("last_runtime_note", ""),
        last_live_result=status_payload.get("last_live_result"),
        paper_current_day=status_payload.get("paper_current_day"),
        last_history=status_payload.get("last_history"),
        paper_state=_paper_state_from_dict(paper_state_payload) if paper_state_payload else None,
        live_engine_state=status_payload.get("live_engine_state"),
    )
    return RuntimeInstanceRecord(config=config, status=status)


def _paper_state_from_dict(payload: dict[str, Any]) -> PaperExecutionState:
    return PaperExecutionState(
        config=PaperExecutionConfig(**dict(payload.get("config") or {})),
        account=PaperAccountSummary(**dict(payload.get("account") or {})),
        kill_switch=KillSwitchState(**dict(payload.get("kill_switch") or {})),
        intents=[OrderIntent(**item) for item in payload.get("intents", [])],
        open_orders=[PaperOrder(**item) for item in payload.get("open_orders", [])],
        open_positions=[PaperPosition(**item) for item in payload.get("open_positions", [])],
        recent_fills=[PaperFill(**item) for item in payload.get("recent_fills", [])],
        recent_closed_positions=[PaperPosition(**item) for item in payload.get("recent_closed_positions", [])],
        cooldowns=dict(payload.get("cooldowns") or {}),
    )


def _normalize_symbol(symbol: str) -> str:
    normalized = (symbol or "").upper().replace("/", "").strip()
    if not normalized:
        raise ValueError("symbol is required")
    return normalized if normalized.endswith("USDT") else f"{normalized}USDT"


def _normalize_interval(interval: str) -> str:
    normalized = (interval or "").strip()
    if normalized not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of: {sorted(VALID_INTERVALS)}")
    return normalized


def _normalize_history_mode(history_mode: str) -> str:
    normalized = (history_mode or "fast_window").strip().lower()
    if normalized not in {"fast_window", "full_history"}:
        raise ValueError("history_mode must be one of: fast_window, full_history")
    return normalized


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_strategy_config(config: RuntimeStrategyConfig) -> RuntimeStrategyConfig:
    return replace(
        config,
        enabled_trigger_modes=tuple(config.enabled_trigger_modes or ("pre_limit",)),
    )


__all__ = ["SubaccountRuntimeManager"]
