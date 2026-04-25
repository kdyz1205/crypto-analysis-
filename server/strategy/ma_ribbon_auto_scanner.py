"""MA-ribbon live auto-execution scanner.

Tick every 60 s:
  1. load state
  2. enforce gates (Task 9: enabled / halted / lock / DD / ramp / 25 cap)
  3. fetch live OHLCV for universe (Task 16)
  4. detect new bull/bear formations (Task 7)
  5. spawn LV1 + register LV2/LV3/LV4 pending (this Task 8)
  6. fire any due LV2/LV3/LV4 pending (this Task 8)
  7. save state

Tasks 9, 10, 16 extend this file.
"""
from __future__ import annotations
import logging
from typing import Any

from server.strategy.ma_ribbon_auto_state import AutoState
from server.strategy.ma_ribbon_auto_adapter import Phase1Signal


_LOG = logging.getLogger(__name__)


_TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
_HIGHER_LAYERS_FOR_LV1: dict[str, str] = {"LV2": "15m", "LV3": "1h", "LV4": "4h"}


def _next_bar_close_after(ts: int, tf_seconds: int) -> int:
    """Smallest k * tf_seconds strictly > ts."""
    return (ts // tf_seconds + 1) * tf_seconds


def register_pending_higher_layers(
    sig: Phase1Signal,
    state: AutoState,
    now_utc: int,
) -> None:
    """Add this signal's LV2/LV3/LV4 to state.pending_signals.
    Caller has already spawned LV1 separately."""
    pending_layers: list[dict[str, Any]] = []
    for layer, tf in _HIGHER_LAYERS_FOR_LV1.items():
        eta = _next_bar_close_after(sig.signal_bar_ts, _TF_SECONDS[tf])
        pending_layers.append({
            "layer": layer,
            "tf": tf,
            "trigger_at_bar_close_after_ts": eta,
        })
    state.pending_signals.append({
        "signal_id": sig.signal_id,
        "symbol": sig.symbol,
        "direction": sig.direction,
        "spawned_layers": ["LV1"],
        "pending_layers": pending_layers,
        "ema21_at_signal": sig.ema21_at_signal,
        "signal_bar_ts": sig.signal_bar_ts,
        "registered_at_utc": now_utc,
    })


def ready_layers_at(state: AutoState, now_utc: int) -> list[dict[str, Any]]:
    """Return list of {signal, layer, tf} entries whose ETA has passed.
    Caller is responsible for verifying ribbon-still-aligned at the layer's TF
    before spawning, and for moving the layer from pending_layers to
    spawned_layers via remove_layer_from_pending after spawn."""
    out = []
    for s in state.pending_signals:
        for layer in s["pending_layers"]:
            if layer["trigger_at_bar_close_after_ts"] <= now_utc:
                out.append({"signal": s, "layer": layer["layer"], "tf": layer["tf"]})
    return out


def remove_layer_from_pending(state: AutoState, signal_id: str, layer: str) -> None:
    for s in state.pending_signals:
        if s["signal_id"] == signal_id:
            s["pending_layers"] = [
                l for l in s["pending_layers"] if l["layer"] != layer
            ]
            if layer not in s["spawned_layers"]:
                s["spawned_layers"].append(layer)
            break


def expire_orphans(state: AutoState, now_utc: int, max_age_seconds: int = 86400) -> None:
    state.pending_signals = [
        s for s in state.pending_signals
        if (now_utc - s.get("registered_at_utc", 0)) < max_age_seconds
    ]
