from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .live_adapter import LiveExecutionAdapter, LiveMode
from .types import OrderIntent

EligibleIntentStatus = Literal["approved", "submitted"]


@dataclass(slots=True)
class LiveBridgeConfig:
    allowed_symbols: tuple[str, ...] = ("BTCUSDT",)
    allowed_timeframes: tuple[str, ...] = ("1h",)
    allowed_trigger_modes: tuple[str, ...] = ("rejection", "failed_breakout")
    max_live_positions: int = 1
    default_mode: LiveMode = "demo"
    max_live_notional: float = 100.0
    reconciliation_max_age_seconds: int = 300


class LiveExecutionEngine:
    def __init__(
        self,
        adapter_provider: Callable[[], LiveExecutionAdapter],
        config: LiveBridgeConfig | None = None,
    ) -> None:
        self._adapter_provider = adapter_provider
        self.config = config or LiveBridgeConfig()
        self.last_preview_result: dict[str, Any] | None = None
        self.last_submission_result: dict[str, Any] | None = None
        self.reconciliation_by_mode: dict[LiveMode, dict[str, Any] | None] = {"demo": None, "live": None}
        self.submissions_by_mode: dict[LiveMode, dict[str, dict[str, Any]]] = {"demo": {}, "live": {}}

    def get_status(self) -> dict[str, Any]:
        adapter = self._adapter_provider()
        enabled_flags = self._enabled_flags()
        return {
            "enabled_flags": enabled_flags,
            "default_mode": self.config.default_mode,
            "api_key_ready": adapter.has_api_keys(),
            "whitelist": {
                "symbols": list(self.config.allowed_symbols),
                "timeframes": list(self.config.allowed_timeframes),
                "trigger_modes": list(self.config.allowed_trigger_modes),
            },
            "limits": {
                "max_live_positions": self.config.max_live_positions,
                "max_live_notional": self.config.max_live_notional,
                "reconciliation_max_age_seconds": self.config.reconciliation_max_age_seconds,
            },
            "reconciliation": {mode: report for mode, report in self.reconciliation_by_mode.items()},
            "reconciliation_required_by_mode": {
                mode: self.reconciliation_by_mode.get(mode) is None for mode in self.reconciliation_by_mode
            },
            "submitted_intent_ids_by_mode": {
                mode: sorted(records.keys()) for mode, records in self.submissions_by_mode.items()
            },
            "last_preview_result": self.last_preview_result,
            "last_submission_result": self.last_submission_result,
            "blocked_reason": self._blocked_reason(enabled_flags, adapter.has_api_keys()),
        }

    async def preview_live_submission(self, intent: OrderIntent, mode: LiveMode | None = None) -> dict[str, Any]:
        resolved_mode = mode or self.config.default_mode
        result = await self._build_preview(intent, resolved_mode)
        self.last_preview_result = result
        return result

    async def execute_live_submission(
        self,
        intent: OrderIntent,
        *,
        mode: LiveMode,
        confirm: bool,
    ) -> dict[str, Any]:
        if not confirm:
            rejected = self._result_base(intent, mode)
            rejected.update({"ok": False, "blocked": True, "reason": "confirm_required"})
            self.last_submission_result = rejected
            return rejected

        existing_submission = self.submissions_by_mode[mode].get(intent.order_intent_id)
        if existing_submission is not None:
            replayed = dict(existing_submission)
            replayed["idempotent_replay"] = True
            self.last_submission_result = replayed
            return replayed

        preview = await self._build_preview(intent, mode)
        self.last_preview_result = preview
        if preview.get("blocked"):
            self.last_submission_result = preview
            return preview

        adapter = self._adapter_provider()
        submitted = await adapter.submit_live_entry(intent, mode)
        result = self._result_base(intent, mode)
        if submitted.get("ok"):
            result.update(
                {
                    "ok": True,
                    "blocked": False,
                    "reason": "",
                    "exchange_order_id": submitted.get("exchange_order_id", ""),
                    "submitted_price": submitted.get("submitted_price", 0.0),
                    "submitted_notional": submitted.get("submitted_notional", preview["would_submit_notional"]),
                    "exchange_response_excerpt": submitted.get("exchange_response_excerpt"),
                }
            )
        else:
            result.update(
                {
                    "ok": False,
                    "blocked": True,
                    "reason": submitted.get("reason", "live_submit_failed"),
                    "exchange_order_id": submitted.get("exchange_order_id", ""),
                    "submitted_price": submitted.get("submitted_price", 0.0),
                    "submitted_notional": submitted.get("submitted_notional", preview["would_submit_notional"]),
                    "exchange_response_excerpt": submitted.get("exchange_response_excerpt"),
                }
            )
        if result.get("ok"):
            self.submissions_by_mode[mode][intent.order_intent_id] = dict(result)
        self.last_submission_result = result
        return result

    async def close_live_position(self, symbol: str, *, mode: LiveMode, confirm: bool) -> dict[str, Any]:
        normalized_symbol = symbol.upper().replace("/", "")
        if not confirm:
            return {
                "ok": False,
                "blocked": True,
                "reason": "confirm_required",
                "mode": mode,
                "symbol": normalized_symbol,
                "exchange_order_id": "",
                "exchange_response_excerpt": None,
            }
        gating_reasons = self._gating_reasons_for_close(mode)
        if gating_reasons:
            return {
                "ok": False,
                "blocked": True,
                "reason": gating_reasons[0],
                "blocking_reasons": gating_reasons,
                "mode": mode,
                "symbol": normalized_symbol,
                "exchange_order_id": "",
                "exchange_response_excerpt": None,
            }

        adapter = self._adapter_provider()
        result = await adapter.submit_live_close(normalized_symbol, mode)
        self.last_submission_result = result
        return result

    async def reconcile_startup(self, mode: LiveMode) -> dict[str, Any]:
        adapter = self._adapter_provider()
        if not adapter.has_api_keys():
            report = {
                "ok": False,
                "blocked": True,
                "mode": mode,
                "reason": "api_keys_missing",
                "positions": [],
                "pending_orders": [],
                "exchange_response_excerpt": None,
            }
            self.reconciliation_by_mode[mode] = report
            return report

        report = await adapter.reconcile_live_state(mode)
        report["checked_at"] = int(time.time())
        self.reconciliation_by_mode[mode] = report
        return report

    async def _build_preview(self, intent: OrderIntent, mode: LiveMode) -> dict[str, Any]:
        reconciliation = self.reconciliation_by_mode.get(mode)
        notional = float(intent.entry_price) * float(intent.quantity)
        reasons = self._gating_reasons_for_intent(intent, mode, reconciliation)
        result = self._result_base(intent, mode)
        result.update(
            {
                "ok": len(reasons) == 0,
                "blocked": len(reasons) > 0,
                "reason": reasons[0] if reasons else "",
                "blocking_reasons": reasons,
                "would_submit_symbol": intent.symbol,
                "would_submit_side": intent.side,
                "would_submit_notional": notional,
                "exchange_order_id": "",
                "submitted_price": float(intent.entry_price),
                "submitted_notional": notional,
                "exchange_response_excerpt": None,
            }
        )
        return result

    def _gating_reasons_for_intent(
        self,
        intent: OrderIntent,
        mode: LiveMode,
        reconciliation: dict[str, Any] | None,
    ) -> list[str]:
        reasons: list[str] = []
        adapter = self._adapter_provider()
        flags = self._enabled_flags()
        if mode not in ("demo", "live"):
            reasons.append("invalid_mode")
        if not flags["enable_live_trading"]:
            reasons.append("enable_live_trading_disabled")
        if mode == "live" and not flags["confirm_live_trading"]:
            reasons.append("confirm_live_trading_disabled")
        if mode == "live" and flags["dry_run"]:
            reasons.append("dry_run_enabled")
        if not adapter.has_api_keys():
            reasons.append("api_keys_missing")
        if reconciliation is None:
            reasons.append("reconciliation_required")
        else:
            checked_at = int(reconciliation.get("checked_at") or 0)
            if checked_at <= 0:
                reasons.append("reconciliation_required")
            elif (time.time() - checked_at) > self.config.reconciliation_max_age_seconds:
                reasons.append("reconciliation_stale")
            if reconciliation.get("blocked"):
                reasons.append(reconciliation.get("reason") or "reconciliation_blocked")
            if len(reconciliation.get("positions", [])) >= self.config.max_live_positions:
                reasons.append("max_live_positions_reached")
        if intent.order_intent_id in self.submissions_by_mode[mode]:
            reasons.append("intent_already_submitted_live")
        if intent.status not in {"approved", "submitted"}:
            reasons.append(f"intent_status_not_live_eligible:{intent.status}")
        if intent.symbol.upper() not in self.config.allowed_symbols:
            reasons.append("symbol_not_whitelisted")
        if intent.timeframe not in self.config.allowed_timeframes:
            reasons.append("timeframe_not_whitelisted")
        if intent.trigger_mode not in self.config.allowed_trigger_modes:
            reasons.append("trigger_mode_not_whitelisted")
        if float(intent.entry_price) * float(intent.quantity) > self.config.max_live_notional:
            reasons.append("live_notional_cap_exceeded")
        return reasons

    def _gating_reasons_for_close(self, mode: LiveMode) -> list[str]:
        reasons: list[str] = []
        adapter = self._adapter_provider()
        flags = self._enabled_flags()
        if not flags["enable_live_trading"]:
            reasons.append("enable_live_trading_disabled")
        if mode == "live" and not flags["confirm_live_trading"]:
            reasons.append("confirm_live_trading_disabled")
        if mode == "live" and flags["dry_run"]:
            reasons.append("dry_run_enabled")
        if not adapter.has_api_keys():
            reasons.append("api_keys_missing")
        return reasons

    def _enabled_flags(self) -> dict[str, bool]:
        return {
            "enable_live_trading": os.environ.get("ENABLE_LIVE_TRADING", "false").lower() == "true",
            "confirm_live_trading": os.environ.get("CONFIRM_LIVE_TRADING", "false").lower() == "true",
            "dry_run": os.environ.get("DRY_RUN", "true").lower() == "true",
        }

    def _blocked_reason(self, enabled_flags: dict[str, bool], api_key_ready: bool) -> str:
        if not enabled_flags["enable_live_trading"]:
            return "enable_live_trading_disabled"
        if not api_key_ready:
            return "api_keys_missing"
        if enabled_flags["dry_run"]:
            return "dry_run_enabled"
        if not enabled_flags["confirm_live_trading"]:
            return "confirm_live_trading_disabled"
        for mode, report in self.reconciliation_by_mode.items():
            if report is None:
                return f"reconciliation_required:{mode}"
        for report in self.reconciliation_by_mode.values():
            if report and report.get("blocked"):
                return str(report.get("reason") or "reconciliation_blocked")
            if report and (time.time() - int(report.get("checked_at") or 0)) > self.config.reconciliation_max_age_seconds:
                return "reconciliation_stale"
        return ""

    @staticmethod
    def _result_base(intent: OrderIntent, mode: LiveMode) -> dict[str, Any]:
        return {
            "mode": mode,
            "order_intent_id": intent.order_intent_id,
            "signal_id": intent.signal_id,
            "symbol": intent.symbol,
            "side": intent.side,
            "timeframe": intent.timeframe,
            "trigger_mode": intent.trigger_mode,
        }


__all__ = ["LiveBridgeConfig", "LiveExecutionEngine"]
