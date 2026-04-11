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
    allowed_symbols: tuple[str, ...] = ("HYPEUSDT", "RIVERUSDT")
    allowed_timeframes: tuple[str, ...] = ("1h",)
    allowed_trigger_modes: tuple[str, ...] = ("pre_limit",)
    max_live_positions: int = 1
    default_mode: LiveMode = "demo"
    max_live_notional: float = 100.0
    reconciliation_max_age_seconds: int = 300

    @classmethod
    def from_env(cls) -> "LiveBridgeConfig":
        return cls(
            allowed_symbols=_csv_env("LIVE_ALLOWED_SYMBOLS", ("HYPEUSDT", "RIVERUSDT")),
            allowed_timeframes=_csv_env("LIVE_ALLOWED_TIMEFRAMES", ("1h",)),
            allowed_trigger_modes=_csv_env(
                "LIVE_ALLOWED_TRIGGER_MODES",
                ("pre_limit",),
            ),
            max_live_positions=_int_env("LIVE_MAX_POSITIONS", 1),
            default_mode=_mode_env("LIVE_DEFAULT_MODE", "demo"),
            max_live_notional=_float_env("LIVE_MAX_NOTIONAL", 100.0),
            reconciliation_max_age_seconds=_int_env("LIVE_RECONCILIATION_MAX_AGE_SECONDS", 300),
        )


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
            "exchange": getattr(adapter, "exchange_name", "live"),
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

    def export_state(self) -> dict[str, Any]:
        return {
            "last_preview_result": self.last_preview_result,
            "last_submission_result": self.last_submission_result,
            "reconciliation_by_mode": self.reconciliation_by_mode,
            "submissions_by_mode": self.submissions_by_mode,
        }

    def load_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        self.last_preview_result = state.get("last_preview_result")
        self.last_submission_result = state.get("last_submission_result")
        reconciliation = state.get("reconciliation_by_mode") or {}
        submissions = state.get("submissions_by_mode") or {}
        self.reconciliation_by_mode = {
            "demo": reconciliation.get("demo"),
            "live": reconciliation.get("live"),
        }
        self.submissions_by_mode = {
            "demo": dict(submissions.get("demo") or {}),
            "live": dict(submissions.get("live") or {}),
        }

    async def preview_live_submission(self, intent: OrderIntent, mode: LiveMode | None = None) -> dict[str, Any]:
        resolved_mode = mode or self.config.default_mode
        result = await self._build_preview(intent, resolved_mode)
        self.last_preview_result = result
        return result

    async def build_preflight(
        self,
        *,
        mode: LiveMode,
        intent: OrderIntent | None = None,
    ) -> dict[str, Any]:
        adapter = self._adapter_provider()
        flags = self._enabled_flags()
        reconciliation = self.reconciliation_by_mode.get(mode)
        preview_result = await self._build_preview(intent, mode) if intent is not None else None

        checks: list[dict[str, Any]] = [
            self._preflight_check(
                "enable_live_trading",
                "ENABLE_LIVE_TRADING",
                flags["enable_live_trading"],
                detail="Set ENABLE_LIVE_TRADING=true to allow any live bridge action.",
            ),
            self._preflight_check(
                "api_keys_ready",
                "Bitget API keys",
                adapter.has_api_keys(),
                detail="Fill BITGET_API_KEY, BITGET_SECRET_KEY, and BITGET_PASSPHRASE.",
            ),
        ]

        if mode == "live":
            checks.extend(
                [
                    self._preflight_check(
                        "confirm_live_trading",
                        "CONFIRM_LIVE_TRADING",
                        flags["confirm_live_trading"],
                        detail="Real-money live mode requires CONFIRM_LIVE_TRADING=true.",
                    ),
                    self._preflight_check(
                        "dry_run_disabled",
                        "DRY_RUN disabled",
                        not flags["dry_run"],
                        detail="Real-money live mode requires DRY_RUN=false.",
                    ),
                ]
            )

        checks.extend(
            [
                self._preflight_check(
                    "reconciliation_present",
                    "Fresh reconciliation",
                    reconciliation is not None,
                    detail=f"Run /api/live-execution/reconcile for mode={mode}.",
                ),
                self._preflight_check(
                    "reconciliation_fresh",
                    "Reconciliation not stale",
                    self._reconciliation_is_fresh(reconciliation),
                    detail=f"Re-run reconciliation within {self.config.reconciliation_max_age_seconds}s of submit.",
                ),
                self._preflight_check(
                    "reconciliation_clear",
                    "Reconciliation unblocked",
                    not bool(reconciliation and reconciliation.get("blocked")),
                    detail=reconciliation.get("reason", "") if reconciliation else "No reconciliation report yet.",
                ),
                self._preflight_check(
                    "intent_selected",
                    "Live-eligible intent selected",
                    intent is not None,
                    detail="Seed at least one approved/submitted paper intent before live submit.",
                ),
            ]
        )

        if intent is not None and preview_result is not None:
            checks.append(
                self._preflight_check(
                    "intent_gating",
                    "Intent passes live gating",
                    not bool(preview_result.get("blocked")),
                    detail=", ".join(preview_result.get("blocking_reasons") or []) or preview_result.get("reason", ""),
                )
            )

        blocking_reasons = [check["check_id"] for check in checks if check["blocking"] and not check["ok"]]
        next_actions = self._preflight_next_actions(mode, checks, preview_result)
        return {
            "ready": len(blocking_reasons) == 0,
            "mode": mode,
            "exchange": getattr(adapter, "exchange_name", "bitget"),
            "selected_intent_id": intent.order_intent_id if intent else None,
            "selected_signal_id": intent.signal_id if intent else None,
            "selected_symbol": intent.symbol if intent else None,
            "selected_timeframe": intent.timeframe if intent else None,
            "selected_trigger_mode": intent.trigger_mode if intent else None,
            "checks": checks,
            "blocking_reasons": blocking_reasons,
            "next_actions": next_actions,
            "preview_result": preview_result,
        }

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
        if self.config.allowed_symbols and intent.symbol.upper() not in self.config.allowed_symbols:
            reasons.append("symbol_not_whitelisted")
        if self.config.allowed_timeframes and intent.timeframe not in self.config.allowed_timeframes:
            reasons.append("timeframe_not_whitelisted")
        if self.config.allowed_trigger_modes and intent.trigger_mode not in self.config.allowed_trigger_modes:
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

    def _reconciliation_is_fresh(self, reconciliation: dict[str, Any] | None) -> bool:
        if reconciliation is None:
            return False
        checked_at = int(reconciliation.get("checked_at") or 0)
        if checked_at <= 0:
            return False
        return (time.time() - checked_at) <= self.config.reconciliation_max_age_seconds

    @staticmethod
    def _preflight_check(check_id: str, label: str, ok: bool, *, detail: str = "", blocking: bool = True) -> dict[str, Any]:
        return {
            "check_id": check_id,
            "label": label,
            "ok": bool(ok),
            "blocking": blocking,
            "detail": detail,
        }

    def _preflight_next_actions(
        self,
        mode: LiveMode,
        checks: list[dict[str, Any]],
        preview_result: dict[str, Any] | None,
    ) -> list[str]:
        actions: list[str] = []
        failed = {check["check_id"] for check in checks if not check["ok"]}
        if "enable_live_trading" in failed:
            actions.append("Set ENABLE_LIVE_TRADING=true.")
        if "api_keys_ready" in failed:
            actions.append("Fill BITGET_API_KEY, BITGET_SECRET_KEY, and BITGET_PASSPHRASE.")
        if "confirm_live_trading" in failed:
            actions.append("Set CONFIRM_LIVE_TRADING=true for real-money mode.")
        if "dry_run_disabled" in failed:
            actions.append("Set DRY_RUN=false before real-money submit.")
        if "reconciliation_present" in failed or "reconciliation_fresh" in failed or "reconciliation_clear" in failed:
            actions.append(f"Run reconcile for mode={mode} and confirm it returns ok/unblocked.")
        if "intent_selected" in failed:
            actions.append("Generate or approve at least one paper intent, then select it in Live Bridge.")
        if "intent_gating" in failed and preview_result is not None:
            actions.append(f"Selected intent is blocked by: {', '.join(preview_result.get('blocking_reasons') or []) or preview_result.get('reason', 'unknown')}.")
        if not actions:
            actions.append(f"Preflight passed for mode={mode}. Demo/real-money submit may proceed under current safeguards.")
        return actions

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


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw.strip():
        return ()  # explicitly empty = no restriction
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or ()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        return default


def _mode_env(name: str, default: LiveMode) -> LiveMode:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"demo", "live"}:
        return raw
    return default
