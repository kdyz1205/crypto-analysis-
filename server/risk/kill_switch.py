from __future__ import annotations

from ..execution.types import KillSwitchState, PaperAccountSummary, PaperExecutionConfig


class PaperKillSwitch:
    def __init__(self) -> None:
        self._state = KillSwitchState()

    def snapshot(self) -> KillSwitchState:
        return KillSwitchState(
            blocked=self._state.blocked,
            manual_blocked=self._state.manual_blocked,
            data_blocked=self._state.data_blocked,
            risk_blocked=self._state.risk_blocked,
            reason=self._state.reason,
        )

    def reset(self) -> None:
        self._state = KillSwitchState()

    def load_state(self, state: KillSwitchState) -> KillSwitchState:
        self._state = KillSwitchState(
            blocked=bool(state.blocked),
            manual_blocked=bool(state.manual_blocked),
            data_blocked=bool(state.data_blocked),
            risk_blocked=bool(state.risk_blocked),
            reason=str(state.reason or ""),
        )
        return self.snapshot()

    def set_manual(self, blocked: bool, reason: str = "") -> KillSwitchState:
        self._state.manual_blocked = blocked
        self._state.reason = reason if blocked else ""
        self._sync_blocked()
        return self.snapshot()

    def set_data_blocked(self, blocked: bool, reason: str = "") -> KillSwitchState:
        self._state.data_blocked = blocked
        if blocked:
            self._state.reason = reason or "data_state_blocked"
        elif not self._state.manual_blocked and not self._state.risk_blocked:
            self._state.reason = ""
        self._sync_blocked()
        return self.snapshot()

    def evaluate(self, account: PaperAccountSummary, config: PaperExecutionConfig) -> KillSwitchState:
        if account.daily_realized_pnl <= -(config.max_daily_loss * account.starting_equity):
            self._state.risk_blocked = True
            self._state.reason = "max_daily_loss_hit"
        self._sync_blocked()
        return self.snapshot()

    def _sync_blocked(self) -> None:
        self._state.blocked = bool(
            self._state.manual_blocked or self._state.data_blocked or self._state.risk_blocked
        )


__all__ = ["PaperKillSwitch"]
