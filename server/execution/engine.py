from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..risk.kill_switch import PaperKillSwitch
from ..risk.risk_rules import cooldown_scope_key, evaluate_signal_risk
from ..strategy.replay import ReplayResult
from .order_manager import PaperOrderManager
from .position_manager import PaperPositionManager
from .types import PaperAccountSummary, PaperExecutionConfig, PaperExecutionState


class PaperExecutionEngine:
    def __init__(
        self,
        config: PaperExecutionConfig | None = None,
        *,
        max_fill_history: int | None = 100,
        max_closed_history: int | None = 100,
    ) -> None:
        self.config = config or PaperExecutionConfig()
        self.order_manager = PaperOrderManager(max_fill_history=max_fill_history)
        self.position_manager = PaperPositionManager(max_closed_history=max_closed_history)
        self.kill_switch = PaperKillSwitch()
        self.realized_pnl = 0.0
        self.daily_realized_pnl = 0.0
        self.current_day: str | None = None
        self.consecutive_losses = 0
        self.cooldowns: dict[str, int] = {}
        self.last_processed_bar_by_stream: dict[str, int] = {}

    def reset(self) -> PaperExecutionState:
        self.order_manager.reset()
        self.position_manager.reset()
        self.kill_switch.reset()
        self.realized_pnl = 0.0
        self.daily_realized_pnl = 0.0
        self.current_day = None
        self.consecutive_losses = 0
        self.cooldowns.clear()
        self.last_processed_bar_by_stream.clear()
        return self.get_state()

    def load_state(self, state: PaperExecutionState, *, current_day: str | None = None) -> PaperExecutionState:
        self.config = state.config
        self.order_manager.load_state(
            intents=list(state.intents),
            open_orders=list(state.open_orders),
            recent_fills=list(state.recent_fills),
        )
        self.position_manager.load_state(
            open_positions=list(state.open_positions),
            recent_closed_positions=list(state.recent_closed_positions),
        )
        self.kill_switch.load_state(state.kill_switch)
        self.realized_pnl = float(state.account.realized_pnl)
        self.daily_realized_pnl = float(state.account.daily_realized_pnl)
        self.current_day = current_day
        self.consecutive_losses = int(state.account.consecutive_losses)
        self.cooldowns = dict(state.cooldowns)
        self.last_processed_bar_by_stream = dict(state.account.last_processed_bar_by_stream)
        return self.get_state()

    def update_config(self, **changes) -> PaperExecutionConfig:
        self.config = replace(self.config, **changes)
        return self.config

    def get_state(self) -> PaperExecutionState:
        return PaperExecutionState(
            config=self.config,
            account=self._build_account_summary(),
            kill_switch=self.kill_switch.snapshot(),
            intents=self.order_manager.get_intents()[-20:],
            open_orders=self.order_manager.get_open_orders(),
            open_positions=self.position_manager.get_open_positions(),
            recent_fills=self.order_manager.get_recent_fills(),
            recent_closed_positions=self.position_manager.get_recent_closed_positions(),
            cooldowns=dict(self.cooldowns),
        )

    def step(
        self,
        symbol: str,
        timeframe: str,
        candles_df: pd.DataFrame,
        replay_result: ReplayResult,
        *,
        bar_index: int | None = None,
        snapshot_offset: int = 0,
    ) -> dict[str, Any]:
        stream_key = f"{symbol}:{timeframe}"
        max_replay_index = snapshot_offset + len(replay_result.snapshots) - 1
        if max_replay_index < snapshot_offset:
            raise ValueError("replay returned no snapshots")

        last_processed = self.last_processed_bar_by_stream.get(stream_key, -1)
        target_bar = (last_processed + 1) if bar_index is None else bar_index
        if target_bar > max_replay_index:
            target_bar = max_replay_index
        if target_bar < last_processed:
            raise ValueError(f"bar_index {target_bar} is behind already-processed bar {last_processed} for {stream_key}")

        processed_bars: list[int] = []
        for current_bar in range(last_processed + 1, target_bar + 1):
            snapshot = replay_result.snapshots[current_bar - snapshot_offset]
            bar = candles_df.iloc[current_bar].to_dict()
            timestamp = snapshot.timestamp

            self._roll_day(timestamp)
            fills = self.order_manager.advance_orders_for_bar(current_bar, bar, timestamp)
            for fill in fills:
                can_open, blocking_reason = self.position_manager.can_open_fill(
                    fill,
                    allow_multiple_same_direction_per_symbol=self.config.allow_multiple_same_direction_per_symbol,
                )
                if not can_open:
                    self.order_manager.reject_filled_order(
                        fill.order_id,
                        fill.fill_id,
                        blocking_reason or "position_open_rejected",
                        current_bar=current_bar,
                    )
                    continue

                opened_position = self.position_manager.open_from_fill(
                    fill,
                    self.order_manager,
                    current_bar=current_bar,
                    current_ts=timestamp,
                    allow_multiple_same_direction_per_symbol=self.config.allow_multiple_same_direction_per_symbol,
                )
                if opened_position is None:
                    self.order_manager.reject_filled_order(
                        fill.order_id,
                        fill.fill_id,
                        "position_open_failed",
                        current_bar=current_bar,
                    )

            closed_positions = self.position_manager.advance_positions_for_bar(current_bar, bar, timestamp)
            for position in closed_positions:
                self.realized_pnl += position.realized_pnl
                self.daily_realized_pnl += position.realized_pnl
                if position.realized_pnl < 0:
                    self.consecutive_losses += 1
                    self.cooldowns[cooldown_scope_key(position.symbol, position.timeframe, position.direction)] = (
                        current_bar + self.config.cooldown_bars_after_loss
                    )
                elif position.realized_pnl > 0:
                    self.consecutive_losses = 0

            self.order_manager.expire_stale_orders(current_bar, self.config.cancel_after_bars)
            self.kill_switch.evaluate(self._build_account_summary(), self.config)

            for signal in snapshot.signals:
                decision = evaluate_signal_risk(
                    signal,
                    self._build_account_summary(),
                    self.position_manager.get_open_positions(),
                    self.config,
                    current_bar=current_bar,
                    cooldowns=self.cooldowns,
                    pending_orders=self.order_manager.get_open_orders(),
                    kill_switch=self.kill_switch.snapshot(),
                )
                intent = self.order_manager.create_order_intent_from_signal(
                    signal,
                    decision,
                    self.config,
                    current_bar=current_bar,
                    current_ts=timestamp,
                )
                if decision.approved:
                    self.order_manager.submit_paper_order(intent)

            self.last_processed_bar_by_stream[stream_key] = current_bar
            processed_bars.append(current_bar)

        return {
            "stream": stream_key,
            "processedBars": processed_bars,
            "lastProcessedBar": self.last_processed_bar_by_stream.get(stream_key, last_processed),
            "state": self.get_state(),
        }

    def set_manual_kill_switch(self, blocked: bool, reason: str = ""):
        return self.kill_switch.set_manual(blocked, reason)

    def _build_account_summary(self) -> PaperAccountSummary:
        open_positions = self.position_manager.get_open_positions()
        open_orders = self.order_manager.get_open_orders()
        unrealized_pnl = sum(position.unrealized_pnl for position in open_positions)
        total_exposure = sum(abs(position.quantity * position.entry_price) for position in open_positions)
        equity = self.config.starting_equity + self.realized_pnl + unrealized_pnl

        return PaperAccountSummary(
            starting_equity=float(self.config.starting_equity),
            equity=float(equity),
            realized_pnl=float(self.realized_pnl),
            unrealized_pnl=float(unrealized_pnl),
            daily_realized_pnl=float(self.daily_realized_pnl),
            consecutive_losses=int(self.consecutive_losses),
            total_exposure=float(total_exposure),
            open_order_count=len(open_orders),
            open_position_count=len(open_positions),
            closed_trade_count=self.position_manager.get_closed_trade_count(),
            last_processed_bar_by_stream=dict(self.last_processed_bar_by_stream),
        )

    def _roll_day(self, timestamp: Any) -> None:
        current_day = self._day_key(timestamp)
        if current_day is None:
            return
        if self.current_day is None:
            self.current_day = current_day
            return
        if current_day != self.current_day:
            self.current_day = current_day
            self.daily_realized_pnl = 0.0

    @staticmethod
    def _day_key(timestamp: Any) -> str | None:
        if timestamp is None:
            return None
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).date().isoformat()
        if isinstance(timestamp, str):
            try:
                value = timestamp.replace("Z", "+00:00")
                return datetime.fromisoformat(value).date().isoformat()
            except ValueError:
                return None
        return None


__all__ = ["PaperExecutionEngine"]
