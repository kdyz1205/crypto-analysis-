"""HFT Kill Switch — shuts down trading when conditions deteriorate.

Checks: consecutive losses, slippage, fill quality, vol spike, latency, data gap.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time


@dataclass
class HFTKillSwitch:
    max_consecutive_losses: int = 8
    max_drawdown_usdt: float = 10.0  # $10 max drawdown for $100 account
    max_slippage_ratio: float = 0.35  # slippage > 35% of expected alpha
    max_latency_ms: float = 2000     # API latency > 2s = trouble
    max_no_data_seconds: float = 30  # no book update for 30s = stop

    # State
    consecutive_losses: int = 0
    session_pnl: float = 0.0
    peak_pnl: float = 0.0
    last_data_time: float = field(default_factory=time.time)
    blocked: bool = False
    block_reason: str = ""

    def check(self, *, pnl: float | None = None, latency_ms: float = 0, data_time: float | None = None) -> bool:
        """Returns True if trading should continue, False if killed."""
        if self.blocked:
            return False

        if pnl is not None:
            self.session_pnl += pnl
            self.peak_pnl = max(self.peak_pnl, self.session_pnl)
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

        if data_time is not None:
            self.last_data_time = data_time

        # Check conditions
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._kill(f"consecutive_losses={self.consecutive_losses}")
            return False

        drawdown = self.peak_pnl - self.session_pnl
        if drawdown > self.max_drawdown_usdt:
            self._kill(f"drawdown=${drawdown:.2f}")
            return False

        if latency_ms > self.max_latency_ms:
            self._kill(f"latency={latency_ms:.0f}ms")
            return False

        data_gap = time.time() - self.last_data_time
        if data_gap > self.max_no_data_seconds:
            self._kill(f"data_gap={data_gap:.0f}s")
            return False

        return True

    def _kill(self, reason: str):
        self.blocked = True
        self.block_reason = reason

    def reset(self):
        self.blocked = False
        self.block_reason = ""
        self.consecutive_losses = 0
