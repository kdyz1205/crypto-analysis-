"""Dataclasses for conditional orders tied to drawn trendlines."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────
# Status transitions
#
#   pending   — watcher is polling; line not touched yet
#   triggered — line was touched, Telegram alert fired. If exchange
#               submit is configured, exchange_order_id is populated.
#   cancelled — user cancelled, or auto-cancelled by stop conditions
#               (line broken / price drifted too far / expired)
#   failed    — triggered but exchange submit errored
# ─────────────────────────────────────────────────────────────────────
ConditionalStatus = Literal["pending", "triggered", "filled", "cancelled", "failed"]


@dataclass(frozen=True, slots=True)
class TriggerConfig:
    """How the watcher decides the line has been "touched"."""
    # Tolerance as fraction of ATR at the current bar. 0.2 = within 0.2 × ATR.
    tolerance_atr: float = 0.2
    # How often the watcher polls, in seconds. Defaults tuned per TF below.
    poll_seconds: int = 60
    # Auto-cancel after this many seconds without triggering. 0 = unlimited.
    # User policy: lines live forever until user deletes. Default disabled.
    max_age_seconds: int = 0
    # Auto-cancel if price drifts beyond this × ATR from the line. 0 = unlimited.
    # User policy: no drift cancel — line stays armed forever.
    max_distance_atr: float = 0.0
    # If price closes through the line by >= this × ATR, the line is "broken".
    # Under the user's strategy this does NOT cancel — it triggers auto-reverse
    # (the line flips polarity and a new conditional is spawned in the opposite
    # direction). See reverse_* on OrderConfig.
    break_threshold_atr: float = 0.5


@dataclass(frozen=True, slots=True)
class OrderConfig:
    """What to do when the trigger fires.

    order_kind determines BOTH the trigger condition AND the natural
    direction (combined with line side):

        line side    kind       direction   trigger
        ---------    ----       ---------   -------
        support      bounce     long        price within tolerance of line
        support      breakout   short       close through line by breakaway_atr
        resistance   bounce     short       price within tolerance of line
        resistance   breakout   long        close through line by breakaway_atr

    The same manual line can have multiple ConditionalOrders with
    different kinds — e.g. one bounce + one breakout, letting the first
    confirmed outcome determine which gets triggered.
    """
    # Direction — computed from (side, kind) if not explicitly provided
    direction: Literal["long", "short"]
    # Trigger kind — "bounce" = touch within tolerance; "breakout" = close through
    order_kind: Literal["bounce", "breakout"] = "bounce"
    # Entry offset in ABSOLUTE price POINTS (e.g. 0.05 = 5 cents away from line).
    # If None, falls back to entry_offset_atr. Points is more intuitive for
    # the user "put my order 0.10 above support".
    entry_offset_points: float | None = None
    # Entry offset as fraction of ATR from the line. Used if _points is None.
    entry_offset_atr: float = 0.0
    # Stop distance in ABSOLUTE price POINTS (takes precedence over stop_atr)
    stop_points: float | None = None
    stop_atr: float = 0.3
    # Take profit — choose ONE:
    #   rr_target:   fixed R multiple (e.g. 2.0 = 2R target)
    #   tp_price:    absolute price
    # If both are None, no TP is set (manual close).
    rr_target: float | None = 2.0
    tp_price: float | None = None
    # Size — choose ONE:
    #   notional_usd:   fixed USD notional
    #   equity_pct:     percentage of total equity
    #   risk_pct:       percentage of equity to risk (respects stop distance)
    notional_usd: float | None = None
    equity_pct: float | None = None
    risk_pct: float | None = 0.005  # default 0.5% risk
    # Whether to actually submit to exchange (True) or only alert (False)
    submit_to_exchange: bool = False
    # Exchange account mode (only if submit_to_exchange=True)
    exchange_mode: Literal["paper", "demo", "live"] = "paper"
    # Line-relative percentages — the ONLY supported offset mode. Every
    # replan re-applies these to the NEW projected line price, so sloped
    # lines get their orders moved every bar. 0 = not set (rejected by
    # the watcher; must be explicitly provided by the API / modal).
    tolerance_pct_of_line: float = 0.0
    stop_offset_pct_of_line: float = 0.0
    # Cross-margin leverage (e.g. 10 for 10x). If set, notional size is
    # computed as account_equity * leverage. Overrides notional_usd /
    # equity_pct / risk_pct. Required for the UI's leverage-based modal.
    leverage: float | None = None
    # Auto-reverse on stop-loss. When the stop gets hit, the watcher
    # spawns a NEW conditional on the SAME manual_line_id with direction
    # flipped and these reverse_* params applied. None = no reverse.
    reverse_enabled: bool = False
    reverse_entry_offset_pct: float = 0.0
    reverse_stop_offset_pct: float = 0.0
    reverse_rr_target: float | None = None
    reverse_leverage: float | None = None


@dataclass(frozen=True, slots=True)
class ConditionalEvent:
    """One log entry on a conditional order's lifetime."""
    ts: int
    kind: Literal[
        "created", "poll", "triggered", "exchange_submitted",
        "exchange_acked", "exchange_error", "cancelled", "expired",
        "line_broken", "drifted_far", "tolerance_check",
    ]
    price: float | None = None
    line_price: float | None = None
    distance_atr: float | None = None
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionalOrder:
    """A conditional order tied to a drawn trendline.

    Mutable because status + events list evolve over time. Write through
    ConditionalOrderStore.update_status / .append_event rather than
    mutating directly.
    """
    conditional_id: str
    manual_line_id: str          # FK to ManualTrendline
    symbol: str
    timeframe: str
    side: Literal["support", "resistance"]
    # Line coords (snapshot at creation — we recompute projection each poll)
    t_start: int
    t_end: int
    price_start: float
    price_end: float

    # Pattern stats snapshot (what the user saw when they approved)
    pattern_stats_at_create: dict[str, Any]

    # Config
    trigger: TriggerConfig
    order: OrderConfig

    # State
    status: ConditionalStatus
    created_at: int
    updated_at: int
    triggered_at: int | None = None
    cancelled_at: int | None = None
    cancel_reason: str = ""

    # Fills
    exchange_order_id: str | None = None
    fill_price: float | None = None
    fill_qty: float | None = None

    # Activity log
    events: list[ConditionalEvent] = field(default_factory=list)

    # Last-known state from the watcher (for UI display)
    last_poll_ts: int | None = None
    last_market_price: float | None = None
    last_line_price: float | None = None
    last_distance_atr: float | None = None

    # Copied from the manual drawing. Most user lines extend right, so a
    # Bitget plan order must keep repricing after the second anchor.
    extend_left: bool = False
    extend_right: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # events is already dicts via asdict
        return d

    def line_price_at(self, ts: int) -> float:
        """Project the line's price at timestamp `ts`.

        Uses LOGARITHMIC interpolation because the chart's price axis is
        in Log mode (default in chart.js: PriceScaleMode.Logarithmic).
        lightweight-charts renders a straight line between 2 points in
        PIXEL space, which — on a log-scale axis — maps to EXPONENTIAL
        interpolation in price space (constant growth rate per unit time),
        NOT linear price interpolation.

        User report 2026-04-21 (BAS/4h): backend's linear projection
        gave 0.010224 at bar_open, but chart visually showed the line at
        0.010177 at the same X — a 0.47% discrepancy. User circled 0.010173
        on the chart. Log interpolation resolves the mismatch.

        Respects the manual drawing's extension flags. With extend_right=True
        the line keeps projecting after the second anchor; with
        extend_left=True it also projects before the first anchor. Otherwise
        it snaps to the nearest anchor outside the anchor window.
        """
        import math
        span = self.t_end - self.t_start
        if span <= 0:
            return self.price_start
        if ts <= self.t_start and not self.extend_left:
            return self.price_start
        if ts >= self.t_end and not self.extend_right:
            return self.price_end
        # Log interpolation: matches visual rendering when chart is Log mode.
        # Guard against non-positive prices (shouldn't happen for real coins
        # but keep the fallback safe).
        if self.price_start <= 0 or self.price_end <= 0:
            slope_per_sec = (self.price_end - self.price_start) / span
            return self.price_start + slope_per_sec * (ts - self.t_start)
        log_start = math.log(self.price_start)
        log_end = math.log(self.price_end)
        ratio = (ts - self.t_start) / span
        return math.exp(log_start + ratio * (log_end - log_start))

    def line_price_at_bar_open(self, ts: int) -> float:
        """DEPRECATED — do not call for new code. Use `line_price_at(ts)`.

        History: this method was introduced 2026-04-21 to fix a 4h click-
        moment drift (~2.5h inside a 4h bar). It snaps `ts` to the
        current TF bar's OPEN time. That worked for intraday TFs
        (bar-open ≤ a few hours stale) but created a far worse bug on
        1d/1w: user's eye is at the RIGHT edge of the live bar ≈ now,
        not at the LEFT edge = bar-open which for 1d is up to 24h old.

        On 2026-04-23 a user's ZEC 1d line placed a trigger 4.8 points
        below visual expectation due to 22h of slope drift. All callers
        (place-line-order, watcher.replan) have been switched to
        `line_price_at(ts)` with `ts = now`. This method is kept only to
        avoid breaking any external caller and will be removed in a
        future commit once grep shows zero callers.
        """
        secs = _TF_SECONDS.get(self.timeframe, 3600)
        bar_open = (int(ts) // secs) * secs
        return self.line_price_at(bar_open)


_TF_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
    "1d": 86400, "1w": 604800,
}


__all__ = [
    "ConditionalStatus",
    "TriggerConfig",
    "OrderConfig",
    "ConditionalEvent",
    "ConditionalOrder",
]
