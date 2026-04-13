"""Core data models — shared across tools, agent, engine, and UI.

Every object in the system lifecycle is defined here.
No logic, only data structures.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ── Factor ───────────────────────────────────────────────────────────────

FactorStage = Literal["core", "candidate", "validated"]

@dataclass
class FactorDefinition:
    """A single trading factor."""
    id: str = field(default_factory=new_id)
    name: str = ""
    description: str = ""
    category: str = ""        # momentum | trend | volatility | volume | structure
    indicator: str = ""       # e.g. "rsi", "adx", "zone_score"
    params: dict[str, Any] = field(default_factory=dict)
    condition: str = "gt"     # gt | lt | cross_above | cross_below | between
    threshold: float = 0.0
    source: str = "system"    # system | ai | manual | paper | external
    stage: FactorStage = "core"
    confidence: float = 0.5   # 0-1
    created_at: float = field(default_factory=time.time)


# ── Strategy Config (structured schema) ─────────────────────────────────

@dataclass
class MarketScope:
    """What to trade and on which timeframes."""
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    main_tf: str = "4h"           # primary analysis timeframe
    confirm_tf: str = ""          # higher-TF trend confirmation (e.g. 1d)
    entry_tf: str = ""            # lower-TF entry precision (e.g. 15m)


@dataclass
class ConditionRule:
    """A single filter condition with parameters."""
    indicator: str = ""           # rsi, adx, bb_width, volume_ratio, zone_score...
    condition: str = "gt"         # gt | lt | between | cross_above | cross_below
    threshold: float = 0.0
    threshold_upper: float = 0.0  # for 'between' condition
    params: dict[str, Any] = field(default_factory=dict)  # e.g. {"period": 14}
    enabled: bool = True


@dataclass
class EntryRules:
    """How to enter a trade."""
    modes: list[str] = field(default_factory=lambda: ["rejection"])  # rejection, failed_breakout, retest, pre_limit
    logic_combine: str = "OR"     # OR = any mode triggers | AND = all must confirm
    offset_pct: float = 0.0       # limit offset from zone


@dataclass
class ExitRules:
    """How to manage stops and targets."""
    stop_type: str = "structure"  # structure | fixed_pct | atr
    stop_pct: float = 1.0         # for fixed_pct mode
    stop_atr_mult: float = 1.5    # for atr mode
    tp_type: str = "rr"           # rr | fixed_pct | trailing | structure
    rr_target: float = 2.0        # for rr mode
    tp_pct: float = 3.0           # for fixed_pct mode
    trailing_pct: float = 1.0     # for trailing mode


@dataclass
class RiskProfile:
    """Multi-dimensional risk controls."""
    risk_per_trade: float = 0.01  # 1% default
    max_concurrent: int = 3       # max open positions
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    max_consecutive_losses: int = 5
    auto_pause_on_dd: bool = True  # pause strategy if DD exceeded


@dataclass
class StrategyConfig:
    """Unified strategy configuration — used by manual builder, AI, templates, agent."""
    market: MarketScope = field(default_factory=MarketScope)
    logic_tags: list[str] = field(default_factory=lambda: ["reversal"])  # reversal, breakout, trend, scalp
    logic_combine: str = "OR"     # how multiple logics interact
    conditions: list[ConditionRule] = field(default_factory=list)
    entry: EntryRules = field(default_factory=EntryRules)
    exit: ExitRules = field(default_factory=ExitRules)
    risk: RiskProfile = field(default_factory=RiskProfile)


# ── Strategy Draft ──────────────────────────────────────────────────────

StrategyStatus = Literal["draft", "pending_simulation", "simulating", "simulated", "ranked", "live_draft", "live_running", "retired"]

@dataclass
class StrategyDraft:
    """A strategy definition — from template, AI, or manual builder."""
    id: str = field(default_factory=new_id)
    name: str = ""
    source: str = "manual"    # ai | manual | template | leaderboard | pattern_engine
    template_id: str = ""     # if from catalog
    config: StrategyConfig = field(default_factory=StrategyConfig)  # structured config
    # Legacy flat fields (kept for backward compat with existing drafts)
    logic_tags: list[str] = field(default_factory=list)
    factor_ids: list[str] = field(default_factory=list)
    trigger_modes: list[str] = field(default_factory=list)
    entry_mode: str = ""
    exit_rules: dict[str, Any] = field(default_factory=dict)
    risk_rules: dict[str, Any] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    status: StrategyStatus = "draft"
    generation: int = 0
    batch_id: str = ""
    parent_strategy_id: str = ""
    # Pattern engine lineage (when source = "pattern_engine")
    source_pattern_id: str = ""  # the 2-touch pattern record that triggered this
    decision_rule: str = ""      # which rule triggered: reversal_strict, breakout_trend_aligned, etc.
    pattern_ev: float = 0.0
    pattern_confidence: float = 0.0
    pattern_decision: str = ""   # reversal | breakout | failed_breakout
    pattern_reason: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ── Simulation ───────────────────────────────────────────────────────────

SimJobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

@dataclass
class SimulationJob:
    """A batch simulation experiment."""
    id: str = field(default_factory=new_id)
    name: str = ""
    strategy_ids: list[str] = field(default_factory=list)
    date_range: dict[str, str] = field(default_factory=dict)  # start, end
    capital: float = 10000.0
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    status: SimJobStatus = "pending"
    generation: int = 0
    batch_id: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    results: list[str] = field(default_factory=list)
    failed_items: list[dict] = field(default_factory=list)  # {strategy_id, symbol, tf, stage, error}


@dataclass
class SimulationResult:
    """Result of one strategy in one simulation job."""
    id: str = field(default_factory=new_id)
    job_id: str = ""
    strategy_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    # Overall metrics
    return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_count: int = 0
    avg_rr: float = 0.0
    score: float = 0.0
    # Train/Val/Test split scores (anti-overfit)
    train_score: float = 0.0
    val_score: float = 0.0
    test_score: float = 0.0
    train_return: float = 0.0
    val_return: float = 0.0
    test_return: float = 0.0
    overfit_flag: str = ""        # "" | "overfit" | "stable" | "degraded"
    # Traceability
    generation: int = 0
    batch_id: str = ""
    tested_at: float = field(default_factory=time.time)


# ── Leaderboard ──────────────────────────────────────────────────────────

@dataclass
class LeaderboardEntry:
    """A ranked strategy in the leaderboard."""
    id: str = field(default_factory=new_id)
    strategy_id: str = ""
    strategy_name: str = ""
    source: str = ""          # ai | manual | template
    symbol: str = ""
    timeframe: str = ""
    trigger_modes: list[str] = field(default_factory=list)
    factor_ids: list[str] = field(default_factory=list)
    # Metrics
    return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_count: int = 0
    score: float = 0.0
    # Deployment
    deployment_eligible: bool = False
    deployed: bool = False
    generation: int = 0
    batch_id: str = ""
    simulation_job_id: str = ""
    ranked_at: float = field(default_factory=time.time)


# ── Live Deployment ──────────────────────────────────────────────────────

DeployStatus = Literal["draft", "pending_approval", "approved", "running", "paused", "stopped", "retired"]

@dataclass
class LiveDeploymentDraft:
    """A deployment candidate — not yet running."""
    id: str = field(default_factory=new_id)
    strategy_id: str = ""
    leaderboard_entry_id: str = ""
    capital_allocation: float = 0.0
    risk_per_trade: float = 0.01
    max_concurrent_positions: int = 3
    exchange: str = "bitget"
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    auto_submit: bool = False
    runtime_days: int = 30
    status: DeployStatus = "draft"
    # Pattern lineage (copied from source strategy draft)
    source_pattern_id: str = ""
    decision_rule: str = ""
    pattern_decision: str = ""
    pattern_ev: float = 0.0
    created_at: float = field(default_factory=time.time)
    approved_at: float = 0.0
    notes: str = ""


@dataclass
class LiveStrategyInstance:
    """A running live strategy."""
    id: str = field(default_factory=new_id)
    deployment_draft_id: str = ""
    strategy_id: str = ""
    running_status: str = "running"   # running | paused | stopped | error
    allocated_capital: float = 0.0
    # P&L is split: virtual (pattern engine simulated outcomes — not real
    # money) vs real (only updated on actual exchange fill). `current_pnl`
    # is DEPRECATED — kept only for backward compat reads, mirrors
    # pattern_virtual_pnl.
    current_pnl: float = 0.0           # DEPRECATED — mirrors pattern_virtual_pnl
    pattern_virtual_pnl: float = 0.0   # pattern engine simulated outcomes
    realized_pnl_usd: float = 0.0      # only set after exchange fill ack
    current_drawdown: float = 0.0
    open_positions: int = 0
    total_trades: int = 0
    # Pattern lineage (copied from draft → strategy)
    source_pattern_id: str = ""
    decision_rule: str = ""
    pattern_decision: str = ""
    pattern_ev_expected: float = 0.0  # what the pattern engine predicted
    symbol: str = ""
    timeframe: str = ""
    # Realized outcome (populated when instance is stopped/retired)
    realized_return_pct: float = 0.0
    realized_return_atr: float = 0.0
    realized_drawdown_pct: float = 0.0
    realized_drawdown_atr: float = 0.0
    bars_held: int = 0
    outcome_class: str = ""  # bounce_success | breakout_fail | stop_out | target_hit | timeout
    outcome_success: bool = False
    outcome_written_back: bool = False  # set to True after pattern writeback
    started_at: float = field(default_factory=time.time)
    last_action_at: float = 0.0
    stopped_at: float = 0.0
    error: str = ""


# ── Audit ────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """Every important system action gets logged."""
    id: str = field(default_factory=new_id)
    timestamp: float = field(default_factory=time.time)
    actor: str = ""           # agent | user | system | risk_engine
    action: str = ""          # factor_created | strategy_generated | sim_started | deployed | stopped
    object_type: str = ""     # factor | strategy | simulation | deployment
    object_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
