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


# ── Strategy ─────────────────────────────────────────────────────────────

StrategyStatus = Literal["draft", "pending_simulation", "simulating", "simulated", "ranked", "live_draft", "live_running", "retired"]

@dataclass
class StrategyDraft:
    """A strategy definition — from template, AI, or manual builder."""
    id: str = field(default_factory=new_id)
    name: str = ""
    source: str = "manual"    # ai | manual | template | leaderboard
    template_id: str = ""     # if from catalog
    logic_tags: list[str] = field(default_factory=list)     # reversal, breakout, trend, scalp
    factor_ids: list[str] = field(default_factory=list)     # which factors used
    trigger_modes: list[str] = field(default_factory=list)  # pre_limit, rejection, etc
    entry_mode: str = ""      # limit | market | touch_confirm
    exit_rules: dict[str, Any] = field(default_factory=dict)  # stop_type, tp_type, rr_target
    risk_rules: dict[str, Any] = field(default_factory=dict)  # risk_per_trade, max_dd
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    status: StrategyStatus = "draft"
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
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    results: list[str] = field(default_factory=list)  # SimulationResult IDs


@dataclass
class SimulationResult:
    """Result of one strategy in one simulation job."""
    id: str = field(default_factory=new_id)
    job_id: str = ""
    strategy_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_count: int = 0
    avg_rr: float = 0.0
    score: float = 0.0        # composite ranking score
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
    current_pnl: float = 0.0
    current_drawdown: float = 0.0
    open_positions: int = 0
    total_trades: int = 0
    started_at: float = field(default_factory=time.time)
    last_action_at: float = 0.0
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
