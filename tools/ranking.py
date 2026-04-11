"""Tool 6: Ranking — update leaderboard, mark deployment eligibility."""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import asdict
from .types import LeaderboardEntry, SimulationResult, LiveDeploymentDraft, new_id

LB_DIR = Path(__file__).parent.parent / "data" / "leaderboard"
LIVE_DRAFTS_DIR = Path(__file__).parent.parent / "data" / "strategies" / "live_drafts"
MAX_LEADERBOARD = 50
MIN_TRADES_FOR_ELIGIBLE = 5
MIN_SCORE_FOR_ELIGIBLE = 0.4


def update_leaderboard(results: list[SimulationResult], strategy_meta: dict = None) -> list[dict]:
    """Add simulation results to leaderboard."""
    LB_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing
    entries = []
    lb_file = LB_DIR / "leaderboard.json"
    if lb_file.exists():
        try:
            entries = [LeaderboardEntry(**e) for e in json.loads(lb_file.read_text(encoding="utf-8"))]
        except Exception:
            entries = []

    # Add new results
    meta = strategy_meta or {}
    for r in results:
        eligible = r.trade_count >= MIN_TRADES_FOR_ELIGIBLE and r.score >= MIN_SCORE_FOR_ELIGIBLE
        entry = LeaderboardEntry(
            strategy_id=r.strategy_id,
            strategy_name=meta.get(r.strategy_id, {}).get("name", r.strategy_id),
            source=meta.get(r.strategy_id, {}).get("source", "ai"),
            symbol=r.symbol,
            timeframe=r.timeframe,
            trigger_modes=meta.get(r.strategy_id, {}).get("trigger_modes", []),
            factor_ids=meta.get(r.strategy_id, {}).get("factor_ids", []),
            return_pct=r.return_pct,
            win_rate=r.win_rate,
            profit_factor=r.profit_factor,
            sharpe_ratio=r.sharpe_ratio,
            max_drawdown_pct=r.max_drawdown_pct,
            trade_count=r.trade_count,
            score=r.score,
            deployment_eligible=eligible,
        )
        entries.append(entry)

    # Sort and cap
    entries.sort(key=lambda e: -e.score)
    entries = entries[:MAX_LEADERBOARD]

    # Save
    lb_file.write_text(json.dumps([asdict(e) for e in entries], indent=2), encoding="utf-8")
    return [asdict(e) for e in entries]


def get_leaderboard(limit: int = 20) -> list[dict]:
    """Read leaderboard."""
    lb_file = LB_DIR / "leaderboard.json"
    if not lb_file.exists():
        return []
    try:
        entries = json.loads(lb_file.read_text(encoding="utf-8"))
        return entries[:limit]
    except Exception:
        return []


def create_live_draft(leaderboard_entry_id: str, capital: float = 100.0, risk_per_trade: float = 0.01) -> dict:
    """Create a live deployment draft from a leaderboard entry."""
    lb = get_leaderboard(50)
    entry = next((e for e in lb if e["id"] == leaderboard_entry_id), None)
    if not entry:
        return {"error": "Entry not found"}

    draft = LiveDeploymentDraft(
        strategy_id=entry["strategy_id"],
        leaderboard_entry_id=leaderboard_entry_id,
        capital_allocation=capital,
        risk_per_trade=risk_per_trade,
        symbols=[entry["symbol"]],
        timeframes=[entry["timeframe"]],
        status="draft",
    )
    LIVE_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DRAFTS_DIR / f"{draft.id}.json").write_text(json.dumps(asdict(draft), indent=2), encoding="utf-8")
    return asdict(draft)
