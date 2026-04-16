"""Tool 6: Ranking — structured leaderboard with history and traceability."""

from __future__ import annotations
import json, time
from pathlib import Path
from dataclasses import asdict
from .types import LeaderboardEntry, SimulationResult, LiveDeploymentDraft, new_id
from .audit import write_audit

LB_DIR = Path(__file__).parent.parent / "data" / "leaderboard"
ENTRIES_DIR = LB_DIR / "entries"
INDEX_FILE = LB_DIR / "index.json"
LIVE_DRAFTS_DIR = Path(__file__).parent.parent / "data" / "strategies" / "live_drafts"
MAX_LEADERBOARD = 50
MIN_TRADES = 5
MIN_SCORE = 0.4


def update_leaderboard(results: list[SimulationResult], strategy_meta: dict = None, generation: int = 0) -> list[dict]:
    """Add results to leaderboard. Each entry stored individually."""
    ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
    index = _load_index()
    meta = strategy_meta or {}

    for r in results:
        eligible = r.trade_count >= MIN_TRADES and r.score >= MIN_SCORE
        entry = LeaderboardEntry(
            strategy_id=r.strategy_id,
            strategy_name=meta.get(r.strategy_id, {}).get("name", r.strategy_id[:8]),
            source=meta.get(r.strategy_id, {}).get("source", "ai"),
            symbol=r.symbol, timeframe=r.timeframe,
            trigger_modes=meta.get(r.strategy_id, {}).get("trigger_modes", []),
            factor_ids=meta.get(r.strategy_id, {}).get("factor_ids", []),
            return_pct=r.return_pct, win_rate=r.win_rate, profit_factor=r.profit_factor,
            sharpe_ratio=r.sharpe_ratio, max_drawdown_pct=r.max_drawdown_pct,
            trade_count=r.trade_count, score=r.score,
            deployment_eligible=eligible, generation=generation,
            batch_id=r.batch_id, simulation_job_id=r.job_id,
        )
        (ENTRIES_DIR / f"{entry.id}.json").write_text(json.dumps(asdict(entry), indent=2), encoding="utf-8")
        index.append({"id": entry.id, "score": entry.score, "strategy_id": entry.strategy_id,
                       "symbol": entry.symbol, "timeframe": entry.timeframe, "generation": generation})

    index.sort(key=lambda e: -e.get("score", 0))
    index = index[:MAX_LEADERBOARD]
    _save_index(index)
    write_audit("system", "leaderboard_updated", "leaderboard", "", {"new": len(results), "total": len(index), "gen": generation})
    return get_leaderboard(20)


def get_leaderboard(limit: int = 20) -> list[dict]:
    index = _load_index()
    entries = []
    for ref in index[:limit]:
        f = ENTRIES_DIR / f"{ref['id']}.json"
        if f.exists():
            try: entries.append(json.loads(f.read_text(encoding="utf-8")))
            except: pass
    return entries


def get_entry(entry_id: str) -> dict | None:
    f = ENTRIES_DIR / f"{entry_id}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def create_live_draft(entry_id: str, capital: float = 100.0, risk: float = 0.01) -> dict:
    entry = get_entry(entry_id)
    if not entry: return {"error": "not found"}
    draft = LiveDeploymentDraft(strategy_id=entry["strategy_id"], leaderboard_entry_id=entry_id,
        capital_allocation=capital, risk_per_trade=risk, symbols=[entry["symbol"]], timeframes=[entry["timeframe"]])
    LIVE_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DRAFTS_DIR / f"{draft.id}.json").write_text(json.dumps(asdict(draft), indent=2), encoding="utf-8")
    write_audit("user", "live_draft_created", "deployment", draft.id, {"strategy_id": entry["strategy_id"], "capital": capital})
    return asdict(draft)


def _load_index():
    if INDEX_FILE.exists():
        try: return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except: pass
    return []

def _save_index(index):
    LB_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")
