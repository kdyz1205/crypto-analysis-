"""Tool 4: Strategy Generation — compose factors into strategy drafts."""

from __future__ import annotations
import json, random
from pathlib import Path
from dataclasses import asdict
from .types import StrategyDraft, new_id

DRAFTS_DIR = Path(__file__).parent.parent / "data" / "strategies" / "drafts"


def create_strategy(draft: StrategyDraft) -> dict:
    """Save a strategy draft."""
    draft.id = draft.id or new_id()
    draft.status = "draft"
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    (DRAFTS_DIR / f"{draft.id}.json").write_text(json.dumps(asdict(draft), indent=2), encoding="utf-8")
    return asdict(draft)


def generate_from_template(template_id: str, symbol: str, timeframe: str, equity: float = 10000) -> dict:
    """Create a strategy draft from a catalog template."""
    from server.strategy.catalog import get_template
    t = get_template(template_id)
    if not t:
        return {"error": f"Unknown template: {template_id}"}
    draft = StrategyDraft(
        name=f"{symbol}-{t.name}",
        source="template",
        template_id=template_id,
        logic_tags=[t.category],
        trigger_modes=list(t.default_trigger_modes),
        symbols=[symbol],
        timeframes=[timeframe],
        params={**t.default_params, "starting_equity": equity},
        status="draft",
    )
    return create_strategy(draft)


def generate_ai_strategy(symbols: list[str] = None, timeframes: list[str] = None) -> dict:
    """AI-generated random strategy from factor pool."""
    from server.factors.factor_engine import FactorEngine
    syms = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT"]
    tfs = timeframes or ["1h", "4h"]
    factors = FactorEngine.generate_candidates(1)[0]
    triggers = random.choice([
        ["rejection", "failed_breakout"],
        ["pre_limit", "rejection"],
        ["rejection", "failed_breakout", "retest"],
    ])
    draft = StrategyDraft(
        name=f"AI-{new_id()[:6]}",
        source="ai",
        logic_tags=["reversal"],
        factor_ids=factors,
        trigger_modes=triggers,
        symbols=syms,
        timeframes=tfs,
        params={"rr_target": random.choice([1.5, 2.0, 2.5, 3.0]), "lookback_bars": random.choice([60, 80, 100])},
        status="pending_simulation",
    )
    return create_strategy(draft)


def list_drafts(status: str = None) -> list[dict]:
    """List all strategy drafts."""
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    drafts = []
    for f in DRAFTS_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if status and d.get("status") != status:
                continue
            drafts.append(d)
        except Exception:
            pass
    return sorted(drafts, key=lambda d: d.get("created_at", 0), reverse=True)


def update_draft_status(draft_id: str, new_status: str) -> bool:
    """Update a draft's status."""
    path = DRAFTS_DIR / f"{draft_id}.json"
    if not path.exists():
        return False
    d = json.loads(path.read_text(encoding="utf-8"))
    d["status"] = new_status
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return True
