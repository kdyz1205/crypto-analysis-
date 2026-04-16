"""Tool 4: Strategy Generation — compose factors into strategy drafts."""

from __future__ import annotations
import json, random
from pathlib import Path
from dataclasses import asdict
from .types import StrategyDraft, StrategyConfig, MarketScope, ConditionRule, EntryRules, ExitRules, RiskProfile, new_id
from .audit import write_audit

DRAFTS_DIR = Path(__file__).parent.parent / "data" / "strategies" / "drafts"


def create_strategy(draft: StrategyDraft) -> dict:
    """Save a strategy draft."""
    draft.id = draft.id or new_id()
    draft.status = "draft"
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    (DRAFTS_DIR / f"{draft.id}.json").write_text(json.dumps(asdict(draft), indent=2), encoding="utf-8")
    write_audit("agent", "strategy_generated", "strategy", draft.id, {
        "name": draft.name, "source": draft.source, "status": draft.status,
        "symbols": draft.symbols, "trigger_modes": draft.trigger_modes,
    })
    return asdict(draft)


def generate_from_template(template_id: str, symbol: str, timeframe: str,
                           equity: float = 10000, generation: int = 0, batch_id: str = "") -> dict:
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
        generation=generation,
        batch_id=batch_id,
    )
    return create_strategy(draft)


def generate_ai_strategy(symbols: list[str] = None, timeframes: list[str] = None,
                         generation: int = 0, batch_id: str = "") -> dict:
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
        generation=generation,
        batch_id=batch_id,
    )
    return create_strategy(draft)


def create_from_config(name: str, config_dict: dict, source: str = "manual",
                       lineage: dict | None = None) -> dict:
    """Create a strategy draft from a structured StrategyConfig dict.

    Optional `lineage` dict can include:
      source_pattern_id, decision_rule, pattern_decision, pattern_ev,
      pattern_confidence, pattern_reason, generation, batch_id
    """
    lineage = lineage or {}
    # Parse config dict into typed structure
    market = config_dict.get("market", {})
    entry = config_dict.get("entry", {})
    exit_r = config_dict.get("exit", {})
    risk = config_dict.get("risk", {})
    conditions = config_dict.get("conditions", [])

    cfg = StrategyConfig(
        market=MarketScope(
            symbols=market.get("symbols", ["BTCUSDT"]),
            main_tf=market.get("main_tf", "4h"),
            confirm_tf=market.get("confirm_tf", ""),
            entry_tf=market.get("entry_tf", ""),
        ),
        logic_tags=config_dict.get("logic_tags", ["reversal"]),
        logic_combine=config_dict.get("logic_combine", "OR"),
        conditions=[ConditionRule(**c) for c in conditions],
        entry=EntryRules(
            modes=entry.get("modes", ["rejection"]),
            logic_combine=entry.get("logic_combine", "OR"),
            offset_pct=entry.get("offset_pct", 0.0),
        ),
        exit=ExitRules(
            stop_type=exit_r.get("stop_type", "structure"),
            stop_pct=exit_r.get("stop_pct", 1.0),
            stop_atr_mult=exit_r.get("stop_atr_mult", 1.5),
            tp_type=exit_r.get("tp_type", "rr"),
            rr_target=exit_r.get("rr_target", 2.0),
            tp_pct=exit_r.get("tp_pct", 3.0),
            trailing_pct=exit_r.get("trailing_pct", 1.0),
        ),
        risk=RiskProfile(
            risk_per_trade=risk.get("risk_per_trade", 0.01),
            max_concurrent=risk.get("max_concurrent", 3),
            max_daily_loss_pct=risk.get("max_daily_loss_pct", 5.0),
            max_drawdown_pct=risk.get("max_drawdown_pct", 10.0),
            max_consecutive_losses=risk.get("max_consecutive_losses", 5),
            auto_pause_on_dd=risk.get("auto_pause_on_dd", True),
        ),
    )

    # Build draft with both new config and legacy flat fields
    draft = StrategyDraft(
        name=name,
        source=source,
        config=cfg,
        # Legacy flat fields for backward compat with backtest engine
        logic_tags=cfg.logic_tags,
        factor_ids=[c.indicator for c in cfg.conditions if c.enabled],
        trigger_modes=cfg.entry.modes,
        entry_mode=cfg.entry.modes[0] if cfg.entry.modes else "",
        exit_rules={"stop_type": cfg.exit.stop_type, "tp_type": cfg.exit.tp_type, "rr_target": cfg.exit.rr_target},
        risk_rules={"risk_per_trade": cfg.risk.risk_per_trade, "max_dd": cfg.risk.max_drawdown_pct},
        symbols=cfg.market.symbols,
        timeframes=[cfg.market.main_tf] + ([cfg.market.confirm_tf] if cfg.market.confirm_tf else []),
        params={"rr_target": cfg.exit.rr_target, "risk_per_trade": cfg.risk.risk_per_trade},
        status="draft",
        # Pattern lineage
        source_pattern_id=lineage.get("source_pattern_id", ""),
        decision_rule=lineage.get("decision_rule", ""),
        pattern_decision=lineage.get("pattern_decision", ""),
        pattern_ev=lineage.get("pattern_ev", 0.0),
        pattern_confidence=lineage.get("pattern_confidence", 0.0),
        pattern_reason=lineage.get("pattern_reason", ""),
        generation=lineage.get("generation", 0),
        batch_id=lineage.get("batch_id", ""),
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
    old_status = d.get("status", "unknown")
    d["status"] = new_status
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    write_audit("system", "strategy_status_changed", "strategy", draft_id, {
        "from": old_status, "to": new_status,
    })
    return True
