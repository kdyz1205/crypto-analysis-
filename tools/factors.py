"""Tool 3: Factor Management — CRUD with lifecycle: core → candidate → validated."""

from __future__ import annotations
import json, time
from pathlib import Path
from dataclasses import asdict
from .types import FactorDefinition, new_id
from .audit import write_audit

DATA_ROOT = Path(__file__).parent.parent / "data" / "factors"


def list_factors(stage: str = "core") -> list[dict]:
    """List factors by stage."""
    from server.factors.factor_engine import FACTOR_POOL
    if stage == "core":
        builtin = [asdict(FactorDefinition(
            id=f.factor_id, name=f.name, category=f.category,
            indicator=f.indicator, params=f.params,
            condition=f.condition, threshold=f.threshold,
            stage="core", source="system", description=f.description,
        )) for f in FACTOR_POOL]
        return builtin + _load_stage("core")
    return _load_stage(stage)


def create_factor(factor: FactorDefinition) -> dict:
    """Create a new factor in candidate pool."""
    factor.stage = "candidate"
    factor.id = factor.id or new_id()
    factor.confidence = 0.0  # untested
    _save_factor(factor)
    write_audit("agent", "factor_created", "factor", factor.id, {"name": factor.name, "stage": "candidate"})
    return asdict(factor)


def record_factor_test(factor_id: str, score: float, trade_count: int, symbol: str = "", timeframe: str = "") -> dict:
    """Record a backtest result for a candidate factor. Returns updated factor."""
    factor = _load_factor(factor_id)
    if not factor:
        return {"error": "not found"}

    # Initialize test history
    if "test_history" not in factor:
        factor["test_history"] = []
    factor["test_history"].append({
        "score": score, "trades": trade_count,
        "symbol": symbol, "timeframe": timeframe,
        "tested_at": time.time(),
    })
    factor["test_count"] = len(factor["test_history"])
    factor["avg_score"] = sum(t["score"] for t in factor["test_history"]) / len(factor["test_history"])
    factor["total_trades"] = sum(t["trades"] for t in factor["test_history"])

    _save_factor_dict(factor)
    return factor


def promote_factor(factor_id: str, to_stage: str = "validated") -> dict:
    """Promote factor with guard check."""
    from risk.guards import check_factor_promotion

    factor = _load_factor(factor_id)
    if not factor:
        return {"error": "not found"}

    test_count = factor.get("test_count", 0)
    avg_score = factor.get("avg_score", 0)
    total_trades = factor.get("total_trades", 0)

    ok, reason = check_factor_promotion(total_trades, avg_score, test_count)
    if not ok:
        return {"error": f"promotion blocked: {reason}", "factor_id": factor_id}

    # Move to new stage
    old_stage = factor.get("stage", "candidate")
    factor["stage"] = to_stage
    factor["promoted_at"] = time.time()

    # Remove from old location
    old_path = DATA_ROOT / old_stage / f"{factor_id}.json"
    if old_path.exists():
        old_path.unlink()

    _save_factor_dict(factor)
    write_audit("system", "factor_promoted", "factor", factor_id,
                {"from": old_stage, "to": to_stage, "score": avg_score, "tests": test_count}, reason="passed guard")
    return factor


def _load_stage(stage: str) -> list[dict]:
    d = DATA_ROOT / stage
    d.mkdir(parents=True, exist_ok=True)
    factors = []
    for f in d.glob("*.json"):
        try:
            factors.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return factors


def _load_factor(factor_id: str) -> dict | None:
    """Load factor by ID — checks disk files first, then FACTOR_POOL for built-ins."""
    for stage in ("candidate", "validated", "core"):
        p = DATA_ROOT / stage / f"{factor_id}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    # Fall back to built-in FACTOR_POOL (core factors stored in memory only)
    try:
        from server.factors.factor_engine import FACTOR_POOL
        for f in FACTOR_POOL:
            if f.factor_id == factor_id:
                return asdict(FactorDefinition(
                    id=f.factor_id, name=f.name, category=f.category,
                    indicator=f.indicator, params=f.params,
                    condition=f.condition, threshold=f.threshold,
                    stage="core", source="system", description=f.description,
                ))
    except Exception:
        pass
    return None


def _save_factor(factor: FactorDefinition):
    d = DATA_ROOT / factor.stage
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{factor.id}.json").write_text(json.dumps(asdict(factor), indent=2), encoding="utf-8")


def _save_factor_dict(factor: dict):
    stage = factor.get("stage", "candidate")
    d = DATA_ROOT / stage
    d.mkdir(parents=True, exist_ok=True)
    fid = factor.get("id", new_id())
    (d / f"{fid}.json").write_text(json.dumps(factor, indent=2), encoding="utf-8")
