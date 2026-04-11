"""Tool 3: Factor Management — read, create, promote factors."""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import asdict
from .types import FactorDefinition, AuditEntry, new_id

DATA_ROOT = Path(__file__).parent.parent / "data" / "factors"


def list_factors(stage: str = "core") -> list[dict]:
    """List factors by stage (core/candidate/validated)."""
    # Built-in factors from factor_engine
    from server.factors.factor_engine import FACTOR_POOL
    builtin = [FactorDefinition(
        id=f.factor_id, name=f.name, category=f.category,
        indicator=f.indicator, params=f.params,
        condition=f.condition, threshold=f.threshold,
        stage="core", source="system", description=f.description,
    ) for f in FACTOR_POOL]

    # Load from disk
    stage_dir = DATA_ROOT / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    disk_factors = []
    for f in stage_dir.glob("*.json"):
        try:
            disk_factors.append(FactorDefinition(**json.loads(f.read_text(encoding="utf-8"))))
        except Exception:
            pass

    if stage == "core":
        return [asdict(f) for f in builtin + disk_factors]
    return [asdict(f) for f in disk_factors]


def create_factor(factor: FactorDefinition) -> dict:
    """Create a new factor in candidate pool."""
    factor.stage = "candidate"
    factor.id = factor.id or new_id()
    stage_dir = DATA_ROOT / "candidate"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / f"{factor.id}.json").write_text(json.dumps(asdict(factor), indent=2), encoding="utf-8")
    return asdict(factor)


def promote_factor(factor_id: str, to_stage: str = "validated") -> bool:
    """Promote factor: candidate → validated, or validated → core."""
    for from_stage in ("candidate", "validated"):
        src = DATA_ROOT / from_stage / f"{factor_id}.json"
        if src.exists():
            dst_dir = DATA_ROOT / to_stage
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{factor_id}.json"
            data = json.loads(src.read_text(encoding="utf-8"))
            data["stage"] = to_stage
            dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
            src.unlink()
            return True
    return False
