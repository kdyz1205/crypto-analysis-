"""Structured audit writer — every important action gets a permanent record."""

from __future__ import annotations
import json, time
from pathlib import Path
from dataclasses import asdict
from .types import AuditEntry

AUDIT_DIR = Path(__file__).parent.parent / "data" / "logs" / "audit"


def write_audit(actor: str, action: str, object_type: str, object_id: str, details: dict = None, reason: str = "") -> AuditEntry:
    """Write a structured audit entry to disk. Returns the entry."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    entry = AuditEntry(
        actor=actor,
        action=action,
        object_type=object_type,
        object_id=object_id,
        details=details or {},
        reason=reason,
    )
    # Append to daily audit file
    day = time.strftime("%Y-%m-%d")
    audit_file = AUDIT_DIR / f"{day}.jsonl"
    with open(audit_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), default=str) + "\n")
    return entry


def read_audit(day: str = None, limit: int = 100) -> list[dict]:
    """Read audit entries. Defaults to today."""
    day = day or time.strftime("%Y-%m-%d")
    audit_file = AUDIT_DIR / f"{day}.jsonl"
    if not audit_file.exists():
        return []
    entries = []
    for line in audit_file.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries[-limit:]
