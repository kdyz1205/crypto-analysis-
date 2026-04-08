"""
Audit subscriber — writes every event to trade_audit.jsonl.

Supplements (does not replace) the existing audit writes in agent_brain.py.
Eventually Phase 3 Task 7 will remove the scattered calls and leave this as the
sole writer, but until then it's additive so we don't lose data.
"""

from pathlib import Path

from ..core.events import bus, Event

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_LOG = PROJECT_ROOT / "trade_audit.jsonl"


async def audit_writer(event: Event):
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
    except Exception as e:
        print(f"[AuditSubscriber] Write failed: {e}")


def register():
    for prefix in [
        "signal.*", "order.*", "position.*", "risk.*",
        "agent.*", "ops.*", "summary.*",
    ]:
        bus.subscribe(prefix, audit_writer)
    print("[AuditSubscriber] Registered for 7 event prefixes")
