"""One-shot: migrate live_instance JSONs to the new P&L schema.

Old: current_pnl (single field, indistinguishable virtual vs real)
New: pattern_virtual_pnl (simulated outcome) + realized_pnl_usd (exchange)

This moves any non-zero current_pnl into pattern_virtual_pnl and
zeroes realized_pnl_usd (no real fills happened historically).

Idempotent: instances already on the new schema are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

INSTANCE_DIR = Path(__file__).parent.parent / "data" / "strategies" / "live_instances"


def main():
    if not INSTANCE_DIR.exists():
        print(f"[migrate_pnl] {INSTANCE_DIR} does not exist — nothing to do.")
        return

    files = sorted(INSTANCE_DIR.glob("*.json"))
    print(f"[migrate_pnl] scanning {len(files)} instance file(s) in {INSTANCE_DIR}")

    migrated = 0
    skipped = 0
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {p.name}: parse error {e}")
            continue

        # Already migrated?
        if "pattern_virtual_pnl" in d and "realized_pnl_usd" in d:
            skipped += 1
            continue

        # Migrate
        old_pnl = float(d.get("current_pnl", 0.0))
        d["pattern_virtual_pnl"] = old_pnl
        d["realized_pnl_usd"] = 0.0
        # Keep current_pnl as alias for backward compat reads
        d["current_pnl"] = old_pnl

        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"  MIGRATED {p.name}: old_pnl={old_pnl} → pattern_virtual_pnl={old_pnl}, realized_pnl_usd=0.0")
        migrated += 1

    print(f"\n[migrate_pnl] done: {migrated} migrated, {skipped} already on new schema")


if __name__ == "__main__":
    main()
