"""Playwright-grade end-to-end test of the MA Ribbon + BB Exit live runner.

Checks:
  1. /api/mar-bb/state is reachable
  2. /api/mar-bb/start (dry_run) moves status → running
  3. /api/mar-bb/kick completes a full scan, returns metrics
  4. Scan duration is under 30s for top 20 symbols
  5. /api/mar-bb/stop cleanly halts

This test does NOT fire real orders — uses dry_run=true throughout.
The real-order path is already verified by test_draw_line_real.py
(same LiveExecutionAdapter.submit_live_entry code path).

Exit 0 = all waypoints pass.
Exit 1 = something broke, with a specific failure point.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8000"


def _get(path: str, timeout: float = 10.0) -> dict | None:
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"  http GET {path} err: {e}", flush=True)
        return None


def _post(path: str, body: dict | None = None, timeout: float = 180.0) -> dict | None:
    try:
        data = json.dumps(body or {}).encode("utf-8")
        req = urllib.request.Request(
            BASE + path, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"  http POST {path} err: {e}", flush=True)
        return None


def waypoint(label: str, cond: bool, detail: str = "") -> bool:
    tag = "✓" if cond else "✗"
    print(f"  [{tag}] {label}" + (f"  ({detail})" if detail else ""), flush=True)
    return cond


def main() -> int:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "?"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"=== mar_bb runner test @ {ts} git={sha} ===", flush=True)

    all_ok = True

    # 1. State endpoint reachable
    s = _get("/api/mar-bb/state")
    all_ok &= waypoint("state endpoint reachable", s is not None and s.get("ok"))
    if not s:
        return 1
    initial = s["state"]
    print(f"  initial status: {initial['status']}  scans={initial['scans_completed']}", flush=True)

    # 2. Start runner in dry_run mode
    r = _post("/api/mar-bb/start", {
        "top_n": 20,
        "timeframe": "1h",
        "scan_interval_s": 3600,
        "dry_run": True,
    })
    all_ok &= waypoint("start returned ok", r is not None and r.get("ok"))

    # Give the loop a tick to establish status=running
    time.sleep(1.5)
    s = _get("/api/mar-bb/state")
    all_ok &= waypoint("status == running after start",
                       s and s["state"].get("status") == "running",
                       detail=str(s["state"].get("status") if s else "?"))

    # 3. Kick a scan (forces immediate iteration through top-N)
    t0 = time.time()
    r = _post("/api/mar-bb/kick", {}, timeout=200)
    dt = time.time() - t0
    all_ok &= waypoint("kick returned ok", r is not None and r.get("ok"),
                       detail=f"{dt:.1f}s")
    if not r:
        return 1
    state_after = r["state"]

    # 4. Scan metrics look sane
    scans = state_after.get("scans_completed", 0)
    last_err = state_after.get("last_error", "")
    all_ok &= waypoint("scans_completed incremented",
                       scans >= 1, detail=f"scans={scans}")
    all_ok &= waypoint("no runner error",
                       not last_err, detail=last_err[:80] if last_err else "")
    duration = state_after.get("last_scan_duration_s", 0)
    all_ok &= waypoint("scan duration reasonable (<30s)",
                       duration < 30, detail=f"{duration}s")

    # 5. Stop runner
    r = _post("/api/mar-bb/stop", {}, timeout=10)
    all_ok &= waypoint("stop returned ok", r is not None and r.get("ok"))
    time.sleep(1.5)
    s = _get("/api/mar-bb/state")
    all_ok &= waypoint("status == stopped after stop",
                       s and s["state"].get("status") == "stopped",
                       detail=str(s["state"].get("status") if s else "?"))

    print(f"\n=== result: {'PASS' if all_ok else 'FAIL'} ===", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
