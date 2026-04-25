"""Prove IndexedDB cache makes F5 refresh instant.

Flow:
  1. Open /v2 (cold IDB)
  2. Wait for chart to show bars — measure how long
  3. Verify IndexedDB got populated with HYPEUSDT 4h bars
  4. Reload the page (F5) — IDB should be populated from step 2
  5. Measure how long until chart shows bars again

Expected:
  Cold load: 1-3s (server fetch + render)
  Warm F5:  < 500ms (IDB quick-paint → chart.setData)

Output: appended to data/logs/ui_tests/idb_perceived_latency.log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = "http://localhost:8000/v2"
LOG_DIR = Path(__file__).resolve().parents[1] / "data" / "logs" / "ui_tests"
LOG_DIR.mkdir(parents=True, exist_ok=True)


INIT_SCRIPT = """
  window.__chartLoadedAt = null;
  window.__chartLoadedAtStart = performance.now();
  const orig = console.log;
  console.log = function(...args) {
    const first = args[0];
    if (typeof first === 'string' && first.includes('[chart] LOADED')) {
      if (window.__chartLoadedAt == null) {
        window.__chartLoadedAt = performance.now();
      }
    }
    return orig.apply(console, args);
  };
"""


def wait_for_chart_bars(page, timeout_ms: int = 10000) -> float:
    """Wait until the LOADED-banner log appears. Relies on the init
    script having been registered BEFORE navigation."""
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        loaded = page.evaluate("window.__chartLoadedAt")
        started = page.evaluate("window.__chartLoadedAtStart")
        if loaded and started:
            return loaded - started
        page.wait_for_timeout(30)
    return float("inf")


def count_idb_bars(page, symbol: str = "HYPEUSDT", interval: str = "4h") -> int:
    """Query the IndexedDB cache directly via the window debug hook."""
    return page.evaluate(f"""
        async () => {{
            if (!window.__ohlcvCache) return -1;
            const row = await window.__ohlcvCache.getCached('{symbol}', '{interval}');
            return row?.bars?.length || 0;
        }}
    """)


def main():
    results = {"ts": int(time.time())}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 900})
        # Install the timing hook BEFORE any page script runs, for both
        # first load and reload.
        ctx.add_init_script(INIT_SCRIPT)
        page = ctx.new_page()
        page.on("console", lambda msg: print(f"[console.{msg.type}] {msg.text[:200]}") if msg.type in ("error", "warning") else None)

        # ── First visit (cold IDB) ──────────────────────────────
        t0 = time.time()
        page.goto(BASE, wait_until="domcontentloaded", timeout=30000)
        cold_ms = wait_for_chart_bars(page, timeout_ms=20000)
        results["cold_chart_visible_ms"] = round(cold_ms, 0)
        print(f"[cold] chart bars visible in: {cold_ms:.0f}ms")

        # Wait a beat for IDB write to settle
        page.wait_for_timeout(1500)
        idb_count = count_idb_bars(page)
        results["idb_bars_after_first_load"] = idb_count
        print(f"[idb] after first load: {idb_count} bars stored")

        # ── 5 consecutive F5 refreshes (warm IDB) ──────────────
        warm_times = []
        for i in range(5):
            page.reload(wait_until="domcontentloaded", timeout=30000)
            ms = wait_for_chart_bars(page, timeout_ms=20000)
            warm_times.append(ms)
            print(f"[warm {i+1}] F5 chart visible in: {ms:.0f}ms")
        results["warm_chart_visible_ms"] = warm_times
        results["warm_median_ms"] = sorted(warm_times)[len(warm_times)//2]

        idb_count2 = count_idb_bars(page)
        results["idb_bars_after_reload"] = idb_count2
        print(f"[idb] after 5 reloads: {idb_count2} bars")

        median = results["warm_median_ms"]
        results["speedup_x"] = round(cold_ms / median, 1) if median > 0 else None
        browser.close()

    log_path = LOG_DIR / f"idb_perceived_latency_{results['ts']}.log"
    log_path.write_text(json.dumps(results, indent=2))
    print()
    print(json.dumps(results, indent=2))
    print(f"\nlog: {log_path}")

    passed = (
        idb_count > 50
        and results["warm_median_ms"] < cold_ms
        and results["warm_median_ms"] < 3000
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
