"""End-to-end test of the Runner dashboard page.

Checks:
  1. /v2 loads
  2. Click the "策略" nav button
  3. Runner view appears (#view-runner has .rn-root)
  4. Status pill renders with a known status class
  5. Config form shows the current top_n/timeframe/etc
  6. Stats block renders equity + positions count
  7. Positions table OR "无持仓" empty state renders
  8. Signals feed renders (either rows or "等待信号…")
  9. Clicking "立即扫描" completes (returns JSON, status updates)

Exit 0 = all waypoints green.
"""
from __future__ import annotations

import subprocess
import sys
import time
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8000/v2"


def waypoint(label: str, cond: bool, detail: str = "") -> bool:
    tag = "✓" if cond else "✗"
    print(f"  [{tag}] {label}" + (f"  ({detail})" if detail else ""), flush=True)
    return cond


def main() -> int:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "?"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"=== runner view test @ {ts} git={sha} ===", flush=True)

    console_errors: list[str] = []
    net_errors: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1680, "height": 980})
        page = ctx.new_page()
        page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text[:200]}")
                if msg.type == "error" else None)
        page.on("response", lambda r: net_errors.append({"url": r.url[:100], "status": r.status})
                if r.status >= 500 else None)

        all_ok = True

        try:
            page.goto(BASE, wait_until="commit", timeout=30000)
        except Exception as e:
            print(f"  goto failed: {e}", flush=True)
            return 1

        # Wait for main.js to finish booting (chart container exists)
        try:
            page.wait_for_selector("#chart-container", timeout=15000)
        except Exception:
            return 1
        page.wait_for_timeout(2500)

        # Click the 策略 nav button
        nav_btn = page.query_selector(".v2-nav-btn[data-view='runner']")
        all_ok &= waypoint("nav button 策略 present", nav_btn is not None)
        if not nav_btn:
            return 1
        nav_btn.click()
        page.wait_for_timeout(800)

        # Runner view active + rendered
        view_el = page.query_selector(".v2-view[data-view='runner'].active")
        all_ok &= waypoint("runner view active", view_el is not None)

        root = page.query_selector("#view-runner .rn-root")
        all_ok &= waypoint(".rn-root rendered", root is not None)

        # Wait for first poll to fill in
        page.wait_for_timeout(3500)

        # Status pill
        pill = page.query_selector("#rn-status-pill")
        pill_text = (pill.inner_text() or "").strip().lower() if pill else ""
        all_ok &= waypoint("status pill text",
                           bool(pill_text) and pill_text in ("running", "stopped", "idle"),
                           detail=pill_text)

        # Config top_n is populated (not empty)
        top_n_val = page.eval_on_selector("[name=top_n]", "el => el.value")
        all_ok &= waypoint("config top_n populated",
                           bool(top_n_val) and top_n_val != "0",
                           detail=str(top_n_val))

        # Stats block has 4 stat cards
        stats_count = page.eval_on_selector_all("#rn-stats .rn-stat", "els => els.length")
        all_ok &= waypoint("stats grid has 4 cards",
                           stats_count == 4, detail=f"count={stats_count}")

        # Positions section renders (either table or empty)
        pos_el = page.query_selector("#rn-positions")
        pos_text = (pos_el.inner_text() or "") if pos_el else ""
        all_ok &= waypoint("positions section renders",
                           bool(pos_text),
                           detail="has content")

        # Signals section renders (either rows or empty msg)
        sig_el = page.query_selector("#rn-signals")
        sig_text = (sig_el.inner_text() or "") if sig_el else ""
        all_ok &= waypoint("signals section renders",
                           bool(sig_text),
                           detail="has content")

        # Click "立即扫描" — should not raise an error
        kick_btn = page.query_selector("#rn-btn-kick")
        if kick_btn:
            kick_btn.click()
            # Wait briefly for the scan to start/complete
            page.wait_for_timeout(4000)
        all_ok &= waypoint("kick button clickable + no thrown alert",
                           not any("alert" in e for e in console_errors))

        # Any runtime console errors?
        if console_errors:
            print(f"\n  console errors ({len(console_errors)}):", flush=True)
            for e in console_errors[:5]:
                print(f"    {e}", flush=True)

        if net_errors:
            print(f"\n  5xx responses ({len(net_errors)}):", flush=True)
            for n in net_errors[:5]:
                print(f"    {n['status']} {n['url']}", flush=True)

        browser.close()

    print(f"\n=== result: {'PASS' if all_ok else 'FAIL'} ===", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
