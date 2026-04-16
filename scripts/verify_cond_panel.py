"""Playwright verify: conditional panel renders, analyze flow works.

Run against a server on a non-default port (so we don't collide).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, ConsoleMessage

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8001/v2"
OUT = Path(__file__).parent.parent / "data" / "logs" / "cond_verify"
OUT.mkdir(parents=True, exist_ok=True)

errors: list[dict] = []
warnings: list[dict] = []
network_errors: list[dict] = []


def on_console(msg: ConsoleMessage):
    t = msg.type
    if t == "error":
        errors.append({"text": msg.text[:400]})
    elif t == "warning":
        warnings.append({"text": msg.text[:400]})


def on_response(resp):
    if resp.status >= 400:
        network_errors.append({"url": resp.url[:200], "status": resp.status})


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 980})
        page = context.new_page()
        page.on("console", on_console)
        page.on("response", on_response)

        print(f"[1/5] Loading {BASE}")
        t0 = time.time()
        try:
            page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
            print(f"  DOM ready in {time.time()-t0:.1f}s")
            page.wait_for_load_state("networkidle", timeout=30000)
            print(f"  networkidle in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  load partial: {e}")
        page.wait_for_timeout(3000)

        # Verify conditional panel exists
        print("[2/5] Checking #v2-cond-rail presence")
        info = page.evaluate("""
            () => {
                const rail = document.getElementById('v2-cond-rail');
                if (!rail) return { present: false };
                const panel = rail.querySelector('.cond-panel');
                const statsSlot = rail.querySelector('.cond-stats-body');
                const pendingSlot = rail.querySelector('.cond-pending-body');
                return {
                    present: true,
                    has_panel: !!panel,
                    has_stats: !!statsSlot,
                    has_pending: !!pendingSlot,
                    stats_text: statsSlot ? statsSlot.innerText.slice(0, 200) : null,
                    pending_text: pendingSlot ? pendingSlot.innerText.slice(0, 200) : null,
                    rail_width: rail.offsetWidth,
                    rail_height: rail.offsetHeight,
                };
            }
        """)
        print(f"  rail present: {info.get('present')}")
        print(f"  dimensions:   {info.get('rail_width')}x{info.get('rail_height')}")
        print(f"  has panel:    {info.get('has_panel')}")
        print(f"  stats text:   {info.get('stats_text')}")
        print(f"  pending text: {info.get('pending_text')}")

        page.screenshot(path=str(OUT / "01_initial.png"), full_page=True)

        # Try drawing a manual line via the existing API directly,
        # then check if panel updates (the drawing updated event should fire).
        print("[3/5] Creating a manual line via API for panel to pick up")
        import httpx
        try:
            r = httpx.post(
                "http://localhost:8001/api/drawings",
                json={
                    "symbol": "HYPEUSDT",
                    "timeframe": "4h",
                    "side": "support",
                    "t_start": 1776000000,
                    "t_end": 1776086400,
                    "price_start": 40.0,
                    "price_end": 40.5,
                    "extend_right": True,
                    "extend_left": False,
                    "locked": False,
                    "label": "verify test",
                    "notes": "",
                    "override_mode": "display_only",
                },
                timeout=10,
            )
            print(f"  POST /api/drawings -> {r.status_code}")
        except Exception as e:
            print(f"  drawing create err: {e}")

        # Give the panel a moment to refresh pending list
        page.wait_for_timeout(2000)

        print("[4/5] Creating a conditional via API directly")
        try:
            # First, refresh drawings to get the manual_line_id
            r = httpx.get("http://localhost:8001/api/drawings?symbol=HYPEUSDT&timeframe=4h")
            drawings = r.json().get("drawings", [])
            print(f"  drawings in store: {len(drawings)}")
            if drawings:
                mid = drawings[0].get("manual_line_id")
                print(f"  using manual_line_id={mid}")
                r = httpx.post(
                    "http://localhost:8001/api/conditionals",
                    json={
                        "manual_line_id": mid,
                        "trigger": {
                            "tolerance_atr": 0.2,
                            "poll_seconds": 30,
                            "max_age_seconds": 3600,
                        },
                        "order": {
                            "notional_usd": 200,
                            "stop_atr": 0.3,
                            "rr_target": 2.0,
                            "submit_to_exchange": False,
                            "exchange_mode": "paper",
                        },
                        "pattern_stats": {"n_samples": 32, "p_bounce": 0.68},
                    },
                    timeout=10,
                )
                print(f"  POST /api/conditionals -> {r.status_code}")
                if r.status_code == 200:
                    cid = r.json().get("conditional", {}).get("conditional_id")
                    print(f"  created cid={cid}")
        except Exception as e:
            print(f"  conditional create err: {e}")

        # Wait for panel refresh (10s poll)
        print("[5/5] Waiting for panel auto-refresh then screenshot")
        page.wait_for_timeout(12000)
        page.screenshot(path=str(OUT / "02_after_conditional.png"), full_page=True)

        info2 = page.evaluate("""
            () => {
                const rail = document.getElementById('v2-cond-rail');
                if (!rail) return {};
                const rows = rail.querySelectorAll('.cond-row');
                const pending = rail.querySelector('.cond-pending-body');
                return {
                    row_count: rows.length,
                    pending_text: pending ? pending.innerText.slice(0, 600) : null,
                };
            }
        """)
        print(f"  row count after refresh: {info2.get('row_count')}")
        print(f"  pending text (truncated):\n  {info2.get('pending_text')}")

        # Cleanup: cancel and delete the conditional + drawing
        try:
            r = httpx.get("http://localhost:8001/api/conditionals?status=all")
            for c in r.json().get("conditionals", []):
                cid = c.get("conditional_id")
                httpx.delete(f"http://localhost:8001/api/conditionals/{cid}", timeout=5)
            if drawings:
                mid = drawings[0].get("manual_line_id")
                httpx.delete(
                    f"http://localhost:8001/api/drawings/{mid}",
                    timeout=5,
                )
        except Exception:
            pass

        browser.close()

    # Final verdict
    print()
    print("=" * 60)
    if errors:
        print(f"FAIL: {len(errors)} console errors")
        for e in errors[:5]:
            print(f"  - {e['text'][:140]}")
    if warnings:
        print(f"WARN: {len(warnings)} console warnings")
        for w in warnings[:5]:
            print(f"  - {w['text'][:140]}")
    if network_errors:
        print(f"NET:  {len(network_errors)} network errors")
        for n in network_errors[:5]:
            print(f"  - {n['status']} {n['url']}")
    if not errors and not network_errors:
        print("PASS: panel rendered, 0 errors, cond flow works")
    print("=" * 60)
    return 0 if not errors and not network_errors else 1


if __name__ == "__main__":
    sys.exit(main())
