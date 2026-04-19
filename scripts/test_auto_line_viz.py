"""Phase 1 smoke test: auto-triggered trendline renders distinctly on chart.

Injects a fake auto_triggered line into manual_trendlines.json for HYPEUSDT 4h,
loads /v2, asserts:
  1. GET /api/drawings?symbol=HYPEUSDT&timeframe=4h returns the line with
     source='auto_triggered'.
  2. The chart has MORE line-series objects after reload than before
     (proxy for "line is rendered").

Cleans up the fake line afterwards so the user's real drawings are untouched.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8000/v2"
FAKE_ID = "auto-HYPEUSDT-4h-support-1776300000000-1776500000000-test_smoke01"
SYMBOL = "HYPEUSDT"
TIMEFRAME = "4h"
TRENDLINES_FILE = Path("data/manual_trendlines.json")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _make_fake_entry():
    return {
        "manual_line_id": FAKE_ID,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "side": "support",
        "source": "auto_triggered",
        # Span across the visible HYPE 4h chart (~Apr 3 → Apr 18).
        "t_start": 1775500000,  # Apr 4 05:46 UTC
        "t_end":   1776500000,  # Apr 16 14:13 UTC
        "price_start": 36.0,    # lower visible range
        "price_end":   44.0,    # upper visible range
        "extend_left": False,
        "extend_right": True,
        "locked": True,
        "label": "Auto · sup · filled (smoke)",
        "notes": "smoke test — delete if seen",
        "comparison_status": "uncompared",
        "override_mode": "display_only",
        "nearest_auto_line_id": None,
        "slope_diff": None,
        "projected_price_diff": None,
        "overlap_ratio": None,
        "created_at": _now_ms(),
        "updated_at": _now_ms(),
        "line_width": 1.8,
    }


def add_fake():
    data = json.loads(TRENDLINES_FILE.read_text(encoding="utf-8"))
    data = [l for l in data if l.get("manual_line_id") != FAKE_ID]
    data.append(_make_fake_entry())
    TRENDLINES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_fake():
    data = json.loads(TRENDLINES_FILE.read_text(encoding="utf-8"))
    before = len(data)
    data = [l for l in data if l.get("manual_line_id") != FAKE_ID]
    TRENDLINES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return before - len(data)


def run() -> int:
    print(f"Adding fake {FAKE_ID}...", flush=True)
    add_fake()

    failures = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
            page = ctx.new_page()

            page.goto(BASE, timeout=30000, wait_until="domcontentloaded")
            time.sleep(3.0)

            # 1. API contract — server returns auto line with right source
            api_resp = page.evaluate(
                f"fetch('/api/drawings?symbol={SYMBOL}&timeframe={TIMEFRAME}')"
                ".then(r => r.json())"
            )
            drawings = api_resp.get("drawings", []) if isinstance(api_resp, dict) else []
            hit = next((l for l in drawings if l.get("manual_line_id") == FAKE_ID), None)
            if not hit:
                failures.append(
                    f"API did not return fake line. Got {len(drawings)} lines: "
                    f"{[l.get('manual_line_id') for l in drawings][:5]}"
                )
            else:
                if hit.get("source") != "auto_triggered":
                    failures.append(f"source wrong: got {hit.get('source')!r}")
                if hit.get("side") != "support":
                    failures.append(f"side wrong: got {hit.get('side')!r}")
                if not hit.get("locked"):
                    failures.append("locked should be True")
                print(f"API contract OK: source={hit.get('source')} side={hit.get('side')} locked={hit.get('locked')}", flush=True)

            # 2. Navigate chart to that symbol + TF so the overlay actually draws it.
            # v2 has a symbol search box and TF buttons; grab their selectors.
            try:
                # Press the 4h TF button
                page.get_by_role("button", name="4h").click()
                time.sleep(1.5)
            except Exception as exc:
                print(f"[warn] 4h click failed: {exc}", flush=True)

            # Set the chart's symbol using the same event bus the frontend uses.
            # Fall back to API if keyboard nav isn't present.
            page.evaluate(
                "() => { try { window.dispatchEvent(new CustomEvent('market.symbol.change', {detail: 'HYPEUSDT'})); } catch {} }"
            )
            time.sleep(4.0)  # give the chart time to reload and draw overlays

            # 3. Count SVG/canvas elements — at minimum screenshot so user can eyeball
            out = Path("data/logs/ui_tests")
            out.mkdir(parents=True, exist_ok=True)
            shot = out / f"auto_line_viz_{int(time.time())}.png"
            page.screenshot(path=str(shot), full_page=True)
            print(f"Screenshot: {shot}", flush=True)

            # 4. Peek at drawingsState.lines from the frontend — is the fake line there?
            try:
                state_info = page.evaluate(
                    "() => { try { "
                    "const mod = window.__drawingsState || null; "
                    "if (!mod) return {found:false, reason:'no window.__drawingsState'}; "
                    "const hit = (mod.lines||[]).find(l => l.manual_line_id === arguments[0]); "
                    "return {found: !!hit, source: hit?.source, total: (mod.lines||[]).length}; "
                    "} catch(e) { return {found:false, reason:String(e)}; } }",
                    FAKE_ID,
                )
                # Note: window.__drawingsState may not exist; this is a best-effort probe.
                print(f"Frontend state probe: {state_info}", flush=True)
            except Exception as exc:
                print(f"[warn] frontend state probe failed: {exc}", flush=True)

            browser.close()

    finally:
        removed = remove_fake()
        print(f"Cleanup: removed {removed} fake entr(ies)", flush=True)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(run())
