"""Real end-to-end test of the draw-line-to-tradeplan-modal flow.

This is the test I should have written BEFORE claiming "画线已经做完了".
It drives a real Chromium window, presses T to enter draw mode, clicks
twice to commit a line, and watches for the modal to appear. Every
intermediate console log is captured.

Exit 0 = flow works end to end.
Exit 1 = flow is broken, with a specific failure point printed.
"""
from __future__ import annotations

import sys
import time
from playwright.sync_api import sync_playwright, ConsoleMessage, Page

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8000/v2"


def main() -> int:
    console_log: list[dict] = []
    network_errors: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1680, "height": 980})
        page = ctx.new_page()

        page.on("console", lambda msg: console_log.append(
            {"type": msg.type, "text": msg.text[:300]}
        ))
        page.on("response", lambda resp: network_errors.append(
            {"url": resp.url[:120], "status": resp.status}
        ) if resp.status >= 400 else None)

        print("[1] loading page...", flush=True)
        t0 = time.time()
        try:
            page.goto(BASE, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"  load failed: {e}")
            return 1
        print(f"  DOM: {time.time()-t0:.1f}s", flush=True)

        # Wait for chart + candle series to actually exist
        try:
            page.wait_for_function(
                "() => window.document.querySelector('#chart-container svg, #chart-container canvas')",
                timeout=15000,
            )
        except Exception:
            print("  chart did not render in 15s", flush=True)

        # Give WS tickers a moment to settle
        page.wait_for_timeout(2500)

        # Probe just for files existing — DO NOT dynamic-import, that
        # creates a second module instance with null state.
        probe = page.evaluate("""async () => {
            const out = {};
            try {
                const r = await fetch('/static/js/workbench/drawings/trade_plan_modal.js');
                out.trade_plan_modal = r.status;
            } catch (e) { out.trade_plan_modal_err = String(e); }
            try {
                const r = await fetch('/static/js/workbench/drawings/chart_drawing.js');
                out.chart_drawing = r.status;
            } catch (e) { out.chart_drawing_err = String(e); }
            return out;
        }""")
        print(f"[2] module probe: {probe}", flush=True)

        if probe.get("trade_plan_modal") != 200:
            print("  ✗ trade_plan_modal.js not served (status=" + str(probe.get("trade_plan_modal")) + ")", flush=True)
            browser.close()
            return 1

        # Enter draw mode via the SAME keyboard handler the real UI uses.
        # chart_drawing.js registered a document-level keydown listener
        # during initChartDrawing (run by main.js at boot). Pressing 'T'
        # drives the already-initialised module, with _container set.
        print("[3] pressing T to enter trendline draw mode...", flush=True)
        # Make sure focus isn't inside an input
        page.evaluate("document.activeElement && document.activeElement.blur && document.activeElement.blur()")
        page.keyboard.press("t")
        page.wait_for_timeout(400)

        # Get chart container bounding box so we can compute click coordinates
        box = page.eval_on_selector("#chart-container", "el => { const r = el.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }")
        print(f"  chart box: {box}", flush=True)

        # Click two points inside the chart (25% / 75% horizontally, 60% / 50% vertically)
        x1 = box["x"] + box["w"] * 0.25
        y1 = box["y"] + box["h"] * 0.60
        x2 = box["x"] + box["w"] * 0.75
        y2 = box["y"] + box["h"] * 0.50
        print(f"[4] clicking first anchor at ({x1:.0f},{y1:.0f})...", flush=True)
        page.mouse.click(x1, y1)
        page.wait_for_timeout(500)
        # Move to second (so draft preview updates)
        page.mouse.move(x2, y2)
        page.wait_for_timeout(200)
        print(f"[5] clicking second anchor at ({x2:.0f},{y2:.0f})...", flush=True)
        page.mouse.click(x2, y2)

        # Wait for POST to finish + modal to open
        print("[6] waiting for modal...", flush=True)
        try:
            page.wait_for_selector(".tp-modal", timeout=8000)
            print("  ✓ modal appeared", flush=True)
            page.screenshot(path="data/logs/browser_review/draw_line_modal.png", full_page=True)
            # Grab modal title for verification
            title = page.eval_on_selector(".tp-title", "el => el.innerText").strip()
            print(f"  modal title: {title}", flush=True)
            success = True
        except Exception as e:
            print(f"  ✗ modal NEVER APPEARED: {e}", flush=True)
            page.screenshot(path="data/logs/browser_review/draw_line_no_modal.png", full_page=True)
            success = False

        # Dump console + network for diagnostics
        print(f"\n[7] console log ({len(console_log)} entries):", flush=True)
        for c in console_log:
            if c["type"] in ("error", "warning"):
                print(f"  [{c['type']}] {c['text']}", flush=True)
        chart_draw_logs = [c for c in console_log if "chart_drawing" in c["text"] or "[chart" in c["text"]]
        print(f"\n  relevant chart_drawing logs: {len(chart_draw_logs)}", flush=True)
        for c in chart_draw_logs[-10:]:
            print(f"  [{c['type']}] {c['text']}", flush=True)

        if network_errors:
            print(f"\n[8] network errors ({len(network_errors)}):", flush=True)
            for n in network_errors[:10]:
                print(f"  {n['status']} {n['url']}", flush=True)

        browser.close()
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
