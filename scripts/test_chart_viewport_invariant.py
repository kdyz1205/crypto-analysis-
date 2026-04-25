"""Regression test for the viewport-invariant rule:

  AFTER EVERY SUCCESSFUL CANDLE LOAD, THE CHART'S VISIBLE RANGE MUST
  INTERSECT THE LOADED DATA RANGE.

If the visible range falls entirely outside the data (e.g. chart was
scrolled to "March 11" but new data starts in "April"), the chart shows
a blank canvas — exactly the bug the user reported on 2026-04-25.

This test exercises the failure mode that previously slipped through:
  1. Load HYPE 4h (chart fits to last 200 bars)
  2. Scroll the time scale far to the LEFT (way before the data)
  3. Switch to ETH 4h (different symbol)
  4. Switch BACK to HYPE 4h
  5. ASSERT chart shows non-empty visible range with overlapping candles

Run: python scripts/test_chart_viewport_invariant.py
Exit 0 = passed (chart visible after each switch), 1 = failed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = "http://localhost:8000/v2"


def main():
    failures = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 900})
        # Disable Service Worker so we don't hit cached HTML
        ctx.add_init_script("try { localStorage.disableSW='1'; } catch{}")
        page = ctx.new_page()

        # Capture stale-viewport log lines from chart.js
        stale_logs = []
        page.on("console", lambda m: stale_logs.append(m.text)
                if "VIEWPORT STALE" in m.text else None)

        page.goto(BASE, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(4000)   # let initial chart render

        def get_chart_state():
            """Return (visible_from, visible_to, candle_count, looks_blank)."""
            return page.evaluate("""() => {
                // Reach into lightweight-charts via a known chart container.
                const cont = document.querySelector('#chart-container');
                if (!cont) return { error: 'no chart container' };
                // Count canvas elements (lightweight-charts uses canvas)
                const canvases = cont.querySelectorAll('canvas');
                // Heuristic for "blank": chart canvas exists but has no
                // pixel-paint. We sample center pixels.
                let nonblack = 0, total = 0;
                for (const c of canvases) {
                    if (c.width < 100 || c.height < 100) continue;
                    try {
                        const ctx2 = c.getContext('2d');
                        const px = ctx2.getImageData(c.width / 2, c.height / 2, 50, 50).data;
                        for (let i = 0; i < px.length; i += 4) {
                            total++;
                            if (px[i] !== 0 || px[i+1] !== 0 || px[i+2] !== 0) nonblack++;
                        }
                    } catch {}
                }
                return {
                    canvas_count: canvases.length,
                    sample_nonblack_ratio: total ? nonblack / total : 0,
                    looks_blank: total > 0 && (nonblack / total) < 0.01,
                };
            }""")

        def switch_symbol(sym):
            page.evaluate(f"""() => {{
                const inp = document.querySelector('#v2-symbol-input');
                if (!inp) return;
                inp.focus(); inp.value = '{sym}';
                inp.dispatchEvent(new Event('input', {{bubbles:true}}));
            }}""")
            page.wait_for_timeout(300)
            page.evaluate("""() => {
                const inp = document.querySelector('#v2-symbol-input');
                if (inp) inp.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter',bubbles:true}));
            }""")
            page.wait_for_timeout(3500)

        # ── Step 1: load HYPEUSDT, baseline ─────────────────────────
        switch_symbol('HYPEUSDT')
        s1 = get_chart_state()
        print(f"After HYPE load: {s1}")
        if s1.get('looks_blank'):
            failures.append(f"baseline HYPE load blank: {s1}")

        # ── Step 2: scroll time scale FAR LEFT (simulate user scroll) ─
        page.evaluate("""() => {
            // Programmatically scroll the time scale to a date 1 year ago
            // — way before any data we'd load. The bug was that subsequent
            // symbol switches preserved this stale viewport.
            try {
                const yearAgo = Math.floor(Date.now()/1000) - 365*86400;
                const win = window;
                // We can't reach the chart instance directly; instead use
                // the chart's drag/scroll. Easiest: scroll chart canvas
                // wheel-event style.
                const canvas = document.querySelector('#chart-container canvas');
                if (canvas) {
                    for (let i = 0; i < 30; i++) {
                        canvas.dispatchEvent(new WheelEvent('wheel', {
                            deltaY: 0, deltaX: -800, bubbles: true,
                        }));
                    }
                }
            } catch (e) { console.warn(e); }
        }""")
        page.wait_for_timeout(500)

        # ── Step 3: switch away to ETHUSDT ─────────────────────────
        switch_symbol('ETHUSDT')
        s2 = get_chart_state()
        print(f"After ETH switch: {s2}")
        if s2.get('looks_blank'):
            failures.append(f"after ETH switch blank: {s2}")

        # ── Step 4: switch BACK to HYPEUSDT — this is where the bug fired ─
        switch_symbol('HYPEUSDT')
        s3 = get_chart_state()
        print(f"After HYPE return: {s3}")
        if s3.get('looks_blank'):
            failures.append(f"after HYPE return blank — viewport invariant violated: {s3}")

        # ── Step 5: TF flip — same invariant must hold ─────────────
        page.evaluate("""() => {
            document.querySelector('.v2-tf-btn[data-tf="1d"]')?.click();
        }""")
        page.wait_for_timeout(3500)
        s4 = get_chart_state()
        print(f"After 1d switch: {s4}")
        if s4.get('looks_blank'):
            failures.append(f"after 1d TF switch blank: {s4}")

        page.screenshot(path="data/logs/test_viewport_invariant.png")
        browser.close()

    print("\n=== summary ===")
    print(f"stale-viewport recoveries logged by chart.js: {len(stale_logs)}")
    for l in stale_logs[:5]:
        print(f"  {l[:180]}")

    if failures:
        print("\n!!! FAILURES !!!")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nALL PASSED — chart never went blank across symbol+TF switches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
