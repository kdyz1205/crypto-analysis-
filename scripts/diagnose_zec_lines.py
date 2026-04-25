"""Open ZEC 1d in browser, dump drawingsState.lines + check SVG content.

User reports drawings panel shows 3 ZEC 1d lines but they don't render
on the chart. This script captures the in-memory state + DOM content
so we can see what's actually happening.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = "http://localhost:8000/v2"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 900})
        page = ctx.new_page()

        page.on("console", lambda m: print(f"[console.{m.type}] {m.text[:300]}")
                if m.type in ("error", "warning") else None)

        # Bypass Service Worker for clean test
        ctx.add_init_script("try { localStorage.disableSW='1'; } catch{}")

        page.goto(BASE, wait_until="domcontentloaded", timeout=30000)
        # Wait for initial chart paint
        page.wait_for_timeout(4000)

        # Switch to ZEC 1d via search box
        page.evaluate("""() => {
            const inp = document.querySelector('#v2-symbol-input');
            if (inp) { inp.focus(); inp.value = 'ZECUSDT'; inp.dispatchEvent(new Event('input', {bubbles:true})); }
        }""")
        page.wait_for_timeout(500)
        # Click 1d TF button
        page.evaluate("""() => {
            const btn = document.querySelector('.v2-tf-btn[data-tf="1d"]');
            if (btn) btn.click();
        }""")
        page.wait_for_timeout(500)
        # Trigger a setSymbol via the search box's pick by Enter
        page.evaluate("""() => {
            // Use the public bus to switch
            const ev = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true });
            const inp = document.querySelector('#v2-symbol-input');
            if (inp) inp.dispatchEvent(ev);
        }""")
        page.wait_for_timeout(5000)

        # Try to read drawingsState — but it's a module-private variable.
        # Use the global event bus instead via window if exposed.
        # Wait extra for markers to load
        page.wait_for_timeout(3000)
        info = page.evaluate("""() => {
            const sym = document.querySelector('#v2-symbol-input')?.value;
            const overlay = document.querySelector('.chart-drawing-overlay');
            const overlayRect = overlay?.getBoundingClientRect();
            // SVG line stroke widths after our fix
            const lineStrokes = overlay ? Array.from(overlay.querySelectorAll('line')).map((e) => ({
                stroke: e.getAttribute('stroke'),
                stroke_width: e.getAttribute('stroke-width'),
                x1: parseFloat(e.getAttribute('x1') || '0').toFixed(0),
                y1: parseFloat(e.getAttribute('y1') || '0').toFixed(0),
                x2: parseFloat(e.getAttribute('x2') || '0').toFixed(0),
                y2: parseFloat(e.getAttribute('y2') || '0').toFixed(0),
            })) : [];
            // Trade marker count: lightweight-charts paints markers on the
            // candle canvas; can't easily inspect. Instead check that our
            // service was called by looking at console.
            return {
                input_value: sym,
                overlay_exists: !!overlay,
                overlay_lines_count: lineStrokes.length,
                line_stroke_samples: lineStrokes.slice(0, 4),
            };
        }""")
        print(json.dumps(info, indent=2, ensure_ascii=False))

        # Network: did /api/drawings get called for ZEC 1d?
        network = []
        def on_response(resp):
            if "/api/drawings" in resp.url:
                network.append({"url": resp.url[:200], "status": resp.status})
        page.on("response", on_response)

        # Force another reload to capture network
        page.evaluate("""() => {
            try {
                // Use the chart module's loadCurrent
                document.querySelector('.v2-tf-btn[data-tf="1d"]')?.click();
            } catch {}
        }""")
        page.wait_for_timeout(2000)
        print("\n=== /api/drawings calls observed ===")
        for n in network[-10:]:
            print(f"  {n['status']} {n['url']}")

        page.screenshot(path="data/logs/diagnose_zec.png", full_page=False)
        print("\nscreenshot saved to data/logs/diagnose_zec.png")

        browser.close()


if __name__ == "__main__":
    main()
