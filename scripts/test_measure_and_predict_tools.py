"""Browser proof: measure tool + long/short prediction tool work on /v2.

Opens the page, mocks OHLCV, presses the 测量 button + drags a measure
box, then presses the 多头 button + clicks twice (entry + tp), asserts
DOM elements appear for each.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000/v2"
LOG_PATH = ROOT / "data" / "logs" / "ui_tests" / "measure_and_predict_tools.log"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8",
        ).strip()
    except Exception:
        return "unknown"


class Logger:
    def __init__(self): self.lines = []
    def log(self, m): print(m, flush=True); self.lines.append(m)
    def flush(self):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            for line in self.lines: fh.write(line + "\n")
            fh.write("\n")


def _mock_ohlcv(count=120):
    now = int(time.time()); step = 4 * 3600
    candles = []; price = 41.0
    for i in range(count):
        t = now - (count - i) * step
        o = price
        c = price * (1 + ((i % 7) - 3) * 0.002)
        h = max(o, c) * 1.003; l = min(o, c) * 0.997
        candles.append({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": 1000 + i*5})
        price = c
    return {"ok": True, "candles": candles, "overlays": {}}


def handle_api(route, request):
    path = urlparse(request.url).path
    def ok(body, status=200):
        route.fulfill(status=status, content_type="application/json", body=json.dumps(body))
    if path == "/api/ohlcv": ok(_mock_ohlcv()); return
    if path in ("/api/drawings/all", "/api/drawings"): ok({"ok": True, "drawings": []}); return
    if path == "/api/conditionals": ok({"ok": True, "count": 0, "conditionals": []}); return
    if path == "/api/live-execution/account":
        ok({"ok": True, "total_equity": 1000.0, "available": 1000.0, "positions": [], "pending_orders": []}); return
    if path.startswith("/api/symbols"):
        ok({"ok": True, "symbols": [{"symbol": "HYPEUSDT", "name": "HYPE/USDT"}]}); return
    if path.startswith("/api/strategy/snapshot") or path.startswith("/api/market"):
        ok({"ok": True}); return
    route.continue_()


def main() -> int:
    log = Logger()
    log.log(f"=== measure_and_predict_tools {datetime.now().astimezone().isoformat(timespec='seconds')} HEAD={_git_head()} ===")
    console_errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.route("**/api/**", handle_api)
            page.add_init_script("window.localStorage.clear()")
            page.goto(BASE + "?symbol=HYPEUSDT&interval=4h", wait_until="commit", timeout=20000)
            page.wait_for_selector("#chart-container", timeout=15000)
            page.wait_for_timeout(1200)

            # 1) Measure tool — click 测量 button to arm mode
            page.click('button[data-tool="measure"]')
            page.wait_for_timeout(300)
            is_active = page.evaluate("""() => {
              const btn = document.querySelector('button[data-tool="measure"]');
              return btn && btn.classList.contains('is-active');
            }""")
            log.log(f"measure button armed = {is_active}")
            if not is_active:
                raise AssertionError("measure button did not become active")

            # Simulate drag on chart container (middle-ish area)
            chart_box = page.locator("#chart-container").bounding_box()
            cx1 = chart_box["x"] + 300
            cy1 = chart_box["y"] + 300
            cx2 = chart_box["x"] + 600
            cy2 = chart_box["y"] + 450
            page.mouse.move(cx1, cy1)
            page.mouse.down()
            page.mouse.move(cx2, cy2, steps=10)
            page.mouse.up()
            page.wait_for_timeout(300)

            measure_box_count = page.locator(".measure-box").count()
            log.log(f"measure boxes rendered = {measure_box_count}")
            if measure_box_count < 1:
                raise AssertionError("drag did not create a measure box")

            has_label = page.evaluate("""() => {
              const box = document.querySelector('.measure-box .measure-label');
              if (!box) return null;
              return box.textContent.slice(0, 200);
            }""")
            log.log(f"measure label text = {has_label!r}")
            if not has_label or "bars" not in has_label:
                raise AssertionError(f"measure label missing expected content: {has_label!r}")

            # Exit measure
            page.keyboard.press("Escape")
            page.wait_for_timeout(200)

            # 2) Long prediction — click 多头, then click entry, click TP above it
            page.click('button[data-tool="predict-long"]')
            page.wait_for_timeout(200)
            is_pred_active = page.evaluate("""() => {
              const btn = document.querySelector('button[data-tool="predict-long"]');
              return btn && btn.classList.contains('is-active');
            }""")
            log.log(f"predict-long armed = {is_pred_active}")
            if not is_pred_active:
                raise AssertionError("predict-long button did not become active")

            # Click 1: entry (low price area)
            page.mouse.click(chart_box["x"] + 400, chart_box["y"] + 500)
            page.wait_for_timeout(200)
            # Click 2: TP (higher up in chart = lower Y coordinate = higher price)
            page.mouse.click(chart_box["x"] + 600, chart_box["y"] + 250)
            page.wait_for_timeout(400)

            predict_box_count = page.locator(".predict-box").count()
            log.log(f"prediction boxes = {predict_box_count}")
            if predict_box_count < 1:
                raise AssertionError("prediction not created after 2 clicks")

            rr_text = page.evaluate("""() => {
              const el = document.querySelector('.predict-box .predict-rr');
              return el ? el.textContent : null;
            }""")
            log.log(f"R:R label = {rr_text!r}")
            if not rr_text or "R:R" not in rr_text:
                raise AssertionError(f"R:R label missing: {rr_text!r}")

            # Check localStorage persisted it
            stored = page.evaluate("() => window.localStorage.getItem('v2.predictions.v1')")
            if not stored or "predict-" not in (stored or "") and "predicted" not in (stored or ""):
                # Fallback: just check we have at least one entry in JSON
                try:
                    parsed = json.loads(stored or "[]")
                    if not isinstance(parsed, list) or len(parsed) == 0:
                        raise AssertionError(f"prediction not persisted to localStorage: {stored!r}")
                except json.JSONDecodeError:
                    raise AssertionError(f"localStorage value not JSON: {stored!r}")

            if console_errors:
                log.log("console_errors=" + json.dumps(console_errors[-5:], ensure_ascii=False))

            log.log("PASS measure tool + long prediction tool both work")
            browser.close()
            return 0
    except Exception as e:
        log.log(f"FAIL {type(e).__name__}: {e}")
        return 1
    finally:
        log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
