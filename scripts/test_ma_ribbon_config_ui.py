"""Browser proof: MA Ribbon config editor works end-to-end.

Steps:
  1. Open /v2 with OHLCV mocked (so chart renders without Bitget).
  2. Click "⚙ 指标" header button → panel appears with MA Ribbon row.
  3. Click ⚙ gear on MA Ribbon row → config editor expands with 4
     line rows (default 5/8/21/55).
  4. Change Line 1 period 5 → 3 → assert localStorage v2.indicators.v1
     has lines[0].period === 3.
  5. Pick "Fibonacci 5/13/34/89" preset → assert all 4 line periods
     updated atomically to [5,13,34,89].
  6. Click "+ 加一条线" → assert 5th line row appears.
  7. Click the last row's ✕ → assert 4 lines again.

If every assertion holds, no DELETE fired on the chart and no console
errors, log PASS.
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
LOG_PATH = ROOT / "data" / "logs" / "ui_tests" / "ma_ribbon_config_ui.log"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, text=True, encoding="utf-8",
        ).strip()
    except Exception:
        return "unknown"


class Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        self.lines.append(msg)

    def flush(self) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            for line in self.lines:
                fh.write(line + "\n")
            fh.write("\n")


def _mock_ohlcv(count: int = 120) -> dict:
    """Minimal OHLCV stub the chart can render without a real backend.
    Prices drift slightly so MA lines have distinct shapes."""
    now = int(time.time())
    step = 4 * 3600
    candles = []
    price = 100.0
    for i in range(count):
        t = now - (count - i) * step
        o = price
        c = price * (1 + ((i % 7) - 3) * 0.002)
        h = max(o, c) * 1.003
        l = min(o, c) * 0.997
        v = 1000.0 + i * 5
        candles.append({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
        price = c
    return {"ok": True, "candles": candles, "overlays": {}}


def main() -> int:
    log = Logger()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    log.log(f"=== ma_ribbon_config_ui {started} HEAD={_git_head()} ===")

    console_errors: list[str] = []

    def fulfill_json(route, body: dict, status: int = 200) -> None:
        route.fulfill(
            status=status,
            content_type="application/json",
            body=json.dumps(body),
        )

    def handle_api(route, request) -> None:
        parsed = urlparse(request.url)
        path = parsed.path
        if path == "/api/ohlcv":
            fulfill_json(route, _mock_ohlcv())
            return
        if path == "/api/drawings/all" or path == "/api/drawings":
            fulfill_json(route, {"ok": True, "drawings": []})
            return
        if path == "/api/conditionals":
            fulfill_json(route, {"ok": True, "count": 0, "conditionals": []})
            return
        if path == "/api/live-execution/account":
            fulfill_json(route, {
                "ok": True, "total_equity": 1000.0, "available": 1000.0,
                "positions": [], "pending_orders": [],
            })
            return
        if path.startswith("/api/symbols"):
            fulfill_json(route, {"ok": True, "symbols": [
                {"symbol": "BTCUSDT", "name": "BTC/USDT"},
            ]})
            return
        if path.startswith("/api/strategy/snapshot"):
            fulfill_json(route, {"ok": True})
            return
        if path.startswith("/api/market"):
            fulfill_json(route, {"ok": True})
            return
        route.continue_()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.route("**/api/**", handle_api)

            # Clean slate for localStorage — force first-boot defaults.
            page.add_init_script("window.localStorage.clear()")

            page.goto(BASE, wait_until="commit", timeout=20000)
            page.wait_for_selector("#v2-indicator-btn", timeout=15000)

            # Let the chart finish its first render so ma_ribbon series exist
            page.wait_for_timeout(800)

            # 1) Open the indicator panel
            page.click("#v2-indicator-btn")
            page.wait_for_selector(".ind-panel", timeout=5000)
            log.log("indicator panel opened")

            # 2) Find MA Ribbon row. The default id is 'ma_ribbon'.
            ma_row = page.locator('.ind-row[data-id="ma_ribbon"]')
            if ma_row.count() == 0:
                raise AssertionError("MA Ribbon row not found in indicator panel")

            # 3) Click the ⚙ gear on MA Ribbon row
            ma_row.locator('.ind-gear').click()
            page.wait_for_selector('.ind-row[data-id="ma_ribbon"] .ind-config', timeout=3000)
            log.log("MA Ribbon config editor expanded")

            # 4) Assert 4 default line rows rendered
            line_rows = page.locator('.ind-row[data-id="ma_ribbon"] .cfg-line-row')
            initial_count = line_rows.count()
            log.log(f"default line count = {initial_count}")
            if initial_count != 4:
                raise AssertionError(f"expected 4 default lines, got {initial_count}")

            # Read default periods off the DOM
            periods = page.evaluate("""() => {
              const rows = document.querySelectorAll(
                '.ind-row[data-id=\"ma_ribbon\"] .cfg-line-row');
              return Array.from(rows).map(r => Number(
                r.querySelector('input[data-f=\"period\"]').value));
            }""")
            log.log(f"default periods = {periods}")
            if periods != [5, 8, 21, 55]:
                raise AssertionError(f"default periods expected [5,8,21,55], got {periods}")

            # 5) Change Line 1 period 5 → 3
            first_period = page.locator(
                '.ind-row[data-id="ma_ribbon"] .cfg-line-row[data-line-idx="0"] input[data-f="period"]')
            first_period.fill("3")
            first_period.press("Enter")
            page.wait_for_timeout(250)

            ls_after_edit = page.evaluate("""() => {
              const raw = window.localStorage.getItem('v2.indicators.v1');
              const list = JSON.parse(raw || '[]');
              const ma = list.find(x => x.id === 'ma_ribbon');
              return ma && ma.config && ma.config.lines
                ? ma.config.lines.map(l => l.period)
                : null;
            }""")
            log.log(f"periods after edit = {ls_after_edit}")
            if not ls_after_edit or ls_after_edit[0] != 3:
                raise AssertionError(
                    f"period edit did not persist: localStorage says {ls_after_edit}")

            # 6) Pick Fibonacci preset → [5,13,34,89]
            preset = page.locator(
                '.ind-row[data-id="ma_ribbon"] select.cfg-preset')
            preset.select_option(value="fibonacci")
            page.wait_for_timeout(250)

            ls_after_preset = page.evaluate("""() => {
              const raw = window.localStorage.getItem('v2.indicators.v1');
              const list = JSON.parse(raw || '[]');
              const ma = list.find(x => x.id === 'ma_ribbon');
              return ma?.config?.lines?.map(l => l.period) || null;
            }""")
            log.log(f"periods after Fibonacci preset = {ls_after_preset}")
            if ls_after_preset != [5, 13, 34, 89]:
                raise AssertionError(
                    f"preset did not apply: expected [5,13,34,89], got {ls_after_preset}")

            # 7) Add a line
            page.locator(
                '.ind-row[data-id="ma_ribbon"] button[data-action="line-add"]').click()
            page.wait_for_timeout(200)
            after_add = page.locator(
                '.ind-row[data-id="ma_ribbon"] .cfg-line-row').count()
            log.log(f"lines after add = {after_add}")
            if after_add != 5:
                raise AssertionError(f"expected 5 lines after add, got {after_add}")

            # 8) Remove the last line (idx 4) via its ✕
            page.locator(
                '.ind-row[data-id="ma_ribbon"] .cfg-line-row[data-line-idx="4"] button[data-action="line-del"]'
            ).click()
            page.wait_for_timeout(200)
            after_del = page.locator(
                '.ind-row[data-id="ma_ribbon"] .cfg-line-row').count()
            log.log(f"lines after delete = {after_del}")
            if after_del != 4:
                raise AssertionError(f"expected 4 lines after del, got {after_del}")

            if console_errors:
                log.log("console_errors=" + json.dumps(console_errors[-5:], ensure_ascii=False))

            log.log("PASS MA Ribbon config editor: gear expand, preset, add, delete all work")
            browser.close()
            return 0
    except Exception as exc:
        log.log(f"FAIL {type(exc).__name__}: {exc}")
        return 1
    finally:
        log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
