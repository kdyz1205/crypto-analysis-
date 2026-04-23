"""Browser proof: trade history row click jumps back to market symbol/TF.

Uses the real /v2 page, mocks only /api/trades/manual-history, and asserts:
  1. trade_history row detail button still opens the JSON detail pane
  2. clicking the row switches to market view
  3. market symbol + timeframe update to the row's symbol/TF
"""
from __future__ import annotations

import json
import subprocess
import sys
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
LOG_PATH = ROOT / "data" / "logs" / "ui_tests" / "trade_history_row_jump.log"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


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


def main() -> int:
    log = Logger()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    log.log(f"=== trade_history_row_jump {started} HEAD={_git_head()} ===")

    payload = {
        "rows": [
            {
                "dt": "2026-04-23T10:00:00Z",
                "symbol": "ETHUSDT",
                "timeframe": "1h",
                "side": "short",
                "entry_price": 2000.0,
                "exit_price": 1900.0,
                "pnl_usd": 50.0,
                "pnl_pct": 0.025,
                "close_reason": "tp",
                "user_label": "good",
                "manual_line_id": "manual-eth-row-jump",
            }
        ],
        "columns": [
            "dt", "symbol", "timeframe", "side", "entry_price", "exit_price",
            "pnl_usd", "pnl_pct", "close_reason", "user_label",
        ],
    }

    def handle_api(route, request) -> None:
        parsed = urlparse(request.url)
        if parsed.path == "/api/trades/manual-history":
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return
        route.continue_()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 960})
            page.route("**/api/trades/manual-history**", handle_api)

            page.goto(BASE, wait_until="commit", timeout=20000)
            page.wait_for_selector('#v2-nav .v2-nav-btn[data-view="trade_history"]', timeout=15000)
            with page.expect_response(lambda resp: "/api/trades/manual-history" in resp.url and resp.status == 200, timeout=20000):
                page.click('#v2-nav .v2-nav-btn[data-view="trade_history"]')
            page.wait_for_selector('#view-trade_history', timeout=15000)
            page.wait_for_selector('#view-trade_history tr[data-row-idx="0"]', timeout=20000)

            page.click('#view-trade_history [data-action="detail-row"][data-row-idx="0"]')
            page.wait_for_selector('#view-trade_history #th-detail:not([hidden])', timeout=5000)
            detail_text = page.text_content('#view-trade_history #th-detail-body') or ''
            log.log(f"detail_contains_symbol={'ETHUSDT' in detail_text}")
            if 'ETHUSDT' not in detail_text:
                raise AssertionError("detail button did not open row JSON")

            page.click('#view-trade_history tr[data-row-idx="0"]')
            page.wait_for_timeout(1200)

            market_active = page.eval_on_selector(
                '#v2-nav .v2-nav-btn[data-view="market"]',
                'el => el.classList.contains("active")',
            )
            symbol_value = page.input_value('#v2-symbol-input')
            tf_active = page.eval_on_selector(
                '#v2-tf-group .v2-tf-btn[data-tf="1h"]',
                'el => el.classList.contains("active")',
            )
            log.log(f"market_active={market_active}")
            log.log(f"symbol_value={symbol_value}")
            log.log(f"timeframe_1h_active={tf_active}")

            if not market_active:
                raise AssertionError("row click did not switch back to market view")
            if symbol_value != "ETHUSDT":
                raise AssertionError(f"expected symbol ETHUSDT, got {symbol_value!r}")
            if not tf_active:
                raise AssertionError("expected 1h timeframe button active after row click")

            log.log("PASS trade history detail button works and row click jumps to market symbol/TF")
            browser.close()
            return 0
    except Exception as exc:
        log.log(f"FAIL {type(exc).__name__}: {exc}")
        return 1
    finally:
        log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
