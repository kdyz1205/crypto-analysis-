"""Browser proof for the quick-trade popup and conditional-panel refresh.

This test opens the real /v2 page, but mocks only the order/account API
calls. It proves the UI chain that the user reported broken:

1. quick setup rows are not visually dimmed before choosing a direction
2. clicking a setup without direction does not submit
3. choosing direction + setup posts place-line-order
4. success text includes the Bitget acknowledgement identifiers
5. conditionals.changed and cond-placed fire
6. the panel reacts by refetching /api/conditionals

It intentionally does NOT prove that a real Bitget order exists. That
requires scripts/test_draw_line_real.py and real exchange credentials.
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
LOG_PATH = ROOT / "data" / "logs" / "ui_tests" / "quick_trade_panel_refresh.log"


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
    head = _git_head()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    log.log(f"=== quick_trade_panel_refresh {started} HEAD={head} ===")

    post_payloads: list[dict] = []
    cond_get_urls: list[str] = []
    account_hits: list[str] = []

    conditional_payload = {
        "conditional_id": "cond_mock_ui_refresh",
        "manual_line_id": "manual-ui-refresh",
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "side": "support",
        "t_start": 1712592000,
        "t_end": 1712678400,
        "price_start": 100.0,
        "price_end": 105.0,
        "status": "triggered",
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "exchange_order_id": "mock-oid-ui-refresh",
        "trigger": {"poll_seconds": 60},
        "order": {
            "direction": "long",
            "order_kind": "bounce",
            "tolerance_pct_of_line": 0.05,
            "stop_offset_pct_of_line": 0.01,
            "rr_target": 2.0,
            "notional_usd": 25.0,
            "submit_to_exchange": True,
            "exchange_mode": "live",
        },
        "events": [
            {
                "ts": int(time.time()),
                "kind": "created",
                "message": "mock order for browser UI refresh test",
                "extra": {},
            }
        ],
    }

    setup_state = {
        "active_id": "ui-test",
        "setups": [
            {
                "id": "ui-test",
                "name": "UI test setup",
                "config": {
                    "direction": "long",
                    "order_kind": "bounce",
                    "buffer_pct": 0.1,
                    "stop_pct": 0.3,
                    "rr_target": 2.0,
                    "leverage": 1,
                    "size_mode": "notional_usd",
                    "notional_usd": 25,
                    "equity_pct": 0,
                    "risk_pct": 0,
                    "exchange_mode": "live",
                    "submit_to_exchange": True,
                    "reverse_enabled": False,
                },
            }
        ],
    }

    def fulfill_json(route, body: dict, status: int = 200) -> None:
        route.fulfill(
            status=status,
            content_type="application/json",
            body=json.dumps(body),
        )

    def handle_api(route, request) -> None:
        url = request.url
        parsed = urlparse(url)
        if parsed.path == "/api/live-execution/account":
            account_hits.append(url)
            fulfill_json(
                route,
                {
                    "ok": True,
                    "mode": "live",
                    "total_equity": 1000.0,
                    "available": 1000.0,
                    "positions": [],
                    "pending_orders": [
                        {
                            "orderId": "mock-oid-ui-refresh",
                            "symbol": "BTCUSDT",
                            "price": "100.05",
                            "status": "live",
                        }
                    ],
                },
            )
            return
        if parsed.path == "/api/drawings/manual/place-line-order":
            raw = request.post_data or "{}"
            try:
                post_payloads.append(json.loads(raw))
            except json.JSONDecodeError:
                post_payloads.append({"_raw": raw})
            fulfill_json(
                route,
                {
                    "ok": True,
                    "message": "mock Bitget acknowledgement",
                    "exchange_order_id": "mock-oid-ui-refresh",
                    "conditional": conditional_payload,
                },
            )
            return
        if parsed.path == "/api/conditionals":
            cond_get_urls.append(url)
            fulfill_json(
                route,
                {
                    "ok": True,
                    "count": 1,
                    "conditionals": [conditional_payload],
                },
            )
            return
        route.continue_()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.route("**/api/**", handle_api)
            setup_json = json.dumps(setup_state)
            page.add_init_script(
                f"""() => {{
                    const state = {setup_json};
                    localStorage.setItem('v2.tradeplan.setups.v1', JSON.stringify(state));
                    localStorage.setItem('v2.tradeplan.defaults.v1', JSON.stringify(state.setups[0].config));
                }}"""
            )

            console_errors: list[str] = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

            page.goto(BASE, wait_until="commit", timeout=20000)
            page.wait_for_selector("body", timeout=10000)
            page.wait_for_function(
                "() => document.body && document.body.innerText.length > 20",
                timeout=10000,
            )
            page.wait_for_selector("#v2-cond-panel", timeout=15000)
            log.log("PASS page loaded and conditional panel mounted")

            page.evaluate(
                """async () => {
                    window.__qtEvents = { changed: 0, placed: 0 };
                    const events = await import('/js/util/events.js');
                    events.subscribe('conditionals.changed', () => { window.__qtEvents.changed += 1; });
                    window.addEventListener('cond-placed', () => { window.__qtEvents.placed += 1; });
                    const modal = await import('/js/workbench/drawings/trade_plan_modal.js');
                    window.__openQuickTradePopup = modal.openQuickTradePopup;
                }"""
            )

            page.evaluate(
                """() => {
                    window.__openQuickTradePopup({
                        manual_line_id: 'manual-ui-refresh',
                        symbol: 'BTCUSDT',
                        timeframe: '4h',
                        side: 'support',
                        t_start: 1712592000,
                        t_end: 1712678400,
                        price_start: 100,
                        price_end: 105,
                        status: 'active'
                    }, 320, 240);
                }"""
            )
            page.wait_for_selector(".tp-quick-popup", timeout=10000)
            page.wait_for_selector(".qt-setup-row", timeout=10000)
            opacity = page.eval_on_selector(
                ".qt-setup-row",
                "el => getComputedStyle(el).opacity",
            )
            if float(opacity) < 0.95:
                raise AssertionError(f"setup row is still visually dimmed: opacity={opacity}")
            log.log(f"PASS setup row opacity={opacity} before direction")

            before_posts = len(post_payloads)
            page.click(".qt-setup-row")
            page.wait_for_timeout(250)
            status_text = page.eval_on_selector("#qt-status", "el => el.textContent")
            if len(post_payloads) != before_posts:
                raise AssertionError("setup click without direction submitted an order")
            if "direction" not in status_text.lower() and "方向" not in status_text:
                raise AssertionError(f"missing direction warning, got status={status_text!r}")
            log.log("PASS setup click without direction warns and does not POST")

            before_cond_gets = len(cond_get_urls)
            page.click(".qt-dir-btn[data-dir='long']")
            page.click(".qt-setup-row")
            page.wait_for_function(
                "() => document.querySelector('#qt-status')?.textContent.includes('Bitget')",
                timeout=10000,
            )
            success_text = page.eval_on_selector("#qt-status", "el => el.textContent")
            if "cond" not in success_text or "oid" not in success_text:
                raise AssertionError(f"success text does not show cond+oid: {success_text!r}")
            log.log(f"PASS success status visible: {success_text}")

            page.wait_for_function(
                "() => window.__qtEvents && window.__qtEvents.changed >= 1 && window.__qtEvents.placed >= 1",
                timeout=10000,
            )
            events = page.evaluate("() => window.__qtEvents")
            log.log(f"PASS UI events fired: {events}")

            deadline = time.time() + 5.0
            while time.time() < deadline and len(cond_get_urls) <= before_cond_gets:
                page.wait_for_timeout(100)
            if len(cond_get_urls) <= before_cond_gets:
                raise AssertionError(
                    f"panel did not refetch /api/conditionals after order; before={before_cond_gets}, after={len(cond_get_urls)}"
                )
            log.log(
                "PASS panel refetched conditionals after order "
                f"before={before_cond_gets} after={len(cond_get_urls)}"
            )

            if len(post_payloads) != 1:
                raise AssertionError(f"expected one place-line-order POST, got {len(post_payloads)}")
            payload = post_payloads[0]
            if payload.get("manual_line_id") != "manual-ui-refresh":
                raise AssertionError(f"wrong manual_line_id in POST: {payload}")
            if payload.get("direction") != "long":
                raise AssertionError(f"wrong direction in POST: {payload}")
            log.log(
                "PASS one place-line-order POST captured "
                f"manual_line_id={payload.get('manual_line_id')} direction={payload.get('direction')}"
            )

            relevant_errors = [
                e for e in console_errors
                if "quick_trade_panel_refresh" in e or "trade_plan_modal" in e
            ]
            if relevant_errors:
                raise AssertionError(f"relevant console errors: {relevant_errors}")
            log.log(f"PASS account API mocked hits={len(account_hits)}")

            browser.close()
    except Exception as exc:
        log.log(f"FAIL {type(exc).__name__}: {exc}")
        log.flush()
        return 1

    log.log("RESULT PASS")
    log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
