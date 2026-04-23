"""Browser proof: triggered plan orders draw entry/SL/TP price-line
markers on the chart, and filled positions draw stronger-color ones.

Mocks /api/conditionals to return one triggered + one filled HYPE cond,
then asserts that the chart series has 6 price lines (3 each) after
refresh.
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
LOG_PATH = ROOT / "data" / "logs" / "ui_tests" / "plan_overlay_markers.log"


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
    now = int(time.time())
    step = 4 * 3600
    candles = []
    price = 41.0
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
    log.log(f"=== plan_overlay_markers {started} HEAD={_git_head()} ===")

    now = int(time.time())

    triggered_cond = {
        "conditional_id": "cond_triggered_1",
        "manual_line_id": "manual-HYPEUSDT-4h-resistance-ui-markers",
        "symbol": "HYPEUSDT",
        "timeframe": "4h",
        "side": "resistance",
        "status": "triggered",
        "fill_price": 41.446,
        "fill_qty": 275.22,
        "exchange_order_id": "fake_oid_1",
        "created_at": now, "updated_at": now, "triggered_at": now,
        "order": {
            "direction": "short",
            "order_kind": "bounce",
            "rr_target": 10.0,
            "stop_points": 0.063,   # SL = fill + 0.063 = 41.509 for short
            "tp_price": 40.825,
            "notional_usd": 11400, "equity_pct": 100, "leverage": 20,
            "submit_to_exchange": True, "exchange_mode": "live",
        },
        "events": [],
    }
    filled_cond = {
        "conditional_id": "cond_filled_1",
        "manual_line_id": "manual-HYPEUSDT-4h-support-pos",
        "symbol": "HYPEUSDT",
        "timeframe": "4h",
        "side": "support",
        "status": "filled",
        "fill_price": 40.0,
        "fill_qty": 50.0,
        "exchange_order_id": "fake_oid_2",
        "created_at": now, "updated_at": now, "triggered_at": now,
        "order": {
            "direction": "long",
            "order_kind": "bounce",
            "rr_target": 3.0,
            "stop_points": 0.2,     # SL = fill - 0.2 = 39.8 for long
            "tp_price": 40.6,
            "notional_usd": 2000, "equity_pct": 20, "leverage": 10,
            "submit_to_exchange": True, "exchange_mode": "live",
        },
        "events": [],
    }

    def fulfill_json(route, body: dict, status: int = 200) -> None:
        route.fulfill(
            status=status, content_type="application/json",
            body=json.dumps(body),
        )

    def handle_api(route, request) -> None:
        path = urlparse(request.url).path
        if path == "/api/ohlcv":
            fulfill_json(route, _mock_ohlcv())
            return
        if path == "/api/drawings/all" or path == "/api/drawings":
            fulfill_json(route, {"ok": True, "drawings": []})
            return
        if path == "/api/conditionals":
            fulfill_json(route, {
                "ok": True, "count": 2,
                "conditionals": [triggered_cond, filled_cond],
            })
            return
        if path == "/api/live-execution/account":
            fulfill_json(route, {
                "ok": True, "total_equity": 1000.0, "available": 1000.0,
                "positions": [], "pending_orders": [],
            })
            return
        if path.startswith("/api/symbols"):
            fulfill_json(route, {"ok": True, "symbols": [
                {"symbol": "HYPEUSDT", "name": "HYPE/USDT"},
            ]})
            return
        if path.startswith("/api/strategy/snapshot") or path.startswith("/api/market"):
            fulfill_json(route, {"ok": True})
            return
        route.continue_()

    console_errors: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.route("**/api/**", handle_api)
            page.add_init_script("window.localStorage.clear()")

            page.goto(BASE + "?symbol=HYPEUSDT&interval=4h", wait_until="commit", timeout=20000)
            page.wait_for_selector("#chart-container", timeout=15000)
            # Allow chart + first plan-overlay refresh to complete
            page.wait_for_timeout(1500)

            # Switch to HYPE explicitly in case the default symbol loaded
            # something else. The symbol picker input lives in #v2-symbol-input.
            page.evaluate("""() => {
              const pickerInput = document.querySelector('#v2-symbol-input');
              if (pickerInput) {
                pickerInput.value = 'HYPEUSDT';
                pickerInput.dispatchEvent(new Event('input', {bubbles: true}));
              }
            }""")
            page.wait_for_timeout(500)

            # Wait for the overlay's test-hook probe to populate. It's
            # set inside refreshPlanOverlay after every draw pass.
            probe = None
            for i in range(40):  # up to 8s
                probe = page.evaluate("() => window.__planOverlayProbe || null")
                if probe and probe.get("rows"):
                    break
                page.wait_for_timeout(200)

            log.log(f"probe = {json.dumps(probe, ensure_ascii=False)[:400]}")
            if not probe or not probe.get("rows"):
                raise AssertionError("plan overlay probe never populated — overlay did not draw")

            rows = probe["rows"]
            cond_ids = {r["cond_id"] for r in rows}
            if "cond_triggered_1" not in cond_ids:
                raise AssertionError(f"triggered cond not rendered: rows={rows}")
            if "cond_filled_1" not in cond_ids:
                raise AssertionError(f"filled cond not rendered: rows={rows}")

            # Each cond should produce 3 priceLines (entry + stop + tp).
            for r in rows:
                if r["line_count"] != 3:
                    raise AssertionError(
                        f"cond {r['cond_id']} produced {r['line_count']} lines, expected 3: {r}"
                    )

            if console_errors:
                log.log("console_errors=" + json.dumps(console_errors[-5:], ensure_ascii=False))

            log.log("PASS triggered plan AND filled position draw entry/SL/TP labels on chart")
            browser.close()
            return 0
    except Exception as exc:
        log.log(f"FAIL {type(exc).__name__}: {exc}")
        return 1
    finally:
        log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
