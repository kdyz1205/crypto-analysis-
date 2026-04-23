"""Browser proof: active orders protect their drawing line from deletion.

Opens the real /v2 page, mocks only the drawing/order list APIs, clicks the
line delete button, and proves the UI does not send DELETE while an active
conditional exists for that line.
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
LOG_PATH = ROOT / "data" / "logs" / "ui_tests" / "active_order_line_delete_guard.log"
LINE_ID = "manual-BTCUSDT-4h-resistance-ui-delete-guard"


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
    log.log(f"=== active_order_line_delete_guard {started} HEAD={_git_head()} ===")

    now = int(time.time())
    drawing = {
        "manual_line_id": LINE_ID,
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "side": "resistance",
        "source": "manual",
        "t_start": 1712592000,
        "t_end": 1712678400,
        "price_start": 100.0,
        "price_end": 105.0,
        "extend_left": False,
        "extend_right": True,
        "locked": False,
        "label": "ui active-order guard",
        "notes": "",
        "comparison_status": "uncompared",
        "override_mode": "display_only",
        "nearest_auto_line_id": None,
        "slope_diff": None,
        "projected_price_diff": None,
        "overlap_ratio": None,
        "created_at": now,
        "updated_at": now,
    }
    conditional = {
        "conditional_id": "cond-ui-delete-guard",
        "manual_line_id": LINE_ID,
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "side": "resistance",
        "t_start": drawing["t_start"],
        "t_end": drawing["t_end"],
        "price_start": drawing["price_start"],
        "price_end": drawing["price_end"],
        "pattern_stats_at_create": {},
        "trigger": {"poll_seconds": 60},
        "order": {
            "direction": "short",
            "order_kind": "bounce",
            "rr_target": 2.0,
            "notional_usd": 25.0,
            "submit_to_exchange": True,
            "exchange_mode": "live",
        },
        "status": "triggered",
        "created_at": now,
        "updated_at": now,
        "triggered_at": now,
        "exchange_order_id": "ui-delete-guard-oid",
        "events": [{"ts": now, "kind": "created", "message": "ui guard"}],
    }

    delete_hits: list[str] = []
    console_errors: list[str] = []

    def fulfill_json(route, body: dict, status: int = 200) -> None:
        route.fulfill(
            status=status,
            content_type="application/json",
            body=json.dumps(body),
        )

    def handle_api(route, request) -> None:
        parsed = urlparse(request.url)
        if request.method == "DELETE" and parsed.path.startswith("/api/drawings/"):
            delete_hits.append(request.url)
            fulfill_json(route, {"ok": False, "reason": "delete_should_not_fire"}, status=500)
            return
        if parsed.path == "/api/drawings/all":
            fulfill_json(route, {"ok": True, "drawings": [drawing]})
            return
        if parsed.path == "/api/drawings":
            fulfill_json(route, {"ok": True, "drawings": [drawing]})
            return
        if parsed.path == "/api/conditionals":
            fulfill_json(route, {"ok": True, "count": 1, "conditionals": [conditional]})
            return
        if parsed.path == "/api/live-execution/account":
            fulfill_json(
                route,
                {
                    "ok": True,
                    "total_equity": 1000.0,
                    "available": 1000.0,
                    "positions": [],
                    "pending_orders": [
                        {"orderId": "ui-delete-guard-oid", "symbol": "BTCUSDT", "status": "live"}
                    ],
                },
            )
            return
        route.continue_()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.route("**/api/**", handle_api)

            page.goto(BASE, wait_until="commit", timeout=20000)
            page.wait_for_selector("#v2-cond-panel", timeout=15000)
            page.wait_for_selector('[data-toggle-symbol="BTCUSDT"]', timeout=15000)
            page.click('[data-toggle-symbol="BTCUSDT"]')
            selector = f'button.mydraw-del[data-line-id="{LINE_ID}"]'
            page.wait_for_selector(selector, state="attached", timeout=15000)
            disabled = page.eval_on_selector(selector, "el => el.disabled")
            log.log(f"delete button disabled={disabled}")
            if not disabled:
                raise AssertionError("line delete button is enabled despite active conditional")

            box = page.locator(selector).bounding_box()
            if not box:
                raise AssertionError("delete button has no bounding box")
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.wait_for_timeout(500)

            log.log(f"delete requests fired={len(delete_hits)}")
            if delete_hits:
                raise AssertionError(f"disabled delete still fired DELETE: {delete_hits}")
            if console_errors:
                log.log("console_errors=" + json.dumps(console_errors[-5:], ensure_ascii=False))

            log.log("PASS active order kept line delete disabled and no DELETE request fired")
            browser.close()
            return 0
    except Exception as exc:
        log.log(f"FAIL {type(exc).__name__}: {exc}")
        return 1
    finally:
        log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
