"""Browser proof: one drag gesture commits exactly one drawing PATCH.

Touches the real /v2 page, draws a real line, drags it once, and asserts
the frontend only emits a single PATCH /api/drawings/{id} for that gesture.
This guards the chart_drawing commit-on-mouseup contract after editing
chart_drawing.js.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000/v2"
LOG_PATH = ROOT / "data" / "logs" / "ui_tests" / "drag_line_single_patch.log"


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


def _delete_line(manual_line_id: str) -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:8000/api/drawings/{urllib.parse.quote(manual_line_id)}",
        method="DELETE",
    )
    try:
        urllib.request.urlopen(req, timeout=8).read()
    except Exception:
        pass


def main() -> int:
    log = Logger()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    log.log(f"=== drag_line_single_patch {started} HEAD={_git_head()} ===")
    manual_line_id = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1680, "height": 960})
            console_all: list[str] = []
            patch_hits: list[str] = []

            def on_console(msg) -> None:
                text = msg.text
                console_all.append(f"[{msg.type}] {text}")

            def on_request(req) -> None:
                nonlocal manual_line_id
                if req.method != "PATCH":
                    return
                if "/api/drawings/" not in req.url:
                    return
                if manual_line_id and manual_line_id not in req.url:
                    return
                patch_hits.append(req.url)

            page.on("console", on_console)
            page.on("request", on_request)

            page.goto(BASE, wait_until="commit", timeout=20000)
            page.wait_for_function(
                """() => {
                    const el = document.querySelector('#chart-container');
                    if (!el) return false;
                    return !!el.querySelector('canvas') && !!el.querySelector('svg.chart-drawing-overlay');
                }""",
                timeout=30000,
            )
            page.wait_for_timeout(2500)

            box = page.eval_on_selector(
                "#chart-container",
                "el => { const r = el.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }",
            )
            log.log(f"chart_box={json.dumps(box)}")

            # Focus chart, arm draw mode, draw one gentle rising line.
            page.mouse.click(box["x"] + 80, box["y"] + 80)
            page.wait_for_timeout(100)
            page.keyboard.press("t")
            page.wait_for_timeout(350)

            x1 = box["x"] + box["w"] * 0.56
            y1 = box["y"] + box["h"] * 0.60
            x2 = box["x"] + box["w"] * 0.82
            y2 = box["y"] + box["h"] * 0.53
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            page.mouse.click(x1, y1)
            page.wait_for_timeout(250)
            page.mouse.move(x2, y2)
            page.wait_for_timeout(150)
            page.mouse.click(x2, y2)
            page.wait_for_timeout(1200)

            for line in reversed(console_all):
                if "commit ok" in line and "manual-" in line:
                    manual_line_id = "manual-" + line.split("manual-", 1)[1].strip()
                    break
            if not manual_line_id:
                raise AssertionError(f"draw commit log missing; tail={console_all[-12:]}")
            log.log(f"manual_line_id={manual_line_id}")

            # Drag once: body mousedown -> move -> mouseup. The contract is
            # one gesture => one PATCH after mouseup.
            page.mouse.move(mid_x, mid_y)
            page.wait_for_timeout(100)
            page.mouse.down()
            page.mouse.move(mid_x + 70, mid_y - 28, steps=8)
            page.wait_for_timeout(120)
            page.mouse.up()
            page.wait_for_timeout(1800)

            log.log(f"patch_hits={len(patch_hits)}")
            if patch_hits:
                log.log("patch_urls=" + json.dumps(patch_hits, ensure_ascii=False))
            if len(patch_hits) != 1:
                raise AssertionError(f"expected 1 drawing PATCH for one drag gesture, got {len(patch_hits)}")

            log.log("PASS one drag gesture emitted exactly one drawing PATCH")
            browser.close()
            return 0
    except Exception as exc:
        log.log(f"FAIL {type(exc).__name__}: {exc}")
        return 1
    finally:
        if manual_line_id:
            _delete_line(manual_line_id)
        log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
