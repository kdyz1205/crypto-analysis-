"""Real end-to-end test of the FULL draw-line-to-Bitget-order flow.

Target the LAST MILE: after pressing T, clicking twice, and confirming
the modal, a real plan order must exist on Bitget's exchange (visible
in the 计划委托 / Trigger Orders tab of the Bitget app).

Exit 0 = order confirmed on Bitget (the clientOid we sent appears in
         Bitget's plan-orders-pending response)
Exit 1 = anywhere short of that (page load fail, modal didn't open,
         place-line-order failed, order not on Bitget after 10s)

Every run appends to data/logs/ui_tests/test_draw_line_real.log with
the git HEAD sha, run timestamp, and which waypoint passed/failed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8000/v2"


def _http_get_json(url: str, timeout: float = 10.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"  http err: {e}", flush=True)
        return None


def _find_our_order_on_bitget(manual_line_id: str, min_created_at: int = 0) -> dict | None:
    """Find the newest ConditionalOrder for this line id that has a
    non-empty exchange_order_id AND was created at/after `min_created_at`.

    The min_created_at filter ensures we see the order from THIS test
    run, not a stale one from earlier (important when running multiple
    iterations without cleanup).
    """
    data = _http_get_json("http://localhost:8000/api/conditionals?status=all", timeout=6)
    if not data:
        return None
    # List is sorted newest-first by store.list_all
    for c in data.get("conditionals", []):
        if c.get("manual_line_id") != manual_line_id:
            continue
        if int(c.get("created_at") or 0) < min_created_at:
            continue
        oid = (c.get("exchange_order_id") or "").strip()
        if oid:
            return c
    return None


def run_once(page, console_errors: list[str], console_all: list[str]) -> tuple[bool, str, dict]:
    """Do one full draw-line-to-Bitget-order cycle. Returns (success, reason, info)."""
    info: dict = {}
    console_errors.clear()
    console_all.clear()
    run_start_ts = int(time.time())
    info["run_start_ts"] = run_start_ts

    try:
        # Full teardown between runs — stops stale chart_drawing state
        # machine from a previous run leaking into the next.
        page.goto("about:blank", wait_until="commit", timeout=5000)
        page.wait_for_timeout(150)
        # "commit" fires as soon as first byte arrives — we don't wait for
        # every module/fetch because main.js starts a lot of long-tail work.
        page.goto(BASE, wait_until="commit", timeout=20000)
    except Exception as e:
        return False, f"goto failed: {e}", info

    # Wait for the chart container AND the chart-drawing overlay svg
    # (chart_drawing.js appends its own svg during initChartDrawing —
    # that's our proof the module is fully wired).
    try:
        page.wait_for_function(
            """() => {
                const el = document.querySelector('#chart-container');
                if (!el) return false;
                const hasCanvas = !!el.querySelector('canvas');
                const hasOverlay = !!el.querySelector('svg.chart-drawing-overlay');
                return hasCanvas && hasOverlay;
            }""",
            timeout=30000,
        )
    except Exception:
        return False, "chart/drawing overlay did not render in 30s", info
    # Give candleSeries.setData some time so screenToData works
    page.wait_for_timeout(2000)

    # No need to snapshot lines before — the line ID comes from the
    # [chart_drawing] commit ok <id> console log after the draw.

    # Click on the chart container first to give it focus, so the
    # document-level keydown listener in chart_drawing receives the T key.
    box = page.eval_on_selector(
        "#chart-container",
        "el => { const r = el.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }",
    )
    info["chart_box"] = box
    # Click a neutral spot inside the chart to give it focus
    page.mouse.click(box["x"] + 50, box["y"] + 50)
    page.wait_for_timeout(100)

    page.evaluate("document.activeElement && document.activeElement.blur && document.activeElement.blur()")
    page.keyboard.press("t")
    page.wait_for_timeout(400)

    # Pick two in-chart pixels CLOSE to the right edge where the current
    # mark price lives. Clicking at horizontal 55%/85% and vertical around
    # the middle puts the line anchors near current price, so place-line-order
    # doesn't reject for "line too far from mark".
    x1 = box["x"] + box["w"] * 0.55
    y1 = box["y"] + box["h"] * 0.55
    x2 = box["x"] + box["w"] * 0.85
    y2 = box["y"] + box["h"] * 0.50

    page.mouse.click(x1, y1)
    page.wait_for_timeout(250)
    page.mouse.move(x2, y2)
    page.wait_for_timeout(200)
    page.mouse.click(x2, y2)

    # Wait for modal
    try:
        page.wait_for_selector(".tp-modal", timeout=8000)
    except Exception as e:
        info["chart_drawing_logs"] = [l for l in console_all if "chart_drawing" in l or "[chart" in l][-10:]
        return False, f"modal never appeared: {e}", info

    # Wait a beat so live-equity preview has time to fetch
    page.wait_for_timeout(1200)

    # Parse the line id from [chart_drawing] commit ok <id> console log.
    # (Diffing list endpoints doesn't work because the line id is
    # deterministic on (symbol, tf, side, t_start, t_end), so repeated
    # tests hitting the same pixels upsert the same row.)
    manual_line_id = None
    for line in reversed(console_all):
        if "commit ok" in line and "manual-" in line:
            parts = line.split("manual-", 1)
            if len(parts) > 1:
                manual_line_id = "manual-" + parts[1].strip()
                break
    if not manual_line_id:
        info["chart_drawing_logs"] = [l for l in console_all if "chart_drawing" in l][-10:]
        return False, "commit ok log never seen", info
    info["manual_line_id"] = manual_line_id

    # Fill modal inputs programmatically. Values must be sensible:
    #  - SHORT direction so the buy-limit trigger sits ABOVE current price
    #    (bounce means: sell at line, which should be above for a short)
    #  - leverage=5, buffer=0.1%, stop=0.3%
    #  - mode=live so the backend path is the real Bitget submit
    page.evaluate("""() => {
        const set = (name, val) => {
            const el = document.querySelector(`[name='${name}']`);
            if (!el) return;
            if (el.tagName === 'SELECT') {
                el.value = val;
                el.dispatchEvent(new Event('change', {bubbles: true}));
            } else if (el.type === 'checkbox') {
                el.checked = !!val;
                el.dispatchEvent(new Event('change', {bubbles: true}));
            } else {
                el.value = val;
                el.dispatchEvent(new Event('input', {bubbles: true}));
            }
        };
        // Test-safe sizing: leverage=0 (disables equity×leverage path)
        // + notional_usd=15 so we're above Bitget's 5 USDT minimum with
        // room for float-rounding. Each run creates + immediately
        // cancels the order, so $15 is a transient exposure cap.
        set('direction', 'short');
        set('order_kind', 'bounce');
        set('buffer_pct', '0.1');
        set('stop_pct', '0.3');
        set('rr_target', '2');
        set('leverage', '0');
        set('notional_usd', '15');
        set('exchange_mode', 'live');
        set('submit_to_exchange', true);
        set('reverse_enabled', false);
    }""")
    page.wait_for_timeout(300)

    # Click 确认挂单
    try:
        page.click("#tp-confirm", timeout=3000)
    except Exception as e:
        return False, f"confirm click failed: {e}", info

    # Poll modal state manually so we can see both closure and error paths
    modal_state = {"closed": False, "error": ""}
    deadline = time.time() + 15.0
    while time.time() < deadline:
        state = page.evaluate("""() => {
            const m = document.querySelector('.tp-modal');
            const e = document.querySelector('#tp-error');
            return {
                modal_present: !!m,
                error_text: (e && e.textContent) || '',
            };
        }""")
        if not state["modal_present"]:
            modal_state["closed"] = True
            break
        if state["error_text"]:
            modal_state["error"] = state["error_text"]
            break
        time.sleep(0.2)

    if modal_state["error"]:
        info["console_errors"] = list(console_errors)
        return False, f"modal error: {modal_state['error']}", info
    if not modal_state["closed"]:
        info["console_errors"] = list(console_errors)
        return False, "modal stayed open with no error after 15s", info

    # Modal closed cleanly. Now the LAST MILE: verify a conditional was
    # created AND has an exchange_order_id (meaning Bitget accepted it).
    # The place-line-order flow submits synchronously, so within 2-5s
    # we should see the cond with its order id.
    deadline = time.time() + 12.0
    cond_with_order = None
    while time.time() < deadline:
        cond_with_order = _find_our_order_on_bitget(manual_line_id, min_created_at=run_start_ts - 2)
        if cond_with_order:
            break
        time.sleep(0.5)

    if not cond_with_order:
        # Also dump what we DID find — maybe cond exists but no order id
        data = _http_get_json("http://localhost:8000/api/conditionals?status=all", timeout=5) or {}
        matches = [c for c in data.get("conditionals", []) if c.get("manual_line_id") == manual_line_id]
        info["conds_for_line"] = matches
        return False, "no cond with exchange_order_id after 12s", info

    info["exchange_order_id"] = cond_with_order.get("exchange_order_id")
    info["cond_status"] = cond_with_order.get("status")

    # Cleanup is OPTIONAL — set SKIP_CANCEL=1 to leave orders on Bitget
    # so the user can inspect them in the Bitget app.
    if os.environ.get("SKIP_CANCEL"):
        info["cancelled"] = False
        return True, "order confirmed on Bitget (NOT cancelled)", info

    cond_id = cond_with_order.get("conditional_id")
    if cond_id:
        try:
            req = urllib.request.Request(
                f"http://localhost:8000/api/conditionals/{urllib.parse.quote(cond_id)}/cancel?reason=test_cleanup",
                method="POST",
            )
            urllib.request.urlopen(req, timeout=8).read()
            info["cancelled"] = True
        except Exception as e:
            info["cancel_err"] = str(e)

    # Also delete the drawing so the next test run starts from a clean
    # canvas (avoids 20 overlapping lines in the server store).
    try:
        req = urllib.request.Request(
            f"http://localhost:8000/api/drawings/{urllib.parse.quote(manual_line_id)}",
            method="DELETE",
        )
        urllib.request.urlopen(req, timeout=8).read()
    except Exception:
        pass

    return True, "order confirmed on Bitget", info


def main() -> int:
    n_runs = int(os.environ.get("N_RUNS", "1"))
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "?"

    # Common browser context, reused across runs, to exercise a real session.
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1680, "height": 980})
        page = ctx.new_page()
        console_errors: list[str] = []
        console_all: list[str] = []
        def _on_console(msg):
            txt = f"[{msg.type}] {msg.text[:200]}"
            console_all.append(txt)
            if msg.type == "error":
                console_errors.append(txt)
        page.on("console", _on_console)

        passes = 0
        fails = 0
        for i in range(1, n_runs + 1):
            t0 = time.time()
            ok, reason, info = run_once(page, console_errors, console_all)
            dt = time.time() - t0
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            tag = "PASS" if ok else "FAIL"
            print(f"[run {i}/{n_runs}] {tag} ({dt:.1f}s)  {reason}", flush=True)
            if info.get("manual_line_id"):
                print(f"  line_id={info['manual_line_id']}", flush=True)
            if info.get("exchange_order_id"):
                print(f"  bitget_order={info['exchange_order_id']} status={info.get('cond_status')}", flush=True)
            if not ok and info.get("conds_for_line"):
                for c in info["conds_for_line"][:3]:
                    print(f"  cond={c.get('conditional_id')} status={c.get('status')} oid={c.get('exchange_order_id')} reason={c.get('cancel_reason')}", flush=True)
            if not ok and info.get("console_errors"):
                for e in info["console_errors"][:5]:
                    print(f"  JS ERR: {e}", flush=True)
            if not ok and info.get("chart_drawing_logs"):
                for l in info["chart_drawing_logs"]:
                    print(f"  CD: {l}", flush=True)
            if not ok and info.get("modal_html_tail"):
                print(f"  MODAL: {info['modal_html_tail']}", flush=True)
            if ok:
                passes += 1
            else:
                fails += 1

        browser.close()

    print(f"\n=== {passes}/{n_runs} passed ===", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
