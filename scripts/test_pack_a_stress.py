"""Pack A stress harness — 20 runs x 4 flows = 80 total.

Covers the four UI flows shipped in Phase 3 Pack A:

  1. start_live_confirm   — dry_run off + click 启动 → 真钱启动确认 modal,
                            OK disabled until user types "LIVE" (case-sensitive).
  2. flatten_all_confirm  — click 紧急平仓 → modal, OK disabled until
                            user types "FLATTEN".
  3. halt_viz             — inject a faked /api/mar-bb/state response that
                            reports daily_risk.halted=true. Assert the
                            #rn-status-pill flips to the halted class with
                            "HALTED", "-" and "TODAY" text; assert
                            #rn-halt-countdown and #rn-btn-reset-halt become
                            visible.
  4. preview_dot_tracks   — start the trendline tool, patch rAF to a
                            synchronous setTimeout(0), fire four mousemove
                            events at distinct client coords, and assert the
                            SVG preview circle's cx/cy matches each target
                            within 3 px.

Each of the four flows is run 20 times against a single shared browser
context (re-navigating /v2 between runs to reset state). Log lines go to
    data/logs/ui_tests/pack_a_stress_{timestamp}.log
in the format
    {iso_ts} {flow_name} run={i}/20 {pass|fail} {detail}

Exit 0 only if all 80 runs pass. Any failures keep the loop going so we
can see ratios, and each flow's 20-run block is printed as a summary at
the end.

Per project CLAUDE.md, this is the canonical "done" gate for Pack A.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

from playwright.sync_api import sync_playwright, Page, BrowserContext, Error as PWError

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8000/v2"
RUNS_PER_FLOW = 20
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "data" / "logs" / "ui_tests"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Log helpers
# ─────────────────────────────────────────────────────────────
_log_file: "any" = None


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def logln(line: str) -> None:
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


def record(flow: str, idx: int, ok: bool, detail: str) -> None:
    logln(f"{_ts()} {flow} run={idx}/{RUNS_PER_FLOW} {'pass' if ok else 'fail'} {detail}")


# ─────────────────────────────────────────────────────────────
# Shared: boot /v2 and wait for the chart + runner modules
# ─────────────────────────────────────────────────────────────
def hard_reset(page: Page) -> None:
    """Full navigation reset so state machines / stuck modals don't leak."""
    try:
        page.goto("about:blank", wait_until="commit", timeout=5000)
        page.wait_for_timeout(100)
    except Exception:
        pass
    page.goto(BASE, wait_until="commit", timeout=30000)
    # Wait for chart container + drawing overlay to signal "boot finished"
    page.wait_for_function(
        """() => {
            const el = document.querySelector('#chart-container');
            if (!el) return false;
            return !!el.querySelector('canvas') && !!el.querySelector('svg.chart-drawing-overlay');
        }""",
        timeout=30000,
    )
    # Extra beat for market data / candleSeries.setData
    page.wait_for_timeout(1500)


def open_runner_view(page: Page) -> None:
    """Click 策略 tab and wait for .rn-root."""
    btn = page.query_selector(".v2-nav-btn[data-view='runner']")
    if btn is None:
        raise RuntimeError("nav button 策略 missing")
    btn.click()
    page.wait_for_selector("#view-runner .rn-root", timeout=10000)
    # Wait for the first poll to populate config form
    page.wait_for_function(
        """() => {
            const el = document.querySelector('[name=top_n]');
            return el && el.value && el.value !== '';
        }""",
        timeout=15000,
    )


# ─────────────────────────────────────────────────────────────
# Flow 1 — start-live confirm gate
# ─────────────────────────────────────────────────────────────
def flow_start_live_confirm(page: Page) -> tuple[bool, str]:
    hard_reset(page)
    open_runner_view(page)

    # Uncheck dry_run (it defaults to whatever backend says; force false).
    page.evaluate("""() => {
        const el = document.querySelector('[name=dry_run]');
        if (!el) throw new Error('dry_run checkbox missing');
        if (el.checked) {
            el.checked = false;
            el.dispatchEvent(new Event('change', {bubbles: true}));
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }
        // Mark form dirty so refresh() doesn't overwrite
        const form = document.getElementById('rn-config');
        if (form) form.dataset.dirty = '1';
    }""")

    # The Start button only pops the modal when runner isn't already running.
    # If state says running, stop it first (no modal on the stop path — it
    # just confirms via window.confirm, which we accept via dialog handler).
    state_running = page.evaluate("""() => {
        // runner_view caches _lastState in module scope; probe via pill class
        const p = document.querySelector('#rn-status-pill');
        return !!p && /running/.test(p.className);
    }""")
    if state_running:
        # Accept the confirm() on stop
        page.once("dialog", lambda d: d.accept())
        page.click("#rn-btn-stop")
        page.wait_for_timeout(800)

    # Click 启动/应用配置 — should open the typed-confirm modal
    page.click("#rn-btn-start")
    try:
        page.wait_for_selector(".rn-modal-overlay .rn-modal", timeout=5000)
    except Exception as e:
        return False, f"modal never appeared: {e}"

    title = page.eval_on_selector(".rn-modal-title", "el => el.textContent || ''")
    if title.strip() != "真钱启动确认":
        # Close whatever modal opened then bail
        page.evaluate("""() => document.querySelectorAll('.rn-modal-overlay').forEach(o => o.remove())""")
        return False, f"modal title mismatch: {title!r}"

    input_sel = ".rn-modal-overlay .rn-modal-input"
    ok_sel = ".rn-modal-overlay [data-act=ok]"
    cancel_sel = ".rn-modal-overlay [data-act=cancel]"

    # Type lowercase "live" → OK must stay disabled
    page.fill(input_sel, "live")
    disabled_after_lower = page.eval_on_selector(ok_sel, "el => el.disabled")
    if not disabled_after_lower:
        page.click(cancel_sel)
        return False, "ok enabled after typing lowercase 'live'"

    # Clear and type uppercase "LIVE" → OK must enable
    page.fill(input_sel, "")
    page.fill(input_sel, "LIVE")
    # The input handler runs synchronously on 'input' event
    enabled_after_upper = page.eval_on_selector(ok_sel, "el => !el.disabled")
    if not enabled_after_upper:
        page.click(cancel_sel)
        return False, "ok stayed disabled after typing 'LIVE'"

    # Click 取消 — DO NOT click ok (we don't want to actually start live).
    page.click(cancel_sel)
    try:
        page.wait_for_selector(".rn-modal-overlay", state="detached", timeout=3000)
    except Exception:
        return False, "modal did not close after cancel"

    return True, "title+disable+enable+cancel ok"


# ─────────────────────────────────────────────────────────────
# Flow 2 — flatten-all confirm gate
# ─────────────────────────────────────────────────────────────
def flow_flatten_all_confirm(page: Page) -> tuple[bool, str]:
    hard_reset(page)
    open_runner_view(page)

    page.click("#rn-btn-flatten")
    try:
        page.wait_for_selector(".rn-modal-overlay .rn-modal", timeout=5000)
    except Exception as e:
        return False, f"flatten modal never appeared: {e}"

    title = page.eval_on_selector(".rn-modal-title", "el => el.textContent || ''")
    if title.strip() != "紧急平仓":
        page.evaluate("""() => document.querySelectorAll('.rn-modal-overlay').forEach(o => o.remove())""")
        return False, f"modal title mismatch: {title!r}"

    input_sel = ".rn-modal-overlay .rn-modal-input"
    ok_sel = ".rn-modal-overlay [data-act=ok]"
    cancel_sel = ".rn-modal-overlay [data-act=cancel]"

    # Type "FLAT" → OK disabled
    page.fill(input_sel, "FLAT")
    disabled_after_partial = page.eval_on_selector(ok_sel, "el => el.disabled")
    if not disabled_after_partial:
        page.click(cancel_sel)
        return False, "ok enabled after typing partial 'FLAT'"

    # Type full "FLATTEN" → OK enabled
    page.fill(input_sel, "")
    page.fill(input_sel, "FLATTEN")
    enabled_after_full = page.eval_on_selector(ok_sel, "el => !el.disabled")
    if not enabled_after_full:
        page.click(cancel_sel)
        return False, "ok stayed disabled after typing 'FLATTEN'"

    page.click(cancel_sel)
    try:
        page.wait_for_selector(".rn-modal-overlay", state="detached", timeout=3000)
    except Exception:
        return False, "modal did not close after cancel"

    return True, "title+disable+enable+cancel ok"


# ─────────────────────────────────────────────────────────────
# Flow 3 — halt viz
# ─────────────────────────────────────────────────────────────
_HALT_FETCH_PATCH = """() => {
    if (window.__packaHaltPatched) return;
    window.__packaHaltPatched = true;
    const origFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
        try {
            const url = typeof input === 'string' ? input : (input && input.url) || '';
            if (url.includes('/api/mar-bb/state')) {
                const body = JSON.stringify({
                    state: {
                        status: 'running',
                        config: { top_n: 100, timeframe: '1h', scan_interval_s: 60,
                                  notional_usd: 12, leverage: 5,
                                  max_concurrent_positions: 5, dry_run: false,
                                  strategies: ['mar_bb'] },
                        daily_risk: { date: '2026-04-19', halted: true,
                                      last_dd_pct: 0.5, limit_pct: 0.3 },
                        last_scan_ts: Math.floor(Date.now()/1000),
                        scans_completed: 1, last_scan_duration_s: 1,
                        signals_detected: 0, orders_submitted: 0, orders_rejected: 0,
                    }
                });
                return new Response(body, { status: 200,
                    headers: { 'Content-Type': 'application/json' } });
            }
        } catch(_) {}
        return origFetch(input, init);
    };
}"""


def flow_halt_viz(page: Page) -> tuple[bool, str]:
    hard_reset(page)
    # Install the fetch override BEFORE switching to the runner view so
    # the first poll after mount sees the halted state.
    page.evaluate(_HALT_FETCH_PATCH)

    # Also invalidate the runner view's cached fetches — the util/fetch.js
    # cache TTL is 30s, so if a prior non-halted response is still cached
    # the override won't take effect. Hard-reset + override installed
    # before runner mount sidesteps the cache entirely.
    open_runner_view(page)

    # runner_view polls every 3s. Wait long enough for at least one poll
    # with the patched fetch to complete and a render to run.
    page.wait_for_timeout(3500)

    info = page.evaluate("""() => {
        const pill = document.querySelector('#rn-status-pill');
        const cd = document.querySelector('#rn-halt-countdown');
        const rbt = document.querySelector('#rn-btn-reset-halt');
        const visible = (el) => {
            if (!el) return false;
            const s = window.getComputedStyle(el);
            if (s.display === 'none' || s.visibility === 'hidden') return false;
            return el.offsetWidth > 0 && el.offsetHeight > 0;
        };
        return {
            pill_class: pill ? pill.className : null,
            pill_text: pill ? pill.innerText : null,
            pill_textContent: pill ? pill.textContent : null,
            cd_visible: visible(cd),
            cd_text: cd ? cd.innerText : '',
            rbt_visible: visible(rbt),
        };
    }""")

    fails: list[str] = []
    cls = info.get("pill_class") or ""
    if "rn-status-halted" not in cls:
        fails.append(f"pill class missing rn-status-halted ({cls!r})")

    # innerText in Chromium applies text-transform, so "HALTED" and "TODAY"
    # will be uppercased from the underlying "today". textContent stays raw.
    raw = info.get("pill_textContent") or ""
    itext = info.get("pill_text") or ""
    combined = itext + "|" + raw
    if "HALTED" not in combined.upper():
        fails.append(f"pill text missing HALTED ({combined!r})")
    if "-" not in combined:
        fails.append(f"pill text missing '-' sign ({combined!r})")
    if "TODAY" not in combined.upper():
        fails.append(f"pill text missing TODAY ({combined!r})")

    if not info.get("cd_visible"):
        fails.append("halt-countdown not visible")
    elif "UTC 午夜" not in (info.get("cd_text") or ""):
        fails.append(f"halt-countdown missing 'UTC 午夜' ({info.get('cd_text')!r})")

    if not info.get("rbt_visible"):
        fails.append("reset-halt button not visible")

    if fails:
        return False, "; ".join(fails)
    return True, f"pill={info['pill_textContent']!r} cd={info['cd_text']!r}"


# ─────────────────────────────────────────────────────────────
# Flow 4 — mouse preview dot tracks in drawing_first_point
# ─────────────────────────────────────────────────────────────
_TRENDLINE_SETUP = """async () => {
    // Dynamic import returns the already-loaded module instance so the
    // singleton _chart/_svg/_container state (from main.js boot) is live.
    const mod = await import('/js/workbench/drawings/chart_drawing.js');
    // Patch rAF so scheduleRender() flushes synchronously-ish. Without
    // this, the first mousemove sets _rafPending=true and the test can
    // race the real 16ms rAF tick.
    window.requestAnimationFrame = (cb) => setTimeout(cb, 0);
    // Start the trendline tool → tx.state = 'drawing_first_point'
    mod.startTrendlineTool();
    return true;
}"""


def flow_preview_dot_tracks(page: Page) -> tuple[bool, str]:
    hard_reset(page)

    try:
        page.evaluate(_TRENDLINE_SETUP)
    except Exception as e:
        return False, f"setup failed: {e}"

    # Resolve chart container rect in client coords so we can pick 4 targets
    # inside its bounds and convert to container-relative expected cx/cy.
    rect = page.evaluate("""() => {
        const el = document.querySelector('#chart-container');
        const r = el.getBoundingClientRect();
        return {x: r.x, y: r.y, w: r.width, h: r.height};
    }""")
    if not rect or rect["w"] < 100 or rect["h"] < 100:
        return False, f"chart rect too small: {rect}"

    # Four targets spread across the interior so at least some of them
    # are in past candle territory (timeToCoordinate succeeds) rather than
    # extrapolated future-X space, where screenToData round-trip is
    # slightly lossier.
    xs = [0.25, 0.45, 0.65, 0.80]
    ys = [0.35, 0.55, 0.45, 0.60]
    targets: list[tuple[float, float]] = []
    for fx, fy in zip(xs, ys):
        cx_client = rect["x"] + rect["w"] * fx
        cy_client = rect["y"] + rect["h"] * fy
        targets.append((cx_client, cy_client))

    failures: list[str] = []
    for i, (cx_client, cy_client) in enumerate(targets):
        # Dispatch a mousemove at the exact client coord the test expects.
        # Using page.mouse.move would also work but dispatchEvent gives
        # exact clientX/clientY control without cursor path interpolation.
        page.evaluate(
            """({x, y}) => {
                const el = document.querySelector('#chart-container');
                const ev = new MouseEvent('mousemove', {
                    bubbles: true, cancelable: true, view: window,
                    clientX: x, clientY: y, button: 0,
                });
                el.dispatchEvent(ev);
            }""",
            {"x": cx_client, "y": cy_client},
        )
        # Flush the setTimeout(0) that replaces rAF
        page.wait_for_timeout(25)

        result = page.evaluate("""() => {
            const svg = document.querySelector('svg.chart-drawing-overlay');
            if (!svg) return {err: 'no svg overlay'};
            const circles = svg.querySelectorAll('circle');
            if (!circles.length) return {err: 'no circle rendered', count: 0};
            // The preview circle is the only one in drawing_first_point
            // state (no selected line → no anchor circles). If multiple
            // got drawn, take the last one (drawn on top per render order).
            const c = circles[circles.length - 1];
            return {
                cx: Number(c.getAttribute('cx')),
                cy: Number(c.getAttribute('cy')),
                count: circles.length,
                rect_x: svg.getBoundingClientRect().left,
                rect_y: svg.getBoundingClientRect().top,
            };
        }""")
        if "err" in result:
            failures.append(f"t{i+1}:{result['err']}")
            continue

        # Expected SVG coords = client - svg_rect (svg has inset:0 so this
        # equals client - container_rect).
        expected_cx = cx_client - result["rect_x"]
        expected_cy = cy_client - result["rect_y"]
        dx = abs(result["cx"] - expected_cx)
        dy = abs(result["cy"] - expected_cy)
        if dx > 3.0 or dy > 3.0:
            failures.append(
                f"t{i+1}:Δ=({dx:.2f},{dy:.2f}) got=({result['cx']:.1f},{result['cy']:.1f}) "
                f"want=({expected_cx:.1f},{expected_cy:.1f})"
            )

    if failures:
        return False, " | ".join(failures)
    return True, f"4/4 targets within 3px"


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────
FLOWS: list[tuple[str, Callable[[Page], tuple[bool, str]]]] = [
    ("start_live_confirm",  flow_start_live_confirm),
    ("flatten_all_confirm", flow_flatten_all_confirm),
    ("halt_viz",            flow_halt_viz),
    ("preview_dot_tracks",  flow_preview_dot_tracks),
]


def main() -> int:
    global _log_file

    ts_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = LOG_DIR / f"pack_a_stress_{ts_stamp}.log"
    _log_file = log_path.open("w", encoding="utf-8", buffering=1)

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=str(ROOT),
    ).stdout.strip() or "?"

    logln(f"=== pack_a_stress @ {_ts()} git={sha} base={BASE} ===")
    logln(f"log_path={log_path}")

    per_flow: dict[str, tuple[int, int]] = {name: (0, 0) for name, _ in FLOWS}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx: BrowserContext = browser.new_context(viewport={"width": 1680, "height": 980})
        page = ctx.new_page()

        console_all: list[str] = []
        page.on("console", lambda m: console_all.append(f"[{m.type}] {m.text[:200]}"))

        for flow_name, flow_fn in FLOWS:
            logln(f"--- flow: {flow_name} ---")
            for i in range(1, RUNS_PER_FLOW + 1):
                t0 = time.time()
                try:
                    ok, detail = flow_fn(page)
                except PWError as e:
                    ok, detail = False, f"playwright_err: {str(e).splitlines()[0][:160]}"
                except Exception as e:
                    tb = traceback.format_exc().splitlines()[-1][:160]
                    ok, detail = False, f"exc: {tb}"
                dt = time.time() - t0
                passes, fails = per_flow[flow_name]
                if ok:
                    per_flow[flow_name] = (passes + 1, fails)
                else:
                    per_flow[flow_name] = (passes, fails + 1)
                record(flow_name, i, ok, f"{detail} (t={dt:.1f}s)")

        browser.close()

    # ── Summary ─────────────────────────────────────────────
    total_pass = sum(p for p, _ in per_flow.values())
    total_fail = sum(f for _, f in per_flow.values())
    total = total_pass + total_fail

    logln("")
    logln("=" * 60)
    logln(f"SUMMARY: {total_pass}/{total} passed")
    for name, (p, f) in per_flow.items():
        logln(f"  {name:22s} {p}/{RUNS_PER_FLOW}")
    logln("=" * 60)

    _log_file.close()
    print(f"\nlog written: {log_path}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
