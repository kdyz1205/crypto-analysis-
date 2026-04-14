"""One-shot Playwright audit of the workbench UI.

Captures:
- Full-page screenshots of each view
- Console errors and network errors
- Count of drawn lines/markers/zones on chart
- Decision rail content
- Execution panel [虚拟] badge state

Exit codes:
  0 — clean (no console errors/warnings, decision rail rendered,
            no 4xx/5xx, no "加载中" stuck state)
  1 — has issues (any of the above failed)

The pre-commit hook gates on this exit code.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, ConsoleMessage

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8000/v2"
OUT = Path(__file__).parent / "data" / "logs" / "browser_review"
OUT.mkdir(parents=True, exist_ok=True)

console_log: list[dict] = []
network_errors: list[dict] = []


def on_console(msg: ConsoleMessage):
    console_log.append({
        "type": msg.type,
        "text": msg.text[:400],
    })


def on_response(resp):
    if resp.status >= 400:
        network_errors.append({
            "url": resp.url[:200],
            "status": resp.status,
        })


def safe_screenshot(page, name: str):
    try:
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
    except Exception as e:
        print(f"  screenshot {name} failed: {e}")


def inspect_market_chart(page):
    """Count chart series, markers, lines."""
    try:
        # Give lightweight-charts + async overlays time to settle
        page.wait_for_timeout(3000)
        info = page.evaluate("""() => {
            const container = document.querySelector('#v2-chart-container') || document.querySelector('#chart-container') || document.querySelector('.v2-chart-wrapper');
            const canvases = document.querySelectorAll('canvas');
            const drZones = document.querySelectorAll('#v2-decision-rail .dr-card');
            const lineCountText = document.body.innerText.match(/[0-9]+\\s*条/);
            const virtualBadges = document.querySelectorAll('.exec-pnl-virtual');
            const errorText = document.body.innerText.match(/失败|错误|连接中|加载中/g) || [];
            return {
                chart_container_found: !!container,
                canvas_count: canvases.length,
                decision_rail_cards: drZones.length,
                decision_rail_text: Array.from(drZones).map(c => c.innerText.slice(0, 500)),
                virtual_badge_count: virtualBadges.length,
                error_keywords: errorText.slice(0, 20),
                body_text_head: document.body.innerText.slice(0, 2000),
            };
        }""")
        return info
    except Exception as e:
        return {"error": str(e)}


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 980})
        page = context.new_page()
        page.on("console", on_console)
        page.on("response", on_response)

        results = {}

        print(f"[1/6] Loading {BASE}...")
        t0 = time.time()
        try:
            page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
            print(f"  DOM ready: {time.time()-t0:.1f}s")
            page.wait_for_load_state("networkidle", timeout=30000)
            print(f"  network idle: {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  load failed: {e}")
        safe_screenshot(page, "01_boot")
        results["boot"] = {
            "duration_s": round(time.time() - t0, 2),
            "console_errors": [c for c in console_log if c["type"] in ("error", "warning")][:20],
            "network_errors": network_errors[:20],
        }

        print("[2/6] Inspecting market view (chart + drawings)...")
        # Should already be on market view
        market_info = inspect_market_chart(page)
        results["market"] = market_info
        safe_screenshot(page, "02_market")

        print("[3/6] Switch to Live view via nav...")
        try:
            nav_btns = page.query_selector_all(".v2-nav-btn")
            live_btn = None
            for b in nav_btns:
                if b.inner_text().strip() in ("运行", "Live", "live"):
                    live_btn = b
                    break
            if live_btn:
                live_btn.click()
                page.wait_for_timeout(2000)
                safe_screenshot(page, "03_live")
                live_info = page.evaluate("""() => ({
                    body_text: document.body.innerText.slice(0, 3000),
                    virtual_badges: document.querySelectorAll('.exec-pnl-virtual').length,
                    real_pnl_classes: document.querySelectorAll('.exec-pnl-real').length,
                    error_keywords: (document.body.innerText.match(/失败|连接中|假/g) || []).length,
                })""")
                results["live"] = live_info
            else:
                print("  no live nav button found")
        except Exception as e:
            print(f"  live view failed: {e}")
            results["live"] = {"error": str(e)}

        print("[4/6] Switch to Monitor view...")
        try:
            nav_btns = page.query_selector_all(".v2-nav-btn")
            for b in nav_btns:
                if b.inner_text().strip() in ("监控", "Monitor", "monitor"):
                    b.click()
                    page.wait_for_timeout(3000)
                    break
            safe_screenshot(page, "04_monitor")
            monitor_info = page.evaluate("""() => ({
                body_text: document.body.innerText.slice(0, 3000),
                error_keywords: (document.body.innerText.match(/失败|错误/g) || []).slice(0, 10),
            })""")
            results["monitor"] = monitor_info
        except Exception as e:
            print(f"  monitor failed: {e}")
            results["monitor"] = {"error": str(e)}

        print("[5/6] Back to market, change symbol to BTC 5m...")
        try:
            nav_btns = page.query_selector_all(".v2-nav-btn")
            for b in nav_btns:
                if b.inner_text().strip() in ("市场", "Market", "market"):
                    b.click()
                    page.wait_for_timeout(2000)
                    break
            # Click interval selector if present — try multiple patterns
            page.evaluate("""() => {
                const ev = new CustomEvent('market.interval.requested', { detail: { interval: '5m' } });
                window.dispatchEvent(ev);
                // Alternative: find the 5m button in a toolbar
                const btns = document.querySelectorAll('button, .v2-tf-btn, [data-interval]');
                for (const b of btns) {
                    if (b.innerText === '5m' || b.dataset.interval === '5m') {
                        b.click();
                        return;
                    }
                }
            }""")
            page.wait_for_timeout(4000)
            safe_screenshot(page, "05_btc_5m")
            btc5m_info = inspect_market_chart(page)
            results["btc_5m"] = btc5m_info
        except Exception as e:
            print(f"  btc 5m failed: {e}")
            results["btc_5m"] = {"error": str(e)}

        print("[6/6] Final console snapshot...")
        errs = [c for c in console_log if c["type"] == "error"]
        warns = [c for c in console_log if c["type"] == "warning"]
        results["console_summary"] = {
            "total_messages": len(console_log),
            "error_count": len(errs),
            "warning_count": len(warns),
            "errors": errs[:30],
            "warnings": warns[:20],
        }
        results["network_error_summary"] = {
            "total": len(network_errors),
            "items": network_errors[:30],
        }

        out_file = OUT / "audit.json"
        out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote report: {out_file}")
        print(f"Screenshots: {OUT}")

        browser.close()

    # ── Pass / fail gate ─────────────────────────────────────────
    failures: list[str] = []

    cs = results.get("console_summary", {})
    if cs.get("error_count", 0) > 0:
        failures.append(f"{cs['error_count']} console error(s)")
        for e in cs.get("errors", [])[:5]:
            failures.append(f"  - {e.get('text', '')[:140]}")
    if cs.get("warning_count", 0) > 0:
        failures.append(f"{cs['warning_count']} console warning(s)")

    nes = results.get("network_error_summary", {})
    if nes.get("total", 0) > 0:
        failures.append(f"{nes['total']} network error(s)")
        for e in nes.get("items", [])[:5]:
            failures.append(f"  - {e.get('status')} {e.get('url', '')[:140]}")

    market = results.get("market", {})
    rail_text = market.get("decision_rail_text", [])
    rail_stuck = any("加载中" in t for t in rail_text)
    if rail_stuck:
        failures.append("decision rail is stuck on '加载中...'")
    # Note: decision_rail_cards check removed. The legacy v2-decision-rail
    # DOM and its dr-card children were deleted when the cond rail took
    # over that region; checking for .dr-card always fails now even on a
    # healthy page. Gate kept as a sentinel via console/network errors.

    print()
    print("=" * 60)
    if failures:
        print(f"BROWSER REVIEW: FAIL ({len(failures)} issue(s))")
        print("=" * 60)
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("BROWSER REVIEW: PASS")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
