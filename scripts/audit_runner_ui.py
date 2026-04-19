"""Audit the current runner/live panel UI state.

Per CLAUDE.md: UI claims need browser evidence. This navigates to every
runner-related view and screenshots it, plus dumps visible text and buttons.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = "http://localhost:8000/v2"
OUT = Path("data/logs/ui_tests/runner_audit")
OUT.mkdir(parents=True, exist_ok=True)

report = {"base": BASE, "views": {}, "console": [], "network_4xx": []}


def audit_view(page, name, nav_selector=None):
    info = {"name": name, "screenshot": None, "headings": [], "buttons": [],
            "status_texts": [], "url": None, "visible_text_head": None}
    try:
        if nav_selector:
            btn = page.locator(nav_selector).first
            if btn.count() > 0 and btn.is_visible():
                btn.click()
                page.wait_for_load_state("networkidle", timeout=8000)
                time.sleep(1.5)
            else:
                info["nav_found"] = False
        info["url"] = page.url
        path = OUT / f"{name}.png"
        page.screenshot(path=str(path), full_page=True)
        info["screenshot"] = str(path)
        info["headings"] = page.eval_on_selector_all(
            "h1, h2, h3, .rn-title, .page-title, .monitor-badge, .rn-status-pill",
            "els => els.map(e => e.textContent?.trim()).filter(Boolean).slice(0, 40)"
        )
        info["buttons"] = page.eval_on_selector_all(
            "button",
            "els => els.map(e => (e.textContent||'').trim()).filter(t => t.length>0 && t.length<60).slice(0, 60)"
        )
        # Get first chunk of visible body text so we can see what user sees
        info["visible_text_head"] = (page.inner_text("body") or "")[:1500]
    except Exception as e:
        info["error"] = repr(e)
    report["views"][name] = info


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = ctx.new_page()
    page.on("console", lambda m: report["console"].append(
        {"t": m.type, "text": (m.text or "")[:200]}))
    page.on("response", lambda r: report["network_4xx"].append(
        {"status": r.status, "url": r.url[:200]}) if r.status >= 400 else None)

    print(f"Opening {BASE}...", flush=True)
    page.goto(BASE, timeout=30000, wait_until="domcontentloaded")
    time.sleep(3.0)

    # Home / market view as loaded
    audit_view(page, "00_home")

    # Nav to each sidebar item. We'll try common selectors.
    nav_targets = [
        ("01_market", 'a[href*="market"], button:has-text("市场")'),
        ("02_strategy", 'a[href*="strategy"], button:has-text("策略")'),
        ("03_runner", 'a[href*="runner"], button:has-text("Runner"), button:has-text("运行器")'),
        ("04_live", 'a[href*="live"], button:has-text("实盘")'),
        ("05_monitor", 'a[href*="monitor"], button:has-text("Monitor"), button:has-text("监控")'),
        ("06_execution", 'a[href*="execution"], button:has-text("执行")'),
    ]
    for name, sel in nav_targets:
        audit_view(page, name, nav_selector=sel)

    browser.close()

# Persist report
rpt_path = OUT / "report.json"
rpt_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nScreenshots: {OUT}")
print(f"Report:      {rpt_path}")
print(f"Console msgs: {len(report['console'])}")
print(f"4xx/5xx: {len(report['network_4xx'])}")
for v, info in report["views"].items():
    sc = info.get("screenshot", "?")
    btns = len(info.get("buttons", []))
    err = info.get("error", "")
    print(f"  {v}: btns={btns}  {sc}  {err}")
