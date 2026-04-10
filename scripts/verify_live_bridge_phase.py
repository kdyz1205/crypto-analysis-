from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
except ImportError as exc:  # pragma: no cover - environment dependency
    raise SystemExit(
        "selenium is required for this verification script. "
        "Install it with: python -m pip install selenium"
    ) from exc


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PORT = 8073
BASE_URL = f"http://127.0.0.1:{PORT}"
CHROME_CANDIDATES = (
    pathlib.Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    pathlib.Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    pathlib.Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
)

SERVER_BOOTSTRAP = r"""
import os

from server.app import app
from server.execution.types import PaperExecutionConfig, RiskDecision
from server.routers.live_execution import live_engine
from server.routers.paper_execution import paper_engine
from server.strategy.types import StrategySignal
import uvicorn

paper_engine.reset()
live_engine.last_preview_result = None
live_engine.last_submission_result = None
live_engine.reconciliation_by_mode = {"demo": None, "live": None}
live_engine.submissions_by_mode = {"demo": {}, "live": {}}

signal = StrategySignal(
    signal_id="sig-live-verify",
    line_id="line-live-verify",
    symbol="HYPEUSDT",
    timeframe="1h",
    signal_type="REJECTION_SHORT",
    direction="short",
    trigger_mode="pre_limit",
    timestamp=1,
    trigger_bar_index=1,
    score=0.8,
    priority_rank=1,
    entry_price=100.0,
    stop_price=105.0,
    tp_price=90.0,
    risk_reward=2.0,
    confirming_touch_count=3,
    bars_since_last_confirming_touch=1,
    distance_to_line=0.1,
    line_side="resistance",
    reason_code="verify",
    factor_components={},
)
decision = RiskDecision(
    signal_id=signal.signal_id,
    approved=True,
    blocking_reason="",
    risk_amount=30.0,
    proposed_quantity=0.5,
    stop_distance=5.0,
    exposure_after_fill=50.0,
)
paper_engine.order_manager.create_order_intent_from_signal(
    signal,
    decision,
    PaperExecutionConfig(),
    current_bar=1,
    current_ts=1,
)

uvicorn.run(app, host="127.0.0.1", port=8073, log_level="warning")
"""


def _find_browser() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    browser = shutil.which("chrome") or shutil.which("msedge")
    if browser:
        return browser
    raise SystemExit("Could not find Chrome or Edge for selenium verification.")


def _start_server() -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["BITGET_API_KEY"] = ""
    env["BITGET_SECRET_KEY"] = ""
    env["BITGET_SECRET"] = ""
    env["BITGET_API_SECRET"] = ""
    env["BITGET_PASSPHRASE"] = ""
    env["ENABLE_LIVE_TRADING"] = "true"
    env["DRY_RUN"] = "true"
    env["CONFIRM_LIVE_TRADING"] = "false"
    return subprocess.Popen(
        [sys.executable, "-c", SERVER_BOOTSTRAP],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_server(timeout_seconds: float = 45.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{BASE_URL}/v2", timeout=2.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # pragma: no cover - retry path
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Server did not start on {BASE_URL}: {last_error}")


def _build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1600,1200")
    options.binary_location = _find_browser()
    return webdriver.Chrome(options=options)


def _shutdown_server(server: subprocess.Popen[str]) -> None:
    server.terminate()
    try:
        server.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - fallback path
        server.kill()
        server.communicate(timeout=5)


def _wait_for_text(driver: webdriver.Chrome, selector: str, needle: str, timeout_seconds: float = 20.0) -> str:
    needle_lower = needle.lower()
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if elements:
            text = elements[0].text
            if needle_lower in text.lower():
                return text
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {needle!r} in {selector}")


def _wait_for_non_empty_section(driver: webdriver.Chrome, selector: str, timeout_seconds: float = 20.0) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if elements:
            text = elements[0].text.strip()
            if text:
                return text
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for non-empty section {selector}")


def _wait_for_button_disabled(driver: webdriver.Chrome, element_id: str, timeout_seconds: float = 20.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            element = driver.find_element(By.ID, element_id)
            return element.get_attribute('disabled') is not None
        except Exception:  # pragma: no cover - transient rerender path
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for disabled state of #{element_id}")


@dataclass
class VerificationResult:
    passed: bool
    details: dict


def _verify_api() -> dict:
    status_response = httpx.get(f"{BASE_URL}/api/live-execution/status", timeout=20.0)
    status_response.raise_for_status()
    status_payload = status_response.json()["status"]

    paper_state = httpx.get(f"{BASE_URL}/api/paper-execution/state", timeout=20.0)
    paper_state.raise_for_status()
    intents = paper_state.json()["state"]["intents"]
    if not intents:
      raise RuntimeError("Expected a seeded paper intent for live bridge verification")
    intent_id = intents[0]["order_intent_id"]

    reconcile_response = httpx.post(
        f"{BASE_URL}/api/live-execution/reconcile",
        json={"mode": "demo"},
        timeout=20.0,
    )
    reconcile_response.raise_for_status()
    reconcile_payload = reconcile_response.json()["reconciliation"]

    preview_response = httpx.post(
        f"{BASE_URL}/api/live-execution/preview",
        json={"order_intent_id": intent_id, "mode": "demo"},
        timeout=20.0,
    )
    preview_response.raise_for_status()
    preview_payload = preview_response.json()["result"]

    submit_response = httpx.post(
        f"{BASE_URL}/api/live-execution/submit",
        json={"order_intent_id": intent_id, "mode": "demo", "confirm": True},
        timeout=20.0,
    )
    submit_response.raise_for_status()
    submit_payload = submit_response.json()["result"]

    preflight_response = httpx.get(
        f"{BASE_URL}/api/live-execution/preflight",
        params={"mode": "live", "order_intent_id": intent_id},
        timeout=20.0,
    )
    preflight_response.raise_for_status()
    preflight_payload = preflight_response.json()["preflight"]

    return {
        "status": {
            "exchange": status_payload["exchange"],
            "api_key_ready": status_payload["api_key_ready"],
            "blocked_reason": status_payload["blocked_reason"],
            "enabled_flags": status_payload["enabled_flags"],
        },
        "intent_id": intent_id,
        "reconcile": {
            "blocked": reconcile_payload["blocked"],
            "reason": reconcile_payload["reason"],
        },
        "preview": {
            "blocked": preview_payload["blocked"],
            "reason": preview_payload["reason"],
        },
        "submit": {
            "blocked": submit_payload["blocked"],
            "reason": submit_payload["reason"],
        },
        "preflight": {
            "ready": preflight_payload["ready"],
            "blocking_reasons": preflight_payload["blocking_reasons"],
        },
    }


def _verify_browser() -> dict:
    driver = _build_driver()
    try:
        driver.get(f"{BASE_URL}/v2")
        _wait_for_non_empty_section(driver, '#chart-header-v2', timeout_seconds=25.0)
        driver.find_element(By.ID, "v2-exec-toggle").click()
        _wait_for_non_empty_section(driver, '#v2-execution-panel', timeout_seconds=15.0)
        driver.find_element(By.CSS_SELECTOR, '.exec-tab[data-tab="execution"]').click()
        _wait_for_text(driver, '#v2-exec-live-section', 'Live Bridge', timeout_seconds=25.0)
        _wait_for_text(driver, '#v2-exec-live-section', 'BITGET KEYS MISSING')
        bridge_text = _wait_for_non_empty_section(driver, '#v2-exec-live-section', timeout_seconds=25.0)
        if 'BITGET KEYS MISSING' not in bridge_text:
            bridge_text = _wait_for_text(driver, '#v2-exec-live-section', 'BITGET KEYS MISSING', timeout_seconds=25.0)
        driver.find_element(By.ID, 'v2-live-preflight-btn').click()
        _wait_for_text(driver, '#v2-exec-live-section', 'Live Preflight')
        preflight_text = _wait_for_text(driver, '#v2-exec-live-section', 'Bitget API keys', timeout_seconds=25.0)
        result_text = _wait_for_text(driver, '#v2-exec-live-section', 'api_keys_missing', timeout_seconds=25.0)
        submit_demo_disabled = _wait_for_button_disabled(driver, 'v2-live-submit-demo-btn')
        submit_live_disabled = _wait_for_button_disabled(driver, 'v2-live-submit-live-btn')
        return {
            "live_section_excerpt": bridge_text[:800],
            "preflight_excerpt": preflight_text[:1200],
            "result_excerpt": result_text[:1200],
            "submit_demo_disabled": submit_demo_disabled,
            "submit_live_disabled": submit_live_disabled,
        }
    finally:
        driver.quit()


def main() -> int:
    server = _start_server()
    try:
        _wait_for_server()
        api_details = _verify_api()
        browser_details = _verify_browser()
        preflight_excerpt_lower = browser_details["preflight_excerpt"].lower()
        live_section_excerpt_lower = browser_details["live_section_excerpt"].lower()
        result_excerpt_lower = browser_details["result_excerpt"].lower()
        expected_preflight_reasons = {
            "api_keys_ready",
            "confirm_live_trading",
            "dry_run_disabled",
            "reconciliation_present",
            "reconciliation_fresh",
            "intent_gating",
        }
        passed = (
            api_details["status"]["exchange"] == "bitget"
            and api_details["status"]["api_key_ready"] is False
            and api_details["status"]["blocked_reason"] == "api_keys_missing"
            and api_details["reconcile"]["reason"] == "api_keys_missing"
            and api_details["preview"]["reason"] == "api_keys_missing"
            and api_details["submit"]["reason"] == "api_keys_missing"
            and api_details["preflight"]["ready"] is False
            and expected_preflight_reasons.issubset(set(api_details["preflight"]["blocking_reasons"]))
            and "hypeusdt, riverusdt" in live_section_excerpt_lower
            and "pre_limit" in live_section_excerpt_lower
            and browser_details["submit_demo_disabled"] is True
            and browser_details["submit_live_disabled"] is True
            and "live preflight" in preflight_excerpt_lower
            and "bitget api keys" in preflight_excerpt_lower
            and "blocking reasons:" in preflight_excerpt_lower
            and "api_keys_missing" in result_excerpt_lower
        )
        print(json.dumps({"passed": passed, "details": {"api": api_details, "browser": browser_details}}, ensure_ascii=True, indent=2))
        return 0 if passed else 1
    finally:
        _shutdown_server(server)


if __name__ == "__main__":
    raise SystemExit(main())
