from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import httpx

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select, WebDriverWait
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "selenium is required for data completeness verification. "
        "Install it with: python -m pip install selenium"
    ) from exc

from verify_execution_stabilization_followup import _find_browser


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8068
BASE_URL = f"http://{HOST}:{PORT}"
VERIFY_LABEL = "History Coverage Verify"


def _start_server() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", HOST, "--port", str(PORT)],
        cwd=REPO_ROOT,
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
        except Exception as exc:  # pragma: no cover
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Server did not start on {BASE_URL}: {last_error}")


def _shutdown_server(server: subprocess.Popen[str]) -> None:
    server.terminate()
    try:
        server.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        server.kill()
        server.communicate(timeout=5)


def _build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1600,1200")
    options.binary_location = _find_browser()
    return webdriver.Chrome(options=options)


def _api_verify_snapshot() -> dict:
    response = httpx.get(
        f"{BASE_URL}/api/strategy/snapshot",
        params={
            "symbol": "HYPEUSDT",
            "interval": "4h",
            "history_mode": "full_history",
            "analysis_bars": 200,
            "days": 365,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    payload = response.json()
    history = payload["history"]
    assert history["historyMode"] == "full_history"
    assert history["analysisInputBarCount"] <= history["loadedBarCount"]
    return {
        "historyMode": history["historyMode"],
        "loadedBarCount": history["loadedBarCount"],
        "analysisInputBarCount": history["analysisInputBarCount"],
        "analysisWasTrimmed": history["analysisWasTrimmed"],
        "dataSourceMode": history.get("dataSourceMode"),
        "dataSourceKind": history.get("dataSourceKind"),
    }


def _api_verify_paper_step() -> dict:
    httpx.post(f"{BASE_URL}/api/paper-execution/reset", json={}, timeout=20.0).raise_for_status()
    response = httpx.post(
        f"{BASE_URL}/api/paper-execution/step",
        json={
            "symbol": "HYPEUSDT",
            "interval": "4h",
            "history_mode": "full_history",
            "days": 365,
            "analysis_bars": 200,
            "trigger_modes": ["pre_limit"],
            "lookback_bars": 80,
            "strategy_window_bars": 100,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    payload = response.json()
    history = payload["history"]
    assert history["historyMode"] == "full_history"
    assert history["analysisInputBarCount"] <= history["loadedBarCount"]
    return {
        "stream": payload["stream"],
        "lastProcessedBar": payload["lastProcessedBar"],
        "history": history,
    }


def _clear_existing_runtime_instances() -> None:
    response = httpx.get(f"{BASE_URL}/api/runtime/instances", timeout=20.0)
    response.raise_for_status()
    for record in response.json().get("instances") or []:
        config = record.get("config") or {}
        if config.get("label") != VERIFY_LABEL:
            continue
        instance_id = config.get("instance_id")
        if not instance_id:
            continue
        try:
            httpx.delete(f"{BASE_URL}/api/runtime/instances/{instance_id}", timeout=20.0).raise_for_status()
        except Exception:
            pass


def _api_verify_runtime() -> tuple[str, dict]:
    _clear_existing_runtime_instances()
    created = httpx.post(
        f"{BASE_URL}/api/runtime/instances",
        json={
            "label": VERIFY_LABEL,
            "symbol": "HYPEUSDT",
            "timeframe": "4h",
            "history_mode": "full_history",
            "analysis_bars": 200,
            "days": 365,
            "live_mode": "disabled",
            "strategy_config": {
                "enabled_trigger_modes": ["pre_limit"],
                "lookback_bars": 80,
                "window_bars": 100,
            },
        },
        timeout=20.0,
    )
    created.raise_for_status()
    instance = created.json()["instance"]
    instance_id = instance["config"]["instance_id"]

    ticked = httpx.post(f"{BASE_URL}/api/runtime/instances/{instance_id}/tick", json={}, timeout=120.0)
    ticked.raise_for_status()
    tick_payload = ticked.json()["instance"]
    history = tick_payload["status"]["last_history"]
    assert history["historyMode"] == "full_history"
    assert history["analysisInputBarCount"] <= history["loadedBarCount"]
    return instance_id, {
        "instanceId": instance_id,
        "history": history,
    }


def _browser_verify(instance_label: str) -> dict:
    driver = _build_driver()
    wait = WebDriverWait(driver, 120)
    try:
        driver.get(f"{BASE_URL}/v2")
        wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#v2-symbol-select option"))

        driver.find_element(By.ID, "v2-exec-toggle").click()
        wait.until(lambda d: "hidden" not in (d.find_element(By.ID, "v2-execution-panel").get_attribute("class") or ""))
        driver.find_element(By.CSS_SELECTOR, '.exec-tab[data-tab="execution"]').click()
        wait.until(lambda d: d.find_element(By.ID, "v2-exec-paper-section").is_displayed())

        paper_form = driver.find_element(By.ID, "v2-paper-step-form")
        symbol_input = paper_form.find_element(By.NAME, "symbol")
        symbol_input.clear()
        symbol_input.send_keys("HYPEUSDT")
        interval_input = paper_form.find_element(By.NAME, "interval")
        interval_input.clear()
        interval_input.send_keys("4h")
        Select(paper_form.find_element(By.NAME, "history_mode")).select_by_value("full_history")
        days_input = paper_form.find_element(By.NAME, "days")
        days_input.clear()
        days_input.send_keys("365")
        analysis_input = paper_form.find_element(By.NAME, "analysis_bars")
        analysis_input.clear()
        analysis_input.send_keys("200")

        driver.find_element(By.ID, "v2-paper-step-btn").click()
        wait.until(lambda d: "Last step history coverage:" in d.find_element(By.ID, "v2-exec-paper-section").text)
        paper_text = driver.find_element(By.ID, "v2-exec-paper-section").text

        wait.until(lambda d: instance_label in d.find_element(By.ID, "v2-exec-runtime-section").text)
        wait.until(lambda d: "Last tick history coverage:" in d.find_element(By.ID, "v2-exec-runtime-section").text)
        runtime_text = driver.find_element(By.ID, "v2-exec-runtime-section").text

        passed = (
            "History Mode" in paper_text
            and "Last step history coverage:" in paper_text
            and "mode full_history" in paper_text
            and instance_label in runtime_text
            and "Last tick history coverage:" in runtime_text
            and "full history / 200 bars / 365d" in runtime_text.lower()
        )
        return {
            "passed": passed,
            "paperExcerpt": paper_text[:1000],
            "runtimeExcerpt": runtime_text[:1000],
        }
    finally:
        driver.quit()


def main() -> int:
    server = _start_server()
    runtime_instance_id = None
    try:
        _wait_for_server()
        snapshot = _api_verify_snapshot()
        paper = _api_verify_paper_step()
        runtime_instance_id, runtime = _api_verify_runtime()
        browser = _browser_verify(VERIFY_LABEL)
        passed = browser["passed"]
        result = {
            "passed": passed,
            "snapshot": snapshot,
            "paper": {
                "stream": paper["stream"],
                "lastProcessedBar": paper["lastProcessedBar"],
                "history": paper["history"],
            },
            "runtime": runtime,
            "browser": browser,
        }
    finally:
        if runtime_instance_id:
            try:
                httpx.delete(f"{BASE_URL}/api/runtime/instances/{runtime_instance_id}", timeout=20.0).raise_for_status()
            except Exception:
                pass
        _shutdown_server(server)

    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
