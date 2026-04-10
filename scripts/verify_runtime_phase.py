from __future__ import annotations

import json
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
except ImportError as exc:  # pragma: no cover - environment dependency
    raise SystemExit(
        "selenium is required for this verification script. "
        "Install it with: python -m pip install selenium"
    ) from exc


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHROME_CANDIDATES = (
    pathlib.Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    pathlib.Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    pathlib.Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
)


def _find_browser() -> str:
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    browser = shutil.which("chrome") or shutil.which("msedge")
    if browser:
        return browser
    raise SystemExit("Could not find Chrome or Edge for selenium verification.")


def _start_server(port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_server(port: int, timeout_seconds: float = 45.0) -> None:
    deadline = time.time() + timeout_seconds
    base = f"http://127.0.0.1:{port}"
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base}/v2", timeout=2.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # pragma: no cover - retry path
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Server did not start on port {port}: {last_error}")


def _build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1600,1200")
    options.binary_location = _find_browser()
    return webdriver.Chrome(options=options)


def _wait_for_condition(driver: webdriver.Chrome, script: str, timeout_seconds: float = 10.0, poll_seconds: float = 0.25):
    deadline = time.time() + timeout_seconds
    last_value = None
    while time.time() < deadline:
        last_value = driver.execute_script(script)
        if last_value:
            return last_value
        time.sleep(poll_seconds)
    return last_value


def _shutdown_server(server: subprocess.Popen[str]) -> None:
    server.terminate()
    try:
        server.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - fallback path
        server.kill()
        server.communicate(timeout=5)


def _create_runtime_instance(base_url: str) -> str:
    _clear_existing_runtime_instances(base_url, label="Browser Verify Runtime")
    payload = {
        "label": "Browser Verify Runtime",
        "symbol": "HYPEUSDT",
        "timeframe": "4h",
        "subaccount_label": "verify-subaccount",
        "history_mode": "fast_window",
        "analysis_bars": 500,
        "days": 365,
        "tick_interval_seconds": 60,
        "auto_restart_on_boot": False,
        "live_mode": "disabled",
        "auto_live_preview": True,
        "auto_live_submit": False,
        "notes": "browser verification",
    }
    response = httpx.post(f"{base_url}/api/runtime/instances", json=payload, timeout=20.0)
    response.raise_for_status()
    instance = response.json()["instance"]
    instance_id = instance["config"]["instance_id"]

    tick = httpx.post(f"{base_url}/api/runtime/instances/{instance_id}/tick", json={}, timeout=20.0)
    tick.raise_for_status()
    _wait_for_instance_record(base_url, instance_id)
    return instance_id


def _delete_runtime_instance(base_url: str, instance_id: str) -> None:
    response = httpx.delete(f"{base_url}/api/runtime/instances/{instance_id}", timeout=20.0)
    response.raise_for_status()


@dataclass
class VerificationResult:
    passed: bool
    details: dict


def _verify_runtime_ui(base_url: str, instance_id: str) -> VerificationResult:
    driver = _build_driver()
    try:
        driver.get(f"{base_url}/v2")
        time.sleep(2.0)
        driver.execute_script(
            """
            document.getElementById('v2-exec-toggle')?.click();
            """
        )
        time.sleep(0.5)
        driver.execute_script(
            """
            document.querySelector('.exec-tab[data-tab="execution"]')?.click();
            """
        )
        execution_details = _wait_for_condition(
            driver,
            """
            const runtimeSection = document.getElementById('v2-exec-runtime-section');
            if (!runtimeSection || !runtimeSection.innerText.includes('Browser Verify Runtime')) return null;
            return {
              runtimeText: runtimeSection?.innerText || '',
              hasCreateForm: !!document.getElementById('v2-runtime-create-form'),
              hasTickButton: !!document.querySelector('[data-runtime-action="tick"]'),
              hasStartButton: !!document.querySelector('[data-runtime-action="start"]'),
              hasKillButton: !!document.querySelector('[data-runtime-action="kill"]'),
            };
            """,
            timeout_seconds=30.0,
        )
        if execution_details is None:
            driver.execute_script(
                """
                document.querySelector('.exec-tab[data-tab="execution"]')?.click();
                """
            )
            time.sleep(1.0)
            execution_details = driver.execute_script(
                """
                const runtimeSection = document.getElementById('v2-exec-runtime-section');
                return {
                  runtimeText: runtimeSection?.innerText || '',
                  hasCreateForm: !!document.getElementById('v2-runtime-create-form'),
                  hasTickButton: !!document.querySelector('[data-runtime-action="tick"]'),
                  hasStartButton: !!document.querySelector('[data-runtime-action="start"]'),
                  hasKillButton: !!document.querySelector('[data-runtime-action="kill"]'),
                };
                """
            )
        driver.execute_script(
            """
            document.querySelector('.exec-tab[data-tab="ops"]')?.click();
            """
        )
        ops_details = _wait_for_condition(
            driver,
            """
            const subtab = document.querySelector('[data-subtab="ops"]');
            if (!subtab || !subtab.innerText.includes('Subaccount Runtime Events')) return null;
            return {
              opsText: subtab?.innerText || '',
            };
            """,
            timeout_seconds=12.0,
        )
        if ops_details is None:
            ops_details = driver.execute_script(
                """
                const subtab = document.querySelector('[data-subtab="ops"]');
                return {
                  opsText: subtab?.innerText || '',
                };
                """
            )
    finally:
        driver.quit()

    events_response = httpx.get(f"{base_url}/api/runtime/events?instance_id={instance_id}&limit=10", timeout=20.0)
    events_response.raise_for_status()
    event_types = [event["event_type"] for event in events_response.json()["events"]]

    details = {
        **execution_details,
        **ops_details,
        "eventTypes": event_types,
    }
    runtime_text = str(details["runtimeText"] or "")
    ops_text = str(details["opsText"] or "")
    passed = (
        details["runtimeText"]
        and "subaccount runtime" in runtime_text.lower()
        and "Browser Verify Runtime" in details["runtimeText"]
        and details["hasCreateForm"]
        and details["hasTickButton"]
        and details["hasStartButton"]
        and details["hasKillButton"]
        and details["opsText"]
        and "subaccount runtime events" in ops_text.lower()
        and any(event_type == "instance.created" for event_type in event_types)
        and any(event_type == "instance.ticked" for event_type in event_types)
    )
    return VerificationResult(passed=passed, details=details)


def main() -> int:
    port = 8065
    server = _start_server(port)
    instance_id = None
    try:
        _wait_for_server(port)
        base_url = f"http://127.0.0.1:{port}"
        instance_id = _create_runtime_instance(base_url)
        result = _verify_runtime_ui(base_url, instance_id)
    finally:
        if instance_id:
            try:
                _delete_runtime_instance(base_url, instance_id)
            except Exception:
                pass
        _shutdown_server(server)

    print(json.dumps({"passed": result.passed, "details": result.details}, ensure_ascii=True, indent=2))
    return 0 if result.passed else 1


def _wait_for_instance_record(base_url: str, instance_id: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = httpx.get(f"{base_url}/api/runtime/instances", timeout=10.0)
        response.raise_for_status()
        records = response.json().get("instances") or []
        if any(record.get("config", {}).get("instance_id") == instance_id for record in records):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Runtime instance {instance_id} did not become visible via API")


def _clear_existing_runtime_instances(base_url: str, *, label: str) -> None:
    response = httpx.get(f"{base_url}/api/runtime/instances", timeout=10.0)
    response.raise_for_status()
    records = response.json().get("instances") or []
    for record in records:
        config = record.get("config") or {}
        if config.get("label") != label:
            continue
        instance_id = config.get("instance_id")
        if not instance_id:
            continue
        try:
            httpx.delete(f"{base_url}/api/runtime/instances/{instance_id}", timeout=10.0).raise_for_status()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
