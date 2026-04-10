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
    from selenium.webdriver.common.by import By
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
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base}/v2", timeout=2.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server did not start on port {port}")


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


@dataclass
class VerificationResult:
    passed: bool
    details: dict


def _verify_chart_modes(base_url: str) -> VerificationResult:
    driver = _build_driver()
    try:
        driver.get(f"{base_url}/v2")
        time.sleep(4.0)

        fast_button = driver.find_element(By.CSS_SELECTOR, '[data-history-mode="fast_window"]')
        full_button = driver.find_element(By.CSS_SELECTOR, '[data-history-mode="full_history"]')
        linear_button = driver.find_element(By.CSS_SELECTOR, '[data-scale-mode="linear"]')
        log_button = driver.find_element(By.CSS_SELECTOR, '[data-scale-mode="log"]')
        meta = driver.find_element(By.ID, "chart-mode-meta")
        header = driver.find_element(By.ID, "chart-header-v2")

        fast_meta = meta.text
        fast_header = header.text

        full_button.click()
        time.sleep(3.0)
        full_meta = meta.text
        full_header = header.text

        log_button.click()
        time.sleep(1.0)
        log_header = header.text

        linear_button.click()
        time.sleep(1.0)
        linear_header = header.text

        details = {
            "fastMeta": fast_meta,
            "fullMeta": full_meta,
            "fastHeader": fast_header,
            "fullHeader": full_header,
            "logHeader": log_header,
            "linearHeader": linear_header,
            "fullButtonActive": "active" in (full_button.get_attribute("class") or ""),
            "logButtonActiveAfterClick": "active" in (log_button.get_attribute("class") or ""),
        }
        passed = (
            "Fast window" in fast_meta
            and "Full history" in full_meta
            and "listing start" in full_meta
            and "/LIN" in fast_header
            and "/LIN" in full_header
            and "/LOG" in log_header
            and "/LIN" in linear_header
        )
        return VerificationResult(passed, details)
    finally:
        driver.quit()


def main() -> int:
    port = 8068
    server = _start_server(port)
    try:
        _wait_for_server(port)
        base_url = f"http://127.0.0.1:{port}"
        result = _verify_chart_modes(base_url)
    finally:
        _shutdown_server(server)

    print(json.dumps({"passed": result.passed, "details": result.details}, ensure_ascii=True, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
