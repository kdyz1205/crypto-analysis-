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
DRAWINGS_STORE_PATH = REPO_ROOT / "data" / "manual_trendlines.json"
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


def _seed_manual_line(base_url: str) -> str:
    httpx.post(
        f"{base_url}/api/drawings/clear",
        params={"symbol": "HYPEUSDT", "timeframe": "4h"},
        timeout=20.0,
    ).raise_for_status()
    payload = {
        "symbol": "HYPEUSDT",
        "timeframe": "4h",
        "side": "resistance",
        "t_start": 1712592000,
        "t_end": 1712678400,
        "price_start": 100.0,
        "price_end": 105.0,
        "label": "browser manual line",
        "notes": "seeded",
        "override_mode": "display_only",
    }
    response = httpx.post(f"{base_url}/api/drawings", json=payload, timeout=20.0)
    response.raise_for_status()
    return response.json()["drawing"]["manual_line_id"]


@dataclass
class VerificationResult:
    passed: bool
    details: dict


def _verify_manual_ui(base_url: str, manual_line_id: str) -> VerificationResult:
    driver = _build_driver()
    try:
        driver.get(f"{base_url}/v2")
        deadline = time.time() + 15.0
        while time.time() < deadline:
            rows = driver.find_elements(By.CSS_SELECTOR, ".manual-line-row")
            if rows:
                break
            time.sleep(0.25)
        rows = driver.find_elements(By.CSS_SELECTOR, ".manual-line-row")
        if not rows:
            return VerificationResult(False, {"reason": "manual line rows not visible"})

        _wait_for_manual_panel_ready(driver)
        rows[0].click()
        _wait_for_selected_inputs(driver)

        driver.execute_script(
            """
            const label = document.getElementById('manual-line-label-input');
            const notes = document.getElementById('manual-line-notes-input');
            if (label) {
              label.value = 'verified manual line';
              label.dispatchEvent(new Event('input', { bubbles: true }));
              label.dispatchEvent(new Event('change', { bubbles: true }));
            }
            if (notes) {
              notes.value = 'manual note from browser verify';
              notes.dispatchEvent(new Event('input', { bubbles: true }));
              notes.dispatchEvent(new Event('change', { bubbles: true }));
            }
            """,
        )

        driver.execute_script(
            """
            const button = document.querySelector('[data-action="save-metadata"]');
            if (button) button.click();
            """,
        )

        current = _wait_for_saved_drawing(
            base_url,
            manual_line_id,
            expected_label="verified manual line",
            expected_notes="manual note from browser verify",
        )

        details = {
            "rowCount": len(rows),
            "hasDrawResistance": bool(driver.find_elements(By.CSS_SELECTOR, '[data-action="draw-resistance"]')),
            "hasDrawSupport": bool(driver.find_elements(By.CSS_SELECTOR, '[data-action="draw-support"]')),
            "hasToggleExtendLeft": bool(driver.find_elements(By.CSS_SELECTOR, '[data-action="toggle-extend-left"]')),
            "hasToggleExtendRight": bool(driver.find_elements(By.CSS_SELECTOR, '[data-action="toggle-extend-right"]')),
            "hasOverrideMode": bool(driver.find_elements(By.CSS_SELECTOR, 'select[data-action="override-mode"]')),
            "savedLabel": current["label"] if current else None,
            "savedNotes": current["notes"] if current else None,
        }
        passed = (
            details["rowCount"] >= 1
            and details["hasDrawResistance"]
            and details["hasDrawSupport"]
            and details["hasToggleExtendLeft"]
            and details["hasToggleExtendRight"]
            and details["hasOverrideMode"]
            and details["savedLabel"] == "verified manual line"
            and details["savedNotes"] == "manual note from browser verify"
        )
        return VerificationResult(passed, details)
    finally:
        driver.quit()


def _wait_for_selected_inputs(driver: webdriver.Chrome, timeout_seconds: float = 15.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        label = driver.find_elements(By.ID, "manual-line-label-input")
        notes = driver.find_elements(By.ID, "manual-line-notes-input")
        save = driver.find_elements(By.CSS_SELECTOR, '[data-action="save-metadata"]')
        if label and notes and save:
            return
        time.sleep(0.25)
    raise RuntimeError("Selected manual line editor did not appear")


def _wait_for_manual_panel_ready(driver: webdriver.Chrome, timeout_seconds: float = 15.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        note = driver.find_elements(By.CSS_SELECTOR, ".manual-panel-note")
        if note and "Loading manual lines..." not in (note[0].text or ""):
            return
        time.sleep(0.25)
    raise RuntimeError("Manual panel did not leave loading state")


def _wait_for_saved_drawing(
    base_url: str,
    manual_line_id: str,
    *,
    expected_label: str,
    expected_notes: str,
    timeout_seconds: float = 60.0,
) -> dict | None:
    deadline = time.time() + timeout_seconds
    latest = None
    while time.time() < deadline:
        latest = _read_saved_drawing_from_store(manual_line_id) or latest
        if latest and latest.get("label") == expected_label and latest.get("notes") == expected_notes:
            return latest
        try:
            list_response = httpx.get(
                f"{base_url}/api/drawings",
                params={"symbol": "HYPEUSDT", "timeframe": "4h"},
                timeout=20.0,
            )
            list_response.raise_for_status()
            drawings = list_response.json()["drawings"]
            latest = next((item for item in drawings if item["manual_line_id"] == manual_line_id), None)
            if latest and latest.get("label") == expected_label and latest.get("notes") == expected_notes:
                return latest
        except Exception:
            pass
        time.sleep(0.5)
    return latest


def _read_saved_drawing_from_store(manual_line_id: str) -> dict | None:
    if not DRAWINGS_STORE_PATH.exists():
        return None
    try:
        payload = json.loads(DRAWINGS_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    return next(
        (
            item
            for item in payload
            if isinstance(item, dict) and item.get("manual_line_id") == manual_line_id
        ),
        None,
    )


def main() -> int:
    port = 8066
    server = _start_server(port)
    try:
        _wait_for_server(port)
        base_url = f"http://127.0.0.1:{port}"
        manual_line_id = _seed_manual_line(base_url)
        result = _verify_manual_ui(base_url, manual_line_id)
    finally:
        _shutdown_server(server)

    print(json.dumps({"passed": result.passed, "details": result.details}, ensure_ascii=True, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
