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
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "selenium is required for browser verification. "
        "Install it with: python -m pip install selenium"
    ) from exc

from verify_execution_stabilization_followup import _find_browser


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8064
BASE_URL = f"http://{HOST}:{PORT}"

DEGRADED_PRELOAD = """
(() => {
  const originalFetch = window.fetch.bind(window);
  const delayedFailure = (init, label, delayMs = 8000) => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} mocked delay/failure`)), delayMs);
    if (init && init.signal && typeof init.signal.addEventListener === 'function') {
      init.signal.addEventListener('abort', () => {
        clearTimeout(timer);
        reject(new DOMException('The operation was aborted.', 'AbortError'));
      }, { once: true });
    }
  });

  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    if (url.includes('/api/agent/status')) {
      return delayedFailure(init, 'agent-status');
    }
    if (url.includes('/api/live-execution/status')) {
      return delayedFailure(init, 'live-status');
    }
    return originalFetch(input, init);
  };
})();
"""


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


def _build_driver(preload_script: str | None = None) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1600,1200")
    options.binary_location = _find_browser()
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    driver = webdriver.Chrome(options=options)
    if preload_script:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": preload_script},
        )
    return driver


def _shutdown_server(server: subprocess.Popen[str]) -> None:
    server.terminate()
    try:
        server.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        server.kill()
        server.communicate(timeout=5)


def _wait_for_text(wait: WebDriverWait, locator: tuple[str, str], needle: str) -> None:
    wait.until(lambda d: needle in d.find_element(*locator).text)


def _wait_for_paper_section_ready(wait: WebDriverWait) -> None:
    wait.until(
        lambda d: (
            "Step once" in d.find_element(By.ID, "v2-exec-paper-section").text
            and "Loading paper execution..." not in d.find_element(By.ID, "v2-exec-paper-section").text
        )
    )


def _open_execution_tab(driver: webdriver.Chrome, wait: WebDriverWait) -> None:
    driver.find_element(By.ID, "v2-exec-toggle").click()
    wait.until(lambda d: "hidden" not in (d.find_element(By.ID, "v2-execution-panel").get_attribute("class") or ""))
    driver.find_element(By.CSS_SELECTOR, '.exec-tab[data-tab="execution"]').click()
    wait.until(lambda d: d.find_element(By.ID, "v2-exec-paper-section").is_displayed())


def _collect_boot_pills(driver: webdriver.Chrome) -> list[dict[str, str]]:
    return driver.execute_script(
        """
        return Array.from(document.querySelectorAll('.boot-status .boot-pill')).map((el) => ({
          text: (el.textContent || '').trim(),
          className: el.className,
        }));
        """
    )


def _scenario_actual() -> dict:
    driver = _build_driver()
    wait = WebDriverWait(driver, 90)
    try:
        start = time.perf_counter()
        driver.get(f"{BASE_URL}/v2")
        wait.until(lambda d: d.execute_script("return document.querySelectorAll('#v2-symbol-select option').length") > 0)
        wait.until(lambda d: d.execute_script(
            "const header=document.querySelector('#chart-header-v2'); return header && header.textContent && header.textContent !== '—';"
        ))
        page_ready = round(time.perf_counter() - start, 3)

        _open_execution_tab(driver, wait)
        _wait_for_paper_section_ready(wait)
        paper_ready = round(time.perf_counter() - start, 3)

        driver.find_element(By.ID, "v2-paper-step-btn").click()
        _wait_for_text(wait, (By.ID, "v2-exec-paper-section"), "Stepping...")
        _wait_for_text(wait, (By.ID, "v2-exec-paper-section"), "Last step:")

        paper_text = driver.find_element(By.ID, "v2-exec-paper-section").text
        severe_logs = [entry for entry in driver.get_log("browser") if entry.get("level") == "SEVERE"]
        return {
            "page_ready_seconds": page_ready,
            "paper_section_ready_seconds": paper_ready,
            "paper_after_step_excerpt": paper_text[:800],
            "boot_pills": _collect_boot_pills(driver),
            "console_severe": severe_logs[:10],
        }
    finally:
        driver.quit()


def _scenario_degraded_agent_live() -> dict:
    driver = _build_driver(DEGRADED_PRELOAD)
    wait = WebDriverWait(driver, 90)
    try:
        start = time.perf_counter()
        driver.get(f"{BASE_URL}/v2")
        wait.until(lambda d: d.execute_script("return document.querySelectorAll('#v2-symbol-select option').length") > 0)
        wait.until(lambda d: d.execute_script(
            "const header=document.querySelector('#chart-header-v2'); return header && header.textContent && header.textContent !== '—';"
        ))

        _open_execution_tab(driver, wait)
        _wait_for_paper_section_ready(wait)
        paper_ready = round(time.perf_counter() - start, 3)

        agent_initial = driver.find_element(By.ID, "v2-exec-agent-section").text
        live_initial = driver.find_element(By.ID, "v2-exec-live-section").text

        wait.until(
            lambda d: (
                "unavailable" in d.find_element(By.ID, "v2-exec-agent-section").text.lower()
                and "unavailable" in d.find_element(By.ID, "v2-exec-live-section").text.lower()
            )
        )

        paper_final = driver.find_element(By.ID, "v2-exec-paper-section").text
        agent_final = driver.find_element(By.ID, "v2-exec-agent-section").text
        live_final = driver.find_element(By.ID, "v2-exec-live-section").text

        return {
            "paper_ready_seconds": paper_ready,
            "agent_initial_excerpt": agent_initial[:200],
            "live_initial_excerpt": live_initial[:200],
            "paper_final_excerpt": paper_final[:400],
            "agent_final_excerpt": agent_final[:300],
            "live_final_excerpt": live_final[:300],
        }
    finally:
        driver.quit()


def main() -> None:
    server = _start_server()
    try:
        _wait_for_server()
        results = {
            "actual": _scenario_actual(),
            "degraded_agent_live": _scenario_degraded_agent_live(),
        }
        print(json.dumps(results, indent=2))
    finally:
        _shutdown_server(server)


if __name__ == "__main__":
    main()
