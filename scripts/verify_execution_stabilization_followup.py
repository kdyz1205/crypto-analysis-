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


def _build_driver(preload_script: str) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1600,1200")
    options.binary_location = _find_browser()
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": preload_script},
    )
    return driver


def _shutdown_server(server: subprocess.Popen[str]) -> str:
    server.terminate()
    try:
        server.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - fallback path
        server.kill()
        server.communicate(timeout=5)
    return ""


def _lightweight_charts_stub() -> str:
    return """
(() => {
  window.__lineSeriesPayloads = [];
  const chartStub = {
    CrosshairMode: { Normal: 0 },
    createChart() {
      return {
        addCandlestickSeries() {
          return {
            setData() {},
            setMarkers() {},
          };
        },
        addHistogramSeries() {
          return { setData() {} };
        },
        addLineSeries() {
          return createLineSeries();
        },
        removeSeries() {},
        applyOptions() {},
        timeScale() {
          return { fitContent() {} };
        },
      };
    },
  };
  function createLineSeries() {
    return {
      setData(data) {
        window.__lineSeriesPayloads.push(data);
      },
    };
  }
  Object.defineProperty(window, 'LightweightCharts', {
    value: chartStub,
    writable: false,
    configurable: false,
  });
})();
"""


BOOT_FAILURE_PRELOAD = (
    _lightweight_charts_stub()
    + """
(() => {
  window.__eventSourceCount = 0;
  window.EventSource = class MockEventSource {
    constructor(url) {
      this.url = url;
      window.__eventSourceCount += 1;
      setTimeout(() => this.onopen && this.onopen(), 10);
    }
    addEventListener() {}
    close() {}
  };

  const jsonResponse = (payload, status = 200) =>
    Promise.resolve(new Response(JSON.stringify(payload), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }));

  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    if (url.includes('/api/symbols')) return jsonResponse(['HYPEUSDT']);
    if (url.includes('/api/ohlcv')) return Promise.reject(new Error('mock ohlcv failure'));
    if (url.includes('/api/agent/summary')) return jsonResponse({});
    if (url.includes('/api/agent/risk-state')) return jsonResponse({});
    if (url.includes('/api/agent/signal-candidates')) return jsonResponse({ count: 0, candidates: [] });
    return jsonResponse({});
  };
})();
"""
)


STALE_FIRST_LOAD_BOOT_PRELOAD = (
    _lightweight_charts_stub()
    + """
(() => {
  window.__eventSourceCount = 0;
  window.__intervalLog = [];
  window.__requestLog = [];

  const originalSetInterval = window.setInterval.bind(window);
  window.setInterval = (fn, delay, ...args) => {
    window.__intervalLog.push({
      delay,
      source: String(fn),
    });
    return originalSetInterval(fn, delay, ...args);
  };

  window.EventSource = class MockEventSource {
    constructor(url) {
      this.url = url;
      window.__eventSourceCount += 1;
      setTimeout(() => this.onopen && this.onopen(), 10);
    }
    addEventListener() {}
    close() {}
  };

  const jsonResponse = (payload, status = 200) =>
    Promise.resolve(new Response(JSON.stringify(payload), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }));

  const candlesPayload = (symbol) => ({
    candles: [
      { time: 1711929600, open: 1.0, high: 2.0, low: 0.5, close: symbol === 'AAAUSDT' ? 2.2 : 1.2 },
      { time: 1711944000, open: 1.2, high: 2.2, low: 1.0, close: symbol === 'AAAUSDT' ? 2.4 : 1.8 },
      { time: 1711958400, open: 1.8, high: 2.4, low: 1.4, close: symbol === 'AAAUSDT' ? 2.6 : 2.0 },
    ],
    volume: [],
    overlays: {},
    pricePrecision: 2,
  });

  const strategyConfig = {
    layerDefaults: {
      trendlines: true,
      touchMarkers: true,
      projectedLine: true,
      signalMarkers: true,
      invalidationMarkers: true,
      orderMarkers: false,
    },
    tickSize: 0.01,
  };

  const strategySnapshot = {
    snapshot: {
      candidate_lines: [],
      active_lines: [],
      line_states: [],
      touch_points: [],
      signals: [],
      signal_states: [],
      invalidations: [],
      orders: [],
    },
  };

  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    if (url.includes('/api/symbols')) return jsonResponse(['HYPEUSDT', 'AAAUSDT']);
    if (url.includes('/api/ohlcv')) {
      const parsed = new URL(url, window.location.origin);
      const symbol = parsed.searchParams.get('symbol');
      const delay = symbol === 'HYPEUSDT' ? 2500 : 100;
      window.__requestLog.push(`ohlcv:${symbol}`);
      return new Promise((resolve) => {
        setTimeout(() => resolve(new Response(JSON.stringify(candlesPayload(symbol)), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })), delay);
      });
    }
    if (url.includes('/api/strategy/config')) return jsonResponse(strategyConfig);
    if (url.includes('/api/strategy/snapshot')) return jsonResponse(strategySnapshot);
    if (url.includes('/api/patterns')) {
      const parsed = new URL(url, window.location.origin);
      const symbol = parsed.searchParams.get('symbol');
      window.__requestLog.push(`patterns:${symbol}`);
      return jsonResponse({
        supportLines: [{ t1: 1711929600, t2: 1711958400, v1: symbol === 'AAAUSDT' ? 111 : 99, v2: symbol === 'AAAUSDT' ? 111 : 99 }],
        resistanceLines: [],
        consolidationZones: [],
        patterns: [],
        currentTrend: 0,
        trendLabel: 'SIDEWAYS',
        trendSlope: 0,
      });
    }
    if (url.includes('/api/agent/summary')) return jsonResponse({});
    if (url.includes('/api/agent/risk-state')) return jsonResponse({});
    if (url.includes('/api/agent/signal-candidates')) return jsonResponse({ count: 0, candidates: [] });
    return jsonResponse({});
  };
})();
"""
)


STALE_OVERLAY_PRELOAD = (
    _lightweight_charts_stub()
    + """
(() => {
  window.__requestLog = [];

  const candlesPayload = (symbol) => ({
    candles: [
      { time: 1711929600, open: 1, high: 2, low: 0.5, close: 1.2 },
      { time: 1711944000, open: 1.2, high: 2.2, low: 1.0, close: 1.8 },
      { time: 1711958400, open: 1.8, high: 2.4, low: 1.4, close: 2.0 },
      { time: 1711972800, open: 2.0, high: 2.6, low: 1.8, close: symbol === 'BBBUSDT' ? 2.3 : 2.1 },
    ],
    volume: [],
    overlays: {},
    pricePrecision: 2,
  });

  const strategyConfig = {
    layerDefaults: {
      trendlines: true,
      touchMarkers: true,
      projectedLine: true,
      signalMarkers: true,
      invalidationMarkers: true,
      orderMarkers: false,
    },
    tickSize: 0.01,
  };

  const strategySnapshot = {
    snapshot: {
      candidate_lines: [],
      active_lines: [],
      line_states: [],
      touch_points: [],
      signals: [],
      signal_states: [],
      invalidations: [],
      orders: [],
    },
  };

  const delayedJson = (payload, delayMs, signal, label) => new Promise((resolve, reject) => {
    const finish = () => resolve(new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }));
    const timer = setTimeout(() => {
      window.__requestLog.push(`${label}:resolved`);
      finish();
    }, delayMs);
    if (signal) {
      signal.addEventListener('abort', () => {
        clearTimeout(timer);
        window.__requestLog.push(`${label}:aborted`);
        reject(new DOMException('Aborted', 'AbortError'));
      }, { once: true });
    }
  });

  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    if (url.includes('/api/symbols')) {
      return Promise.resolve(new Response(JSON.stringify(['HYPEUSDT', 'AAAUSDT', 'BBBUSDT']), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }));
    }
    if (url.includes('/api/ohlcv')) {
      const parsed = new URL(url, window.location.origin);
      const symbol = parsed.searchParams.get('symbol');
      window.__requestLog.push(`ohlcv:${symbol}`);
      return Promise.resolve(new Response(JSON.stringify(candlesPayload(symbol)), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }));
    }
    if (url.includes('/api/strategy/config')) {
      return Promise.resolve(new Response(JSON.stringify(strategyConfig), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }));
    }
    if (url.includes('/api/strategy/snapshot')) {
      const parsed = new URL(url, window.location.origin);
      const symbol = parsed.searchParams.get('symbol');
      window.__requestLog.push(`strategy:${symbol}`);
      return Promise.resolve(new Response(JSON.stringify(strategySnapshot), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }));
    }
    if (url.includes('/api/patterns')) {
      const parsed = new URL(url, window.location.origin);
      const symbol = parsed.searchParams.get('symbol');
      const value = symbol === 'AAAUSDT' ? 111 : 222;
      const delay = symbol === 'AAAUSDT' ? 2500 : 100;
      window.__requestLog.push(`patterns:${symbol}:scheduled`);
      return delayedJson({
        supportLines: [{ t1: 1711929600, t2: 1711972800, v1: value, v2: value }],
        resistanceLines: [],
        consolidationZones: [],
        patterns: [],
        currentTrend: 0,
        trendLabel: 'SIDEWAYS',
        trendSlope: 0,
      }, delay, init.signal, `patterns:${symbol}`);
    }
    if (url.includes('/api/agent/summary')) return Promise.resolve(new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    if (url.includes('/api/agent/risk-state')) return Promise.resolve(new Response(JSON.stringify({}), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    if (url.includes('/api/agent/signal-candidates')) return Promise.resolve(new Response(JSON.stringify({ count: 0, candidates: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    return Promise.resolve(new Response(JSON.stringify({}), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }));
  };
})();
"""
)


@dataclass
class VerificationResult:
    name: str
    passed: bool
    details: dict


def verify_boot_failure(base_url: str) -> VerificationResult:
    driver = _build_driver(BOOT_FAILURE_PRELOAD)
    try:
      driver.get(f"{base_url}/v2")
      time.sleep(4)
      details = driver.execute_script(
          """
          const pills = document.querySelectorAll('#v2-boot-status .boot-pill');
          return {
            chartText: pills[1]?.innerText || '',
            chartTitle: pills[1]?.title || '',
            chartClass: pills[1]?.className || '',
            patternsText: pills[2]?.innerText || '',
            patternsTitle: pills[2]?.title || '',
            patternsClass: pills[2]?.className || '',
            streamText: pills[4]?.innerText || '',
            streamTitle: pills[4]?.title || '',
            streamClass: pills[4]?.className || '',
            eventSourceCount: window.__eventSourceCount || 0,
          };
          """
      )
    finally:
      driver.quit()

    passed = (
        "boot-error" in details["chartClass"]
        and "boot-error" in details["patternsClass"]
        and details["eventSourceCount"] >= 1
    )
    return VerificationResult("boot_failure", passed, details)


def verify_stale_first_load_boot_recovery(base_url: str) -> VerificationResult:
    driver = _build_driver(STALE_FIRST_LOAD_BOOT_PRELOAD)
    try:
      driver.get(f"{base_url}/v2")
      time.sleep(1.0)
      driver.execute_script(
          """
          const select = document.getElementById('v2-symbol-select');
          select.value = 'AAAUSDT';
          select.dispatchEvent(new Event('change', { bubbles: true }));
          """
      )
      time.sleep(13)
      details = driver.execute_script(
          """
          const pills = document.querySelectorAll('#v2-boot-status .boot-pill');
          const intervalLog = window.__intervalLog || [];
          return {
            chartClass: pills[1]?.className || '',
            chartTitle: pills[1]?.title || '',
            patternsClass: pills[2]?.className || '',
            patternsTitle: pills[2]?.title || '',
            streamClass: pills[4]?.className || '',
            streamTitle: pills[4]?.title || '',
            eventSourceCount: window.__eventSourceCount || 0,
            liveUpdateIntervalCount: intervalLog.filter((entry) => entry.source.includes('loadCurrent')).length,
            intervalLog,
            requestLog: window.__requestLog || [],
          };
          """
      )
    finally:
      driver.quit()

    request_log = details["requestLog"]
    passed = (
        "boot-ok" in details["chartClass"]
        and "boot-ok" in details["patternsClass"]
        and details["eventSourceCount"] == 1
        and details["liveUpdateIntervalCount"] == 1
        and any(entry == "ohlcv:HYPEUSDT" for entry in request_log)
        and any(entry == "ohlcv:AAAUSDT" for entry in request_log)
    )
    return VerificationResult("stale_first_load_boot_recovery", passed, details)


def verify_stale_overlay_guard(base_url: str) -> VerificationResult:
    driver = _build_driver(STALE_OVERLAY_PRELOAD)
    try:
      driver.get(f"{base_url}/v2")
      time.sleep(1.5)
      driver.execute_script(
          """
          const select = document.getElementById('v2-symbol-select');
          select.value = 'AAAUSDT';
          select.dispatchEvent(new Event('change', { bubbles: true }));
          setTimeout(() => {
            select.value = 'BBBUSDT';
            select.dispatchEvent(new Event('change', { bubbles: true }));
          }, 2300);
          """
      )
      time.sleep(7)
      details = driver.execute_script(
          """
          const lastSeries = window.__lineSeriesPayloads[window.__lineSeriesPayloads.length - 1] || [];
          return {
            requestLog: window.__requestLog || [],
            lineSeriesPayloads: window.__lineSeriesPayloads || [],
            lastPatternValue: lastSeries[0]?.value ?? null,
            patternBootTitle: document.querySelectorAll('#v2-boot-status .boot-pill')[2]?.title || '',
          };
          """
      )
    finally:
      driver.quit()

    request_log = details["requestLog"]
    passed = (
        details["lastPatternValue"] == 222
        and any(entry == "patterns:AAAUSDT:aborted" for entry in request_log)
        and any(entry == "patterns:BBBUSDT:resolved" for entry in request_log)
    )
    return VerificationResult("stale_overlay_guard", passed, details)


def main() -> int:
    port = 8062
    server = _start_server(port)
    try:
        _wait_for_server(port)
        base_url = f"http://127.0.0.1:{port}"
        results = [
            verify_boot_failure(base_url),
            verify_stale_first_load_boot_recovery(base_url),
            verify_stale_overlay_guard(base_url),
        ]
    finally:
        server_log = _shutdown_server(server)

    print(json.dumps(
        {
            "results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "details": result.details,
                }
                for result in results
            ],
            "server_log_tail": server_log.splitlines()[-40:] if server_log else [],
        },
        ensure_ascii=True,
        indent=2,
    ))

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
