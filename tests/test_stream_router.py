from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/api/health", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise AssertionError("server did not become healthy in time")


def test_sse_stream_does_not_block_regular_api_requests():
    repo_root = Path(__file__).resolve().parents[1]
    port = _get_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(base_url)

        result: dict[str, object] = {}
        done = threading.Event()

        def _open_stream() -> None:
            try:
                with requests.get(f"{base_url}/api/stream", stream=True, timeout=(5, 5)) as response:
                    result["status"] = response.status_code
                    iterator = response.iter_lines(decode_unicode=True)
                    result["first_line"] = next(iterator)
                    result["second_line"] = next(iterator)
                    done.wait(2)
            except Exception as exc:  # pragma: no cover - surfaced via assertions below
                result["error"] = repr(exc)

        thread = threading.Thread(target=_open_stream, daemon=True)
        thread.start()
        time.sleep(1.5)

        health_response = requests.get(f"{base_url}/api/health", timeout=5)
        assert health_response.status_code == 200
        assert "Server is running" in health_response.text

        thread.join(timeout=8)
        done.set()

        assert "error" not in result, result.get("error")
        assert result.get("status") == 200
        assert result.get("first_line") == "event: connected"
        assert str(result.get("second_line", "")).startswith("data:")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
