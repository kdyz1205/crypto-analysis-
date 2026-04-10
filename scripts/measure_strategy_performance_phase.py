import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8062
BASE_URL = f"http://{HOST}:{PORT}"
SNAPSHOT_PATH = "/api/strategy/snapshot?symbol=HYPEUSDT&interval=4h&analysis_bars=500"
STRUCTURE_PATH = "/api/market/structure-summary?symbol=HYPEUSDT&interval=4h"
PAPER_PATH = "/api/paper-execution/state"


def _start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _wait_for_server(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get(f"{BASE_URL}{PAPER_PATH}")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("Server did not become ready in time")


async def _measure_request(client: httpx.AsyncClient, path: str) -> dict:
    start = time.perf_counter()
    response = await client.get(f"{BASE_URL}{path}")
    elapsed = time.perf_counter() - start
    return {
        "path": path,
        "status": response.status_code,
        "seconds": round(elapsed, 4),
    }


def _stop_server(server: subprocess.Popen) -> None:
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()


async def _measure_cold_and_warm(path: str) -> dict:
    server = _start_server()
    try:
        await _wait_for_server()
        async with httpx.AsyncClient(timeout=120.0) as client:
            cold = await _measure_request(client, path)
            warm = await _measure_request(client, path)
            paper = await _measure_request(client, PAPER_PATH)
        return {"cold": cold, "warm": warm, "paper_only": paper}
    finally:
        _stop_server(server)


async def _measure_cold_concurrent(path: str, label: str) -> dict:
    server = _start_server()
    try:
        await _wait_for_server()
        async with httpx.AsyncClient(timeout=120.0) as client:
            paper_result, heavy_result = await asyncio.gather(
                _measure_request(client, PAPER_PATH),
                _measure_request(client, path),
            )
        return {"paper": paper_result, label: heavy_result}
    finally:
        _stop_server(server)


async def main() -> None:
    snapshot = await _measure_cold_and_warm(SNAPSHOT_PATH)
    structure = await _measure_cold_and_warm(STRUCTURE_PATH)
    cold_snapshot_concurrency = await _measure_cold_concurrent(SNAPSHOT_PATH, "snapshot")
    cold_structure_concurrency = await _measure_cold_concurrent(STRUCTURE_PATH, "structure_summary")
    print(
        json.dumps(
            {
                "snapshot": snapshot,
                "structure_summary": structure,
                "paper_with_snapshot_cold": cold_snapshot_concurrency,
                "paper_with_structure_summary_cold": cold_structure_concurrency,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
