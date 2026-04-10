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
PORT = 8067
BASE_URL = f"http://{HOST}:{PORT}"
REPLAY_TAIL_PATH = "/api/strategy/replay?symbol=HYPEUSDT&interval=4h&analysis_bars=500&tail=2"
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


async def _measure_tail_replay() -> dict:
    server = _start_server()
    try:
        await _wait_for_server()
        async with httpx.AsyncClient(timeout=120.0) as client:
            cold = await _measure_request(client, REPLAY_TAIL_PATH)
            warm = await _measure_request(client, REPLAY_TAIL_PATH)
            paper = await _measure_request(client, PAPER_PATH)
        return {"cold": cold, "warm": warm, "paper_only": paper}
    finally:
        _stop_server(server)


async def _measure_tail_replay_concurrency() -> dict:
    server = _start_server()
    try:
        await _wait_for_server()
        async with httpx.AsyncClient(timeout=120.0) as client:
            paper_result, replay_result = await asyncio.gather(
                _measure_request(client, PAPER_PATH),
                _measure_request(client, REPLAY_TAIL_PATH),
            )
        return {"paper": paper_result, "replay_tail": replay_result}
    finally:
        _stop_server(server)


async def main() -> None:
    replay = await _measure_tail_replay()
    replay_concurrency = await _measure_tail_replay_concurrency()
    print(
        json.dumps(
            {
                "replay_tail": replay,
                "paper_with_replay_tail_cold": replay_concurrency,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
