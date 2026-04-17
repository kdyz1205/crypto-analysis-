"""Vercel serverless entry point.

Vercel installs root requirements.txt and bundles reachable imports into one
Python function. The full trading backend imports data-science packages that
are too large for Vercel's 500 MB function limit, so Vercel deploys expose a
lightweight API shell and static UI. Run live trading on the uvicorn backend.
"""
import os
import sys

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Add project root to path so server package is importable outside Vercel.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Crypto Analysis Vercel API")


@app.get("/api/health")
async def api_health():
    return {
        "ok": True,
        "runtime": "vercel-lightweight",
        "full_backend": "external",
    }


@app.get("/api/live-execution/status")
async def api_live_execution_status():
    return {
        "ok": False,
        "runtime": "vercel-lightweight",
        "reason": "live trading backend is not hosted on Vercel",
    }


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def api_not_available(path: str):
    return JSONResponse(
        status_code=503,
        content={
            "ok": False,
            "runtime": "vercel-lightweight",
            "path": f"/api/{path}",
            "reason": "full trading backend requires the non-serverless uvicorn process",
        },
    )


if os.environ.get("VERCEL") != "1":
    from server.app import app  # noqa: E402
