"""
Ops routes: health, static files, telegram config, agent logs, healer lifecycle.
"""

import asyncio

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse

from ..core.config import PROJECT_ROOT, FRONTEND_DIR
from ..core.log_buffer import _LOG_BUFFER
from ..core.dependencies import get_agent, get_healer

router = APIRouter(tags=["ops"])


# ── Health & Static ──────────────────────────────────────────────────────

@router.get("/api/health")
async def health_check():
    """Health check endpoint to verify server is running."""
    return {"status": "ok", "message": "Server is running"}


@router.get("/")
async def index():
    return FileResponse(
        str(FRONTEND_DIR / "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/style.css")
async def serve_css():
    return FileResponse(
        str(FRONTEND_DIR / "style.css"),
        media_type="text/css",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


@router.get("/app.js")
async def serve_js():
    return FileResponse(
        str(FRONTEND_DIR / "app.js"),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


# ── Telegram Config ──────────────────────────────────────────────────────

@router.post("/api/agent/telegram-config")
async def api_agent_telegram_config(request: Request):
    """Save Telegram notification config and send a test message."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "Invalid JSON"}

    bot_token = body.get("bot_token", "").strip()
    chat_id = body.get("chat_id", "").strip()
    if not bot_token or not chat_id:
        return {"ok": False, "reason": "bot_token and chat_id required"}

    # Store config in agent state
    agent = get_agent()
    agent._telegram_config = {
        "bot_token": bot_token,
        "chat_id": chat_id,
        "notify_signals": body.get("notify_signals", True),
        "notify_fills": body.get("notify_fills", True),
        "notify_errors": body.get("notify_errors", False),
        "notify_daily": body.get("notify_daily", False),
    }

    # Send test message
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": "✅ Crypto Agent connected! Notifications enabled.", "parse_mode": "HTML"},
            )
            if resp.status_code == 200:
                print(f"[Telegram] Config saved, test message sent to {chat_id}")
                return {"ok": True}
            else:
                try:
                    err = resp.json().get("description", resp.text)
                except Exception:
                    err = resp.text
                return {"ok": False, "reason": f"Telegram API error: {err}"}
    except Exception as e:
        return {"ok": False, "reason": f"Network error: {e}"}


# ── Agent Logs ───────────────────────────────────────────────────────────

@router.get("/api/agent/logs")
async def api_agent_logs(
    limit: int = Query(50, ge=1, le=200),
    filter: str = Query("agent", description="Filter: 'agent' for agent-only, 'all' for everything"),
):
    """Get recent agent logs from the ring buffer."""
    all_logs = list(_LOG_BUFFER)
    if filter == "agent":
        # Only show agent/trader/strategy relevant logs, not HTTP access logs
        keywords = ["[Agent]", "[OKX]", "[Healer]", "[Data]", "V6", "Layer",
                     "signal", "position", "trade", "Evolution", "SL:", "TP:"]
        all_logs = [l for l in all_logs if any(kw in l.get("msg", "") for kw in keywords)]
    logs = all_logs[-limit:]
    return {"logs": logs}


# ── Self-Healer ──────────────────────────────────────────────────────────

@router.get("/api/healer/status")
async def api_healer_status():
    """Get self-healer status: fix count, recent errors, last fix time."""
    return get_healer().get_status()


@router.post("/api/healer/trigger")
async def api_healer_trigger():
    """Manually trigger a heal attempt right now."""
    asyncio.create_task(get_healer().try_heal())
    return {"ok": True, "message": "Heal attempt triggered"}


@router.post("/api/healer/stop")
async def api_healer_stop():
    get_healer().stop()
    return {"ok": True, "message": "Healer stopped"}


@router.post("/api/healer/start")
async def api_healer_start():
    get_healer().start()
    return {"ok": True, "message": "Healer started"}
