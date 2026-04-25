"""
Ops routes: health, static files, telegram config, agent logs, healer lifecycle.
"""

import asyncio

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

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
    """Legacy v1 UI was deleted 2026-04-19. Root now redirects to v2 so
    old bookmarks and run.py launches still land on a working page."""
    return RedirectResponse(url="/v2", status_code=302)


@router.get("/v2")
async def index_v2():
    """Main Trading OS UI (layered v2 app)."""
    return FileResponse(
        str(FRONTEND_DIR / "v2.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/v2.html")
async def index_v2_html():
    return FileResponse(
        str(FRONTEND_DIR / "v2.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/sw.js")
async def serve_service_worker():
    """2026-04-23: Service Worker for static + OHLCV stale-while-revalidate.
    MUST be served at ROOT (/sw.js) for its scope to cover the whole site.
    Browser cache disabled so SW upgrades don't get stuck."""
    return FileResponse(
        str(FRONTEND_DIR / "sw.js"),
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Service-Worker-Allowed": "/",
        },
    )


@router.get("/agents")
async def index_agents():
    """Live Claude x Codex dialogue viewer (pixel-style chat bubbles).

    Reads real task exchanges from data/logs/agent_dialogue.jsonl.
    """
    return FileResponse(
        str(FRONTEND_DIR / "agent_dialogue.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/api/agent_dialogue")
async def api_agent_dialogue():
    """Serve the agent dialogue JSONL as plain text so the viewer can
    parse it line-by-line. Each line is one turn."""
    from pathlib import Path as _P
    p = _P(__file__).resolve().parents[2] / "data" / "logs" / "agent_dialogue.jsonl"
    if not p.exists():
        return JSONResponse({"ok": False, "reason": "no dialogue log yet"}, status_code=404)
    return FileResponse(
        str(p),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


def _conditional_file_response(path, media_type: str, request: Request):
    """Shared helper: serve static file with ETag + If-Modified-Since 304."""
    from fastapi import Response
    import email.utils as _eu
    stat = path.stat()
    etag = f'W/"{int(stat.st_mtime)}-{stat.st_size}"'
    last_modified = _eu.formatdate(stat.st_mtime, usegmt=True)
    inm = request.headers.get("if-none-match")
    ims = request.headers.get("if-modified-since")
    not_modified = False
    if inm and inm == etag:
        not_modified = True
    elif ims:
        try:
            if int(_eu.parsedate_to_datetime(ims).timestamp()) >= int(stat.st_mtime):
                not_modified = True
        except Exception:
            pass
    headers = {
        "Cache-Control": "max-age=30, must-revalidate",
        "ETag": etag,
        "Last-Modified": last_modified,
    }
    if not_modified:
        return Response(status_code=304, headers=headers)
    return FileResponse(str(path), media_type=media_type, headers=headers)


@router.get("/v2.css")
async def serve_v2_css(request: Request):
    return _conditional_file_response(FRONTEND_DIR / "v2.css", "text/css", request)


@router.get("/style.css")
async def serve_css(request: Request):
    return _conditional_file_response(FRONTEND_DIR / "style.css", "text/css", request)


@router.get("/v2-theme-enhanced.css")
async def serve_v2_theme_enhanced(request: Request):
    """Optional theme override loaded after v2.css — higher-contrast palette."""
    return _conditional_file_response(
        FRONTEND_DIR / "v2-theme-enhanced.css",
        "text/css",
        request,
    )


@router.get("/js/{subpath:path}")
async def serve_js_module(subpath: str, request: Request):
    """Serve Phase 2 ES modules from frontend/js/.

    Optimizations for page-refresh speed (2026-04-20: user reported slow
    reloads — 61 modules × per-file round-trip was 2-3s on local dev):

    1. Honor `If-None-Match` / `If-Modified-Since` and return 304 so the
       browser reuses its cached copy. Previous code always returned
       200 with full body, so cache headers were useless.
    2. Short `max-age=30` so rapid repeat refreshes skip the server
       entirely. Edits propagate after 30s (or Ctrl+Shift+R forces).
    """
    if ".." in subpath or subpath.startswith("/"):
        return {"error": "invalid path"}
    full_path = FRONTEND_DIR / "js" / subpath
    if not full_path.is_file():
        return {"error": "not found", "path": subpath}

    stat = full_path.stat()
    # Weak ETag: mtime + size. Enough to detect edits.
    etag = f'W/"{int(stat.st_mtime)}-{stat.st_size}"'
    import email.utils as _eu
    last_modified = _eu.formatdate(stat.st_mtime, usegmt=True)

    inm = request.headers.get("if-none-match")
    ims = request.headers.get("if-modified-since")
    not_modified = False
    if inm and inm == etag:
        not_modified = True
    elif ims:
        try:
            ims_ts = _eu.parsedate_to_datetime(ims).timestamp()
            if int(ims_ts) >= int(stat.st_mtime):
                not_modified = True
        except Exception:
            pass

    common_headers = {
        # max-age=0: always revalidate, but 304 (no body) is fast.
        # Previous 30s window was caching stale JS across multiple
        # rapid edits during active development. User 2026-04-21:
        # "刷新之后还是旧的 JS, 看不到 fee 行". With max-age=0 the
        # browser always asks the server; server returns 304 when
        # unchanged (cheap) or 200 with new body when mtime changed.
        "Cache-Control": "max-age=0, must-revalidate",
        "ETag": etag,
        "Last-Modified": last_modified,
    }
    if not_modified:
        from fastapi import Response
        return Response(status_code=304, headers=common_headers)

    return FileResponse(
        str(full_path),
        media_type="application/javascript",
        headers=common_headers,
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
