"""
Server-Sent Events stream router.

Delivers every event published to the bus to connected clients in real time.
Consumed by the frontend Decision Rail, Glassbox, and (future) the Telegram bot.
"""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..core.events import bus
from ..subscribers.sse_broadcast import add_client, remove_client, client_count

router = APIRouter(tags=["stream"])


async def _event_generator(queue: asyncio.Queue):
    try:
        # Initial connected message so clients know the stream is live
        yield f"event: connected\ndata: {json.dumps({'ok': True})}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                data = json.dumps(event.to_dict(), default=str)
                yield f"event: {event.type}\ndata: {data}\n\n"
            except asyncio.TimeoutError:
                # keepalive ping
                yield f"event: ping\ndata: {{}}\n\n"
    finally:
        remove_client(queue)


@router.get("/api/stream")
async def api_stream():
    """Server-Sent Events stream of all agent events."""
    queue = add_client()
    return StreamingResponse(
        _event_generator(queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/api/stream/status")
async def api_stream_status():
    """Debug endpoint: see connected client count + recent events."""
    return {
        "clients": client_count(),
        "subscribers": bus.subscriber_count(),
        "recent_events": [e.to_dict() for e in bus.get_recent(20)],
    }


@router.get("/api/events/history")
async def api_events_history(limit: int = 100):
    """Return the last N events from the in-memory ring buffer."""
    return {"events": [e.to_dict() for e in bus.get_recent(limit)]}
