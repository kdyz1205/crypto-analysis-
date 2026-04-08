"""
SSE broadcast subscriber — pushes events into per-client asyncio queues
that feed the /api/stream Server-Sent Events endpoint.
"""

import asyncio

from ..core.events import bus, Event

# Per-client queues (set of asyncio.Queue)
_client_queues: set[asyncio.Queue] = set()


def add_client() -> asyncio.Queue:
    """Register a new SSE client and return its queue."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _client_queues.add(q)
    return q


def remove_client(q: asyncio.Queue) -> None:
    _client_queues.discard(q)


def client_count() -> int:
    return len(_client_queues)


async def _broadcast(event: Event):
    dead: list[asyncio.Queue] = []
    for q in _client_queues:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _client_queues.discard(q)


def register():
    for prefix in [
        "signal.*", "order.*", "position.*", "risk.*",
        "agent.*", "ops.*", "summary.*",
    ]:
        bus.subscribe(prefix, _broadcast)
    print("[SSESubscriber] Registered broadcast for 7 event prefixes")
