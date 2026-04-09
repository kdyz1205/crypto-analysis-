"""
SSE broadcast subscriber that fans events out into per-client thread-safe queues.

Using standard library queues keeps the streaming response work off the main
asyncio event loop so one long-lived SSE connection cannot starve normal API
requests on single-process dev servers.
"""

from __future__ import annotations

import queue as thread_queue

from ..core.events import Event, bus

_client_queues: set[thread_queue.Queue] = set()


def add_client() -> thread_queue.Queue:
    """Register a new SSE client and return its queue."""
    q: thread_queue.Queue = thread_queue.Queue(maxsize=200)
    _client_queues.add(q)
    return q


def remove_client(q: thread_queue.Queue) -> None:
    _client_queues.discard(q)


def client_count() -> int:
    return len(_client_queues)


async def _broadcast(event: Event) -> None:
    dead: list[thread_queue.Queue] = []
    for q in _client_queues:
        try:
            q.put_nowait(event)
        except thread_queue.Full:
            dead.append(q)
    for q in dead:
        _client_queues.discard(q)


def register() -> None:
    for prefix in [
        "signal.*",
        "order.*",
        "position.*",
        "risk.*",
        "agent.*",
        "ops.*",
        "summary.*",
    ]:
        bus.subscribe(prefix, _broadcast)
    print("[SSESubscriber] Registered broadcast for 7 event prefixes")
