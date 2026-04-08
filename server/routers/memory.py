"""
Memory API — Hermes-inspired persistent memory.
GET  /api/memory/namespaces          — list all namespaces
GET  /api/memory/{namespace}         — all entries in a namespace
GET  /api/memory/{namespace}/{key}   — one entry
POST /api/memory/{namespace}/{key}   — save entry (body: { content, metadata })
DELETE /api/memory/{namespace}/{key} — delete entry
GET  /api/memory/search?q=...        — substring search
"""

from fastapi import APIRouter, Query, Request

from ..core import memory as mem

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/namespaces")
async def list_namespaces():
    return {"namespaces": mem.list_namespaces()}


@router.get("/search")
async def search_memory(q: str = Query(...), limit: int = Query(20, ge=1, le=100)):
    return {"query": q, "results": mem.search_memory(q, limit)}


@router.get("/{namespace}")
async def get_namespace(namespace: str):
    data = mem.get_memory(namespace)
    return {"namespace": namespace, "entries": data or {}}


@router.get("/{namespace}/{key}")
async def get_entry(namespace: str, key: str):
    entry = mem.get_memory(namespace, key)
    if entry is None:
        return {"ok": False, "reason": "not found"}
    return {"ok": True, "entry": entry}


@router.post("/{namespace}/{key}")
async def save_entry(namespace: str, key: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "invalid JSON"}
    content = body.get("content", "")
    metadata = body.get("metadata", {})
    entry = mem.save_memory(namespace, key, content, metadata)
    return {"ok": True, "entry": entry}


@router.delete("/{namespace}/{key}")
async def delete_entry(namespace: str, key: str):
    ok = mem.delete_memory(namespace, key)
    return {"ok": ok}
