"""
Persistent memory subsystem (Hermes-inspired).

Stores agent/user memories across sessions in a simple JSON file.
Every memory is tagged by namespace (e.g. 'project', 'user_pref',
'trade_lesson') and has free-form text + structured metadata.

Memories can be:
- Saved      save_memory(namespace, key, content, metadata)
- Retrieved  get_memory(namespace, key) / get_memory(namespace) for all in namespace
- Searched   search_memory(query) — substring search across all namespaces
- Listed     list_namespaces()
- Deleted    delete_memory(namespace, key)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from .config import PROJECT_ROOT

MEMORY_FILE = PROJECT_ROOT / "agent_memory.json"
_lock = Lock()


def _load() -> dict:
    if not MEMORY_FILE.exists():
        return {}
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Memory] Load failed: {e}")
        return {}


def _save(data: dict) -> None:
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        print(f"[Memory] Save failed: {e}")


def save_memory(namespace: str, key: str, content: str, metadata: dict | None = None) -> dict:
    """Store a memory. Returns the saved entry."""
    with _lock:
        data = _load()
        ns = data.setdefault(namespace, {})
        entry = {
            "content": content,
            "metadata": metadata or {},
            "created_at": ns.get(key, {}).get("created_at") or datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        ns[key] = entry
        _save(data)
        return entry


def get_memory(namespace: str, key: str | None = None) -> dict | None:
    """Fetch one memory (key provided) or all memories in a namespace."""
    with _lock:
        data = _load()
        ns = data.get(namespace, {})
        if key is None:
            return ns
        return ns.get(key)


def search_memory(query: str, limit: int = 20) -> list[dict]:
    """Substring search across all namespaces. Returns list of matching entries."""
    q = query.lower().strip()
    if not q:
        return []
    results: list[dict] = []
    with _lock:
        data = _load()
        for ns, entries in data.items():
            for key, entry in entries.items():
                content = str(entry.get("content", ""))
                if q in content.lower() or q in key.lower():
                    results.append({
                        "namespace": ns,
                        "key": key,
                        **entry,
                    })
                    if len(results) >= limit:
                        return results
    return results


def list_namespaces() -> list[dict]:
    """Return list of namespaces with their entry counts."""
    with _lock:
        data = _load()
        return [{"namespace": ns, "count": len(entries)} for ns, entries in data.items()]


def delete_memory(namespace: str, key: str) -> bool:
    with _lock:
        data = _load()
        ns = data.get(namespace)
        if not ns or key not in ns:
            return False
        del ns[key]
        if not ns:
            del data[namespace]
        _save(data)
        return True


def clear_namespace(namespace: str) -> int:
    """Delete all memories in a namespace. Returns count deleted."""
    with _lock:
        data = _load()
        if namespace not in data:
            return 0
        count = len(data[namespace])
        del data[namespace]
        _save(data)
        return count
