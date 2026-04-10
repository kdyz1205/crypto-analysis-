from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ..core.config import PROJECT_ROOT
from .types import ManualTrendline

DEFAULT_DRAWINGS_STORE = PROJECT_ROOT / "data" / "manual_trendlines.json"


class ManualTrendlineStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_DRAWINGS_STORE
        self._lock = threading.Lock()

    def list(self, *, symbol: str | None = None, timeframe: str | None = None) -> list[ManualTrendline]:
        with self._lock:
            items = self._read_all()
        if symbol:
            items = [item for item in items if item.symbol == symbol]
        if timeframe:
            items = [item for item in items if item.timeframe == timeframe]
        return sorted(items, key=lambda item: (item.symbol, item.timeframe, item.created_at, item.manual_line_id))

    def get(self, manual_line_id: str) -> ManualTrendline | None:
        with self._lock:
            items = self._read_all()
        for item in items:
            if item.manual_line_id == manual_line_id:
                return item
        return None

    def upsert(self, drawing: ManualTrendline) -> ManualTrendline:
        with self._lock:
            items = self._read_all()
            updated = False
            for index, item in enumerate(items):
                if item.manual_line_id == drawing.manual_line_id:
                    items[index] = drawing
                    updated = True
                    break
            if not updated:
                items.append(drawing)
            self._write_all(items)
        return drawing

    def delete(self, manual_line_id: str) -> bool:
        with self._lock:
            items = self._read_all()
            kept = [item for item in items if item.manual_line_id != manual_line_id]
            if len(kept) == len(items):
                return False
            self._write_all(kept)
        return True

    def clear(self, *, symbol: str | None = None, timeframe: str | None = None) -> int:
        with self._lock:
            items = self._read_all()
            kept: list[ManualTrendline] = []
            removed = 0
            for item in items:
                if symbol and item.symbol != symbol:
                    kept.append(item)
                    continue
                if timeframe and item.timeframe != timeframe:
                    kept.append(item)
                    continue
                if not symbol and not timeframe:
                    removed += 1
                    continue
                removed += 1
            self._write_all(kept)
        return removed

    def _read_all(self) -> list[ManualTrendline]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        items: list[ManualTrendline] = []
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            try:
                items.append(ManualTrendline(**raw))
            except TypeError:
                continue
        return items

    def _write_all(self, items: list[ManualTrendline]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [item.to_dict() for item in items]
        self.path.write_text(json.dumps(serializable, ensure_ascii=True, indent=2), encoding="utf-8")


def now_ts() -> int:
    return int(time.time())


__all__ = ["ManualTrendlineStore", "DEFAULT_DRAWINGS_STORE", "now_ts"]
