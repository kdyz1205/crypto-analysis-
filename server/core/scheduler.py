"""
Natural-language task scheduler (Hermes-inspired).

Parses simple recurrence patterns like:
  - "every 15 minutes"
  - "every hour"
  - "every day at 08:00"
  - "every 4 hours"
  - "daily 16:00"

Runs async tasks at scheduled times via a single asyncio background loop.
Tasks are callables registered by name — the scheduler stores (name, cron_string,
interval_seconds, last_run, next_run, enabled).

This is NOT a full cron implementation — just what we need for trading bot
use cases: daily reports, periodic scans, hourly summaries.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Awaitable, Callable

from .config import PROJECT_ROOT

SCHEDULE_FILE = PROJECT_ROOT / "agent_schedule.json"
_lock = Lock()


@dataclass
class ScheduledTask:
    id: str
    name: str                # human-readable
    recurrence: str          # raw text the user typed
    interval_seconds: int    # how often (0 = use target_time)
    target_hour: int | None = None  # for "daily at HH:MM"
    target_minute: int | None = None
    action: str = ""         # tool name to invoke
    params: dict = field(default_factory=dict)
    enabled: bool = True
    last_run: str | None = None
    next_run: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


# Handler registry — name → async function(params) -> str
_handlers: dict[str, Callable[[dict], Awaitable[str]]] = {}
_tasks: dict[str, ScheduledTask] = {}
_loop_running = False


def register_handler(name: str, fn: Callable[[dict], Awaitable[str]]) -> None:
    """Register an async task handler by name."""
    _handlers[name] = fn


def _load_from_disk():
    if not SCHEDULE_FILE.exists():
        return
    try:
        raw = json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
        for tid, d in raw.items():
            _tasks[tid] = ScheduledTask(**d)
    except Exception as e:
        print(f"[Scheduler] Load failed: {e}")


import threading
_disk_lock = threading.Lock()

def _save_to_disk():
    try:
        with _disk_lock:
            SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
            SCHEDULE_FILE.write_text(
                json.dumps({tid: t.to_dict() for tid, t in _tasks.items()}, indent=2, default=str),
                encoding="utf-8",
            )
    except Exception as e:
        print(f"[Scheduler] Save failed: {e}")


def parse_recurrence(text: str) -> dict:
    """
    Parse natural-language recurrence into structured config.
    Returns { interval_seconds, target_hour, target_minute }.
    """
    t = text.lower().strip()

    # "every N minutes/hours/seconds"
    m = re.match(r"every\s+(\d+)\s*(second|minute|hour|day|week)s?", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}[unit]
        return {"interval_seconds": n * mult, "target_hour": None, "target_minute": None}

    # "every minute/hour/day"
    m = re.match(r"every\s+(second|minute|hour|day|week)", t)
    if m:
        unit = m.group(1)
        mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}[unit]
        return {"interval_seconds": mult, "target_hour": None, "target_minute": None}

    # "daily at HH:MM" or "every day at HH:MM"
    m = re.search(r"(?:daily|every day)\s+(?:at\s+)?(\d{1,2}):(\d{2})", t)
    if m:
        return {
            "interval_seconds": 86400,
            "target_hour": int(m.group(1)),
            "target_minute": int(m.group(2)),
        }

    # "daily HH:MM"
    m = re.match(r"daily\s+(\d{1,2}):(\d{2})", t)
    if m:
        return {
            "interval_seconds": 86400,
            "target_hour": int(m.group(1)),
            "target_minute": int(m.group(2)),
        }

    # Default: treat as hourly
    return {"interval_seconds": 3600, "target_hour": None, "target_minute": None}


def _compute_next_run(task: ScheduledTask) -> datetime:
    now = datetime.now(timezone.utc)
    if task.target_hour is not None:
        # Scheduled daily at specific time
        candidate = now.replace(
            hour=task.target_hour,
            minute=task.target_minute or 0,
            second=0, microsecond=0,
        )
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate
    # Interval-based
    if task.last_run:
        last = datetime.fromisoformat(task.last_run)
        return last + timedelta(seconds=task.interval_seconds)
    return now + timedelta(seconds=task.interval_seconds)


def create_task(name: str, recurrence: str, action: str, params: dict | None = None) -> ScheduledTask:
    """Create + persist a new task."""
    parsed = parse_recurrence(recurrence)
    task_id = f"task_{int(datetime.now().timestamp())}_{len(_tasks)}"
    task = ScheduledTask(
        id=task_id,
        name=name,
        recurrence=recurrence,
        interval_seconds=parsed["interval_seconds"],
        target_hour=parsed["target_hour"],
        target_minute=parsed["target_minute"],
        action=action,
        params=params or {},
    )
    task.next_run = _compute_next_run(task).isoformat()
    with _lock:
        _tasks[task_id] = task
        _save_to_disk()
    print(f"[Scheduler] Created task {task_id}: {name} ({recurrence}) next={task.next_run}")
    return task


def list_tasks() -> list[dict]:
    with _lock:
        return [t.to_dict() for t in _tasks.values()]


def delete_task(task_id: str) -> bool:
    with _lock:
        if task_id in _tasks:
            del _tasks[task_id]
            _save_to_disk()
            return True
        return False


def toggle_task(task_id: str, enabled: bool) -> bool:
    with _lock:
        if task_id in _tasks:
            _tasks[task_id].enabled = enabled
            _save_to_disk()
            return True
        return False


async def _run_task(task: ScheduledTask):
    handler = _handlers.get(task.action)
    if not handler:
        print(f"[Scheduler] No handler for action: {task.action}")
        return
    try:
        result = await handler(task.params)
        task.last_run = datetime.now(timezone.utc).isoformat()
        task.next_run = _compute_next_run(task).isoformat()
        _save_to_disk()
        print(f"[Scheduler] Ran {task.name}: {str(result)[:100]}")
    except Exception as e:
        print(f"[Scheduler] Task {task.name} failed: {e}")


async def _scheduler_loop():
    global _loop_running
    _loop_running = True
    _load_from_disk()
    print(f"[Scheduler] Loop started ({len(_tasks)} tasks loaded)")

    while _loop_running:
        try:
            now = datetime.now(timezone.utc)
            due = []
            with _lock:
                for task in list(_tasks.values()):
                    if not task.enabled or not task.next_run:
                        continue
                    try:
                        nr = datetime.fromisoformat(task.next_run)
                        if nr <= now:
                            due.append(task)
                    except Exception:
                        pass

            for task in due:
                await _run_task(task)
        except Exception as e:
            print(f"[Scheduler] Loop error: {e}")

        await asyncio.sleep(30)  # Check every 30s


def start_scheduler():
    """Start the scheduler loop as an asyncio background task."""
    asyncio.create_task(_scheduler_loop())


def stop_scheduler():
    global _loop_running
    _loop_running = False
