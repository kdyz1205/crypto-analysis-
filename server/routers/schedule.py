"""
Schedule API — Hermes-inspired task scheduler.
POST /api/schedule/create      — create task { name, recurrence, action, params }
GET  /api/schedule/list        — list all tasks
DELETE /api/schedule/{id}      — delete task
POST /api/schedule/{id}/toggle — enable/disable { enabled }
"""

from fastapi import APIRouter, Request

from ..core import scheduler as sched

router = APIRouter(prefix="/api/schedule", tags=["schedule"])


@router.post("/create")
async def create_task(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "invalid JSON"}
    name = body.get("name")
    recurrence = body.get("recurrence")
    action = body.get("action")
    params = body.get("params", {})
    if not (name and recurrence and action):
        return {"ok": False, "reason": "name, recurrence, action required"}
    task = sched.create_task(name, recurrence, action, params)
    return {"ok": True, "task": task.to_dict()}


@router.get("/list")
async def list_tasks():
    return {"tasks": sched.list_tasks()}


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    return {"ok": sched.delete_task(task_id)}


@router.post("/{task_id}/toggle")
async def toggle_task(task_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "reason": "invalid JSON"}
    enabled = bool(body.get("enabled", True))
    return {"ok": sched.toggle_task(task_id, enabled)}
