from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..execution.live_adapter import LiveExecutionAdapter
from ..execution.live_engine import LiveBridgeConfig, LiveExecutionEngine
from ..execution.types import OrderIntent
from ..schemas.live_execution import (
    LiveCloseRequest,
    LiveCloseResponse,
    LiveFlattenAllRequest,
    LiveFlattenAllResponse,
    LivePreflightResponse,
    LiveExecutionStatusResponse,
    LivePreviewRequest,
    LivePreviewResponse,
    LiveReconcileRequest,
    LiveReconcileResponse,
    LiveSubmitRequest,
    LiveSubmitResponse,
)
from .paper_execution import paper_engine

router = APIRouter(prefix="/api/live-execution", tags=["live-execution"])


def _adapter_provider() -> LiveExecutionAdapter:
    return LiveExecutionAdapter()


live_engine = LiveExecutionEngine(
    adapter_provider=_adapter_provider,
    config=LiveBridgeConfig.from_env(),
)


@router.get("/status", response_model=LiveExecutionStatusResponse)
async def api_live_execution_status():
    return {"status": live_engine.get_status()}


@router.get("/account")
async def api_live_execution_account(mode: str = Query("demo")):
    """Get Bitget account summary — balance, positions, pending orders."""
    normalized = _normalize_mode(mode)
    adapter = _adapter_provider()
    if not adapter.has_api_keys():
        return {"ok": False, "reason": "api_keys_missing"}
    return await adapter.get_live_account_status(normalized)


@router.get("/preflight", response_model=LivePreflightResponse)
async def api_live_execution_preflight(
    mode: str = Query("live"),
    order_intent_id: str | None = Query(None),
    signal_id: str | None = Query(None),
):
    normalized_mode = _normalize_mode(mode)
    intent = _resolve_optional_intent(order_intent_id, signal_id)
    preflight = await live_engine.build_preflight(mode=normalized_mode, intent=intent)
    return {"preflight": preflight}


@router.post("/reconcile", response_model=LiveReconcileResponse)
async def api_live_execution_reconcile(req: LiveReconcileRequest):
    mode = _normalize_mode(req.mode)
    reconciliation = await live_engine.reconcile_startup(mode)
    return {"reconciliation": reconciliation}


@router.post("/preview", response_model=LivePreviewResponse)
async def api_live_execution_preview(req: LivePreviewRequest):
    mode = _normalize_mode(req.mode)
    intent = _resolve_intent(req.order_intent_id, req.signal_id)
    result = await live_engine.preview_live_submission(intent, mode=mode)
    return {"result": result}


@router.post("/submit", response_model=LiveSubmitResponse)
async def api_live_execution_submit(req: LiveSubmitRequest):
    mode = _normalize_mode(req.mode)
    intent = _resolve_intent(req.order_intent_id, None)
    result = await live_engine.execute_live_submission(intent, mode=mode, confirm=req.confirm)
    return {"result": result}


@router.post("/close", response_model=LiveCloseResponse)
async def api_live_execution_close(req: LiveCloseRequest):
    mode = _normalize_mode(req.mode)
    result = await live_engine.close_live_position(req.symbol, mode=mode, confirm=req.confirm)
    return {"result": result}


@router.post("/flatten-all", response_model=LiveFlattenAllResponse)
async def api_live_execution_flatten_all(req: LiveFlattenAllRequest):
    """Emergency kill switch.
    Closes every position + cancels every plan order for the given mode.
    Protected by a literal confirmation code to prevent accidental hits.
    """
    if (req.confirm_code or "").strip().upper() != "FLATTEN":
        return {
            "ok": False,
            "mode": req.mode,
            "attempted": 0,
            "closed": 0,
            "failures": [],
            "plan_orders_cancelled": 0,
            "plan_orders_failed": 0,
            "reason": "confirm_code_mismatch — send {'confirm_code': 'FLATTEN'}",
        }

    mode = _normalize_mode(req.mode)
    adapter = _adapter_provider()
    if not adapter.has_api_keys():
        return {
            "ok": False,
            "mode": mode,
            "attempted": 0,
            "closed": 0,
            "failures": [],
            "plan_orders_cancelled": 0,
            "plan_orders_failed": 0,
            "reason": "api_keys_missing",
        }

    # 1. Fetch live positions for the mode.
    account = await adapter.get_live_account_status(mode)
    positions = account.get("positions") or []
    attempted = 0
    closed = 0
    failures: list[dict] = []
    for pos in positions:
        sym = str(pos.get("symbol") or "").upper()
        if not sym:
            continue
        size = float(pos.get("total") or 0)
        if size <= 0:
            continue
        attempted += 1
        try:
            result = await live_engine.close_live_position(sym, mode=mode, confirm=True)
            if result.get("ok"):
                closed += 1
            else:
                failures.append({
                    "symbol": sym,
                    "reason": result.get("reason") or result.get("blocking_reasons") or "unknown",
                })
        except Exception as exc:
            failures.append({"symbol": sym, "reason": f"exception: {exc}"})

    plan_cancelled = 0
    plan_failed = 0
    if req.cancel_plans:
        try:
            from ..strategy.trendline_order_manager import cancel_all_trendline_plan_orders
            cfg = {"mode": mode}
            cancel_resp = await cancel_all_trendline_plan_orders(cfg, status="flatten_all")
            plan_cancelled = int(cancel_resp.get("cancelled") or 0)
            plan_failed = int(cancel_resp.get("failed") or 0)
        except Exception as exc:
            print(f"[flatten-all] cancel plans err: {exc}", flush=True)
            plan_failed = -1  # signal exception

    ok = (attempted == closed and plan_failed >= 0)
    print(
        f"[flatten-all] mode={mode} attempted={attempted} closed={closed} "
        f"plan_cancelled={plan_cancelled} plan_failed={plan_failed} "
        f"failures={len(failures)}",
        flush=True,
    )
    return {
        "ok": ok,
        "mode": mode,
        "attempted": attempted,
        "closed": closed,
        "failures": failures,
        "plan_orders_cancelled": plan_cancelled,
        "plan_orders_failed": plan_failed,
        "reason": None if ok else "see failures[]",
    }


def _mark_local_cond_cancelled(order_id: str, reason: str = "cancelled via api") -> bool:
    """Find any local ConditionalOrder pointing to this Bitget order_id and
    mark it cancelled, so the watcher's _maybe_replan won't resurrect it.
    Returns True if a cond was updated."""
    try:
        from ..conditionals.store import ConditionalOrderStore
        from ..conditionals.types import ConditionalEvent
        import time as _t
        store = ConditionalOrderStore()
        all_conds = store.list_all() if hasattr(store, "list_all") else store.list()
        target = str(order_id).strip()
        updated = False
        for c in all_conds:
            cond_oid = str(getattr(c, "exchange_order_id", "") or "").strip()
            if cond_oid == target and c.status != "cancelled":
                store.set_status(c.conditional_id, "cancelled", reason=reason)
                try:
                    store.append_event(c.conditional_id, ConditionalEvent(
                        ts=int(_t.time()), kind="cancelled",
                        message=f"cancelled via api: bitget_order_id={order_id}",
                    ))
                except Exception as e:
                    print(f"[cancel-order] append_event err: {e}", flush=True)
                updated = True
                print(f"[cancel-order] marked local cond {c.conditional_id} cancelled (oid={target})", flush=True)
        if not updated:
            print(f"[cancel-order] no local cond found with oid={target}", flush=True)
        return updated
    except Exception as e:
        import traceback
        print(f"[cancel-order] local cond mark failed: {e}\n{traceback.format_exc()}", flush=True)
    return False


@router.post("/cancel-order")
async def api_live_cancel_order(
    symbol: str = Query(...),
    order_id: str = Query(...),
    mode: str = Query("live"),
):
    """Cancel a Bitget order. Tries the regular order endpoint first;
    falls back to the plan-order endpoint when the order id belongs to
    a trigger / plan order (different namespace on Bitget).

    CRITICAL: Always marks the matching local ConditionalOrder as
    cancelled — otherwise the watcher's replan loop will see a
    `triggered` cond with a dead exchange_order_id, "replan" it, and
    place a new ghost order on Bitget."""
    adapter = _adapter_provider()
    if not adapter.has_api_keys():
        return {"ok": False, "reason": "api_keys_missing"}
    sym = symbol.upper()
    norm_mode = _normalize_mode(mode)

    # Try regular order cancel first
    regular_resp = await adapter.cancel_order(sym, order_id, norm_mode)
    if regular_resp.get("ok"):
        local_updated = _mark_local_cond_cancelled(order_id, reason="user cancel via api")
        return {"ok": True, "order_id": order_id, "kind": "order",
                "bitget": regular_resp.get("exchange_response_excerpt"), "local_cond_updated": local_updated}

    # Fall through to plan / trigger order cancel
    plan_resp = await adapter.cancel_plan_order_any_type(sym, order_id, norm_mode)
    if plan_resp.get("ok"):
        local_updated = _mark_local_cond_cancelled(order_id, reason="user cancel via api (plan)")
        return {"ok": True, "order_id": order_id, "kind": "plan",
                "bitget": plan_resp.get("exchange_response_excerpt"), "local_cond_updated": local_updated}

    # Bitget did not confirm the cancel. Keep local state active so the UI
    # does not hide a potentially-live exchange order. Reconcile can clean
    # up later only after exchange truth proves the order is gone.
    local_updated = False
    return {
        "ok": False,
        "order_id": order_id,
        "reason": plan_resp.get("reason") or regular_resp.get("reason") or "both cancel paths failed",
        "raw_order": regular_resp,
        "raw_plan": plan_resp,
        "local_cond_updated": local_updated,
    }


@router.get("/plan-orders")
async def api_live_plan_orders(mode: str = Query("live")):
    """Fetch all pending Bitget plan / trigger orders for USDT futures."""
    adapter = _adapter_provider()
    if not adapter.has_api_keys():
        return {"ok": False, "reason": "api_keys_missing"}
    resp = await adapter._bitget_request(
        "GET", "/api/v2/mix/order/orders-plan-pending",
        mode=_normalize_mode(mode),
        params={"productType": "USDT-FUTURES", "planType": "normal_plan"},
    )
    if resp.get("code") != "00000":
        return {"ok": False, "reason": resp.get("msg") or str(resp)[:300], "raw": resp}
    data = resp.get("data") or {}
    rows = data.get("entrustedList") or data.get("orderList") or []
    return {
        "ok": True,
        "count": len(rows),
        "plan_orders": rows,
    }


@router.post("/cancel-all")
async def api_live_cancel_all(symbol: str = Query(...), mode: str = Query("live")):
    """Cancel ALL pending orders on a symbol. Use to clean up stale tests."""
    adapter = _adapter_provider()
    if not adapter.has_api_keys():
        return {"ok": False, "reason": "api_keys_missing"}
    body = {
        "symbol": symbol.upper(),
        "productType": "USDT-FUTURES",
        "marginCoin": "USDT",
    }
    resp = await adapter._bitget_request("POST", "/api/v2/mix/order/cancel-symbol-orders", mode=_normalize_mode(mode), body=body)
    if resp.get("code") == "00000":
        return {"ok": True, "bitget": resp.get("data")}
    return {"ok": False, "reason": resp.get("msg") or str(resp)[:300], "raw": resp}


def _resolve_intent(order_intent_id: str | None, signal_id: str | None) -> OrderIntent:
    if signal_id:
        intent = paper_engine.order_manager.get_intent(signal_id)
        if intent is not None:
            return intent
        raise HTTPException(404, f"Unknown signal_id: {signal_id}")

    if order_intent_id:
        for intent in paper_engine.order_manager.get_intents():
            if intent.order_intent_id == order_intent_id:
                return intent
        raise HTTPException(404, f"Unknown order_intent_id: {order_intent_id}")

    raise HTTPException(400, "order_intent_id or signal_id is required")


def _resolve_optional_intent(order_intent_id: str | None, signal_id: str | None) -> OrderIntent | None:
    if order_intent_id or signal_id:
        return _resolve_intent(order_intent_id, signal_id)

    intents = [
        intent
        for intent in paper_engine.order_manager.get_intents()
        if intent.status in {"approved", "submitted"}
    ]
    if not intents:
        return None
    intents.sort(key=lambda intent: (intent.created_at_bar, intent.order_intent_id), reverse=True)
    return intents[0]


def _normalize_mode(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    if normalized not in {"demo", "live"}:
        raise HTTPException(400, "mode must be one of: demo, live")
    return normalized


__all__ = ["live_engine", "router"]
