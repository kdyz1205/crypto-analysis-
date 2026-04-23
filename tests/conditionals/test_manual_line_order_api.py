import pytest

from server.conditionals import ConditionalOrder, OrderConfig, TriggerConfig
from server.drawings.types import ManualTrendline
from server.routers import conditionals as router


class _FakeDrawingStore:
    def __init__(self, drawing):
        self.drawing = drawing

    def get(self, manual_line_id: str):
        return self.drawing if manual_line_id == self.drawing.manual_line_id else None


class _FakeConditionalStore:
    def __init__(self):
        self.created = None
        self.events = []

    def create(self, cond):
        self.created = cond
        return cond

    def append_event(self, conditional_id, event):
        self.events.append((conditional_id, event))
        return self.created


class _FakeAdapter:
    instances = []

    def __init__(self):
        self.intent = None
        self.mode = None
        self.trigger_price = None
        self.requests = []
        _FakeAdapter.instances.append(self)

    def has_api_keys(self):
        return True

    async def _bitget_request(self, method, path, *, mode, body=None, params=None):
        self.requests.append((method, path, mode, body, params))
        return {"code": "00000", "data": {}}

    async def submit_live_entry(self, intent, mode):
        raise AssertionError("manual line orders must use Bitget plan orders")

    async def submit_live_plan_entry(self, intent, mode, trigger_price):
        self.intent = intent
        self.mode = mode
        self.trigger_price = trigger_price
        return {
            "ok": True,
            "exchange_order_id": "plan-order-1",
            "submitted_price": trigger_price,
            "submitted_order_type": intent.order_type,
        }

    async def cancel_order(self, symbol, order_id, mode):
        return {"ok": True}

    async def cancel_plan_order_any_type(self, symbol, order_id, mode):
        return {"ok": True}

    async def get_pending_orders(self, mode, *, symbol=None):
        return []

    async def get_pending_plan_orders(self, mode, *, plan_type="normal_plan", symbol=None):
        return []


class _FakeCancelStore:
    def __init__(self, cond):
        self.cond = cond
        self.events = []
        self.deleted = []

    def get(self, conditional_id):
        return self.cond if conditional_id == self.cond.conditional_id else None

    def append_event(self, conditional_id, event):
        self.events.append((conditional_id, event))
        if conditional_id == self.cond.conditional_id:
            self.cond.events.append(event)
        return self.cond

    def set_status(self, conditional_id, status, *, reason=""):
        if conditional_id != self.cond.conditional_id:
            return None
        self.cond.status = status
        if status == "cancelled":
            self.cond.cancel_reason = reason
        return self.cond

    def delete(self, conditional_id):
        if conditional_id != self.cond.conditional_id:
            return False
        self.deleted.append(conditional_id)
        return True


class _FakeCancelAdapter:
    cancel_regular_ok = False
    cancel_plan_ok = False
    pending_oids = {"plan-order-1"}

    def has_api_keys(self):
        return True

    async def cancel_order(self, symbol, order_id, mode):
        return {"ok": self.cancel_regular_ok, "reason": "regular_failed"}

    async def cancel_plan_order_any_type(self, symbol, order_id, mode):
        return {"ok": self.cancel_plan_ok, "reason": "plan_failed"}

    async def get_pending_orders(self, mode, *, symbol=None):
        return [{"orderId": oid, "symbol": symbol or "HYPEUSDT"} for oid in self.pending_oids]

    async def get_pending_plan_orders(self, mode, *, plan_type="normal_plan", symbol=None):
        return [{"orderId": oid, "symbol": symbol or "HYPEUSDT"} for oid in self.pending_oids]


def _triggered_cond():
    return ConditionalOrder(
        conditional_id="cond-live-1",
        manual_line_id="line-1",
        symbol="HYPEUSDT",
        timeframe="1h",
        side="support",
        t_start=1_000,
        t_end=1_600,
        price_start=100.0,
        price_end=106.0,
        pattern_stats_at_create={},
        trigger=TriggerConfig(poll_seconds=60),
        order=OrderConfig(
            direction="long",
            order_kind="bounce",
            tolerance_pct_of_line=0.05,
            stop_offset_pct_of_line=0.01,
            rr_target=8.0,
            notional_usd=10.0,
            submit_to_exchange=True,
            exchange_mode="live",
        ),
        status="triggered",
        created_at=1_000,
        updated_at=1_000,
        exchange_order_id="plan-order-1",
        extend_right=True,
    )


@pytest.mark.asyncio
async def test_place_line_order_submits_trigger_market_plan_at_line_buffer(monkeypatch, tmp_path):
    drawing = ManualTrendline(
        manual_line_id="line-1",
        symbol="HYPEUSDT",
        timeframe="1h",
        side="support",
        source="manual",
        t_start=1_000,
        t_end=1_600,
        price_start=100.0,
        price_end=106.0,
        extend_left=False,
        extend_right=True,
        locked=False,
        label="",
        notes="",
        comparison_status="uncompared",
        override_mode="display_only",
        nearest_auto_line_id=None,
        slope_diff=None,
        projected_price_diff=None,
        overlap_ratio=None,
        created_at=1_000,
        updated_at=1_000,
    )
    fake_store = _FakeConditionalStore()
    _FakeAdapter.instances = []

    monkeypatch.setattr(router, "_drawings", _FakeDrawingStore(drawing))
    monkeypatch.setattr(router, "_store", fake_store)
    monkeypatch.setattr(router, "now_ts", lambda: 1_600)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("CONFIRM_LIVE_TRADING", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    async def _mark_price(symbol):
        return 110.0

    monkeypatch.setattr(router, "_fetch_bitget_mark_price", _mark_price)

    async def _vol_ctx(symbol, timeframe):
        return {"atr_pct": 1.0}

    monkeypatch.setattr(router, "volatility_context", _vol_ctx)

    import server.execution.live_adapter as live_adapter

    monkeypatch.setattr(live_adapter, "LiveExecutionAdapter", _FakeAdapter)

    from server.strategy import drawing_learner

    monkeypatch.setattr(drawing_learner, "ML_DRAWINGS_FILE", tmp_path / "user_drawings_ml.jsonl")

    req = router.PlaceLineOrderReq(
        manual_line_id="line-1",
        direction="long",
        tolerance_pct=0.05,
        size_usdt=10.0,
        mode="live",
        rr_target=8.0,
    )

    result = await router.api_place_line_order(req)

    adapter = _FakeAdapter.instances[-1]
    assert result["ok"] is True
    assert "trigger-market plan" in result["message"]
    assert adapter.intent.order_type == "market"
    assert adapter.intent.post_only is False
    # Placement now projects the line at NOW, matching what the user sees on
    # the right edge of the live bar.
    assert adapter.trigger_price == pytest.approx(106.053)
    assert adapter.intent.entry_price == pytest.approx(106.053)
    assert adapter.intent.stop_price == pytest.approx(105.9576)
    assert adapter.intent.tp_price == pytest.approx(106.8162)
    assert adapter.intent.quantity == pytest.approx(10.0 / 106.053)
    assert fake_store.created.status == "triggered"
    assert fake_store.created.exchange_order_id == "plan-order-1"
    assert fake_store.created.order.stop_offset_pct_of_line == 0.04

    ml_rows = (tmp_path / "user_drawings_ml.jsonl").read_text(encoding="utf-8").splitlines()
    assert '"event": "user_order_intent"' in ml_rows[-1]
    assert '"label_reason": "user_placed_line_order"' in ml_rows[-1]


@pytest.mark.asyncio
async def test_cancel_conditional_refuses_local_cancel_when_bitget_cancel_not_confirmed(monkeypatch):
    cond = _triggered_cond()
    fake_store = _FakeCancelStore(cond)
    _FakeCancelAdapter.cancel_regular_ok = False
    _FakeCancelAdapter.cancel_plan_ok = False
    _FakeCancelAdapter.pending_oids = {"plan-order-1"}
    monkeypatch.setattr(router, "_store", fake_store)

    import server.execution.live_adapter as live_adapter

    monkeypatch.setattr(live_adapter, "LiveExecutionAdapter", _FakeCancelAdapter)

    with pytest.raises(router.HTTPException) as exc:
        await router.api_cancel_conditional("cond-live-1")

    assert exc.value.status_code == 409
    assert cond.status == "triggered"
    assert any(event.kind == "exchange_error" for _, event in fake_store.events)


@pytest.mark.asyncio
async def test_cancel_conditional_marks_cancelled_after_plan_cancel_confirmed(monkeypatch):
    cond = _triggered_cond()
    fake_store = _FakeCancelStore(cond)
    _FakeCancelAdapter.cancel_regular_ok = False
    _FakeCancelAdapter.cancel_plan_ok = True
    _FakeCancelAdapter.pending_oids = {"plan-order-1"}
    monkeypatch.setattr(router, "_store", fake_store)

    import server.execution.live_adapter as live_adapter

    monkeypatch.setattr(live_adapter, "LiveExecutionAdapter", _FakeCancelAdapter)

    result = await router.api_cancel_conditional("cond-live-1")

    assert result["ok"] is True
    assert result["bitget_cancelled"] is True
    assert cond.status == "cancelled"
    assert any(event.kind == "cancelled" for _, event in fake_store.events)
