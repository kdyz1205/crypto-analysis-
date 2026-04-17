import pytest

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
        self.requests = []
        _FakeAdapter.instances.append(self)

    def has_api_keys(self):
        return True

    async def _bitget_request(self, method, path, *, mode, body=None, params=None):
        self.requests.append((method, path, mode, body, params))
        return {"code": "00000", "data": {}}

    async def submit_live_entry(self, intent, mode):
        self.intent = intent
        self.mode = mode
        return {
            "ok": True,
            "exchange_order_id": "regular-order-1",
            "submitted_price": intent.entry_price,
        }

    async def cancel_order(self, symbol, order_id, mode):
        return {"ok": True}

    async def cancel_plan_order_any_type(self, symbol, order_id, mode):
        return {"ok": True}


@pytest.mark.asyncio
async def test_place_line_order_submits_post_only_limit_at_line_buffer(monkeypatch):
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
    async def _mark_price(symbol):
        return 110.0

    monkeypatch.setattr(router, "_fetch_bitget_mark_price", _mark_price)

    async def _vol_ctx(symbol, timeframe):
        return {"atr_pct": 1.0}

    monkeypatch.setattr(router, "volatility_context", _vol_ctx)

    import server.execution.live_adapter as live_adapter

    monkeypatch.setattr(live_adapter, "LiveExecutionAdapter", _FakeAdapter)

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
    assert "post-only limit" in result["message"]
    assert adapter.intent.order_type == "limit"
    assert adapter.intent.post_only is True
    assert adapter.intent.entry_price == pytest.approx(106.053)
    assert adapter.intent.stop_price == pytest.approx(106.0)
    assert adapter.intent.tp_price == pytest.approx(106.477)
    assert adapter.intent.quantity == pytest.approx(10.0 / 106.053)
    assert fake_store.created.status == "triggered"
    assert fake_store.created.exchange_order_id == "regular-order-1"
    assert fake_store.created.order.stop_offset_pct_of_line == 0.0
