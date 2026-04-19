import pytest

from server.conditionals import watcher
from server.conditionals.types import ConditionalOrder, OrderConfig, TriggerConfig


class _FakeStore:
    def __init__(self, cond):
        self.cond = cond
        self.updated = []
        self.events = []

    def get(self, conditional_id):
        return self.cond if conditional_id == self.cond.conditional_id else None

    def update(self, cond):
        self.cond = cond
        self.updated.append(cond)
        return cond

    def append_event(self, conditional_id, event):
        self.events.append(event)
        self.cond.events.append(event)
        return self.cond


class _FakeAdapter:
    instances = []

    def __init__(self):
        self.cancelled_regular = []
        self.cancelled_plan = []
        self.plan_submits = []
        self.regular_submits = []
        _FakeAdapter.instances.append(self)

    def has_api_keys(self):
        return True

    async def cancel_order(self, symbol, order_id, mode):
        self.cancelled_regular.append((symbol, order_id, mode))
        return {"ok": True}

    async def cancel_plan_order_any_type(self, symbol, order_id, mode):
        self.cancelled_plan.append((symbol, order_id, mode))
        return {"ok": True}

    async def submit_live_entry(self, intent, mode):
        self.regular_submits.append((intent, mode))
        raise AssertionError("replan must not submit regular limit orders")

    async def submit_live_plan_entry(self, intent, mode, trigger_price):
        self.plan_submits.append((intent, mode, trigger_price))
        return {"ok": True, "exchange_order_id": "new-plan-1"}


def _cond(**overrides):
    base = dict(
        conditional_id="cond_replan",
        manual_line_id="line_replan",
        symbol="HYPEUSDT",
        timeframe="5m",
        side="support",
        t_start=1_000,
        t_end=1_300,
        price_start=40.0,
        price_end=43.0,
        pattern_stats_at_create={},
        trigger=TriggerConfig(poll_seconds=60),
        order=OrderConfig(
            direction="long",
            order_kind="bounce",
            tolerance_pct_of_line=0.1,
            rr_target=2.0,
            notional_usd=10.0,
            submit_to_exchange=True,
            exchange_mode="live",
        ),
        status="triggered",
        created_at=1_000,
        updated_at=1_000,
        triggered_at=1_000,
        exchange_order_id="old-order-1",
        fill_price=40.04,
        fill_qty=0.25,
        last_poll_ts=1_000,
        extend_right=True,
    )
    base.update(overrides)
    return ConditionalOrder(**base)


@pytest.mark.asyncio
async def test_replan_uses_trigger_market_plan_order(monkeypatch):
    cond = _cond()
    store = _FakeStore(cond)
    _FakeAdapter.instances = []
    monkeypatch.setattr(watcher, "_store", store)
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeAdapter)

    await watcher._maybe_replan(cond, 1_300)

    adapter = _FakeAdapter.instances[-1]
    assert adapter.cancelled_regular == [("HYPEUSDT", "old-order-1", "live")]
    assert adapter.regular_submits == []
    assert len(adapter.plan_submits) == 1
    intent, mode, trigger_price = adapter.plan_submits[0]
    assert mode == "live"
    assert intent.order_type == "market"
    assert intent.post_only is False
    assert trigger_price == pytest.approx(intent.entry_price)
    assert cond.exchange_order_id == "new-plan-1"
    assert "trigger-market plan" in store.events[-1].message


@pytest.mark.asyncio
async def test_replan_waits_until_next_timeframe_bar(monkeypatch):
    cond = _cond()
    store = _FakeStore(cond)
    _FakeAdapter.instances = []
    monkeypatch.setattr(watcher, "_store", store)
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeAdapter)

    await watcher._maybe_replan(cond, 1_199)

    assert _FakeAdapter.instances == []
    assert store.updated == []
