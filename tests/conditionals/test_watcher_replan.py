import pytest

from server.conditionals import watcher
from server.conditionals.types import ConditionalOrder, OrderConfig, TriggerConfig


class _FakeStore:
    def __init__(self, cond):
        self.cond = cond
        self.updated = []
        self.events = []
        self.status_changes = []

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

    def set_status(self, conditional_id, status, *, reason=""):
        if conditional_id != self.cond.conditional_id:
            return None
        self.cond.status = status
        if status == "cancelled":
            self.cond.cancel_reason = reason
        self.status_changes.append((status, reason))
        return self.cond

    def set_status_if(self, conditional_id, *, from_status, to_status, reason=""):
        """Mirror store.set_status_if atomic CAS for tests."""
        if conditional_id != self.cond.conditional_id:
            return None
        if self.cond.status != from_status:
            return None
        self.cond.status = to_status
        if to_status == "cancelled":
            self.cond.cancel_reason = reason
        self.status_changes.append((to_status, reason))
        return self.cond


class _FakeAdapter:
    instances = []
    # Test overrides: set before calling _maybe_replan to force a specific
    # cancel outcome. Default both True (happy path).
    cancel_regular_ok = True
    cancel_plan_ok = True
    # If set to an exception instance, the next cancel call raises it.
    cancel_regular_raise = None
    cancel_plan_raise = None

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
        if _FakeAdapter.cancel_regular_raise:
            raise _FakeAdapter.cancel_regular_raise
        return {"ok": _FakeAdapter.cancel_regular_ok}

    async def cancel_plan_order_any_type(self, symbol, order_id, mode):
        self.cancelled_plan.append((symbol, order_id, mode))
        if _FakeAdapter.cancel_plan_raise:
            raise _FakeAdapter.cancel_plan_raise
        return {"ok": _FakeAdapter.cancel_plan_ok}

    async def get_pending_orders(self, mode):
        return [{"orderId": "old-order-1"}]

    async def get_pending_plan_orders(self, mode, plan_type="normal_plan"):
        return [{"orderId": "old-order-1"}]

    async def submit_live_entry(self, intent, mode):
        self.regular_submits.append((intent, mode))
        raise AssertionError("replan must not submit regular limit orders")

    async def submit_live_plan_entry(self, intent, mode, trigger_price):
        self.plan_submits.append((intent, mode, trigger_price))
        return {"ok": True, "exchange_order_id": "new-plan-1"}


def _reset_fake_adapter():
    _FakeAdapter.instances = []
    _FakeAdapter.cancel_regular_ok = True
    _FakeAdapter.cancel_plan_ok = True
    _FakeAdapter.cancel_regular_raise = None
    _FakeAdapter.cancel_plan_raise = None


async def _mark_returns(value):
    """Factory for a mocked _fetch_mark_price_strict that returns `value`."""
    async def _inner(symbol):
        return value
    return _inner


def _patch_mark(monkeypatch, value):
    """Helper: patch _fetch_mark_price_strict to return a fixed value."""
    async def _fake(symbol):
        return value
    monkeypatch.setattr(watcher, "_fetch_mark_price_strict", _fake)
    # Also patch the pending-oids cache to bypass real Bitget calls.
    async def _fake_pending(mode):
        return {"old-order-1"}
    monkeypatch.setattr(watcher, "_get_cached_pending_oids", _fake_pending)


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
    _reset_fake_adapter()
    monkeypatch.setattr(watcher, "_store", store)
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeAdapter)
    # Mock mark at line_now value (43.0) so neither long nor short line-broken
    # check fires; replan path should run.
    _patch_mark(monkeypatch, 43.0)

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
    _reset_fake_adapter()
    monkeypatch.setattr(watcher, "_store", store)
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeAdapter)
    _patch_mark(monkeypatch, 43.0)

    await watcher._maybe_replan(cond, 1_199)

    assert _FakeAdapter.instances == []
    assert store.updated == []


# ─────────────────────────────────────────────────────────────
# Line-broken invalidation tests (user 2026-04-22 BAS incident)
# ─────────────────────────────────────────────────────────────

def _short_descending_cond(**overrides):
    """Mirror the real BAS 2026-04-22 setup: descending line, SHORT direction,
    tolerance_pct=1.0, stop_offset_pct=0.1.
    Line goes from 0.01950 at t=1000 to 0.01199 at t=1300 (descending).
    """
    base = dict(
        conditional_id="cond_bas_short",
        manual_line_id="line_bas",
        symbol="BASUSDT",
        timeframe="5m",
        side="support",
        t_start=1_000,
        t_end=1_300,
        price_start=0.01950,
        price_end=0.01199,
        pattern_stats_at_create={},
        trigger=TriggerConfig(poll_seconds=60),
        order=OrderConfig(
            direction="short",
            order_kind="bounce",
            tolerance_pct_of_line=1.0,
            stop_offset_pct_of_line=0.1,
            rr_target=5.0,
            notional_usd=300.0,
            submit_to_exchange=True,
            exchange_mode="live",
        ),
        status="triggered",
        created_at=1_000,
        updated_at=1_000,
        triggered_at=1_000,
        exchange_order_id="old-order-1",
        fill_price=0.01493,   # the original trigger price
        fill_qty=20094.0,
        last_poll_ts=1_000,
        extend_right=True,
    )
    base.update(overrides)
    return ConditionalOrder(**base)


@pytest.mark.asyncio
async def test_line_broken_short_cancels_when_mark_above_line_plus_stop_pct(monkeypatch):
    """BAS 2026-04-22 scenario: line descended under a flat price, mark is
    above line × (1 + stop_pct). Expected: cancel, status → cancelled,
    line_broken event logged, NO replan (no plan_submits).
    NOTE: ts=1300 with 5m TF snaps to bar_open=1200, where log-interpolated
    line ≈ 0.01411. Mark must be > 0.01411 × 1.001 = 0.01412 to trigger."""
    cond = _short_descending_cond()
    store = _FakeStore(cond)
    _reset_fake_adapter()
    monkeypatch.setattr(watcher, "_store", store)
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeAdapter)
    # Mark=0.016 safely above line threshold
    _patch_mark(monkeypatch, 0.016)

    await watcher._maybe_replan(cond, 1_300)

    adapter = _FakeAdapter.instances[-1]
    # Confirm cancel fired, NOT replan
    assert len(adapter.cancelled_regular) == 1, f"expected 1 cancel, got {adapter.cancelled_regular}"
    assert adapter.plan_submits == [], "replan MUST NOT run when line-broken fires"
    # Status transitioned via CAS
    assert cond.status == "cancelled"
    assert any(s == "cancelled" for s, _ in store.status_changes)
    # line_broken event was appended
    assert any(e.kind == "line_broken" for e in store.events), \
        f"line_broken event missing; events: {[e.kind for e in store.events]}"


@pytest.mark.asyncio
async def test_line_broken_cancel_FAILS_both_paths_leaves_triggered(monkeypatch):
    """Bug 1+2 coverage: if BOTH Bitget cancel paths return ok=False,
    we MUST NOT set status=cancelled (would orphan the live Bitget plan).
    Cond must stay as 'triggered' for next-bar retry."""
    cond = _short_descending_cond()
    store = _FakeStore(cond)
    _reset_fake_adapter()
    _FakeAdapter.cancel_regular_ok = False
    _FakeAdapter.cancel_plan_ok = False
    monkeypatch.setattr(watcher, "_store", store)
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeAdapter)
    _patch_mark(monkeypatch, 0.016)

    await watcher._maybe_replan(cond, 1_300)

    adapter = _FakeAdapter.instances[-1]
    # Both cancel paths were attempted
    assert len(adapter.cancelled_regular) == 1
    assert len(adapter.cancelled_plan) == 1
    # Status MUST NOT have moved to cancelled
    assert cond.status == "triggered", \
        f"CRITICAL: cancel failed but status={cond.status} (should stay triggered for retry)"
    assert not any(s == "cancelled" for s, _ in store.status_changes)
    # exchange_error event was logged so user can see the orphan risk
    assert any(e.kind == "exchange_error" for e in store.events), \
        f"exchange_error event missing; events: {[e.kind for e in store.events]}"
    # Replan still must NOT have run
    assert adapter.plan_submits == []


@pytest.mark.asyncio
async def test_line_broken_cancel_first_path_raises_then_fallback_succeeds(monkeypatch):
    """Bug 2 coverage: cancel_order raises (network blip). The fallback
    cancel_plan_order_any_type must STILL be attempted. If fallback
    succeeds, overall cancel is ok and we transition to cancelled."""
    cond = _short_descending_cond()
    store = _FakeStore(cond)
    _reset_fake_adapter()
    _FakeAdapter.cancel_regular_raise = RuntimeError("network blip")
    _FakeAdapter.cancel_plan_ok = True
    monkeypatch.setattr(watcher, "_store", store)
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeAdapter)
    _patch_mark(monkeypatch, 0.016)

    await watcher._maybe_replan(cond, 1_300)

    adapter = _FakeAdapter.instances[-1]
    # First path raised but fallback was attempted and succeeded
    assert len(adapter.cancelled_regular) == 1
    assert len(adapter.cancelled_plan) == 1, "fallback cancel must be attempted after regular raises"
    # Overall cancel succeeded → status transitioned to cancelled
    assert cond.status == "cancelled"
    assert any(e.kind == "line_broken" for e in store.events)


@pytest.mark.asyncio
async def test_mark_fetch_returns_none_aborts_maybe_replan(monkeypatch):
    """Bug 4 coverage: if _fetch_mark_price_strict returns None, the whole
    _maybe_replan must abort — NO replan, NO cancel. Prevents the BAS
    re-failure mode where a transient ticker outage re-triggers the
    original bug."""
    cond = _short_descending_cond()
    store = _FakeStore(cond)
    _reset_fake_adapter()
    monkeypatch.setattr(watcher, "_store", store)
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeAdapter)
    _patch_mark(monkeypatch, None)

    await watcher._maybe_replan(cond, 1_300)

    adapter_instances = _FakeAdapter.instances
    # NO cancel, NO replan — all cancel / submit lists must be empty
    for adapter in adapter_instances:
        assert adapter.cancelled_regular == [], "Bug 4: mark=None triggered a cancel"
        assert adapter.plan_submits == [], "Bug 4: mark=None triggered a replan"
    # Cond stays triggered, last_poll_ts updated to gate next tick
    assert cond.status == "triggered"
    assert cond.last_poll_ts == 1_300
    # A poll event was logged so user can see the skip
    assert any(e.kind == "poll" and "ABORTED" in (e.message or "") for e in store.events), \
        f"abort event missing; events: {[(e.kind, e.message) for e in store.events]}"


@pytest.mark.asyncio
async def test_line_broken_cas_lost_skips_reverse_spawn(monkeypatch):
    """Bug 5 coverage: if CAS from_status=triggered fails (someone else
    already transitioned), we must NOT spawn reverse. Simulate by pre-
    setting cond.status to 'cancelled' before _maybe_replan runs."""
    import dataclasses
    cond = _short_descending_cond()
    # OrderConfig is frozen — use dataclasses.replace to enable reverse.
    cond = dataclasses.replace(
        cond,
        order=dataclasses.replace(
            cond.order,
            reverse_enabled=True,
            reverse_entry_offset_pct=0.5,
            reverse_stop_offset_pct=0.3,
        ),
    )
    store = _FakeStore(cond)
    _reset_fake_adapter()
    monkeypatch.setattr(watcher, "_store", store)
    monkeypatch.setattr("server.execution.live_adapter.LiveExecutionAdapter", _FakeAdapter)
    _patch_mark(monkeypatch, 0.016)

    reverse_spawn_called = []
    async def _fake_reverse(src, reason):
        reverse_spawn_called.append((src.conditional_id, reason))
    monkeypatch.setattr(watcher, "_spawn_reverse_conditional", _fake_reverse)

    # Pre-flip status to simulate a racing path that already cancelled.
    cond.status = "cancelled"

    await watcher._maybe_replan(cond, 1_300)

    # CAS from_status='triggered' should have failed (status is already cancelled).
    # Reverse spawn MUST NOT have been called.
    assert reverse_spawn_called == [], \
        f"Bug 5: reverse spawned despite losing CAS: {reverse_spawn_called}"


# ─── Regression: reconcile must not nuke plan-type conds on 429 ─────
#
# Bug observed 2026-04-23 00:48 LA (user real money): Bitget returned
# 429 "too many requests" on get_pending_plan_orders while
# get_pending_orders (regular) succeeded. Our reconcile's gate at
# the time was `if not (ok_regular or ok_plan): skip` — i.e. "skip
# only if BOTH fail". That let plan-fetch failure + regular success
# fall through, union yielded only regular (non-plan) oids, and every
# plan-type cond was judged "not in pending" → CAS-cancelled even
# though Bitget still held them live (HYPE 275.22 @ 41.446, plus 2 more).
#
# The fix strengthens the gate to `if not (ok_regular AND ok_plan)`:
# any fetch failure now skips the mode, so reconcile retries next cycle.
@pytest.mark.asyncio
async def test_reconcile_skips_on_plan_pending_429(monkeypatch):
    """When Bitget returns 429 on the plan-pending fetch but the regular
    fetch succeeds, reconcile MUST NOT cancel plan-type conds. The union
    of an empty plan list and a small regular list looks identical to
    'order is gone' — a false positive that nukes real money orders."""
    # Build a triggered plan-type cond matching the real HYPE one.
    cond = _cond(
        conditional_id="cond_hype_rate_limit",
        symbol="HYPEUSDT",
        timeframe="4h",
        exchange_order_id="plan_oid_123",
        status="triggered",
    )

    class _FakeStoreList:
        def __init__(self, cond):
            self.cond = cond
            self.status_changes: list[tuple[str, str]] = []

        def list_all(self, status=None):
            if status is None or self.cond.status == status:
                return [self.cond]
            return []

        def get(self, cid):
            return self.cond if cid == self.cond.conditional_id else None

        def update(self, cond):
            self.cond = cond
            return cond

        def append_event(self, cid, event):
            self.cond.events.append(event)
            return self.cond

        def set_status_if(self, cid, *, from_status, to_status, reason=""):
            if self.cond.status != from_status:
                return None
            self.cond.status = to_status
            self.status_changes.append((to_status, reason))
            return self.cond

    store = _FakeStoreList(cond)
    monkeypatch.setattr(watcher, "_store", store)

    class _RateLimitedAdapter:
        def has_api_keys(self):
            return True

        async def get_pending_orders(self, mode):
            # Regular (non-plan) succeeds but returns empty — realistic
            # when user has no limit orders, only plan-type triggers.
            return []

        async def get_pending_plan_orders(self, mode, plan_type="normal_plan"):
            # Plan fetch hits 429 → adapter raises RuntimeError (matches
            # what live_adapter.get_pending_plan_orders does on non-00000).
            raise RuntimeError(
                'plan pending fetch failed: {"code":"429","msg":"too many requests"}'
            )

        async def _bitget_request(self, method, path, *, mode=None, params=None, body=None):
            # position/all-position is the only other bitget call reconcile
            # makes when one fetch fails — respond empty.
            if "position/all-position" in path:
                return {"code": "00000", "data": []}
            return {"code": "00000", "data": []}

        @staticmethod
        def _as_rows(data):
            if not data: return []
            if isinstance(data, list): return data
            return data.get("list") or data.get("entrustedList") or []

    monkeypatch.setattr(
        "server.execution.live_adapter.LiveExecutionAdapter",
        _RateLimitedAdapter,
    )

    reverse_calls: list[tuple[str, str]] = []
    async def _fake_reverse(src, reason):
        reverse_calls.append((src.conditional_id, reason))
    monkeypatch.setattr(watcher, "_spawn_reverse_conditional", _fake_reverse)

    await watcher._reconcile_against_bitget()

    # REGRESSION: the cond MUST still be 'triggered'. The 429 on plan
    # fetch is a temporary API blip; we'd rather wait 30s and retry
    # than nuke a live real-money order.
    assert cond.status == "triggered", \
        f"429 on plan-pending nuked plan cond: status={cond.status} changes={store.status_changes}"
    assert store.status_changes == [], \
        f"CAS-cancel fired on 429: {store.status_changes}"
    # Reverse spawn must also not have been called.
    assert reverse_calls == [], \
        f"reverse spawned on 429 false-positive: {reverse_calls}"


@pytest.mark.asyncio
async def test_reconcile_skips_on_regular_pending_failure(monkeypatch):
    """Mirror of the 429-on-plan test: if the regular pending fetch
    fails but plan succeeds, we still skip the mode. Yes, the user
    currently has no non-plan conds, but the gate must be symmetric —
    future types (e.g. limit-entry manual orders) would hit the same
    trap in reverse."""
    cond = _cond(
        conditional_id="cond_hype_regular_fail",
        symbol="HYPEUSDT",
        exchange_order_id="plan_oid_456",
        status="triggered",
    )

    class _FakeStoreList:
        def __init__(self, cond):
            self.cond = cond
            self.status_changes: list[tuple[str, str]] = []

        def list_all(self, status=None):
            if status is None or self.cond.status == status:
                return [self.cond]
            return []

        def get(self, cid):
            return self.cond if cid == self.cond.conditional_id else None

        def update(self, cond):
            self.cond = cond
            return cond

        def append_event(self, cid, event):
            self.cond.events.append(event)
            return self.cond

        def set_status_if(self, cid, *, from_status, to_status, reason=""):
            if self.cond.status != from_status:
                return None
            self.cond.status = to_status
            self.status_changes.append((to_status, reason))
            return self.cond

    store = _FakeStoreList(cond)
    monkeypatch.setattr(watcher, "_store", store)

    class _AsymFailAdapter:
        def has_api_keys(self):
            return True

        async def get_pending_orders(self, mode):
            raise RuntimeError("regular fetch transient failure")

        async def get_pending_plan_orders(self, mode, plan_type="normal_plan"):
            # Even though plan fetch returned a list that doesn't contain
            # our oid, we should NOT cancel — because regular failed and
            # the order could in theory be a regular one (not our case
            # today, but the invariant must hold).
            return [{"orderId": "some_other_plan"}]

        async def _bitget_request(self, method, path, *, mode=None, params=None, body=None):
            return {"code": "00000", "data": []}

        @staticmethod
        def _as_rows(data):
            if not data: return []
            if isinstance(data, list): return data
            return data.get("list") or data.get("entrustedList") or []

    monkeypatch.setattr(
        "server.execution.live_adapter.LiveExecutionAdapter",
        _AsymFailAdapter,
    )

    reverse_calls: list[tuple[str, str]] = []
    async def _fake_reverse(src, reason):
        reverse_calls.append((src.conditional_id, reason))
    monkeypatch.setattr(watcher, "_spawn_reverse_conditional", _fake_reverse)

    await watcher._reconcile_against_bitget()

    assert cond.status == "triggered", \
        f"regular-fetch-failure falsely cancelled cond: status={cond.status} changes={store.status_changes}"
    assert store.status_changes == []
    assert reverse_calls == []


class _FakeStoreList:
    """Minimal store stub matching the reconcile path's contract."""
    def __init__(self, cond):
        self.cond = cond
        self.status_changes: list[tuple[str, str]] = []

    def list_all(self, status=None):
        if status is None or self.cond.status == status:
            return [self.cond]
        return []

    def get(self, cid):
        return self.cond if cid == self.cond.conditional_id else None

    def update(self, cond):
        self.cond = cond
        return cond

    def append_event(self, cid, event):
        self.cond.events.append(event)
        return self.cond

    def set_status_if(self, cid, *, from_status, to_status, reason=""):
        if self.cond.status != from_status:
            return None
        self.cond.status = to_status
        self.status_changes.append((to_status, reason))
        return self.cond


class _HistoryAdapter:
    """Configurable fake adapter used for reconcile scenario tests.

    Tell it what pending-plan / pending-regular / history / positions
    should return; everything else is canned."""
    def __init__(self, *,
                 plan_pending_rows: list[dict] | Exception = None,
                 regular_pending_rows: list[dict] | Exception = None,
                 history_rows: list[dict] | Exception = None,
                 history_code: str = "00000",
                 positions_rows: list[dict] | None = None):
        self.plan_pending_rows = plan_pending_rows if plan_pending_rows is not None else []
        self.regular_pending_rows = regular_pending_rows if regular_pending_rows is not None else []
        self.history_rows = history_rows if history_rows is not None else []
        self.history_code = history_code
        self.positions_rows = positions_rows or []

    def has_api_keys(self): return True

    async def get_pending_orders(self, mode, symbol=None):
        if isinstance(self.regular_pending_rows, Exception):
            raise self.regular_pending_rows
        return self.regular_pending_rows

    async def get_pending_plan_orders(self, mode, plan_type="normal_plan", symbol=None):
        if isinstance(self.plan_pending_rows, Exception):
            raise self.plan_pending_rows
        return self.plan_pending_rows

    async def _bitget_request(self, method, path, *, mode=None, params=None, body=None):
        if "position/all-position" in path:
            return {"code": "00000", "data": self.positions_rows}
        if "orders-plan-history" in path:
            if isinstance(self.history_rows, Exception):
                raise self.history_rows
            return {"code": self.history_code, "data": {"entrustedList": self.history_rows}}
        return {"code": "00000", "data": []}

    @staticmethod
    def _as_rows(data):
        if not data: return []
        if isinstance(data, list): return data
        return data.get("list") or data.get("entrustedList") or []


# ─── The three contract tests the user explicitly asked for ─────────

@pytest.mark.asyncio
async def test_reconcile_pending_429_but_history_shows_live(monkeypatch):
    """Scenario 1 (the exact bug from 2026-04-23 00:48 LA):
    pending fetch fails with 429, but if we query history Bitget shows
    our oid is still live. Cond MUST stay triggered — no false cancel."""
    cond = _cond(
        conditional_id="cond_429_but_live",
        symbol="HYPEUSDT",
        exchange_order_id="oid_alive_on_bitget",
        status="triggered",
    )
    store = _FakeStoreList(cond)
    monkeypatch.setattr(watcher, "_store", store)

    adapter = _HistoryAdapter(
        plan_pending_rows=RuntimeError(
            'plan pending fetch failed: {"code":"429","msg":"too many requests"}'
        ),
        regular_pending_rows=[],
        history_rows=[{
            "orderId": "oid_alive_on_bitget",
            "planStatus": "live",
            "symbol": "HYPEUSDT",
        }],
    )
    monkeypatch.setattr(
        "server.execution.live_adapter.LiveExecutionAdapter",
        lambda: adapter,
    )

    reverse_calls: list = []
    async def _fake_reverse(src, reason): reverse_calls.append((src.conditional_id, reason))
    monkeypatch.setattr(watcher, "_spawn_reverse_conditional", _fake_reverse)

    await watcher._reconcile_against_bitget()

    assert cond.status == "triggered", \
        f"Test 1 FAIL: 429 + history-live produced status={cond.status}. This is the exact bug we fixed; it must NEVER regress."
    assert store.status_changes == []
    assert reverse_calls == []


@pytest.mark.asyncio
async def test_reconcile_pending_absent_history_cancelled(monkeypatch):
    """Scenario 2: pending does not have the oid AND history explicitly
    shows state=cancelled. This is the ONLY legitimate local cancel path
    under the new design. Cond MUST transition to cancelled + reverse spawn."""
    cond = _cond(
        conditional_id="cond_truly_cancelled",
        symbol="HYPEUSDT",
        exchange_order_id="oid_really_gone",
        status="triggered",
    )
    store = _FakeStoreList(cond)
    monkeypatch.setattr(watcher, "_store", store)

    adapter = _HistoryAdapter(
        plan_pending_rows=[{"orderId": "other_order_unrelated"}],
        regular_pending_rows=[],
        history_rows=[{
            "orderId": "oid_really_gone",
            "planStatus": "cancelled",
            "symbol": "HYPEUSDT",
        }],
    )
    monkeypatch.setattr(
        "server.execution.live_adapter.LiveExecutionAdapter",
        lambda: adapter,
    )

    reverse_calls: list = []
    async def _fake_reverse(src, reason): reverse_calls.append((src.conditional_id, reason))
    monkeypatch.setattr(watcher, "_spawn_reverse_conditional", _fake_reverse)

    await watcher._reconcile_against_bitget()

    assert cond.status == "cancelled", \
        f"Test 2 FAIL: affirmative history cancel was ignored, status={cond.status}"
    assert any("history-confirmed cancel" in reason for _, reason in store.status_changes), \
        f"expected history-confirmed cancel reason, got {store.status_changes}"


@pytest.mark.asyncio
async def test_reconcile_pending_absent_history_also_absent_stays_triggered(monkeypatch):
    """Scenario 3: pending does not have the oid AND history also does not
    contain it (either oldthan 48h window, or API returned empty). This is
    the UNKNOWN state — we must NEVER transition. Stay triggered and retry
    next cycle."""
    cond = _cond(
        conditional_id="cond_unknown",
        symbol="HYPEUSDT",
        exchange_order_id="oid_in_limbo",
        status="triggered",
    )
    store = _FakeStoreList(cond)
    monkeypatch.setattr(watcher, "_store", store)

    adapter = _HistoryAdapter(
        plan_pending_rows=[{"orderId": "unrelated"}],
        regular_pending_rows=[],
        history_rows=[],   # history successfully queried but contains no row for our oid
    )
    monkeypatch.setattr(
        "server.execution.live_adapter.LiveExecutionAdapter",
        lambda: adapter,
    )

    reverse_calls: list = []
    async def _fake_reverse(src, reason): reverse_calls.append((src.conditional_id, reason))
    monkeypatch.setattr(watcher, "_spawn_reverse_conditional", _fake_reverse)

    await watcher._reconcile_against_bitget()

    assert cond.status == "triggered", \
        f"Test 3 FAIL: UNKNOWN classification falsely transitioned cond, status={cond.status}"
    assert store.status_changes == []
    assert reverse_calls == []


@pytest.mark.asyncio
async def test_reconcile_history_api_error_stays_triggered(monkeypatch):
    """Bonus (should also hold): pending absent, history API itself fails
    (429/5xx/timeout). Cond MUST stay triggered. This locks the invariant
    that we NEVER act on API failures, only on affirmative responses."""
    cond = _cond(
        conditional_id="cond_history_error",
        symbol="HYPEUSDT",
        exchange_order_id="oid_api_fail",
        status="triggered",
    )
    store = _FakeStoreList(cond)
    monkeypatch.setattr(watcher, "_store", store)

    adapter = _HistoryAdapter(
        plan_pending_rows=[],
        regular_pending_rows=[],
        history_rows=RuntimeError("history endpoint 503"),
    )
    monkeypatch.setattr(
        "server.execution.live_adapter.LiveExecutionAdapter",
        lambda: adapter,
    )

    reverse_calls: list = []
    async def _fake_reverse(src, reason): reverse_calls.append((src.conditional_id, reason))
    monkeypatch.setattr(watcher, "_spawn_reverse_conditional", _fake_reverse)

    await watcher._reconcile_against_bitget()

    assert cond.status == "triggered"
    assert store.status_changes == []


@pytest.mark.asyncio
async def test_reconcile_history_shows_filled_transitions_to_filled(monkeypatch):
    """Positive FILLED path: pending absent (order filled), history shows
    state=triggered (Bitget's terminology for a fired plan). Cond should
    transition to filled. No reverse spawn (those are for cancel paths)."""
    cond = _cond(
        conditional_id="cond_filled_via_history",
        symbol="HYPEUSDT",
        exchange_order_id="oid_filled",
        status="triggered",
    )
    store = _FakeStoreList(cond)
    monkeypatch.setattr(watcher, "_store", store)

    adapter = _HistoryAdapter(
        plan_pending_rows=[],
        regular_pending_rows=[],
        history_rows=[{
            "orderId": "oid_filled",
            "planStatus": "triggered",   # Bitget's "fired" state
            "symbol": "HYPEUSDT",
        }],
        positions_rows=[],   # position not yet visible, but history is enough
    )
    monkeypatch.setattr(
        "server.execution.live_adapter.LiveExecutionAdapter",
        lambda: adapter,
    )

    reverse_calls: list = []
    async def _fake_reverse(src, reason): reverse_calls.append((src.conditional_id, reason))
    monkeypatch.setattr(watcher, "_spawn_reverse_conditional", _fake_reverse)

    # Also stub the manual-trailing register so we don't need chart data.
    monkeypatch.setattr(
        watcher, "_register_manual_trailing_if_position_open",
        lambda *a, **kw: None,
    )

    await watcher._reconcile_against_bitget()

    assert cond.status == "filled", \
        f"Test bonus FAIL: history-filled should transition status=filled, got {cond.status}"
    assert reverse_calls == [], "reverse must NOT fire on filled"
