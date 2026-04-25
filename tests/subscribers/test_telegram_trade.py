"""Tests for trade-event classifier + Telegram formatter
(server/subscribers/telegram_trade.py).

Coverage matrix:
  classify_bitget_ws_order:
    - normal_plan + status=filled + tradeSide=open  → entry_filled
    - profit_loss + status=filled + pnl > 0          → tp_hit
    - profit_loss + status=filled + pnl < 0          → sl_hit
    - profit_loss + planType=pos_loss (no pnl)       → sl_hit
    - profit_loss + planType=pos_profit (no pnl)     → tp_hit
    - status=cancelled                                → cancelled
    - tradeSide=close                                 → position_closed
    - status=live (non-terminal)                      → None (no event)
    - missing symbol                                  → None
    - unknown tradeSide                               → None

  dedup:
    - same (source, oid, type) within 60 s            → second emit blocked
    - different oid                                   → both emit
    - different event_type                            → both emit

  formatter:
    - entry_filled with all fields                    → expected lines
    - sl_hit with pnl                                 → red emoji + negative pnl
    - tp_hit with pnl                                 → bullseye + positive pnl
    - cancelled minimal payload                       → no crash, basic message
    - paper source                                    → [paper] tag
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.subscribers.telegram_trade import (
    TradeEvent,
    classify_bitget_ws_order,
    classify_watcher_event,
    format_trade_event,
    _should_emit,
    _reset_dedup_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_dedup():
    _reset_dedup_for_tests()
    yield
    _reset_dedup_for_tests()


# ─── classify_bitget_ws_order ────────────────────────────────────────

def test_classify_entry_filled_long():
    row = {
        "symbol": "TSLAUSDT",
        "orderId": "999",
        "clientOid": "replan_cond_abc123_1234",
        "side": "buy",
        "tradeSide": "open",
        "planType": "normal_plan",
        "status": "filled",
        "fillPrice": "377.5",
        "fillSize": "0.02",
        "triggerPrice": "319",
    }
    te = classify_bitget_ws_order(row)
    assert te is not None
    assert te.event_type == "entry_filled"
    assert te.direction == "long"
    assert te.symbol == "TSLAUSDT"
    assert te.fill_price == 377.5
    assert te.size == 0.02
    assert te.exchange_order_id == "999"
    assert te.source == "bitget"


def test_classify_entry_filled_short():
    row = {
        "symbol": "XAUUSDT", "orderId": "1", "side": "sell",
        "tradeSide": "open", "planType": "normal_plan",
        "status": "filled", "fillPrice": "4750",
    }
    te = classify_bitget_ws_order(row)
    assert te.direction == "short"
    assert te.event_type == "entry_filled"


def test_classify_tp_hit_via_pnl_positive():
    row = {
        "symbol": "ETHUSDT", "orderId": "2", "side": "sell",
        "tradeSide": "close", "planType": "profit_loss",
        "status": "filled", "fillPrice": "3000", "totalProfits": "12.5",
    }
    te = classify_bitget_ws_order(row)
    assert te.event_type == "tp_hit"
    assert te.pnl_usd == 12.5


def test_classify_sl_hit_via_pnl_negative():
    row = {
        "symbol": "ETHUSDT", "orderId": "3", "side": "sell",
        "tradeSide": "close", "planType": "profit_loss",
        "status": "filled", "fillPrice": "2800", "totalProfits": "-15.0",
    }
    te = classify_bitget_ws_order(row)
    assert te.event_type == "sl_hit"
    assert te.pnl_usd == -15.0


def test_classify_sl_hit_via_plantype_pos_loss():
    """When pnl missing, fall back to plan_type variant."""
    row = {
        "symbol": "BTCUSDT", "orderId": "4", "side": "sell",
        "tradeSide": "close", "planType": "pos_loss",
        "status": "filled", "fillPrice": "50000",
    }
    te = classify_bitget_ws_order(row)
    assert te.event_type == "sl_hit"


def test_classify_tp_hit_via_plantype_pos_profit():
    row = {
        "symbol": "BTCUSDT", "orderId": "5", "side": "sell",
        "tradeSide": "close", "planType": "pos_profit",
        "status": "filled", "fillPrice": "60000",
    }
    te = classify_bitget_ws_order(row)
    assert te.event_type == "tp_hit"


def test_classify_cancelled():
    row = {
        "symbol": "ZECUSDT", "orderId": "6", "side": "buy",
        "tradeSide": "open", "planType": "normal_plan",
        "status": "cancelled",
    }
    te = classify_bitget_ws_order(row)
    assert te.event_type == "cancelled"


def test_classify_close_without_planmatch_is_position_closed():
    row = {
        "symbol": "SOLUSDT", "orderId": "7", "side": "sell",
        "tradeSide": "close", "planType": "normal_plan",
        "status": "filled", "fillPrice": "150.0",
    }
    te = classify_bitget_ws_order(row)
    assert te.event_type == "position_closed"


def test_classify_live_status_returns_none():
    row = {
        "symbol": "TSLAUSDT", "orderId": "8", "status": "live",
    }
    assert classify_bitget_ws_order(row) is None


def test_classify_partial_fill_returns_none():
    row = {
        "symbol": "TSLAUSDT", "orderId": "9", "status": "partial-fill",
    }
    assert classify_bitget_ws_order(row) is None


def test_classify_missing_symbol_returns_none():
    row = {"orderId": "10", "status": "filled", "tradeSide": "open"}
    assert classify_bitget_ws_order(row) is None


def test_classify_unknown_tradeside_returns_none():
    row = {
        "symbol": "X", "orderId": "11", "status": "filled",
        "tradeSide": "weirdo", "planType": "normal_plan",
    }
    assert classify_bitget_ws_order(row) is None


def test_classify_non_dict_returns_none():
    assert classify_bitget_ws_order(None) is None
    assert classify_bitget_ws_order("string") is None
    assert classify_bitget_ws_order([1, 2, 3]) is None


# ─── Dedup ───────────────────────────────────────────────────────────

def test_dedup_blocks_second_emit_within_ttl():
    assert _should_emit("bitget", "999", "entry_filled") is True
    assert _should_emit("bitget", "999", "entry_filled") is False  # blocked


def test_dedup_allows_different_oid():
    assert _should_emit("bitget", "111", "entry_filled") is True
    assert _should_emit("bitget", "222", "entry_filled") is True


def test_dedup_allows_different_event_type_for_same_oid():
    """Same orderId but different event_type → both emit (entry then close)."""
    assert _should_emit("bitget", "X", "entry_filled") is True
    assert _should_emit("bitget", "X", "tp_hit") is True
    assert _should_emit("bitget", "X", "tp_hit") is False  # second tp_hit blocked


def test_dedup_allows_different_source_for_same_oid():
    assert _should_emit("bitget", "X", "entry_filled") is True
    assert _should_emit("hyperliquid", "X", "entry_filled") is True


def test_dedup_handles_no_oid():
    assert _should_emit("paper", None, "entry_filled") is True
    assert _should_emit("paper", None, "entry_filled") is False  # blocked even w/o oid


# ─── Formatter ────────────────────────────────────────────────────────

def test_format_entry_filled_full_payload():
    p = TradeEvent(
        event_type="entry_filled", source="bitget", symbol="TSLAUSDT",
        direction="long", fill_price=377.5, trigger_price=319.0,
        entry_price=None, size=0.02, pnl_usd=None, pnl_pct=None,
        reason="plan-order trigger fired", exchange_order_id="999",
        client_order_id=None, conditional_id=None, timestamp=1000,
    ).to_payload()
    msg = format_trade_event(p)
    assert "🟢" in msg                    # entry emoji
    assert "入场成交" in msg
    assert "LONG" in msg
    assert "TSLAUSDT" in msg
    assert "377.50" in msg or "377.5" in msg
    assert "319" in msg                   # trigger displayed when ≠ fill
    assert "0.02" in msg
    assert "999" in msg                    # oid


def test_format_sl_hit_uses_red_and_shows_pnl():
    p = TradeEvent(
        event_type="sl_hit", source="bitget", symbol="ETHUSDT",
        direction="long", fill_price=2800.0, trigger_price=None,
        entry_price=2900.0, size=0.5, pnl_usd=-50.0, pnl_pct=-1.7,
        reason="profit_loss fired", exchange_order_id="3",
        client_order_id=None, conditional_id=None, timestamp=1000,
    ).to_payload()
    msg = format_trade_event(p)
    assert "🔴" in msg
    assert "止损触发" in msg
    assert "-$50.00" in msg or "-50.00" in msg
    assert "2900" in msg                  # entry shown


def test_format_tp_hit_uses_bullseye_and_positive_pnl():
    p = TradeEvent(
        event_type="tp_hit", source="bitget", symbol="BTCUSDT",
        direction="long", fill_price=60000.0, trigger_price=None,
        entry_price=58000.0, size=0.01, pnl_usd=20.0, pnl_pct=3.4,
        reason="profit_loss fired", exchange_order_id="5",
        client_order_id=None, conditional_id=None, timestamp=1000,
    ).to_payload()
    msg = format_trade_event(p)
    assert "🎯" in msg
    assert "止盈触发" in msg
    assert "+$20.00" in msg
    assert "+3.40%" in msg


def test_format_cancelled_minimal_payload():
    p = TradeEvent(
        event_type="cancelled", source="bitget", symbol="ZECUSDT",
        direction="long", fill_price=None, trigger_price=317.0,
        entry_price=None, size=None, pnl_usd=None, pnl_pct=None,
        reason="user/system cancelled", exchange_order_id="6",
        client_order_id=None, conditional_id=None, timestamp=1000,
    ).to_payload()
    msg = format_trade_event(p)
    assert "↩" in msg or "↩️" in msg
    assert "挂单取消" in msg
    assert "ZECUSDT" in msg
    assert "317" in msg
    # Must NOT crash on missing fill_price / size / pnl
    assert "None" not in msg


def test_format_paper_source_shows_tag():
    p = TradeEvent(
        event_type="entry_filled", source="paper", symbol="HYPEUSDT",
        direction="long", fill_price=42.0, trigger_price=None,
        entry_price=None, size=10.0, pnl_usd=None, pnl_pct=None,
        reason="paper sim", exchange_order_id=None,
        client_order_id=None, conditional_id=None, timestamp=1000,
    ).to_payload()
    msg = format_trade_event(p)
    assert "[paper]" in msg


def test_format_bitget_source_no_tag_in_message():
    """Default source bitget — no [bitget] tag (would be redundant)."""
    p = TradeEvent(
        event_type="entry_filled", source="bitget", symbol="TSLAUSDT",
        direction="long", fill_price=377.5, trigger_price=None,
        entry_price=None, size=0.02, pnl_usd=None, pnl_pct=None,
        reason="x", exchange_order_id="x", client_order_id=None,
        conditional_id=None, timestamp=1000,
    ).to_payload()
    msg = format_trade_event(p)
    assert "[bitget]" not in msg


# ─── classify_watcher_event ──────────────────────────────────────────

def test_watcher_cancelled_event():
    payload = {
        "kind": "cancelled", "symbol": "ZECUSDT", "direction": "long",
        "exchange_order_id": "1234", "exchange_mode": "live",
        "message": "user manually cancelled", "price": 317.5,
    }
    te = classify_watcher_event(payload)
    assert te is not None
    assert te.event_type == "cancelled"
    assert te.source == "bitget"
    assert te.symbol == "ZECUSDT"
    assert te.exchange_order_id == "1234"
    assert "manually cancelled" in te.reason


def test_watcher_exchange_acked_becomes_entry_filled():
    payload = {
        "kind": "exchange_acked", "symbol": "TSLAUSDT", "direction": "long",
        "exchange_order_id": "999", "exchange_mode": "live",
        "message": "fill captured", "price": 377.8,
    }
    te = classify_watcher_event(payload)
    assert te.event_type == "entry_filled"
    assert te.fill_price == 377.8


def test_watcher_line_broken_becomes_cancelled():
    payload = {
        "kind": "line_broken", "symbol": "BTCUSDT",
        "exchange_order_id": "x", "exchange_mode": "live",
        "message": "line broken at $50000",
    }
    te = classify_watcher_event(payload)
    assert te.event_type == "cancelled"
    assert "line broken" in te.reason


def test_watcher_paper_mode_marked_as_paper_source():
    payload = {
        "kind": "cancelled", "symbol": "TESTUSDT",
        "exchange_order_id": "1", "exchange_mode": "paper",
    }
    te = classify_watcher_event(payload)
    assert te.source == "paper"


def test_watcher_unhandled_kinds_return_none():
    """exchange_submitted / exchange_error / triggered already covered by
    legacy on_conditional_event — don't double-fire."""
    for kind in ["exchange_submitted", "exchange_error", "triggered", "breakout"]:
        assert classify_watcher_event({"kind": kind, "symbol": "X"}) is None, kind


def test_watcher_non_dict_returns_none():
    assert classify_watcher_event(None) is None
    assert classify_watcher_event("string") is None


def test_format_handles_html_special_chars_in_reason():
    """Reasons with < or > shouldn't break Telegram HTML parsing."""
    p = TradeEvent(
        event_type="cancelled", source="bitget", symbol="TEST",
        direction="long", fill_price=None, trigger_price=None,
        entry_price=None, size=None, pnl_usd=None, pnl_pct=None,
        reason="price drift > 5%, cancel <auto>",
        exchange_order_id="x", client_order_id=None,
        conditional_id=None, timestamp=1000,
    ).to_payload()
    msg = format_trade_event(p)
    assert "&lt;auto&gt;" in msg            # escaped
    assert "<auto>" not in msg              # raw not present
