import json

import pytest

from server.routers import trades


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_manual_trade_history_user_label_supports_quality_and_legacy_fields(monkeypatch, tmp_path):
    ml_file = tmp_path / "user_drawings_ml.jsonl"
    labels_file = tmp_path / "user_drawing_labels.jsonl"
    cond_file = tmp_path / "conditional_orders.json"

    _write_jsonl(
        ml_file,
        [
            {
                "event": "position_closed_from_drawing",
                "ts": 30,
                "dt": "2026-04-23T10:00:00Z",
                "symbol": "BTCUSDT",
                "timeframe": "4h",
                "side": "long",
                "manual_line_id": "line-quality",
                "features_at_close": {
                    "entry_price": 100.0,
                    "close_price": 110.0,
                    "margin_used": 20.0,
                },
                "pnl_usd": 10.0,
                "pnl_pct": 0.1,
                "close_reason": "tp",
            },
            {
                "event": "position_closed_from_drawing",
                "ts": 20,
                "dt": "2026-04-23T09:00:00Z",
                "symbol": "ETHUSDT",
                "timeframe": "1h",
                "side": "short",
                "manual_line_id": "line-legacy-win",
                "features_at_close": {
                    "entry_price": 200.0,
                    "close_price": 180.0,
                    "margin_used": 30.0,
                },
                "pnl_usd": 8.0,
                "pnl_pct": 0.04,
                "close_reason": "tp",
            },
            {
                "event": "position_closed_from_drawing",
                "ts": 10,
                "dt": "2026-04-23T08:00:00Z",
                "symbol": "SOLUSDT",
                "timeframe": "15m",
                "side": "long",
                "manual_line_id": "line-legacy-label",
                "features_at_close": {
                    "entry_price": 50.0,
                    "close_price": 49.0,
                    "margin_used": 12.0,
                },
                "pnl_usd": -2.0,
                "pnl_pct": -0.04,
                "close_reason": "sl_or_tp",
            },
        ],
    )
    _write_jsonl(
        labels_file,
        [
            {"event": "user_labeled_drawing", "manual_line_id": "line-quality", "quality": "good"},
            {"event": "user_drawing_label", "manual_line_id": "line-legacy-win", "label_trade_win": 1},
            {"event": "user_drawing_label", "manual_line_id": "line-legacy-label", "label": "watch"},
        ],
    )
    cond_file.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(trades, "_ML_FILE", ml_file)
    monkeypatch.setattr(trades, "_LABELS_FILE", labels_file)
    monkeypatch.setattr(trades, "_COND_FILE", cond_file)

    payload = await trades.api_manual_trade_history(limit=50, symbol=None)

    rows = {row["manual_line_id"]: row for row in payload["rows"]}
    assert rows["line-quality"]["user_label"] == "good"
    assert rows["line-legacy-win"]["user_label"] == "win"
    assert rows["line-legacy-label"]["user_label"] == "watch"
