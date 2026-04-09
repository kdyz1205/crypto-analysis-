from __future__ import annotations

from decimal import Decimal


def _as_decimal(value: str | int | float | None, default: str = "0") -> Decimal:
    raw = default if value in (None, "") else str(value)
    return Decimal(raw)


def _bitget_tick_size(row: dict) -> str:
    price_place = int(str(row.get("pricePlace") or "0"))
    price_end_step = _as_decimal(row.get("priceEndStep"), "1")
    tick = price_end_step / (Decimal(10) ** price_place)
    normalized = format(tick.normalize(), "f")
    return normalized if "." in normalized else normalized


def bitget_contracts_to_symbol_map(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol.endswith("USDT"):
            continue

        tick_sz = _bitget_tick_size(row)
        price_precision = int(str(row.get("pricePlace") or "0"))
        result[symbol] = {
            "instId": symbol,
            "symbol": symbol,
            "tickSz": tick_sz,
            "lotSz": str(row.get("sizeMultiplier") or "0.001"),
            "pricePrecision": price_precision,
            "volumePrecision": int(str(row.get("volumePlace") or "0")),
            "minTradeNum": row.get("minTradeNum"),
            "state": row.get("symbolStatus", "normal"),
            "productType": row.get("productType", "usdt-futures"),
        }
    return result


def bitget_tickers_to_volume_symbols(rows: list[dict], top_n: int = 20) -> list[str]:
    pairs: list[tuple[str, float]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol.endswith("USDT"):
            continue
        try:
            volume = float(row.get("usdtVolume") or row.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        pairs.append((symbol, volume))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in pairs[:top_n]]


def bitget_candles_to_records(rows: list[list[str]]) -> list[list[float | int]]:
    records: list[list[float | int]] = []
    for row in rows:
        if len(row) < 7:
            continue
        ts_ms = int(row[0])
        open_price = float(row[1])
        high_price = float(row[2])
        low_price = float(row[3])
        close_price = float(row[4])
        base_volume = float(row[5])
        quote_volume = float(row[6])
        records.append(
            [
                ts_ms,
                open_price,
                high_price,
                low_price,
                close_price,
                base_volume,
                ts_ms,
                quote_volume,
                0,
                0.0,
                0.0,
                0,
            ]
        )
    return records
