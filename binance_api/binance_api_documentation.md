# Binance API Documentation

## Endpoints

| Market | Endpoint | Description |
|--------|----------|-------------|
| Futures (USDS-margined) | `https://fapi.binance.com/fapi/v1/klines` | Kline/candlestick data |
| Spot | `https://api.binance.com/api/v3/klines` | Kline/candlestick data |

## Klines Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `symbol` | Yes | e.g. `ENSOUSDT` |
| `interval` | Yes | `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, `1M` |
| `startTime` | No | Start time in ms. If omitted, returns most recent candles |
| `endTime` | No | End time in ms |
| `limit` | No | Default 500, max 1500 |

## Rate Limits

### Weight Budget

| API | Budget |
|-----|--------|
| Futures | 2,400 weight / minute / IP |
| Spot | 6,000 weight / minute / IP |

### Klines Weight Tiers (Futures)

Weight is determined by the `limit` parameter, not by interval or symbol:

| `limit` range | Weight |
|---------------|--------|
| 1–99 | 1 |
| 100–499 | 2 |
| 500–999 | 5 |
| 1000–1500 | 10 |

### Practical Throughput (Futures, limit=1500)

- 240 requests/min (2400 / 10)
- 360,000 candles/min
- More than enough for any single-symbol historical download

### Rate Limit Enforcement

- Limits are per IP, not per API key
- HTTP 429 = rate limited, must back off
- HTTP 418 = IP banned (2 min to 3 days, escalating for repeat offenders)
- Response headers include `X-MBX-USED-WEIGHT-(intervalNum)(intervalLetter)` showing current usage

## Data Freshness

- **REST API**: The last candle in the response is the **current unclosed candle**, updating live with each trade. No documented latency — effectively real-time on each request.
- **WebSocket** (`<symbol>@kline_<interval>`): Pushes updates every **250ms**. Each payload includes `"x": true/false` indicating whether the candle is closed.
- For historical analysis, discard the last candle if you only want closed/finalized candles.

## Response Format

Each kline is an array of 12 elements:

| Index | Field |
|-------|-------|
| 0 | Open time (ms) |
| 1 | Open |
| 2 | High |
| 3 | Low |
| 4 | Close |
| 5 | Volume |
| 6 | Close time (ms) |
| 7 | Quote asset volume |
| 8 | Number of trades |
| 9 | Taker buy base asset volume |
| 10 | Taker buy quote asset volume |
| 11 | Ignore |

## Sources

- [Futures Klines Endpoint](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data)
- [Spot API Limits](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/limits)
- [Spot Market Data Endpoints](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)
