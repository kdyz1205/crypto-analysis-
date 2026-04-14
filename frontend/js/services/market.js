// frontend/js/services/market.js
import { fetchJson } from '../util/fetch.js';

export const getSymbols = (includeExtended = false) =>
  fetchJson(`/api/symbols${includeExtended ? '?include_extended=true' : ''}`);

export const getSymbolInfo = (symbol) =>
  fetchJson(`/api/symbol-info?symbol=${encodeURIComponent(symbol)}`);

// ─────────────────────────────────────────────────────────────
// Frontend OHLCV result cache.
// When the user flicks TFs (4h → 1h → 4h), we don't need to pay
// the backend round-trip every time. A tiny in-memory cache keyed
// by (symbol, interval, days, historyMode) serves the repeat calls
// instantly and lets the chart swap in under 50ms.
// TTL is short: fast TFs (1m/3m/5m) expire quickly so live bars
// stay fresh; slow TFs get longer TTL.
// ─────────────────────────────────────────────────────────────
const _ohlcvCache = new Map();   // key -> { ts, data }

function _cacheTtlMs(interval) {
  const fast = { '1m': 3000, '3m': 5000, '5m': 10000, '15m': 15000, '30m': 30000 };
  return fast[interval] || 60000;   // slow TFs cached 60s
}

/**
 * Peek the cache without triggering a fetch. Returns cached data or
 * null. Used by chart.js for optimistic instant-swap on TF switch.
 */
export function peekOhlcvCache(symbol, interval, days = 30, historyMode = 'fast_window') {
  const key = `${symbol}|${interval}|${days}|${historyMode}`;
  const hit = _ohlcvCache.get(key);
  if (!hit) return null;
  if (Date.now() - hit.ts > _cacheTtlMs(interval)) {
    _ohlcvCache.delete(key);
    return null;
  }
  return hit.data;
}

export const getOhlcv = async (symbol, interval, days = 30, endTime = null, historyMode = 'fast_window') => {
  const key = `${symbol}|${interval}|${days}|${historyMode}`;
  // Skip cache when running a replay (endTime set)
  if (!endTime) {
    const hit = _ohlcvCache.get(key);
    if (hit && Date.now() - hit.ts < _cacheTtlMs(interval)) {
      return hit.data;
    }
  }
  const params = new URLSearchParams({
    symbol, interval, days: String(days), history_mode: historyMode,
    include_overlays: 'false',   // compute MA/BB client-side — server skips loops + shrinks payload
  });
  if (endTime) params.set('end_time', endTime);
  // noCache on fetch layer: we manage our own freshness via the cache
  // above; the fetch-layer 30s cache would hold stale bar counts.
  const data = await fetchJson(`/api/ohlcv?${params}`, { noCache: true });
  if (!endTime) {
    _ohlcvCache.set(key, { ts: Date.now(), data });
    // Cap cache size
    if (_ohlcvCache.size > 24) {
      const oldest = [..._ohlcvCache.entries()].sort((a, b) => a[1].ts - b[1].ts)[0];
      if (oldest) _ohlcvCache.delete(oldest[0]);
    }
  }
  return data;
};

/**
 * Lazy-load older bars for a chart.
 * Used when the user scrolls to the left edge — we fetch another
 * ~500 bars ending strictly before the earliest currently-loaded bar
 * and prepend them to the series.
 */
export const getOhlcvBackfill = (symbol, interval, beforeTs, bars = 500) => {
  const params = new URLSearchParams({
    symbol, interval,
    before_ts: String(beforeTs),
    bars: String(bars),
  });
  return fetchJson(`/api/ohlcv/backfill?${params}`, { noCache: true });
};

export const getChart = (symbol, interval, days = 365, endTime = null, historyMode = 'fast_window') => {
  const params = new URLSearchParams({ symbol, interval, days: String(days), history_mode: historyMode });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/chart?${params}`);
};

export const getTopVolume = (n = 20) => fetchJson(`/api/top-volume?n=${n}`);

export const getDataInfo = (symbol, interval) =>
  fetchJson(`/api/data-info?symbol=${encodeURIComponent(symbol)}&interval=${interval}`);
