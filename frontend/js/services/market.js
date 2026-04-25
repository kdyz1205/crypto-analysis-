// frontend/js/services/market.js
import { fetchJson } from '../util/fetch.js';
import { publish } from '../util/events.js';
import * as ohlcvCache from './ohlcv_cache.js';

export const getSymbols = (includeExtended = false) =>
  fetchJson(`/api/symbols${includeExtended ? '?include_extended=true' : ''}`);

// Sortable-picker payload — richer than getSymbols(). Each row carries
// {symbol, last_price, change24h, volume_usdt}. Backend caches 10 min
// so calling this on every dropdown open is cheap.
export const getSymbolsExtended = (topN = 200) =>
  fetchJson(`/api/symbols/extended?top_n=${topN}`);

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

async function _fetchOhlcvFromServer(symbol, interval, days, endTime, historyMode) {
  const params = new URLSearchParams({
    symbol, interval, days: String(days), history_mode: historyMode,
    include_overlays: 'false',
  });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/ohlcv?${params}`, { noCache: true });
}

export const getOhlcv = async (symbol, interval, days = 30, endTime = null, historyMode = 'fast_window') => {
  const key = `${symbol}|${interval}|${days}|${historyMode}`;
  // Skip all caches during replay (endTime set) — replay needs exact bars.
  if (endTime) {
    return _fetchOhlcvFromServer(symbol, interval, days, endTime, historyMode);
  }

  // Tier 1 — hot in-memory hit (sub-ms, survives TF flips).
  const hit = _ohlcvCache.get(key);
  if (hit && Date.now() - hit.ts < _cacheTtlMs(interval)) {
    return hit.data;
  }

  // Tier 2 — IndexedDB persistent hit (survives F5 / server restart).
  //   If cached bars cover the requested range and the last bar is still
  //   fresh enough for this TF, return IMMEDIATELY and fire a small delta
  //   refresh in the background. Otherwise fall through to full fetch.
  //
  // 2026-04-23: biggest "F5 feels fast" win. Historical bars are immutable
  // so there's no reason to re-download 2,936 bars every time.
  const idbHit = await ohlcvCache.getCached(symbol, interval).catch(() => null);
  if (idbHit && ohlcvCache.isFreshEnough(idbHit, interval)) {
    const data = {
      candles: idbHit.bars,
      volume: idbHit.volume || [],
      pricePrecision: idbHit.pricePrecision,
      fromIndexedDB: true,
    };
    _ohlcvCache.set(key, { ts: Date.now(), data });
    // Fire a background delta refresh. Don't await — user sees cached now.
    _backgroundRefresh(symbol, interval, days, historyMode, key).catch((err) => {
      console.warn('[ohlcv] bg refresh failed', err);
    });
    return data;
  }

  // Tier 3 — server fetch. Populate both caches on success.
  const data = await _fetchOhlcvFromServer(symbol, interval, days, endTime, historyMode);
  _ohlcvCache.set(key, { ts: Date.now(), data });
  if (_ohlcvCache.size > 24) {
    const oldest = [..._ohlcvCache.entries()].sort((a, b) => a[1].ts - b[1].ts)[0];
    if (oldest) _ohlcvCache.delete(oldest[0]);
  }
  const bars = Array.isArray(data?.candles) ? data.candles : [];
  if (bars.length > 0) {
    ohlcvCache.storeBars(symbol, interval, bars, {
      volume: data.volume || [],
      pricePrecision: data.pricePrecision,
    }).catch(() => {});
  }
  return data;
};

/**
 * Background-refresh: fetch the last ~7 days of bars, merge into IDB,
 * then publish `ohlcv.delta` so subscribers (chart.js) can append the
 * newest bars without a full redraw.
 */
async function _backgroundRefresh(symbol, interval, days, historyMode, hotKey) {
  // We only need the tail for delta — 7 days covers any TF's new bars
  // comfortably without asking the server to ship ALL history again.
  const deltaDays = 7;
  try {
    const data = await _fetchOhlcvFromServer(symbol, interval, deltaDays, null, historyMode);
    const bars = Array.isArray(data?.candles) ? data.candles : [];
    if (bars.length === 0) return;
    const { bars: mergedBars, volume: mergedVol } = await ohlcvCache.mergeBars(
      symbol, interval, bars, data.volume || [], { pricePrecision: data.pricePrecision },
    );
    const mergedData = {
      candles: mergedBars,
      volume: mergedVol,
      pricePrecision: data.pricePrecision,
      fromIndexedDB: true,
    };
    _ohlcvCache.set(hotKey, { ts: Date.now(), data: mergedData });
    publish('ohlcv.delta', {
      symbol, interval,
      newest: bars[bars.length - 1],
      count: bars.length,
      data: mergedData,
    });
  } catch (err) {
    console.warn('[ohlcv] background refresh error', err);
  }
}

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
  // P0 2026-04-23: noCache per P12 — this endpoint includes pattern
  // detection state that the user reads when deciding to enter; 30s
  // stale data on a TF switch means looking at outdated structure.
  return fetchJson(`/api/chart?${params}`, { noCache: true });
};

export const getTopVolume = (n = 20) => fetchJson(`/api/top-volume?n=${n}`);

export const getDataInfo = (symbol, interval) =>
  fetchJson(`/api/data-info?symbol=${encodeURIComponent(symbol)}&interval=${interval}`);
