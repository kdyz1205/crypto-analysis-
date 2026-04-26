// frontend/js/workbench/overlays/trade_markers_overlay.js
//
// 2026-04-25: OKX/Bitget-style fill markers on the chart.
//
//   ▲ green below bar  = LONG entry filled
//   ▼ red   above bar  = SHORT entry filled
//   🔴 red  above bar  = SL hit
//   🎯 amber above bar = TP hit
//   ⬛ gray  above bar  = manual/other close
//
// 2026-04-25 (later, per user request): **Per-bar dedup.** "一次多空就好了
// — 在一个时间框架里面，你多次的买入就会显示一次。只有打开到最小的时间
// 框架，你才会看到每次。" Multiple entries on the SAME side within the
// SAME chart bar collapse to ONE marker. Switching to a smaller TF
// (1m / 5m) splits them apart again because each fill now lands in
// its own bucket.
//
// Implementation: bucket entries by `(floor(time / TF_seconds), side)`.
// Keep one representative fill per bucket (latest within the bucket).
// Exits stay per-fill — SL / TP / manual closes are individual story-
// points the trader needs to see.

import { fetchJson } from '../../util/fetch.js';

let _candleSeries = null;
let _currentSymbol = null;
let _currentInterval = null;     // needed to compute per-bar buckets
let _refreshTimer = null;

const REFRESH_MS = 60_000;        // refetch every 60s in case new fills arrive
const LOOKBACK_DAYS = 30;          // keep marker count manageable

// TF → seconds. Used for floor-to-bar bucketing of entry markers.
// UTC-aligned; lightweight-charts snaps marker.time to the bar
// containing it, so per-second precision past the bucket boundary
// doesn't matter.
const TF_SECONDS = {
  '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
  '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200,
  '1d': 86400, '1w': 604800,
};

export function initTradeMarkers(candleSeries) {
  _candleSeries = candleSeries;
}

/**
 * Pull fills for `symbol` and apply as markers. Replaces any prior set.
 * `interval` is required for per-bar bucketing — same fill list, different
 * TF → different marker count.
 */
export async function refreshTradeMarkers(symbol, interval) {
  if (!_candleSeries || !symbol) return;
  _currentSymbol = String(symbol).toUpperCase();
  if (interval) _currentInterval = interval;
  let payload;
  try {
    payload = await fetchJson(
      `/api/trades/fills?symbol=${encodeURIComponent(_currentSymbol)}&days=${LOOKBACK_DAYS}`,
      { timeout: 8000, noCache: true },
    );
  } catch (err) {
    console.warn('[trade_markers] fetch failed:', err?.message || err);
    return;
  }
  // Race guard: if symbol switched while fetch was in flight, drop.
  if (String(_currentSymbol).toUpperCase() !== String(symbol).toUpperCase()) return;
  const fills = Array.isArray(payload?.fills) ? payload.fills : [];
  const intervalSec = TF_SECONDS[_currentInterval] || 3600;  // 1h fallback
  const markers = _bucketEntriesPerBar(fills, intervalSec)
    .map(_fillToMarker)
    .filter(Boolean);
  // lightweight-charts requires markers sorted by time ascending
  markers.sort((a, b) => a.time - b.time);
  try {
    _candleSeries.setMarkers(markers);
  } catch (err) {
    console.warn('[trade_markers] setMarkers failed:', err);
  }
}

export function startTradeMarkersAutoRefresh() {
  stopTradeMarkersAutoRefresh();
  _refreshTimer = setInterval(() => {
    if (_currentSymbol) void refreshTradeMarkers(_currentSymbol, _currentInterval);
  }, REFRESH_MS);
}

export function stopTradeMarkersAutoRefresh() {
  if (_refreshTimer) { clearInterval(_refreshTimer); _refreshTimer = null; }
}

export function clearTradeMarkers() {
  if (!_candleSeries) return;
  try { _candleSeries.setMarkers([]); } catch {}
}

// ─── Internals ──────────────────────────────────────────────────────────

/**
 * Collapse entry fills that share `(bar_bucket, side)` into a single
 * representative fill. Exits and other types pass through unchanged.
 *
 * Why floor-bucketing works even when bars have non-UTC-midnight opens
 * (Bitget 1d opens at UTC 16:00): lightweight-charts snaps marker.time
 * to the bar CONTAINING that time, so the bucket key only needs to be
 * a stable function of the time within the desired granularity. Two
 * fills with `floor(t / 86400)` matching share a calendar day, and
 * lightweight-charts snaps both into the same daily bar regardless of
 * the bar's open offset.
 */
function _bucketEntriesPerBar(fills, intervalSec) {
  const entryBuckets = new Map();   // key `${bucket}|${side}` → latest fill
  const passthrough = [];

  for (const f of fills) {
    if (!f || typeof f.time !== 'number') continue;
    const type = (f.type || '').toLowerCase();
    if (type !== 'entry') {
      passthrough.push(f);
      continue;
    }
    const side = (f.side || '').toLowerCase();
    const bucket = Math.floor(f.time / intervalSec) * intervalSec;
    const key = `${bucket}|${side}`;
    const existing = entryBuckets.get(key);
    if (!existing || existing.time < f.time) {
      entryBuckets.set(key, f);
    }
  }

  return [...entryBuckets.values(), ...passthrough];
}

function _fillToMarker(f) {
  if (!f || typeof f.time !== 'number') return null;
  const type = (f.type || '').toLowerCase();
  const side = (f.side || '').toLowerCase();
  const closeReason = (f.close_reason || '').toLowerCase();

  // 2026-04-25: User asked for TradingView/OKX/Bitget-style — just the
  // 多/空 character, no price, no qty. Arrow shape + color + position
  // already conveys side; text is purely a Chinese-language hint.
  if (type === 'entry') {
    const isLong = side === 'long';
    return {
      time: f.time,
      position: isLong ? 'belowBar' : 'aboveBar',
      color: isLong ? '#26a69a' : '#ef5350',
      shape: isLong ? 'arrowUp' : 'arrowDown',
      text: isLong ? '多' : '空',
    };
  }

  if (type === 'exit') {
    // Distinguish SL / TP / manual via close_reason
    if (closeReason.includes('sl') || closeReason.includes('stop')) {
      return {
        time: f.time,
        position: 'aboveBar',
        color: '#ff1744',
        shape: 'circle',
        text: `SL ${_priceFmt(f.price)}${_pnlSuffix(f.pnl_usd)}`,
      };
    }
    if (closeReason.includes('tp') || closeReason.includes('profit')) {
      return {
        time: f.time,
        position: 'aboveBar',
        color: '#fbbf24',
        shape: 'circle',
        text: `TP ${_priceFmt(f.price)}${_pnlSuffix(f.pnl_usd)}`,
      };
    }
    // Manual / other close
    return {
      time: f.time,
      position: 'aboveBar',
      color: '#9e9e9e',
      shape: 'square',
      text: `平仓 ${_priceFmt(f.price)}${_pnlSuffix(f.pnl_usd)}`,
    };
  }

  return null;
}

function _priceFmt(p) {
  if (p == null) return '';
  const v = Number(p);
  if (!Number.isFinite(v)) return '';
  if (v >= 1000) return v.toFixed(2);
  if (v >= 10) return v.toFixed(3);
  if (v >= 1) return v.toFixed(4);
  return v.toFixed(6);
}

function _pnlSuffix(pnl) {
  if (pnl == null) return '';
  const v = Number(pnl);
  if (!Number.isFinite(v)) return '';
  const sign = v >= 0 ? '+' : '';
  return ` (${sign}$${v.toFixed(2)})`;
}
