// frontend/js/workbench/overlays/trade_markers_overlay.js
//
// 2026-04-25: OKX/Bitget-style fill markers on the chart.
//
// Each filled cond + each closed-position event produces a marker:
//
//   ▲ green below bar  = LONG entry filled
//   ▼ red   above bar  = SHORT entry filled
//   ▼ gray  above bar  = LONG exit (positions closed from below)
//   ▲ gray  below bar  = SHORT exit
//   🔴 red  above bar  = SL hit
//   🎯 amber above bar = TP hit
//
// The marker timestamp is the actual fill_time, so:
//   - On 1m chart, the marker pinpoints the exact minute
//   - On 1d chart, the marker maps to the day-bar containing that minute
//
// This file ONLY does setMarkers; it doesn't change candle data or
// drawings. Safe to plug into any chart that already has a candleSeries.

import { fetchJson } from '../../util/fetch.js';

let _candleSeries = null;
let _currentSymbol = null;
let _refreshTimer = null;

const REFRESH_MS = 60_000;        // refetch every 60s in case new fills arrive
const LOOKBACK_DAYS = 30;          // keep marker count manageable

export function initTradeMarkers(candleSeries) {
  _candleSeries = candleSeries;
}

/**
 * Pull fills for `symbol` and apply as markers. Replaces any prior set.
 * Caller decides when to call (typically on chart.load.succeeded).
 */
export async function refreshTradeMarkers(symbol) {
  if (!_candleSeries || !symbol) return;
  _currentSymbol = String(symbol).toUpperCase();
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
  const markers = fills.map(_fillToMarker).filter(Boolean);
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
    if (_currentSymbol) void refreshTradeMarkers(_currentSymbol);
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

function _fillToMarker(f) {
  if (!f || typeof f.time !== 'number') return null;
  const type = (f.type || '').toLowerCase();
  const side = (f.side || '').toLowerCase();
  const closeReason = (f.close_reason || '').toLowerCase();

  // Build a label that fits OKX/Bitget style: emoji + abbreviation
  // (no full sentences — chart space is precious).
  if (type === 'entry') {
    const isLong = side === 'long';
    return {
      time: f.time,
      position: isLong ? 'belowBar' : 'aboveBar',
      color: isLong ? '#26a69a' : '#ef5350',
      shape: isLong ? 'arrowUp' : 'arrowDown',
      text: _entryLabel(side, f.price, f.qty),
      // size: 1, // default
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

function _entryLabel(side, price, qty) {
  const sideLabel = side === 'long' ? '多' : '空';
  const qtyTxt = (qty != null && Number.isFinite(qty)) ? ` x${_qtyFmt(qty)}` : '';
  return `${sideLabel} ${_priceFmt(price)}${qtyTxt}`;
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

function _qtyFmt(q) {
  const v = Number(q);
  if (!Number.isFinite(v)) return '';
  return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
}

function _pnlSuffix(pnl) {
  if (pnl == null) return '';
  const v = Number(pnl);
  if (!Number.isFinite(v)) return '';
  const sign = v >= 0 ? '+' : '';
  return ` (${sign}$${v.toFixed(2)})`;
}
