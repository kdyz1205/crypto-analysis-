// frontend/js/workbench/ma_overlay.js — MA ribbon + BB overlays

import { $ } from '../util/dom.js';

const MA_COLORS = {
  ma5:      { color: '#ffeb3b', width: 1, title: 'MA5' },
  ma8:      { color: '#ff9800', width: 1, title: 'MA8' },
  ema21:    { color: '#2196f3', width: 2, title: 'EMA21' },
  ma55:     { color: '#e91e63', width: 2, title: 'MA55' },
  bb_upper: { color: 'rgba(156,39,176,0.5)', width: 1, title: 'BB Up', lineStyle: 2 },
  bb_lower: { color: 'rgba(156,39,176,0.5)', width: 1, title: 'BB Lo', lineStyle: 2 },
};

const seriesRefs = {};
let visible = true;

export function initMAOverlays(chart) {
  // Empty init, series created lazily on first draw
  return {};
}

export function drawMAOverlays(chart, overlays, _candleTimes) {
  if (!chart || !overlays) return;
  for (const [key, cfg] of Object.entries(MA_COLORS)) {
    const arr = overlays[key];
    if (!Array.isArray(arr) || arr.length === 0) continue;

    // API returns [{time, value}, ...] — pass through directly
    const data = arr
      .map((pt) => ({
        time: typeof pt.time === 'string'
          ? Math.floor(new Date(pt.time).getTime() / 1000)
          : pt.time,
        value: Number(pt.value),
      }))
      .filter((x) => x.time != null && isFinite(x.value));

    if (!seriesRefs[key]) {
      seriesRefs[key] = chart.addLineSeries({
        color: cfg.color,
        lineWidth: cfg.width,
        lineStyle: cfg.lineStyle ?? 0,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,  // no dot on hover
        title: cfg.title,
        autoscaleInfoProvider: () => null,
      });
    }
    try { seriesRefs[key].setData(data); }
    catch (err) { console.warn('[ma] setData failed', key, err); }
    seriesRefs[key].applyOptions({ visible });
  }
  // MA series drawn silently
}

export function toggleMAOverlays() {
  visible = !visible;
  for (const s of Object.values(seriesRefs)) {
    try { s.applyOptions({ visible }); } catch {}
  }
  return visible;
}

export function clearMAOverlays(chart) {
  for (const s of Object.values(seriesRefs)) {
    try { chart.removeSeries(s); } catch {}
  }
  Object.keys(seriesRefs).forEach((k) => delete seriesRefs[k]);
}

// ─────────────────────────────────────────────────────────────
// Client-side overlay computation.
// The server used to ship MA/EMA/BB arrays with every /api/ohlcv
// response. For a 500-bar chart that's 6 extra 500-element arrays
// inflating the JSON payload. Computing on client is O(n) JS and
// takes <5ms for 500 bars — cheaper than the bytes over the wire.
// ─────────────────────────────────────────────────────────────

function _sma(values, period) {
  const n = values.length;
  const out = new Array(n).fill(null);
  if (n < period) return out;
  let sum = 0;
  for (let i = 0; i < period; i++) sum += values[i];
  out[period - 1] = sum / period;
  for (let i = period; i < n; i++) {
    sum += values[i] - values[i - period];
    out[i] = sum / period;
  }
  return out;
}

function _ema(values, period) {
  const n = values.length;
  const out = new Array(n).fill(null);
  if (n < period) return out;
  const k = 2 / (period + 1);
  let prev = 0;
  for (let i = 0; i < period; i++) prev += values[i];
  prev /= period;
  out[period - 1] = prev;
  for (let i = period; i < n; i++) {
    prev = values[i] * k + prev * (1 - k);
    out[i] = prev;
  }
  return out;
}

function _bb(values, period, stdMult) {
  const n = values.length;
  const upper = new Array(n).fill(null);
  const lower = new Array(n).fill(null);
  if (n < period) return { upper, lower };
  for (let i = period - 1; i < n; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) sum += values[j];
    const mean = sum / period;
    let vsq = 0;
    for (let j = i - period + 1; j <= i; j++) {
      vsq += (values[j] - mean) * (values[j] - mean);
    }
    const std = Math.sqrt(vsq / period);
    upper[i] = mean + stdMult * std;
    lower[i] = mean - stdMult * std;
  }
  return { upper, lower };
}

/**
 * Compute MA/EMA/BB arrays client-side from candle data.
 * Returns overlays object in the same format the server used to send:
 * { ma5: [{time, value}...], ma8: [...], ema21: [...], ma55: [...],
 *   bb_upper: [...], bb_lower: [...] }
 */
export function computeOverlaysFromCandles(candles) {
  if (!Array.isArray(candles) || candles.length < 55) return {};
  const closes = candles.map((c) => Number(c.close));
  const times = candles.map((c) => c.time);
  const ma5 = _sma(closes, 5);
  const ma8 = _sma(closes, 8);
  const ema21 = _ema(closes, 21);
  const ma55 = _sma(closes, 55);
  const { upper: bbUp, lower: bbLo } = _bb(closes, 21, 2.5);
  const _pack = (arr) =>
    arr.map((v, i) => (v == null ? null : { time: times[i], value: v }))
       .filter((x) => x != null);
  return {
    ma5: _pack(ma5),
    ma8: _pack(ma8),
    ema21: _pack(ema21),
    ma55: _pack(ma55),
    bb_upper: _pack(bbUp),
    bb_lower: _pack(bbLo),
  };
}
