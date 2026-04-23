// frontend/js/workbench/ma_overlay.js — MA ribbon + BB overlays
//
// 2026-04-23: MA Ribbon refactored to be config-driven. The ribbon used
// to be hardcoded 5/8/21/55 with fixed colors. Now each `ma_ribbon`
// indicator instance owns a `config.lines` array:
//   lines: [{ period: int, type: 'sma'|'ema', color: '#hex', width: 1..3 }]
// so the user can add/remove lines, change periods, pick colors, and
// flip SMA/EMA from the indicator-panel ⚙ gear without a code change.
//
// Two render paths coexist:
//   1. drawMARibbon(chart, candles, config, instanceId)  ← preferred,
//      used by indicator_controller.applyIndicators. Computes from
//      candles directly, tracks series per instanceId so a re-render
//      can clean up its own series without nuking BB/Wyckoff etc.
//   2. drawMAOverlays(chart, overlays, _candleTimes)     ← legacy
//      fallback, triggers only if applyIndicators throws before the
//      chart has lines. Uses the default 5/8/21/55 visual.

import { $ } from '../util/dom.js';

// ─── Default ribbon (used by drawMAOverlays legacy path + first-load
//     seed for ma_ribbon indicator instance). Matches what the chart
//     showed before the refactor so nothing looks different on upgrade.
export const DEFAULT_RIBBON_LINES = [
  { period: 5,  type: 'sma', color: '#ffeb3b', width: 1 },
  { period: 8,  type: 'sma', color: '#ff9800', width: 1 },
  { period: 21, type: 'ema', color: '#2196f3', width: 2 },
  { period: 55, type: 'sma', color: '#e91e63', width: 2 },
];

// ─── Preset library (users pick a named set from the gear dropdown).
//     Findings from /project_ma_ribbon_*.md memory files drive which
//     numbers are offered: 4h sweep winner [8,21,55,89] and the real
//     best [5,13,34,89] are worth a one-click switch.
export const RIBBON_PRESETS = [
  {
    id: 'default',
    label: '默认 5/8/21/55',
    lines: [
      { period: 5,  type: 'sma', color: '#ffeb3b', width: 1 },
      { period: 8,  type: 'sma', color: '#ff9800', width: 1 },
      { period: 21, type: 'ema', color: '#2196f3', width: 2 },
      { period: 55, type: 'sma', color: '#e91e63', width: 2 },
    ],
  },
  {
    id: 'sweep_4h_winner',
    label: '4h 冠军 8/21/55/89',
    lines: [
      { period: 8,  type: 'sma', color: '#ffeb3b', width: 1 },
      { period: 21, type: 'sma', color: '#ff9800', width: 1 },
      { period: 55, type: 'ema', color: '#2196f3', width: 2 },
      { period: 89, type: 'sma', color: '#e91e63', width: 2 },
    ],
  },
  {
    id: 'fibonacci',
    label: 'Fibonacci 5/13/34/89',
    lines: [
      { period: 5,  type: 'sma', color: '#ffeb3b', width: 1 },
      { period: 13, type: 'sma', color: '#ff9800', width: 1 },
      { period: 34, type: 'ema', color: '#2196f3', width: 2 },
      { period: 89, type: 'sma', color: '#e91e63', width: 2 },
    ],
  },
  {
    id: 'scalp_short',
    label: '短线 3/5/8/13',
    lines: [
      { period: 3,  type: 'ema', color: '#ffeb3b', width: 1 },
      { period: 5,  type: 'ema', color: '#ff9800', width: 1 },
      { period: 8,  type: 'ema', color: '#2196f3', width: 1 },
      { period: 13, type: 'ema', color: '#e91e63', width: 2 },
    ],
  },
  {
    id: 'classic_pair',
    label: '经典 50/200',
    lines: [
      { period: 50,  type: 'sma', color: '#ffb300', width: 2 },
      { period: 200, type: 'sma', color: '#e91e63', width: 2 },
    ],
  },
];

// Legacy hardcoded colors — used ONLY by drawMAOverlays fallback path.
const MA_COLORS = {
  ma5:      { color: '#ffeb3b', width: 1, title: 'MA5' },
  ma8:      { color: '#ff9800', width: 1, title: 'MA8' },
  ema21:    { color: '#2196f3', width: 2, title: 'EMA21' },
  ma55:     { color: '#e91e63', width: 2, title: 'MA55' },
  bb_upper: { color: 'rgba(156,39,176,0.5)', width: 1, title: 'BB Up', lineStyle: 2 },
  bb_lower: { color: 'rgba(156,39,176,0.5)', width: 1, title: 'BB Lo', lineStyle: 2 },
};

// Legacy-path series store (by key)
const seriesRefs = {};
// New config-driven store: instanceId -> { lineIdx -> series }
const _ribbonSeries = new Map();
let visible = true;

export function initMAOverlays(chart) {
  return {};
}

/** New config-driven MA ribbon renderer.
 *
 * @param chart       LightweightCharts instance
 * @param candles     [{time, open, high, low, close, volume}...]
 * @param config      { lines: [{period, type, color, width}...] }
 * @param instanceId  indicator id from indicator_controller (so we can
 *                    clean up only this instance's series on re-render)
 */
export function drawMARibbon(chart, candles, config, instanceId = 'ma_ribbon') {
  if (!chart || !Array.isArray(candles) || candles.length === 0) return;
  const lines = Array.isArray(config?.lines) && config.lines.length > 0
    ? config.lines
    : DEFAULT_RIBBON_LINES;

  // Clear this instance's previous series (period / color / etc may have
  // changed, simpler to rebuild than reconcile).
  clearRibbonInstance(chart, instanceId);

  const closes = candles.map((c) => Number(c.close));
  const times = candles.map((c) => c.time);
  const store = {};

  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i] || {};
    const period = Math.max(1, Math.floor(Number(ln.period) || 0));
    const type = (ln.type === 'ema') ? 'ema' : 'sma';
    const color = ln.color || '#ffeb3b';
    const width = Math.max(1, Math.min(4, Math.floor(Number(ln.width) || 1)));
    if (period < 1 || closes.length < period) continue;
    const values = type === 'ema' ? _ema(closes, period) : _sma(closes, period);
    const data = values
      .map((v, idx) => (v == null ? null : { time: times[idx], value: v }))
      .filter((x) => x != null);

    const series = chart.addLineSeries({
      color,
      lineWidth: width,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      title: `${type.toUpperCase()}${period}`,
      autoscaleInfoProvider: () => null,
    });
    try { series.setData(data); }
    catch (err) { console.warn('[ma_ribbon] setData failed', period, err); }
    series.applyOptions({ visible });
    store[i] = series;
  }
  _ribbonSeries.set(instanceId, store);
}

export function clearRibbonInstance(chart, instanceId) {
  const store = _ribbonSeries.get(instanceId);
  if (!store) return;
  for (const s of Object.values(store)) {
    try { chart.removeSeries(s); } catch {}
  }
  _ribbonSeries.delete(instanceId);
}

export function setRibbonVisible(visibleFlag, instanceId) {
  const store = _ribbonSeries.get(instanceId);
  if (!store) return;
  for (const s of Object.values(store)) {
    try { s.applyOptions({ visible: !!visibleFlag }); } catch {}
  }
}

/** LEGACY: hardcoded 5/8/21/55 + BB overlay path. Kept only for the
 *  failure-recovery branch in chart.js where applyIndicators() threw
 *  before the ribbon loaded. New code should use drawMARibbon(). */
export function drawMAOverlays(chart, overlays, _candleTimes) {
  if (!chart || !overlays) return;
  for (const [key, cfg] of Object.entries(MA_COLORS)) {
    const arr = overlays[key];
    if (!Array.isArray(arr) || arr.length === 0) continue;
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
        crosshairMarkerVisible: false,
        title: cfg.title,
        autoscaleInfoProvider: () => null,
      });
    }
    try { seriesRefs[key].setData(data); }
    catch (err) { console.warn('[ma] setData failed', key, err); }
    seriesRefs[key].applyOptions({ visible });
  }
}

export function toggleMAOverlays() {
  visible = !visible;
  for (const s of Object.values(seriesRefs)) {
    try { s.applyOptions({ visible }); } catch {}
  }
  // Also flip every config-driven ribbon instance.
  for (const store of _ribbonSeries.values()) {
    for (const s of Object.values(store)) {
      try { s.applyOptions({ visible }); } catch {}
    }
  }
  return visible;
}

export function setMAOverlaysVisible(v) {
  visible = !!v;
  for (const s of Object.values(seriesRefs)) {
    try { s.applyOptions({ visible }); } catch {}
  }
  for (const store of _ribbonSeries.values()) {
    for (const s of Object.values(store)) {
      try { s.applyOptions({ visible }); } catch {}
    }
  }
}

export function clearMAOverlays(chart) {
  for (const s of Object.values(seriesRefs)) {
    try { chart.removeSeries(s); } catch {}
  }
  Object.keys(seriesRefs).forEach((k) => delete seriesRefs[k]);
  for (const [id] of _ribbonSeries) clearRibbonInstance(chart, id);
}

// ─────────────────────────────────────────────────────────────
// Client-side overlay computation (still used for BB since BB
// shares a single overlays object in chart.js).
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
  const middle = new Array(n).fill(null);
  const lower = new Array(n).fill(null);
  if (n < period) return { upper, middle, lower };
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
    middle[i] = mean;
    lower[i] = mean - stdMult * std;
  }
  return { upper, middle, lower };
}

/**
 * Compute MA/EMA/BB arrays client-side for the *legacy* overlay shape
 * that chart.js passes along for BB rendering. MA keys are kept for
 * backwards-compat with drawMAOverlays fallback path; bb_middle added
 * 2026-04-23 so indicator_controller.renderBB stops silently failing.
 */
export function computeOverlaysFromCandles(candles) {
  if (!Array.isArray(candles) || candles.length < 55) return {};
  const closes = candles.map((c) => Number(c.close));
  const times = candles.map((c) => c.time);
  const ma5 = _sma(closes, 5);
  const ma8 = _sma(closes, 8);
  const ema21 = _ema(closes, 21);
  const ma55 = _sma(closes, 55);
  const { upper: bbUp, middle: bbMid, lower: bbLo } = _bb(closes, 21, 2.5);
  const _pack = (arr) =>
    arr.map((v, i) => (v == null ? null : { time: times[i], value: v }))
       .filter((x) => x != null);
  return {
    ma5: _pack(ma5),
    ma8: _pack(ma8),
    ema21: _pack(ema21),
    ma55: _pack(ma55),
    bb_upper: _pack(bbUp),
    bb_middle: _pack(bbMid),
    bb_lower: _pack(bbLo),
  };
}
