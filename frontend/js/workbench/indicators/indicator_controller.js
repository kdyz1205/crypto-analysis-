// frontend/js/workbench/indicators/indicator_controller.js
//
// Central indicator state + render dispatch. Owns:
//   - the list of indicators the user wants visible (persisted)
//   - the LightweightCharts series instances for each indicator
//   - the apply() call invoked by chart.js after every candle load
//
// Schema (localStorage `v2.indicators.v1`):
//   [{ id: string, type: string, name: string, visible: boolean,
//      config: {...} }]
//
// Types registered here:
//   'ma_ribbon'  — delegates to ma_overlay.js (existing)
//   'bb'         — Bollinger Bands (computed client-side)
//   'wyckoff'    — delegates to wyckoff_overlay.js (existing)
//   'rsi'        — RSI(14) subpanel
//   'macd'       — MACD(12,26,9) subpanel
//   'volume_ma'  — SMA(20) line over the volume series
//   'atr'        — ATR(14) subpanel

import {
  computeRSI, computeMACD, computeVolumeMA, computeATR,
  factorRSIOversold, factorRSIOverbought, factorVolumeSurge,
  factorMACDBullCross, factorMACDBearCross,
} from './indicator_math.js';

const LS_KEY = 'v2.indicators.v1';

/** Default list — matches the old hardcoded overlays so nothing visually
 *  disappears on first boot. New users can disable from the panel. */
const DEFAULT_INDICATORS = [
  { id: 'ma_ribbon', type: 'ma_ribbon', name: 'MA Ribbon (5/8/21/55)', visible: true, config: {} },
  { id: 'bb',        type: 'bb',        name: 'Bollinger Bands (20, 2)', visible: true, config: { period: 20, stddev: 2 } },
  // Optional / heavier — off by default
  { id: 'wyckoff',   type: 'wyckoff',   name: 'Wyckoff (vol-ratio)', visible: false, config: {} },
  { id: 'rsi',       type: 'rsi',       name: 'RSI (14)', visible: false, config: { period: 14 } },
  { id: 'macd',      type: 'macd',      name: 'MACD (12, 26, 9)', visible: false, config: { fast: 12, slow: 26, signal: 9 } },
  { id: 'volume_ma', type: 'volume_ma', name: 'Volume MA (20)', visible: false, config: { period: 20 } },
  { id: 'atr',       type: 'atr',       name: 'ATR (14)', visible: false, config: { period: 14 } },
];

/** Catalog of indicators available to add (category → list). Used by the
 *  "+ 添加指标" dropdown. */
export const INDICATOR_CATALOG = [
  {
    category: '趋势 Trend',
    items: [
      { type: 'ma_ribbon', name: 'MA Ribbon (5/8/21/55)' },
    ],
  },
  {
    category: '波动率 Volatility',
    items: [
      { type: 'bb',  name: 'Bollinger Bands (20, 2)', defaultConfig: { period: 20, stddev: 2 } },
      { type: 'atr', name: 'ATR (14)',                defaultConfig: { period: 14 } },
    ],
  },
  {
    category: '动量 Momentum',
    items: [
      { type: 'rsi',  name: 'RSI (14)',              defaultConfig: { period: 14 } },
      { type: 'macd', name: 'MACD (12, 26, 9)',      defaultConfig: { fast: 12, slow: 26, signal: 9 } },
    ],
  },
  {
    category: '成交量 Volume',
    items: [
      { type: 'volume_ma', name: 'Volume MA (20)',   defaultConfig: { period: 20 } },
    ],
  },
  {
    category: '结构 Structure',
    items: [
      { type: 'wyckoff', name: 'Wyckoff (vol-ratio)' },
    ],
  },
  // ─── Factor library — binary signals rendered as candle markers ───
  // Mirrors the server-side factors in data/factors/core/. These are
  // boolean per-bar conditions ("RSI < 30", "volume > 2× 20SMA", ...).
  // Each factor added by the user places triangles on candles where the
  // condition fires.
  {
    category: '因子 Factor',
    items: [
      { type: 'factor_rsi_oversold',   name: 'RSI Oversold (<30)',    defaultConfig: { period: 14, threshold: 30 } },
      { type: 'factor_rsi_overbought', name: 'RSI Overbought (>70)',  defaultConfig: { period: 14, threshold: 70 } },
      { type: 'factor_volume_surge',   name: 'Volume Surge (>2×SMA20)', defaultConfig: { period: 20, multiple: 2.0 } },
      { type: 'factor_macd_bull',      name: 'MACD Bull Cross',       defaultConfig: { fast: 12, slow: 26, signal: 9 } },
      { type: 'factor_macd_bear',      name: 'MACD Bear Cross',       defaultConfig: { fast: 12, slow: 26, signal: 9 } },
    ],
  },
];

let _indicators = loadIndicators();
const _series = new Map();   // id -> array of LightweightCharts series refs
const _listeners = new Set();

function loadIndicators() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        // Merge with defaults: keep user's visibility prefs, add any new
        // defaults that weren't in their saved list.
        const byId = new Map(parsed.map((x) => [x.id, x]));
        for (const d of DEFAULT_INDICATORS) {
          if (!byId.has(d.id)) byId.set(d.id, { ...d });
        }
        return Array.from(byId.values());
      }
    }
  } catch (err) { console.warn('[indicators] load err:', err); }
  return JSON.parse(JSON.stringify(DEFAULT_INDICATORS));
}

function saveIndicators() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(_indicators)); } catch {}
}

function notify() {
  for (const fn of _listeners) {
    try { fn(getIndicators()); } catch (err) { console.warn('[indicators] listener err:', err); }
  }
}

// ─── Public API ──────────────────────────────────────────────────

export function getIndicators() {
  return _indicators.map((x) => ({ ...x, config: { ...x.config } }));
}

export function subscribe(fn) {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

export function setVisible(id, visible) {
  const ind = _indicators.find((x) => x.id === id);
  if (!ind) return;
  ind.visible = !!visible;
  saveIndicators();
  notify();
}

export function removeIndicator(id) {
  const idx = _indicators.findIndex((x) => x.id === id);
  if (idx < 0) return;
  _indicators.splice(idx, 1);
  saveIndicators();
  notify();
}

export function addIndicator(type, name, config = {}) {
  // Unique id per instance (user can have two MACDs with different periods)
  const id = `${type}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`;
  _indicators.push({ id, type, name, visible: true, config });
  saveIndicators();
  notify();
  return id;
}

export function isVisible(type) {
  return _indicators.some((x) => x.type === type && x.visible);
}

// ─── Render dispatch ─────────────────────────────────────────────

/** Called by chart.js after every candle load. Applies the current
 *  visibility state to all indicator series, creating/destroying as
 *  needed. Delegates MA / BB / Wyckoff to existing overlay modules for
 *  backward compat; renders RSI / MACD / Volume MA / ATR here. */
export async function applyIndicators(chart, candleSeries, candles, overlays) {
  if (!chart) return;

  // MA Ribbon — delegate (existing) ----------------------------------
  const maOn = isVisible('ma_ribbon');
  try {
    const mod = await import('../ma_overlay.js');
    if (maOn && overlays) {
      const candleTimes = candles.map((c) => c.time);
      mod.drawMAOverlays(chart, overlays, candleTimes);
    } else if (typeof mod.toggleMAOverlays === 'function' && !maOn) {
      // ma_overlay has a "hide all" via its own toggle that expects a
      // next-visible value. Easier: set explicit false.
      try { mod.setMAOverlaysVisible?.(false); } catch {}
    }
  } catch (err) { console.warn('[indicators] ma_ribbon err:', err); }

  // Wyckoff — delegate (existing, now opt-in) ------------------------
  const wyOn = isVisible('wyckoff');
  window.__wyckoffEnabled = wyOn;
  try {
    const mod = await import('../wyckoff_overlay.js');
    if (wyOn) {
      mod.drawWyckoffOverlay(chart, candles);
    } else if (typeof mod.clearWyckoffOverlay === 'function') {
      mod.clearWyckoffOverlay(chart);
    }
  } catch (err) { console.warn('[indicators] wyckoff err:', err); }

  // BB — client-side ----------------------------------------------
  const bbInstances = _indicators.filter((x) => x.type === 'bb' && x.visible);
  clearSeriesGroup('bb');
  if (bbInstances.length > 0 && overlays?.bb_upper && overlays?.bb_middle && overlays?.bb_lower) {
    for (const ind of bbInstances) {
      renderBB(chart, candles, overlays, ind);
    }
  }

  // RSI ------------------------------------------------------------
  renderSubplot(chart, candles, 'rsi', renderRSI);

  // MACD -----------------------------------------------------------
  renderSubplot(chart, candles, 'macd', renderMACD);

  // ATR ------------------------------------------------------------
  renderSubplot(chart, candles, 'atr', renderATR);

  // Volume MA — overlays the existing volume series on 'volume' scale
  renderSubplot(chart, candles, 'volume_ma', renderVolumeMA);

  // Factor markers — triangles on candles where the factor fires.
  // All factor markers get MERGED into one setMarkers() call so
  // lightweight-charts doesn't fight over marker ownership.
  renderFactorMarkers(chart, candleSeries, candles);
}

function renderSubplot(chart, candles, type, fn) {
  const list = _indicators.filter((x) => x.type === type && x.visible);
  clearSeriesGroup(type);
  for (const ind of list) {
    try { fn(chart, candles, ind); }
    catch (err) { console.warn(`[indicators] ${type} err:`, err); }
  }
}

function clearSeriesGroup(type) {
  for (const [id, list] of _series.entries()) {
    if (!id.startsWith(type)) continue;
    for (const s of list) {
      try { window.__chartRef?.removeSeries?.(s); } catch {}
    }
    _series.delete(id);
  }
}

function trackSeries(id, series) {
  const arr = _series.get(id) || [];
  arr.push(series);
  _series.set(id, arr);
}

// ─── Renderers ───────────────────────────────────────────────────

function renderBB(chart, candles, overlays, ind) {
  const times = candles.map((c) => c.time);
  const upper = overlays.bb_upper || [];
  const middle = overlays.bb_middle || [];
  const lower = overlays.bb_lower || [];
  const mk = (data, color, style = 0) => {
    const s = chart.addLineSeries({
      color, lineWidth: 1, lineStyle: style,
      priceLineVisible: false, lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    s.setData(data.map((v, i) => ({ time: times[i], value: v })).filter((p) => p.value != null));
    trackSeries(ind.id, s);
  };
  mk(upper, 'rgba(186, 104, 200, 0.55)', 2);
  mk(middle, 'rgba(186, 104, 200, 0.4)', 2);
  mk(lower, 'rgba(186, 104, 200, 0.55)', 2);
}

function renderRSI(chart, candles, ind) {
  const period = ind.config?.period || 14;
  const closes = candles.map((c) => c.close);
  const values = computeRSI(closes, period);
  const series = chart.addLineSeries({
    color: '#38bdf8',
    lineWidth: 1.5,
    priceScaleId: 'rsi',
    priceLineVisible: false,
    lastValueVisible: true,
    crosshairMarkerVisible: false,
    title: `RSI(${period})`,
  });
  try {
    chart.priceScale('rsi').applyOptions({
      scaleMargins: { top: 0.72, bottom: 0.12 },
      borderVisible: false,
    });
  } catch {}
  series.setData(
    candles.map((c, i) => ({ time: c.time, value: values[i] })).filter((p) => p.value != null),
  );
  // Overbought/oversold reference lines
  const ref = (value, color) => {
    const s = chart.addLineSeries({
      color, lineWidth: 1, lineStyle: 2,
      priceScaleId: 'rsi',
      priceLineVisible: false, lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    if (candles.length >= 2) {
      s.setData([
        { time: candles[0].time, value },
        { time: candles[candles.length - 1].time, value },
      ]);
    }
    trackSeries(ind.id, s);
  };
  ref(70, 'rgba(239, 83, 80, 0.35)');
  ref(30, 'rgba(38, 166, 154, 0.35)');
  trackSeries(ind.id, series);
}

function renderMACD(chart, candles, ind) {
  const { fast = 12, slow = 26, signal = 9 } = ind.config || {};
  const closes = candles.map((c) => c.close);
  const { macd, signal: sig, hist } = computeMACD(closes, fast, slow, signal);

  try {
    chart.priceScale('macd').applyOptions({
      scaleMargins: { top: 0.88, bottom: 0 },
      borderVisible: false,
    });
  } catch {}

  const macdLine = chart.addLineSeries({
    color: '#38bdf8', lineWidth: 1.5,
    priceScaleId: 'macd',
    priceLineVisible: false, lastValueVisible: false,
    crosshairMarkerVisible: false,
    title: 'MACD',
  });
  macdLine.setData(candles.map((c, i) => ({ time: c.time, value: macd[i] })).filter((p) => p.value != null));
  trackSeries(ind.id, macdLine);

  const sigLine = chart.addLineSeries({
    color: '#f59e0b', lineWidth: 1.5,
    priceScaleId: 'macd',
    priceLineVisible: false, lastValueVisible: false,
    crosshairMarkerVisible: false,
    title: 'Signal',
  });
  sigLine.setData(candles.map((c, i) => ({ time: c.time, value: sig[i] })).filter((p) => p.value != null));
  trackSeries(ind.id, sigLine);

  const histBar = chart.addHistogramSeries({
    priceScaleId: 'macd',
    priceFormat: { type: 'price' },
    priceLineVisible: false, lastValueVisible: false,
  });
  histBar.setData(
    candles.map((c, i) => {
      const v = hist[i];
      if (v == null) return null;
      return { time: c.time, value: v, color: v >= 0 ? 'rgba(38, 166, 154, 0.6)' : 'rgba(239, 83, 80, 0.6)' };
    }).filter(Boolean),
  );
  trackSeries(ind.id, histBar);
}

function renderATR(chart, candles, ind) {
  const period = ind.config?.period || 14;
  const values = computeATR(candles, period);
  try {
    chart.priceScale('atr').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0.05 },
      borderVisible: false,
    });
  } catch {}
  const series = chart.addLineSeries({
    color: '#eab308', lineWidth: 1.5,
    priceScaleId: 'atr',
    priceLineVisible: false, lastValueVisible: true,
    crosshairMarkerVisible: false,
    title: `ATR(${period})`,
  });
  series.setData(
    candles.map((c, i) => ({ time: c.time, value: values[i] })).filter((p) => p.value != null),
  );
  trackSeries(ind.id, series);
}

// ─── Factor markers ─────────────────────────────────────────────
// Each factor returns a boolean[] aligned to candles. We build a
// marker per true bar (position: below for oversold / bull, above
// for overbought / bear) and set them all in one call.

const FACTOR_STYLES = {
  factor_rsi_oversold:   { fn: factorRSIOversold,   label: 'RSI<30', color: '#26a69a', position: 'belowBar', shape: 'arrowUp' },
  factor_rsi_overbought: { fn: factorRSIOverbought, label: 'RSI>70', color: '#ef5350', position: 'aboveBar', shape: 'arrowDown' },
  factor_volume_surge:   { fn: factorVolumeSurge,   label: 'VolSurge', color: '#38bdf8', position: 'aboveBar', shape: 'circle' },
  factor_macd_bull:      { fn: factorMACDBullCross, label: 'MACD↑', color: '#26a69a', position: 'belowBar', shape: 'arrowUp' },
  factor_macd_bear:      { fn: factorMACDBearCross, label: 'MACD↓', color: '#ef5350', position: 'aboveBar', shape: 'arrowDown' },
};

function renderFactorMarkers(chart, candleSeries, candles) {
  const active = _indicators.filter((x) => x.visible && x.type.startsWith('factor_'));
  // Always rewrite the whole marker set so removing a factor cleans up.
  const markers = [];
  for (const ind of active) {
    const style = FACTOR_STYLES[ind.type];
    if (!style) continue;
    try {
      const args = [candles];
      // Feed optional config values into the math fn in a predictable
      // positional order (period, threshold/multiple/slow/signal). The
      // math fns use defaults if args are missing.
      if (ind.config) {
        if (ind.type === 'factor_rsi_oversold' || ind.type === 'factor_rsi_overbought') {
          args.push(ind.config.period ?? 14, ind.config.threshold);
        } else if (ind.type === 'factor_volume_surge') {
          args.push(ind.config.period ?? 20, ind.config.multiple ?? 2.0);
        } else {
          args.push(ind.config.fast ?? 12, ind.config.slow ?? 26, ind.config.signal ?? 9);
        }
      }
      const flags = style.fn(...args);
      for (let i = 0; i < flags.length; i++) {
        if (!flags[i]) continue;
        markers.push({
          time: candles[i].time,
          position: style.position,
          color: style.color,
          shape: style.shape,
          text: style.label,
          size: 0.7,
        });
      }
    } catch (err) { console.warn(`[factor] ${ind.type} render err:`, err); }
  }
  // Merge with any existing candle markers (execution arrows, Wyckoff
  // spring/utad), dedup by (time, shape, color, position) so a re-apply
  // doesn't stack.
  try {
    const existing = candleSeries.markers ? candleSeries.markers() : [];
    // Drop any previous factor markers — we identify them by our unique
    // text prefixes to avoid nuking execution markers. (执行侧用中文"买"
    // "卖"; Wyckoff 用 "S"/"UT"; factor markers use the labels above.)
    const factorLabels = new Set(Object.values(FACTOR_STYLES).map((s) => s.label));
    const kept = (existing || []).filter((m) => !factorLabels.has(m.text));
    const merged = [...kept, ...markers].sort((a, b) => a.time - b.time);
    candleSeries.setMarkers(merged);
  } catch (err) {
    console.warn('[factor] setMarkers failed:', err);
  }
}

function renderVolumeMA(chart, candles, ind) {
  const period = ind.config?.period || 20;
  const values = computeVolumeMA(candles, period);
  const series = chart.addLineSeries({
    color: '#94a3b8', lineWidth: 1.5,
    priceScaleId: 'volume',
    priceLineVisible: false, lastValueVisible: false,
    crosshairMarkerVisible: false,
    title: `Vol MA(${period})`,
  });
  series.setData(
    candles.map((c, i) => ({ time: c.time, value: values[i] })).filter((p) => p.value != null),
  );
  trackSeries(ind.id, series);
}
