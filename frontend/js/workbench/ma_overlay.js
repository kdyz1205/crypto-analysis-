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
        title: cfg.title,
        autoscaleInfoProvider: () => null,  // exclude from auto-scale calculation
      });
    }
    try { seriesRefs[key].setData(data); }
    catch (err) { console.warn('[ma] setData failed', key, err); }
    seriesRefs[key].applyOptions({ visible });
  }
  console.log('[ma] drew', Object.keys(seriesRefs).length, 'MA series');
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
