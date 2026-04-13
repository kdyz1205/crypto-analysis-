/**
 * Trendline overlay — draws S/R trendlines.
 *
 * Reworked to:
 * - Accept 2-touch candidate lines (user philosophy: 2-touch is DRAW,
 *   3-touch is the trade edge we're hunting)
 * - Snap anchor times to real candle times so lightweight-charts plots
 *   the series (times not in the time scale get silently dropped)
 * - Project forward to the last loaded candle (not beyond, which also
 *   gets silently dropped)
 * - Style by touch_count: confirmed 3+ solid orange, 2-touch dashed gold
 * - Publish window.__activeTrendlines for decision rail / click handler
 */

const lineSeriesRefs = [];
window.__activeTrendlines = [];

function clearSeries(chart, refs) {
  for (const series of refs) {
    try { chart.removeSeries(series); } catch {}
  }
  refs.length = 0;
}

export function clearTrendlineOverlay(chart) {
  if (!chart) return;
  clearSeries(chart, lineSeriesRefs);
}

/**
 * Binary search: snap arbitrary timestamp to the nearest existing candle time.
 * lightweight-charts line series can ONLY plot at times that exist in the
 * main time scale (built from the candle series). Using a raw timestamp that
 * doesn't match a candle time results in the data point being silently
 * dropped from render — invisible failure.
 */
function snapToCandle(t, candleTimes) {
  if (!candleTimes || !candleTimes.length) return t;
  let lo = 0, hi = candleTimes.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (Number(candleTimes[mid]) < t) lo = mid + 1;
    else hi = mid;
  }
  const a = Number(candleTimes[Math.max(0, lo - 1)]);
  const b = Number(candleTimes[lo]);
  return Math.abs(a - t) < Math.abs(b - t) ? a : b;
}

function lineColor(line) {
  if (line.is_invalidated) return 'rgba(148, 163, 184, 0.5)';
  const isConfirmed = (line.confirming_touch_count || 0) >= 3;
  if (line.side === 'resistance') {
    return isConfirmed
      ? 'rgba(255, 140, 0, 1.0)'      // bright orange
      : 'rgba(255, 196, 82, 0.85)';   // gold
  }
  return isConfirmed
    ? 'rgba(255, 140, 0, 1.0)'
    : 'rgba(255, 196, 82, 0.85)';
}

export function drawTrendlineOverlay(chart, snapshot, layerVisibility = {}, candleTimes = null) {
  if (!chart) return;
  clearTrendlineOverlay(chart);
  if (!snapshot) return;
  if (layerVisibility.primaryTrendlines === false) return;

  // Merge active_lines + candidate_lines. We WANT candidate (2-touch) lines:
  // the fade+flip strategy trades the 3rd touch, so drawing stops when the
  // line is already confirmed would miss every trade.
  const allLines = [
    ...(Array.isArray(snapshot.active_lines) ? snapshot.active_lines : []),
    ...(Array.isArray(snapshot.candidate_lines) ? snapshot.candidate_lines : []),
  ];

  const seen = new Set();
  const filtered = allLines
    .filter(line => {
      if (!line) return false;
      if (line.is_invalidated || line.invalidation_reason) return false;
      if (line.display_class === 'debug' && !layerVisibility.debugTrendlines) return false;
      if (line.line_id && seen.has(line.line_id)) return false;
      if (line.line_id) seen.add(line.line_id);
      // Required fields
      if (line.t_start == null || line.t_end == null) return false;
      if (line.price_start == null || line.price_end == null) return false;
      if (!Number.isFinite(Number(line.price_start)) || !Number.isFinite(Number(line.price_end))) return false;
      if (Number(line.t_end) <= Number(line.t_start)) return false;
      return true;
    })
    .sort((a, b) => {
      // confirmed first, then by touch count, then by display_rank
      const ta = (a.confirming_touch_count || 0) >= 3 ? 0 : 1;
      const tb = (b.confirming_touch_count || 0) >= 3 ? 0 : 1;
      if (ta !== tb) return ta - tb;
      if (a.confirming_touch_count !== b.confirming_touch_count) {
        return (b.confirming_touch_count || 0) - (a.confirming_touch_count || 0);
      }
      return (a.display_rank || 999) - (b.display_rank || 999);
    })
    .slice(0, 8);

  // Publish for decision rail + click handler
  window.__activeTrendlines = filtered.map(l => ({
    line_id: l.line_id,
    side: l.side,
    state: l.state,
    confirming_touch_count: l.confirming_touch_count,
    display_class: l.display_class,
    anchor_indices: l.anchor_indices,
    anchor_prices: l.anchor_prices,
    price_start: l.price_start,
    price_end: l.price_end,
    t_start: l.t_start,
    t_end: l.t_end,
  }));
  try {
    window.dispatchEvent(new CustomEvent('trendlines-updated', { detail: window.__activeTrendlines }));
  } catch (e) {}

  if (filtered.length === 0) return;

  // Resolve last candle time for projection clamp
  let lastCandleTime = null;
  if (candleTimes && candleTimes.length > 0) {
    lastCandleTime = Number(candleTimes[candleTimes.length - 1]);
  } else {
    try {
      const r = chart.timeScale?.()?.getVisibleRange?.();
      if (r) lastCandleTime = Number(r.to);
    } catch (e) {}
  }

  for (const line of filtered) {
    try {
      const tStart = Math.floor(Number(line.t_start));
      const tEnd = Math.floor(Number(line.t_end));
      const pStart = Number(line.price_start);
      const pEnd = Number(line.price_end);
      const dt = tEnd - tStart;
      if (dt <= 0) continue;
      const slopePerSec = (pEnd - pStart) / dt;

      // Project from the first anchor all the way to the last candle
      const projectTo = (lastCandleTime && lastCandleTime > tEnd) ? lastCandleTime : tEnd;

      // Snap both endpoints to actual candle times (required for render)
      const snappedStart = snapToCandle(tStart, candleTimes);
      const snappedEnd = snapToCandle(projectTo, candleTimes);
      if (snappedStart === snappedEnd) continue;

      // Recompute line values at snapped times
      const vStart = pStart + slopePerSec * (snappedStart - tStart);
      const vEnd = pStart + slopePerSec * (snappedEnd - tStart);
      if (!Number.isFinite(vStart) || !Number.isFinite(vEnd)) continue;

      const isConfirmed = (line.confirming_touch_count || 0) >= 3;
      const color = lineColor(line);
      const lineWidth = isConfirmed ? 3 : 2;
      const lineStyle = 0;  // always solid — dashed is invisible over candles

      const series = chart.addLineSeries({
        color,
        lineWidth,
        lineStyle,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
        autoscaleInfoProvider: () => ({ priceRange: null }),
        title: `${line.side === 'support' ? 'S' : 'R'} ${line.confirming_touch_count || 2}t`,
      });

      const points = [
        { time: snappedStart, value: vStart },
        { time: snappedEnd, value: vEnd },
      ];
      points.sort((a, b) => a.time - b.time);
      series.setData(points);
      lineSeriesRefs.push(series);
    } catch (err) {
      console.warn('[trendline] draw failed:', line?.line_id, err);
    }
  }
}
