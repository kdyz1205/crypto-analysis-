const manualLineSeriesRefs = [];
// Map line_id → its LineSeries object (for highlight-on-hover)
const lineSeriesById = new Map();
let _highlightedId = null;
let _highlightTimer = null;
// Persistently-selected line_id. Stays lit until explicitly changed.
let _selectedId = null;

function clearSeries(chart, refs) {
  for (const series of refs) {
    try { chart.removeSeries(series); } catch {}
  }
  refs.length = 0;
  lineSeriesById.clear();
  _highlightedId = null;
}

function manualColor(line, selectedLineId) {
  if (line.manual_line_id === selectedLineId) return 'rgba(56, 189, 248, 1)';
  // Broken lines (price crossed through after they were drawn) — dim grey.
  if (line._broken) return 'rgba(148, 163, 184, 0.55)';
  if (line.override_mode === 'promote_to_active' || line.override_mode === 'strategy_input_enabled') {
    return line.side === 'resistance' ? 'rgba(251, 146, 60, 0.95)' : 'rgba(45, 212, 191, 0.95)';
  }
  return line.side === 'resistance' ? 'rgba(125, 211, 252, 0.9)' : 'rgba(103, 232, 249, 0.9)';
}

function projectPrice(line, targetTime) {
  const dt = Math.max(line.t_end - line.t_start, 1);
  const slope = (Number(line.price_end) - Number(line.price_start)) / dt;
  return Number(line.price_start) + (slope * (targetTime - line.t_start));
}

export function resolveManualTrendlinePoints(line, options = {}) {
  const latestTime = Number(options.latestTime || 0);
  const earliestTime = Number(options.earliestTime || 0);
  const startTime = line.extend_left && earliestTime > 0 && earliestTime < line.t_start
    ? earliestTime
    : Number(line.t_start);
  const endTime = line.extend_right && latestTime > line.t_end
    ? latestTime
    : Number(line.t_end);
  const startPrice = startTime !== Number(line.t_start) ? projectPrice(line, startTime) : Number(line.price_start);
  const endPrice = endTime !== Number(line.t_end) ? projectPrice(line, endTime) : Number(line.price_end);
  return [
    { time: Math.floor(startTime), value: Number(startPrice) },
    { time: Math.floor(endTime), value: Number(endPrice) },
  ];
}

export function clearManualTrendlineOverlay(chart) {
  if (!chart) return;
  clearSeries(chart, manualLineSeriesRefs);
}

export function drawManualTrendlineOverlay(chart, lines, options = {}) {
  if (!chart) return;
  clearManualTrendlineOverlay(chart);
  const visibleLines = Array.isArray(lines) ? lines : [];

  for (const line of visibleLines) {
    try {
      const isSelected = line.manual_line_id === options.selectedLineId;
      const series = chart.addLineSeries({
        color: manualColor(line, options.selectedLineId || null),
        lineWidth: isSelected ? 3 : 2,
        lineStyle: 0,  // solid — dashed is hard to see
        priceLineVisible: false,
        lastValueVisible: false,
        autoscaleInfoProvider: () => null,
      });
      series.setData(resolveManualTrendlinePoints(line, options));
      manualLineSeriesRefs.push(series);
      lineSeriesById.set(line.manual_line_id, { series, line });
    } catch (err) {
      console.warn('[manual overlay] failed to draw line', line.manual_line_id, err);
    }
  }
}

/**
 * Temporarily highlight a line by flashing its color + width.
 * Called by the conditional panel when user hovers over a line in the list.
 */
function _restore(id) {
  if (!id) return;
  const entry = lineSeriesById.get(id);
  if (!entry) return;
  try {
    entry.series.applyOptions({
      color: manualColor(entry.line, _selectedId),
      lineWidth: entry.line.manual_line_id === _selectedId ? 3 : 2,
    });
  } catch {}
}

export function highlightManualLine(manualLineId) {
  if (_highlightedId === manualLineId) return;
  if (_highlightedId && _highlightedId !== _selectedId) _restore(_highlightedId);
  _highlightedId = manualLineId;
  if (_highlightTimer) { clearTimeout(_highlightTimer); _highlightTimer = null; }
  if (!manualLineId) return;
  const entry = lineSeriesById.get(manualLineId);
  if (!entry) return;
  try {
    entry.series.applyOptions({
      color: 'rgba(251, 191, 36, 1)',
      lineWidth: 4,
    });
  } catch {}
  _highlightTimer = setTimeout(() => {
    if (_highlightedId === manualLineId) {
      _highlightedId = null;
      if (manualLineId !== _selectedId) _restore(manualLineId);
    }
  }, 3000);
}

/** Persistently mark a line as selected — stays lit until changed. */
export function selectManualLineOnChart(manualLineId) {
  const prev = _selectedId;
  _selectedId = manualLineId || null;
  if (prev && prev !== _selectedId) _restore(prev);
  if (!_selectedId) return;
  const entry = lineSeriesById.get(_selectedId);
  if (!entry) return;
  try {
    entry.series.applyOptions({
      color: 'rgba(56, 189, 248, 1)',  // cyan, distinct from hover gold
      lineWidth: 4,
    });
  } catch {}
}
