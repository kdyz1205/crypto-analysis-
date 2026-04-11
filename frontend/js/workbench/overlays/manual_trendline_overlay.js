const manualLineSeriesRefs = [];

function clearSeries(chart, refs) {
  for (const series of refs) {
    try { chart.removeSeries(series); } catch {}
  }
  refs.length = 0;
}

function manualColor(line, selectedLineId) {
  if (line.manual_line_id === selectedLineId) return 'rgba(56, 189, 248, 1)';
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
      const series = chart.addLineSeries({
        color: manualColor(line, options.selectedLineId || null),
        lineWidth: line.manual_line_id === options.selectedLineId ? 3 : 2,
        lineStyle: line.locked ? 0 : 2,
        priceLineVisible: false,
        lastValueVisible: false,
        autoscaleInfoProvider: () => null,
      });
      series.setData(resolveManualTrendlinePoints(line, options));
      manualLineSeriesRefs.push(series);
    } catch (err) {
      console.warn('[manual overlay] failed to draw line', line.manual_line_id, err);
    }
  }
}
