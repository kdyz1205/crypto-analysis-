const lineSeriesRefs = [];
const projectedSeriesRefs = [];

function clearSeries(chart, refs) {
  for (const series of refs) {
    try { chart.removeSeries(series); } catch {}
  }
  refs.length = 0;
}

function lineStyleForState(state) {
  if (state === 'candidate') return 2;
  if (state === 'invalidated' || state === 'expired') return 1;
  return 0;
}

function lineColor(line) {
  if (line.is_invalidated) return 'rgba(148, 163, 184, 0.75)';
  if (line.side === 'resistance') {
    return line.is_active ? 'rgba(255, 92, 92, 0.95)' : 'rgba(255, 146, 43, 0.65)';
  }
  return line.is_active ? 'rgba(0, 230, 118, 0.95)' : 'rgba(34, 197, 94, 0.65)';
}

function projectedColor(line) {
  if (line.side === 'resistance') return 'rgba(255, 196, 82, 0.8)';
  return 'rgba(110, 231, 183, 0.8)';
}

export function clearTrendlineOverlay(chart) {
  if (!chart) return;
  clearSeries(chart, lineSeriesRefs);
  clearSeries(chart, projectedSeriesRefs);
}

export function drawTrendlineOverlay(chart, snapshot, layerVisibility = {}) {
  if (!chart) return;
  clearTrendlineOverlay(chart);
  if (!snapshot) return;

  const allLines = Array.isArray(snapshot.candidate_lines) ? [...snapshot.candidate_lines] : [];
  // Only show active primary lines to reduce noise — max 6 total
  const primaryLines = allLines
    .filter((line) => line.display_class === 'primary' && line.is_active)
    .sort((a, b) => (a.display_rank || 999) - (b.display_rank || 999))
    .slice(0, 6);
  const debugLines = layerVisibility.debugTrendlines
    ? allLines.filter((line) => line.display_class === 'debug').sort((a, b) => (a.display_rank || 999) - (b.display_rank || 999))
    : [];
  const visibleLines = [
    ...(layerVisibility.primaryTrendlines !== false ? primaryLines : []),
    ...debugLines,
  ];

  if (visibleLines.length > 0) {
    for (const line of visibleLines) {
      try {
        const series = chart.addLineSeries({
          color: lineColor(line),
          lineWidth: line.display_class === 'primary' ? 3 : line.is_active ? 2 : 1,
          lineStyle: line.display_class === 'debug' ? 2 : lineStyleForState(line.state),
          priceLineVisible: false,
          lastValueVisible: false,
          autoscaleInfoProvider: () => null,
        });
        series.setData([
          { time: Math.floor(line.t_start), value: Number(line.price_start) },
          { time: Math.floor(line.t_end), value: Number(line.price_end) },
        ]);
        lineSeriesRefs.push(series);
      } catch (err) {
        console.warn('[strategy overlay] failed to draw trendline', line.line_id, err);
      }
    }
  }

  if (layerVisibility.projectedLine !== false) {
    for (const line of visibleLines.filter((item) => item.is_active && item.display_class !== 'debug')) {
      try {
        const projectedSeries = chart.addLineSeries({
          color: projectedColor(line),
          lineWidth: 1,
          lineStyle: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          autoscaleInfoProvider: () => null,
        });
        projectedSeries.setData([
          { time: Math.floor(line.projected_time_current), value: Number(line.projected_price_current) },
          { time: Math.floor(line.projected_time_next), value: Number(line.projected_price_next) },
        ]);
        projectedSeriesRefs.push(projectedSeries);
      } catch (err) {
        console.warn('[strategy overlay] failed to draw projected line', line.line_id, err);
      }
    }
  }
}
