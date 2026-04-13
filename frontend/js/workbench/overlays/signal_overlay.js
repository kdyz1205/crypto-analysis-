function triggerLabel(triggerMode) {
  if (triggerMode === 'failed_breakout') return 'FB';
  if (triggerMode === 'rejection') return 'REJ';
  return 'PRE';
}

export function clearSignalOverlay(candleSeries) {
  if (!candleSeries || typeof candleSeries.setMarkers !== 'function') return;
  try { candleSeries.setMarkers([]); } catch {}
}

export function drawSignalOverlay(candleSeries, snapshot, layerVisibility = {}) {
  if (!candleSeries || typeof candleSeries.setMarkers !== 'function') return;
  if (!snapshot) {
    clearSignalOverlay(candleSeries);
    return;
  }

  const markers = [];
  const activeLineIds = new Set((snapshot.active_lines || []).map((line) => line.line_id));
  const touchPoints = Array.isArray(snapshot.touch_points) ? snapshot.touch_points : [];
  const lineLookup = new Map((snapshot.candidate_lines || []).map((line) => [line.line_id, line]));

  const scopedTouches = activeLineIds.size > 0
    ? touchPoints.filter((point) => activeLineIds.has(point.line_id) || point.display_visible)
    : touchPoints;

  if (layerVisibility.confirmingTouches !== false || layerVisibility.barTouches === true) {
    const visibleTouches = scopedTouches.filter((point) => {
      if (!point.display_visible) return false;
      if (point.display_class === 'confirming') return layerVisibility.confirmingTouches !== false;
      if (point.display_class === 'bar') return layerVisibility.barTouches === true;
      return false;
    });
    for (const point of visibleTouches) {
      markers.push({
        time: Math.floor(point.timestamp),
        position: point.side === 'resistance' ? 'aboveBar' : 'belowBar',
        color: point.is_confirming_touch ? '#8b5cf6' : 'rgba(148, 163, 184, 0.85)',
        shape: point.is_confirming_touch ? 'square' : 'circle',
        text: point.is_confirming_touch ? 'C' : 'T',
      });
    }
  }

  if (layerVisibility.signalMarkers !== false) {
    for (const signal of (snapshot.signals || [])) {
      markers.push({
        time: Math.floor(signal.timestamp),
        position: signal.direction === 'short' ? 'aboveBar' : 'belowBar',
        color: signal.direction === 'short' ? '#ff1744' : '#00e676',
        shape: signal.direction === 'short' ? 'arrowDown' : 'arrowUp',
        text: triggerLabel(signal.trigger_mode),
      });
    }
  }

  if (layerVisibility.collapsedInvalidations !== false) {
    for (const invalidation of (snapshot.invalidations || [])) {
      if (invalidation.display_class === 'debug' || invalidation.display_class === 'debug_invalidation') {
        continue;
      }
      const line = lineLookup.get(invalidation.line_id);
      const markerTime = invalidation.invalidation_timestamp ?? line?.invalidation_timestamp ?? snapshot.timestamp;
      // Skip invalid timestamps — earlier code happily passed NaN through
      // Math.floor(NaN)=NaN and lightweight-charts plotted markers at T=0.
      if (markerTime == null || !Number.isFinite(Number(markerTime))) continue;
      markers.push({
        time: Math.floor(Number(markerTime)),
        position: invalidation.side === 'resistance' ? 'aboveBar' : 'belowBar',
        color: '#94a3b8',
        shape: 'square',
        text: invalidation.collapsed_invalidation_count > 1 ? `X${invalidation.collapsed_invalidation_count}` : 'X',
      });
    }
  }

  // Merge with existing markers (e.g. exec markers from the order overlay)
  // instead of clobbering them. Dedupe by (time, shape, color) tuple.
  let existing = [];
  try {
    existing = candleSeries.markers?.() || [];
  } catch {}
  const merged = [...existing, ...markers];
  const seen = new Set();
  const dedup = [];
  for (const m of merged) {
    const key = `${m.time}-${m.shape}-${m.color}-${m.position}`;
    if (seen.has(key)) continue;
    seen.add(key);
    dedup.push(m);
  }
  dedup.sort((a, b) => a.time - b.time);
  try {
    candleSeries.setMarkers(dedup);
  } catch (err) {
    console.warn('[strategy overlay] failed to set markers', err);
  }
}
