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

  if (layerVisibility.touchMarkers !== false) {
    const scopedTouches = activeLineIds.size > 0
      ? touchPoints.filter((point) => activeLineIds.has(point.line_id))
      : touchPoints;
    for (const point of scopedTouches.slice(-80)) {
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

  if (layerVisibility.invalidationMarkers !== false) {
    for (const invalidation of (snapshot.invalidations || [])) {
      const line = lineLookup.get(invalidation.line_id);
      const markerTime = line?.t_end ?? snapshot.timestamp;
      markers.push({
        time: Math.floor(markerTime),
        position: invalidation.side === 'resistance' ? 'aboveBar' : 'belowBar',
        color: '#94a3b8',
        shape: 'square',
        text: 'X',
      });
    }
  }

  markers.sort((a, b) => a.time - b.time);
  try {
    candleSeries.setMarkers(markers);
  } catch (err) {
    console.warn('[strategy overlay] failed to set markers', err);
  }
}
