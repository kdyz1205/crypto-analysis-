import { publish } from '../util/events.js';

export const drawingsState = {
  lines: [],
  loading: false,
  error: null,
  drawSide: null,
  pendingAnchor: null,
  selectedLineId: null,
  // Multi-select: set of line ids. When non-empty, single-select UI
  // still works — selectedLineId is always the latest added. Batch
  // operations (delete, toggle visibility) use this set.
  multiSelectedIds: new Set(),
  // Per-line visibility toggle. Lines in this set are NOT rendered
  // on the chart even though they still exist + their tradePlans are
  // still running server-side. Purely a display flag.
  hiddenLineIds: new Set(),
  // Conditional count per line id, refreshed by the panel.
  tradePlanCounts: {},   // { [manual_line_id]: count }
  editTarget: null,
  viewMode: 'mixed',
};

export function toggleMultiSelected(lineId) {
  if (!lineId) return;
  if (drawingsState.multiSelectedIds.has(lineId)) {
    drawingsState.multiSelectedIds.delete(lineId);
  } else {
    drawingsState.multiSelectedIds.add(lineId);
  }
  publish('drawings.multiSelected', drawingsState.multiSelectedIds);
}

export function clearMultiSelected() {
  drawingsState.multiSelectedIds.clear();
  publish('drawings.multiSelected', drawingsState.multiSelectedIds);
}

export function toggleLineVisibility(lineId) {
  if (!lineId) return;
  if (drawingsState.hiddenLineIds.has(lineId)) {
    drawingsState.hiddenLineIds.delete(lineId);
  } else {
    drawingsState.hiddenLineIds.add(lineId);
  }
  publish('drawings.visibility', drawingsState.hiddenLineIds);
}

export function setTradePlanCounts(counts) {
  drawingsState.tradePlanCounts = counts || {};
  publish('drawings.tradePlanCounts', drawingsState.tradePlanCounts);
}

export function setManualDrawings(lines) {
  drawingsState.lines = Array.isArray(lines) ? lines : [];
  publish('drawings.updated', drawingsState.lines);
}

export function setDrawingsLoading(loading) {
  drawingsState.loading = !!loading;
  publish('drawings.loading', drawingsState.loading);
}

export function setDrawingsError(error) {
  drawingsState.error = error ? String(error) : null;
  publish('drawings.error', drawingsState.error);
}

export function setDrawSide(side) {
  drawingsState.drawSide = side;
  publish('drawings.mode', drawingsState.drawSide);
}

export function setPendingAnchor(anchor) {
  drawingsState.pendingAnchor = anchor || null;
  publish('drawings.anchor', drawingsState.pendingAnchor);
}

export function setSelectedManualLine(manualLineId) {
  drawingsState.selectedLineId = manualLineId || null;
  publish('drawings.selected', drawingsState.selectedLineId);
}

export function setEditTarget(target) {
  drawingsState.editTarget = target || null;
  publish('drawings.editTarget', drawingsState.editTarget);
}

export function setDrawingViewMode(mode) {
  drawingsState.viewMode = mode || 'mixed';
  publish('drawings.viewMode', drawingsState.viewMode);
}
