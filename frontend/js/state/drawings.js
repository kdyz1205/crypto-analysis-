import { publish } from '../util/events.js';

export const drawingsState = {
  lines: [],
  loading: false,
  error: null,
  drawSide: null,
  pendingAnchor: null,
  selectedLineId: null,
  editTarget: null,
  viewMode: 'mixed',
};

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
