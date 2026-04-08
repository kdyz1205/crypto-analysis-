// frontend/js/state/patterns.js
import { publish } from '../util/events.js';

export const patternsState = {
  toolMode: 'recognize',    // 'recognize' | 'draw' | 'assist' | null
  drawingMode: null,         // 'trend' | 'horizontal' | null
  pendingDrawPoint: null,
  userDrawnLines: [],
  rawPatternData: null,
  patternStatsData: null,
  selectedLineIndex: null,
  similarLineIndices: [],
  srLineSegments: [],
  srVisible: true,
  maxSRLines: 0,
  patternResponseCache: new Map(),
};

export function setToolMode(mode) {
  if (patternsState.toolMode === mode) return;
  patternsState.toolMode = mode;
  publish('patterns.tool.changed', mode);
}

export function setDrawingMode(mode) {
  patternsState.drawingMode = mode;
  publish('patterns.drawing.changed', mode);
}

export function setSrVisible(v) {
  patternsState.srVisible = !!v;
  publish('patterns.sr.visibility', patternsState.srVisible);
}

export function setMaxSRLines(n) {
  patternsState.maxSRLines = Number(n) || 0;
  publish('patterns.sr.max', patternsState.maxSRLines);
}

export function setSelectedLineIndex(idx) {
  patternsState.selectedLineIndex = idx;
  publish('patterns.line.selected', idx);
}

export function addUserDrawnLine(line) {
  patternsState.userDrawnLines.push(line);
  publish('patterns.drawn.added', line);
}

export function popUserDrawnLine() {
  const line = patternsState.userDrawnLines.pop();
  publish('patterns.drawn.popped', line);
  return line;
}

export function clearUserDrawnLines() {
  patternsState.userDrawnLines = [];
  publish('patterns.drawn.cleared', null);
}

export function setRawPatternData(data) {
  patternsState.rawPatternData = data;
  publish('patterns.raw.updated', data);
}
