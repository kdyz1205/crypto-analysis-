// frontend/js/workbench/drawings/chart_drawing.js
//
// Single-module trendline drawing/editing built on the architectural
// rules the user laid out:
//
//   - persistent state lives in drawingsState (lines[], selectedLineId)
//   - transient state lives in this module (state, cursor, draftLine, dragging)
//   - mousemove only updates transient state, never persistent state
//   - all geometry goes through dataToScreen / screenToData; no raw pixels
//     ever touch a persistent line object
//   - render is RAF-coalesced and reads both state layers
//
// State machine:
//
//      idle ──[T / 画线]──> drawing_first_point
//      drawing_first_point ──[click]──> drawing_second_point  (set draft.start)
//      drawing_second_point ──[mousemove]──> stays            (update draft.end)
//      drawing_second_point ──[click]──> drawing_first_point  (commit, sticky)
//      drawing_* ──[Esc]──> idle
//      idle ──[mousedown anchor]──> dragging_anchor
//      idle ──[mousedown body]──> dragging_line
//      dragging_* ──[mousemove]──> stays (update dragging preview)
//      dragging_* ──[mouseup]──> idle (PATCH commit)

import { drawingsState, setManualDrawings, setSelectedManualLine } from '../../state/drawings.js';
import { marketState } from '../../state/market.js';
import { subscribe } from '../../util/events.js';
import { fetchJson } from '../../util/fetch.js';
import * as drawingsSvc from '../../services/drawings.js';
import { openTradePlanModal } from './trade_plan_modal.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const ANCHOR_R = 6;        // visible handle radius
const ANCHOR_HIT = 12;     // hit test radius for anchors
const BODY_HIT = 8;        // hit test distance for line body

// ─────────────────────────────────────────────────────────────
// Module deps (set by init)
// ─────────────────────────────────────────────────────────────
let _chart = null;
let _candleSeries = null;
let _container = null;
let _svg = null;

// ─────────────────────────────────────────────────────────────
// Transient state (mousemove writes here, never persists)
// ─────────────────────────────────────────────────────────────
const tx = {
  state: 'idle',
  /** @type {{time: number, price: number} | null} */
  cursor: null,
  /** @type {{start: {time, price}, end: {time, price}} | null} */
  draftLine: null,
  /** @type {{lineId: string, mode: 'anchor_start'|'anchor_end'|'body', origLine: any, grabCursor: {time, price}} | null} */
  dragging: null,
  hoveredLineId: null,
  /** @type {{t_start, t_end, price_start, price_end, outcome?}[]} */
  similarLines: [],
};

let _rafPending = false;

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────
export function initChartDrawing(chart, candleSeries, container) {
  _chart = chart;
  _candleSeries = candleSeries;
  _container = container;

  _svg = document.createElementNS(SVG_NS, 'svg');
  _svg.setAttribute('class', 'chart-drawing-overlay');
  Object.assign(_svg.style, {
    position: 'absolute',
    inset: '0',
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
    zIndex: '20',
  });
  container.style.position = container.style.position || 'relative';
  container.appendChild(_svg);

  // Single source of truth for click and mousemove on the chart.
  // We use container-level DOM events because we need clicks in the
  // rightOffset (future) area too, which subscribeClick can miss.
  container.addEventListener('mousedown', onMouseDown);
  container.addEventListener('mousemove', onMouseMove);
  container.addEventListener('click',     onClick);
  container.addEventListener('contextmenu', onContextMenu);
  document.addEventListener('mousemove', onDocMouseMove);
  document.addEventListener('mouseup',   onDocMouseUp);
  document.addEventListener('keydown',   onKeyDown);

  // Re-render whenever chart pan/zoom or data changes
  chart.timeScale().subscribeVisibleTimeRangeChange(scheduleRender);
  chart.timeScale().subscribeVisibleLogicalRangeChange?.(scheduleRender);
  window.addEventListener('resize', scheduleRender);

  subscribe('drawings.updated', scheduleRender);
  subscribe('drawings.selected', scheduleRender);
  subscribe('chart.load.succeeded', scheduleRender);

  // Cancel any in-flight drawing AND force-thaw the chart on symbol/tf
  // change. Defensive belt-and-braces — any earlier stuck freeze gets
  // recovered here, even if our state machine thinks it was already idle.
  const recover = (reason) => {
    transition('idle', reason);
    try { chart.applyOptions({ handleScroll: true, handleScale: true }); } catch {}
  };
  subscribe('market.symbol.changed', () => recover('symbol_changed'));
  subscribe('market.interval.changed', () => recover('tf_changed'));

  scheduleRender();
  console.log('[chart_drawing] initialized');
}

/** Called by the toolbar / keyboard shortcut. */
export function startTrendlineTool() {
  transition('drawing_first_point', 'tool_armed');
}

/** Exposed so refresh paths can avoid race conditions during drag. */
export function isDragging() {
  return tx.state === 'dragging_anchor' || tx.state === 'dragging_line';
}

/**
 * Scroll the chart's visible range so `line` is centered. If the line
 * is already visible, just flashes the selection (no pan).
 */
export function jumpToLine(lineId) {
  const line = (drawingsState.lines || []).find((l) => l.manual_line_id === lineId);
  if (!line || !_chart) return;
  setSelectedManualLine(lineId);
  try {
    const ts = _chart.timeScale();
    const vr = ts.getVisibleRange?.();
    const tA = Math.min(line.t_start, line.t_end);
    const tB = Math.max(line.t_start, line.t_end);
    // If line overlaps visible range, no pan needed
    if (vr && tA <= vr.to && tB >= vr.from) {
      scheduleRender();
      return;
    }
    // Pan so the midpoint is centered
    const mid = (tA + tB) / 2;
    const span = (vr && vr.to - vr.from) || Math.max(60, (tB - tA) * 2);
    ts.setVisibleRange({ from: mid - span / 2, to: mid + span / 2 });
    scheduleRender();
  } catch (e) {
    console.warn('[chart_drawing] jumpToLine failed', e);
  }
}

/** Called by the conditional panel when user hovers a line in the list. */
export function setHoveredLineFromPanel(lineId) {
  if (tx.hoveredLineId === lineId) return;
  tx.hoveredLineId = lineId || null;
  scheduleRender();
}

/**
 * Set the historical "similar pattern" lines to overlay as faded ghosts.
 * Pass an empty array to clear. These are NEVER persisted; pure visualization.
 */
export function setSimilarLines(lines) {
  tx.similarLines = Array.isArray(lines) ? lines : [];
  scheduleRender();
}

// ─────────────────────────────────────────────────────────────
// Coordinate conversion — the only place pixels meet (time, price)
// ─────────────────────────────────────────────────────────────
function dataToScreen(point) {
  if (!point || !_chart || !_candleSeries) return null;
  const ts = _chart.timeScale();
  let x = ts.timeToCoordinate(point.time);
  if (x == null) {
    // Future or out-of-range: extrapolate from barSpacing
    x = futureTimeToX(point.time);
  }
  if (x == null) return null;
  let y;
  try { y = _candleSeries.priceToCoordinate(point.price); } catch { return null; }
  if (y == null) return null;
  return { x, y };
}

function screenToData(x, y) {
  if (!_chart || !_candleSeries) return null;
  const ts = _chart.timeScale();
  let time = null;
  try {
    const t = ts.coordinateToTime(x);
    if (t != null) time = typeof t === 'object' ? Number(t.timestamp ?? t) : Number(t);
  } catch {}
  if (time == null) time = xToFutureTime(x);
  if (time == null) return null;
  let price;
  try { price = _candleSeries.coordinateToPrice(y); } catch { return null; }
  if (price == null || !isFinite(price)) return null;
  return { time: Math.floor(time), price: Number(price) };
}

function barIntervalSec() {
  const c = marketState.lastCandles || [];
  if (c.length < 2) return 3600;
  return Math.max(60, c[c.length - 1].time - c[c.length - 2].time);
}

function futureTimeToX(time) {
  const c = marketState.lastCandles || [];
  if (!c.length || !_chart) return null;
  const last = c[c.length - 1];
  const ts = _chart.timeScale();
  const lastX = ts.timeToCoordinate(last.time);
  if (lastX == null) return null;
  const spacing = ts.options().barSpacing || 8;
  const bars = (time - last.time) / barIntervalSec();
  return lastX + bars * spacing;
}

function xToFutureTime(x) {
  const c = marketState.lastCandles || [];
  if (!c.length || !_chart) return null;
  const last = c[c.length - 1];
  const ts = _chart.timeScale();
  const lastX = ts.timeToCoordinate(last.time);
  if (lastX == null) return null;
  const spacing = ts.options().barSpacing || 8;
  const bars = (x - lastX) / spacing;
  return Math.floor(last.time + bars * barIntervalSec());
}

// Pixel from a DOM event (relative to chart container)
function eventPixel(ev) {
  const rect = _container.getBoundingClientRect();
  return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
}

// Hit-test against persistent lines. Returns {lineId, mode} or null.
function hitTest(px, py) {
  const lines = drawingsState.lines || [];
  // Anchors first (priority)
  for (const line of lines) {
    const a = dataToScreen(toPoint(line, 'start'));
    const b = dataToScreen(toPoint(line, 'end'));
    if (!a || !b) continue;
    if (Math.hypot(px - a.x, py - a.y) <= ANCHOR_HIT) return { lineId: line.manual_line_id, mode: 'anchor_start' };
    if (Math.hypot(px - b.x, py - b.y) <= ANCHOR_HIT) return { lineId: line.manual_line_id, mode: 'anchor_end' };
  }
  // Then line bodies
  for (const line of lines) {
    const a = dataToScreen(toPoint(line, 'start'));
    const b = dataToScreen(toPoint(line, 'end'));
    if (!a || !b) continue;
    if (distancePointToSegment(px, py, a.x, a.y, b.x, b.y) <= BODY_HIT) {
      return { lineId: line.manual_line_id, mode: 'body' };
    }
  }
  return null;
}

function distancePointToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  if (dx === 0 && dy === 0) return Math.hypot(px - x1, py - y1);
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

function toPoint(line, which) {
  return which === 'start'
    ? { time: line.t_start, price: line.price_start }
    : { time: line.t_end, price: line.price_end };
}

// ─────────────────────────────────────────────────────────────
// State machine transition (single function — never set tx.state directly)
// ─────────────────────────────────────────────────────────────
function transition(next, reason = '') {
  const prev = tx.state;
  console.log(`[chart_drawing] ${prev} → ${next} (${reason})`);

  // Helpers
  const freezeChart = () => {
    try { _chart.applyOptions({ handleScroll: false, handleScale: false }); } catch {}
  };
  const thawChart = () => {
    try { _chart.applyOptions({ handleScroll: true, handleScale: true }); } catch {}
  };
  const isInteractive = (s) =>
    s === 'drawing_first_point' || s === 'drawing_second_point'
    || s === 'dragging_anchor' || s === 'dragging_line';

  // Exit actions (run even on no-op, in case state got stuck)
  if (prev === 'drawing_first_point' || prev === 'drawing_second_point') {
    if (next !== 'drawing_first_point' && next !== 'drawing_second_point') {
      tx.draftLine = null;
    }
  }
  if (prev === 'dragging_anchor' || prev === 'dragging_line') {
    if (next !== 'dragging_anchor' && next !== 'dragging_line') {
      tx.dragging = null;
    }
  }

  tx.state = next;

  // Chart freeze/thaw is IDEMPOTENT and always re-asserted, so a stuck
  // freeze from any earlier race gets recovered every time we land in idle.
  if (isInteractive(next)) {
    freezeChart();
  } else {
    thawChart();
  }

  // Cursor crosshair only during drawing modes
  if (next === 'drawing_first_point' || next === 'drawing_second_point') {
    _container.classList.add('draw-mode-crosshair');
  } else {
    _container.classList.remove('draw-mode-crosshair');
  }

  if (prev !== next) scheduleRender();
}

// ─────────────────────────────────────────────────────────────
// Event handlers
// ─────────────────────────────────────────────────────────────
function onMouseDown(ev) {
  if (ev.button !== 0) return;       // left click only
  if (tx.state === 'drawing_first_point' || tx.state === 'drawing_second_point') {
    return;  // drawing mode handles this in onClick
  }
  // Idle: check if clicking on an existing line's anchor or body
  const pixel = eventPixel(ev);
  const hit = hitTest(pixel.x, pixel.y);
  if (!hit) return;
  ev.preventDefault();

  const line = (drawingsState.lines || []).find((l) => l.manual_line_id === hit.lineId);
  if (!line) return;
  setSelectedManualLine(hit.lineId);

  const cursor = screenToData(pixel.x, pixel.y);
  if (!cursor) return;

  tx.dragging = {
    lineId: hit.lineId,
    mode: hit.mode,
    origLine: { ...line },
    grabCursor: cursor,
  };
  transition(hit.mode === 'body' ? 'dragging_line' : 'dragging_anchor', 'mousedown_hit');
}

function onMouseMove(ev) {
  // Update cursor in data coords
  const pixel = eventPixel(ev);
  const data = screenToData(pixel.x, pixel.y);
  tx.cursor = data;

  // Hover detection in idle
  if (tx.state === 'idle') {
    const hit = hitTest(pixel.x, pixel.y);
    const hovered = hit?.lineId || null;
    if (hovered !== tx.hoveredLineId) {
      tx.hoveredLineId = hovered;
      _container.style.cursor = hovered ? 'pointer' : '';
      scheduleRender();
    }
    return;
  }

  // Drawing preview
  if (tx.state === 'drawing_second_point' && data) {
    if (!tx.draftLine) return;
    tx.draftLine.end = data;
    scheduleRender();
    return;
  }
}

function onDocMouseMove(ev) {
  // Handles dragging — fires even if mouse leaves the chart
  if (tx.state !== 'dragging_anchor' && tx.state !== 'dragging_line') return;
  const pixel = eventPixel(ev);
  const data = screenToData(pixel.x, pixel.y);
  if (!data) return;
  tx.cursor = data;

  // The dragging mutation only happens here, only on transient state.
  // We don't touch drawingsState until mouseup.
  scheduleRender();
}

function onClick(ev) {
  if (tx.state !== 'drawing_first_point' && tx.state !== 'drawing_second_point') return;
  ev.preventDefault();
  ev.stopPropagation();
  const pixel = eventPixel(ev);
  let data = screenToData(pixel.x, pixel.y);

  // FALLBACK 1: if direct conversion failed at the click instant (rare race
  // with lightweight-charts internals), reuse the last cursor position from
  // mousemove. The preview line was already drawn there so it's valid.
  if (!data && tx.cursor) {
    console.warn('[chart_drawing] click->data null, fallback to tx.cursor', tx.cursor);
    data = { ...tx.cursor };
  }

  // FALLBACK 2: if cursor is also null (impossible in practice but...),
  // try ONCE more with a 1-pixel offset — sometimes the exact pixel is
  // on a boundary the API doesn't accept.
  if (!data) {
    data = screenToData(pixel.x + 1, pixel.y + 1);
  }

  if (!data) {
    console.error('[chart_drawing] could not resolve click after fallbacks', { pixel, cursor: tx.cursor });
    // Last resort: don't swallow silently. Surface this so the user
    // doesn't think the click did nothing.
    alert('画线失败: 点击位置无法解析,请向 chart 中间区域点击');
    return;
  }

  if (tx.state === 'drawing_first_point') {
    tx.draftLine = { start: data, end: data };
    transition('drawing_second_point', 'first_anchor');
    return;
  }

  if (tx.state === 'drawing_second_point') {
    if (!tx.draftLine) {
      console.error('[chart_drawing] second click but no draftLine — state corruption');
      transition('idle', 'recovery');
      return;
    }
    // Reject zero-length (user double-clicked same pixel)
    if (tx.draftLine.start.time === data.time && tx.draftLine.start.price === data.price) {
      console.warn('[chart_drawing] second click on same point as first — ignoring (use mousemove first)');
      return;
    }
    tx.draftLine.end = data;
    void commitDraft();
  }
}

function onDocMouseUp() {
  if (tx.state === 'dragging_anchor' || tx.state === 'dragging_line') {
    void commitDrag();
  }
}

function onKeyDown(ev) {
  // Don't grab keys while user is typing in an input
  const t = ev.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;

  if (ev.key === 'Escape' && tx.state !== 'idle') {
    transition('idle', 'esc');
    return;
  }
  if ((ev.key === 't' || ev.key === 'T') && tx.state === 'idle') {
    startTrendlineTool();
  }
}

function onContextMenu(ev) {
  if (tx.state !== 'idle') return;
  const pixel = eventPixel(ev);
  const hit = hitTest(pixel.x, pixel.y);
  if (!hit) return;
  ev.preventDefault();
  openContextMenu(ev, hit.lineId);
}

// ─────────────────────────────────────────────────────────────
// Commits — the only place persistent state changes from drawing
// ─────────────────────────────────────────────────────────────
async function commitDraft() {
  if (!tx.draftLine) return;
  const { start, end } = tx.draftLine;
  const [a, b] = start.time <= end.time ? [start, end] : [end, start];
  tx.draftLine = null;
  transition('idle', 'commit_done');

  // Derive role from slope (just a label; server doesn't use it for direction).
  const slopeEps = Math.max(Math.abs(a.price), Math.abs(b.price)) * 0.001;
  const delta = b.price - a.price;
  const lineSide = delta > slopeEps ? 'support'
    : (delta < -slopeEps ? 'resistance' : 'support');

  // OPTIMISTIC INSERT — show the line on the chart IMMEDIATELY with a
  // temporary client-side ID. Otherwise commit→POST→render has a 1-8s
  // window where the line is invisible (POST hits _enrich_drawing on
  // server which loads OHLCV + builds snapshot = slow).
  const tempId = `temp-${marketState.currentSymbol}-${marketState.currentInterval}-${Date.now()}`;
  const tempLine = {
    manual_line_id: tempId,
    symbol: marketState.currentSymbol,
    timeframe: marketState.currentInterval,
    side: lineSide,
    source: 'manual',
    t_start: a.time,
    t_end: b.time,
    price_start: a.price,
    price_end: b.price,
    extend_left: false,
    extend_right: true,
    locked: false,
    label: `trendline ${new Date().toISOString().slice(11, 16)}`,
    notes: '',
    created_at: Math.floor(Date.now() / 1000),
    updated_at: Math.floor(Date.now() / 1000),
    _pending: true,   // marker for UI ("saving...")
  };
  const linesNow = (drawingsState.lines || []).slice();
  linesNow.push(tempLine);
  setManualDrawings(linesNow);
  setSelectedManualLine(tempId);

  // Now POST in background. Replace temp with real on success, remove on failure.
  let real = null;
  try {
    const resp = await drawingsSvc.createManualDrawing({
      symbol: marketState.currentSymbol,
      timeframe: marketState.currentInterval,
      side: lineSide,
      t_start: a.time,
      t_end: b.time,
      price_start: a.price,
      price_end: b.price,
      extend_left: false,
      extend_right: true,
      locked: false,
      label: tempLine.label,
      notes: '',
      override_mode: 'display_only',
    });
    real = resp?.drawing;
    console.log('[chart_drawing] commit ok', real?.manual_line_id);
  } catch (err) {
    // AbortError is NOT a failure — it means a later refresh superseded
    // this POST mid-flight. The temp line is still in the store; the
    // next refresh will replace it with the real server line. Stay silent.
    const isAbort = err?.name === 'AbortError'
      || (typeof err?.message === 'string' && err.message.includes('aborted'));
    if (isAbort) {
      console.debug('[chart_drawing] commit POST aborted by superseding request — temp line kept');
      return;
    }
    console.error('[chart_drawing] commit failed', err);
    // Real error: roll back optimistic insert
    const cur = (drawingsState.lines || []).filter((l) => l.manual_line_id !== tempId);
    setManualDrawings(cur);
    if (drawingsState.selectedLineId === tempId) setSelectedManualLine(null);
    alert(`画线失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
    return;
  }
  if (!real) return;
  // Replace temp with real, preserve order
  const cur = drawingsState.lines || [];
  const next = cur.map((l) => (l.manual_line_id === tempId ? real : l));
  if (!next.find((l) => l.manual_line_id === real.manual_line_id)) {
    next.push(real);   // safety: temp wasn't in list (race), append real
  }
  setManualDrawings(next);
  if (drawingsState.selectedLineId === tempId) {
    setSelectedManualLine(real.manual_line_id);
  }

  // Auto-open TradePlan modal so the user can immediately attach an
  // order to the line they just drew. If they cancel, the line still
  // exists — they can right-click it later to attach a plan.
  try {
    await openTradePlanModal(real);
  } catch (err) {
    console.warn('[chart_drawing] trade plan modal error', err);
  }
}

async function commitDrag() {
  const drag = tx.dragging;
  if (!drag) { transition('idle', 'noop'); return; }
  const cursor = tx.cursor;
  if (!cursor) { transition('idle', 'no_cursor'); return; }

  // Compute final values
  let newStart, newEnd;
  if (drag.mode === 'anchor_start') {
    newStart = cursor;
    newEnd = { time: drag.origLine.t_end, price: drag.origLine.price_end };
  } else if (drag.mode === 'anchor_end') {
    newStart = { time: drag.origLine.t_start, price: drag.origLine.price_start };
    newEnd = cursor;
  } else { // body
    const dt = cursor.time - drag.grabCursor.time;
    const dp = cursor.price - drag.grabCursor.price;
    newStart = { time: drag.origLine.t_start + dt, price: drag.origLine.price_start + dp };
    newEnd   = { time: drag.origLine.t_end   + dt, price: drag.origLine.price_end   + dp };
  }
  // Normalize so t_start <= t_end
  if (newEnd.time < newStart.time) {
    [newStart, newEnd] = [newEnd, newStart];
  }
  // No-op?
  if (newStart.time === drag.origLine.t_start && newStart.price === drag.origLine.price_start
   && newEnd.time === drag.origLine.t_end && newEnd.price === drag.origLine.price_end) {
    transition('idle', 'noop_drag');
    return;
  }

  // Optimistic local mutation so the line stays in its new place.
  // Use setManualDrawings to publish drawings.updated so OTHER subscribers
  // (conditional_panel side panel, etc) know to re-fetch analyze stats.
  const allLines = drawingsState.lines || [];
  const idx = allLines.findIndex((l) => l.manual_line_id === drag.lineId);
  let snapshot = null;
  if (idx >= 0) {
    snapshot = { ...allLines[idx] };  // backup for rollback
    const updated = {
      ...allLines[idx],
      t_start: newStart.time, price_start: newStart.price,
      t_end:   newEnd.time,   price_end:   newEnd.price,
    };
    const next = allLines.slice();
    next[idx] = updated;
    setManualDrawings(next);
  }
  transition('idle', 'drag_committed');

  // PATCH backend
  try {
    await fetchJson(`/api/drawings/${encodeURIComponent(drag.lineId)}`, {
      method: 'PATCH',
      body: {
        t_start: newStart.time,
        t_end:   newEnd.time,
        price_start: newStart.price,
        price_end:   newEnd.price,
      },
      timeout: 8000,
    });
    console.log('[chart_drawing] drag PATCH ok');
  } catch (err) {
    console.error('[chart_drawing] drag PATCH failed', err);
    // Rollback to snapshot
    if (snapshot && idx >= 0) {
      const cur = drawingsState.lines || [];
      const rollback = cur.slice();
      const ridx = rollback.findIndex((l) => l.manual_line_id === drag.lineId);
      if (ridx >= 0) {
        rollback[ridx] = snapshot;
        setManualDrawings(rollback);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Render
// ─────────────────────────────────────────────────────────────
function scheduleRender() {
  if (_rafPending) return;
  _rafPending = true;
  requestAnimationFrame(() => {
    _rafPending = false;
    render();
  });
}

function render() {
  if (!_svg) return;
  // 1. Wipe SVG (cheap — we only have a handful of elements)
  while (_svg.firstChild) _svg.removeChild(_svg.firstChild);

  const lines = drawingsState.lines || [];
  const selId = drawingsState.selectedLineId;
  const hovId = tx.hoveredLineId;
  const dragLineId = tx.dragging?.lineId;

  // 1.5 Draw similar historical lines (FIRST so they sit beneath user lines)
  for (const sim of tx.similarLines) {
    const a = dataToScreen({ time: sim.t_start, price: sim.price_start });
    const b = dataToScreen({ time: sim.t_end, price: sim.price_end });
    if (!a || !b) continue;
    // Color by outcome: green if bounced, red if broke
    let color = 'rgba(120,120,120,0.35)';
    if (sim.outcome?.bounced) color = 'rgba(0,230,118,0.4)';
    else if (sim.outcome?.broke) color = 'rgba(255,82,82,0.4)';
    appendLine(a.x, a.y, b.x, b.y, color, 1.5, '4,3');
  }

  // 2. Draw persistent lines
  const hidden = drawingsState.hiddenLineIds || new Set();
  for (const line of lines) {
    if (hidden.has(line.manual_line_id)) continue;   // user hid this one
    const aData = toPoint(line, 'start');
    const bData = toPoint(line, 'end');

    // If this line is being dragged, show its TRANSIENT preview position
    let drawA = aData, drawB = bData;
    if (dragLineId === line.manual_line_id && tx.dragging && tx.cursor) {
      const drag = tx.dragging;
      if (drag.mode === 'anchor_start') drawA = tx.cursor;
      else if (drag.mode === 'anchor_end') drawB = tx.cursor;
      else if (drag.mode === 'body') {
        const dt = tx.cursor.time - drag.grabCursor.time;
        const dp = tx.cursor.price - drag.grabCursor.price;
        drawA = { time: drag.origLine.t_start + dt, price: drag.origLine.price_start + dp };
        drawB = { time: drag.origLine.t_end + dt,   price: drag.origLine.price_end + dp };
      }
    }

    const a = dataToScreen(drawA);
    const b = dataToScreen(drawB);
    if (!a || !b) continue;

    const isSel = line.manual_line_id === selId;
    const isHov = line.manual_line_id === hovId;
    const color = (isSel || isHov) ? 'rgba(251,191,36,1)' : 'rgba(251,191,36,0.45)';
    const width = isSel ? 3 : (isHov ? 2.5 : 1.8);

    // Visible line
    appendLine(a.x, a.y, b.x, b.y, color, width);

    // Anchors only when selected or hovered
    if (isSel || isHov) {
      appendCircle(a.x, a.y, ANCHOR_R, color);
      appendCircle(b.x, b.y, ANCHOR_R, color);
    }
  }

  // 3. Draft preview (draws on top of everything)
  if (tx.draftLine) {
    const a = dataToScreen(tx.draftLine.start);
    const b = dataToScreen(tx.draftLine.end);
    if (a && b) {
      appendLine(a.x, a.y, b.x, b.y, 'rgba(251,191,36,0.95)', 2, '6,4');
      appendCircle(a.x, a.y, ANCHOR_R, 'rgba(251,191,36,1)');
      appendCircle(b.x, b.y, ANCHOR_R, 'rgba(251,191,36,1)');
    }
  } else if (tx.state === 'drawing_first_point' && tx.cursor) {
    // Show a tiny preview dot at cursor while waiting for first click
    const c = dataToScreen(tx.cursor);
    if (c) appendCircle(c.x, c.y, 4, 'rgba(251,191,36,0.7)');
  }
}

function appendLine(x1, y1, x2, y2, stroke, strokeWidth, dasharray = null) {
  const ln = document.createElementNS(SVG_NS, 'line');
  ln.setAttribute('x1', x1); ln.setAttribute('y1', y1);
  ln.setAttribute('x2', x2); ln.setAttribute('y2', y2);
  ln.setAttribute('stroke', stroke);
  ln.setAttribute('stroke-width', String(strokeWidth));
  if (dasharray) ln.setAttribute('stroke-dasharray', dasharray);
  ln.setAttribute('stroke-linecap', 'round');
  _svg.appendChild(ln);
}

function appendCircle(cx, cy, r, fill) {
  const c = document.createElementNS(SVG_NS, 'circle');
  c.setAttribute('cx', cx); c.setAttribute('cy', cy);
  c.setAttribute('r', String(r));
  c.setAttribute('fill', fill);
  c.setAttribute('stroke', '#fff');
  c.setAttribute('stroke-width', '1.5');
  c.setAttribute('filter', 'drop-shadow(0 0 2px rgba(0,0,0,0.8))');
  _svg.appendChild(c);
}

// ─────────────────────────────────────────────────────────────
// Right-click context menu
// ─────────────────────────────────────────────────────────────
let _menu = null;
function openContextMenu(ev, lineId) {
  closeContextMenu();
  _menu = document.createElement('div');
  Object.assign(_menu.style, {
    position: 'fixed',
    left: ev.clientX + 'px',
    top: ev.clientY + 'px',
    background: '#141a26',
    border: '1px solid #2a3548',
    borderRadius: '4px',
    padding: '4px 0',
    minWidth: '140px',
    boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
    zIndex: '9999',
    fontSize: '12px',
    color: '#d8dde8',
  });
  _menu.innerHTML = `
    <div data-act="select" style="padding:6px 14px;cursor:pointer">选中此线</div>
    <div data-act="create_trade_plan" style="padding:6px 14px;cursor:pointer;color:#38bdf8">+ 创建交易计划</div>
    <div data-act="delete" style="padding:6px 14px;cursor:pointer;color:#ff5252">删除此线</div>
  `;
  _menu.addEventListener('click', async (e) => {
    const item = e.target.closest('[data-act]');
    if (!item) return;
    const act = item.dataset.act;
    closeContextMenu();
    if (act === 'select') {
      setSelectedManualLine(lineId);
    } else if (act === 'create_trade_plan') {
      const line = (drawingsState.lines || []).find((l) => l.manual_line_id === lineId);
      if (line) {
        try { await openTradePlanModal(line); }
        catch (err) { console.warn('[chart_drawing] trade plan modal', err); }
      }
    } else if (act === 'delete') {
      await deleteLine(lineId);
    }
  });
  _menu.querySelectorAll('[data-act]').forEach((el) => {
    el.addEventListener('mouseenter', () => { el.style.background = '#1e2738'; });
    el.addEventListener('mouseleave', () => { el.style.background = 'transparent'; });
  });
  document.body.appendChild(_menu);
  setTimeout(() => {
    document.addEventListener('mousedown', onOutsideMenu, { once: true });
  }, 0);
}
function onOutsideMenu(ev) {
  if (_menu && !_menu.contains(ev.target)) closeContextMenu();
}
function closeContextMenu() {
  if (_menu && _menu.parentNode) _menu.parentNode.removeChild(_menu);
  _menu = null;
}

async function deleteLine(lineId) {
  const keep = (drawingsState.lines || []).filter((l) => l.manual_line_id !== lineId);
  setManualDrawings(keep);
  try {
    await fetchJson(`/api/drawings/${encodeURIComponent(lineId)}`, { method: 'DELETE', timeout: 8000 });
  } catch (err) {
    alert(`删除失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
  }
}
