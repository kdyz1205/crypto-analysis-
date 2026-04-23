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
//      drawing_second_point ──[click]──> idle                 (commit once)
//      drawing_* ──[Esc]──> idle
//      idle ──[mousedown anchor]──> dragging_anchor
//      idle ──[mousedown body]──> dragging_line
//      dragging_* ──[mousemove]──> stays (update dragging preview)
//      dragging_* ──[mouseup]──> idle (PATCH commit)

import { drawingsState, setManualDrawings, setSelectedManualLine } from '../../state/drawings.js';
import { marketState } from '../../state/market.js';
import { publish, subscribe } from '../../util/events.js';
import { fetchJson } from '../../util/fetch.js';
import * as drawingsSvc from '../../services/drawings.js';
import { openTradePlanModal, openSetupPickerPopup } from './trade_plan_modal.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const ANCHOR_R = 6;        // visible handle radius
const ANCHOR_HIT = 12;     // hit test radius for anchors
const BODY_HIT = 8;        // hit test distance for line body
// Thinner than the 1.8 original. Iterated 2026-04-20:
//   1.8 → 1.0 → 0.3 (final). User asked for "base = thinnest".
// Presets now: 0.3 (default/base) / 1.0 (medium) / 2.0 (bold).
const DEFAULT_LINE_WIDTH = 0.3;

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
// True while the T key is physically held down. Prevents the OS keyboard
// auto-repeat (firing keydown every ~50ms) from re-entering draw mode the
// instant a line is committed — the symptom was "I need to draw two
// consecutive lines before draw mode exits" (reported 2026-04-20).
let _tKeyHeld = false;

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
  document.addEventListener('keyup',     onKeyUp);

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
  if (isFutureX(x)) {
    time = xToFutureTime(x);
  } else if (time == null) {
    time = xToFutureTime(x);
  }
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

function isFutureX(x) {
  const c = marketState.lastCandles || [];
  if (!c.length || !_chart) return false;
  const last = c[c.length - 1];
  const ts = _chart.timeScale();
  const lastX = ts.timeToCoordinate(last.time);
  if (lastX == null) return false;
  const spacing = ts.options().barSpacing || 8;
  return x > lastX + spacing * 0.5;
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
    const [displayA, displayB] = displayEndpoints(line);
    const a = dataToScreen(displayA);
    const b = dataToScreen(displayB);
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

function numericTime(value) {
  if (value == null) return null;
  if (typeof value === 'object') return Number(value.timestamp ?? value.time ?? value);
  return Number(value);
}

function projectLinePrice(a, b, time) {
  const span = b.time - a.time;
  if (!span) return a.price;
  const slope = (b.price - a.price) / span;
  return a.price + slope * (time - a.time);
}

function visibleTimeBounds(fallbackA, fallbackB) {
  const candles = marketState.lastCandles || [];
  let from = candles.length ? Number(candles[0].time) : Number(fallbackA.time);
  let to = candles.length ? Number(candles[candles.length - 1].time) + barIntervalSec() * 8 : Number(fallbackB.time);
  try {
    const vr = _chart?.timeScale?.().getVisibleRange?.();
    const vf = numericTime(vr?.from);
    const vt = numericTime(vr?.to);
    if (Number.isFinite(vf)) from = vf;
    if (Number.isFinite(vt)) to = vt;
  } catch {}
  return { from, to };
}

function displayEndpoints(line, start = toPoint(line, 'start'), end = toPoint(line, 'end')) {
  if (!start || !end) return [start, end];
  let a = start;
  let b = end;
  if (b.time < a.time) [a, b] = [b, a];

  let from = a.time;
  let to = b.time;
  const bounds = visibleTimeBounds(a, b);
  if (line?.extend_left && Number.isFinite(bounds.from) && bounds.from < from) {
    from = Math.floor(bounds.from);
  }
  if (line?.extend_right !== false && Number.isFinite(bounds.to) && bounds.to > to) {
    to = Math.floor(bounds.to);
  }

  return [
    { time: from, price: projectLinePrice(a, b, from) },
    { time: to, price: projectLinePrice(a, b, to) },
  ];
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
  publish('drawtool.mode', {
    state: next,
    previous: prev,
    reason,
    active: next === 'drawing_first_point' || next === 'drawing_second_point',
    phase: next === 'drawing_first_point'
      ? 'picking_first'
      : (next === 'drawing_second_point' ? 'picking_second' : 'idle'),
  });

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

  // BUGFIX: while waiting for the first click, the preview dot is drawn
  // from tx.cursor (see render() step 3 for drawing_first_point). Without
  // scheduling a render here, the dot freezes at whatever position had the
  // last render even though cursor moves, producing the "cursor is far from
  // the mouse" symptom. RAF coalescer in scheduleRender prevents spam.
  if (tx.state === 'drawing_first_point') {
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
  if (ev.key === 't' || ev.key === 'T') {
    // Honor only the FIRST keydown in a press — subsequent auto-repeat
    // keydowns are ignored so that committing a line (state → idle) does
    // not immediately re-arm another draw from the still-held-down key.
    if (ev.repeat || _tKeyHeld) return;
    _tKeyHeld = true;
    if (tx.state === 'idle') {
      startTrendlineTool();
    }
  }
  // TradingView-style "reset chart" — re-enables autoScale + fitContent.
  // Only fires when NOT in a drawing state so it doesn't interfere.
  // User 2026-04-22 asked for a way back after dragging disabled auto-fit.
  if ((ev.key === 'r' || ev.key === 'R') && tx.state === 'idle') {
    if (ev.repeat) return;
    // Ignore if user is typing in any input / textarea / contenteditable
    const tgt = ev.target;
    if (tgt && (tgt.tagName === 'INPUT' || tgt.tagName === 'TEXTAREA' || tgt.isContentEditable)) return;
    import('../chart.js').then((m) => m.resetChartViewport?.()).catch(() => {});
  }
}

function onKeyUp(ev) {
  if (ev.key === 't' || ev.key === 'T') {
    _tKeyHeld = false;
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

  // Derive role: look at where the CURRENT PRICE sits relative to the line,
  // not the slope direction. A rising line CONNECTING TWO HIGHS is a
  // RESISTANCE (price below, user shorts). A rising line connecting two
  // lows is a SUPPORT (price above, user longs). Slope alone can't tell.
  //
  // Old slope-only rule mislabeled ascending resistance as support, which
  // flipped the trade_plan_modal default direction from short→long and
  // caused user to submit buys when they intended sells (2026-04-20 bug).
  let lineSide;
  try {
    const data = _candleSeries?.data?.() || [];
    const latest = data.length > 0 ? data[data.length - 1] : null;
    const lastClose = latest?.close ?? latest?.value ?? null;
    // Line price projected at the later-in-time anchor; that's the freshest
    // "meaning" of the line relative to current price.
    const linePriceNow = Number(b.price);
    if (lastClose != null && isFinite(lastClose) && linePriceNow > 0) {
      lineSide = lastClose < linePriceNow ? 'resistance' : 'support';
    } else {
      // Fallback: slope-based (old behavior) when we can't read close.
      const slopeEps = Math.max(Math.abs(a.price), Math.abs(b.price)) * 0.001;
      const delta = b.price - a.price;
      lineSide = delta > slopeEps ? 'support'
        : (delta < -slopeEps ? 'resistance' : 'support');
    }
  } catch {
    lineSide = 'support';
  }

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
    extend_right: false,
    locked: false,
    label: `trendline ${new Date().toISOString().slice(11, 16)}`,
    notes: '',
    line_width: DEFAULT_LINE_WIDTH,
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
      extend_right: false,
      locked: false,
      label: tempLine.label,
      notes: '',
      line_width: DEFAULT_LINE_WIDTH,
      override_mode: 'display_only',
    });
    real = resp?.drawing ? {
      ...resp.drawing,
      t_start: tempLine.t_start,
      t_end: tempLine.t_end,
      price_start: tempLine.price_start,
      price_end: tempLine.price_end,
      line_width: tempLine.line_width,
    } : null;
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

  // Stay idle after one line. The user explicitly chooses the next action:
  // press T to draw another line, or right-click this line to create a plan.
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

  // PATCH backend. Larger timeout (25s) because the handler does
  // force_replan_line which can call Bitget cancel+replace on every
  // attached cond — slow when >3 conds or Bitget is backed up.
  // User 2026-04-21: old 8s timeout was aborting legit PATCHes,
  // frontend rolled back to snapshot, line "jumped" to original
  // position after each drag — making fine adjustment impossible.
  // Also: on abort specifically, DO NOT roll back. The request may
  // have actually reached the server and stored; a rollback plus the
  // server's successful state creates a ping-pong where next refresh
  // restores the server's (new) state and "un-jumps". Simpler: keep
  // optimistic local state and show a console warn. If the PATCH
  // truly didn't land, next manual refresh will correct it.
  try {
    await fetchJson(`/api/drawings/${encodeURIComponent(drag.lineId)}`, {
      method: 'PATCH',
      body: {
        t_start: newStart.time,
        t_end:   newEnd.time,
        price_start: newStart.price,
        price_end:   newEnd.price,
      },
      timeout: 25000,
    });
    console.log('[chart_drawing] drag PATCH ok');
  } catch (err) {
    const msg = String(err?.message || err || '');
    const isAbort = /abort|timeout|signal/i.test(msg);
    if (isAbort) {
      // Keep the optimistic local state. Server may have actually
      // accepted — or not — but rolling back here makes adjustment
      // impossible (line jumps back on every slow network blip).
      console.warn('[chart_drawing] drag PATCH timed out, keeping local state', err);
      return;
    }
    // Non-abort errors (e.g. 404, 400) imply the server didn't accept
    // the change. Roll back to previous position so user sees a real
    // failure and can fix the underlying issue.
    console.error('[chart_drawing] drag PATCH rejected', err);
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
  // User 2026-04-21: add a small ★ marker near each similar line's end so
  // they're visually identifiable as "相似" and not confused with real lines.
  for (const sim of tx.similarLines) {
    const a = dataToScreen({ time: sim.t_start, price: sim.price_start });
    const b = dataToScreen({ time: sim.t_end, price: sim.price_end });
    if (!a || !b) continue;
    // Color by outcome: green if bounced, red if broke
    let color = 'rgba(120,120,120,0.55)';
    let label = '★';          // default: unknown outcome
    let labelColor = 'rgba(200,200,200,0.85)';
    if (sim.outcome?.bounced) {
      color = 'rgba(0,230,118,0.55)';
      label = '★✓';           // bounced = reached RR target
      labelColor = 'rgba(0,230,118,1)';
    } else if (sim.outcome?.broke) {
      color = 'rgba(255,82,82,0.55)';
      label = '★✗';           // broke
      labelColor = 'rgba(255,82,82,1)';
    }
    appendLine(a.x, a.y, b.x, b.y, color, 1.5, '4,3');
    // Label at the end anchor, slightly offset so it doesn't overlap the line
    appendText(b.x + 4, b.y, label, labelColor, 10);
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

    const [displayA, displayB] = displayEndpoints(line, drawA, drawB);
    const a = dataToScreen(displayA);
    const b = dataToScreen(displayB);
    if (!a || !b) continue;

    const isSel = line.manual_line_id === selId;
    const isHov = line.manual_line_id === hovId;
    const isAuto = line.source === 'auto_triggered';
    // Auto-triggered (system-drawn) lines use a warm red/amber palette so they
    // are immediately distinguishable from user-drawn manual lines.
    let color;
    if (isSel || isHov) {
      color = isAuto ? 'rgba(248,113,113,1)' : 'rgba(251,191,36,1)';
    } else {
      color = isAuto
        ? (line.side === 'resistance' ? 'rgba(248,113,113,0.85)' : 'rgba(250,204,21,0.85)')
        : 'rgba(251,191,36,0.45)';
    }
    const baseWidth = lineStrokeWidth(line);
    const width = isSel ? baseWidth + 1.0 : (isHov ? baseWidth + 0.6 : baseWidth);
    // Dashed for auto lines → "not drawn by you, drawn by the system".
    const dasharray = isAuto ? '8,5' : null;

    // Visible line
    appendLine(a.x, a.y, b.x, b.y, color, width, dasharray);

    // Anchors only when selected or hovered
    if (isSel || isHov) {
      const anchorA = dataToScreen(drawA);
      const anchorB = dataToScreen(drawB);
      if (anchorA) appendCircle(anchorA.x, anchorA.y, ANCHOR_R, color);
      if (anchorB) appendCircle(anchorB.x, anchorB.y, ANCHOR_R, color);
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

function appendText(x, y, text, fill = '#fbbf24', fontSize = 10) {
  const t = document.createElementNS(SVG_NS, 'text');
  t.setAttribute('x', String(x));
  t.setAttribute('y', String(y));
  t.setAttribute('fill', fill);
  t.setAttribute('font-size', String(fontSize));
  t.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
  t.setAttribute('font-weight', '600');
  t.setAttribute('text-anchor', 'start');
  t.setAttribute('dominant-baseline', 'middle');
  t.setAttribute('pointer-events', 'none');
  t.textContent = text;
  _svg.appendChild(t);
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
    <div data-act="quick_trade" style="padding:6px 14px;cursor:pointer;color:#00e676;font-weight:600">⚡ 交易 (快捷挂单)</div>
    <div data-act="create_trade_plan" style="padding:6px 14px;cursor:pointer;color:#38bdf8">+ 创建交易计划 (完整配置)</div>
    <div data-act="add_alert" style="padding:6px 14px;cursor:pointer;color:#fbbf24">🔔 添加价格警报</div>
    <div style="height:1px;background:#2a3548;margin:4px 0"></div>
    <div style="padding:5px 14px;color:#8a92a5">标记形态质量 (喂给 ML)</div>
    <div data-act="label_good" style="padding:6px 14px;cursor:pointer;color:#00e676">⭐ 好形态</div>
    <div data-act="label_mediocre" style="padding:6px 14px;cursor:pointer;color:#94a3b8">〜 一般</div>
    <div data-act="label_bad" style="padding:6px 14px;cursor:pointer;color:#ff5252">✗ 差形态</div>
    <div style="height:1px;background:#2a3548;margin:4px 0"></div>
    <div style="padding:5px 14px;color:#8a92a5">线宽</div>
    <div data-act="set_width" data-width="0.3" style="padding:6px 14px;cursor:pointer">细 (默认)</div>
    <div data-act="set_width" data-width="1.0" style="padding:6px 14px;cursor:pointer">中</div>
    <div data-act="set_width" data-width="2.0" style="padding:6px 14px;cursor:pointer">粗</div>
    <div data-act="delete" style="padding:6px 14px;cursor:pointer;color:#ff5252">删除此线</div>
  `;
  _menu.addEventListener('click', async (e) => {
    const item = e.target.closest('[data-act]');
    if (!item) return;
    const act = item.dataset.act;
    closeContextMenu();
    if (act === 'select') {
      setSelectedManualLine(lineId);
    } else if (act === 'quick_trade') {
      const line = (drawingsState.lines || []).find((l) => l.manual_line_id === lineId);
      if (line) {
        // 2026-04-21 user spec: after drawing a line, "交易" button
        // opens a small popup next to the menu with Direction toggle +
        // Reverse toggle + setup drop-down. Pick setup → instant place.
        // No full configure modal. Under 5 clicks from line → Bitget.
        try {
          const { openQuickTradePopup } = await import('./trade_plan_modal.js');
          await openQuickTradePopup(line, ev.clientX, ev.clientY);
        } catch (err) { console.warn('[chart_drawing] quick-trade popup', err); }
      }
    } else if (act === 'create_trade_plan') {
      const line = (drawingsState.lines || []).find((l) => l.manual_line_id === lineId);
      if (line) {
        // 2026-04-20: open a lightweight setup picker first; picker
        // decides whether to drop straight into the full Trade Plan
        // modal pre-filled with the chosen setup.
        try {
          await openSetupPickerPopup(line, ev.clientX, ev.clientY);
        } catch (err) { console.warn('[chart_drawing] setup picker', err); }
      }
    } else if (act === 'add_alert') {
      const line = (drawingsState.lines || []).find((l) => l.manual_line_id === lineId);
      if (line) openAlertDialog(line);
    } else if (act === 'set_width') {
      await setLineWidth(lineId, Number(item.dataset.width || DEFAULT_LINE_WIDTH));
    } else if (act === 'label_good' || act === 'label_mediocre' || act === 'label_bad') {
      // User 2026-04-22: rate pattern quality so ML learns from the
      // ones the user considers visually valid.
      const quality = act === 'label_good' ? 'good'
                    : act === 'label_bad'  ? 'bad'
                    : 'mediocre';
      const label = quality === 'good' ? '⭐ 好形态'
                  : quality === 'bad'  ? '✗ 差形态'
                  : '〜 一般';
      const patternType = prompt(
        `标记为"${label}"。可选: 输入形态类型 (如 双顶/三角/通道/头肩, 留空跳过)`,
        ''
      );
      if (patternType === null) return;  // cancelled
      const notes = quality === 'good'
        ? (prompt('为什么觉得这个形态好? (可选, 随便写)', '') || '')
        : '';
      try {
        const { fetchJson } = await import('../../util/fetch.js');
        await fetchJson(`/api/drawings/${encodeURIComponent(lineId)}/label`, {
          method: 'POST',
          body: {
            quality,
            pattern_type: patternType.trim() || null,
            notes: notes.trim() || null,
          },
          timeout: 10000,
        });
        // Small toast via title flash; no UI noise
        console.log(`[label] saved: ${lineId} → ${quality} (${patternType || 'no-type'})`);
      } catch (err) {
        alert(`标记失败: ${err?.message || err}`);  // SAFE: alert renders text
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

  // Clamp to viewport so the menu never gets clipped by the chart's
  // overflow:hidden or the browser bottom edge. Lines drawn near the
  // bottom of the chart used to open menus that extended below the
  // visible area — user had to zoom out to get access (2026-04-20).
  try {
    const rect = _menu.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const pad = 8;
    let newLeft = ev.clientX;
    let newTop = ev.clientY;
    if (newLeft + rect.width + pad > vw) newLeft = vw - rect.width - pad;
    if (newTop + rect.height + pad > vh) newTop = vh - rect.height - pad;
    if (newLeft < pad) newLeft = pad;
    if (newTop < pad) newTop = pad;
    _menu.style.left = `${newLeft}px`;
    _menu.style.top = `${newTop}px`;
  } catch {}

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

// ─── Alert dialog ──────────────────────────────────────────
function openAlertDialog(line) {
  const overlay = document.createElement('div');
  Object.assign(overlay.style, {
    position: 'fixed', inset: '0', background: 'rgba(0,0,0,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: '10000',
  });
  const slope = (line.price_end - line.price_start) / (line.t_end - line.t_start) || 0;
  const intercept = line.price_start - slope * line.t_start;
  const kind = line.price_end >= line.price_start ? 'support' : 'resistance';
  overlay.innerHTML = `
    <div style="background:#141a26;border:1px solid #2a3548;border-radius:8px;padding:20px;min-width:300px;color:#d8dde8">
      <h3 style="margin:0 0 12px;color:#38bdf8">🔔 添加价格警报</h3>
      <div style="margin-bottom:10px;font-size:12px;color:#8899aa">
        ${line.symbol || ''} · ${kind} 线
      </div>
      <label style="font-size:12px;display:block;margin-bottom:4px">触发范围 (%)</label>
      <input id="alert-threshold" type="number" value="0.3" step="0.05" min="0.01" max="5"
        style="width:100%;padding:6px;background:#0d1117;border:1px solid #2a3548;color:#fff;border-radius:4px;margin-bottom:10px">
      <label style="font-size:12px;display:block;margin-bottom:4px">模式</label>
      <select id="alert-mode" style="width:100%;padding:6px;background:#0d1117;border:1px solid #2a3548;color:#fff;border-radius:4px;margin-bottom:10px">
        <option value="single">单次触发</option>
        <option value="repeat">重复触发 (5分钟冷却)</option>
      </select>
      <label style="font-size:12px;display:block;margin-bottom:4px">备注 (可选)</label>
      <input id="alert-label" type="text" placeholder="我的支撑线警报"
        style="width:100%;padding:6px;background:#0d1117;border:1px solid #2a3548;color:#fff;border-radius:4px;margin-bottom:14px">
      <div style="display:flex;gap:8px">
        <button id="alert-confirm" style="flex:1;padding:8px;background:#38bdf8;color:#000;border:none;border-radius:4px;cursor:pointer;font-weight:bold">确认添加</button>
        <button id="alert-cancel" style="flex:1;padding:8px;background:#2a3548;color:#d8dde8;border:none;border-radius:4px;cursor:pointer">取消</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.querySelector('#alert-cancel').onclick = () => overlay.remove(); // SAFE: programmatic, not inline HTML
  overlay.querySelector('#alert-confirm').onclick = async () => { // SAFE: programmatic, not inline HTML
    const threshold = parseFloat(overlay.querySelector('#alert-threshold').value) / 100;
    const mode = overlay.querySelector('#alert-mode').value;
    const label = overlay.querySelector('#alert-label').value;
    try {
      const sym = drawingsState.currentSymbol || line.symbol || 'BTC/USDT:USDT';
      const tf = drawingsState.currentTimeframe || line.timeframe || '4h';
      const resp = await fetch('/api/alerts/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: sym, timeframe: tf,
          slope, intercept, kind, mode, label,
          threshold,
        }),
      });
      const data = await resp.json();
      if (data.ok) {
        overlay.remove();
        console.log('[alert] added:', data.alert_id);
      }
    } catch (err) {
      console.error('[alert] failed:', err);
    }
  };
  overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
}

async function deleteLine(lineId) {
  if (!lineId) return;
  const previous = drawingsState.lines || [];
  const keep = (drawingsState.lines || []).filter((l) => l.manual_line_id !== lineId);
  setManualDrawings(keep);
  if (drawingsState.selectedLineId === lineId) setSelectedManualLine(null);
  if (lineId.startsWith('temp-')) return;
  try {
    await drawingsSvc.deleteManualDrawing(lineId);
  } catch (err) {
    setManualDrawings(previous);
    alert(`删除失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
  }
}

async function setLineWidth(lineId, width) {
  // Min 0.2 (hair-thin, for 0.1% buffer scales). Cap 8 px.
  const lineWidth = Math.max(0.2, Math.min(Number(width) || DEFAULT_LINE_WIDTH, 8));
  const lines = drawingsState.lines || [];
  const idx = lines.findIndex((l) => l.manual_line_id === lineId);
  if (idx < 0) return;
  const previous = lines.slice();
  const next = lines.slice();
  next[idx] = { ...next[idx], line_width: lineWidth };
  setManualDrawings(next);
  if (lineId.startsWith('temp-')) return;
  try {
    await drawingsSvc.updateManualDrawing(lineId, { line_width: lineWidth });
  } catch (err) {
    setManualDrawings(previous);
    alert(`线宽保存失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
  }
}

function lineStrokeWidth(line) {
  const width = Number(line?.line_width);
  if (!Number.isFinite(width)) return DEFAULT_LINE_WIDTH;
  return Math.max(0.5, Math.min(width, 8));
}
