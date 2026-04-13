// frontend/js/workbench/drawings/draw_tool.js
//
// TradingView-style drawing tool built on top of lightweight-charts.
//
// Lifecycle:
//   idle → user clicks toolbar icon → mode = "picking_first"
//   cursor becomes crosshair, first click captures anchor A
//   → mode = "picking_second", mousemove draws ghost line A → cursor
//   second click captures anchor B, POSTs to /api/drawings, creates
//   permanent line series, mode = "idle"
//
//   Escape key at any point cancels and removes the ghost.
//
// Extensibility:
//   - activeTool is a string ('trendline' | 'horizontal' | future 'channel' etc.)
//   - registerTool(name, { onFirstClick, onMove, onSecondClick }) can add new tools
//   - The permanent line creation goes through drawingsSvc.createManualDrawing
//     so it flows into the existing conditional-panel analyze pipeline
//     (drawings.updated event → panel auto-analyzes → user can create conditional)
//
// This file is the ONLY place that talks to lightweight-charts for drawing.
// The controller + panel can be refactored later to use this tool exclusively.

import { marketState } from '../../state/market.js';
import * as drawingsSvc from '../../services/drawings.js';
import { publish } from '../../util/events.js';

// ─────────────────────────────────────────────────────────────
// Module state
// ─────────────────────────────────────────────────────────────
let _chart = null;
let _candleSeries = null;
let _container = null;      // DOM element that owns the chart canvas

// Tool state
let _activeTool = null;      // null | 'trendline-support' | 'trendline-resistance' | 'horizontal'
let _phase = 'idle';         // 'idle' | 'picking_first' | 'picking_second'
let _anchorA = null;         // { time, price }
let _ghostSeries = null;     // LineSeries currently rendered as the rubber-band preview

// RAF throttle for crosshair move: LC fires it at 60fps+ which freezes the
// browser when setData forces a full redraw on 3000+ candles. We coalesce
// all moves within a frame into one pending update.
let _pendingB = null;
let _rafId = null;

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────
export function initDrawTool(chart, candleSeries, container) {
  _chart = chart;
  _candleSeries = candleSeries;
  _container = container;

  // Chart click: both phases handled here
  if (typeof chart.subscribeClick === 'function') {
    chart.subscribeClick((param) => onChartClick(param));
  }

  // Crosshair move: drives the rubber-band preview
  if (typeof chart.subscribeCrosshairMove === 'function') {
    chart.subscribeCrosshairMove((param) => onCrosshairMove(param));
  }

  // Keyboard: Escape to cancel
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _phase !== 'idle') {
      cancelDrawing('user_escape');
    }
  });

  console.log('[draw_tool] initialized');
}

/**
 * Activate a drawing tool. Call this from the toolbar button click.
 *   startDrawing('trendline-support')
 *   startDrawing('trendline-resistance')
 *   startDrawing('horizontal')
 */
export function startDrawing(tool) {
  if (!_chart) {
    console.warn('[draw_tool] not initialized yet');
    return;
  }
  cancelDrawing('new_tool_started');
  _activeTool = tool;
  _phase = 'picking_first';
  if (_container) _container.classList.add('draw-mode-crosshair');
  publish('drawtool.mode', { tool, phase: _phase });
  console.log(`[draw_tool] ▶ startDrawing(${tool}) phase=picking_first`);
}

export function cancelDrawing(reason = 'cancel') {
  if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
  _pendingB = null;
  if (_ghostSeries) {
    try { _chart.removeSeries(_ghostSeries); } catch {}
    _ghostSeries = null;
  }
  _activeTool = null;
  _phase = 'idle';
  _anchorA = null;
  if (_container) _container.classList.remove('draw-mode-crosshair');
  publish('drawtool.mode', { tool: null, phase: 'idle', reason });
  console.log(`[draw_tool] ■ cancelDrawing(${reason})`);
}

export function getDrawState() {
  return { activeTool: _activeTool, phase: _phase, anchorA: _anchorA };
}

// ─────────────────────────────────────────────────────────────
// Event handlers
// ─────────────────────────────────────────────────────────────
function onChartClick(param) {
  console.log(`[draw_tool] chart click phase=${_phase}`, {
    time: param?.time,
    hasPoint: !!param?.point,
    pointX: param?.point?.x,
    pointY: param?.point?.y,
  });
  if (_phase === 'idle') return;

  const anchor = resolveAnchor(param);
  if (!anchor) {
    console.warn('[draw_tool] ⚠ could not resolve click to (time, price)', param);
    return;
  }

  if (_phase === 'picking_first') {
    _anchorA = anchor;
    _phase = 'picking_second';
    // Pre-create the ghost series NOW so the first mousemove doesn't have
    // to allocate inside a RAF callback.
    ensureGhostSeries();
    publish('drawtool.mode', { tool: _activeTool, phase: _phase, anchor });
    console.log(`[draw_tool] ▶ anchor A captured`, anchor);
    return;
  }

  if (_phase === 'picking_second') {
    console.log(`[draw_tool] ▶ anchor B captured, committing`, anchor);
    commitLine(_anchorA, anchor);
    cancelDrawing('committed');
  }
}

function onCrosshairMove(param) {
  if (_phase !== 'picking_second' || !_anchorA) return;
  if (!param || !param.point) return;

  // Convert pixel x → time, pixel y → price (cheap, O(1) each)
  const ts = _chart.timeScale();
  let time = param.time;
  if (time == null) {
    try { time = ts.coordinateToTime(param.point.x); } catch {}
  }
  if (time == null) return;

  let price = null;
  if (_candleSeries && typeof _candleSeries.coordinateToPrice === 'function') {
    try { price = _candleSeries.coordinateToPrice(param.point.y); } catch {}
  }
  if (price == null) return;

  // RAF-throttle: store the latest pending endpoint and schedule one redraw.
  // Coalesces 60+ moves/sec into 1 update/frame. This is the fix for the
  // "page freezes while drawing" bug — setData on every raw move forces
  // LC to redraw all 3000+ candles at 60fps, killing the main thread.
  _pendingB = { time: Number(time), price: Number(price) };
  if (_rafId == null) {
    _rafId = requestAnimationFrame(() => {
      _rafId = null;
      if (_pendingB && _anchorA && _phase === 'picking_second') {
        updateGhost(_anchorA, _pendingB);
      }
    });
  }
}

function ensureGhostSeries() {
  if (_ghostSeries || !_chart) return;
  try {
    _ghostSeries = _chart.addLineSeries({
      color: ghostColor(),
      lineWidth: 2,
      lineStyle: 2,          // dashed
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      autoscaleInfoProvider: () => ({ priceRange: null }),
    });
    console.log('[draw_tool] ghost series created');
  } catch (e) {
    console.error('[draw_tool] addLineSeries failed', e);
  }
}

// ─────────────────────────────────────────────────────────────
// Ghost line (rubber-band preview)
// ─────────────────────────────────────────────────────────────
function updateGhost(a, b) {
  if (!_chart || !a || !b) return;
  ensureGhostSeries();
  if (!_ghostSeries) return;

  // LC requires ASCENDING time order
  const pts = (a.time <= b.time)
    ? [{ time: a.time, value: a.price }, { time: b.time, value: b.price }]
    : [{ time: b.time, value: b.price }, { time: a.time, value: a.price }];

  // Also ensure both times are integer seconds (LC is picky about types)
  pts[0].time = Math.floor(Number(pts[0].time));
  pts[1].time = Math.floor(Number(pts[1].time));
  if (pts[0].time === pts[1].time) {
    // A single-point line is illegal; skip
    return;
  }

  try {
    _ghostSeries.setData(pts);
  } catch (e) {
    console.warn('[draw_tool] ghost setData failed', e, pts);
  }
}

function ghostColor() {
  if (_activeTool === 'trendline-support') return 'rgba(0, 230, 118, 0.85)';
  if (_activeTool === 'trendline-resistance') return 'rgba(255, 82, 82, 0.85)';
  return 'rgba(255, 255, 255, 0.6)';
}

// ─────────────────────────────────────────────────────────────
// Commit to backend
// ─────────────────────────────────────────────────────────────
async function commitLine(a, b) {
  const side = _activeTool === 'trendline-resistance' ? 'resistance' : 'support';
  // Ensure a.time < b.time
  const [start, end] = (a.time <= b.time) ? [a, b] : [b, a];

  try {
    await drawingsSvc.createManualDrawing({
      symbol: marketState.currentSymbol,
      timeframe: marketState.currentInterval,
      side,
      t_start: Math.floor(Number(start.time)),
      t_end: Math.floor(Number(end.time)),
      price_start: Number(start.price),
      price_end: Number(end.price),
      extend_left: false,
      extend_right: true,
      locked: false,
      label: `${side} ${new Date().toISOString().slice(11, 16)}`,
      notes: '',
      override_mode: 'display_only',
    });
    // Publish an event so the conditional panel + manual_trendline_controller
    // refresh their state. The existing refreshManualDrawings() in the
    // controller will pick this up via 'drawings.updated'.
    publish('drawtool.committed', { side, start, end });
    // Kick a reload — the controller subscribes to symbol/interval change
    // events but not to drawtool.committed specifically, so force it:
    try {
      const mod = await import('./manual_trendline_controller.js');
      await mod.refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    } catch (e) { console.warn('[draw_tool] refresh fail', e); }
  } catch (err) {
    console.error('[draw_tool] commit failed', err);
    alert(`画线失败: ${err?.message || err}`);
  }
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
function resolveAnchor(param) {
  if (!param || !param.point) return null;

  // Time: try param.time, fall back to coordinateToTime
  let time = param.time;
  if (time == null && _chart?.timeScale) {
    try { time = _chart.timeScale().coordinateToTime(param.point.x); } catch {}
  }
  if (time == null) {
    // Final fallback: snap to last candle
    const candles = marketState.lastCandles || [];
    if (candles.length) time = Number(candles[candles.length - 1].time);
  }
  if (time == null) return null;

  // Price: candleSeries.coordinateToPrice
  let price = null;
  if (_candleSeries && typeof _candleSeries.coordinateToPrice === 'function') {
    try { price = _candleSeries.coordinateToPrice(param.point.y); } catch {}
  }
  if (price == null) return null;

  return { time: Number(time), price: Number(price) };
}
