// frontend/js/workbench/drawings/measure_tool.js
//
// TradingView-style measure tool. User presses M (or clicks 📏 button),
// then clicks + drags on the chart to draw a measurement box. Box shows
// ΔPrice (absolute + %), ΔTime (bars + human duration), ΔBars count.
//
// Two modes share the same drag mechanic:
//   - 'price' : emphasis on price (defaults to a 2-pt vertical strip)
//   - 'date'  : emphasis on time (defaults to a horizontal strip)
//   - 'rect'  : general rectangle (default when user clicks 测量)
//
// Release mouse → measurement frozen until user clicks elsewhere or Esc.
// Supports multiple simultaneous measurements (each click-drag adds one).

import { publish } from '../../util/events.js';

let _chart = null;
let _candleSeries = null;
let _container = null;
let _active = false;        // is measure mode armed?
let _mode = 'rect';         // 'rect' | 'price' | 'date'
let _overlayLayer = null;
let _dragState = null;      // { startX, startY, curX, curY, boxEl }
let _measurements = [];     // persisted boxes on chart

export function initMeasureTool(chart, candleSeries, container) {
  _chart = chart;
  _candleSeries = candleSeries;
  _container = container;
  if (!_overlayLayer) {
    _overlayLayer = document.createElement('div');
    _overlayLayer.className = 'measure-overlay-layer';
    _overlayLayer.style.cssText = `
      position: absolute; inset: 0;
      pointer-events: none;
      overflow: hidden;
      z-index: 15;
    `;
    container.appendChild(_overlayLayer);
  }
  injectStyles();

  // Keep boxes aligned to candle times as the user pans/zooms.
  chart.timeScale().subscribeVisibleLogicalRangeChange(_repositionAll);
  window.addEventListener('resize', _repositionAll);

  // ESC cancels the in-progress drag
  document.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape') return;
    if (_dragState) _cancelDrag();
    if (_active) exitMeasureMode();
  });
}

export function enterMeasureMode(mode = 'rect') {
  _active = true;
  _mode = mode;
  _container.style.cursor = 'crosshair';
  _container.addEventListener('mousedown', _onMouseDown, true);
  publish('measure.mode', { active: true, mode });
}

export function exitMeasureMode() {
  _active = false;
  _container.style.cursor = '';
  _container.removeEventListener('mousedown', _onMouseDown, true);
  if (_dragState) _cancelDrag();
  publish('measure.mode', { active: false });
}

export function clearAllMeasurements() {
  for (const m of _measurements) {
    try { m.el.remove(); } catch {}
  }
  _measurements = [];
}

// ─── Drag lifecycle ─────────────────────────────────────────────

function _onMouseDown(ev) {
  if (!_active || ev.button !== 0) return;
  // Only intercept events that land inside the chart area (not over
  // the toolbar buttons floating on top of the container).
  const t = ev.target;
  if (t && t.closest && (t.closest('.draw-toolbar') || t.closest('.v2-cond-rail'))) return;

  ev.preventDefault();
  ev.stopPropagation();
  const rect = _container.getBoundingClientRect();
  const x0 = ev.clientX - rect.left;
  const y0 = ev.clientY - rect.top;
  const boxEl = document.createElement('div');
  boxEl.className = 'measure-box';
  _overlayLayer.appendChild(boxEl);
  _dragState = { startX: x0, startY: y0, curX: x0, curY: y0, boxEl };
  _updateBox();

  document.addEventListener('mousemove', _onMouseMove, true);
  document.addEventListener('mouseup', _onMouseUp, true);
}

function _onMouseMove(ev) {
  if (!_dragState) return;
  const rect = _container.getBoundingClientRect();
  _dragState.curX = ev.clientX - rect.left;
  _dragState.curY = ev.clientY - rect.top;
  _updateBox();
}

function _onMouseUp(ev) {
  document.removeEventListener('mousemove', _onMouseMove, true);
  document.removeEventListener('mouseup', _onMouseUp, true);
  if (!_dragState) return;
  const ds = _dragState;
  const { startX, startY, curX, curY, boxEl } = ds;
  // Reject tiny clicks (no real drag)
  if (Math.abs(curX - startX) < 3 && Math.abs(curY - startY) < 3) {
    _cancelDrag();
    return;
  }
  // Anchor box to data-space so pan/zoom keeps it aligned.
  const t1 = _chart.timeScale().coordinateToTime(startX);
  const t2 = _chart.timeScale().coordinateToTime(curX);
  const p1 = _candleSeries.coordinateToPrice(startY);
  const p2 = _candleSeries.coordinateToPrice(curY);
  if (t1 == null || t2 == null || p1 == null || p2 == null) {
    _cancelDrag();
    return;
  }
  _measurements.push({
    el: boxEl,
    time1: Number(t1), time2: Number(t2),
    price1: Number(p1), price2: Number(p2),
    mode: _mode,
  });
  // Click on the box → remove it
  boxEl.addEventListener('click', (cev) => {
    cev.stopPropagation();
    const idx = _measurements.findIndex((m) => m.el === boxEl);
    if (idx >= 0) {
      _measurements[idx].el.remove();
      _measurements.splice(idx, 1);
    }
  });
  // Enable pointer events AFTER drag completes so click-to-remove works.
  boxEl.style.pointerEvents = 'auto';
  _dragState = null;
  // Stay in measure mode so user can drop more boxes. Esc exits.
}

function _cancelDrag() {
  if (_dragState?.boxEl) _dragState.boxEl.remove();
  _dragState = null;
  document.removeEventListener('mousemove', _onMouseMove, true);
  document.removeEventListener('mouseup', _onMouseUp, true);
}

// ─── Box rendering ──────────────────────────────────────────────

function _updateBox() {
  if (!_dragState) return;
  const { startX, startY, curX, curY, boxEl } = _dragState;
  _renderBoxAt(boxEl, startX, startY, curX, curY);
}

function _renderBoxAt(boxEl, x1, y1, x2, y2) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const width = Math.abs(x2 - x1);
  const height = Math.abs(y2 - y1);
  boxEl.style.left = `${left}px`;
  boxEl.style.top = `${top}px`;
  boxEl.style.width = `${width}px`;
  boxEl.style.height = `${height}px`;

  // Convert to price/time for the label
  const p1 = _candleSeries.coordinateToPrice(y1);
  const p2 = _candleSeries.coordinateToPrice(y2);
  const t1 = _chart.timeScale().coordinateToTime(x1);
  const t2 = _chart.timeScale().coordinateToTime(x2);
  if (p1 == null || p2 == null || t1 == null || t2 == null) return;

  const dp = p2 - p1;          // signed (down = negative)
  const base = p1 !== 0 ? Math.abs(p1) : 1;
  const pct = (dp / base) * 100;
  const dt = Number(t2) - Number(t1);
  const dtAbs = Math.abs(dt);

  const dir = dp > 0 ? '▲' : (dp < 0 ? '▼' : '—');
  const dirColor = dp > 0 ? '#26a69a' : (dp < 0 ? '#ef5350' : '#94a3b8');

  const lines = [];
  lines.push(`<span style="color:${dirColor};font-weight:600">${dir} ${_fmtPrice(Math.abs(dp))}  (${_fmtPct(pct)})</span>`);
  lines.push(`<span>${_fmtDuration(dtAbs)}  ·  ${_fmtBarCount(dtAbs)}</span>`);
  lines.push(`<span style="color:#94a3b8;font-size:10px">${_fmtPrice(p1)} → ${_fmtPrice(p2)}</span>`);

  boxEl.innerHTML = `<div class="measure-label">${lines.join('<br>')}</div>`;
}

function _repositionAll() {
  for (const m of _measurements) {
    const x1 = _chart.timeScale().timeToCoordinate(m.time1);
    const x2 = _chart.timeScale().timeToCoordinate(m.time2);
    const y1 = _candleSeries.priceToCoordinate(m.price1);
    const y2 = _candleSeries.priceToCoordinate(m.price2);
    if (x1 == null || x2 == null || y1 == null || y2 == null) {
      m.el.style.display = 'none';
      continue;
    }
    m.el.style.display = '';
    _renderBoxAt(m.el, x1, y1, x2, y2);
  }
}

// ─── Formatting helpers ─────────────────────────────────────────

function _fmtPrice(v) {
  const x = Number(v);
  if (!Number.isFinite(x)) return '—';
  const a = Math.abs(x);
  if (a >= 1000) return x.toFixed(2);
  if (a >= 100)  return x.toFixed(3);
  if (a >= 1)    return x.toFixed(4);
  if (a >= 0.01) return x.toFixed(6);
  return x.toFixed(7);
}

function _fmtPct(v) {
  const x = Number(v);
  if (!Number.isFinite(x)) return '—';
  const sign = x > 0 ? '+' : '';
  return `${sign}${x.toFixed(2)}%`;
}

function _fmtDuration(secs) {
  const s = Math.abs(Number(secs) || 0);
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  if (s < 86400) return `${(s / 3600).toFixed(2)}h`;
  return `${(s / 86400).toFixed(2)}d`;
}

function _fmtBarCount(secs) {
  // Derive bar duration from current TF (chart.js exposes currentInterval).
  // Fallback: 1h.
  let barSec = 3600;
  try {
    const tf = (window?.marketState?.currentInterval || '').toLowerCase();
    const map = {
      '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
      '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200,
      '1d': 86400, '1w': 604800,
    };
    if (map[tf]) barSec = map[tf];
  } catch {}
  const bars = Math.abs(Number(secs) || 0) / barSec;
  return `${bars.toFixed(bars >= 10 ? 0 : 1)} bars`;
}

function injectStyles() {
  if (document.getElementById('measure-tool-styles')) return;
  const s = document.createElement('style');
  s.id = 'measure-tool-styles';
  s.textContent = `
    .measure-box {
      position: absolute;
      box-sizing: border-box;
      border: 1px dashed rgba(96, 165, 250, 0.9);
      background: rgba(96, 165, 250, 0.12);
      pointer-events: none;
    }
    .measure-box:hover { border-style: solid; }
    .measure-label {
      position: absolute;
      top: 2px; left: 4px;
      padding: 3px 7px;
      background: rgba(14, 20, 31, 0.92);
      border: 1px solid rgba(96, 165, 250, 0.6);
      border-radius: 4px;
      color: #e8edf5;
      font-size: 11px;
      font-family: ui-monospace, "SF Mono", Consolas, monospace;
      line-height: 1.4;
      white-space: nowrap;
      pointer-events: auto;
      cursor: pointer;
    }
    .measure-label:hover { background: rgba(239, 68, 68, 0.15); }
  `;
  document.head.appendChild(s);
}
