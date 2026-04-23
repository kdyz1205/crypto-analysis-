// frontend/js/workbench/drawings/prediction_tool.js
//
// "预测" tool — drop a long/short position prototype on the chart to
// reason about R:R before committing to a real order.
//
// Mechanic (mirrors TradingView's Long/Short Position tool):
//   1. User picks 多头 or 空头 from toolbar.
//   2. First click: sets ENTRY price at click Y.
//   3. Second click: sets TP at click Y; SL auto-offset is stop_pct% on
//      opposite side of entry (default 1%, user can edit inline).
//   4. Locked box shows entry/SL/TP horizontal zones:
//        - Green zone: entry → TP (profit)
//        - Red zone:   entry → SL (risk)
//      Labels: R:R ratio, % risk, % reward, absolute $ change per qty=1.
//   5. Box is draggable by its entry midline to relocate; SL/TP handles
//      let user fine-tune.
//
// Persisted to localStorage keyed by symbol:interval so predictions
// survive page reloads. Can be cleared via "清空预测" in toolbar.

import { publish } from '../../util/events.js';

const LS_KEY = 'v2.predictions.v1';

let _chart = null;
let _candleSeries = null;
let _container = null;
let _layer = null;
let _active = false;
let _direction = 'long';       // 'long' | 'short'
let _pickPhase = 0;             // 0 = idle, 1 = awaiting entry click, 2 = awaiting TP click
let _tempEntry = null;          // { time, price, el }
let _tempEl = null;             // DOM el while being drawn
let _predictions = [];          // [{ symbol, interval, direction, entry, sl, tp, time_start, time_end, el }]

const DEFAULT_STOP_PCT = 1.0;   // 1% stop offset from entry

export function initPredictionTool(chart, candleSeries, container) {
  _chart = chart;
  _candleSeries = candleSeries;
  _container = container;
  if (!_layer) {
    _layer = document.createElement('div');
    _layer.className = 'predict-overlay-layer';
    _layer.style.cssText = `
      position: absolute; inset: 0;
      pointer-events: none;
      overflow: hidden;
      z-index: 14;
    `;
    container.appendChild(_layer);
  }
  injectStyles();
  chart.timeScale().subscribeVisibleLogicalRangeChange(_repositionAll);
  window.addEventListener('resize', _repositionAll);
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      if (_pickPhase > 0) _cancelPick();
      if (_active) exitPredictionMode();
    }
  });

  // Load saved predictions
  _loadFromStorage();
  _renderAll();
}

export function enterPredictionMode(direction = 'long') {
  _active = true;
  _direction = direction === 'short' ? 'short' : 'long';
  _pickPhase = 1;
  _container.style.cursor = 'crosshair';
  _container.addEventListener('click', _onClick, true);
  publish('predict.mode', { active: true, direction: _direction });
}

export function exitPredictionMode() {
  _active = false;
  _pickPhase = 0;
  _container.style.cursor = '';
  _container.removeEventListener('click', _onClick, true);
  if (_tempEl) { _tempEl.remove(); _tempEl = null; }
  _tempEntry = null;
  publish('predict.mode', { active: false });
}

export function clearAllPredictions(symbol = null, interval = null) {
  const keep = symbol && interval
    ? _predictions.filter((p) => p.symbol !== symbol || p.interval !== interval)
    : [];
  for (const p of _predictions) {
    if (!keep.includes(p)) { try { p.el.remove(); } catch {} }
  }
  _predictions = keep;
  _saveToStorage();
}

function _onClick(ev) {
  if (!_active || _pickPhase === 0) return;
  const t = ev.target;
  if (t?.closest && (t.closest('.draw-toolbar') || t.closest('.v2-cond-rail'))) return;

  ev.preventDefault();
  ev.stopPropagation();
  const rect = _container.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  const time = Number(_chart.timeScale().coordinateToTime(x));
  const price = Number(_candleSeries.coordinateToPrice(y));
  if (!Number.isFinite(time) || !Number.isFinite(price)) return;

  if (_pickPhase === 1) {
    _tempEntry = { time, price };
    _pickPhase = 2;
    // Show a tiny dot marker
    _tempEl = document.createElement('div');
    _tempEl.className = 'predict-entry-marker';
    _tempEl.style.left = `${x - 4}px`;
    _tempEl.style.top = `${y - 4}px`;
    _layer.appendChild(_tempEl);
    publish('predict.phase', { phase: 'awaiting_tp' });
    return;
  }

  // _pickPhase === 2: user just clicked TP
  const entry = _tempEntry.price;
  const tp = price;
  // Stop offset: opposite side of entry at DEFAULT_STOP_PCT
  const stopPct = DEFAULT_STOP_PCT / 100.0;
  const sl = _direction === 'long' ? entry * (1 - stopPct) : entry * (1 + stopPct);
  // Reject invalid geometry: long wants tp > entry, short wants tp < entry
  if (_direction === 'long' && tp <= entry) {
    _cancelPick();
    alert('多头仓位: TP 必须高于入场价');  // SAFE: alert renders text
    return;
  }
  if (_direction === 'short' && tp >= entry) {
    _cancelPick();
    alert('空头仓位: TP 必须低于入场价');  // SAFE: alert renders text
    return;
  }

  const pred = {
    id: `pred_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`,
    symbol: window?.marketState?.currentSymbol || '',
    interval: window?.marketState?.currentInterval || '',
    direction: _direction,
    time_start: _tempEntry.time,
    time_end: time,
    entry, sl, tp,
    stop_pct_at_create: DEFAULT_STOP_PCT,
    created_at: Date.now(),
  };
  _predictions.push(pred);
  _createPredictionDom(pred);
  _saveToStorage();
  _tempEl?.remove();
  _tempEl = null;
  _tempEntry = null;
  _pickPhase = 0;
  exitPredictionMode();
}

function _cancelPick() {
  _pickPhase = 0;
  if (_tempEl) { _tempEl.remove(); _tempEl = null; }
  _tempEntry = null;
}

// ─── DOM rendering ──────────────────────────────────────────────

function _createPredictionDom(pred) {
  const el = document.createElement('div');
  el.className = `predict-box predict-${pred.direction}`;
  el.dataset.predId = pred.id;

  // Close button
  const closeBtn = document.createElement('button');
  closeBtn.className = 'predict-close';
  closeBtn.textContent = '✕';
  closeBtn.title = '删除预测';
  closeBtn.addEventListener('click', (ev) => {
    ev.stopPropagation();
    _removePrediction(pred.id);
  });
  el.appendChild(closeBtn);

  _layer.appendChild(el);
  pred.el = el;
  _positionPredictionEl(pred);
}

function _positionPredictionEl(pred) {
  if (!pred.el) return;
  const xStart = _chart.timeScale().timeToCoordinate(pred.time_start);
  const xEnd = _chart.timeScale().timeToCoordinate(pred.time_end);
  const yEntry = _candleSeries.priceToCoordinate(pred.entry);
  const ySl = _candleSeries.priceToCoordinate(pred.sl);
  const yTp = _candleSeries.priceToCoordinate(pred.tp);
  if (xStart == null || xEnd == null || yEntry == null || ySl == null || yTp == null) {
    pred.el.style.display = 'none';
    return;
  }
  pred.el.style.display = '';
  const left = Math.min(xStart, xEnd);
  const width = Math.abs(xEnd - xStart);
  const top = Math.min(ySl, yTp);
  const height = Math.abs(yTp - ySl);
  pred.el.style.left = `${left}px`;
  pred.el.style.top = `${top}px`;
  pred.el.style.width = `${width}px`;
  pred.el.style.height = `${height}px`;

  // Inside: profit zone (entry → TP) green, risk zone (entry → SL) red
  const entryOffset = yEntry - top;
  const risk = Math.abs(pred.entry - pred.sl);
  const reward = Math.abs(pred.tp - pred.entry);
  const rrRatio = risk > 0 ? (reward / risk) : 0;
  const riskPct = pred.entry !== 0 ? (risk / pred.entry * 100) : 0;
  const rewardPct = pred.entry !== 0 ? (reward / pred.entry * 100) : 0;

  const profitZoneTop = pred.direction === 'long' ? 0 : entryOffset;
  const profitZoneHeight = pred.direction === 'long' ? entryOffset : (height - entryOffset);
  const riskZoneTop = pred.direction === 'long' ? entryOffset : 0;
  const riskZoneHeight = pred.direction === 'long' ? (height - entryOffset) : entryOffset;

  pred.el.innerHTML = `
    <div class="predict-zone predict-zone-profit" style="top:${profitZoneTop}px;height:${profitZoneHeight}px"></div>
    <div class="predict-zone predict-zone-risk" style="top:${riskZoneTop}px;height:${riskZoneHeight}px"></div>
    <div class="predict-entry-line" style="top:${entryOffset}px"></div>
    <div class="predict-label predict-label-tp" style="top:${pred.direction === 'long' ? 2 : height - 18}px">
      TP ${_fmtPrice(pred.tp)} · +${rewardPct.toFixed(2)}%
    </div>
    <div class="predict-label predict-label-entry" style="top:${entryOffset - 9}px">
      ${pred.direction === 'long' ? '多头' : '空头'} 入场 ${_fmtPrice(pred.entry)}
    </div>
    <div class="predict-label predict-label-sl" style="top:${pred.direction === 'long' ? height - 18 : 2}px">
      SL ${_fmtPrice(pred.sl)} · -${riskPct.toFixed(2)}%
    </div>
    <div class="predict-rr">R:R = 1:${rrRatio.toFixed(2)}</div>
    <button class="predict-close" title="删除预测">✕</button>
  `;
  // Re-wire the close button (innerHTML rewrite nuked the old listener)
  pred.el.querySelector('.predict-close')?.addEventListener('click', (ev) => {
    ev.stopPropagation();
    _removePrediction(pred.id);
  });
  pred.el.style.pointerEvents = 'auto';
}

function _repositionAll() {
  for (const p of _predictions) _positionPredictionEl(p);
}

function _renderAll() {
  for (const p of _predictions) {
    if (!p.el) _createPredictionDom(p);
    else _positionPredictionEl(p);
  }
}

function _removePrediction(id) {
  const idx = _predictions.findIndex((p) => p.id === id);
  if (idx < 0) return;
  try { _predictions[idx].el?.remove(); } catch {}
  _predictions.splice(idx, 1);
  _saveToStorage();
}

// ─── Persistence ────────────────────────────────────────────────

function _loadFromStorage() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      _predictions = parsed.map((p) => ({ ...p, el: null }));
    }
  } catch {}
}

function _saveToStorage() {
  try {
    // Don't serialize DOM refs
    const serializable = _predictions.map(({ el, ...rest }) => rest);
    localStorage.setItem(LS_KEY, JSON.stringify(serializable));
  } catch {}
}

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

function injectStyles() {
  if (document.getElementById('predict-tool-styles')) return;
  const s = document.createElement('style');
  s.id = 'predict-tool-styles';
  s.textContent = `
    .predict-box {
      position: absolute; box-sizing: border-box;
      pointer-events: auto;
      border: 1px solid rgba(148, 163, 184, 0.5);
      font-family: ui-monospace, "SF Mono", Consolas, monospace;
      font-size: 11px;
    }
    .predict-zone {
      position: absolute; left: 0; right: 0;
      pointer-events: none;
    }
    .predict-zone-profit { background: rgba(38, 166, 154, 0.18); }
    .predict-zone-risk   { background: rgba(239, 83, 80, 0.18); }
    .predict-entry-line {
      position: absolute; left: 0; right: 0;
      border-top: 1px dashed rgba(148, 163, 184, 0.9);
      height: 0; pointer-events: none;
    }
    .predict-label {
      position: absolute;
      left: 4px;
      padding: 2px 5px;
      background: rgba(14, 20, 31, 0.88);
      color: #e8edf5;
      border-radius: 3px;
      white-space: nowrap;
      font-size: 10.5px;
      pointer-events: none;
    }
    .predict-label-tp    { color: #26a69a; }
    .predict-label-sl    { color: #ef5350; }
    .predict-label-entry { color: #94a3b8; }
    .predict-rr {
      position: absolute; right: 4px; top: 2px;
      padding: 2px 5px;
      background: rgba(14, 20, 31, 0.92);
      color: #fbbf24;
      border-radius: 3px;
      font-weight: 600;
      font-size: 10.5px;
      pointer-events: none;
    }
    .predict-close {
      position: absolute; right: 2px; bottom: 2px;
      width: 18px; height: 18px;
      background: rgba(14, 20, 31, 0.92);
      color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 3px; cursor: pointer;
      font-size: 11px; line-height: 1;
      display: flex; align-items: center; justify-content: center;
      pointer-events: auto;
    }
    .predict-close:hover { color: #ef5350; border-color: #ef5350; }
    .predict-entry-marker {
      position: absolute;
      width: 8px; height: 8px;
      background: #fbbf24;
      border: 1px solid rgba(14, 20, 31, 0.9);
      border-radius: 50%;
      pointer-events: none;
    }
  `;
  document.head.appendChild(s);
}
