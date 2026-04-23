import { marketState } from '../../state/market.js';
import {
  drawingsState,
  setDrawingViewMode,
  setDrawingsError,
  setDrawingsLoading,
  setDrawSide,
  setEditTarget,
  setManualDrawings,
  setPendingAnchor,
  setSelectedManualLine,
  setTradePlanCounts,
  toggleLineVisibility,
  toggleMultiSelected,
  clearMultiSelected,
} from '../../state/drawings.js';
import * as condSvc from '../../services/conditionals.js';
import * as drawingsSvc from '../../services/drawings.js';
import { subscribe } from '../../util/events.js';
import { clearManualTrendlineOverlay } from '../overlays/manual_trendline_overlay.js';

let controllerChart = null;
let controllerContainer = null;
let controllerInitialized = false;
let drawingsRequestSeq = 0;
let selectedDraft = null;

export function initManualTrendlineController(chart, container) {
  controllerChart = chart;
  controllerContainer = container;
  if (controllerInitialized || !chart || !container) return;
  controllerInitialized = true;

  const panel = document.createElement('div');
  panel.className = 'manual-drawing-panel';
  panel.id = 'manual-drawing-panel';
  container.appendChild(panel);

  panel.addEventListener('click', onPanelClick);
  panel.addEventListener('change', onPanelChange);
  panel.addEventListener('input', onPanelInput);

  // DISABLED: chart_drawing.js is now the single source of truth for
  // chart click → drawing. Two competing subscribers were racing to
  // consume the second click of two-click trendline mode, leaving the
  // user's preview line stuck to the cursor with no way to commit.
  // The old controller's sidebar/panel features (refresh, render via
  // legacy overlay) still work, but its click subscription is off.
  // if (typeof chart.subscribeClick === 'function') {
  //   chart.subscribeClick((param) => {
  //     console.log('[manual-draw] chart click', { ... });
  //     void handleChartClick(param);
  //   });
  // }

  subscribe('drawings.updated', () => { renderPanel(); void refreshTradePlanCounts(); });
  subscribe('conditionals.changed', () => { void refreshTradePlanCounts(); });
  subscribe('drawings.loading', () => renderPanel());
  subscribe('drawings.error', () => renderPanel());
  subscribe('drawings.mode', () => renderPanel());
  subscribe('drawings.anchor', () => renderPanel());
  subscribe('drawings.selected', () => renderPanel());
  subscribe('drawings.editTarget', () => renderPanel());
  subscribe('drawings.multiSelected', () => renderPanel());
  subscribe('drawings.visibility', () => renderPanel());
  subscribe('drawings.tradePlanCounts', () => renderPanel());
  subscribe('strategy.snapshot.updated', () => renderManualLines());
  // NOTE: no refreshManualDrawings subscriber on market.symbol.changed /
  // market.interval.changed — loadCurrent() in chart.js already calls
  // refreshManualDrawings after it reloads OHLCV, so adding it here
  // double-fires the GET /api/drawings and collides via AbortController.

  renderPanel();
}

export async function refreshManualDrawings(symbol, timeframe) {
  if (!symbol || !timeframe) return;
  // Don't overwrite drawingsState.lines while the user is mid-drag — it
  // would snap the line back to server state and lose their gesture.
  try {
    const { isDragging } = await import('./chart_drawing.js');
    if (isDragging()) {
      console.log('[drawings] refresh skipped — drag in progress');
      return;
    }
  } catch {}
  const requestSeq = ++drawingsRequestSeq;
  setDrawingsLoading(true);
  try {
    const response = await drawingsSvc.getManualDrawings(symbol, timeframe);
    if (requestSeq !== drawingsRequestSeq) return;
    // Merge: keep any local _pending lines (optimistic inserts whose POST
    // is still in flight) so the user doesn't see their just-drawn line
    // vanish during a background refresh. Only lines for THIS symbol+tf.
    const serverLines = response.drawings || [];
    const pending = (drawingsState.lines || []).filter(
      (l) => l._pending && l.symbol === symbol && l.timeframe === timeframe
        && !serverLines.some((s) => s.manual_line_id === l.manual_line_id),
    );
    setManualDrawings([...serverLines, ...pending]);
    setDrawingsError(null);
  } catch (err) {
    if (requestSeq !== drawingsRequestSeq) return;
    setDrawingsError(safeErrorMessage(err));
  } finally {
    if (requestSeq !== drawingsRequestSeq) return;
    setDrawingsLoading(false);
    renderManualLines();
  }
}

export function renderManualLines() {
  // Manual lines are now rendered by the SVG overlay (svg_line_editor.js).
  // Clear any legacy lightweight-charts line series so they don't duplicate.
  if (controllerChart) clearManualTrendlineOverlay(controllerChart);
}

export function getSuppressedAutoLineIds() {
  const suppressed = new Set();
  for (const line of drawingsState.lines) {
    if (line.override_mode === 'suppress_nearest_auto_line' && line.nearest_auto_line_id) {
      suppressed.add(line.nearest_auto_line_id);
    }
  }
  return suppressed;
}

function renderPanel() {
  const panel = document.getElementById('manual-drawing-panel');
  if (!panel) return;
  const selected = drawingsState.lines.find((line) => line.manual_line_id === drawingsState.selectedLineId) || null;
  panel.innerHTML = `
    <div class="manual-panel-title">Manual Lines</div>
    <div class="manual-panel-actions">
      <button class="btn ${drawingsState.drawSide === 'resistance' ? 'active' : ''}" data-action="draw-resistance">Draw Resistance</button>
      <button class="btn ${drawingsState.drawSide === 'support' ? 'active' : ''}" data-action="draw-support">Draw Support</button>
      <button class="btn" data-action="cancel-draw">Cancel</button>
    </div>
    <div class="manual-panel-actions">
      <label class="manual-inline-label">
        View
        <select data-action="view-mode">
          ${renderOption('mixed', 'Mixed', drawingsState.viewMode)}
          ${renderOption('manual_only', 'Manual Only', drawingsState.viewMode)}
          ${renderOption('auto_only', 'Auto Only', drawingsState.viewMode)}
        </select>
      </label>
      <button class="btn" data-action="clear-all">Clear All</button>
    </div>
    <div class="manual-panel-note">${renderStatusLine(selected)}</div>
    <div class="manual-panel-list">
      ${renderLineList(drawingsState.lines, drawingsState.selectedLineId)}
    </div>
    ${selected ? renderSelectedActions(selected) : '<div class="manual-panel-empty">Select a manual line to edit or override.</div>'}
  `;
}

function renderOption(value, label, currentValue) {
  return `<option value="${value}" ${value === currentValue ? 'selected' : ''}>${label}</option>`;
}

function renderStatusLine(selected) {
  if (drawingsState.loading) return 'Loading manual lines...';
  if (drawingsState.error) return `Load failed: ${escapeHtml(drawingsState.error)}`;
  if (drawingsState.editTarget && selected) {
    return `Click a candle to update the ${drawingsState.editTarget} anchor for ${escapeHtml(selected.label || selected.manual_line_id)}.`;
  }
  if (drawingsState.pendingAnchor && drawingsState.drawSide) {
    return `First ${drawingsState.drawSide} anchor captured at ${drawingsState.pendingAnchor.time}. Click the second anchor.`;
  }
  if (drawingsState.drawSide) {
    return `Draw mode active for ${drawingsState.drawSide}. Click two candles to create a line.`;
  }
  return 'Manual lines persist in the backend store and can override auto lines.';
}

function renderLineList(lines, selectedLineId) {
  if (!lines.length) {
    return '<div class="manual-panel-empty">No manual lines for this chart yet.</div>';
  }
  const hidden = drawingsState.hiddenLineIds || new Set();
  const multi = drawingsState.multiSelectedIds || new Set();
  const counts = drawingsState.tradePlanCounts || {};
  const multiBlocked = Array.from(multi).some((id) => (counts[id] || 0) > 0);
  const multiBar = multi.size > 0
    ? `<div class="manual-panel-actions manual-multi-bar">
         <span class="manual-multi-count">${multi.size} selected</span>
         <button class="btn btn-danger" data-action="multi-delete"
           ${multiBlocked ? 'disabled title="Cancel active orders before deleting lines."' : ''}>Delete Selected</button>
         <button class="btn" data-action="multi-clear">Clear</button>
       </div>`
    : '';
  const rows = lines
    .map((line) => {
      const id = line.manual_line_id;
      const isHidden = hidden.has(id);
      const isMulti = multi.has(id);
      const isSel = id === selectedLineId;
      const n = counts[id] || 0;
      const tpBadge = n > 0
        ? `<span class="manual-tp-badge" title="${n} trade plan(s)">${n}</span>`
        : '';
      const classes = [
        'manual-line-row',
        isSel ? 'is-selected' : '',
        isMulti ? 'is-multi' : '',
        isHidden ? 'is-hidden' : '',
      ].filter(Boolean).join(' ');
      return `
        <div class="${classes}" data-line-id="${escapeHtml(id)}">
          <input type="checkbox" class="manual-line-check" data-action="multi-toggle"
            data-line-id="${escapeHtml(id)}" ${isMulti ? 'checked' : ''}/>
          <button class="manual-line-eye" data-action="toggle-visibility"
            data-line-id="${escapeHtml(id)}" title="${isHidden ? 'Show' : 'Hide'}">
            ${isHidden ? '○' : '●'}
          </button>
          <button class="manual-line-body" data-action="jump-line" data-line-id="${escapeHtml(id)}">
            <span class="manual-line-main">${escapeHtml(line.label || id)}</span>
            <span class="manual-line-meta">${escapeHtml(line.side)} · ${escapeHtml(line.comparison_status || 'uncompared')}</span>
          </button>
          ${tpBadge}
        </div>
      `;
    })
    .join('');
  return multiBar + rows;
}

async function refreshTradePlanCounts() {
  try {
    const resp = await condSvc.listConditionals('all', marketState.currentSymbol);
    const counts = {};
    for (const c of resp?.conditionals || []) {
      if (c.status !== 'pending' && c.status !== 'triggered') continue;
      const id = c.manual_line_id;
      if (!id) continue;
      counts[id] = (counts[id] || 0) + 1;
    }
    setTradePlanCounts(counts);
  } catch (err) {
    console.warn('[manual-panel] tradePlan count refresh failed', err);
  }
}

function renderSelectedActions(selected) {
  const draft = getSelectedDraft(selected);
  const counts = drawingsState.tradePlanCounts || {};
  const activePlans = counts[selected.manual_line_id] || 0;
  const deleteAttrs = activePlans > 0
    ? 'disabled title="Cancel active orders before deleting this line."'
    : '';
  return `
    <div class="manual-selected-actions">
      <div class="manual-selected-title">Selected: ${escapeHtml(selected.label || selected.manual_line_id)}</div>
      <div class="manual-selected-meta">
        nearest auto: ${escapeHtml(selected.nearest_auto_line_id || 'none')} |
        status: ${escapeHtml(selected.comparison_status || 'uncompared')} |
        slope diff: ${selected.slope_diff ?? '-'} |
        projected diff: ${selected.projected_price_diff ?? '-'} |
        overlap: ${selected.overlap_ratio ?? '-'}
      </div>
      <div class="manual-panel-actions">
        <button class="btn" data-action="edit-start" ${selected.locked ? 'disabled' : ''}>Edit Start</button>
        <button class="btn" data-action="edit-end" ${selected.locked ? 'disabled' : ''}>Edit End</button>
        <button class="btn" data-action="toggle-lock">${selected.locked ? 'Unlock' : 'Lock'}</button>
        <button class="btn" data-action="toggle-extend-left">${selected.extend_left ? 'Stop Extend Left' : 'Extend Left'}</button>
        <button class="btn" data-action="toggle-extend-right">${selected.extend_right ? 'Stop Extend' : 'Extend Right'}</button>
        <button class="btn btn-danger" data-action="delete-selected" ${deleteAttrs}>Delete</button>
      </div>
      <div class="manual-panel-actions">
        <label class="manual-inline-label">
          Override
          <select data-action="override-mode">
            ${renderOption('display_only', 'Display Only', selected.override_mode)}
            ${renderOption('compare_only', 'Compare Only', selected.override_mode)}
            ${renderOption('promote_to_active', 'Promote To Active', selected.override_mode)}
            ${renderOption('suppress_nearest_auto_line', 'Suppress Nearest Auto', selected.override_mode)}
            ${renderOption('strategy_input_enabled', 'Strategy Input', selected.override_mode)}
          </select>
        </label>
      </div>
      <div class="manual-panel-form">
        <label class="manual-inline-label manual-inline-field">
          Label
          <input type="text" id="manual-line-label-input" value="${escapeHtml(draft.label)}" placeholder="Line label" />
        </label>
        <label class="manual-inline-label manual-inline-field">
          Notes
          <textarea id="manual-line-notes-input" rows="3" placeholder="Optional notes">${escapeHtml(draft.notes)}</textarea>
        </label>
        <div class="manual-panel-actions">
          <button class="btn" data-action="save-metadata">Save Label / Notes</button>
        </div>
      </div>
    </div>
  `;
}

async function onPanelClick(event) {
  const target = event.target.closest('[data-action]');
  if (!(target instanceof HTMLElement)) return;
  const action = target.dataset.action;
  const selected = drawingsState.lines.find((line) => line.manual_line_id === drawingsState.selectedLineId) || null;

  if (action === 'draw-resistance') {
    setDrawSide('resistance');
    setPendingAnchor(null);
    setEditTarget(null);
    return;
  }
  if (action === 'draw-support') {
    setDrawSide('support');
    setPendingAnchor(null);
    setEditTarget(null);
    return;
  }
  if (action === 'cancel-draw') {
    setDrawSide(null);
    setPendingAnchor(null);
    setEditTarget(null);
    return;
  }
  if (action === 'clear-all') {
    await drawingsSvc.clearManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    setSelectedManualLine(null);
    selectedDraft = null;
    await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    return;
  }
  if (action === 'select-line' || action === 'jump-line') {
    const lineId = target.dataset.lineId || null;
    setSelectedManualLine(lineId);
    const current = drawingsState.lines.find((line) => line.manual_line_id === lineId) || null;
    selectedDraft = current
      ? { manualLineId: current.manual_line_id, label: current.label || '', notes: current.notes || '' }
      : null;
    setEditTarget(null);
    if (action === 'jump-line') {
      // Scroll chart to line
      try {
        const mod = await import('./chart_drawing.js');
        mod.jumpToLine(lineId);
      } catch {}
    }
    return;
  }
  if (action === 'toggle-visibility') {
    event.stopPropagation();
    toggleLineVisibility(target.dataset.lineId || null);
    return;
  }
  if (action === 'multi-toggle') {
    event.stopPropagation();
    toggleMultiSelected(target.dataset.lineId || null);
    return;
  }
  if (action === 'multi-clear') {
    clearMultiSelected();
    return;
  }
  if (action === 'multi-delete') {
    const ids = Array.from(drawingsState.multiSelectedIds || []);
    if (!ids.length) return;
    const counts = drawingsState.tradePlanCounts || {};
    const blocked = ids.filter((id) => (counts[id] || 0) > 0);
    if (blocked.length) {
      alert('Selected lines still have active orders. Cancel those orders first; the lines stay as the order rationale.');
      return;
    }
    if (!confirm(`Delete ${ids.length} line(s)?`)) return;
    for (const id of ids) {
      try { await drawingsSvc.deleteManualDrawing(id); }
      catch (err) { console.warn('[manual-panel] delete failed', id, err); }
    }
    clearMultiSelected();
    setSelectedManualLine(null);
    await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    return;
  }
  if (!selected) return;

  if (action === 'edit-start') {
    setEditTarget('start');
    setDrawSide(null);
    setPendingAnchor(null);
    return;
  }
  if (action === 'edit-end') {
    setEditTarget('end');
    setDrawSide(null);
    setPendingAnchor(null);
    return;
  }
  if (action === 'toggle-lock') {
    await drawingsSvc.updateManualDrawing(selected.manual_line_id, { locked: !selected.locked });
    await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    return;
  }
  if (action === 'toggle-extend-left') {
    await drawingsSvc.updateManualDrawing(selected.manual_line_id, { extend_left: !selected.extend_left });
    await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    return;
  }
  if (action === 'toggle-extend-right') {
    await drawingsSvc.updateManualDrawing(selected.manual_line_id, { extend_right: !selected.extend_right });
    await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    return;
  }
  if (action === 'save-metadata') {
    const draft = getSelectedDraft(selected);
    await drawingsSvc.updateManualDrawing(selected.manual_line_id, {
      label: draft.label,
      notes: draft.notes,
    });
    await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    return;
  }
  if (action === 'delete-selected') {
    const counts = drawingsState.tradePlanCounts || {};
    if ((counts[selected.manual_line_id] || 0) > 0) {
      alert('This line still has active orders. Cancel those orders first; the line stays as the order rationale.');
      return;
    }
    await drawingsSvc.deleteManualDrawing(selected.manual_line_id);
    setSelectedManualLine(null);
    selectedDraft = null;
    setEditTarget(null);
    await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
  }
}

async function onPanelChange(event) {
  const target = event.target;
  if (!(target instanceof HTMLSelectElement)) return;
  const action = target.dataset.action;
  const selected = drawingsState.lines.find((line) => line.manual_line_id === drawingsState.selectedLineId) || null;
  if (action === 'view-mode') {
    setDrawingViewMode(target.value);
    renderManualLines();
    return;
  }
  if (action === 'override-mode' && selected) {
    await drawingsSvc.updateManualDrawing(selected.manual_line_id, { override_mode: target.value });
    await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
  }
}

function onPanelInput(event) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) return;
  const selected = drawingsState.lines.find((line) => line.manual_line_id === drawingsState.selectedLineId) || null;
  if (!selected) return;
  const draft = getSelectedDraft(selected);
  if (target.id === 'manual-line-label-input') {
    draft.label = target.value;
  } else if (target.id === 'manual-line-notes-input') {
    draft.notes = target.value;
  }
  selectedDraft = draft;
}

async function handleChartClick(param) {
  if (!drawingsState.drawSide && !drawingsState.editTarget) return;
  const candleAnchor = resolveCandleAnchor(param, drawingsState.drawSide || inferSelectedSide());
  if (!candleAnchor) return;

  if (drawingsState.editTarget) {
    const selected = drawingsState.lines.find((line) => line.manual_line_id === drawingsState.selectedLineId);
    if (!selected || selected.locked) return;
    const payload = buildEditedLinePayload(selected, drawingsState.editTarget, candleAnchor);
    await drawingsSvc.updateManualDrawing(selected.manual_line_id, payload);
    setEditTarget(null);
    await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    return;
  }

  if (!drawingsState.pendingAnchor) {
    setPendingAnchor(candleAnchor);
    return;
  }

  const ordered = normalizeAnchors(drawingsState.pendingAnchor, candleAnchor);
  await drawingsSvc.createManualDrawing({
    symbol: marketState.currentSymbol,
    timeframe: marketState.currentInterval,
    side: drawingsState.drawSide,
    t_start: ordered.start.time,
    t_end: ordered.end.time,
    price_start: ordered.start.price,
    price_end: ordered.end.price,
    label: `${drawingsState.drawSide} manual line`,
    override_mode: 'display_only',
  });
  setPendingAnchor(null);
  setDrawSide(null);
  await refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
}

function buildEditedLinePayload(line, target, anchor) {
  const start = target === 'start'
    ? { time: anchor.time, price: anchor.price }
    : { time: line.t_start, price: line.price_start };
  const end = target === 'end'
    ? { time: anchor.time, price: anchor.price }
    : { time: line.t_end, price: line.price_end };
  const ordered = normalizeAnchors(start, end);
  return {
    t_start: ordered.start.time,
    t_end: ordered.end.time,
    price_start: ordered.start.price,
    price_end: ordered.end.price,
  };
}

function normalizeAnchors(left, right) {
  if (left.time <= right.time) {
    return { start: left, end: right };
  }
  return { start: right, end: left };
}

function inferSelectedSide() {
  const selected = drawingsState.lines.find((line) => line.manual_line_id === drawingsState.selectedLineId);
  return selected?.side || 'resistance';
}

function getSelectedDraft(selected) {
  if (!selectedDraft || selectedDraft.manualLineId !== selected.manual_line_id) {
    selectedDraft = {
      manualLineId: selected.manual_line_id,
      label: selected.label || '',
      notes: selected.notes || '',
    };
  }
  return selectedDraft;
}

function resolveCandleAnchor(param, side) {
  const candles = marketState.lastCandles || [];
  if (!candles.length) {
    console.warn('[manual-draw] no candles loaded yet');
    return null;
  }

  // Step 1: try the direct time from the click event
  let time = normalizeChartTime(param?.time);

  // Step 2: if time missing (click landed between bars / on indicator),
  // use x-pixel → time via lightweight-charts API
  if (time == null && param?.point && controllerChart?.timeScale) {
    try {
      const ts = controllerChart.timeScale().coordinateToTime(param.point.x);
      time = normalizeChartTime(ts);
    } catch (err) {
      console.warn('[manual-draw] coordinateToTime failed:', err);
    }
  }

  // Step 3: still null? fall back to the nearest candle by x-pixel guess.
  if (time == null) {
    // Last resort: use the most recent candle — user will see the line and
    // can adjust. Better than silently doing nothing.
    const last = candles[candles.length - 1];
    time = Number(last.time);
    console.warn('[manual-draw] fallback: using latest candle time', time);
  }

  // Step 4: find the exact or nearest candle
  let candle = candles.find((c) => Number(c.time) === Number(time));
  if (!candle) {
    // Snap to the closest candle
    let best = null;
    let bestDiff = Infinity;
    for (const c of candles) {
      const diff = Math.abs(Number(c.time) - Number(time));
      if (diff < bestDiff) {
        best = c;
        bestDiff = diff;
      }
    }
    candle = best;
    if (candle) {
      time = Number(candle.time);
      console.log('[manual-draw] snapped to nearest candle', time);
    }
  }

  if (!candle) {
    console.warn('[manual-draw] could not resolve any candle for click');
    return null;
  }

  return {
    time: Number(time),
    price: side === 'support' ? Number(candle.low) : Number(candle.high),
  };
}

function normalizeChartTime(time) {
  if (typeof time === 'number') return Math.floor(time);
  if (time && typeof time === 'object' && typeof time.timestamp === 'number') {
    return Math.floor(time.timestamp);
  }
  return null;
}

function safeErrorMessage(err) {
  if (!err) return 'Unknown error';
  if (typeof err === 'string') return err;
  if (err.body && typeof err.body === 'object' && err.body.detail) return String(err.body.detail);
  if (err.message) return String(err.message);
  return String(err);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
