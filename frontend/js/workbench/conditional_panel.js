// frontend/js/workbench/conditional_panel.js
//
// Unified trading panel. Layout:
//
//   ┌─ 我的手画线 (list, hover-highlights on chart) ─────────────
//   ├─ (if selected) 线信息 + 市场上下文 + 历史概率
//   ├─ (if selected) [反弹单] [突破单]
//   └─ 条件单 (pending orders list, live updating)
//
// This replaces:
//   - The old sidebar manual drawing panel (hidden via CSS)
//   - The decision_rail market-state / regime / S/R / pattern-stats
//     cards on the market view (hidden while cond_panel is mounted)
//
// Single source of truth for "I drew a line, now what".

import { subscribe } from '../util/events.js';
import { drawingsState, setSelectedManualLine } from '../state/drawings.js';
import { marketState, setSymbol, setIntervalTF } from '../state/market.js';
import * as condSvc from '../services/conditionals.js';
import { fetchJson } from '../util/fetch.js';
import { setHoveredLineFromPanel, setSimilarLines } from './drawings/chart_drawing.js';

let panel = null;
let currentAnalyze = null;

// Adaptive price precision — small-cap coins need more decimals.
// 0.4 → 5 decimals, 4 → 4 decimals, 40 → 3 decimals, 400 → 2, 4000+ → 2.
function fmtPrice(p, extra = 0) {
  const n = Number(p);
  if (!isFinite(n) || n === 0) return '0';
  const abs = Math.abs(n);
  let dp;
  if (abs >= 1000) dp = 2;
  else if (abs >= 100) dp = 3;
  else if (abs >= 10) dp = 3;
  else if (abs >= 1) dp = 4;
  else if (abs >= 0.1) dp = 5;
  else if (abs >= 0.01) dp = 6;
  else dp = 7;
  return n.toFixed(Math.min(8, dp + extra));
}
let currentLine = null;
let marketContext = null;
let pendingRefreshTimer = null;
let knownConditionals = [];
let bitgetAccount = null;       // real Bitget account snapshot (positions + pending + balance)
let bitgetTimer = null;
// All drawings across every symbol — populated by fetchAllDrawings().
// Used by the grouped "我的手画线 · 全部币种" section.
let allDrawings = [];
let allDrawingsTimer = null;
// UI state: which symbol group is expanded (null = none).
let _expandedSymbol = null;
// Race-guard for jump-line click: latest requested lineId. If user
// rapid-clicks another line before the 350ms defer fires, the earlier
// selection is discarded. Code-reviewer S4 2026-04-21.
let _pendingJumpLineId = null;

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
function esc(v) {
  if (v == null) return '';
  return String(v)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(0)}%`;
}
function fmtR(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(2) + 'R';
}
function fmtTime(ts) {
  if (!ts) return '—';
  try { return new Date(ts * 1000).toLocaleTimeString(); } catch { return '—'; }
}
function fmtDateShort(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts * 1000);
    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
  } catch { return '—'; }
}

// ─────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────
export function initConditionalPanel(container) {
  if (panel) return panel;
  panel = document.createElement('div');
  panel.className = 'cond-panel';
  panel.id = 'v2-cond-panel';
  container.appendChild(panel);

  panel.addEventListener('click', onPanelClick);
  panel.addEventListener('mouseover', onPanelHover);
  panel.addEventListener('mouseout', onPanelLeave);

  // When user draws or loads a line, refresh the list (but don't auto-select).
  // ALSO: if the currently selected line was mutated (e.g. user dragged it),
  // re-load the analyze stats — geometry changed so smart recs are stale.
  subscribe('drawings.updated', () => {
    if (currentLine) {
      const sel = drawingsState.selectedLineId;
      const lines = drawingsState.lines || [];
      const found = sel ? lines.find((l) => l.manual_line_id === sel) : null;
      if (sel && !found) {
        currentLine = null;
        currentAnalyze = null;
        marketContext = null;
        setSelectedManualLine(null);
        render();
        return;
      }
      if (found) {
        const geomChanged =
          found.t_start !== currentLine.t_start ||
          found.t_end !== currentLine.t_end ||
          found.price_start !== currentLine.price_start ||
          found.price_end !== currentLine.price_end;
        if (geomChanged) {
          currentLine = found;
          void loadAnalyzeAndContext();
          return;
        }
        currentLine = found;
      }
    }
    render();
  });

  // When a line is explicitly selected (by user click in our list)
  subscribe('drawings.selected', () => {
    const sel = drawingsState.selectedLineId;
    const lines = drawingsState.lines || [];
    currentLine = sel ? lines.find((l) => l.manual_line_id === sel) : null;
    // chart_drawing.js highlights selected line via drawings.selected event
    void loadAnalyzeAndContext();
  });

  // Bug fix: panel must reset when symbol/timeframe changes, otherwise
  // it shows stale lines for the previous symbol.
  subscribe('market.symbol.changed', () => {
    currentLine = null;
    currentAnalyze = null;
    marketContext = null;
    render();
  });
  subscribe('market.interval.changed', () => {
    currentLine = null;
    currentAnalyze = null;
    marketContext = null;
    render();
  });

  void refreshPending();
  pendingRefreshTimer = setInterval(() => void refreshPending(), 10000);
  subscribe('conditionals.changed', () => {
    void refreshPending();
    void refreshBitgetAccount();
  });

  void refreshBitgetAccount();
  bitgetTimer = setInterval(() => void refreshBitgetAccount(), 5000);

  // Immediate refresh when trade_plan_modal fires 'cond-placed' — user
  // 2026-04-22 complaint: "挂单后 panel 看不到,必须刷新". Backend
  // persists within the request; Bitget pending list updates by the
  // time we return; but our 5s poll was the bottleneck. Dispatching
  // the event cuts that wait to ~one request round-trip.
  const onCondPlaced = () => {
    void refreshPending();
    void refreshBitgetAccount();
    void refreshAllDrawings();
    // Kick one more after Bitget-propagation delay (~1.5s) in case the
    // plan order shows up a tick late on Bitget's pending-list API.
    setTimeout(() => { void refreshBitgetAccount(); }, 1500);
  };
  window.addEventListener('cond-placed', onCondPlaced);

  // All-drawings grouped view — refresh every 15s and on PATCH/POST/DELETE.
  void refreshAllDrawings();
  allDrawingsTimer = setInterval(() => void refreshAllDrawings(), 15000);
  subscribe('drawings.updated', () => { void refreshAllDrawings(); });

  render();
  return panel;
}

async function refreshBitgetAccount() {
  if (!panel) return;
  try {
    // noCache: polling endpoint, the 30s default TTL would freeze balance
    const r = await fetchJson('/api/live-execution/account?mode=live', {
      timeout: 8000, noCache: true,
    });
    if (r && r.ok !== false) {
      bitgetAccount = r;
      const el = panel.querySelector('[data-bitget-section]');
      if (el) el.outerHTML = renderBitgetSection();
    }
  } catch (err) {
    // AbortError is expected on panel unmount / slow network — not a bug.
    // Anything else is worth logging at debug level, not warn (the browser
    // review gate treats warn as failure).
    if (err?.name !== 'AbortError') {
      console.debug('[cond_panel] refreshBitgetAccount err', err);
    }
  }
}

async function fetchMarkPriceOnce(symbol) {
  if (!symbol) return null;
  try {
    const r = await fetchJson(
      `/api/market/mark-price?symbol=${encodeURIComponent(symbol)}`,
      { timeout: 5000, noCache: true },
    );
    if (r && r.ok) return r.mark_price;
  } catch (err) {
    console.warn('[cond_panel] fetchMarkPriceOnce err', err);
  }
  return null;
}

export function destroyConditionalPanel() {
  if (pendingRefreshTimer) { clearInterval(pendingRefreshTimer); pendingRefreshTimer = null; }
  if (bitgetTimer) { clearInterval(bitgetTimer); bitgetTimer = null; }
  if (allDrawingsTimer) { clearInterval(allDrawingsTimer); allDrawingsTimer = null; }
  if (panel && panel.parentNode) panel.parentNode.removeChild(panel);
  panel = null;
  currentAnalyze = null;
  currentLine = null;
  marketContext = null;
  knownConditionals = [];
}

// ─────────────────────────────────────────────────────────────
// Data loaders
// ─────────────────────────────────────────────────────────────
async function loadAnalyzeAndContext() {
  if (!currentLine) {
    currentAnalyze = null;
    marketContext = null;
    setSimilarLines([]);
    render();
    return;
  }
  render({ loading: true });
  const [analyzeResp, mkt] = await Promise.allSettled([
    condSvc.analyzeDrawing(currentLine.manual_line_id),
    fetchJson(
      `/api/market/structure-summary?symbol=${encodeURIComponent(currentLine.symbol)}&interval=${encodeURIComponent(currentLine.timeframe)}`,
      { timeout: 20000 }
    ).catch(() => null),
  ]);
  currentAnalyze = analyzeResp.status === 'fulfilled' ? analyzeResp.value : { ok: false, error: 'analyze failed' };
  marketContext = mkt.status === 'fulfilled' ? mkt.value : null;

  // Push similar historical lines onto the chart overlay (if user has the
  // toggle on). Default: show.
  if (showSimilarOnChart && currentAnalyze?.ok && currentAnalyze.similar_lines) {
    setSimilarLines(currentAnalyze.similar_lines);
  } else {
    setSimilarLines([]);
  }
  render();
}

let showSimilarOnChart = true;

async function refreshPending() {
  if (!panel) return;
  try {
    const resp = await condSvc.listConditionals('all');
    knownConditionals = (resp && resp.conditionals) || [];
    renderPendingSection();
  } catch (err) {
    console.warn('[cond_panel] refreshPending err', err);
  }
}

async function refreshAllDrawings() {
  if (!panel) return;
  try {
    const { getAllManualDrawings } = await import('../services/drawings.js');
    const resp = await getAllManualDrawings();
    allDrawings = (resp && resp.drawings) || [];
    render();
  } catch (err) {
    console.warn('[cond_panel] refreshAllDrawings err', err);
  }
}

// ─────────────────────────────────────────────────────────────
// Event handling
// ─────────────────────────────────────────────────────────────
async function onPanelClick(e) {
  // First: check if user clicked a pending order row (not the 撤 button)
  // → jump chart to that symbol + tf. User 2026-04-21.
  const pendingRow = e.target.closest('.cond-pending-clickable');
  if (pendingRow && !e.target.closest('button')) {
    const sym = pendingRow.dataset.jumpSymbol;
    const tf = pendingRow.dataset.jumpTf;
    if (sym) {
      try {
        setSymbol(sym);
        if (tf) setIntervalTF(tf);
      } catch (err) {
        console.warn('[cond_panel] jump failed', err);
      }
    }
    return;
  }

  // MyDrawings grouped list: click symbol header → toggle expand/collapse
  const symHead = e.target.closest('[data-toggle-symbol]');
  if (symHead && !e.target.closest('button')) {
    const sym = symHead.dataset.toggleSymbol;
    _expandedSymbol = (_expandedSymbol === sym) ? '__none__' : sym;
    render();
    return;
  }

  // MyDrawings line row: click → jump to symbol+tf + select line
  const jumpLine = e.target.closest('[data-jump-line]');
  if (jumpLine && !e.target.closest('button')) {
    const sym = jumpLine.dataset.lineSymbol;
    const tf = jumpLine.dataset.lineTf;
    const lineId = jumpLine.dataset.lineId;
    try {
      if (sym && sym !== marketState.currentSymbol) setSymbol(sym);
      if (tf && tf !== marketState.currentInterval) setIntervalTF(tf);
      // Selecting immediately works ONLY if the target symbol's drawings
      // are already loaded. Since setSymbol triggers an async reload,
      // defer the selection to when drawings refresh. Track the latest
      // requested lineId so rapid clicks across different symbols don't
      // end up selecting a stale lineId on the wrong symbol's chart.
      _pendingJumpLineId = lineId;
      setTimeout(() => {
        if (_pendingJumpLineId === lineId) {
          setSelectedManualLine(lineId);
          _pendingJumpLineId = null;
        }
      }, 350);
    } catch (err) {
      console.warn('[cond_panel] jump-line failed', err);
    }
    return;
  }

  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;

  // Delete a line from the grouped list (any symbol, not just current)
  if (action === 'delete-line-any') {
    e.stopPropagation();
    const lineId = btn.dataset.lineId;
    if (!lineId) return;
    if (!confirm('删除这条线?')) return;
    try {
      const { deleteManualDrawing } = await import('../services/drawings.js');
      await deleteManualDrawing(lineId);
      await refreshAllDrawings();
      // Also refresh current-symbol list if affected
      const sym = btn.dataset.lineSymbol;
      const tf = btn.dataset.lineTf;
      if (sym === marketState.currentSymbol && tf === marketState.currentInterval) {
        const { refreshManualDrawings } = await import('./drawings/manual_trendline_controller.js');
        await refreshManualDrawings(sym, tf);
      }
    } catch (err) {
      alert(`删除失败: ${err?.message || err}`);  // SAFE: alert renders text
    }
    return;
  }

  if (action === 'refresh-all-drawings') {
    void refreshAllDrawings();
    return;
  }

  if (action === 'select-line') {
    const id = btn.dataset.lineId;
    // Toggle: clicking the same line again releases it
    if (drawingsState.selectedLineId === id) {
      setSelectedManualLine(null);
    } else {
      setSelectedManualLine(id);
    }
    return;
  }

  if (action === 'deselect-line') {
    setSelectedManualLine(null);
    return;
  }

  if (action === 'setup-order') {
    openOrderModal();
    return;
  }

  if (action === 'toggle-similar') {
    showSimilarOnChart = !showSimilarOnChart;
    if (showSimilarOnChart && currentAnalyze?.ok && currentAnalyze.similar_lines) {
      setSimilarLines(currentAnalyze.similar_lines);
    } else {
      setSimilarLines([]);
    }
    return;
  }

  if (action === 'bitget-cancel') {
    const sym = btn.dataset.symbol;
    const oid = btn.dataset.oid;
    if (!sym || !oid) return;
    if (!confirm(`撤掉 Bitget 上的订单 …${oid.slice(-8)}?`)) return;
    try {
      const r = await fetchJson(
        `/api/live-execution/cancel-order?symbol=${encodeURIComponent(sym)}&order_id=${encodeURIComponent(oid)}&mode=live`,
        { method: 'POST', timeout: 35000, noCache: true },
      );
      if (r && r.ok === false) {
        alert(`撤单失败: ${r.reason || '未知'}`);  // SAFE: alert() renders text, not HTML
      }
    } catch (err) {
      const detail = err?.body?.detail || err?.body?.reason || err?.message || String(err);
      alert(`撤单失败: ${detail}`);
    }
    void refreshBitgetAccount();
    void refreshPending();
    return;
  }

  if (action === 'cancel-cond') {
    const cid = btn.dataset.cid;
    if (cid && confirm('取消这个 pending 条件单?')) {
      try {
        await condSvc.cancelConditional(cid);
      } catch (err) {
        const detail = err?.body?.detail || err?.body?.reason || err?.message || String(err);
        alert(`取消失败: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
      }
      void refreshPending();
    }
    return;
  }

  // Cancel a TRIGGERED cond by hitting Bitget cancel-order endpoint
  // (which now also marks the local cond cancelled in one shot).
  if (action === 'bitget-cancel-from-cond') {
    const sym = btn.dataset.symbol;
    const oid = btn.dataset.oid;
    if (!sym || !oid) return;
    if (!confirm(`撤掉 Bitget 上的订单 …${oid.slice(-8)} 并标记本地 cancelled?`)) return;
    try {
      const r = await fetchJson(
        `/api/live-execution/cancel-order?symbol=${encodeURIComponent(sym)}&order_id=${encodeURIComponent(oid)}&mode=live`,
        { method: 'POST', timeout: 35000, noCache: true },
      );
      console.log('[cond_panel] bitget cancel-from-cond resp', r);
    } catch (err) {
      const detail = err?.body?.detail || err?.body?.reason || err?.message || String(err);
      alert(`撤单失败: ${detail}`);
    }
    void refreshPending();
    void refreshBitgetAccount();
    return;
  }

  if (action === 'delete-cond') {
    const cid = btn.dataset.cid;
    if (!cid) return;
    // Find the cond in our local list to check if it has a live Bitget order
    const cond = knownConditionals.find((c) => c.conditional_id === cid);
    const hasLiveBitget = cond && cond.status === 'triggered' && cond.exchange_order_id;
    let msg = '删除这条本地记录?';
    if (hasLiveBitget) {
      msg = '⚠ 这条 cond 还在 Bitget 上挂着!\n' +
            '删除前会先撤 Bitget 实单；如果撤单无法确认，后端会拒绝删除。\n\n' +
            '继续?';
    }
    if (!confirm(msg)) return;
    try {
      await condSvc.deleteConditional(cid);
    } catch (err) {
      const detail = err?.body?.detail || err?.body?.reason || err?.message || String(err);
      alert(`删除失败: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
    }
    void refreshPending();
    return;
  }

  if (action === 'expand-cond') {
    const cid = btn.dataset.cid;
    const el = panel.querySelector(`[data-events-for="${cid}"]`);
    if (el) el.classList.toggle('open');
    return;
  }

  if (action === 'delete-line') {
    const id = btn.dataset.lineId;
    if (!id) return;
    console.log('[cond_panel] delete-line clicked', id);
    const oldText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '删除中...';
    try {
      const { deleteManualDrawing } = await import('../services/drawings.js');
      const resp = await deleteManualDrawing(id);
      console.log('[cond_panel] delete response', resp);
    } catch (err) {
      console.error('[cond_panel] delete failed', err);
      const detail = err?.body?.detail || err?.body?.reason || err?.userHint || err?.message || String(err);
      alert(`删除失败: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);  // SAFE: alert() renders text, not HTML
      // rollback — re-fetch the real list
      try {
        const mod = await import('./drawings/manual_trendline_controller.js');
        await mod.refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
      } catch (refreshErr) {
        console.warn('[cond_panel] refresh after failed delete failed', refreshErr);
      }
      btn.disabled = false;
      btn.textContent = oldText || '删除';
      return;
    }
    if (currentLine?.manual_line_id === id) {
      currentLine = null;
      currentAnalyze = null;
    }
    // Hard resync after success to ensure chart overlay and state agree
    try {
      const mod = await import('./drawings/manual_trendline_controller.js');
      await mod.refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    } catch (refreshErr) {
      console.warn('[cond_panel] refresh after delete failed', refreshErr);
    }
    return;
  }

  if (action === 'clear-all-lines') {
    if (!confirm('清空所有画线?')) return;
    console.log('[cond_panel] clear-all-lines clicked');
    try {
      const { clearManualDrawings } = await import('../services/drawings.js');
      await clearManualDrawings(marketState.currentSymbol, marketState.currentInterval);
      const { setManualDrawings } = await import('../state/drawings.js');
      setManualDrawings([]);
      currentLine = null;
      currentAnalyze = null;
      render();
    } catch (err) {
      console.error('[cond_panel] clear failed', err);
      const detail = err?.body?.detail || err?.body?.reason || err?.userHint || err?.message || String(err);
      alert(`清空失败: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);  // SAFE: alert() renders text, not HTML
    }
    return;
  }
}

function onPanelHover(e) {
  const row = e.target.closest('[data-hover-line]');
  if (!row) return;
  const id = row.dataset.hoverLine;
  if (id) setHoveredLineFromPanel(id);
}

function onPanelLeave(e) {
  const row = e.target.closest('[data-hover-line]');
  if (!row) return;
  const related = e.relatedTarget;
  if (related && row.contains(related)) return;
  setHoveredLineFromPanel(null);
}

// ─────────────────────────────────────────────────────────────
// Main render
// ─────────────────────────────────────────────────────────────
function render(opts = {}) {
  if (!panel) return;
  const lines = drawingsState.lines || [];

  panel.innerHTML = `
    ${renderBitgetSection()}
    ${renderLineListSection(lines)}
    ${currentLine ? renderSelectedSection(opts.loading) : ''}
  `;
  // Conditionals-list section (renderPendingSection) intentionally
  // omitted 2026-04-20 per user: cumulative cancelled rows piled up to
  // dozens, making the useful sections (手画线 + selected) hard to scan.
  // The data + API still exists for programmatic use; only the sidebar
  // list is hidden.
}

function renderBitgetSection() {
  const a = bitgetAccount || {};
  const equity = Number(a.total_equity || 0);
  const free = Number(a.usdt_available || 0);
  const positions = (a.positions || []).filter((p) => Number(p.total) > 0);
  const pending = (a.pending_orders || []).filter((p) => p.orderId);

  const posRows = positions.length
    ? positions.map((p) => {
        const dir = p.holdSide || '';
        const dirColor = dir === 'long' ? '#00e676' : '#ff5252';
        // SL / TP line — shows Bitget-side protection so the user can see
        // at a glance their short is covered (instead of assuming it's
        // naked). 2026-04-21: user reported "我不知道这个单有没有 SL/TP"
        // because sidebar only showed entry + mark.
        const sl = Number(p.sl_trigger);
        const tp = Number(p.tp_trigger);
        const slCount = Number(p.sl_count || 0);
        const tpCount = Number(p.tp_count || 0);
        const dupWarn = (slCount > 1 || tpCount > 1)
          ? `<span title="Bitget 上挂了 ${slCount} 份 SL + ${tpCount} 份 TP,建议到 Bitget app 清理重复" style="color:#fbbf24;margin-left:6px">⚠ 重复</span>`
          : '';
        const sltpLine = (sl > 0 || tp > 0)
          ? `<div style="font-size:10px;color:#94a1b7;padding-left:8px">
               SL <b style="color:#ef5350">${sl > 0 ? sl.toFixed(4) : '—'}</b>
               · TP <b style="color:#26a69a">${tp > 0 ? tp.toFixed(4) : '—'}</b>
               ${dupWarn}
             </div>`
          : `<div style="font-size:10px;color:#ff9800;padding-left:8px">⚠ 无 SL/TP 保护</div>`;
        return `<div class="cond-row cond-pending-clickable" style="font-size:10px;cursor:pointer" data-jump-symbol="${esc(p.symbol || '')}" data-jump-tf="" title="点击跳转到 ${esc(p.symbol || '')} 图表">
          <b style="color:${dirColor}">${esc(dir.toUpperCase())}</b> ${esc(p.symbol || '')}
          · qty <b>${esc(String(p.total ?? '?'))}</b>
          · entry ${esc(String(p.averageOpenPrice ?? '?'))}
          · mark ${esc(String(p.markPrice ?? '?'))}
          ${sltpLine}
        </div>`;
      }).join('')
    : '<p class="muted" style="font-size:10px;margin:4px 0">无持仓</p>';

  // For each pending order, find the matching local conditional (by
  // exchange_order_id) so we can surface its timeframe. Click the row
  // → jump the chart to that symbol + tf. User 2026-04-21: "这些依然
  // 无法点击 然后跳转到所属的币的tf".
  const condsByOid = new Map();
  for (const c of knownConditionals) {
    if (c.exchange_order_id) condsByOid.set(String(c.exchange_order_id), c);
  }
  const orderRows = pending.length
    ? pending.map((o) => {
        const sym = o.symbol || '';
        const oid = String(o.orderId || '');
        const matchedCond = condsByOid.get(oid);
        const tf = matchedCond?.timeframe || '';
        const tfTag = tf ? `<span style="background:#223049;padding:1px 4px;border-radius:3px;margin-left:3px;font-size:9px;color:#8ac4ff">${esc(tf)}</span>` : '';
        return `<div class="cond-row cond-pending-clickable" style="font-size:10px;cursor:pointer" data-jump-symbol="${esc(sym)}" data-jump-tf="${esc(tf)}" title="点击跳转到 ${esc(sym)} ${esc(tf)} 图表">
          ${esc(sym)} ${tfTag} · ${esc(o.side || '')} · ${esc(o.size || '')}
          · @ ${esc(String(o.price ?? '?'))}
          · <span class="cond-status">${esc(o.status || '')}</span>
          <button class="cond-btn-sm cond-btn-warn" data-action="bitget-cancel" data-symbol="${esc(sym)}" data-oid="${esc(oid)}" title="撤销挂单">撤</button>
        </div>`;
      }).join('')
    : '<p class="muted" style="font-size:10px;margin:4px 0">无挂单</p>';

  return `
    <div class="cond-section" data-bitget-section>
      <div class="cond-header">
        Bitget 实盘
        <span class="cond-spacer"></span>
        <span style="color:#888;font-size:10px">权益 <b style="color:#00e676">$${equity.toFixed(2)}</b> · 可用 <b>$${free.toFixed(2)}</b></span>
      </div>
      <div style="font-size:10px;color:#fbbf24;margin-top:4px">持仓 ${positions.length}</div>
      ${posRows}
      <div style="font-size:10px;color:#fbbf24;margin-top:4px">挂单 ${pending.length}</div>
      ${orderRows}
    </div>
  `;
}

function renderLineListSection(lines) {
  // 2026-04-21 redesign per user: group by symbol across ALL drawings.
  // Click a symbol → expand to see all that symbol's lines. Click a
  // line → jump chart to that symbol+tf + select it. Much cleaner than
  // the old flat "7 lines stacked vertically" look.
  const all = Array.isArray(allDrawings) ? allDrawings : [];
  const currentSym = marketState.currentSymbol;

  // Group by symbol
  const bySymbol = new Map();
  for (const d of all) {
    const sym = (d.symbol || '').toUpperCase();
    if (!sym) continue;
    if (!bySymbol.has(sym)) bySymbol.set(sym, []);
    bySymbol.get(sym).push(d);
  }
  // Sort symbols: current symbol first, then by most-recent drawing
  const symbols = [...bySymbol.keys()].sort((a, b) => {
    if (a === currentSym) return -1;
    if (b === currentSym) return 1;
    const la = Math.max(...bySymbol.get(a).map((d) => d.updated_at || d.created_at || 0));
    const lb = Math.max(...bySymbol.get(b).map((d) => d.updated_at || d.created_at || 0));
    return lb - la;
  });

  if (all.length === 0) {
    return `
      <div class="cond-section">
        <div class="cond-header">我的手画线 <span class="cond-count">0</span></div>
        <p class="muted">按 T 画一条线开始</p>
      </div>
    `;
  }

  const groups = symbols.map((sym) => {
    const items = bySymbol.get(sym).slice().sort((a, b) => (b.updated_at || b.created_at || 0) - (a.updated_at || a.created_at || 0));
    const tfSet = new Set(items.map((d) => d.timeframe));
    const tfList = [...tfSet].join(' · ');
    const isExpanded = _expandedSymbol === sym || (_expandedSymbol === null && sym === currentSym);
    const isCurrent = sym === currentSym;
    const arrow = isExpanded ? '▾' : '▸';

    const rows = !isExpanded ? '' : items.map((l) => {
      const isSel = l.manual_line_id === drawingsState.selectedLineId;
      const tfTag = `<span class="mydraw-tf">${esc(l.timeframe)}</span>`;
      const sideColor = l.side === 'support' ? '#2dd4bf' : '#fb923c';
      return `
        <div class="mydraw-line-row ${isSel ? 'is-selected' : ''}" data-jump-line="1" data-line-id="${esc(l.manual_line_id)}" data-line-symbol="${esc(l.symbol)}" data-line-tf="${esc(l.timeframe)}" title="点击跳转到 ${esc(l.symbol)} ${esc(l.timeframe)} 并选中此线">
          <span class="mydraw-dot" style="background:${sideColor}"></span>
          ${tfTag}
          <span class="mydraw-prices">${esc(fmtPrice(l.price_start))} → ${esc(fmtPrice(l.price_end))}</span>
          <span class="mydraw-time">${esc(fmtDateShort(l.updated_at || l.created_at))}</span>
          <button class="mydraw-del" data-action="delete-line-any" data-line-id="${esc(l.manual_line_id)}" data-line-symbol="${esc(l.symbol)}" data-line-tf="${esc(l.timeframe)}" title="删除此线">×</button>
        </div>
      `;
    }).join('');

    return `
      <div class="mydraw-group ${isCurrent ? 'is-current' : ''}">
        <div class="mydraw-group-head" data-toggle-symbol="${esc(sym)}">
          <span class="mydraw-arrow">${arrow}</span>
          <b class="mydraw-sym">${esc(sym)}</b>
          <span class="mydraw-group-count">${items.length} 线</span>
          <span class="mydraw-group-tfs">${esc(tfList)}</span>
        </div>
        ${rows ? `<div class="mydraw-lines">${rows}</div>` : ''}
      </div>
    `;
  }).join('');

  return `
    <div class="cond-section">
      <div class="cond-header">
        我的手画线 <span class="cond-count">${all.length}</span>
        <span class="cond-spacer"></span>
        <button class="cond-btn-sm" data-action="refresh-all-drawings" title="刷新">⟳</button>
        <button class="cond-btn-sm cond-btn-danger" data-action="clear-all-lines" title="清空当前币种的画线">清空</button>
      </div>
      <div class="mydraw-groups">${groups}</div>
      <p class="muted" style="font-size:10px;margin-top:4px">点币种展开 · 点线跳转并选中 · × 删除</p>
    </div>
  `;
}

function renderSelectedSection(loading) {
  if (!currentLine) return '';
  const sideColor = '#fbbf24';  // yellow — neutral line, no S/R distinction

  let analyzeHtml = '';
  if (loading) {
    analyzeHtml = '<p class="muted" style="font-size:10px">分析中...</p>';
  } else if (!currentAnalyze?.ok) {
    analyzeHtml = '<p class="muted" style="font-size:10px">无历史样本 — 可以直接挂单</p>';
  } else {
    const s = currentAnalyze.stats || {};
    const n = s.sample_size || 0;
    if (n === 0) {
      analyzeHtml = '<p class="muted" style="font-size:10px">无历史样本 — 可以直接挂单</p>';
    } else {
      const trustColor = { high: '#00e676', medium: '#fbbf24', low: '#ff8a80', none: '#888' }[s.trustworthiness || 'none'] || '#888';
      analyzeHtml = `
        <div class="cond-stats-grid">
          <div><span>样本</span><b>${n}</b></div>
          <div><span>反弹</span><b style="color:${s.p_bounce >= 0.5 ? '#00e676' : '#888'}">${fmtPct(s.p_bounce)}</b></div>
          <div><span>破位</span><b>${fmtPct(s.p_break)}</b></div>
          <div><span>期望值</span><b style="color:${s.expected_value > 0 ? '#00e676' : '#ff8a80'}">${fmtR(s.expected_value)}</b></div>
          <div style="grid-column:span 2"><span>可信度</span><b style="color:${trustColor}">${esc(s.trustworthiness || 'none')}</b></div>
        </div>
      `;
    }
  }

  // Smart-features panel — pulled directly from currentAnalyze top-level fields
  let smartHtml = '';
  if (currentAnalyze?.ok) {
    const v = currentAnalyze.volatility || null;
    const c = currentAnalyze.coherence || null;
    const lq = currentAnalyze.line_quality || null;
    const ds = currentAnalyze.distance || null;
    const r = currentAnalyze.recommendations || null;
    const lines = [];
    if (v) {
      const regimeColor = v.regime === 'wide' ? '#ff5252' : v.regime === 'tight' ? '#60a5fa' : '#fbbf24';
      let line = `ATR <b style="color:${regimeColor}">${(v.atr_pct||0).toFixed(2)}%</b> · ${esc(v.regime||'')}`;
      if (v.bb_state && v.bb_state !== 'normal') {
        const bbColor = v.bb_state === 'squeeze' ? '#fbbf24' : '#00e676';
        const bbZh = v.bb_state === 'squeeze' ? '挤压' : '扩张';
        line += ` · BB <b style="color:${bbColor}">${bbZh}</b>`;
      }
      lines.push(line);
    }
    if (c?.score > 0) {
      const matchStr = (c.matches || []).map(m => `${esc(m.tf)}@${esc(fmtPrice(m.price))}`).join(', ');
      lines.push(`📐 高TF对齐 <b style="color:#00e676">${c.score}</b>: ${matchStr}`);
    }
    if (lq?.touch_count > 2) {
      lines.push(`👆 触点 <b style="color:#00e676">${lq.touch_count}</b> (强度 ${Math.round((lq.strength||0)*100)}%)`);
    }
    if (ds?.zone && ds.zone !== 'medium') {
      const dColor = ds.zone === 'near' ? '#fbbf24' : '#888';
      lines.push(`📏 距线 <b style="color:${dColor}">${esc(ds.zone)}</b> (${(ds.gap_pct||0).toFixed(2)}%)`);
    }
    if (r) {
      lines.push(`<span style="color:#94a3b8">推荐: Buffer <b style="color:#fbbf24">${r.tolerance_pct}%</b> · SL=线 · RR <b style="color:#fbbf24">${r.rr_target}</b></span>`);
    }
    if (lines.length) {
      smartHtml = `
        <div class="cond-smart" style="margin:6px 0;padding:6px 8px;background:rgba(96,165,250,0.07);border:1px solid rgba(96,165,250,0.25);border-radius:4px;font-size:10px;line-height:1.7;color:#94a3b8">
          <b style="color:#60a5fa">🧠 智能调参</b><br>
          ${lines.join('<br>')}
        </div>
      `;
    }
  }

  let contextHtml = '';
  if (marketContext && !marketContext.error) {
    const mc = marketContext;
    const trendClass = mc.trend_label === 'UPTREND' ? 'pnl-pos' : mc.trend_label === 'DOWNTREND' ? 'pnl-neg' : '';
    const trendZh = { UPTREND: '上升', DOWNTREND: '下降', RANGE: '震荡' }[mc.trend_label] || mc.trend_label || '—';
    contextHtml = `
      <div class="cond-ctx">
        <div><span>趋势</span><b class="${trendClass}">${esc(trendZh)} ${mc.trend_slope_pct > 0 ? '+' : ''}${mc.trend_slope_pct}%</b></div>
        <div><span>MA排列</span><b>${esc({ BULL_ORDERED: '多头', BEAR_ORDERED: '空头', MIXED: '混合', NEUTRAL: '中性' }[mc.ma_alignment] || mc.ma_alignment || '—')}</b></div>
      </div>
    `;
  }

  return `
    <div class="cond-section cond-selected">
      <div class="cond-header">
        <span class="cond-sel-dot" style="background:${sideColor}"></span>
        <strong>选中的线</strong>
        <span class="cond-sel-price">${esc(fmtPrice(currentLine.price_start))} → ${esc(fmtPrice(currentLine.price_end))}</span>
        <span class="cond-spacer"></span>
        <button class="cond-btn-sm" data-action="deselect-line">取消选中</button>
      </div>
      ${contextHtml}
      ${analyzeHtml}
      ${smartHtml}
      <div style="margin:6px 0;display:flex;gap:6px;align-items:center;font-size:10px">
        <label style="display:flex;align-items:center;gap:4px;cursor:pointer">
          <input type="checkbox" data-action="toggle-similar" ${showSimilarOnChart ? 'checked' : ''} />
          <span>图上显示历史相似线</span>
        </label>
      </div>
      <div class="cond-actions-dual">
        <button class="cond-btn cond-btn-primary" data-action="setup-order" style="width:100%">挂单到 Bitget →</button>
      </div>
    </div>
  `;
}

function renderPendingSection(returnHtml = false) {
  const pending = knownConditionals.filter((c) => c.status === 'pending');
  const triggered = knownConditionals.filter((c) => c.status === 'triggered');
  const cancelled = knownConditionals.filter((c) => c.status === 'cancelled');

  if (!knownConditionals.length) {
    const html = `
      <div class="cond-section">
        <div class="cond-header">条件单 <span class="cond-count">0</span></div>
        <p class="muted" style="font-size:10px">选一条线后挂反弹单或突破单</p>
      </div>
    `;
    if (returnHtml) return html;
    const body = panel?.querySelector('.cond-pending-body');
    if (body) body.innerHTML = '<p class="muted" style="font-size:10px">暂无条件单</p>';
    return '';
  }

  const row = (c) => {
    const statusColor = { pending: '#fbbf24', triggered: '#00e676', cancelled: '#888', failed: '#ff1744' }[c.status] || '#888';
    const distanceText = c.last_distance_atr != null ? `${c.last_distance_atr.toFixed(2)} ATR` : '未探测';
    const marketText = c.last_market_price != null ? fmtPrice(c.last_market_price) : '—';
    const lineText = c.last_line_price != null ? fmtPrice(c.last_line_price) : '—';
    const kindZh = c.order.order_kind === 'breakout' ? '突破' : '反弹';
    const kindColor = c.order.order_kind === 'breakout' ? '#ffa726' : '#00e676';
    const events = (c.events || []).slice(-5).reverse();
    const notional = Number(c.order.notional_usd);
    const notionalStr = notional > 0 ? `${notional.toFixed(0)}U` : '—';
    const oid = c.exchange_order_id || '';
    const oidShort = oid ? oid.slice(-8) : '';   // last 8 — more distinguishable than first

    return `
      <div class="cond-row" data-cid="${esc(c.conditional_id)}">
        <div class="cond-row-header">
          <span class="cond-dot" style="background:${statusColor}"></span>
          <strong>${esc(c.symbol)} ${esc(c.timeframe)}</strong>
          <span class="cond-kind" style="color:${kindColor}">${esc(kindZh)}</span>
          <span class="cond-dir">${esc(c.order.direction)}</span>
          <span class="cond-status" style="color:${statusColor}">${esc(c.status)}</span>
          <span class="cond-spacer"></span>
          <button class="cond-btn-sm" data-action="expand-cond" data-cid="${esc(c.conditional_id)}">log</button>
          ${c.status === 'pending'
            ? `<button class="cond-btn-sm cond-btn-warn" data-action="cancel-cond" data-cid="${esc(c.conditional_id)}">取消</button>`
            : ''}
          ${c.status === 'triggered' && oid
            ? `<button class="cond-btn-sm cond-btn-warn" data-action="bitget-cancel-from-cond" data-cid="${esc(c.conditional_id)}" data-symbol="${esc(c.symbol)}" data-oid="${esc(oid)}" title="撤掉 Bitget 上的挂单 + 标记本地 cancelled">撤Bitget</button>`
            : ''}
          <button class="cond-btn-sm cond-btn-danger" data-action="delete-cond" data-cid="${esc(c.conditional_id)}" title="${c.status === 'triggered' ? '先撤 Bitget,确认后再删本地记录' : '删除本地记录'}">×</button>
        </div>
        <div class="cond-row-stats">
          mkt <b>${marketText}</b> · line <b>${lineText}</b> · <b>${esc(distanceText)}</b>
          · ${notionalStr} · RR ${esc(String(c.order.rr_target ?? '—'))}
          ${oid ? `· <span style="color:#00e676">✓ Bitget …${esc(oidShort)}</span>` : '· 仅警报'}
        </div>
        <div class="cond-events-panel" data-events-for="${esc(c.conditional_id)}">
          ${events.map(e => `
            <div class="cond-event">
              <span class="cond-event-time">${fmtTime(e.ts)}</span>
              <span class="cond-event-kind">${esc(e.kind)}</span>
              <span class="cond-event-msg">${esc(e.message || '')}</span>
            </div>
          `).join('') || '<div class="cond-event muted">无事件</div>'}
        </div>
      </div>
    `;
  };

  const html = `
    <div class="cond-section">
      <div class="cond-header">
        条件单 <span class="cond-count">${pending.length} 活跃 · ${triggered.length} 触发 · ${cancelled.length} 取消</span>
      </div>
      <div class="cond-pending-body">${knownConditionals.map(row).join('')}</div>
    </div>
  `;
  if (returnHtml) return html;
  // When called from refreshPending, re-render the whole panel so all
  // sections stay in sync (the old approach updated only the pending
  // body and lost scroll / hover state, confusing users).
  render();
  return '';
}

// ─────────────────────────────────────────────────────────────
// Order modal
// ─────────────────────────────────────────────────────────────
async function openOrderModal() {
  if (!currentLine) return;
  const line = {
    manual_line_id: currentLine.manual_line_id,
    symbol: currentLine.symbol,
    timeframe: currentLine.timeframe,
    t_start: Number(currentLine.t_start),
    t_end: Number(currentLine.t_end),
    price_start: Number(currentLine.price_start),
    price_end: Number(currentLine.price_end),
    extend_left: !!currentLine.extend_left,
    extend_right: currentLine.extend_right !== false,
  };

  // Show an instant loading scrim so user sees IMMEDIATE feedback after
  // clicking the button. Otherwise modal sits invisible for up to 15s
  // while balance fetch runs and the user thinks the click didn't register.
  const loadingScrim = document.createElement('div');
  loadingScrim.className = 'cond-modal-bg';
  loadingScrim.innerHTML = `
    <div class="cond-modal" id="cm-loading" style="text-align:center;padding:30px;min-width:320px">
      <div style="font-size:13px;color:#94a3b8;margin-bottom:10px">挂单 · ${esc(line.symbol)} ${esc(line.timeframe)}</div>
      <div style="color:#fbbf24;font-size:11px">⏳ 加载账户 / mark / 分析中…</div>
      <div style="color:#666;font-size:9px;margin-top:8px">最长 15s · Esc 取消</div>
    </div>
  `;
  let scrimEscHandler = null;
  const removeScrim = () => {
    try { loadingScrim.remove(); } catch {}
    if (scrimEscHandler) { document.removeEventListener('keydown', scrimEscHandler); scrimEscHandler = null; }
  };
  let cancelled = false;
  scrimEscHandler = (ev) => { if (ev.key === 'Escape') { cancelled = true; removeScrim(); } };
  document.addEventListener('keydown', scrimEscHandler);
  loadingScrim.addEventListener('click', (ev) => { if (ev.target === loadingScrim) { cancelled = true; removeScrim(); } });
  document.body.appendChild(loadingScrim);

  // refPrice projection. Match backend extension semantics so the modal
  // shows the same moving line price the Bitget plan order will track.
  let refPrice = line.price_end || line.price_start || 0;
  try {
    const ts = Math.floor(Date.now() / 1000);
    const span = line.t_end - line.t_start;
    if (span > 0) {
      const slope = (line.price_end - line.price_start) / span;
      const projected = line.price_start + slope * (ts - line.t_start);
      if (ts < line.t_start && !line.extend_left) refPrice = line.price_start;
      else if (ts > line.t_end && !line.extend_right) refPrice = line.price_end;
      else refPrice = projected;
    }
  } catch {}

  let balance = null;
  let liveMark = null;
  let recs = null;
  let analyzeStats = null;
  let acctErr = null;
  let markErr = null;

  // Kick analyze in BACKGROUND — it takes 5-8s and is not required for
  // order submission. Modal should render as soon as account+mark are
  // done; analyze results land in a slot later.
  const analyzePromise = condSvc.analyzeDrawing(line.manual_line_id)
    .catch(() => null);

  try {
    const [acct, mark] = await Promise.all([
      fetchJson('/api/live-execution/account?mode=live', { timeout: 8000, noCache: true })
        .catch((e) => { acctErr = e; return null; }),
      fetchMarkPriceOnce(line.symbol).catch((e) => { markErr = e; return null; }),
    ]);
    if (cancelled) return;   // user pressed Esc / clicked outside
    balance = Number(acct?.usdt_available ?? acct?.total_equity ?? 0);
    liveMark = mark;
  } catch (e) { console.error('[cond_panel] modal init fetch err', e); }
  console.log('[cond_panel] modal data (fast path)', { balance, liveMark, refPrice, acctErr, markErr });

  const defTolerance = recs?.tolerance_pct ?? 0.1;
  const defRR = recs?.rr_target ?? 2;
  const defaultPct = 20;
  const defaultSize = balance ? Math.max(5, Math.floor(balance * defaultPct / 100)) : 10;

  // Role label — only show if mark loaded; otherwise say "未知 mark"
  let autoSide;
  if (liveMark == null) {
    autoSide = '⚠ mark 未加载，无法判断';
  } else if (refPrice > liveMark) {
    autoSide = `阻力(线 ${fmtPrice(refPrice)} > mark ${fmtPrice(liveMark)})`;
  } else {
    autoSide = `支撑(线 ${fmtPrice(refPrice)} < mark ${fmtPrice(liveMark)})`;
  }

  const modal = document.createElement('div');
  modal.className = 'cond-modal-bg';
  modal.innerHTML = `
    <div class="cond-modal" id="cm-box">
      <h3>挂单 · ${esc(line.symbol)} ${esc(line.timeframe)}</h3>
      <div class="cond-modal-desc">
        线价(at now) <b>${esc(fmtPrice(refPrice))}</b> · Bitget 实时 <b style="color:#fbbf24">${liveMark != null ? esc(fmtPrice(liveMark)) : '加载失败'}</b> ·
        可用 <b style="color:${balance > 0 ? '#00e676' : '#ff5252'}">$${balance ? balance.toFixed(2) : '加载失败'}</b><br>
        <span style="color:#888;font-size:10px">${esc(autoSide)}</span>
      </div>

      <div class="cond-modal-row">
        <label>类型</label>
        <div style="flex:1;display:flex;gap:6px">
          <button type="button" name="kind" data-kind="bounce" class="is-active" style="flex:1;padding:6px;background:#0d4a2a;color:#00e676;border:1px solid #00e676;border-radius:3px;cursor:pointer">反弹</button>
          <button type="button" name="kind" data-kind="break" style="flex:1;padding:6px;background:#0b0f17;color:#888;border:1px solid #2a3548;border-radius:3px;cursor:pointer">突破</button>
        </div>
      </div>

      <div class="cond-modal-row">
        <label>方向</label>
        <div style="flex:1;display:flex;gap:6px;align-items:center">
          <span data-auto-dir style="flex:1;padding:5px 10px;background:#0b0f17;color:#fbbf24;border:1px dashed #fbbf24;border-radius:3px;font-size:11px">自动</span>
          <button type="button" name="dir-override" data-dir="long" style="padding:5px 8px;background:#0b0f17;color:#888;border:1px solid #2a3548;border-radius:3px;cursor:pointer;font-size:10px">强制 LONG</button>
          <button type="button" name="dir-override" data-dir="short" style="padding:5px 8px;background:#0b0f17;color:#888;border:1px solid #2a3548;border-radius:3px;cursor:pointer;font-size:10px">强制 SHORT</button>
        </div>
      </div>

      <div class="cond-modal-stats-slot" id="cm-analyze-slot" style="margin:8px 0;padding:6px 10px;background:rgba(138,149,166,0.06);border:1px solid rgba(138,149,166,0.25);border-radius:4px;font-size:10px;color:#8a95a6">
        ⏳ 历史分析中 (~6s)… 不必等,直接填参数可挂单
      </div>

      <div class="cond-modal-row">
        <label>容差 (%)</label>
        <input type="number" name="tolerance_pct" value="${defTolerance}" step="0.05" min="0.01" />
        <small style="color:#888;margin-left:4px;font-size:10px">价格摸到线 ± 容差时触发</small>
      </div>
      <div class="cond-modal-row">
        <label>止损</label>
        <div style="flex:1;color:#fbbf24;background:#0b0f17;border:1px solid #2a3548;border-radius:3px;padding:5px 8px;font-size:11px">SL = trendline</div>
        <small style="color:#888;margin-left:4px;font-size:10px">价格穿过线即离场</small>
      </div>
      <div class="cond-modal-row">
        <label>RR 目标</label>
        <input type="number" name="rr" value="${defRR}" step="0.5" min="0.5" max="10" />
      </div>
      <div class="cond-modal-row">
        <label>仓位</label>
        ${balance > 0 ? `
        <div style="flex:1;display:flex;gap:6px;align-items:center">
          <input type="number" name="size_pct" value="${defaultPct}" step="5" min="1" max="100" style="flex:1" />
          <span style="color:#888;font-size:10px">% of $${balance.toFixed(2)} =</span>
          <b data-size-usdt style="color:#fbbf24">$${defaultSize}</b>
        </div>
        ` : `
        <div style="flex:1;display:flex;gap:6px;align-items:center">
          <input type="number" name="size_usdt_direct" value="6" step="1" min="5" max="10000" style="flex:1" />
          <span style="color:#fbbf24;font-size:10px">USDT (余额未加载,请直接填)</span>
        </div>
        `}
      </div>
      <div class="cond-modal-row">
        <label>杠杆</label>
        <input type="number" name="leverage" value="5" step="1" min="1" max="50" />
      </div>
      <div class="cond-modal-status" data-status hidden></div>
      <div class="cond-modal-actions">
        <button class="cond-btn" type="button" name="cancel">取消</button>
        <button class="cond-btn cond-btn-primary" type="button" name="confirm">确认挂到 Bitget</button>
      </div>
    </div>
  `;
  removeScrim();
  document.body.appendChild(modal);

  // Analyze is still running in the background — when it resolves, fill
  // the slot. Modal is already usable; this is non-blocking enrichment.
  (async () => {
    try {
      const analyzeResp = await analyzePromise;
      if (!analyzeResp?.ok) return;
      const slot = modal.querySelector('#cm-analyze-slot');
      if (!slot) return;
      const stats = analyzeResp.stats || {};
      const r = analyzeResp.recommendations || null;
      if (!stats.sample_size || stats.sample_size <= 0) {
        slot.innerHTML = '<span style="color:#8a95a6">无历史样本 — 直接按默认参数挂单</span>';
        return;
      }
      const pct = (x) => Math.round((x||0)*100);
      const num = (x, d=2) => (Number(x)||0).toFixed(d);
      slot.style.background = 'rgba(251,191,36,0.06)';
      slot.style.border = '1px solid rgba(251,191,36,0.3)';
      slot.innerHTML = `
        <b style="color:#fbbf24">📊 历史 ${stats.sample_size} 条相似形态</b><br>
        反弹 <b style="color:#00e676">${pct(stats.p_bounce)}%</b> ·
        破位 <b style="color:#ff5252">${pct(stats.p_break)}%</b> ·
        EV <b style="color:${(stats.expected_value||0)>=0?'#00e676':'#ff5252'}">${num(stats.expected_value)}R</b><br>
        可信度 <b style="color:${stats.trustworthiness==='high'?'#00e676':stats.trustworthiness==='medium'?'#fbbf24':'#888'}">${esc(stats.trustworthiness||'low')}</b>
        ${r ? `· 推荐 Buffer ${r.tolerance_pct}% · SL=线 · RR ${r.rr_target}` : ''}
      `;
    } catch (err) {
      console.warn('[cond_panel] analyze background fill err', err);
    }
  })();

  const getField = (name) => modal.querySelector(`[name="${name}"]`);
  let _onKeyRef = null;
  const close = () => {
    try { modal.remove(); } catch {}
    if (_onKeyRef) { document.removeEventListener('keydown', _onKeyRef); _onKeyRef = null; }
  };

  // Live-update the $ value as user changes percentage (only when balance loaded)
  const sizePctInput = getField('size_pct');
  const sizeUsdtDirectInput = getField('size_usdt_direct');
  const sizeDisplay = modal.querySelector('[data-size-usdt]');
  if (sizePctInput && sizeDisplay) {
    const updateSize = () => {
      const pct = parseFloat(sizePctInput.value) || 0;
      const usd = balance ? (balance * pct / 100) : 0;
      sizeDisplay.textContent = `$${usd.toFixed(2)}`;
    };
    sizePctInput.addEventListener('input', updateSize);
  }

  // 反弹/突破 toggle
  let chosenKind = 'bounce';
  const kindBtns = modal.querySelectorAll('[name="kind"]');
  const syncKindBtns = () => {
    kindBtns.forEach((b) => {
      const isActive = b.dataset.kind === chosenKind;
      b.classList.toggle('is-active', isActive);
      b.style.background = isActive ? '#0d4a2a' : '#0b0f17';
      b.style.color = isActive ? '#00e676' : '#888';
      b.style.borderColor = isActive ? '#00e676' : '#2a3548';
    });
  };

  // Direction: 'auto' | 'long' | 'short'. Auto is computed at submit time.
  let dirOverride = 'auto';
  const dirBtns = modal.querySelectorAll('[name="dir-override"]');
  const syncDirBtns = () => {
    const autoSpan = modal.querySelector('[data-auto-dir]');
    if (autoSpan) {
      autoSpan.style.opacity = dirOverride === 'auto' ? '1' : '0.4';
    }
    dirBtns.forEach((b) => {
      const isActive = b.dataset.dir === dirOverride;
      b.style.background = isActive ? (b.dataset.dir === 'long' ? '#0d4a2a' : '#4a0d0d') : '#0b0f17';
      b.style.color = isActive ? (b.dataset.dir === 'long' ? '#00e676' : '#ff5252') : '#888';
      b.style.borderColor = isActive ? (b.dataset.dir === 'long' ? '#00e676' : '#ff5252') : '#2a3548';
    });
  };

  // Compute direction from line position vs current mark + bounce/break
  function computeDirection() {
    // Manual override always wins.
    if (dirOverride !== 'auto') return dirOverride;
    if (liveMark == null) return 'long';
    // Position-based: line BELOW mark = support → bounce LONG; line ABOVE = resistance → bounce SHORT.
    // Slope is irrelevant to the role — what matters is which side of the line price is on.
    const lineAbove = refPrice > liveMark;
    if (chosenKind === 'bounce') {
      return lineAbove ? 'short' : 'long';
    }
    return lineAbove ? 'long' : 'short';
  }

  const submit = async (ev) => {
    if (ev) { ev.preventDefault(); ev.stopPropagation(); }
    const confirmBtn = getField('confirm');
    confirmBtn.disabled = true;
    confirmBtn.textContent = '提交中...';
    // Null-safe field reads — modal might be in an unexpected DOM state
    const fieldVal = (name, fallback) => {
      const el = getField(name);
      if (!el) return fallback;
      const v = parseFloat(el.value);
      return Number.isFinite(v) && v > 0 ? v : fallback;
    };
    const tolerancePct = fieldVal('tolerance_pct', 0.1);
    const rrTarget = fieldVal('rr', 2.0);
    const leverage = Math.max(1, Math.min(100, Math.round(fieldVal('leverage', 5))));
    // Two modes: percentage of balance (when balance loaded) OR direct USDT
    let sizeUsdt;
    if (sizePctInput && balance > 0) {
      const sizePct = parseFloat(sizePctInput.value) || 0;
      sizeUsdt = balance * sizePct / 100;
    } else if (sizeUsdtDirectInput) {
      sizeUsdt = parseFloat(sizeUsdtDirectInput.value) || 0;
    } else {
      sizeUsdt = 0;
    }
    const direction = computeDirection();
    console.log('[cond_panel] computed direction:', direction, 'kind:', chosenKind);

    if (sizeUsdt < 5) {
      alert('Bitget 最小下单 5 USDT,请增加仓位%');
      confirmBtn.disabled = false;
      confirmBtn.textContent = '确认挂到 Bitget';
      return;
    }

    const statusEl = modal.querySelector('[data-status]');
    const showStatus = (html, kind = 'error') => {
      if (!statusEl) return;
      statusEl.hidden = false;
      statusEl.className = `cond-modal-status is-${kind}`;
      statusEl.innerHTML = html;
    };

    if (!line?.manual_line_id) {
      showStatus('<b>状态错误</b><br>选中的线丢失了 (currentLine null)。重新选中再点。', 'error');
      confirmBtn.disabled = false;
      confirmBtn.textContent = '确认挂到 Bitget';
      return;
    }

    try {
      // Place a real Bitget plan order. The backend persists it as a
      // triggered conditional so the watcher can move it with the line by
      // cancelling and replacing the exchange-side plan order.
      const resp = await condSvc.placeLineOrder({
        manual_line_id: line.manual_line_id,
        direction,
        kind: chosenKind,
        tolerance_pct: tolerancePct,
        stop_offset_pct: 0,
        size_usdt: sizeUsdt,
        leverage,
        mode: 'live',
        rr_target: rrTarget,
      });
      console.log('[cond_panel] placeLineOrder response', resp);

      if (!resp?.ok) {
        // Humanised rejection messages
        const reason = resp?.reason || resp?.result?.reason || resp?.error || 'unknown';
        let msg = `<b>下单被拒</b><br>`;
        if (reason === 'entry_above_market_would_fill_immediately') {
          msg += `你选的方向是 <b>LONG</b>,但 entry 价 <b>${resp.proposed_entry?.toFixed(4)}</b> 高于 Bitget 实时 mark <b>${resp.mark_price?.toFixed(4)}</b>。<br>LONG limit 必须低于市价才能 pending,否则立刻成交。<br><b>解决</b>: 把"进场距离 %"调大,让 entry 落到 ${resp.mark_price?.toFixed(4)} 以下,或改 SHORT。`;
        } else if (reason === 'entry_below_market_would_fill_immediately') {
          msg += `你选的方向是 <b>SHORT</b>,但 entry 价 <b>${resp.proposed_entry?.toFixed(4)}</b> 低于 Bitget 实时 mark <b>${resp.mark_price?.toFixed(4)}</b>。<br>SHORT limit 必须高于市价才能 pending。<br><b>解决</b>: 调大"进场距离 %",或改 LONG。`;
        } else if (String(reason).includes('less than the minimum amount')) {
          msg += `Bitget 最小下单 5 USDT,增加仓位%。`;
        } else {
          msg += `<code style="font-size:10px">${String(reason).slice(0,300)}</code>`;
        }
        showStatus(msg, 'error');
        confirmBtn.disabled = false;
        confirmBtn.textContent = '确认挂到 Bitget';
        return;
      }

      // Success - Bitget plan order is live and tracked by the watcher.
      const cid = resp.conditional?.conditional_id || '?';
      showStatus(
        `<b style="color:#00e676">✓ Bitget plan order 已挂出</b><br>` +
        `id: <code>${cid.slice(0, 18)}</code><br>` +
        `${esc(resp.message || '')}<br>` +
        `<small>watcher 会按周期移动这个 plan order，让它跟随斜线投影。<br>2 秒后关闭。</small>`,
        'success'
      );
      await refreshPending();
      setTimeout(close, 2000);
    } catch (err) {
      console.error('[cond_panel] live-pending failed', err);
      confirmBtn.disabled = false;
      confirmBtn.textContent = '确认挂到 Bitget';
      // FetchError carries body — pull detail/reason from it
      const body = err?.body;
      const detail = body?.detail || body?.reason || body?.error || err?.message || String(err);
      let humanMsg = `<b>下单被拒</b><br>`;
      const ds = String(detail);
      if (ds.startsWith('line_too_far_from_mark')) {
        humanMsg += `线离当前 mark 太远，无法挂单。可能原因:<br>`
          + `<small>• 你画的线已经过期(价格早就走过去了)<br>`
          + `• 线被反向外推到不合理位置(画在远未来 + 陡斜率)<br><br>`
          + `服务端返回: <code style="font-size:9px">${ds}</code></small>`;
      } else if (ds.startsWith('line has no usable price')) {
        humanMsg += `线投影到当前时间得到无效价格(<=0)。<br><small>${ds}</small>`;
      } else if (ds.includes('size_below_min_trade') || ds.includes('minimum')) {
        humanMsg += `Bitget 最小下单 5 USDT，请增加仓位(% 或 直接 USDT)。`;
      } else {
        humanMsg += `<code style="font-size:10px">${ds.slice(0,400)}</code>`;
      }
      showStatus(humanMsg, 'error');
    }
  };

  // Event delegation inside the modal — most robust against propagation quirks.
  modal.addEventListener('click', (ev) => {
    const t = ev.target.closest('button');
    if (!t) {
      if (ev.target === modal) close();
      return;
    }
    ev.stopPropagation();
    if (t.dataset && t.dataset.kind) {
      chosenKind = t.dataset.kind;
      syncKindBtns();
      return;
    }
    if (t.name === 'dir-override' && t.dataset.dir) {
      // Click again to toggle back to auto
      dirOverride = (dirOverride === t.dataset.dir) ? 'auto' : t.dataset.dir;
      syncDirBtns();
      return;
    }
    if (t.name === 'cancel') close();
    else if (t.name === 'confirm') void submit(ev);
  }, true);

  // Initial style sync
  syncKindBtns();
  syncDirBtns();

  // Escape closes
  _onKeyRef = (ev) => {
    if (ev.key === 'Escape') close();
  };
  document.addEventListener('keydown', _onKeyRef);
}
