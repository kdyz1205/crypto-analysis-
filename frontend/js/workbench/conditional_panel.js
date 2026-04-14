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
import { marketState } from '../state/market.js';
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

  void refreshBitgetAccount();
  bitgetTimer = setInterval(() => void refreshBitgetAccount(), 5000);

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

// ─────────────────────────────────────────────────────────────
// Event handling
// ─────────────────────────────────────────────────────────────
async function onPanelClick(e) {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;

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
        { method: 'POST', timeout: 10000, noCache: true },
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
      void condSvc.cancelConditional(cid).then(() => refreshPending());
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
        { method: 'POST', timeout: 10000, noCache: true },
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
            '删除只清本地记录,Bitget 那一单不会撤,会留下"鬼单"\n\n' +
            '继续删除? (建议先点"撤Bitget"按钮)';
    }
    if (!confirm(msg)) return;
    void condSvc.deleteConditional(cid).then(() => refreshPending());
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

    // Optimistic: remove from local state immediately so UI reacts NOW,
    // even if the backend call is slow or the refresh races.
    const localLines = (drawingsState.lines || []).filter(l => l.manual_line_id !== id);
    try {
      const { setManualDrawings } = await import('../state/drawings.js');
      setManualDrawings(localLines);
    } catch {}
    if (currentLine?.manual_line_id === id) {
      currentLine = null;
      currentAnalyze = null;
    }
    render();

    try {
      const resp = await fetchJson(`/api/drawings/${encodeURIComponent(id)}`, {
        method: 'DELETE', timeout: 8000,
      });
      console.log('[cond_panel] delete response', resp);
    } catch (err) {
      console.error('[cond_panel] delete failed', err);
      alert(`删除失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
      // rollback — re-fetch the real list
      try {
        const mod = await import('./drawings/manual_trendline_controller.js');
        await mod.refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
      } catch {}
      return;
    }
    // Hard resync after success to ensure chart overlay and state agree
    try {
      const mod = await import('./drawings/manual_trendline_controller.js');
      await mod.refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
    } catch {}
    return;
  }

  if (action === 'clear-all-lines') {
    if (!confirm('清空所有画线?')) return;
    console.log('[cond_panel] clear-all-lines clicked');
    try {
      await fetchJson(
        `/api/drawings/clear?symbol=${encodeURIComponent(marketState.currentSymbol)}&timeframe=${encodeURIComponent(marketState.currentInterval)}`,
        { method: 'POST', timeout: 8000 }
      );
      const { setManualDrawings } = await import('../state/drawings.js');
      setManualDrawings([]);
      currentLine = null;
      currentAnalyze = null;
      render();
    } catch (err) {
      console.error('[cond_panel] clear failed', err);
      alert(`清空失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
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
    ${renderPendingSection(true)}
  `;
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
        return `<div class="cond-row" style="font-size:10px">
          <b style="color:${dirColor}">${esc(dir.toUpperCase())}</b> ${esc(p.symbol || '')}
          · qty <b>${esc(String(p.total ?? '?'))}</b>
          · entry ${esc(String(p.averageOpenPrice ?? '?'))}
          · mark ${esc(String(p.markPrice ?? '?'))}
        </div>`;
      }).join('')
    : '<p class="muted" style="font-size:10px;margin:4px 0">无持仓</p>';

  const orderRows = pending.length
    ? pending.map((o) => `<div class="cond-row" style="font-size:10px">
        ${esc(o.symbol || '')} · ${esc(o.side || '')} · ${esc(o.size || '')}
        · @ ${esc(String(o.price ?? '?'))}
        · <span class="cond-status">${esc(o.status || '')}</span>
        <button class="cond-btn-sm cond-btn-warn" data-action="bitget-cancel" data-symbol="${esc(o.symbol)}" data-oid="${esc(o.orderId)}">撤</button>
      </div>`).join('')
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
  if (!lines.length) {
    return `
      <div class="cond-section">
        <div class="cond-header">我的手画线 <span class="cond-count">0</span></div>
        <p class="muted">按工具栏画一条线开始。<br>T 画线 · Esc 取消</p>
      </div>
    `;
  }

  const rows = lines.slice().reverse().map((l) => {
    const isSel = l.manual_line_id === drawingsState.selectedLineId;
    const sideZh = '线';
    const sideColor = '#fbbf24';
    return `
      <div class="cond-line-row ${isSel ? 'is-selected' : ''}"
           data-hover-line="${esc(l.manual_line_id)}">
        <button class="cond-line-body" data-action="select-line" data-line-id="${esc(l.manual_line_id)}">
          <span class="cond-line-dot" style="background:${sideColor}"></span>
          <span class="cond-line-side">${esc(sideZh)}</span>
          <span class="cond-line-price">${esc(fmtPrice(l.price_start))} → ${esc(fmtPrice(l.price_end))}</span>
          <span class="cond-line-time">${esc(fmtDateShort(l.created_at))}</span>
        </button>
        <button class="cond-line-del" data-action="delete-line" data-line-id="${esc(l.manual_line_id)}" title="删除这条线">×</button>
      </div>
    `;
  }).join('');

  return `
    <div class="cond-section">
      <div class="cond-header">
        我的手画线 <span class="cond-count">${lines.length}</span>
        <span class="cond-spacer"></span>
        <button class="cond-btn-sm cond-btn-danger" data-action="clear-all-lines" title="清空所有画线">清空</button>
      </div>
      <div class="cond-line-list">${rows}</div>
      <p class="muted" style="font-size:10px;margin-top:4px">悬停高亮 · 点击选中 · × 删除</p>
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
      lines.push(`<span style="color:#94a3b8">推荐: 容差 <b style="color:#fbbf24">${r.tolerance_pct}%</b> · 止损 <b style="color:#fbbf24">${r.stop_pct}%</b> · RR <b style="color:#fbbf24">${r.rr_target}</b></span>`);
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
          <button class="cond-btn-sm cond-btn-danger" data-action="delete-cond" data-cid="${esc(c.conditional_id)}" title="${c.status === 'triggered' ? '⚠ 仅删本地记录,不撤Bitget,请先点撤Bitget' : '删除本地记录'}">×</button>
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

  // refPrice projection (clamped)
  let refPrice = line.price_end || line.price_start || 0;
  try {
    const ts = Math.floor(Date.now() / 1000);
    const span = line.t_end - line.t_start;
    if (span > 0) {
      const slope = (line.price_end - line.price_start) / span;
      const projected = line.price_start + slope * (ts - line.t_start);
      if (ts < line.t_start) refPrice = line.price_start;
      else if (ts > line.t_end) refPrice = line.price_end;
      else refPrice = projected;
    }
  } catch {}

  let balance = null;
  let liveMark = null;
  let recs = null;
  let analyzeStats = null;
  let acctErr = null;
  let markErr = null;
  try {
    const [acct, mark, analyzeResp] = await Promise.all([
      fetchJson('/api/live-execution/account?mode=live', { timeout: 15000, noCache: true })
        .catch((e) => { acctErr = e; return null; }),
      fetchMarkPriceOnce(line.symbol).catch((e) => { markErr = e; return null; }),
      condSvc.analyzeDrawing(line.manual_line_id).catch(() => null),
    ]);
    if (cancelled) return;   // user pressed Esc / clicked outside
    balance = Number(acct?.usdt_available ?? acct?.total_equity ?? 0);
    liveMark = mark;
    if (analyzeResp?.ok) {
      recs = analyzeResp.recommendations || null;
      analyzeStats = analyzeResp.stats || null;
      analyzeStats = analyzeStats || {};
      analyzeStats._volatility = analyzeResp.volatility || null;
      analyzeStats._coherence = analyzeResp.coherence || null;
      analyzeStats._lineQuality = analyzeResp.line_quality || null;
      analyzeStats._distance = analyzeResp.distance || null;
    }
  } catch (e) { console.error('[cond_panel] modal init fetch err', e); }
  console.log('[cond_panel] modal data', { balance, liveMark, refPrice, acctErr, markErr });

  const defTolerance = recs?.tolerance_pct ?? 0.1;
  const defStop = recs?.stop_pct ?? 0.3;
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

      ${analyzeStats && analyzeStats.sample_size > 0 ? `
      <div class="cond-modal-stats" style="margin:8px 0;padding:8px 10px;background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.3);border-radius:4px;font-size:10px;line-height:1.6">
        <b style="color:#fbbf24">📊 历史 ${analyzeStats.sample_size} 条相似形态</b><br>
        反弹 <b style="color:#00e676">${Math.round((analyzeStats.p_bounce||0)*100)}%</b> ·
        破位 <b style="color:#ff5252">${Math.round((analyzeStats.p_break||0)*100)}%</b> ·
        EV <b style="color:${analyzeStats.expected_value>=0?'#00e676':'#ff5252'}">${(analyzeStats.expected_value||0).toFixed(2)}R</b><br>
        平均反弹 ${(analyzeStats.avg_return_atr||0).toFixed(2)} ATR ·
        平均回撤 ${(analyzeStats.avg_drawdown_atr||0).toFixed(2)} ATR<br>
        可信度 <b style="color:${analyzeStats.trustworthiness==='high'?'#00e676':analyzeStats.trustworthiness==='medium'?'#fbbf24':'#888'}">${esc(analyzeStats.trustworthiness||'low')}</b> ·
        ${esc(analyzeStats.overfit_flag||'')}
        ${recs ? `<br><span style="color:#888">推荐: 容差 ${recs.tolerance_pct}% · 止损 ${recs.stop_pct}% · RR ${recs.rr_target}</span>` : ''}
      </div>
      ` : ''}

      ${recs ? `
      <div class="cond-modal-smart" style="margin:6px 0;padding:6px 10px;background:rgba(96,165,250,0.06);border:1px solid rgba(96,165,250,0.25);border-radius:4px;font-size:10px;line-height:1.5;color:#94a3b8">
        <b style="color:#60a5fa">🧠 智能调参</b><br>
        ${analyzeStats?._volatility ? `ATR <b style="color:${analyzeStats._volatility.regime==='wide'?'#ff5252':analyzeStats._volatility.regime==='tight'?'#60a5fa':'#fbbf24'}">${(analyzeStats._volatility.atr_pct||0).toFixed(2)}%</b> · ${esc(analyzeStats._volatility.regime||'')}` : ''}
        ${analyzeStats?._volatility?.bb_state && analyzeStats._volatility.bb_state !== 'normal' ? ` · BB <b style="color:${analyzeStats._volatility.bb_state==='squeeze'?'#fbbf24':'#00e676'}">${analyzeStats._volatility.bb_state==='squeeze'?'挤压':'扩张'}</b>` : ''}
        ${analyzeStats?._coherence?.score > 0 ? `<br>📐 高TF对齐 <b style="color:#00e676">${analyzeStats._coherence.score}</b>: ${(analyzeStats._coherence.matches||[]).map(m=>`${esc(m.tf)}@${(m.price||0).toFixed(3)}`).join(', ')}` : ''}
        ${analyzeStats?._lineQuality?.touch_count > 2 ? `<br>👆 触点 <b style="color:#00e676">${analyzeStats._lineQuality.touch_count}</b> (强度 ${(analyzeStats._lineQuality.strength*100).toFixed(0)}%)` : ''}
        ${analyzeStats?._distance?.zone && analyzeStats._distance.zone !== 'medium' ? `<br>📏 距线 <b style="color:${analyzeStats._distance.zone==='near'?'#fbbf24':'#888'}">${analyzeStats._distance.zone}</b> (${(analyzeStats._distance.gap_pct||0).toFixed(2)}%)` : ''}
      </div>
      ` : ''}

      <div class="cond-modal-row">
        <label>容差 (%)</label>
        <input type="number" name="tolerance_pct" value="${defTolerance}" step="0.05" min="0.01" />
        <small style="color:#888;margin-left:4px;font-size:10px">价格摸到线 ± 容差时触发</small>
      </div>
      <div class="cond-modal-row">
        <label>止损 (%)</label>
        <input type="number" name="stop_pct" value="${defStop}" step="0.05" min="0.05" />
        <small style="color:#888;margin-left:4px;font-size:10px">穿过线该% → 离场</small>
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
    const stopPct = fieldVal('stop_pct', 0.3);
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
      const resp = await condSvc.placeLineOrder({
        manual_line_id: line.manual_line_id,
        direction,
        kind: chosenKind,
        tolerance_pct: tolerancePct,
        stop_offset_pct: stopPct,
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

      // Success — pending conditional now armed in watcher
      const cid = resp.conditional?.conditional_id || '?';
      showStatus(
        `<b style="color:#00e676">✓ 条件单已 armed</b><br>` +
        `id: <code>${cid.slice(0, 18)}</code><br>` +
        `${esc(resp.message || '')}<br>` +
        `<small>watcher 每秒检查一次。价格摸到线 ± ${tolerancePct}% 时立刻打到 Bitget。<br>2 秒后关闭。</small>`,
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
