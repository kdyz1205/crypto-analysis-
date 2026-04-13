// frontend/js/workbench/conditional_panel.js
//
// A self-contained UI panel for manual-line → conditional-order workflow.
// Injected into the market view, independent of the existing
// manual_trendline_controller. Reacts to:
//   - drawings.selected event  → fetch analyze, show stats card
//   - user clicks "Setup Order" → show modal
//   - user confirms in modal    → POST /api/conditionals
//   - polls every 10s           → refresh pending list

import { subscribe } from '../util/events.js';
import { drawingsState } from '../state/drawings.js';
import * as condSvc from '../services/conditionals.js';

let panel = null;
let currentAnalyze = null;      // stats payload for currently selected line
let currentLine = null;          // ManualTrendline dict
let pendingRefreshTimer = null;
let knownConditionals = [];

// Simple HTML escape
function esc(v) {
  if (v == null) return '';
  return String(v)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(1)}%`;
}

function fmtR(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(2) + 'R';
}

function fmtTime(ts) {
  if (!ts) return '—';
  try {
    return new Date(ts * 1000).toLocaleTimeString();
  } catch {
    return '—';
  }
}

export function initConditionalPanel(container) {
  if (panel) return panel;
  panel = document.createElement('div');
  panel.className = 'cond-panel';
  panel.id = 'v2-cond-panel';
  panel.innerHTML = `
    <div class="cond-section cond-stats-slot">
      <div class="cond-header">历史概率 / 手画线分析</div>
      <div class="cond-stats-body">
        <p class="muted">选择或绘制一条支撑/阻力线以查看历史概率</p>
      </div>
    </div>
    <div class="cond-section cond-pending-slot">
      <div class="cond-header">条件单 <span class="cond-count"></span></div>
      <div class="cond-pending-body">
        <p class="muted">加载中...</p>
      </div>
    </div>
  `;
  container.appendChild(panel);

  // Single delegation listener for all buttons inside the panel
  panel.addEventListener('click', onPanelClick);

  // React when user selects a drawn line
  subscribe('drawings.selected', () => onDrawingSelected());

  // Also react when drawings list updates (user may draw a new line)
  subscribe('drawings.updated', () => {
    // Auto-select the newly created line if there was none before
    if (!currentLine && drawingsState.lines.length) {
      onDrawingSelected();
    }
  });

  // Pending orders refresh loop
  void refreshPending();
  pendingRefreshTimer = setInterval(() => void refreshPending(), 10000);

  return panel;
}

export function destroyConditionalPanel() {
  if (pendingRefreshTimer) {
    clearInterval(pendingRefreshTimer);
    pendingRefreshTimer = null;
  }
  if (panel && panel.parentNode) {
    panel.parentNode.removeChild(panel);
  }
  panel = null;
  currentAnalyze = null;
  currentLine = null;
  knownConditionals = [];
}

// ──────────────────────────────────────────────────────────────
// Event handlers
// ──────────────────────────────────────────────────────────────
async function onDrawingSelected() {
  const selId = drawingsState.selectedLineId;
  const line = selId
    ? drawingsState.lines.find((l) => l.manual_line_id === selId)
    : drawingsState.lines[drawingsState.lines.length - 1];
  if (!line) return;
  currentLine = line;
  renderStatsSlot({ loading: true });
  try {
    const resp = await condSvc.analyzeDrawing(line.manual_line_id);
    currentAnalyze = resp;
    renderStatsSlot({ loading: false });
  } catch (err) {
    currentAnalyze = { ok: false, error: err?.message || 'analyze failed' };
    renderStatsSlot({ loading: false });
  }
}

function onPanelClick(e) {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;
  if (action === 'setup-order') {
    openOrderModal();
  } else if (action === 'cancel-cond') {
    const cid = btn.dataset.cid;
    if (cid && confirm('Cancel this conditional?')) {
      void condSvc.cancelConditional(cid).then(() => refreshPending());
    }
  } else if (action === 'delete-cond') {
    const cid = btn.dataset.cid;
    if (cid && confirm('Delete this conditional? (cannot undo)')) {
      void condSvc.deleteConditional(cid).then(() => refreshPending());
    }
  } else if (action === 'expand-cond') {
    const cid = btn.dataset.cid;
    const el = panel.querySelector(`[data-events-for="${cid}"]`);
    if (el) el.classList.toggle('open');
  }
}

// ──────────────────────────────────────────────────────────────
// Stats slot render
// ──────────────────────────────────────────────────────────────
function renderStatsSlot({ loading }) {
  const body = panel?.querySelector('.cond-stats-body');
  if (!body) return;

  if (loading) {
    body.innerHTML = '<p class="muted">分析中...</p>';
    return;
  }

  if (!currentLine) {
    body.innerHTML = '<p class="muted">选择或绘制一条支撑/阻力线以查看历史概率</p>';
    return;
  }

  if (!currentAnalyze?.ok) {
    const err = currentAnalyze?.error || '未知错误';
    body.innerHTML = `
      <div class="cond-line-info">
        <strong>${esc(currentLine.side === 'support' ? '支撑' : '阻力')}</strong>
        @ ${esc(currentLine.symbol)} ${esc(currentLine.timeframe)}
      </div>
      <p class="muted">分析失败: ${esc(err)}</p>
      <button data-action="setup-order" class="cond-btn cond-btn-primary">
        仍然设置条件单 →
      </button>
    `;
    return;
  }

  const stats = currentAnalyze.stats || {};
  const n = stats.sample_size || 0;
  const pBounce = stats.p_bounce;
  const pBreak = stats.p_break;
  const pFake = stats.p_fake_break;
  const ev = stats.expected_value;
  const conf = stats.confidence;
  const trust = stats.trustworthiness || 'none';
  const trustColor = { high: '#00e676', medium: '#fbbf24', low: '#ff8a80', none: '#888' }[trust] || '#888';

  body.innerHTML = `
    <div class="cond-line-info">
      <strong>${esc(currentLine.side === 'support' ? '支撑' : '阻力')}</strong>
      @ ${esc(currentLine.symbol)} ${esc(currentLine.timeframe)}
      · ${esc((currentLine.price_start || 0).toFixed(2))} → ${esc((currentLine.price_end || 0).toFixed(2))}
    </div>
    <div class="cond-stats-grid">
      <div><span>样本</span><b>${n}</b></div>
      <div><span>反弹概率</span><b style="color:${pBounce >= 0.5 ? '#00e676' : '#888'}">${fmtPct(pBounce)}</b></div>
      <div><span>破位概率</span><b>${fmtPct(pBreak)}</b></div>
      <div><span>假突破</span><b>${fmtPct(pFake)}</b></div>
      <div><span>期望值</span><b style="color:${ev > 0 ? '#00e676' : '#ff8a80'}">${fmtR(ev)}</b></div>
      <div><span>可信度</span><b style="color:${trustColor}">${esc(trust)} ${conf != null ? fmtPct(conf) : ''}</b></div>
    </div>
    <div class="cond-actions">
      <button data-action="setup-order" class="cond-btn cond-btn-primary">
        设置条件单 →
      </button>
    </div>
  `;
}

// ──────────────────────────────────────────────────────────────
// Order modal
// ──────────────────────────────────────────────────────────────
function openOrderModal() {
  if (!currentLine) return;
  const dir = currentLine.side === 'support' ? 'long' : 'short';

  const modal = document.createElement('div');
  modal.className = 'cond-modal-bg';
  modal.innerHTML = `
    <div class="cond-modal">
      <h3>配置条件单 · ${esc(currentLine.symbol)} ${esc(currentLine.timeframe)}</h3>
      <div class="cond-modal-row">
        <label>方向</label>
        <select id="cm-direction">
          <option value="long" ${dir === 'long' ? 'selected' : ''}>long (从 support 反弹)</option>
          <option value="short" ${dir === 'short' ? 'selected' : ''}>short (从 resistance 拒绝)</option>
        </select>
      </div>
      <div class="cond-modal-row">
        <label>进场偏移 (ATR)</label>
        <input type="number" id="cm-offset" value="0" step="0.05" min="-2" max="2" />
      </div>
      <div class="cond-modal-row">
        <label>Stop (ATR)</label>
        <input type="number" id="cm-stop" value="0.3" step="0.05" min="0.1" max="2" />
      </div>
      <div class="cond-modal-row">
        <label>RR 目标</label>
        <input type="number" id="cm-rr" value="2.0" step="0.5" min="0.5" max="10" />
      </div>
      <div class="cond-modal-row">
        <label>仓位 (USDT)</label>
        <input type="number" id="cm-notional" value="200" step="50" min="10" />
      </div>
      <div class="cond-modal-row">
        <label>触发容差 (ATR)</label>
        <input type="number" id="cm-tol" value="0.2" step="0.05" min="0.05" max="1" />
      </div>
      <div class="cond-modal-row">
        <label>超时 (小时)</label>
        <input type="number" id="cm-expiry" value="48" step="1" min="1" max="720" />
      </div>
      <div class="cond-modal-row">
        <label>挂到交易所</label>
        <select id="cm-submit">
          <option value="false" selected>不,只警报 (推荐)</option>
          <option value="true">是,实盘下单</option>
        </select>
      </div>
      <div class="cond-modal-row">
        <label>模式</label>
        <select id="cm-mode">
          <option value="paper" selected>paper</option>
          <option value="live">live (真钱)</option>
        </select>
      </div>
      <div class="cond-modal-actions">
        <button class="cond-btn" id="cm-cancel">取消</button>
        <button class="cond-btn cond-btn-primary" id="cm-confirm">确认挂单</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);

  const q = (s) => modal.querySelector(s);
  q('#cm-cancel').onclick = () => modal.remove();
  q('#cm-confirm').onclick = async () => {
    const btn = q('#cm-confirm');
    btn.disabled = true;
    btn.textContent = '创建中...';
    const payload = {
      manual_line_id: currentLine.manual_line_id,
      trigger: {
        tolerance_atr: parseFloat(q('#cm-tol').value) || 0.2,
        poll_seconds: 60,  // server auto-derives if 60 is default
        max_age_seconds: Math.round(parseFloat(q('#cm-expiry').value || 48) * 3600),
        max_distance_atr: 5.0,
        break_threshold_atr: 0.5,
      },
      order: {
        direction: q('#cm-direction').value,
        entry_offset_atr: parseFloat(q('#cm-offset').value) || 0,
        stop_atr: parseFloat(q('#cm-stop').value) || 0.3,
        rr_target: parseFloat(q('#cm-rr').value) || 2.0,
        notional_usd: parseFloat(q('#cm-notional').value) || 200,
        submit_to_exchange: q('#cm-submit').value === 'true',
        exchange_mode: q('#cm-mode').value,
      },
      pattern_stats: currentAnalyze?.stats || {},
    };
    try {
      const resp = await condSvc.createConditional(payload);
      if (!resp?.ok) throw new Error(resp?.error || 'create failed');
      modal.remove();
      await refreshPending();
    } catch (err) {
      btn.disabled = false;
      btn.textContent = '确认挂单';
      alert(`创建失败: ${err?.message || err}`);
    }
  };
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.remove();
  });
}

// ──────────────────────────────────────────────────────────────
// Pending panel
// ──────────────────────────────────────────────────────────────
async function refreshPending() {
  if (!panel) return;
  try {
    const resp = await condSvc.listConditionals('all');
    knownConditionals = resp?.conditionals || [];
    renderPendingSlot();
  } catch (err) {
    // Silent — log only
    console.warn('[cond] list failed:', err);
  }
}

function renderPendingSlot() {
  const body = panel?.querySelector('.cond-pending-body');
  const countEl = panel?.querySelector('.cond-count');
  if (!body) return;

  const pending = knownConditionals.filter((c) => c.status === 'pending');
  const triggered = knownConditionals.filter((c) => c.status === 'triggered');
  const cancelled = knownConditionals.filter((c) => c.status === 'cancelled');

  if (countEl) {
    countEl.textContent = `${pending.length} 活跃 / ${triggered.length} 触发 / ${cancelled.length} 已取消`;
  }

  if (!knownConditionals.length) {
    body.innerHTML = '<p class="muted">暂无条件单</p>';
    return;
  }

  const row = (c) => {
    const statusColor = {
      pending: '#fbbf24',
      triggered: '#00e676',
      cancelled: '#888',
      failed: '#ff1744',
    }[c.status] || '#888';
    const distanceText =
      c.last_distance_atr != null ? `${c.last_distance_atr.toFixed(2)} ATR away` : '未探测';
    const marketText =
      c.last_market_price != null ? c.last_market_price.toFixed(4) : '—';
    const lineText =
      c.last_line_price != null ? c.last_line_price.toFixed(4) : '—';

    const eventsHtml = (c.events || [])
      .slice(-10)
      .reverse()
      .map(
        (e) =>
          `<div class="cond-event">
            <span class="cond-event-time">${fmtTime(e.ts)}</span>
            <span class="cond-event-kind" data-kind="${esc(e.kind)}">${esc(e.kind)}</span>
            <span class="cond-event-msg">${esc(e.message || '')}</span>
          </div>`
      )
      .join('');

    return `
      <div class="cond-row" data-cid="${esc(c.conditional_id)}">
        <div class="cond-row-header">
          <span class="cond-dot" style="background:${statusColor}"></span>
          <strong>${esc(c.symbol)} ${esc(c.timeframe)}</strong>
          <span class="cond-side">${esc(c.side)}</span>
          <span class="cond-dir">${esc(c.order.direction)}</span>
          <span class="cond-status" style="color:${statusColor}">${esc(c.status)}</span>
          <span class="cond-spacer"></span>
          <button class="cond-btn-sm" data-action="expand-cond" data-cid="${esc(c.conditional_id)}">events</button>
          ${c.status === 'pending'
            ? `<button class="cond-btn-sm cond-btn-warn" data-action="cancel-cond" data-cid="${esc(c.conditional_id)}">取消</button>`
            : ''}
          <button class="cond-btn-sm cond-btn-danger" data-action="delete-cond" data-cid="${esc(c.conditional_id)}">×</button>
        </div>
        <div class="cond-row-stats">
          market <b>${marketText}</b>
          · line <b>${lineText}</b>
          · ${esc(distanceText)}
          · size <b>${esc(String(c.order.notional_usd ?? '—'))}</b> USDT
          · RR <b>${esc(String(c.order.rr_target ?? '—'))}</b>
          · ${c.order.submit_to_exchange ? '<span style="color:#ff1744">自动下单</span>' : '仅警报'}
        </div>
        <div class="cond-events-panel" data-events-for="${esc(c.conditional_id)}">
          ${eventsHtml || '<div class="cond-event muted">无事件</div>'}
        </div>
      </div>
    `;
  };

  body.innerHTML = knownConditionals.map(row).join('');
}
