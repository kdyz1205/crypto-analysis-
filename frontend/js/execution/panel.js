// frontend/js/execution/panel.js — minimal Execution Center with 4 sub-tabs

import { $, $$, setHtml, on, show, hide } from '../util/dom.js';
import { agentState, setActiveSubTab, setPanelOpen, setLastStatus } from '../state/agent.js';
import { subscribe, publish } from '../util/events.js';
import * as agentSvc from '../services/agent.js';
import * as riskSvc from '../services/risk.js';
import * as execSvc from '../services/execution.js';
import * as opsSvc from '../services/ops.js';
import { formatUsd, formatPct, pnlColorClass } from '../util/format.js';

let pollTimer = null;

export function initExecutionPanel() {
  buildShell();
  wireEvents();
}

function buildShell() {
  const root = $('#v2-execution-panel');
  if (!root) return;
  setHtml(root, `
    <div class="exec-header">
      <h3>Execution Center</h3>
      <div class="exec-mode-badge" id="v2-exec-mode">—</div>
      <button class="exec-close" id="v2-exec-close">×</button>
    </div>
    <nav class="exec-tabs">
      <button class="exec-tab active" data-tab="overview">Overview</button>
      <button class="exec-tab" data-tab="execution">Execution</button>
      <button class="exec-tab" data-tab="risk">Risk</button>
      <button class="exec-tab" data-tab="ops">Ops</button>
    </nav>
    <div class="exec-body">
      <div class="exec-subtab" data-subtab="overview"></div>
      <div class="exec-subtab hidden" data-subtab="execution"></div>
      <div class="exec-subtab hidden" data-subtab="risk"></div>
      <div class="exec-subtab hidden" data-subtab="ops"></div>
    </div>
  `);
}

function wireEvents() {
  const root = $('#v2-execution-panel');
  if (!root) return;

  on('#v2-exec-close', 'click', () => closePanel());

  $$('.exec-tab', root).forEach((btn) => {
    on(btn, 'click', () => switchSubTab(btn.dataset.tab));
  });

  subscribe('agent.subtab.changed', (tab) => {
    $$('.exec-tab', root).forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
    $$('.exec-subtab', root).forEach((s) => s.classList.toggle('hidden', s.dataset.subtab !== tab));
    renderActive();
  });
}

function switchSubTab(tab) {
  setActiveSubTab(tab);
}

export function openPanel() {
  setPanelOpen(true);
  show('#v2-execution-panel');
  renderActive();
  startPolling();
}

export function closePanel() {
  setPanelOpen(false);
  hide('#v2-execution-panel');
  stopPolling();
}

export function togglePanel() {
  agentState.panelOpen ? closePanel() : openPanel();
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(async () => {
    try {
      const status = await agentSvc.getStatus();
      setLastStatus(status);
      renderActive();
    } catch (err) { console.warn('[exec] poll failed:', err); }
  }, 5000);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

async function renderActive() {
  const tab = agentState.activeSubTab;
  switch (tab) {
    case 'overview': return renderOverview();
    case 'execution': return renderExecution();
    case 'risk': return renderRisk();
    case 'ops': return renderOps();
  }
}

async function renderOverview() {
  const container = $('[data-subtab="overview"]');
  if (!container) return;
  const s = agentState.lastStatus || await agentSvc.getStatus();
  setLastStatus(s);

  const mode = s.mode?.toUpperCase() || '—';
  $('#v2-exec-mode') && ($('#v2-exec-mode').textContent = mode);

  const positions = Object.values(s.positions || {});
  const trades = s.recent_trades || [];

  setHtml(container, `
    <div class="exec-hero">
      <div class="hero-stat">
        <div class="hero-label">Total Equity</div>
        <div class="hero-value">${formatUsd(s.equity)}</div>
      </div>
      <div class="hero-stat">
        <div class="hero-label">P&L</div>
        <div class="hero-value ${pnlColorClass(s.total_pnl_usd)}">${formatUsd(s.total_pnl_usd)}</div>
      </div>
      <div class="hero-stat">
        <div class="hero-label">Today</div>
        <div class="hero-value ${pnlColorClass(s.daily_pnl)}">${formatUsd(s.daily_pnl)}</div>
      </div>
      <div class="hero-stat">
        <div class="hero-label">Win Rate</div>
        <div class="hero-value">${(s.win_rate ?? 0).toFixed(1)}%</div>
      </div>
    </div>
    <div class="exec-stats-grid">
      <div class="stat"><div class="stat-label">Status</div><div class="stat-value">${s.running ? 'RUNNING' : 'STOPPED'}</div></div>
      <div class="stat"><div class="stat-label">Cash</div><div class="stat-value">${formatUsd(s.cash)}</div></div>
      <div class="stat"><div class="stat-label">Trades</div><div class="stat-value">${s.total_trades ?? 0}</div></div>
      <div class="stat"><div class="stat-label">Regime</div><div class="stat-value">${s.harness?.market_regime || '—'}</div></div>
      <div class="stat"><div class="stat-label">Gen</div><div class="stat-value">${s.generation ?? 0}</div></div>
      <div class="stat"><div class="stat-label">Phase</div><div class="stat-value">${s.cycle_phase || '—'}</div></div>
    </div>
    <h4>Open positions (${positions.length})</h4>
    <div class="exec-positions">
      ${positions.length === 0 ? '<p class="muted">No open positions</p>' : positions.map(p => `
        <div class="pos-row">
          <span class="pos-sym">${p.symbol}</span>
          <span class="pos-side ${p.side}">${p.side?.toUpperCase()}</span>
          <span>${formatUsd(p.size_usd || p.size)}</span>
          <span class="${pnlColorClass(p.unrealized_pnl_pct)}">${formatPct(p.unrealized_pnl_pct)}</span>
        </div>
      `).join('')}
    </div>
    <h4>Recent trades</h4>
    <div class="exec-trades">
      ${trades.length === 0 ? '<p class="muted">No trades yet</p>' : trades.slice(-5).reverse().map(t => `
        <div class="trade-row">
          <span>${t.symbol}</span>
          <span class="${t.side}">${t.side?.toUpperCase()}</span>
          <span class="${pnlColorClass(t.pnl_pct)}">${formatPct(t.pnl_pct)}</span>
          <span class="muted">${t.reason || ''}</span>
        </div>
      `).join('')}
    </div>
  `);
}

async function renderExecution() {
  const container = $('[data-subtab="execution"]');
  if (!container) return;
  const s = agentState.lastStatus || await agentSvc.getStatus();
  const signals = s.last_signals || {};
  const entries = Object.entries(signals);

  setHtml(container, `
    <div class="exec-controls">
      <button class="btn btn-primary" id="v2-start-btn">Start</button>
      <button class="btn" id="v2-stop-btn">Stop</button>
      <button class="btn" id="v2-revive-btn">Revive</button>
      <button class="btn" id="v2-scan-btn">Scan Now</button>
      <button class="btn ${s.mode === 'paper' ? 'active' : ''}" id="v2-paper-btn">Paper</button>
      <button class="btn ${s.mode === 'live' ? 'active' : ''}" id="v2-live-btn">Live</button>
    </div>
    <h4>Current signals</h4>
    <div class="signals-list">
      ${entries.length === 0 ? '<p class="muted">No signals yet</p>' : entries.map(([sym, sig]) => `
        <div class="signal-row ${sig.blocked ? 'blocked' : ''}">
          <span class="signal-sym">${sym}</span>
          <span class="signal-action ${sig.action}">${(sig.action || '—').toUpperCase()}</span>
          <span class="signal-conf">${sig.confidence ? Math.round(sig.confidence * 100) + '%' : '—'}</span>
          <div class="signal-reason">${sig.reason || ''}</div>
          ${sig.blocked ? `<div class="signal-block">Blocked: ${(sig.block_reasons || []).join('; ')}</div>` : ''}
        </div>
      `).join('')}
    </div>
  `);

  on('#v2-start-btn', 'click', async () => { await agentSvc.start(); renderExecution(); });
  on('#v2-stop-btn', 'click', async () => { await agentSvc.stop(); renderExecution(); });
  on('#v2-revive-btn', 'click', async () => { await agentSvc.revive(); renderExecution(); });
  on('#v2-scan-btn', 'click', async () => { await agentSvc.getSignals(); renderExecution(); });
  on('#v2-paper-btn', 'click', async () => { await agentSvc.setConfig({ mode: 'paper' }); renderExecution(); });
  on('#v2-live-btn', 'click', async () => { await agentSvc.setConfig({ mode: 'live' }); renderExecution(); });
}

async function renderRisk() {
  const container = $('[data-subtab="risk"]');
  if (!container) return;
  const s = agentState.lastStatus || await agentSvc.getStatus();
  const r = s.risk_limits || {};

  // Calculate usage meters
  const dailyLossPct = Math.max(0, -s.daily_pnl / s.equity * 100);
  const ddPct = Math.max(0, (s.peak_equity - s.equity) / s.peak_equity * 100);
  const positionsUsed = Object.keys(s.positions || {}).length;

  const meter = (label, cur, max, unit = '%') => {
    const pct = max > 0 ? Math.min(100, (cur / max) * 100) : 0;
    const cls = pct < 50 ? 'green' : pct < 80 ? 'yellow' : 'red';
    return `
      <div class="meter ${cls}">
        <div class="meter-label">${label}</div>
        <div class="meter-bar"><div class="meter-fill" style="width:${pct}%"></div></div>
        <div class="meter-value">${cur.toFixed(2)}${unit} / ${Number(max).toFixed(2)}${unit}</div>
      </div>
    `;
  };

  setHtml(container, `
    <h4>Live risk meters</h4>
    <div class="risk-meters">
      ${meter('Daily Loss', dailyLossPct, r.max_daily_loss_pct)}
      ${meter('Drawdown', ddPct, r.max_drawdown_pct)}
      ${meter('Positions', positionsUsed, r.max_positions, '')}
    </div>
    <h4>Risk limits config</h4>
    <form class="risk-form" id="v2-risk-form">
      <label>Max Position %<input type="number" name="max_position_pct" value="${r.max_position_pct ?? 5}" step="0.5"></label>
      <label>Max Exposure %<input type="number" name="max_total_exposure_pct" value="${r.max_total_exposure_pct ?? 15}" step="1"></label>
      <label>Max Daily Loss %<input type="number" name="max_daily_loss_pct" value="${r.max_daily_loss_pct ?? 2}" step="0.1"></label>
      <label>Max Drawdown %<input type="number" name="max_drawdown_pct" value="${r.max_drawdown_pct ?? 5}" step="1"></label>
      <label>Max Positions<input type="number" name="max_positions" value="${r.max_positions ?? 3}" step="1"></label>
      <label>Cooldown (s)<input type="number" name="cooldown_seconds" value="${r.cooldown_seconds ?? 3600}" step="60"></label>
      <button type="submit" class="btn btn-primary">Save</button>
    </form>
  `);

  on('#v2-risk-form', 'submit', async (e) => {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target));
    Object.keys(data).forEach(k => data[k] = Number(data[k]));
    await riskSvc.setRiskLimits(data);
    renderRisk();
  });
}

async function renderOps() {
  const container = $('[data-subtab="ops"]');
  if (!container) return;

  let healer = null, okx = null;
  try { healer = await opsSvc.getHealerStatus(); } catch {}
  try { okx = await execSvc.getOkxStatus(); } catch {}

  setHtml(container, `
    <h4>OKX Connection</h4>
    <div class="ops-section">
      <p>Has keys: <strong>${okx?.has_keys ? 'YES' : 'NO'}</strong></p>
      <p>Mode: <strong>${okx?.mode || '—'}</strong></p>
      ${okx?.balance ? `<pre>${JSON.stringify(okx.balance, null, 2)}</pre>` : ''}
    </div>
    <h4>Self-Healer</h4>
    <div class="ops-section">
      <p>Running: <strong>${healer?.running ? 'YES' : 'NO'}</strong></p>
      <p>Fix count: <strong>${healer?.fix_count ?? 0}</strong></p>
      <p>AI: <strong>${healer?.has_ai ? 'enabled' : 'disabled'}</strong></p>
      <button class="btn" id="v2-healer-trigger">Trigger heal</button>
    </div>
  `);

  on('#v2-healer-trigger', 'click', async () => {
    await opsSvc.triggerHealer();
    renderOps();
  });
}
