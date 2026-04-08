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
    paintLoading();
    renderActive();
  });
}

function switchSubTab(tab) {
  setActiveSubTab(tab);
}

export function openPanel() {
  setPanelOpen(true);
  show('#v2-execution-panel');
  // Paint loading state synchronously so the panel is never blank
  paintLoading();
  renderActive().catch((err) => {
    console.error('[exec] render failed:', err);
    const container = $(`[data-subtab="${agentState.activeSubTab}"]`);
    if (container) setHtml(container, `<div class="muted">Error loading: ${err.message || err}</div>`);
  });
  startPolling();
}

function paintLoading() {
  const container = $(`[data-subtab="${agentState.activeSubTab}"]`);
  if (container && !container.innerHTML.trim()) {
    setHtml(container, '<div class="muted" style="padding:40px;text-align:center">Loading...</div>');
  }
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
  try {
    switch (tab) {
      case 'overview': await renderOverview(); break;
      case 'execution': await renderExecution(); break;
      case 'risk': await renderRisk(); break;
      case 'ops': await renderOps(); break;
    }
  } catch (err) {
    console.error(`[exec] render ${tab} failed:`, err);
    const container = $(`[data-subtab="${tab}"]`);
    if (container) {
      setHtml(container, `<div class="muted" style="padding:20px">
        <strong>Render error:</strong> ${err.message || err}<br>
        <button class="btn" onclick="location.reload()">Reload page</button>
      </div>`);
    }
  }
}

async function renderOverview() {
  const container = $('[data-subtab="overview"]');
  if (!container) {
    console.warn('[exec] overview container not found');
    return;
  }
  console.log('[exec] renderOverview starting...');
  const s = await agentSvc.getStatus();
  setLastStatus(s);
  console.log('[exec] renderOverview got status:', s);

  const mode = s.mode?.toUpperCase() || '—';
  const modeBadge = $('#v2-exec-mode');
  if (modeBadge) modeBadge.textContent = mode;

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
  const s = await agentSvc.getStatus();
  const r = s.risk_limits || {};

  // Backend stores limits as decimals (0.05 for 5%). Convert to percent.
  const maxPosPct = (r.max_position_pct ?? 0.05) * 100;
  const maxExpPct = (r.max_total_exposure_pct ?? 0.15) * 100;
  const maxDailyLossPct = (r.max_daily_loss_pct ?? 0.02) * 100;
  const maxDrawdownPct = (r.max_drawdown_pct ?? 0.05) * 100;
  const maxPositions = r.max_positions ?? 3;
  const cooldown = r.cooldown_seconds ?? 3600;

  // Current usage (percent)
  const dailyLossPct = Math.max(0, -(s.daily_pnl ?? 0) / (s.equity || 1) * 100);
  const ddPct = Math.max(0, ((s.peak_equity || 0) - (s.equity || 0)) / (s.peak_equity || 1) * 100);
  const positionsUsed = Object.keys(s.positions || {}).length;

  const meter = (label, cur, max, unit = '%') => {
    const pct = max > 0 ? Math.min(100, (cur / max) * 100) : 0;
    const cls = pct < 50 ? 'green' : pct < 80 ? 'yellow' : 'red';
    return `
      <div class="meter ${cls}">
        <div class="meter-label">${label}</div>
        <div class="meter-bar"><div class="meter-fill" style="width:${pct}%"></div></div>
        <div class="meter-value">${Number(cur).toFixed(2)}${unit} / ${Number(max).toFixed(2)}${unit}</div>
      </div>
    `;
  };

  setHtml(container, `
    <h4>Live risk meters</h4>
    <div class="risk-meters">
      ${meter('Daily Loss', dailyLossPct, maxDailyLossPct)}
      ${meter('Drawdown', ddPct, maxDrawdownPct)}
      ${meter('Positions', positionsUsed, maxPositions, '')}
    </div>
    <h4>Risk limits config</h4>
    <form class="risk-form" id="v2-risk-form">
      <label>Max Position %<input type="number" name="max_position_pct" value="${maxPosPct.toFixed(1)}" step="0.5"></label>
      <label>Max Exposure %<input type="number" name="max_total_exposure_pct" value="${maxExpPct.toFixed(1)}" step="1"></label>
      <label>Max Daily Loss %<input type="number" name="max_daily_loss_pct" value="${maxDailyLossPct.toFixed(2)}" step="0.1"></label>
      <label>Max Drawdown %<input type="number" name="max_drawdown_pct" value="${maxDrawdownPct.toFixed(1)}" step="1"></label>
      <label>Max Positions<input type="number" name="max_positions" value="${maxPositions}" step="1"></label>
      <label>Cooldown (s)<input type="number" name="cooldown_seconds" value="${cooldown}" step="60"></label>
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

  // Fetch all ops data in parallel
  const [s, okx, healer, presetsResp, logsResp] = await Promise.all([
    agentSvc.getStatus().catch(() => null),
    execSvc.getOkxStatus().catch(() => null),
    opsSvc.getHealerStatus().catch(() => null),
    agentSvc.getPresets().catch(() => ({ presets: {} })),
    opsSvc.getLogs(30, 'agent').catch(() => ({ logs: [] })),
  ]);

  const params = s?.strategy_params || {};
  const presets = presetsResp?.presets || {};
  const logs = logsResp?.logs || [];
  const watchSymbols = s?.watch_symbols || [];
  const tickInterval = s?.tick_interval_sec ?? 60;
  const signalInterval = s?.signal_interval || '4h';

  setHtml(container, `
    <h4>Strategy Config</h4>
    <form class="ops-form" id="v2-cfg-form">
      <label>Timeframe
        <select name="timeframe">
          ${['5m','15m','1h','4h','1d'].map(tf => `<option value="${tf}" ${tf === signalInterval ? 'selected' : ''}>${tf}</option>`).join('')}
        </select>
      </label>
      <label>Symbols (comma-separated)
        <input type="text" name="symbols" value="${watchSymbols.join(',')}" placeholder="BTCUSDT,ETHUSDT,...">
      </label>
      <label>Tick Interval (sec)
        <input type="number" name="tick_interval" value="${tickInterval}" min="10" max="600">
      </label>
      <label>Max Position %
        <input type="number" name="max_position_pct" value="${(s?.risk_limits?.max_position_pct || 5).toFixed(1)}" step="0.5" min="0.5" max="25">
      </label>
      <label>Max Positions
        <input type="number" name="max_positions" value="${s?.risk_limits?.max_positions || 3}" min="1" max="10">
      </label>
      <button type="submit" class="btn btn-primary">Apply Config</button>
    </form>

    <h4>Strategy Params (V6)</h4>
    <form class="ops-form params-grid" id="v2-params-form">
      ${Object.entries(params).map(([k, v]) => `
        <label>${k}<input type="number" name="${k}" value="${v}" step="${typeof v === 'number' && !Number.isInteger(v) ? '0.01' : '1'}"></label>
      `).join('')}
      <button type="submit" class="btn btn-primary">Save Params</button>
    </form>

    <h4>Strategy Presets</h4>
    <div class="ops-section">
      <div class="preset-row">
        <select id="v2-preset-select">
          <option value="">-- Select preset --</option>
          ${Object.keys(presets).map(name => `<option value="${name}">${name}</option>`).join('')}
        </select>
        <button class="btn" id="v2-preset-load">Load</button>
        <button class="btn" id="v2-preset-delete">Delete</button>
      </div>
      <div class="preset-row">
        <input type="text" id="v2-preset-name" placeholder="New preset name">
        <button class="btn btn-primary" id="v2-preset-save">Save Current</button>
      </div>
      <div class="muted" id="v2-preset-status"></div>
    </div>

    <h4>OKX API Keys</h4>
    <form class="ops-form" id="v2-okx-form">
      <div class="muted">Status: ${okx?.has_keys ? '<span class="pnl-pos">Connected</span>' : 'No keys configured'}</div>
      ${okx?.balance ? `<pre>${JSON.stringify(okx.balance, null, 2).slice(0, 300)}</pre>` : ''}
      <label>API Key<input type="password" name="api_key" placeholder="${okx?.has_keys ? 'already set — fill to update' : 'enter key'}"></label>
      <label>Secret<input type="password" name="secret"></label>
      <label>Passphrase<input type="password" name="passphrase"></label>
      <button type="submit" class="btn btn-primary">Save & Verify</button>
    </form>

    <h4>Telegram Notifications</h4>
    <form class="ops-form" id="v2-tg-form">
      <label>Bot Token<input type="password" name="bot_token" placeholder="token from @BotFather"></label>
      <label>Chat ID<input type="text" name="chat_id" placeholder="your chat id"></label>
      <div class="checkbox-row">
        <label><input type="checkbox" name="notify_signals" checked> Signals</label>
        <label><input type="checkbox" name="notify_fills" checked> Fills</label>
        <label><input type="checkbox" name="notify_errors"> Errors</label>
        <label><input type="checkbox" name="notify_daily"> Daily</label>
      </div>
      <button type="submit" class="btn btn-primary">Save & Test</button>
    </form>

    <h4>Agent Logs</h4>
    <div class="ops-section">
      <div class="agent-logs-view">
        ${logs.length === 0 ? '<div class="muted">No logs yet</div>' : logs.slice().reverse().map(l =>
          `<div class="log-line"><span class="log-time">${l.time || ''}</span> <span class="log-msg">${(l.msg || '').replace(/</g, '&lt;')}</span></div>`
        ).join('')}
      </div>
    </div>

    <h4>Self-Healer</h4>
    <div class="ops-section">
      <p>Running: <strong>${healer?.running ? 'YES' : 'NO'}</strong></p>
      <p>Fix count: <strong>${healer?.fix_count ?? 0}</strong></p>
      <p>AI: <strong>${healer?.has_ai ? 'enabled' : 'disabled'}</strong></p>
      <button class="btn" id="v2-healer-trigger">Trigger heal</button>
    </div>
  `);

  wireOpsForms();
}

function wireOpsForms() {
  // Strategy config
  on('#v2-cfg-form', 'submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const body = {
      timeframe: fd.get('timeframe'),
      symbols: (fd.get('symbols') || '').toString().split(',').map(s => s.trim()).filter(Boolean),
      tick_interval: Number(fd.get('tick_interval')),
      max_position_pct: Number(fd.get('max_position_pct')),
      max_positions: Number(fd.get('max_positions')),
    };
    try {
      await agentSvc.setStrategyConfig(body);
      setStatusMsg('Config applied');
      renderOps();
    } catch (err) { setStatusMsg('Config failed: ' + err.message); }
  });

  // Strategy params
  on('#v2-params-form', 'submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const params = {};
    fd.forEach((v, k) => { params[k] = Number(v); });
    try {
      await agentSvc.setStrategyParams(params);
      setStatusMsg('Params saved');
    } catch (err) { setStatusMsg('Params failed: ' + err.message); }
  });

  // Presets
  on('#v2-preset-load', 'click', async () => {
    const name = $('#v2-preset-select')?.value;
    if (!name) return;
    try { await agentSvc.loadPreset(name); setPresetStatus('Loaded: ' + name); renderOps(); }
    catch (err) { setPresetStatus('Load failed'); }
  });
  on('#v2-preset-delete', 'click', async () => {
    const name = $('#v2-preset-select')?.value;
    if (!name) return;
    try { await agentSvc.deletePreset(name); setPresetStatus('Deleted: ' + name); renderOps(); }
    catch (err) { setPresetStatus('Delete failed'); }
  });
  on('#v2-preset-save', 'click', async () => {
    const name = $('#v2-preset-name')?.value.trim();
    if (!name) { setPresetStatus('Enter a name'); return; }
    try { await agentSvc.savePreset(name); setPresetStatus('Saved: ' + name); renderOps(); }
    catch (err) { setPresetStatus('Save failed'); }
  });

  // OKX
  on('#v2-okx-form', 'submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const api_key = fd.get('api_key')?.toString().trim();
    const secret = fd.get('secret')?.toString().trim();
    const passphrase = fd.get('passphrase')?.toString().trim();
    if (!api_key || !secret || !passphrase) { setStatusMsg('All 3 OKX fields required'); return; }
    try {
      const r = await execSvc.setOkxKeys(api_key, secret, passphrase);
      setStatusMsg(r.ok ? 'OKX verified' : 'OKX failed: ' + (r.reason || ''));
      renderOps();
    } catch (err) { setStatusMsg('OKX request failed: ' + err.message); }
  });

  // Telegram
  on('#v2-tg-form', 'submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const body = {
      bot_token: fd.get('bot_token'),
      chat_id: fd.get('chat_id'),
      notify_signals: fd.get('notify_signals') === 'on',
      notify_fills: fd.get('notify_fills') === 'on',
      notify_errors: fd.get('notify_errors') === 'on',
      notify_daily: fd.get('notify_daily') === 'on',
    };
    try {
      const r = await opsSvc.setTelegramConfig(body);
      setStatusMsg(r.ok ? 'Telegram test message sent' : 'Telegram failed: ' + (r.reason || ''));
    } catch (err) { setStatusMsg('Telegram failed: ' + err.message); }
  });

  // Healer
  on('#v2-healer-trigger', 'click', async () => {
    await opsSvc.triggerHealer();
    renderOps();
  });
}

function setStatusMsg(msg) {
  console.log('[ops]', msg);
  // Could add a toast here. For now console.
}

function setPresetStatus(msg) {
  const el = $('#v2-preset-status');
  if (el) el.textContent = msg;
}
