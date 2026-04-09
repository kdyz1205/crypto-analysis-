import { $, $$, setHtml, on, show, hide } from '../util/dom.js';
import { agentState, setActiveSubTab, setPanelOpen, setLastStatus } from '../state/agent.js';
import {
  clearPaperExecutionError,
  isPaperExecutionBusy,
  paperExecutionState,
  setPaperExecutionConfig,
  setPaperExecutionError,
  setPaperExecutionKillSwitchUpdating,
  setPaperExecutionLastStep,
  setPaperExecutionLoadingConfig,
  setPaperExecutionLoadingState,
  setPaperExecutionResetting,
  setPaperExecutionStepping,
  setPaperExecutionState,
} from '../state/paper_execution.js';
import { marketState } from '../state/market.js';
import { subscribe } from '../util/events.js';
import * as agentSvc from '../services/agent.js';
import * as riskSvc from '../services/risk.js';
import * as execSvc from '../services/execution.js';
import * as opsSvc from '../services/ops.js';
import * as paperExecSvc from '../services/paper_execution.js';
import { formatPct, formatPrice, formatUsd, pnlColorClass } from '../util/format.js';

const PANEL_POLL_MS = 15000;

let pollTimer = null;

export function initExecutionPanel() {
  buildShell();
  wireEvents();
}

function buildShell() {
  const root = $('#v2-execution-panel');
  if (!root) return;
  setHtml(
    root,
    `
      <div class="exec-header">
        <h3>Execution Center</h3>
        <div class="exec-mode-badge" id="v2-exec-mode">-</div>
        <button class="exec-close" id="v2-exec-close">x</button>
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
    `,
  );
}

function wireEvents() {
  const root = $('#v2-execution-panel');
  if (!root) return;

  on('#v2-exec-close', 'click', () => closePanel());

  $$('.exec-tab', root).forEach((btn) => {
    on(btn, 'click', () => switchSubTab(btn.dataset.tab));
  });

  subscribe('agent.subtab.changed', (tab) => {
    $$('.exec-tab', root).forEach((button) => button.classList.toggle('active', button.dataset.tab === tab));
    $$('.exec-subtab', root).forEach((section) => section.classList.toggle('hidden', section.dataset.subtab !== tab));
    paintLoading();
    renderActive().catch((err) => renderTabError(tab, err));
  });
}

function switchSubTab(tab) {
  setActiveSubTab(tab);
}

export function openPanel() {
  setPanelOpen(true);
  show('#v2-execution-panel');
  paintLoading();
  renderActive().catch((err) => renderTabError(agentState.activeSubTab, err));
  startPolling();
}

export function closePanel() {
  setPanelOpen(false);
  hide('#v2-execution-panel');
  stopPolling();
}

export function togglePanel() {
  if (agentState.panelOpen) closePanel();
  else openPanel();
}

function paintLoading() {
  const container = $(`[data-subtab="${agentState.activeSubTab}"]`);
  if (container && !container.innerHTML.trim()) {
    setHtml(container, '<div class="muted" style="padding:40px;text-align:center">Loading...</div>');
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(async () => {
    if (!agentState.panelOpen) return;
    await Promise.allSettled([refreshAgentStatus(), loadPaperExecutionState()]);
    if (agentState.activeSubTab === 'risk' && !paperExecutionState.config) {
      await loadPaperExecutionConfig().catch(() => null);
    }
    renderActive(true).catch((err) => console.warn('[exec] poll render failed:', err));
  }, PANEL_POLL_MS);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

async function renderActive(useCached = false) {
  const tab = agentState.activeSubTab;
  switch (tab) {
    case 'overview':
      await renderOverview(useCached);
      break;
    case 'execution':
      await renderExecution(useCached);
      break;
    case 'risk':
      await renderRisk(useCached);
      break;
    case 'ops':
      await renderOps(useCached);
      break;
    default:
      break;
  }
}

function renderTabError(tab, err) {
  console.error(`[exec] render ${tab} failed:`, err);
  const container = $(`[data-subtab="${tab}"]`);
  if (!container) return;
  setHtml(
    container,
    `<div class="paper-section"><div class="paper-error">Load failed: ${escapeHtml(safeErrorMessage(err))}</div></div>`,
  );
}

async function refreshAgentStatus() {
  const status = await agentSvc.getStatus();
  setLastStatus(status);
  return status;
}

async function loadPaperExecutionState(force = true) {
  if (!force && paperExecutionState.state) return paperExecutionState.state;
  if (paperExecutionState.loadingState) return paperExecutionState.state;
  setPaperExecutionLoadingState(true);
  try {
    const state = await paperExecSvc.getPaperExecutionState();
    setPaperExecutionState(state);
    if (!paperExecutionState.config && state?.config) {
      setPaperExecutionConfig(state.config);
    }
    clearPaperExecutionError();
    return state;
  } catch (err) {
    setPaperExecutionError(safeErrorMessage(err));
    throw err;
  } finally {
    setPaperExecutionLoadingState(false);
  }
}

async function loadPaperExecutionConfig(force = true) {
  if (!force && paperExecutionState.config) return paperExecutionState.config;
  if (paperExecutionState.loadingConfig) return paperExecutionState.config;
  setPaperExecutionLoadingConfig(true);
  try {
    const config = await paperExecSvc.getPaperExecutionConfig();
    setPaperExecutionConfig(config);
    clearPaperExecutionError();
    return config;
  } catch (err) {
    setPaperExecutionError(safeErrorMessage(err));
    throw err;
  } finally {
    setPaperExecutionLoadingConfig(false);
  }
}

async function renderOverview(useCached = false) {
  const container = $('[data-subtab="overview"]');
  if (!container) return;

  let agentError = null;
  let paperError = null;

  if (!useCached || !agentState.lastStatus) {
    try {
      await refreshAgentStatus();
    } catch (err) {
      agentError = safeErrorMessage(err);
    }
  }
  if (!useCached || !paperExecutionState.state) {
    try {
      await loadPaperExecutionState(!useCached);
    } catch (err) {
      paperError = safeErrorMessage(err);
    }
  }

  const status = agentState.lastStatus;
  const paperState = paperExecutionState.state;
  const modeBadge = $('#v2-exec-mode');
  if (modeBadge) {
    modeBadge.textContent = status?.mode ? `AGENT ${String(status.mode).toUpperCase()}` : 'PAPER';
  }

  setHtml(
    container,
    [
      renderAgentOverviewSection(status, agentError),
      renderPaperOverviewSection(paperState, paperError || paperExecutionState.lastError),
    ].join(''),
  );
}

async function renderExecution(useCached = false) {
  const container = $('[data-subtab="execution"]');
  if (!container) return;

  let agentError = null;
  let paperError = null;

  if (!useCached || !agentState.lastStatus) {
    try {
      await refreshAgentStatus();
    } catch (err) {
      agentError = safeErrorMessage(err);
    }
  }
  if (!useCached || !paperExecutionState.state) {
    try {
      await loadPaperExecutionState(!useCached);
    } catch (err) {
      paperError = safeErrorMessage(err);
    }
  }

  const status = agentState.lastStatus;
  const paperState = paperExecutionState.state;

  setHtml(
    container,
    [
      renderAgentExecutionSection(status, agentError),
      renderPaperExecutionSection(paperState, paperError || paperExecutionState.lastError, paperExecutionState.lastStepResult),
    ].join(''),
  );

  wireLegacyExecutionControls();
  wirePaperExecutionControls();
}

async function renderRisk(useCached = false) {
  const container = $('[data-subtab="risk"]');
  if (!container) return;

  let agentError = null;
  let paperError = null;

  if (!useCached || !agentState.lastStatus) {
    try {
      await refreshAgentStatus();
    } catch (err) {
      agentError = safeErrorMessage(err);
    }
  }
  if (!useCached || !paperExecutionState.state) {
    try {
      await loadPaperExecutionState(!useCached);
    } catch (err) {
      paperError = safeErrorMessage(err);
    }
  }
  if (!useCached || !paperExecutionState.config) {
    try {
      await loadPaperExecutionConfig(!useCached);
    } catch (err) {
      paperError = safeErrorMessage(err);
    }
  }

  setHtml(
    container,
    [
      renderAgentRiskSection(agentState.lastStatus, agentError),
      renderPaperRiskSection(
        paperExecutionState.config || paperExecutionState.state?.config,
        paperError || paperExecutionState.lastError,
      ),
    ].join(''),
  );

  wireLegacyRiskForm();
  wirePaperRiskForm();
}

async function renderOps(useCached = false) {
  const container = $('[data-subtab="ops"]');
  if (!container) return;

  let paperError = null;
  if (!useCached || !paperExecutionState.state) {
    try {
      await loadPaperExecutionState(!useCached);
    } catch (err) {
      paperError = safeErrorMessage(err);
    }
  }

  const [statusResult, okxResult, healerResult, presetsResult, logsResult] = await Promise.allSettled([
    refreshAgentStatus(),
    execSvc.getOkxStatus(),
    opsSvc.getHealerStatus(),
    agentSvc.getPresets(),
    opsSvc.getLogs(30, 'agent'),
  ]);

  const status = statusResult.status === 'fulfilled' ? statusResult.value : agentState.lastStatus;
  const okx = okxResult.status === 'fulfilled' ? okxResult.value : null;
  const healer = healerResult.status === 'fulfilled' ? healerResult.value : null;
  const presets = presetsResult.status === 'fulfilled' ? presetsResult.value?.presets || {} : {};
  const logs = logsResult.status === 'fulfilled' ? logsResult.value?.logs || [] : [];

  setHtml(
    container,
    [
      renderPaperOpsSection(paperExecutionState.state, paperError || paperExecutionState.lastError),
      renderLegacyOpsSection(status, okx, healer, presets, logs),
    ].join(''),
  );

  wireOpsForms();
}

function renderAgentOverviewSection(status, error) {
  if (!status) {
    return renderUnavailableSection('Agent Overview', error || 'Agent status unavailable');
  }

  const positions = Object.values(status.positions || {});
  const trades = status.recent_trades || [];

  return `
    <section class="paper-section">
      <div class="paper-section-header">
        <h4>Agent Overview</h4>
        <span class="paper-badge ${status.running ? 'is-ok' : 'is-muted'}">${status.running ? 'RUNNING' : 'STOPPED'}</span>
      </div>
      <div class="exec-hero">
        <div class="hero-stat">
          <div class="hero-label">Total Equity</div>
          <div class="hero-value">${formatUsd(status.equity)}</div>
        </div>
        <div class="hero-stat">
          <div class="hero-label">P&L</div>
          <div class="hero-value ${pnlColorClass(status.total_pnl_usd)}">${formatUsd(status.total_pnl_usd)}</div>
        </div>
        <div class="hero-stat">
          <div class="hero-label">Today</div>
          <div class="hero-value ${pnlColorClass(status.daily_pnl)}">${formatUsd(status.daily_pnl)}</div>
        </div>
        <div class="hero-stat">
          <div class="hero-label">Win Rate</div>
          <div class="hero-value">${(status.win_rate ?? 0).toFixed(1)}%</div>
        </div>
      </div>
      <div class="exec-stats-grid">
        <div class="stat"><div class="stat-label">Cash</div><div class="stat-value">${formatUsd(status.cash)}</div></div>
        <div class="stat"><div class="stat-label">Trades</div><div class="stat-value">${status.total_trades ?? 0}</div></div>
        <div class="stat"><div class="stat-label">Regime</div><div class="stat-value">${escapeHtml(status.harness?.market_regime || '-')}</div></div>
        <div class="stat"><div class="stat-label">Generation</div><div class="stat-value">${status.generation ?? 0}</div></div>
        <div class="stat"><div class="stat-label">Phase</div><div class="stat-value">${escapeHtml(status.cycle_phase || '-')}</div></div>
        <div class="stat"><div class="stat-label">Mode</div><div class="stat-value">${escapeHtml(String(status.mode || '-').toUpperCase())}</div></div>
      </div>
      <div class="paper-subgrid">
        <div>
          <h4>Open positions (${positions.length})</h4>
          <div class="exec-positions">
            ${positions.length === 0
              ? '<div class="paper-empty">No open positions</div>'
              : positions
                  .map(
                    (position) => `
                      <div class="pos-row">
                        <span class="pos-sym">${escapeHtml(position.symbol || '-')}</span>
                        <span class="pos-side ${escapeHtml(position.side || '')}">${escapeHtml(String(position.side || '-').toUpperCase())}</span>
                        <span>${formatUsd(position.size_usd || position.size)}</span>
                        <span class="${pnlColorClass(position.unrealized_pnl_pct)}">${formatPct(position.unrealized_pnl_pct)}</span>
                      </div>
                    `,
                  )
                  .join('')}
          </div>
        </div>
        <div>
          <h4>Recent trades</h4>
          <div class="exec-trades">
            ${trades.length === 0
              ? '<div class="paper-empty">No trades yet</div>'
              : trades
                  .slice(-5)
                  .reverse()
                  .map(
                    (trade) => `
                      <div class="trade-row">
                        <span>${escapeHtml(trade.symbol || '-')}</span>
                        <span class="${escapeHtml(trade.side || '')}">${escapeHtml(String(trade.side || '-').toUpperCase())}</span>
                        <span class="${pnlColorClass(trade.pnl_pct)}">${formatPct(trade.pnl_pct)}</span>
                        <span class="muted">${escapeHtml(trade.reason || '')}</span>
                      </div>
                    `,
                  )
                  .join('')}
          </div>
        </div>
      </div>
    </section>
  `;
}

function renderPaperOverviewSection(state, error) {
  if (!state) {
    return renderUnavailableSection('Paper Execution', error || 'Paper execution state unavailable');
  }

  const account = state.account || {};
  const killSwitch = state.kill_switch || {};
  const recentFills = (state.recent_fills || []).slice(-8).reverse();
  const recentClosed = (state.recent_closed_positions || []).slice(-8).reverse();
  const cooldownCount = Object.keys(state.cooldowns || {}).length;

  return `
    <section class="paper-section">
      <div class="paper-section-header">
        <h4>Paper Execution</h4>
        <span class="paper-badge ${killSwitch.blocked ? 'is-danger' : 'is-ok'}">
          ${killSwitch.blocked ? 'KILL SWITCH ON' : 'READY'}
        </span>
      </div>
      <div class="exec-hero">
        <div class="hero-stat">
          <div class="hero-label">Equity</div>
          <div class="hero-value">${formatUsd(account.equity)}</div>
        </div>
        <div class="hero-stat">
          <div class="hero-label">Realized</div>
          <div class="hero-value ${pnlColorClass(account.realized_pnl)}">${formatUsd(account.realized_pnl)}</div>
        </div>
        <div class="hero-stat">
          <div class="hero-label">Unrealized</div>
          <div class="hero-value ${pnlColorClass(account.unrealized_pnl)}">${formatUsd(account.unrealized_pnl)}</div>
        </div>
        <div class="hero-stat">
          <div class="hero-label">Exposure</div>
          <div class="hero-value">${formatUsd(account.total_exposure)}</div>
        </div>
      </div>
      <div class="exec-stats-grid">
        <div class="stat"><div class="stat-label">Daily Realized</div><div class="stat-value ${pnlColorClass(account.daily_realized_pnl)}">${formatUsd(account.daily_realized_pnl)}</div></div>
        <div class="stat"><div class="stat-label">Open Orders</div><div class="stat-value">${account.open_order_count ?? 0}</div></div>
        <div class="stat"><div class="stat-label">Open Positions</div><div class="stat-value">${account.open_position_count ?? 0}</div></div>
        <div class="stat"><div class="stat-label">Closed Trades</div><div class="stat-value">${account.closed_trade_count ?? 0}</div></div>
        <div class="stat"><div class="stat-label">Consecutive Losses</div><div class="stat-value">${account.consecutive_losses ?? 0}</div></div>
        <div class="stat"><div class="stat-label">Cooldowns</div><div class="stat-value">${cooldownCount}</div></div>
      </div>
      <div class="paper-kill-banner ${killSwitch.blocked ? 'is-danger' : 'is-ok'}">
        <strong>${killSwitch.blocked ? 'Blocked' : 'Unblocked'}</strong>
        <span>${escapeHtml(killSwitch.reason || 'No active paper-execution block')}</span>
      </div>
      ${error ? `<div class="paper-error">${escapeHtml(error)}</div>` : ''}
      <div class="paper-subgrid">
        <div>
          <h4>Recent fills</h4>
          ${renderPaperFillRows(recentFills)}
        </div>
        <div>
          <h4>Recent closed positions</h4>
          ${renderPaperClosedPositionRows(recentClosed)}
        </div>
      </div>
    </section>
  `;
}

function renderAgentExecutionSection(status, error) {
  if (!status) {
    return renderUnavailableSection('Agent Execution', error || 'Agent execution controls unavailable');
  }

  const signals = status.last_signals || {};
  const entries = Object.entries(signals);

  return `
    <section class="paper-section">
      <div class="paper-section-header">
        <h4>Agent Execution</h4>
        <span class="paper-badge ${status.mode === 'live' ? 'is-danger' : 'is-muted'}">${escapeHtml(String(status.mode || '-').toUpperCase())}</span>
      </div>
      <div class="exec-controls">
        <button class="btn btn-primary" id="v2-start-btn">Start</button>
        <button class="btn" id="v2-stop-btn">Stop</button>
        <button class="btn" id="v2-revive-btn">Revive</button>
        <button class="btn" id="v2-scan-btn">Scan Now</button>
        <button class="btn ${status.mode === 'paper' ? 'active' : ''}" id="v2-paper-btn">Paper</button>
        <button class="btn ${status.mode === 'live' ? 'active' : ''}" id="v2-live-btn">Live</button>
      </div>
      <h4>Current signals</h4>
      <div class="signals-list">
        ${entries.length === 0
          ? '<div class="paper-empty">No signals yet</div>'
          : entries
              .map(
                ([symbol, signal]) => `
                  <div class="signal-row ${signal.blocked ? 'blocked' : ''}">
                    <span class="signal-sym">${escapeHtml(symbol)}</span>
                    <span class="signal-action ${escapeHtml(signal.action || '')}">${escapeHtml(String(signal.action || '-').toUpperCase())}</span>
                    <span class="signal-conf">${signal.confidence ? `${Math.round(signal.confidence * 100)}%` : '-'}</span>
                    <div class="signal-reason">${escapeHtml(signal.reason || '')}</div>
                    ${signal.blocked ? `<div class="signal-block">Blocked: ${escapeHtml((signal.block_reasons || []).join('; '))}</div>` : ''}
                  </div>
                `,
              )
              .join('')}
      </div>
    </section>
  `;
}

function renderPaperExecutionSection(state, error, lastStepResult) {
  if (!state) {
    return renderUnavailableSection('Paper Orders / Positions', error || 'Paper execution state unavailable');
  }

  const killSwitch = state.kill_switch || {};
  const orders = state.open_orders || [];
  const positions = state.open_positions || [];
  const cooldowns = state.cooldowns || {};
  const currentSymbol = marketState.currentSymbol || 'HYPEUSDT';
  const currentInterval = marketState.currentInterval || '4h';
  const busy = isPaperExecutionBusy();
  const stepDisabled = busy || paperExecutionState.loadingState || paperExecutionState.loadingConfig;
  const resetDisabled = busy || paperExecutionState.loadingState || paperExecutionState.loadingConfig;
  const killDisabled = busy || paperExecutionState.loadingState || paperExecutionState.loadingConfig;
  const stepLabel = paperExecutionState.stepping ? 'Stepping...' : 'Step once';
  const resetLabel = paperExecutionState.resetting ? 'Resetting...' : 'Reset paper';
  const killOnLabel = paperExecutionState.killSwitchUpdating && killSwitch.blocked ? 'Updating...' : 'Kill switch on';
  const killOffLabel = paperExecutionState.killSwitchUpdating && !killSwitch.blocked ? 'Updating...' : 'Kill switch off';

  return `
    <section class="paper-section">
      <div class="paper-section-header">
        <h4>Paper Orders / Positions</h4>
        <span class="paper-badge ${killSwitch.blocked ? 'is-danger' : 'is-ok'}">${killSwitch.blocked ? 'Blocked' : 'Interactive'}</span>
      </div>
      <div class="paper-actions">
        <button class="btn btn-primary" id="v2-paper-step-btn" ${stepDisabled ? 'disabled' : ''}>${stepLabel}</button>
        <button class="btn" id="v2-paper-reset-btn" ${resetDisabled ? 'disabled' : ''}>${resetLabel}</button>
        <button class="btn ${killSwitch.blocked ? '' : 'active'}" id="v2-paper-kill-on" ${killDisabled ? 'disabled' : ''}>${killOnLabel}</button>
        <button class="btn ${killSwitch.blocked ? 'active' : ''}" id="v2-paper-kill-off" ${killDisabled ? 'disabled' : ''}>${killOffLabel}</button>
      </div>
      <form class="paper-form" id="v2-paper-step-form">
        <label>Symbol<input type="text" name="symbol" value="${escapeHtml(currentSymbol)}" ${stepDisabled ? 'disabled' : ''} /></label>
        <label>Interval<input type="text" name="interval" value="${escapeHtml(currentInterval)}" ${stepDisabled ? 'disabled' : ''} /></label>
        <label>Bar Index<input type="number" name="bar_index" placeholder="auto next bar" ${stepDisabled ? 'disabled' : ''} /></label>
        <label>Analysis Bars<input type="number" name="analysis_bars" value="500" min="50" step="50" ${stepDisabled ? 'disabled' : ''} /></label>
      </form>
      ${busy ? '<div class="paper-note">Paper execution request in progress...</div>' : ''}
      ${error ? `<div class="paper-error">${escapeHtml(error)}</div>` : ''}
      ${renderPaperLastStep(lastStepResult)}
      <div class="paper-subgrid">
        <div>
          <h4>Open orders (${orders.length})</h4>
          ${renderPaperOrderRows(orders)}
        </div>
        <div>
          <h4>Open positions (${positions.length})</h4>
          ${renderPaperPositionRows(positions)}
        </div>
      </div>
      <div>
        <h4>Cooldowns</h4>
        ${renderPaperCooldownRows(cooldowns)}
      </div>
    </section>
  `;
}

function renderAgentRiskSection(status, error) {
  if (!status) {
    return renderUnavailableSection('Agent Risk', error || 'Agent risk data unavailable');
  }

  const limits = status.risk_limits || {};
  const maxPositionPct = (limits.max_position_pct ?? 0.05) * 100;
  const maxExposurePct = (limits.max_total_exposure_pct ?? 0.15) * 100;
  const maxDailyLossPct = (limits.max_daily_loss_pct ?? 0.02) * 100;
  const maxDrawdownPct = (limits.max_drawdown_pct ?? 0.05) * 100;
  const maxPositions = limits.max_positions ?? 3;
  const cooldown = limits.cooldown_seconds ?? 3600;
  const dailyLossPct = Math.max(0, (-Number(status.daily_pnl || 0) / (Number(status.equity) || 1)) * 100);
  const drawdownPct = Math.max(
    0,
    ((Number(status.peak_equity || 0) - Number(status.equity || 0)) / (Number(status.peak_equity) || 1)) * 100,
  );
  const positionsUsed = Object.keys(status.positions || {}).length;

  return `
    <section class="paper-section">
      <div class="paper-section-header">
        <h4>Agent Risk</h4>
        <span class="paper-badge is-muted">Legacy Agent</span>
      </div>
      <h4>Live risk meters</h4>
      <div class="risk-meters">
        ${renderRiskMeter('Daily Loss', dailyLossPct, maxDailyLossPct, '%')}
        ${renderRiskMeter('Drawdown', drawdownPct, maxDrawdownPct, '%')}
        ${renderRiskMeter('Positions', positionsUsed, maxPositions, '')}
      </div>
      <h4>Risk limits config</h4>
      <form class="risk-form" id="v2-risk-form">
        <label>Max Position %<input type="number" name="max_position_pct" value="${maxPositionPct.toFixed(1)}" step="0.5" /></label>
        <label>Max Exposure %<input type="number" name="max_total_exposure_pct" value="${maxExposurePct.toFixed(1)}" step="1" /></label>
        <label>Max Daily Loss %<input type="number" name="max_daily_loss_pct" value="${maxDailyLossPct.toFixed(2)}" step="0.1" /></label>
        <label>Max Drawdown %<input type="number" name="max_drawdown_pct" value="${maxDrawdownPct.toFixed(1)}" step="1" /></label>
        <label>Max Positions<input type="number" name="max_positions" value="${maxPositions}" step="1" /></label>
        <label>Cooldown (s)<input type="number" name="cooldown_seconds" value="${cooldown}" step="60" /></label>
        <button type="submit" class="btn btn-primary">Save</button>
      </form>
    </section>
  `;
}

function renderPaperRiskSection(config, error) {
  if (!config) {
    return renderUnavailableSection('Paper Risk Config', error || 'Paper risk config unavailable');
  }

  const busy = isPaperExecutionBusy() || paperExecutionState.loadingConfig;

  return `
    <section class="paper-section">
      <div class="paper-section-header">
        <h4>Paper Execution Risk Config</h4>
        <span class="paper-badge is-ok">/api/paper-execution/config</span>
      </div>
      ${error ? `<div class="paper-error">${escapeHtml(error)}</div>` : ''}
      <form class="paper-form paper-risk-form" id="v2-paper-risk-form">
        <label>Risk Per Trade<input type="number" name="risk_per_trade" value="${Number(config.risk_per_trade).toFixed(4)}" step="0.0005" ${busy ? 'disabled' : ''} /></label>
        <label>Max Concurrent Positions<input type="number" name="max_concurrent_positions" value="${config.max_concurrent_positions}" step="1" ${busy ? 'disabled' : ''} /></label>
        <label>Max Positions Per Symbol<input type="number" name="max_positions_per_symbol" value="${config.max_positions_per_symbol}" step="1" ${busy ? 'disabled' : ''} /></label>
        <label>Max Total Exposure<input type="number" name="max_total_exposure" value="${Number(config.max_total_exposure).toFixed(2)}" step="0.05" ${busy ? 'disabled' : ''} /></label>
        <label>Max Daily Loss<input type="number" name="max_daily_loss" value="${Number(config.max_daily_loss).toFixed(4)}" step="0.001" ${busy ? 'disabled' : ''} /></label>
        <label>Max Consecutive Losses<input type="number" name="max_consecutive_losses" value="${config.max_consecutive_losses}" step="1" ${busy ? 'disabled' : ''} /></label>
        <label>Cancel After Bars<input type="number" name="cancel_after_bars" value="${config.cancel_after_bars}" step="1" ${busy ? 'disabled' : ''} /></label>
        <label>Cooldown Bars After Loss<input type="number" name="cooldown_bars_after_loss" value="${config.cooldown_bars_after_loss}" step="1" ${busy ? 'disabled' : ''} /></label>
        <label>Starting Equity<input type="number" name="starting_equity" value="${Number(config.starting_equity).toFixed(2)}" step="100" ${busy ? 'disabled' : ''} /></label>
        <label class="paper-checkbox">
          <input type="checkbox" name="allow_multiple_same_direction_per_symbol" ${config.allow_multiple_same_direction_per_symbol ? 'checked' : ''} ${busy ? 'disabled' : ''} />
          Allow multiple same-direction positions per symbol
        </label>
        <button type="submit" class="btn btn-primary" ${busy ? 'disabled' : ''}>${paperExecutionState.loadingConfig ? 'Loading...' : 'Save Paper Config'}</button>
      </form>
    </section>
  `;
}

function renderPaperOpsSection(state, error) {
  if (!state) {
    return renderUnavailableSection('Paper Diagnostics', error || 'Paper diagnostics unavailable');
  }

  const streamMap = state.account?.last_processed_bar_by_stream || {};
  const entries = Object.entries(streamMap);
  const killSwitch = state.kill_switch || {};

  return `
    <section class="paper-section">
      <div class="paper-section-header">
        <h4>Paper Execution Diagnostics</h4>
        <span class="paper-badge ${killSwitch.blocked ? 'is-danger' : 'is-ok'}">${killSwitch.blocked ? 'Blocked' : 'Healthy'}</span>
      </div>
      <div class="paper-diagnostics-grid">
        <div class="stat"><div class="stat-label">Streams</div><div class="stat-value">${entries.length}</div></div>
        <div class="stat"><div class="stat-label">Last Error</div><div class="stat-value">${escapeHtml(paperExecutionState.lastError || 'None')}</div></div>
        <div class="stat"><div class="stat-label">Kill Switch</div><div class="stat-value">${killSwitch.blocked ? 'BLOCKED' : 'CLEAR'}</div></div>
        <div class="stat"><div class="stat-label">Manual Block</div><div class="stat-value">${killSwitch.manual_blocked ? 'ON' : 'OFF'}</div></div>
      </div>
      ${error ? `<div class="paper-error">${escapeHtml(error)}</div>` : ''}
      <h4>Current stream keys</h4>
      ${
        entries.length === 0
          ? '<div class="paper-empty">No processed streams yet</div>'
          : entries
              .map(
                ([stream, lastBar]) => `
                  <div class="paper-cooldown-row">
                    <span>${escapeHtml(stream)}</span>
                    <span class="muted">last bar ${lastBar}</span>
                  </div>
                `,
              )
              .join('')
      }
      <div class="paper-kill-banner ${killSwitch.blocked ? 'is-danger' : 'is-ok'}">
        <strong>Kill switch snapshot</strong>
        <span>${escapeHtml(killSwitch.reason || 'No active block reason')}</span>
      </div>
    </section>
  `;
}

function renderLegacyOpsSection(status, okx, healer, presets, logs) {
  const params = status?.strategy_params || {};
  const watchSymbols = status?.watch_symbols || [];
  const tickInterval = status?.tick_interval_sec ?? 60;
  const signalInterval = status?.signal_interval || '4h';

  return `
    <section class="paper-section">
      <div class="paper-section-header">
        <h4>Ops</h4>
        <span class="paper-badge is-muted">Legacy Agent</span>
      </div>
      <h4>Strategy Config</h4>
      <form class="ops-form" id="v2-cfg-form">
        <label>Timeframe
          <select name="timeframe">
            ${['5m', '15m', '1h', '4h', '1d']
              .map((timeframe) => `<option value="${timeframe}" ${timeframe === signalInterval ? 'selected' : ''}>${timeframe}</option>`)
              .join('')}
          </select>
        </label>
        <label>Symbols (comma-separated)
          <input type="text" name="symbols" value="${escapeHtml(watchSymbols.join(','))}" placeholder="BTCUSDT,ETHUSDT,..." />
        </label>
        <label>Tick Interval (sec)
          <input type="number" name="tick_interval" value="${tickInterval}" min="10" max="600" />
        </label>
        <label>Max Position %
          <input type="number" name="max_position_pct" value="${Number(status?.risk_limits?.max_position_pct || 5).toFixed(1)}" step="0.5" min="0.5" max="25" />
        </label>
        <label>Max Positions
          <input type="number" name="max_positions" value="${status?.risk_limits?.max_positions || 3}" min="1" max="10" />
        </label>
        <button type="submit" class="btn btn-primary">Apply Config</button>
      </form>

      <h4>Strategy Params (V6)</h4>
      <form class="ops-form params-grid" id="v2-params-form">
        ${Object.entries(params)
          .map(
            ([key, value]) => `
              <label>${escapeHtml(key)}<input type="number" name="${escapeHtml(key)}" value="${value}" step="${
                typeof value === 'number' && !Number.isInteger(value) ? '0.01' : '1'
              }" /></label>
            `,
          )
          .join('')}
        <button type="submit" class="btn btn-primary">Save Params</button>
      </form>

      <h4>Strategy Presets</h4>
      <div class="ops-section">
        <div class="preset-row">
          <select id="v2-preset-select">
            <option value="">-- Select preset --</option>
            ${Object.keys(presets)
              .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
              .join('')}
          </select>
          <button class="btn" id="v2-preset-load">Load</button>
          <button class="btn" id="v2-preset-delete">Delete</button>
        </div>
        <div class="preset-row">
          <input type="text" id="v2-preset-name" placeholder="New preset name" />
          <button class="btn btn-primary" id="v2-preset-save">Save Current</button>
        </div>
        <div class="muted" id="v2-preset-status"></div>
      </div>

      <h4>OKX API Keys</h4>
      <form class="ops-form" id="v2-okx-form">
        <div class="muted">Status: ${okx?.has_keys ? '<span class="pnl-pos">Connected</span>' : 'No keys configured'}</div>
        ${okx?.balance ? `<pre>${escapeHtml(JSON.stringify(okx.balance, null, 2).slice(0, 300))}</pre>` : ''}
        <label>API Key<input type="password" name="api_key" placeholder="${okx?.has_keys ? 'already set - fill to update' : 'enter key'}" /></label>
        <label>Secret<input type="password" name="secret" /></label>
        <label>Passphrase<input type="password" name="passphrase" /></label>
        <button type="submit" class="btn btn-primary">Save and Verify</button>
      </form>

      <h4>Telegram Notifications</h4>
      <form class="ops-form" id="v2-tg-form">
        <label>Bot Token<input type="password" name="bot_token" placeholder="token from @BotFather" /></label>
        <label>Chat ID<input type="text" name="chat_id" placeholder="your chat id" /></label>
        <div class="checkbox-row">
          <label><input type="checkbox" name="notify_signals" checked /> Signals</label>
          <label><input type="checkbox" name="notify_fills" checked /> Fills</label>
          <label><input type="checkbox" name="notify_errors" /> Errors</label>
          <label><input type="checkbox" name="notify_daily" /> Daily</label>
        </div>
        <button type="submit" class="btn btn-primary">Save and Test</button>
      </form>

      <h4>Agent Logs</h4>
      <div class="ops-section">
        <div class="agent-logs-view">
          ${
            logs.length === 0
              ? '<div class="muted">No logs yet</div>'
              : logs
                  .slice()
                  .reverse()
                  .map(
                    (log) => `
                      <div class="log-line">
                        <span class="log-time">${escapeHtml(log.time || '')}</span>
                        <span class="log-msg">${escapeHtml(log.msg || '')}</span>
                      </div>
                    `,
                  )
                  .join('')
          }
        </div>
      </div>

      <h4>Self-Healer</h4>
      <div class="ops-section">
        <p>Running: <strong>${healer?.running ? 'YES' : 'NO'}</strong></p>
        <p>Fix count: <strong>${healer?.fix_count ?? 0}</strong></p>
        <p>AI: <strong>${healer?.has_ai ? 'enabled' : 'disabled'}</strong></p>
        <button class="btn" id="v2-healer-trigger">Trigger heal</button>
      </div>
    </section>
  `;
}

function renderPaperOrderRows(orders) {
  if (!orders || orders.length === 0) {
    return '<div class="paper-empty">No open paper orders</div>';
  }

  return orders
    .map(
      (order) => `
        <div class="paper-order-row">
          <div class="paper-order-main">
            <span class="pos-sym">${escapeHtml(order.symbol)}</span>
            <span class="paper-badge is-muted">${escapeHtml(String(order.side || '-').toUpperCase())}</span>
            <span>${escapeHtml(String(order.order_type || '-').toUpperCase())}</span>
            <span>${formatPrice(order.price)}</span>
            <span>qty ${Number(order.quantity).toFixed(4)}</span>
          </div>
          <div class="paper-order-meta">
            <span class="muted">${escapeHtml(order.signal_id)}</span>
            <span class="muted">bar ${order.created_at_bar}</span>
            <span class="muted">${escapeHtml(order.status || '')}</span>
          </div>
        </div>
      `,
    )
    .join('');
}

function renderPaperPositionRows(positions) {
  if (!positions || positions.length === 0) {
    return '<div class="paper-empty">No open paper positions</div>';
  }

  return positions
    .map(
      (position) => `
        <div class="paper-order-row">
          <div class="paper-order-main">
            <span class="pos-sym">${escapeHtml(position.symbol)}</span>
            <span class="paper-badge ${position.direction === 'long' ? 'is-ok' : 'is-danger'}">${escapeHtml(
              String(position.direction || '-').toUpperCase(),
            )}</span>
            <span>entry ${formatPrice(position.entry_price)}</span>
            <span>mark ${formatPrice(position.mark_price)}</span>
            <span class="${pnlColorClass(position.unrealized_pnl)}">${formatUsd(position.unrealized_pnl)}</span>
          </div>
          <div class="paper-order-meta">
            <span class="muted">qty ${Number(position.quantity).toFixed(4)}</span>
            <span class="muted">stop ${formatPrice(position.stop_price)}</span>
            <span class="muted">tp ${formatPrice(position.tp_price)}</span>
          </div>
        </div>
      `,
    )
    .join('');
}

function renderPaperFillRows(fills) {
  if (!fills || fills.length === 0) {
    return '<div class="paper-empty">No paper fills yet</div>';
  }

  return fills
    .map(
      (fill) => `
        <div class="paper-fill-row">
          <span class="pos-sym">${escapeHtml(fill.symbol)}</span>
          <span class="paper-badge is-muted">${escapeHtml(String(fill.side || '-').toUpperCase())}</span>
          <span>${formatPrice(fill.fill_price)}</span>
          <span>qty ${Number(fill.quantity).toFixed(4)}</span>
          <span class="muted">bar ${fill.filled_at_bar}</span>
        </div>
      `,
    )
    .join('');
}

function renderPaperClosedPositionRows(positions) {
  if (!positions || positions.length === 0) {
    return '<div class="paper-empty">No closed paper positions yet</div>';
  }

  return positions
    .map(
      (position) => `
        <div class="paper-fill-row">
          <span class="pos-sym">${escapeHtml(position.symbol)}</span>
          <span class="paper-badge is-muted">${escapeHtml(String(position.direction || '-').toUpperCase())}</span>
          <span class="${pnlColorClass(position.realized_pnl)}">${formatUsd(position.realized_pnl)}</span>
          <span class="muted">${escapeHtml(position.exit_reason || '-')}</span>
          <span class="muted">bar ${position.closed_at_bar ?? '-'}</span>
        </div>
      `,
    )
    .join('');
}

function renderPaperCooldownRows(cooldowns) {
  const entries = Object.entries(cooldowns || {});
  if (entries.length === 0) {
    return '<div class="paper-empty">No active cooldowns</div>';
  }

  return entries
    .map(
      ([scope, untilBar]) => `
        <div class="paper-cooldown-row">
          <span>${escapeHtml(scope)}</span>
          <span class="muted">until bar ${untilBar}</span>
        </div>
      `,
    )
    .join('');
}

function renderPaperLastStep(lastStepResult) {
  if (!lastStepResult) {
    return '<div class="paper-note">No paper step executed yet in this session.</div>';
  }
  return `
    <div class="paper-note">
      <strong>Last step:</strong>
      stream ${escapeHtml(lastStepResult.stream || '-')} |
      processed ${escapeHtml((lastStepResult.processedBars || []).join(', ') || '-')} |
      lastProcessedBar ${lastStepResult.lastProcessedBar ?? '-'}
    </div>
  `;
}

function renderRiskMeter(label, currentValue, maxValue, unit) {
  const pct = maxValue > 0 ? Math.min(100, (currentValue / maxValue) * 100) : 0;
  const tone = pct < 50 ? 'green' : pct < 80 ? 'yellow' : 'red';
  return `
    <div class="meter ${tone}">
      <div class="meter-label">${label}</div>
      <div class="meter-bar"><div class="meter-fill" style="width:${pct}%"></div></div>
      <div class="meter-value">${Number(currentValue).toFixed(2)}${unit} / ${Number(maxValue).toFixed(2)}${unit}</div>
    </div>
  `;
}

function renderUnavailableSection(title, reason) {
  return `
    <section class="paper-section">
      <div class="paper-section-header">
        <h4>${escapeHtml(title)}</h4>
        <span class="paper-badge is-danger">Unavailable</span>
      </div>
      <div class="paper-error">${escapeHtml(reason || 'Unavailable')}</div>
    </section>
  `;
}

function wireLegacyExecutionControls() {
  on('#v2-start-btn', 'click', async () => {
    await agentSvc.start();
    await refreshAgentStatus().catch(() => null);
    renderExecution(true).catch((err) => console.warn('[exec] start refresh failed:', err));
  });
  on('#v2-stop-btn', 'click', async () => {
    await agentSvc.stop();
    await refreshAgentStatus().catch(() => null);
    renderExecution(true).catch((err) => console.warn('[exec] stop refresh failed:', err));
  });
  on('#v2-revive-btn', 'click', async () => {
    await agentSvc.revive();
    await refreshAgentStatus().catch(() => null);
    renderExecution(true).catch((err) => console.warn('[exec] revive refresh failed:', err));
  });
  on('#v2-scan-btn', 'click', async () => {
    await agentSvc.getSignals();
    await refreshAgentStatus().catch(() => null);
    renderExecution(true).catch((err) => console.warn('[exec] signal refresh failed:', err));
  });
  on('#v2-paper-btn', 'click', async () => {
    await agentSvc.setConfig({ mode: 'paper' });
    await refreshAgentStatus().catch(() => null);
    renderExecution(true).catch((err) => console.warn('[exec] mode refresh failed:', err));
  });
  on('#v2-live-btn', 'click', async () => {
    await agentSvc.setConfig({ mode: 'live' });
    await refreshAgentStatus().catch(() => null);
    renderExecution(true).catch((err) => console.warn('[exec] mode refresh failed:', err));
  });
}

function wirePaperExecutionControls() {
  on('#v2-paper-step-btn', 'click', async () => {
    const form = $('#v2-paper-step-form');
    if (!(form instanceof HTMLFormElement)) return;
    await runPaperAction(setPaperExecutionStepping, async () => {
      const result = await paperExecSvc.stepPaperExecution(readStepPayload(form));
      setPaperExecutionLastStep(result);
      setPaperExecutionState(result.state);
      if (result.state?.config) setPaperExecutionConfig(result.state.config);
      clearPaperExecutionError();
    });
    renderExecution(true).catch((err) => renderTabError('execution', err));
  });

  on('#v2-paper-step-form', 'submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    await runPaperAction(setPaperExecutionStepping, async () => {
      const result = await paperExecSvc.stepPaperExecution(readStepPayload(form));
      setPaperExecutionLastStep(result);
      setPaperExecutionState(result.state);
      if (result.state?.config) setPaperExecutionConfig(result.state.config);
      clearPaperExecutionError();
    });
    renderExecution(true).catch((err) => renderTabError('execution', err));
  });

  on('#v2-paper-reset-btn', 'click', async () => {
    await runPaperAction(setPaperExecutionResetting, async () => {
      const state = await paperExecSvc.resetPaperExecution();
      setPaperExecutionState(state);
      if (state?.config) setPaperExecutionConfig(state.config);
      setPaperExecutionLastStep(null);
      clearPaperExecutionError();
    });
    renderExecution(true).catch((err) => renderTabError('execution', err));
  });

  on('#v2-paper-kill-on', 'click', async () => {
    await runPaperAction(setPaperExecutionKillSwitchUpdating, async () => {
      const state = await paperExecSvc.setPaperKillSwitch({ blocked: true, reason: 'manual_frontend_toggle' });
      setPaperExecutionState(state);
      if (state?.config) setPaperExecutionConfig(state.config);
      clearPaperExecutionError();
    });
    renderExecution(true).catch((err) => renderTabError('execution', err));
  });

  on('#v2-paper-kill-off', 'click', async () => {
    await runPaperAction(setPaperExecutionKillSwitchUpdating, async () => {
      const state = await paperExecSvc.setPaperKillSwitch({ blocked: false, reason: '' });
      setPaperExecutionState(state);
      if (state?.config) setPaperExecutionConfig(state.config);
      clearPaperExecutionError();
    });
    renderExecution(true).catch((err) => renderTabError('execution', err));
  });
}

function wireLegacyRiskForm() {
  on('#v2-risk-form', 'submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    const data = Object.fromEntries(new FormData(form));
    Object.keys(data).forEach((key) => {
      data[key] = Number(data[key]);
    });
    await riskSvc.setRiskLimits(data);
    renderRisk().catch((err) => renderTabError('risk', err));
  });
}

function wirePaperRiskForm() {
  on('#v2-paper-risk-form', 'submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    const formData = new FormData(form);
    const payload = {
      risk_per_trade: Number(formData.get('risk_per_trade')),
      max_concurrent_positions: Number(formData.get('max_concurrent_positions')),
      max_positions_per_symbol: Number(formData.get('max_positions_per_symbol')),
      max_total_exposure: Number(formData.get('max_total_exposure')),
      max_daily_loss: Number(formData.get('max_daily_loss')),
      max_consecutive_losses: Number(formData.get('max_consecutive_losses')),
      cancel_after_bars: Number(formData.get('cancel_after_bars')),
      cooldown_bars_after_loss: Number(formData.get('cooldown_bars_after_loss')),
      starting_equity: Number(formData.get('starting_equity')),
      allow_multiple_same_direction_per_symbol: formData.get('allow_multiple_same_direction_per_symbol') === 'on',
    };

    await runPaperAction(setPaperExecutionLoadingConfig, async () => {
      const config = await paperExecSvc.setPaperExecutionConfig(payload);
      setPaperExecutionConfig(config);
      if (paperExecutionState.state) {
        setPaperExecutionState({ ...paperExecutionState.state, config });
      }
      clearPaperExecutionError();
    });
    renderRisk(true).catch((err) => renderTabError('risk', err));
  });
}

function wireOpsForms() {
  on('#v2-cfg-form', 'submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    const formData = new FormData(form);
    const body = {
      timeframe: formData.get('timeframe'),
      symbols: String(formData.get('symbols') || '')
        .split(',')
        .map((symbol) => symbol.trim())
        .filter(Boolean),
      tick_interval: Number(formData.get('tick_interval')),
      max_position_pct: Number(formData.get('max_position_pct')),
      max_positions: Number(formData.get('max_positions')),
    };
    try {
      await agentSvc.setStrategyConfig(body);
      setStatusMsg('Config applied');
      renderOps().catch((err) => renderTabError('ops', err));
    } catch (err) {
      setStatusMsg(`Config failed: ${safeErrorMessage(err)}`);
    }
  });

  on('#v2-params-form', 'submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    const formData = new FormData(form);
    const params = {};
    formData.forEach((value, key) => {
      params[key] = Number(value);
    });
    try {
      await agentSvc.setStrategyParams(params);
      setStatusMsg('Params saved');
    } catch (err) {
      setStatusMsg(`Params failed: ${safeErrorMessage(err)}`);
    }
  });

  on('#v2-preset-load', 'click', async () => {
    const name = $('#v2-preset-select')?.value;
    if (!name) return;
    try {
      await agentSvc.loadPreset(name);
      setPresetStatus(`Loaded: ${name}`);
      renderOps().catch((err) => renderTabError('ops', err));
    } catch (err) {
      setPresetStatus(`Load failed: ${safeErrorMessage(err)}`);
    }
  });

  on('#v2-preset-delete', 'click', async () => {
    const name = $('#v2-preset-select')?.value;
    if (!name) return;
    try {
      await agentSvc.deletePreset(name);
      setPresetStatus(`Deleted: ${name}`);
      renderOps().catch((err) => renderTabError('ops', err));
    } catch (err) {
      setPresetStatus(`Delete failed: ${safeErrorMessage(err)}`);
    }
  });

  on('#v2-preset-save', 'click', async () => {
    const name = $('#v2-preset-name')?.value.trim();
    if (!name) {
      setPresetStatus('Enter a name');
      return;
    }
    try {
      await agentSvc.savePreset(name);
      setPresetStatus(`Saved: ${name}`);
      renderOps().catch((err) => renderTabError('ops', err));
    } catch (err) {
      setPresetStatus(`Save failed: ${safeErrorMessage(err)}`);
    }
  });

  on('#v2-okx-form', 'submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    const formData = new FormData(form);
    const apiKey = String(formData.get('api_key') || '').trim();
    const secret = String(formData.get('secret') || '').trim();
    const passphrase = String(formData.get('passphrase') || '').trim();
    if (!apiKey || !secret || !passphrase) {
      setStatusMsg('All 3 OKX fields required');
      return;
    }
    try {
      const response = await execSvc.setOkxKeys(apiKey, secret, passphrase);
      setStatusMsg(response.ok ? 'OKX verified' : `OKX failed: ${response.reason || ''}`);
      renderOps().catch((err) => renderTabError('ops', err));
    } catch (err) {
      setStatusMsg(`OKX request failed: ${safeErrorMessage(err)}`);
    }
  });

  on('#v2-tg-form', 'submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    const formData = new FormData(form);
    const body = {
      bot_token: formData.get('bot_token'),
      chat_id: formData.get('chat_id'),
      notify_signals: formData.get('notify_signals') === 'on',
      notify_fills: formData.get('notify_fills') === 'on',
      notify_errors: formData.get('notify_errors') === 'on',
      notify_daily: formData.get('notify_daily') === 'on',
    };
    try {
      const response = await opsSvc.setTelegramConfig(body);
      setStatusMsg(response.ok ? 'Telegram test message sent' : `Telegram failed: ${response.reason || ''}`);
    } catch (err) {
      setStatusMsg(`Telegram failed: ${safeErrorMessage(err)}`);
    }
  });

  on('#v2-healer-trigger', 'click', async () => {
    await opsSvc.triggerHealer();
    renderOps().catch((err) => renderTabError('ops', err));
  });
}

function readStepPayload(form) {
  const formData = new FormData(form);
  const symbol = String(formData.get('symbol') || marketState.currentSymbol || 'HYPEUSDT').trim();
  const interval = String(formData.get('interval') || marketState.currentInterval || '4h').trim();
  const barIndexRaw = String(formData.get('bar_index') || '').trim();
  const analysisBarsRaw = String(formData.get('analysis_bars') || '500').trim();
  const payload = {
    symbol,
    interval,
    analysis_bars: Number(analysisBarsRaw || 500),
  };
  if (barIndexRaw !== '') {
    payload.bar_index = Number(barIndexRaw);
  }
  return payload;
}

async function runPaperAction(setter, fn) {
  if (isPaperExecutionBusy()) return;
  setter(true);
  renderActive(true).catch((err) => console.warn('[exec] paper mutation pre-render failed:', err));
  try {
    await fn();
  } catch (err) {
    setPaperExecutionError(safeErrorMessage(err));
  } finally {
    setter(false);
    renderActive(true).catch((err) => console.warn('[exec] paper mutation post-render failed:', err));
  }
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

function setStatusMsg(message) {
  console.log('[ops]', message);
}

function setPresetStatus(message) {
  const element = $('#v2-preset-status');
  if (element) element.textContent = message;
}
