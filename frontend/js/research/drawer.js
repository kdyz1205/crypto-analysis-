// frontend/js/research/drawer.js — research drawer with backtest + ribbon

import { $, setHtml, on } from '../util/dom.js';
import { uiState, setResearchDrawer, setResearchSubTab } from '../state/ui.js';
import { marketState } from '../state/market.js';
import { subscribe } from '../util/events.js';
import * as researchSvc from '../services/research.js';

let drawer = null;

function build() {
  const existing = $('#v2-research-drawer');
  if (existing) return existing;

  drawer = document.createElement('aside');
  drawer.id = 'v2-research-drawer';
  drawer.className = 'research-drawer hidden';
  drawer.innerHTML = `
    <div class="research-header">
      <span class="research-title">Research Lab</span>
      <nav class="research-tabs">
        <button class="research-tab active" data-tab="backtest">Backtest</button>
        <button class="research-tab" data-tab="ribbon">MA Ribbon</button>
      </nav>
      <button class="research-close" id="v2-research-close">×</button>
    </div>
    <div class="research-body">
      <div class="research-pane" data-pane="backtest">
        <div class="backtest-controls">
          <button class="btn btn-primary" id="v2-run-backtest">Run Backtest</button>
          <span class="muted" id="v2-backtest-status">Click to run on current symbol/timeframe</span>
        </div>
        <div id="v2-backtest-result"></div>
      </div>
      <div class="research-pane hidden" data-pane="ribbon">
        <button class="btn btn-primary" id="v2-run-ribbon">Load MA Ribbon</button>
        <div id="v2-ribbon-result"></div>
      </div>
    </div>
  `;
  document.body.appendChild(drawer);
  return drawer;
}

function switchTab(tab) {
  setResearchSubTab(tab);
  const tabs = drawer.querySelectorAll('.research-tab');
  const panes = drawer.querySelectorAll('.research-pane');
  tabs.forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  panes.forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== tab));
}

async function runBacktest() {
  const { currentSymbol, currentInterval } = marketState;
  const status = $('#v2-backtest-status');
  const result = $('#v2-backtest-result');
  if (status) status.textContent = `Running backtest ${currentSymbol} ${currentInterval}...`;
  try {
    const data = await researchSvc.runBacktest(currentSymbol, currentInterval, 365);
    if (data.error) {
      setHtml(result, `<div class="muted">Error: ${data.error}</div>`);
      return;
    }
    setHtml(result, `
      <div class="backtest-metrics">
        <div class="metric"><div class="metric-label">Trades</div><div class="metric-value">${data.total_trades ?? '—'}</div></div>
        <div class="metric"><div class="metric-label">Wins</div><div class="metric-value pnl-pos">${data.wins ?? '—'}</div></div>
        <div class="metric"><div class="metric-label">Losses</div><div class="metric-value pnl-neg">${data.losses ?? '—'}</div></div>
        <div class="metric"><div class="metric-label">Win Rate</div><div class="metric-value">${(data.win_rate ?? 0).toFixed(1)}%</div></div>
        <div class="metric"><div class="metric-label">Total PnL</div><div class="metric-value ${(data.total_pnl_pct ?? 0) >= 0 ? 'pnl-pos' : 'pnl-neg'}">${(data.total_pnl_pct ?? 0).toFixed(2)}%</div></div>
        <div class="metric"><div class="metric-label">Sharpe</div><div class="metric-value">${(data.sharpe ?? 0).toFixed(2)}</div></div>
        <div class="metric"><div class="metric-label">Max DD</div><div class="metric-value pnl-neg">${(data.max_drawdown_pct ?? 0).toFixed(2)}%</div></div>
        <div class="metric"><div class="metric-label">Avg Trade</div><div class="metric-value">${(data.avg_trade_pct ?? 0).toFixed(2)}%</div></div>
      </div>
    `);
    if (status) status.textContent = `${currentSymbol} ${currentInterval} — done`;
  } catch (err) {
    setHtml(result, `<div class="muted">Error: ${err.message}</div>`);
  }
}

async function runRibbon() {
  const { currentSymbol } = marketState;
  const result = $('#v2-ribbon-result');
  setHtml(result, '<div class="muted">Loading...</div>');
  try {
    const data = await researchSvc.getMaRibbon(currentSymbol);
    if (data.error) {
      setHtml(result, `<div class="muted">Error: ${data.error}</div>`);
      return;
    }
    const tiers = data.timeframes || {};
    setHtml(result, `
      <h4>${currentSymbol} — Score: ${data.score ?? '—'}/10 (${data.tier || '—'})</h4>
      <div class="ribbon-grid">
        ${Object.entries(tiers).map(([tf, info]) => `
          <div class="ribbon-row">
            <span class="ribbon-tf">${tf}</span>
            <span class="ribbon-state ${info.aligned ? 'aligned' : ''}">${info.aligned ? '✅ aligned' : '❌ not aligned'}</span>
            <span class="ribbon-dir">${info.direction || ''}</span>
          </div>
        `).join('')}
      </div>
    `);
  } catch (err) {
    setHtml(result, `<div class="muted">Error: ${err.message}</div>`);
  }
}

export function initResearchDrawer() {
  build();
  on('#v2-research-close', 'click', () => setResearchDrawer(false));
  on('#v2-run-backtest', 'click', runBacktest);
  on('#v2-run-ribbon', 'click', runRibbon);

  drawer.querySelectorAll('.research-tab').forEach((btn) => {
    on(btn, 'click', () => switchTab(btn.dataset.tab));
  });

  subscribe('ui.research.toggled', (open) => {
    if (open) drawer.classList.remove('hidden');
    else drawer.classList.add('hidden');
  });
}

export function openResearchDrawer() { setResearchDrawer(true); }
