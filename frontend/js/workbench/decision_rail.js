// frontend/js/workbench/decision_rail.js
// 4 cards: Market State · Current Setup · Risk Gate · Trade Candidate

import { $, setHtml } from '../util/dom.js';
import { marketState } from '../state/market.js';
import { subscribe } from '../util/events.js';
import { fetchJson } from '../util/fetch.js';

let pollTimer = null;

async function fetchAll(symbol, interval) {
  const [structure, summary, risk, candidates] = await Promise.all([
    fetchJson(`/api/market/structure-summary?symbol=${symbol}&interval=${interval}`).catch(() => null),
    fetchJson('/api/agent/summary').catch(() => null),
    fetchJson('/api/agent/risk-state').catch(() => null),
    fetchJson('/api/agent/signal-candidates').catch(() => null),
  ]);
  return { structure, summary, risk, candidates };
}

function cardMarketState(structure) {
  if (!structure || structure.error) {
    return `<div class="dr-card"><h3>Market State</h3><p class="muted">${structure?.error || 'Loading...'}</p></div>`;
  }
  const trendClass = structure.trend_label === 'UPTREND' ? 'pnl-pos' : structure.trend_label === 'DOWNTREND' ? 'pnl-neg' : '';
  return `
    <div class="dr-card">
      <h3>Market State</h3>
      <div class="dr-row">
        <span class="dr-label">Trend</span>
        <span class="dr-value ${trendClass}"><strong>${structure.trend_label}</strong> ${structure.trend_slope_pct > 0 ? '+' : ''}${structure.trend_slope_pct}%</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">MA Align</span>
        <span class="dr-value">${structure.ma_alignment}</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">Ribbon Score</span>
        <span class="dr-value">${structure.ribbon_score}/${structure.ribbon_max}</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">BB</span>
        <span class="dr-value">${structure.bb_position}${structure.bb_distance_pct != null ? ` (${structure.bb_distance_pct}%)` : ''}</span>
      </div>
      ${structure.nearest_support ? `
        <div class="dr-row">
          <span class="dr-label">Nearest Support</span>
          <span class="dr-value">$${structure.nearest_support} (−${structure.distance_to_support_pct}%)</span>
        </div>
      ` : ''}
      ${structure.nearest_resistance ? `
        <div class="dr-row">
          <span class="dr-label">Nearest Resistance</span>
          <span class="dr-value">$${structure.nearest_resistance} (+${structure.distance_to_resistance_pct}%)</span>
        </div>
      ` : ''}
    </div>
  `;
}

function cardCurrentSetup(candidates, symbol) {
  if (!candidates || !candidates.candidates || candidates.candidates.length === 0) {
    return `
      <div class="dr-card">
        <h3>Current Setup</h3>
        <p class="muted">No active setup on any symbol</p>
      </div>
    `;
  }
  const sigForSym = candidates.candidates.find((c) => c.symbol === symbol) || candidates.candidates[0];
  const biasClass = sigForSym.side === 'long' ? 'pnl-pos' : 'pnl-neg';
  return `
    <div class="dr-card">
      <h3>Current Setup ${sigForSym.symbol !== symbol ? `<span class="muted" style="font-size:10px">(on ${sigForSym.symbol})</span>` : ''}</h3>
      <div class="dr-row">
        <span class="dr-label">Bias</span>
        <span class="dr-value ${biasClass}"><strong>${sigForSym.side?.toUpperCase()}</strong></span>
      </div>
      <div class="dr-row">
        <span class="dr-label">Setup Score</span>
        <span class="dr-value">${sigForSym.setup_score}/100</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">Exec Score</span>
        <span class="dr-value">${sigForSym.execution_score}/100</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">RR</span>
        <span class="dr-value">${sigForSym.rr_estimate || '—'}R</span>
      </div>
      ${sigForSym.reason ? `<div class="dr-reason">${sigForSym.reason.slice(0, 120)}</div>` : ''}
    </div>
  `;
}

function cardRiskGate(risk) {
  if (!risk) return `<div class="dr-card"><h3>Risk Gate</h3><p class="muted">Loading...</p></div>`;
  const stateClass = {
    NORMAL: 'pnl-pos',
    WATCH: '',
    COOLDOWN: '',
    HALTED: 'pnl-neg',
  }[risk.state] || '';

  const m = risk.meters || {};
  const miniMeter = (label, obj) => {
    if (!obj) return '';
    const pct = obj.max > 0 ? Math.min(100, (obj.current / obj.max) * 100) : 0;
    const cls = pct < 50 ? 'green' : pct < 80 ? 'yellow' : 'red';
    return `
      <div class="dr-meter ${cls}">
        <div class="dr-meter-label">${label}: ${obj.current}${obj.unit} / ${obj.max}${obj.unit}</div>
        <div class="dr-meter-bar"><div class="dr-meter-fill" style="width:${pct}%"></div></div>
      </div>
    `;
  };

  return `
    <div class="dr-card">
      <h3>Risk Gate</h3>
      <div class="dr-row">
        <span class="dr-label">State</span>
        <span class="dr-value ${stateClass}"><strong>${risk.state}</strong></span>
      </div>
      ${risk.state_reason ? `<div class="dr-reason">${risk.state_reason}</div>` : ''}
      ${miniMeter('Exposure', m.exposure)}
      ${miniMeter('Daily Loss', m.daily_loss)}
      ${miniMeter('Drawdown', m.drawdown)}
      ${miniMeter('Positions', m.positions)}
      ${risk.cooldown_remaining_sec > 0 ? `
        <div class="dr-row">
          <span class="dr-label">Cooldown</span>
          <span class="dr-value">${risk.cooldown_remaining_sec}s</span>
        </div>
      ` : ''}
    </div>
  `;
}

function cardTradeCandidate(candidates, summary, symbol) {
  if (!candidates || candidates.count === 0) {
    return `
      <div class="dr-card">
        <h3>Trade Candidate</h3>
        <p class="muted">${summary?.last_block_reason ? 'Blocked: ' + summary.last_block_reason : 'No candidate'}</p>
      </div>
    `;
  }
  const c = candidates.candidates.find((x) => x.symbol === symbol) || candidates.candidates[0];
  return `
    <div class="dr-card">
      <h3>Trade Candidate</h3>
      <div class="dr-row">
        <span class="dr-label">Symbol</span>
        <span class="dr-value">${c.symbol} ${c.side?.toUpperCase()}</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">Entry</span>
        <span class="dr-value">$${c.entry}</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">Stop</span>
        <span class="dr-value pnl-neg">$${c.stop}</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">Target</span>
        <span class="dr-value pnl-pos">$${c.target}</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">RR</span>
        <span class="dr-value"><strong>${c.rr_estimate || '—'}R</strong></span>
      </div>
      <div class="dr-row">
        <span class="dr-label">State</span>
        <span class="dr-value ${c.signal_state === 'READY' ? 'pnl-pos' : ''}">${c.signal_state}</span>
      </div>
      ${c.block_reason ? `<div class="dr-block">⚠ ${c.block_reason}</div>` : ''}
    </div>
  `;
}

async function render() {
  const container = $('#v2-decision-rail');
  if (!container) return;

  try {
    const { structure, summary, risk, candidates } = await fetchAll(marketState.currentSymbol, marketState.currentInterval);
    setHtml(container, `
      ${cardMarketState(structure)}
      ${cardCurrentSetup(candidates, marketState.currentSymbol)}
      ${cardRiskGate(risk)}
      ${cardTradeCandidate(candidates, summary, marketState.currentSymbol)}
      <div class="dr-card dr-summary">
        <h3>Agent</h3>
        <div class="dr-row">
          <span class="dr-label">State</span>
          <span class="dr-value ${summary?.runtime_state === 'RUNNING' ? 'pnl-pos' : ''}">${summary?.runtime_state || '—'}</span>
        </div>
        <div class="dr-row">
          <span class="dr-label">Mode</span>
          <span class="dr-value">${summary?.mode?.toUpperCase() || '—'}</span>
        </div>
        <div class="dr-row">
          <span class="dr-label">Equity</span>
          <span class="dr-value">$${(summary?.equity ?? 0).toFixed(2)}</span>
        </div>
        <div class="dr-row">
          <span class="dr-label">Daily PnL</span>
          <span class="dr-value ${summary?.daily_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">$${(summary?.daily_pnl ?? 0).toFixed(2)}</span>
        </div>
        <div class="dr-row">
          <span class="dr-label">Health</span>
          <span class="dr-value">${summary?.health_score ?? '—'}/100</span>
        </div>
      </div>
    `);
  } catch (err) {
    console.error('[decision-rail] render failed:', err);
    setHtml(container, `<div class="dr-card"><h3>Decision Rail</h3><p class="muted">Error: ${err.message}</p></div>`);
  }
}

export function initDecisionRail() {
  render();
  pollTimer = setInterval(render, 8000);
  subscribe('market.symbol.changed', render);
  subscribe('market.interval.changed', render);
}

export function stopDecisionRail() {
  if (pollTimer) clearInterval(pollTimer);
}
