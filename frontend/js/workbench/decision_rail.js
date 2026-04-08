// frontend/js/workbench/decision_rail.js — decision card shown to the right of chart

import { $, setHtml } from '../util/dom.js';
import * as agentSvc from '../services/agent.js';
import * as researchSvc from '../services/research.js';
import { marketState } from '../state/market.js';
import { subscribe } from '../util/events.js';

let pollTimer = null;

async function render() {
  const container = $('#v2-decision-rail');
  if (!container) return;

  try {
    const [status, ribbon] = await Promise.all([
      agentSvc.getStatus(),
      researchSvc.getMaRibbon(marketState.currentSymbol).catch(() => null),
    ]);

    const regime = status.harness?.market_regime || 'unknown';
    const regimeConf = status.harness?.regime_confidence ?? 0;
    const signals = status.last_signals || {};
    const sig = signals[marketState.currentSymbol];

    const ribbonScore = ribbon?.score ?? '—';
    const ribbonTier = ribbon?.tier ?? '—';

    const bias = sig?.action === 'long' ? 'LONG' : sig?.action === 'short' ? 'SHORT' : 'NEUTRAL';
    const biasClass = bias === 'LONG' ? 'pnl-pos' : bias === 'SHORT' ? 'pnl-neg' : '';
    const conf = sig?.confidence ? Math.round(sig.confidence * 100) : '—';
    const blockReason = sig?.blocked ? sig.block_reasons?.join('; ') : null;

    setHtml(container, `
      <div class="dr-card">
        <h3>Decision Rail</h3>
        <div class="dr-row">
          <span class="dr-label">Symbol</span>
          <span class="dr-value">${marketState.currentSymbol}</span>
        </div>
        <div class="dr-row">
          <span class="dr-label">Regime</span>
          <span class="dr-value">${regime} (${Math.round(regimeConf * 100)}%)</span>
        </div>
        <div class="dr-row">
          <span class="dr-label">MA Ribbon</span>
          <span class="dr-value">${ribbonTier} (${ribbonScore}/10)</span>
        </div>
        <div class="dr-row">
          <span class="dr-label">Bias</span>
          <span class="dr-value ${biasClass}"><strong>${bias}</strong></span>
        </div>
        <div class="dr-row">
          <span class="dr-label">Confidence</span>
          <span class="dr-value">${conf}${conf !== '—' ? '%' : ''}</span>
        </div>
        ${sig?.reason ? `<div class="dr-reason">${sig.reason}</div>` : ''}
        ${blockReason ? `<div class="dr-block">⚠ Blocked: ${blockReason}</div>` : ''}
        <div class="dr-row">
          <span class="dr-label">Agent mode</span>
          <span class="dr-value">${status.mode?.toUpperCase() || '—'}</span>
        </div>
        <div class="dr-row">
          <span class="dr-label">Equity</span>
          <span class="dr-value">$${(status.equity ?? 0).toFixed(2)}</span>
        </div>
        <div class="dr-row">
          <span class="dr-label">Open positions</span>
          <span class="dr-value">${Object.keys(status.positions || {}).length}</span>
        </div>
      </div>
    `);
  } catch (err) {
    setHtml(container, `<div class="dr-card"><p>Decision rail unavailable: ${err.message}</p></div>`);
  }
}

export function initDecisionRail() {
  render();
  pollTimer = setInterval(render, 10000);
  subscribe('market.symbol.changed', render);
  subscribe('agent.status.updated', render);
}

export function stopDecisionRail() {
  if (pollTimer) clearInterval(pollTimer);
}
