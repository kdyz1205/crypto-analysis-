// frontend/js/workbench/decision_rail.js
// 4 cards: Market State · Current Setup · Risk Gate · Trade Candidate

import { $, setHtml } from '../util/dom.js';
import { marketState } from '../state/market.js';
import { subscribe } from '../util/events.js';
import { fetchJson } from '../util/fetch.js';

let pollTimer = null;
const DECISION_RAIL_TIMEOUT_MS = 4000;

async function fetchAll(symbol, interval) {
  const [structure, summary, risk, candidates, snapshot] = await Promise.all([
    fetchJson(`/api/market/structure-summary?symbol=${symbol}&interval=${interval}`, { timeout: DECISION_RAIL_TIMEOUT_MS }).catch(() => null),
    fetchJson('/api/agent/summary', { timeout: DECISION_RAIL_TIMEOUT_MS }).catch(() => null),
    fetchJson('/api/agent/risk-state', { timeout: DECISION_RAIL_TIMEOUT_MS }).catch(() => null),
    fetchJson('/api/agent/signal-candidates', { timeout: DECISION_RAIL_TIMEOUT_MS }).catch(() => null),
    fetchJson(`/api/strategy/snapshot?symbol=${symbol}&interval=${interval}&analysis_bars=500`, { timeout: 8000 }).catch(() => null),
  ]);
  return { structure, summary, risk, candidates, snapshot };
}

function cardMarketState(structure) {
  if (!structure || structure.error) {
    return `<div class="dr-card"><h3>市场状态</h3><p class="muted">${structure?.error || '加载中...'}</p></div>`;
  }
  const trendLabel = structure.trend_label === 'UPTREND' ? '上升趋势' : structure.trend_label === 'DOWNTREND' ? '下降趋势' : '震荡';
  const trendClass = structure.trend_label === 'UPTREND' ? 'pnl-pos' : structure.trend_label === 'DOWNTREND' ? 'pnl-neg' : '';
  return `
    <div class="dr-card">
      <h3>市场状态</h3>
      <div class="dr-row">
        <span class="dr-label">趋势</span>
        <span class="dr-value ${trendClass}"><strong>${trendLabel}</strong> ${structure.trend_slope_pct > 0 ? '+' : ''}${structure.trend_slope_pct}%</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">MA 排列</span>
        <span class="dr-value">${structure.ma_alignment}</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">Ribbon</span>
        <span class="dr-value">${structure.ribbon_score}/${structure.ribbon_max}</span>
      </div>
      ${structure.nearest_support ? `
        <div class="dr-row">
          <span class="dr-label">最近支撑</span>
          <span class="dr-value pnl-pos">$${structure.nearest_support} (${structure.distance_to_support_pct}%)</span>
        </div>
      ` : ''}
      ${structure.nearest_resistance ? `
        <div class="dr-row">
          <span class="dr-label">最近阻力</span>
          <span class="dr-value pnl-neg">$${structure.nearest_resistance} (+${structure.distance_to_resistance_pct}%)</span>
        </div>
      ` : ''}
    </div>
  `;
}

function cardSRZones(snapshot, currentPrice) {
  const zones = snapshot?.snapshot?.horizontal_zones || [];
  const signals = snapshot?.snapshot?.signals || [];
  if (!zones.length) {
    return `<div class="dr-card"><h3>S/R 区域</h3><p class="muted">暂无检测到区域</p></div>`;
  }

  const sup = zones.filter(z => z.side === 'support').slice(0, 2);
  const res = zones.filter(z => z.side === 'resistance').slice(0, 2);
  const price = currentPrice || 0;

  const zoneRow = (z) => {
    const dist = price > 0 ? ((z.price_center - price) / price * 100).toFixed(1) : '?';
    const color = z.side === 'support' ? 'pnl-pos' : 'pnl-neg';
    return `
      <div class="dr-row">
        <span class="dr-label ${color}">${z.side === 'support' ? '支撑' : '阻力'} ${z.touches}t</span>
        <span class="dr-value">$${z.price_center.toFixed(2)} <small>(${dist}%)</small></span>
      </div>
    `;
  };

  const sigRows = signals.slice(0, 2).map(s => `
    <div class="dr-row" style="background:rgba(255,235,59,0.08);margin:2px 0;padding:2px 4px;border-radius:3px;">
      <span class="dr-label" style="color:#fbbf24">${s.direction === 'long' ? '做多' : '做空'}</span>
      <span class="dr-value" style="color:#fbbf24">$${s.entry_price.toFixed(2)} RR=${s.risk_reward.toFixed(1)}</span>
    </div>
  `).join('');

  return `
    <div class="dr-card">
      <h3>S/R 区域 <small style="color:var(--v2-muted)">${zones.length}个</small></h3>
      ${res.map(zoneRow).join('')}
      ${sup.map(zoneRow).join('')}
      ${sigRows || '<div class="dr-row"><span class="dr-label muted">等待入场信号...</span></div>'}
    </div>
  `;
}

function cardCurrentSetup(candidates, symbol) {
  if (!candidates || !candidates.candidates || candidates.candidates.length === 0) {
    return `
      <div class="dr-card">
        <h3>当前机会</h3>
        <p class="muted">暂无交易机会</p>
      </div>
    `;
  }
  const sigForSym = candidates.candidates.find((c) => c.symbol === symbol) || candidates.candidates[0];
  const biasClass = sigForSym.side === 'long' ? 'pnl-pos' : 'pnl-neg';
  return `
    <div class="dr-card">
      <h3>当前机会 ${sigForSym.symbol !== symbol ? `<span class="muted" style="font-size:10px">(${sigForSym.symbol})</span>` : ''}</h3>
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
  if (!risk) return `<div class="dr-card"><h3>风控</h3><p class="muted">加载中...</p></div>`;
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
      <h3>风控</h3>
      <div class="dr-row">
        <span class="dr-label">State</span>
        <span class="dr-value ${stateClass}"><strong>${risk.state}</strong></span>
      </div>
      ${risk.state_reason ? `<div class="dr-reason">${risk.state_reason}</div>` : ''}
      ${miniMeter('敞口', m.exposure)}
      ${miniMeter('日亏损', m.daily_loss)}
      ${miniMeter('回撤', m.drawdown)}
      ${miniMeter('持仓', m.positions)}
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
    const { structure, summary, risk, candidates, snapshot } = await fetchAll(marketState.currentSymbol, marketState.currentInterval);
    const lastPrice = structure?.last_price || marketState.lastCandles?.at?.(-1)?.close || 0;
    setHtml(container, `
      ${cardMarketState(structure)}
      ${cardSRZones(snapshot, lastPrice)}
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

let lastRender = 0;
const MIN_RENDER_GAP = 12000; // 12s minimum between renders

async function safeRender() {
  const now = Date.now();
  if (now - lastRender < MIN_RENDER_GAP) return;
  lastRender = now;
  await render();
}

function paintLoading() {
  const container = $('#v2-decision-rail');
  if (!container) return;
  const loadingCard = (title) => `
    <div class="dr-card dr-loading">
      <h3>${title}</h3>
      <div class="dr-skel-bar"></div>
      <div class="dr-skel-bar short"></div>
      <div class="dr-skel-bar"></div>
    </div>
  `;
  container.innerHTML = [
    loadingCard('市场状态'),
    loadingCard('S/R 区域'),
    loadingCard('风控'),
    loadingCard('交易候选'),
    loadingCard('策略'),
  ].join('');
}

export function initDecisionRail() {
  paintLoading();  // instant skeleton
  pollTimer = setInterval(safeRender, 30000);
  subscribe('market.symbol.changed', safeRender);
  subscribe('market.interval.changed', safeRender);
}

export function stopDecisionRail() {
  if (pollTimer) clearInterval(pollTimer);
}

export async function refreshDecisionRail() {
  await safeRender();
}
