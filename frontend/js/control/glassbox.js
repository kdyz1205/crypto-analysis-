// frontend/js/control/glassbox.js — agent timeline (consumes SSE events)

import { $, setHtml, esc } from '../util/dom.js';
import { subscribe } from '../util/events.js';

const timeline = [];
const MAX = 50;

const ICONS = {
  'signal.detected': '🎯',
  'signal.validated': '✅',
  'signal.blocked': '🚫',
  'signal.expired': '⏰',
  'order.submitted': '📤',
  'order.filled': '✔️',
  'order.rejected': '❌',
  'position.opened': '🟢',
  'position.closed': '⚫',
  'position.sl_hit': '🛑',
  'position.tp_hit': '🎯',
  'risk.limit.hit': '⚠️',
  'risk.cooldown.started': '⏸️',
  'agent.regime.changed': '🔄',
  'agent.started': '🚀',
  'agent.stopped': '🛑',
  'summary.daily': '📊',
};

function formatEntry(type, data) {
  const icon = ICONS[type] || '•';
  const time = new Date(data.ts || Date.now()).toLocaleTimeString();
  const p = data.payload || data;
  let text = type;

  if (type === 'signal.detected') {
    // SAFE: text is escaped at the consumer (formatEntry return)
    text = `${p.symbol} ${p.side?.toUpperCase()} conf=${Math.round((p.confidence || 0) * 100)}% — ${p.reason || ''}`;
  } else if (type === 'signal.blocked') {
    text = `${p.symbol} ${p.side?.toUpperCase()} blocked: ${(p.block_reasons || []).join('; ')}`;
  } else if (type === 'position.opened') {
    text = `Opened ${p.symbol} ${p.side?.toUpperCase()} size=$${p.size_usd?.toFixed(0) ?? '?'}`;
  } else if (type === 'position.closed') {
    const pct = p.pnl_pct?.toFixed(2) ?? '?';
    const usd = p.pnl_usd?.toFixed(2) ?? '?';
    // SAFE: text is escaped at the consumer (formatEntry return)
    text = `Closed ${p.symbol} P&L ${pct}% ($${usd}) — ${p.reason || ''}`;
  } else if (type === 'agent.regime.changed') {
    text = `Regime ${p.from} → ${p.to} (${(p.confidence || 0).toFixed(2)})`;
  } else if (type === 'agent.started') {
    text = `Agent started (${p.mode}, gen ${p.generation}, $${p.equity?.toFixed(2)})`;
  } else if (type === 'risk.limit.hit') {
    text = `Risk limit ${p.limit}: ${p.current} / ${p.max}`;
  } else if (type === 'summary.daily') {
    text = `Daily: equity $${p.equity?.toFixed(2)}, daily PnL $${p.daily_pnl?.toFixed(2)}`;
  } else if (p.symbol) {
    text = `${p.symbol} ${JSON.stringify(p).slice(0, 80)}`;
  }

  // Escape `text` once at the consumer — its components (p.reason etc)
  // are server-controlled but rendered into innerHTML via setHtml below.
  return `<div class="glass-row"><span class="glass-icon">${icon}</span><span class="glass-time">${esc(time)}</span><span class="glass-text">${esc(text)}</span></div>`;
}

function render() {
  const container = $('#v2-glassbox');
  if (!container) return;
  if (timeline.length === 0) {
    setHtml(container, '<div class="glass-empty">Waiting for agent events...</div>');
    return;
  }
  setHtml(container, timeline.slice().reverse().map((e) => formatEntry(e.type, e.data)).join(''));
}

function addEntry(type) {
  return (data) => {
    timeline.push({ type, data });
    if (timeline.length > MAX) timeline.shift();
    render();
  };
}

export function initGlassbox() {
  for (const t of Object.keys(ICONS)) {
    subscribe(t, addEntry(t));
  }
  render();
}
