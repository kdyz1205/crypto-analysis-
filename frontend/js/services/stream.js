// frontend/js/services/stream.js — EventSource wrapper for /api/stream

import { publish } from '../util/events.js';

let source = null;
let reconnectTimer = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_BACKOFF_MS = 3000;
const MAX_BACKOFF_MS = 30000;

const EVENT_TYPES = [
  'connected', 'ping',
  'signal.detected', 'signal.validated', 'signal.blocked', 'signal.expired',
  'order.submitted', 'order.filled', 'order.rejected', 'order.cancelled',
  'position.opened', 'position.closed', 'position.sl_hit', 'position.tp_hit',
  'risk.limit.hit', 'risk.cooldown.started', 'risk.cooldown.ended',
  'agent.started', 'agent.stopped', 'agent.mode.changed',
  'agent.regime.changed', 'agent.error.raised',
  'summary.daily', 'ops.healer.triggered',
  // 2026-04-23: private Bitget WS push events (see server/bitget_private_ws.py)
  'order.ws_update', 'account.ws_update', 'position.ws_update',
];

export function connectStream() {
  if (source) return source;
  try {
    source = new EventSource('/api/stream');
  } catch (err) {
    console.error('[stream] failed to connect:', err);
    return null;
  }

  for (const t of EVENT_TYPES) {
    source.addEventListener(t, (ev) => {
      try {
        const data = ev.data ? JSON.parse(ev.data) : {};
        publish(t, data);
        // 2026-04-23: bridge server-side order/position events → the UI's
        // local 'conditionals.changed' bus so open panels refresh immediately
        // when a fill / cancel comes in over SSE. Before this, frontend only
        // fired 'conditionals.changed' after USER actions (local POST);
        // server-triggered events (SL hit, TP hit, Bitget background-fill)
        // had to wait for the 10s poll to show.
        if (t.startsWith('order.') || t.startsWith('position.')) {
          publish('conditionals.changed', { action: t, data, ts: Date.now() });
        }
      } catch (err) {
        console.warn('[stream] parse error for', t, err);
      }
    });
  }

  source.onopen = () => {
    console.log('[stream] connected');
    reconnectAttempts = 0;  // reset backoff on successful connect
  };
  source.onerror = () => {
    source?.close();
    source = null;
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.warn('[stream] giving up after', reconnectAttempts, 'attempts');
      return;
    }
    // Exponential backoff: 3s × 1.5^attempts, capped at 30s
    const backoff = Math.min(
      BASE_BACKOFF_MS * Math.pow(1.5, reconnectAttempts),
      MAX_BACKOFF_MS,
    );
    reconnectAttempts += 1;
    console.warn(`[stream] disconnected (attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS}), retry in ${Math.round(backoff/1000)}s`);
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectStream();
      }, backoff);
    }
  };

  return source;
}

export function disconnectStream() {
  if (source) {
    source.close();
    source = null;
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}
