// frontend/js/services/stream.js — EventSource wrapper for /api/stream

import { publish } from '../util/events.js';

let source = null;
let reconnectTimer = null;

const EVENT_TYPES = [
  'connected', 'ping',
  'signal.detected', 'signal.validated', 'signal.blocked', 'signal.expired',
  'order.submitted', 'order.filled', 'order.rejected', 'order.cancelled',
  'position.opened', 'position.closed', 'position.sl_hit', 'position.tp_hit',
  'risk.limit.hit', 'risk.cooldown.started', 'risk.cooldown.ended',
  'agent.started', 'agent.stopped', 'agent.mode.changed',
  'agent.regime.changed', 'agent.error.raised',
  'summary.daily', 'ops.healer.triggered',
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
      } catch (err) {
        console.warn('[stream] parse error for', t, err);
      }
    });
  }

  source.onopen = () => console.log('[stream] connected');
  source.onerror = () => {
    console.warn('[stream] disconnected, will retry in 3s');
    source?.close();
    source = null;
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectStream();
      }, 3000);
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
