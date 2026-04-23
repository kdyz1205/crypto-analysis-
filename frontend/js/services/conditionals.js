// frontend/js/services/conditionals.js
// Thin wrapper over /api/conditionals/* + /api/drawings/manual/analyze

import { fetchJson, invalidateCachePrefix } from '../util/fetch.js';
import { publish } from '../util/events.js';

function notifyConditionalsChanged(action, data) {
  invalidateCachePrefix('/api/conditionals');
  publish('conditionals.changed', { action, data, ts: Date.now() });
}

export async function analyzeDrawing(manualLineId, k = 30) {
  return fetchJson('/api/drawings/manual/analyze', {
    method: 'POST',
    body: { manual_line_id: manualLineId, k },
    timeout: 20000,
  });
}

export async function createConditional(payload) {
  return fetchJson('/api/conditionals', {
    method: 'POST',
    body: payload,
    timeout: 15000,
  }).then((data) => {
    if (data?.ok !== false) notifyConditionalsChanged('create', data);
    return data;
  });
}

export async function listConditionals(status = 'all', symbol = null) {
  const qs = new URLSearchParams({ status });
  if (symbol) qs.set('symbol', symbol);
  // noCache: this is polled every 10s — stale 30s cache would freeze the UI
  return fetchJson(`/api/conditionals?${qs}`, { timeout: 10000, noCache: true });
}

export async function getConditional(id) {
  return fetchJson(`/api/conditionals/${encodeURIComponent(id)}`, {
    timeout: 10000, noCache: true,
  });
}

export async function cancelConditional(id, reason = 'manual_cancel') {
  return fetchJson(
    `/api/conditionals/${encodeURIComponent(id)}/cancel?reason=${encodeURIComponent(reason)}`,
    { method: 'POST', timeout: 35000 }
  ).then((data) => {
    if (data?.ok !== false) notifyConditionalsChanged('cancel', data);
    return data;
  });
}

export async function deleteConditional(id) {
  return fetchJson(`/api/conditionals/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    timeout: 35000,
  }).then((data) => {
    if (data?.ok !== false) notifyConditionalsChanged('delete', data);
    return data;
  });
}

/**
 * Place a REAL Bitget plan order derived from a drawn line.
 * The backend keeps a local conditional record so the watcher can move
 * the exchange-side trigger as the sloped line projection changes.
 */
export async function placeLineOrder(payload) {
  // Backend: set-leverage (~2-3s) + submit_live_plan_entry (~3-5s) +
  // store persist = 5-12s typical, can spike to 20s+ on Bitget slow
  // periods. Short timeout caused "signal is aborted without reason"
  // when order WAS placed successfully — user saw error, Bitget had
  // the order, watcher replanned it. Bump to 30s. User 2026-04-21.
  return fetchJson('/api/drawings/manual/place-line-order', {
    method: 'POST',
    body: payload,
    timeout: 30000,
  }).then((data) => {
    if (data?.ok !== false) notifyConditionalsChanged('place-line-order', data);
    return data;
  });
}
