// frontend/js/services/conditionals.js
// Thin wrapper over /api/conditionals/* + /api/drawings/manual/analyze

import { fetchJson } from '../util/fetch.js';

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
    { method: 'POST', timeout: 10000 }
  );
}

export async function deleteConditional(id) {
  return fetchJson(`/api/conditionals/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    timeout: 10000,
  });
}

/**
 * Place a REAL Bitget plan order derived from a drawn line.
 * The backend keeps a local conditional record so the watcher can move
 * the exchange-side trigger as the sloped line projection changes.
 */
export async function placeLineOrder(payload) {
  return fetchJson('/api/drawings/manual/place-line-order', {
    method: 'POST',
    body: payload,
    timeout: 12000,
  });
}
