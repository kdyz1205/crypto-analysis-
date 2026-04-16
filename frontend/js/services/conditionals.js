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
 * Place a REAL limit order on Bitget derived from a drawn line.
 * The order goes directly to the exchange orderbook — visible in the
 * Bitget app immediately. Cancel from app or via cancelConditional.
 */
export async function placeLineOrder(payload) {
  return fetchJson('/api/drawings/manual/place-line-order', {
    method: 'POST',
    body: payload,
    timeout: 20000,
  });
}
