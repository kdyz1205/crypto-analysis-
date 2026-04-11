import { fetchJson } from '../util/fetch.js';

export async function getLiveExecutionStatus() {
  const response = await fetchJson('/api/live-execution/status', {
    timeout: 15000,
  });
  return response.status;
}

export async function getLiveAccount(mode = 'live') {
  return fetchJson(`/api/live-execution/account?mode=${encodeURIComponent(mode)}`, {
    timeout: 30000,
  });
}

export async function getLiveExecutionPreflight(params = {}) {
  const search = new URLSearchParams();
  if (params.mode) search.set('mode', params.mode);
  if (params.order_intent_id) search.set('order_intent_id', params.order_intent_id);
  if (params.signal_id) search.set('signal_id', params.signal_id);
  const suffix = search.toString() ? `?${search}` : '';
  const response = await fetchJson(`/api/live-execution/preflight${suffix}`, {
    timeout: 30000,
  });
  return response.preflight;
}

export async function reconcileLiveExecution(payload) {
  const response = await fetchJson('/api/live-execution/reconcile', {
    method: 'POST',
    body: payload,
    timeout: 30000,
  });
  return response.reconciliation;
}

export async function previewLiveExecution(payload) {
  const response = await fetchJson('/api/live-execution/preview', {
    method: 'POST',
    body: payload,
    timeout: 30000,
  });
  return response.result;
}

export async function submitLiveExecution(payload) {
  const response = await fetchJson('/api/live-execution/submit', {
    method: 'POST',
    body: payload,
    timeout: 30000,
  });
  return response.result;
}

export async function closeLiveExecution(payload) {
  const response = await fetchJson('/api/live-execution/close', {
    method: 'POST',
    body: payload,
    timeout: 30000,
  });
  return response.result;
}
