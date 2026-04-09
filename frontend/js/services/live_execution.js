import { fetchJson } from '../util/fetch.js';

export async function getLiveExecutionStatus() {
  const response = await fetchJson('/api/live-execution/status');
  return response.status;
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
