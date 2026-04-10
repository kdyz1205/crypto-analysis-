import { fetchJson } from '../util/fetch.js';

export const getRuntimeInstances = () => fetchJson('/api/runtime/instances');

export const createRuntimeInstance = (payload) =>
  fetchJson('/api/runtime/instances', { method: 'POST', body: payload });

export const updateRuntimeInstance = (instanceId, payload) =>
  fetchJson(`/api/runtime/instances/${encodeURIComponent(instanceId)}`, { method: 'PATCH', body: payload });

export const deleteRuntimeInstance = (instanceId) =>
  fetchJson(`/api/runtime/instances/${encodeURIComponent(instanceId)}`, { method: 'DELETE' });

export const startRuntimeInstance = (instanceId) =>
  fetchJson(`/api/runtime/instances/${encodeURIComponent(instanceId)}/start`, { method: 'POST' });

export const stopRuntimeInstance = (instanceId) =>
  fetchJson(`/api/runtime/instances/${encodeURIComponent(instanceId)}/stop`, { method: 'POST' });

export const tickRuntimeInstance = (instanceId, payload = {}) =>
  fetchJson(`/api/runtime/instances/${encodeURIComponent(instanceId)}/tick`, { method: 'POST', body: payload, timeout: 30000 });

export const setRuntimeKillSwitch = (instanceId, payload) =>
  fetchJson(`/api/runtime/instances/${encodeURIComponent(instanceId)}/kill-switch`, { method: 'POST', body: payload });

export const getRuntimeEvents = (instanceId = null, limit = 20) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (instanceId) params.set('instance_id', instanceId);
  return fetchJson(`/api/runtime/events?${params}`);
};
