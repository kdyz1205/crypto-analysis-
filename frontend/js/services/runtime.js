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

export const getStrategyCatalog = () => fetchJson('/api/runtime/catalog');
export const getLeaderboard = (limit = 20) => fetchJson(`/api/runtime/leaderboard?limit=${limit}`);
export const startEvolution = () => fetchJson('/api/runtime/leaderboard/start', { method: 'POST' });
export const stopEvolution = () => fetchJson('/api/runtime/leaderboard/stop', { method: 'POST' });
export const copyVariant = (variantId, liveMode = 'disabled') => fetchJson(`/api/runtime/leaderboard/${encodeURIComponent(variantId)}/copy?live_mode=${liveMode}`, { method: 'POST' });

export const launchFromCatalog = (templateId, symbol, timeframe, liveMode = 'disabled', equity = 10000) =>
  fetchJson(`/api/runtime/catalog/${encodeURIComponent(templateId)}/launch?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}&live_mode=${encodeURIComponent(liveMode)}&starting_equity=${equity}`, { method: 'POST' });

export const getRuntimeEvents = (instanceId = null, limit = 20) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (instanceId) params.set('instance_id', instanceId);
  return fetchJson(`/api/runtime/events?${params}`);
};
