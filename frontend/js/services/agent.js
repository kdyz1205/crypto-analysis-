// frontend/js/services/agent.js
import { fetchJson } from '../util/fetch.js';

export const getStatus = (requestOptions = {}) => fetchJson('/api/agent/status', requestOptions);
export const start = () => fetchJson('/api/agent/start', { method: 'POST' });
export const stop = () => fetchJson('/api/agent/stop', { method: 'POST' });
export const revive = () => fetchJson('/api/agent/revive', { method: 'POST' });
export const setConfig = (body) => fetchJson('/api/agent/config', { method: 'POST', body });

export const getSignals = () => fetchJson('/api/agent/signals');
export const getAuditLog = (limit = 50) => fetchJson(`/api/agent/audit-log?limit=${limit}`);
export const getLessons = () => fetchJson('/api/agent/lessons');

export const setStrategyConfig = (body) => fetchJson('/api/agent/strategy-config', { method: 'POST', body });
export const setStrategyParams = (params) => fetchJson('/api/agent/strategy-params', { method: 'POST', body: params });

export const getPresets = () => fetchJson('/api/agent/strategy-presets');
export const savePreset = (name) => fetchJson('/api/agent/strategy-presets/save', { method: 'POST', body: { name } });
export const loadPreset = (name) => fetchJson('/api/agent/strategy-presets/load', { method: 'POST', body: { name } });
export const deletePreset = (name) => fetchJson('/api/agent/strategy-presets/delete', { method: 'POST', body: { name } });
