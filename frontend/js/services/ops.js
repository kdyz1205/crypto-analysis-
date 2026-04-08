// frontend/js/services/ops.js
import { fetchJson } from '../util/fetch.js';

export const getHealth = () => fetchJson('/api/health');
export const setTelegramConfig = (body) => fetchJson('/api/agent/telegram-config', { method: 'POST', body });
export const getLogs = (limit = 50, filter = 'agent') =>
  fetchJson(`/api/agent/logs?limit=${limit}&filter=${filter}`);
export const getHealerStatus = () => fetchJson('/api/healer/status');
export const triggerHealer = () => fetchJson('/api/healer/trigger', { method: 'POST' });
export const stopHealer = () => fetchJson('/api/healer/stop', { method: 'POST' });
export const startHealer = () => fetchJson('/api/healer/start', { method: 'POST' });
