// frontend/js/services/chat.js
import { fetchJson } from '../util/fetch.js';

export const sendChat = (message, sessionId = 'default', model = null) =>
  fetchJson('/api/chat', { method: 'POST', body: { message, session_id: sessionId, model } });

export const getModels = () => fetchJson('/api/chat/models');
export const getHistory = (sessionId = 'default') =>
  fetchJson(`/api/chat/history?session_id=${sessionId}`);
export const clearHistory = (sessionId = 'default') =>
  fetchJson(`/api/chat/clear?session_id=${sessionId}`, { method: 'POST' });
