// frontend/js/services/execution.js
import { fetchJson } from '../util/fetch.js';

export const setOkxKeys = (api_key, secret, passphrase) =>
  fetchJson('/api/agent/okx-keys', { method: 'POST', body: { api_key, secret, passphrase } });

export const getOkxStatus = () => fetchJson('/api/agent/okx-status');
