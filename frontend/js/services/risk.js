// frontend/js/services/risk.js
import { fetchJson } from '../util/fetch.js';

export const setRiskLimits = (body) => fetchJson('/api/agent/risk-limits', { method: 'POST', body });
