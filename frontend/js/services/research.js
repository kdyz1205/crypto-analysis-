// frontend/js/services/research.js
import { fetchJson } from '../util/fetch.js';

export const runBacktest = (symbol, interval, days = 365, overrides = {}) => {
  const params = new URLSearchParams({ symbol, interval, days: String(days) });
  Object.entries(overrides).forEach(([k, v]) => {
    if (v != null && v !== '') params.set(k, String(v));
  });
  return fetchJson(`/api/backtest?${params}`);
};

export const optimizeBacktest = (symbol, interval, days = 365, objective = 'total_pnl', maxiter = 80, method = 'L-BFGS-B') =>
  fetchJson(
    `/api/backtest/optimize?symbol=${symbol}&interval=${interval}&days=${days}&objective=${objective}&maxiter=${maxiter}&method=${method}`,
    { method: 'POST' }
  );

export const getMaRibbon = (symbol) => fetchJson(`/api/ma-ribbon?symbol=${symbol}`);

export const getMaRibbonBacktest = (symbol, opts = {}) => {
  const params = new URLSearchParams({ symbol });
  Object.entries(opts).forEach(([k, v]) => params.set(k, String(v)));
  return fetchJson(`/api/ma-ribbon/backtest?${params}`);
};
