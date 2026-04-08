// frontend/js/services/market.js
import { fetchJson } from '../util/fetch.js';

export const getSymbols = (includeExtended = false) =>
  fetchJson(`/api/symbols${includeExtended ? '?include_extended=true' : ''}`);

export const getSymbolInfo = (symbol) =>
  fetchJson(`/api/symbol-info?symbol=${encodeURIComponent(symbol)}`);

export const getOhlcv = (symbol, interval, days = 30, endTime = null) => {
  const params = new URLSearchParams({ symbol, interval, days: String(days) });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/ohlcv?${params}`);
};

export const getChart = (symbol, interval, days = 365, endTime = null) => {
  const params = new URLSearchParams({ symbol, interval, days: String(days) });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/chart?${params}`);
};

export const getTopVolume = (n = 20) => fetchJson(`/api/top-volume?n=${n}`);

export const getDataInfo = (symbol, interval) =>
  fetchJson(`/api/data-info?symbol=${encodeURIComponent(symbol)}&interval=${interval}`);
