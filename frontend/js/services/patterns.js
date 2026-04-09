// frontend/js/services/patterns.js
import { fetchJson } from '../util/fetch.js';

export const getPatterns = (symbol, interval, days = 30, mode = 'full', endTime = null, requestOptions = {}) => {
  const params = new URLSearchParams({ symbol, interval, days: String(days), mode });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/patterns?${params}`, requestOptions);
};

export const getPatternStatsBacktest = (symbol, interval, days = 365) =>
  fetchJson(`/api/pattern-stats/backtest?symbol=${symbol}&interval=${interval}&days=${days}`);

export const getPatternFeatures = (symbol, interval, endTime = null) => {
  const params = new URLSearchParams({ symbol, interval });
  if (endTime) params.set('end_time', endTime);
  return fetchJson(`/api/pattern-stats/features?${params}`);
};

export const getCurrentVsHistory = (symbol, interval, days = 365, epsilon = 0.35) =>
  fetchJson(`/api/pattern-stats/current-vs-history?symbol=${symbol}&interval=${interval}&days=${days}&epsilon=${epsilon}`);

export const getLineSimilar = (symbol, interval, days, x1, y1, x2, y2, epsilon = 0.4, maxLines = 15) => {
  const params = new URLSearchParams({
    symbol, interval, days: String(days),
    x1: String(x1), y1: String(y1), x2: String(x2), y2: String(y2),
    epsilon: String(epsilon), max_lines: String(maxLines),
  });
  return fetchJson(`/api/pattern-stats/line-similar?${params}`);
};
