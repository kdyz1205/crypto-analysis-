import { fetchJson } from '../util/fetch.js';

function buildParams(symbol, interval, options = {}) {
  const params = new URLSearchParams({
    symbol,
    interval,
  });
  if (options.endTime) params.set('end_time', options.endTime);
  if (options.days != null) params.set('days', String(options.days));
  if (options.analysisBars != null) params.set('analysis_bars', String(options.analysisBars));
  if (options.tail != null) params.set('tail', String(options.tail));
  return params;
}

export const getStrategyConfig = (symbol, interval) => {
  const params = new URLSearchParams({ symbol, interval });
  return fetchJson(`/api/strategy/config?${params}`);
};

export const getStrategySnapshot = (symbol, interval, options = {}) =>
  fetchJson(`/api/strategy/snapshot?${buildParams(symbol, interval, options)}`);

export const getStrategyReplay = (symbol, interval, options = {}) =>
  fetchJson(`/api/strategy/replay?${buildParams(symbol, interval, options)}`);
