import { fetchJson } from '../util/fetch.js';

export function getManualDrawings(symbol, timeframe) {
  const params = new URLSearchParams({ symbol, timeframe });
  // noCache: stale 30s cache showed deleted lines after switching symbols
  return fetchJson(`/api/drawings?${params}`, { noCache: true });
}

export function createManualDrawing(payload) {
  return fetchJson('/api/drawings', {
    method: 'POST',
    body: payload,
  });
}

export function updateManualDrawing(manualLineId, payload) {
  return fetchJson(`/api/drawings/${encodeURIComponent(manualLineId)}`, {
    method: 'PATCH',
    body: payload,
  });
}

export function deleteManualDrawing(manualLineId) {
  return fetchJson(`/api/drawings/${encodeURIComponent(manualLineId)}`, {
    method: 'DELETE',
  });
}

export function clearManualDrawings(symbol, timeframe) {
  const params = new URLSearchParams();
  if (symbol) params.set('symbol', symbol);
  if (timeframe) params.set('timeframe', timeframe);
  return fetchJson(`/api/drawings/clear?${params}`, {
    method: 'POST',
  });
}
