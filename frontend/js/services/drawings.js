import { fetchJson, invalidateCachePrefix } from '../util/fetch.js';

export function getManualDrawings(symbol, timeframe) {
  const params = new URLSearchParams({ symbol, timeframe });
  // noCache: stale 30s cache showed deleted lines after switching symbols
  return fetchJson(`/api/drawings?${params}`, { noCache: true });
}

export function getAllManualDrawings() {
  // All drawings across every symbol / TF. Used by the sidebar's
  // "我的手画线 (全部币种)" grouped view.
  return fetchJson('/api/drawings/all', { noCache: true });
}

export function createManualDrawing(payload) {
  return fetchJson('/api/drawings', {
    method: 'POST',
    body: payload,
  }).then((data) => {
    invalidateCachePrefix('/api/drawings');
    return data;
  });
}

export function updateManualDrawing(manualLineId, payload) {
  return fetchJson(`/api/drawings/${encodeURIComponent(manualLineId)}`, {
    method: 'PATCH',
    body: payload,
  }).then((data) => {
    invalidateCachePrefix('/api/drawings');
    return data;
  });
}

export async function deleteManualDrawing(manualLineId) {
  try {
    const data = await fetchJson(`/api/drawings/${encodeURIComponent(manualLineId)}`, {
      method: 'DELETE',
    });
    invalidateCachePrefix('/api/drawings');
    return data;
  } catch (err) {
    if (err?.status === 404) {
      invalidateCachePrefix('/api/drawings');
      return { removed: 0, already_deleted: true };
    }
    throw err;
  }
}

export function clearManualDrawings(symbol, timeframe) {
  const params = new URLSearchParams();
  if (symbol) params.set('symbol', symbol);
  if (timeframe) params.set('timeframe', timeframe);
  return fetchJson(`/api/drawings/clear?${params}`, {
    method: 'POST',
  }).then((data) => {
    invalidateCachePrefix('/api/drawings');
    return data;
  });
}
