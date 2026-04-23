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
      // Backend 2026-04-22: local cancel is synchronous (<100ms),
      // Bitget plan cancels run in a background task so this endpoint
      // returns immediately. 10s is plenty for the happy path.
      timeout: 10000,
    });
    invalidateCachePrefix('/api/drawings');
    return data;
  } catch (err) {
    if (err?.status === 404) {
      invalidateCachePrefix('/api/drawings');
      return { removed: 0, already_deleted: true };
    }
    // Abort/timeout: the local delete MAY have succeeded on the server
    // even if our response read aborted. Tell the user to refresh and
    // manually check Bitget app for any orphan plan orders.
    const msg = String(err?.message || err || '');
    if (/abort|timeout|signal/i.test(msg)) {
      err.userHint = '请求超时:本地可能已删,但 Bitget 挂单需要你手动去 app 核对有没有残留';
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
    timeout: 30000,   // may cascade-cancel many conds on bulk clear
  }).then((data) => {
    invalidateCachePrefix('/api/drawings');
    return data;
  });
}
