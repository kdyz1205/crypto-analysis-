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
      // Backend 2026-04-22: Bitget cancel is synchronous before local
      // line removal, so the UI never hides a still-live plan order.
      timeout: 35000,
    });
    invalidateCachePrefix('/api/drawings');
    return data;
  } catch (err) {
    if (err?.status === 404) {
      invalidateCachePrefix('/api/drawings');
      return { removed: 0, already_deleted: true };
    }
    // 409 = backend refused because one or more Bitget cancels didn't
    // confirm. Server preserved both the local line AND the cond records
    // with status=triggered, so the sidebar panel will keep showing the
    // pending Bitget order on its next 5s poll. Caller's catch should
    // restore the optimistic line hide. This is the SAFE outcome:
    // "cancel failed = effectively nothing was deleted."
    if (err?.status === 409) {
      const detail = err?.detail || err?.body || {};
      const reason = detail.reason || 'bitget_cancel_unconfirmed';
      err.userHint = `Bitget 撤单未确认 (${reason}) — 本地线和挂单都没删,请去 Bitget app 自己撤单后重试,或追加 ?force=true 强删本地`;
      throw err;
    }
    // Abort/timeout: could be mid-flight. Ambiguous state — safest to
    // refresh and check the app.
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
