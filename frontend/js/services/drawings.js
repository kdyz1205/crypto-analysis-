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
      // Current rule: backend REFUSES deletion while active orders still
      // point at this line. Deletion does not cascade-cancel live orders.
      timeout: 35000,
    });
    invalidateCachePrefix('/api/drawings');
    return data;
  } catch (err) {
    if (err?.status === 404) {
      invalidateCachePrefix('/api/drawings');
      return { removed: 0, already_deleted: true };
    }
    // 409 = backend refused to delete the line because active orders still
    // use it as their rationale. Cancel the orders first; line deletion must
    // never cascade-cancel live money by surprise.
    if (err?.status === 409) {
      const detail = err?.detail || err?.body || {};
      const reason = detail.reason || 'active_orders_protect_line';
      err.userHint = `这条线还有活跃挂单 (${reason}) — 先撤掉对应订单,本地线会保留作为下单依据`;
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
    timeout: 30000,   // bulk active-order checks can still take a moment
  }).then((data) => {
    invalidateCachePrefix('/api/drawings');
    return data;
  });
}
