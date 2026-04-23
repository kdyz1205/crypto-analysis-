// frontend/js/workbench/overlays/plan_order_overlay.js
//
// Horizontal price-line markers on the chart for every TRIGGERED plan
// order (what Bitget calls 计划委托) and every FILLED position for the
// currently-viewed symbol + timeframe.
//
// Per cond we draw three lines:
//   - Entry  (触发 / 入场)  blue solid,  label "入场 @ $price"
//   - Stop   (止损)         red  dashed, label "止损 @ $price"
//   - Target (止盈)         green dashed, label "止盈 @ $price"
//
// When cond.status is 'triggered' (plan sitting on Bitget waiting for
// trigger) we prefix labels with "计划" so the user can tell waiting
// from live at a glance. When status is 'filled' (position open) we
// prefix with "持仓".
//
// Uses lightweight-charts native `candleSeries.createPriceLine()` —
// these render as horizontal lines with right-edge price tags and
// follow price-scale zoom/log changes automatically. They are not
// chart series so they don't show up in the legend, and they auto-
// position on the right axis rather than tracking candle time.
//
// Added 2026-04-23 after user reported flying blind: "挂了单之后没
// 清楚标注入场/止损/止盈,要像坐标一样跟着."

import { listConditionals } from '../../services/conditionals.js';
import { marketState } from '../../state/market.js';

// One map per symbol:interval, so the overlay is still correct when
// user switches symbols. Each entry: cond_id -> [priceLine refs].
const _linesByKey = new Map();
let _candleSeriesRef = null;
let _lastPullMs = 0;
let _pollTimer = null;
let _visible = true;

const COLORS = {
  entryPlan:  { color: '#60a5fa', style: 0 }, // blue solid
  entryFill:  { color: '#3b82f6', style: 0 }, // stronger blue
  stopPlan:   { color: '#ef4444', style: 2 }, // red dashed
  stopFill:   { color: '#dc2626', style: 0 }, // red solid when position live
  tpPlan:     { color: '#22c55e', style: 2 }, // green dashed
  tpFill:     { color: '#16a34a', style: 0 },
};

function _key(symbol, interval) {
  return `${symbol || ''}:${interval || ''}`;
}

function _fmtPrice(p) {
  const v = Number(p);
  if (!Number.isFinite(v) || v <= 0) return '—';
  if (v >= 1000) return v.toFixed(2);
  if (v >= 100)  return v.toFixed(3);
  if (v >= 10)   return v.toFixed(3);
  if (v >= 1)    return v.toFixed(4);
  if (v >= 0.1)  return v.toFixed(5);
  if (v >= 0.01) return v.toFixed(6);
  return v.toFixed(7);
}

function _fmtQty(q) {
  const v = Number(q);
  if (!Number.isFinite(v) || v <= 0) return '';
  if (v >= 100) return v.toFixed(1);
  if (v >= 1)   return v.toFixed(2);
  return v.toFixed(4);
}

/** Clear all price-line markers for a given key (or everything if null). */
export function clearPlanOverlay(key = null) {
  if (!_candleSeriesRef) return;
  if (key == null) {
    for (const map of _linesByKey.values()) {
      for (const refs of map.values()) {
        for (const line of refs) {
          try { _candleSeriesRef.removePriceLine(line); } catch {}
        }
      }
    }
    _linesByKey.clear();
    return;
  }
  const map = _linesByKey.get(key);
  if (!map) return;
  for (const refs of map.values()) {
    for (const line of refs) {
      try { _candleSeriesRef.removePriceLine(line); } catch {}
    }
  }
  _linesByKey.delete(key);
}

/** Draw/refresh plan-order + position markers for the currently-loaded
 *  symbol & interval. Safe to call repeatedly — stale lines are removed
 *  before new ones are added. */
export async function refreshPlanOverlay(chart, candleSeries, opts = {}) {
  if (!candleSeries) return;
  _candleSeriesRef = candleSeries;

  const symbol = (opts.symbol || marketState.currentSymbol || '').toUpperCase();
  const interval = opts.interval || marketState.currentInterval || '';
  if (!symbol) return;

  const key = _key(symbol, interval);
  // Wipe previous lines for THIS key only
  clearPlanOverlay(key);
  if (!_visible) return;

  let conds = [];
  try {
    const resp = await listConditionals('all', symbol);
    conds = resp?.conditionals || resp?.data || resp || [];
  } catch (err) {
    console.warn('[plan_overlay] listConditionals failed:', err);
    return;
  }

  const active = conds.filter((c) => {
    if (c.symbol?.toUpperCase() !== symbol) return false;
    if (interval && c.timeframe && c.timeframe !== interval) return false;
    return c.status === 'triggered' || c.status === 'filled';
  });
  if (active.length === 0) return;

  const map = new Map();

  for (const c of active) {
    const isFilled = c.status === 'filled';
    const dir = c.order?.direction || 'long';
    const dirTag = dir === 'long' ? '多' : '空';
    const statusTag = isFilled ? '持仓' : '计划';
    const qty = _fmtQty(c.fill_qty);

    const entryPrice = Number(c.fill_price);
    const stopPrice = Number(c.order?.stop_points != null && c.fill_price != null
      ? (dir === 'long'
          ? Number(c.fill_price) - Number(c.order.stop_points)
          : Number(c.fill_price) + Number(c.order.stop_points))
      : NaN);
    const tpPrice = Number(c.order?.tp_price);

    const refs = [];
    // ── Entry ────
    if (Number.isFinite(entryPrice) && entryPrice > 0) {
      const style = isFilled ? COLORS.entryFill : COLORS.entryPlan;
      const title = `${statusTag} ${dirTag} @ ${_fmtPrice(entryPrice)}${qty ? ` (${qty})` : ''}`;
      try {
        const line = candleSeries.createPriceLine({
          price: entryPrice,
          color: style.color,
          lineStyle: style.style,
          lineWidth: 1.5,
          axisLabelVisible: true,
          title,
        });
        refs.push(line);
      } catch (err) { console.warn('[plan_overlay] entry line err:', err); }
    }
    // ── Stop ────
    if (Number.isFinite(stopPrice) && stopPrice > 0) {
      const style = isFilled ? COLORS.stopFill : COLORS.stopPlan;
      const title = `${statusTag}止损 ${_fmtPrice(stopPrice)}`;
      try {
        const line = candleSeries.createPriceLine({
          price: stopPrice,
          color: style.color,
          lineStyle: style.style,
          lineWidth: 1.5,
          axisLabelVisible: true,
          title,
        });
        refs.push(line);
      } catch (err) { console.warn('[plan_overlay] stop line err:', err); }
    }
    // ── Target ────
    if (Number.isFinite(tpPrice) && tpPrice > 0) {
      const style = isFilled ? COLORS.tpFill : COLORS.tpPlan;
      const title = `${statusTag}止盈 ${_fmtPrice(tpPrice)}`;
      try {
        const line = candleSeries.createPriceLine({
          price: tpPrice,
          color: style.color,
          lineStyle: style.style,
          lineWidth: 1.5,
          axisLabelVisible: true,
          title,
        });
        refs.push(line);
      } catch (err) { console.warn('[plan_overlay] tp line err:', err); }
    }

    if (refs.length > 0) map.set(c.conditional_id, refs);
  }

  _linesByKey.set(key, map);
  _lastPullMs = Date.now();
  // Test hook: expose line metadata so Playwright can assert without
  // trying to read lightweight-charts canvas. Only written, never read
  // by real code.
  try {
    const dump = [];
    for (const [k, m] of _linesByKey) {
      for (const [cid, refs] of m) {
        dump.push({ key: k, cond_id: cid, line_count: refs.length });
      }
    }
    window.__planOverlayProbe = { updated_at: _lastPullMs, rows: dump };
  } catch {}
}

/** Start a polling timer that refreshes markers every N seconds so replan
 *  updates propagate without a full chart reload. */
export function startPlanOverlayPoll(chart, candleSeries, intervalMs = 10000) {
  if (_pollTimer) clearInterval(_pollTimer);
  _candleSeriesRef = candleSeries;
  _pollTimer = setInterval(() => {
    refreshPlanOverlay(chart, candleSeries).catch(() => {});
  }, intervalMs);
}

export function stopPlanOverlayPoll() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

/** Force-refresh (e.g. when user just placed/cancelled an order). */
export function forcePlanOverlayRefresh(chart, candleSeries) {
  return refreshPlanOverlay(chart, candleSeries);
}

export function setPlanOverlayVisible(on) {
  _visible = !!on;
  if (!on) clearPlanOverlay();
}
