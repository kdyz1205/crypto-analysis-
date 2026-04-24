// frontend/js/workbench/overlays/plan_order_overlay.js
//
// Price-level markers on the chart for ACTIVE trades only:
//   - Pending plan orders (status='triggered' AND Bitget still holds it)
//   - Open positions (status='filled' AND Bitget still shows the position)
//
// Closed / stale conds are hidden. The ground truth is the live account
// snapshot at /api/live-execution/account — local cond status is used
// to pick the draw style, but cross-checked against Bitget so a stale
// `filled` row whose position was closed on the app does NOT pollute
// the chart. (2026-04-23: user screenshot showed 6 phantom shorts for
// HYPE while Bitget held 0 positions.)
//
// Rendering:
//   - A thin horizontal line per price level (via lightweight-charts
//     priceLine, but `axisLabelVisible: false` so the chunky price-axis
//     pill is suppressed).
//   - A small HTML label overlaid on the chart body at (right-edge -
//     axis-width - 4px, priceToCoordinate(price)). This keeps labels
//     inside the candle area per user request — anchored to the exact
//     price coordinate, not the axis column.
//
// Labels stay pinned to the right price during pan / zoom / resize via
// a requestAnimationFrame loop that runs only while ≥ 1 label exists.
//
// Previously: axis-pill labels with big text per cond, 10 s poll, no
// closed-position filter. See git log for that iteration.

import { listConditionals } from '../../services/conditionals.js';
import { marketState } from '../../state/market.js';
import { fetchJson } from '../../util/fetch.js';

// One map per symbol:interval, so the overlay is still correct when
// user switches symbols. Each entry: cond_id -> { lines: [priceLine], labels: [{price, text, color, fg}] }.
const _byKey = new Map();
let _candleSeriesRef = null;
let _chartRef = null;
let _chartContainerRef = null;
let _labelLayer = null;
let _rafHandle = null;
let _lastPullMs = 0;
let _pollTimer = null;
let _visible = true;

// Shared cache so every refresh doesn't re-fetch the account.
let _accountCache = { ts: 0, data: null };
const ACCOUNT_TTL_MS = 20000;

const COLORS = {
  entryPlan:  { line: '#60a5fa', bg: '#1e3a8a', fg: '#dbeafe', style: 0 }, // blue
  entryFill:  { line: '#3b82f6', bg: '#1d4ed8', fg: '#eff6ff', style: 0 },
  stopPlan:   { line: '#ef4444', bg: '#7f1d1d', fg: '#fee2e2', style: 2 }, // red dashed
  stopFill:   { line: '#dc2626', bg: '#991b1b', fg: '#fff1f2', style: 0 }, // red solid when live
  tpPlan:     { line: '#22c55e', bg: '#14532d', fg: '#dcfce7', style: 2 }, // green dashed
  tpFill:     { line: '#16a34a', bg: '#166534', fg: '#f0fdf4', style: 0 },
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

// ──────────────────────────────────────────────────────────────
// Label layer (HTML overlaid on chart body)
// ──────────────────────────────────────────────────────────────

let _labelStylesInjected = false;
function _injectLabelStyles() {
  if (_labelStylesInjected) return;
  _labelStylesInjected = true;
  const s = document.createElement('style');
  s.setAttribute('data-plan-overlay-styles', '1');
  s.textContent = `
    .plan-overlay-layer {
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 3;
      overflow: hidden;
    }
    .plan-overlay-label {
      position: absolute;
      padding: 1px 5px;
      font-size: 9px;
      line-height: 12px;
      letter-spacing: 0.3px;
      border-radius: 2px;
      font-family: inherit;
      pointer-events: none;
      white-space: nowrap;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
      transform: translateY(-50%);
      opacity: 0.92;
    }
  `;
  document.head.appendChild(s);
}

function _ensureLabelLayer() {
  if (_labelLayer && _labelLayer.isConnected) return _labelLayer;
  // Find the chart container. chart.js initialises LightweightCharts
  // into #chart-container by default.
  const container = _chartContainerRef
    || document.querySelector('#chart-container')
    || document.querySelector('#v2-chart');
  if (!container) return null;
  _chartContainerRef = container;
  // Ensure the container is positioned for absolute children
  const posValue = window.getComputedStyle(container).position;
  if (posValue === 'static') container.style.position = 'relative';
  _injectLabelStyles();
  _labelLayer = document.createElement('div');
  _labelLayer.className = 'plan-overlay-layer';
  _labelLayer.setAttribute('data-testid', 'plan-overlay-layer');
  container.appendChild(_labelLayer);
  return _labelLayer;
}

function _clearLabelLayer() {
  if (_labelLayer) _labelLayer.innerHTML = '';
  if (_rafHandle) {
    cancelAnimationFrame(_rafHandle);
    _rafHandle = null;
  }
}

function _collectAllLabels() {
  // Flatten every visible cond's label descriptors
  const out = [];
  for (const entry of _byKey.values()) {
    for (const [cid, rec] of entry) {
      for (const lbl of rec.labels || []) {
        out.push({ cid, ...lbl });
      }
    }
  }
  return out;
}

function _priceAxisWidthPx() {
  if (!_chartRef) return 60;
  try {
    const ps = _chartRef.priceScale?.('right');
    const w = ps?.width?.();
    if (Number.isFinite(w) && w > 0) return w;
  } catch {}
  return 60;
}

function _positionLabels() {
  if (!_labelLayer || !_candleSeriesRef) return;
  const labels = _collectAllLabels();
  if (labels.length === 0) {
    _labelLayer.innerHTML = '';
    return;
  }

  // 1. Ensure child count matches labels count. This MUST happen even
  //    when the layer is currently zero-size (mid symbol-switch reflow)
  //    — otherwise children never materialise and the RAF loop dies
  //    with children.length === 0 forever. Bug caught 2026-04-23 on
  //    RAVEUSDT switch: probe said rec.labels=3 but DOM had 0 children.
  const existing = Array.from(_labelLayer.children);
  while (existing.length < labels.length) {
    const el = document.createElement('div');
    el.className = 'plan-overlay-label';
    _labelLayer.appendChild(el);
    existing.push(el);
  }
  while (existing.length > labels.length) {
    const el = existing.pop();
    el.remove();
  }

  // 2. Positioning requires a real layer size + a working priceScale.
  //    If either is missing (typically: chart just rebuilt), retain
  //    children and let RAF retry next frame.
  const layerRect = _labelLayer.getBoundingClientRect();
  if (layerRect.width <= 0 || layerRect.height <= 0) return;
  const axisW = _priceAxisWidthPx();
  const rightPad = Math.max(4, axisW + 4);

  for (let i = 0; i < labels.length; i++) {
    const lbl = labels[i];
    const el = existing[i];
    let y;
    try { y = _candleSeriesRef.priceToCoordinate(lbl.price); } catch { y = null; }
    if (y == null || y < -12 || y > layerRect.height + 12) {
      el.style.display = 'none';
      continue;
    }
    el.style.display = '';
    el.textContent = lbl.text;
    el.style.background = lbl.bg;
    el.style.color = lbl.fg;
    el.style.right = `${rightPad}px`;
    el.style.top = `${y}px`;
  }
}

function _startRafLoop() {
  if (_rafHandle) return;
  const tick = () => {
    _rafHandle = null;
    _positionLabels();
    // Keep running while there are labels in the _byKey state. Checking
    // children.length was wrong: on symbol-switch the layer may briefly
    // have 0 children during resize, which used to kill the loop and
    // labels never recovered.
    if (_collectAllLabels().length > 0) {
      _rafHandle = requestAnimationFrame(tick);
    }
  };
  _rafHandle = requestAnimationFrame(tick);
}

// ──────────────────────────────────────────────────────────────
// Live-account cross-check (ground truth)
// ──────────────────────────────────────────────────────────────

async function _getLiveAccount() {
  const now = Date.now();
  if (_accountCache.data && (now - _accountCache.ts < ACCOUNT_TTL_MS)) {
    return _accountCache.data;
  }
  try {
    const r = await fetchJson('/api/live-execution/account?mode=live', {
      timeout: 6000,
      noCache: true,
    });
    if (r && r.ok !== false) {
      _accountCache = { ts: now, data: r };
      return r;
    }
  } catch (err) {
    // Fall through to stale cache
  }
  return _accountCache.data; // may be null if we've never succeeded
}

function _buildActiveSets(account) {
  // Return { haveSnapshot, openPositionKeys, pendingOids } — callers
  // use `haveSnapshot` to decide whether to trust the filter (fail-safe:
  // if we never got an account snapshot, show everything rather than
  // blank the chart).
  if (!account) return { haveSnapshot: false, openPositionKeys: new Set(), pendingOids: new Set() };
  const openPositionKeys = new Set();
  for (const p of account.positions || []) {
    const sym = String(p.symbol || '').toUpperCase();
    const size = Number(p.total || p.size || p.available || 0);
    if (!sym || size <= 0) continue;
    const hs = String(p.holdSide || p.posSide || p.side || '').toLowerCase();
    const dir = hs === 'long' || hs === 'buy' ? 'long'
              : hs === 'short' || hs === 'sell' ? 'short'
              : '';
    if (!dir) continue;
    openPositionKeys.add(`${sym}:${dir}`);
  }
  const pendingOids = new Set();
  for (const o of account.pending_orders || []) {
    const oid = String(o.orderId || o.order_id || o.planOrderId || o.clientOid || o.client_oid || '');
    if (oid) pendingOids.add(oid);
  }
  return { haveSnapshot: true, openPositionKeys, pendingOids };
}

// ──────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────

/** Clear all price-line markers + labels for a given key (or everything). */
export function clearPlanOverlay(key = null) {
  if (key == null) {
    if (_candleSeriesRef) {
      for (const entry of _byKey.values()) {
        for (const rec of entry.values()) {
          for (const line of rec.lines || []) {
            try { _candleSeriesRef.removePriceLine(line); } catch {}
          }
        }
      }
    }
    _byKey.clear();
    _clearLabelLayer();
    return;
  }
  const entry = _byKey.get(key);
  if (!entry) return;
  if (_candleSeriesRef) {
    for (const rec of entry.values()) {
      for (const line of rec.lines || []) {
        try { _candleSeriesRef.removePriceLine(line); } catch {}
      }
    }
  }
  _byKey.delete(key);
  // Don't nuke the whole layer — other symbols may still have labels
  if (_labelLayer && _byKey.size === 0) _clearLabelLayer();
}

/** Draw/refresh markers for the currently-loaded symbol & interval. */
export async function refreshPlanOverlay(chart, candleSeries, opts = {}) {
  if (!candleSeries) return;
  _candleSeriesRef = candleSeries;
  if (chart) _chartRef = chart;

  const symbol = (opts.symbol || marketState.currentSymbol || '').toUpperCase();
  const interval = opts.interval || marketState.currentInterval || '';
  if (!symbol) return;

  const key = _key(symbol, interval);
  // Wipe previous lines for THIS key only
  clearPlanOverlay(key);
  if (!_visible) return;

  // Fetch conds + live account in parallel
  let conds = [];
  let account = null;
  try {
    const [condResp, acctResp] = await Promise.allSettled([
      listConditionals('all', symbol),
      _getLiveAccount(),
    ]);
    if (condResp.status === 'fulfilled') {
      const r = condResp.value;
      conds = r?.conditionals || r?.data || r || [];
    }
    if (acctResp.status === 'fulfilled') {
      account = acctResp.value;
    }
  } catch (err) {
    console.warn('[plan_overlay] fetch failed:', err);
    return;
  }

  const { haveSnapshot, openPositionKeys, pendingOids } = _buildActiveSets(account);

  // Counts for diagnostics — exposed via probe even when nothing is shown.
  let counts = { total: conds.length, on_symbol_tf: 0, hidden_filled_stale: 0, hidden_triggered_stale: 0, shown_triggered: 0, shown_filled: 0 };

  // Step 1: pre-filter by symbol/TF + status, sort by created_at desc
  const bySymTf = conds
    .filter((c) => c.symbol?.toUpperCase() === symbol)
    .filter((c) => !interval || !c.timeframe || c.timeframe === interval);
  counts.on_symbol_tf = bySymTf.length;
  bySymTf.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

  // Step 2: dedup filled conds per (symbol, direction). Bitget aggregates
  // multiple fills into ONE position; drawing N label-sets for N filled
  // conds clutters the chart with phantom TPs/SLs that don't correspond
  // to the actual live position. Keep only the most-recent filled cond
  // per direction. 2026-04-24 user complaint: 84 filled ZEC long conds
  // produced duplicate 持仓止损 / 持仓止盈 labels at 322.156 / 319.869 /
  // 346.978 / 380.963 / 383.690 — all stale from prior test placements.
  const filledSeenByDir = new Set();
  const active = bySymTf.filter((c) => {
    if (c.status === 'triggered') {
      // Plan order waiting on Bitget. Fail-SAFE: if account API is
      // unavailable we trust local state (the backend classifier only
      // transitions to 'triggered' on affirmative Bitget evidence).
      if (!haveSnapshot) { counts.shown_triggered += 1; return true; }
      const oid = String(c.exchange_order_id || '');
      if (oid && pendingOids.has(oid)) { counts.shown_triggered += 1; return true; }
      counts.hidden_triggered_stale += 1;
      return false;
    }
    if (c.status === 'filled') {
      // Fail-CLOSED: hide when Bitget snapshot unavailable (avoid lying
      // about phantom positions). See 2026-04-23 HYPE incident.
      if (!haveSnapshot) { counts.hidden_filled_stale += 1; return false; }
      const dir = c.order?.direction || 'long';
      const key = `${c.symbol.toUpperCase()}:${dir}`;
      if (!openPositionKeys.has(key)) {
        counts.hidden_filled_stale += 1;
        return false;
      }
      // DEDUP: only the FIRST cond we see for this (symbol, dir) —
      // since we sorted desc by created_at, that's the most recent.
      if (filledSeenByDir.has(key)) {
        counts.hidden_filled_stale += 1;
        return false;
      }
      filledSeenByDir.add(key);
      counts.shown_filled += 1;
      return true;
    }
    return false;
  });

  // Always publish probe (even when we show 0 rows) so UI tests +
  // the user's "why is my chart empty?" debug can see what was
  // filtered and why.
  try {
    window.__planOverlayProbe = {
      updated_at: Date.now(),
      symbol, interval,
      account_snapshot_ok: haveSnapshot,
      open_position_keys: Array.from(openPositionKeys),
      pending_oid_count: pendingOids.size,
      counts,
      active_cond_count: active.length,
      rows: [],
    };
  } catch {}

  if (active.length === 0) return;

  _ensureLabelLayer();
  const byCond = new Map();

  for (const c of active) {
    const isFilled = c.status === 'filled';
    const dir = c.order?.direction || 'long';
    const dirTag = dir === 'long' ? '多' : '空';
    const statusTag = isFilled ? '持仓' : '计划';
    const qty = _fmtQty(c.fill_qty);

    const entryPrice = Number(c.fill_price);
    // SL geometry:
    //   entry sits `entry_offset_points` AWAY from the line
    //     (long: above / short: below).
    //   SL sits `stop_points` further PAST the line (on the same side
    //     as the direction's loss).
    //   Total distance from entry to SL = entry_offset + stop_points.
    // Pre-2026-04-23 this code read only `stop_points`, which gave the
    // distance from LINE to SL — resulting in chart labels drawn way
    // too close to entry (313.46 instead of real 312.18 on ZEC). The
    // real Bitget-submitted SL is at line ± stop_offset, NOT at entry
    // ± stop_points.
    const entryOff = Number(c.order?.entry_offset_points || 0);
    const stopOff = Number(c.order?.stop_points || 0);
    const slDist = entryOff + stopOff;
    const stopPrice = Number.isFinite(entryPrice) && entryPrice > 0 && slDist > 0
      ? (dir === 'long' ? entryPrice - slDist : entryPrice + slDist)
      : NaN;
    const tpPrice = Number(c.order?.tp_price);

    const lines = [];
    const labels = [];
    const draw = (price, kind) => {
      const style = COLORS[kind];
      // Horizontal line on chart; axis pill suppressed so the price
      // axis column stays clean — label goes on the chart body instead.
      try {
        const line = candleSeries.createPriceLine({
          price,
          color: style.line,
          lineStyle: style.style,
          lineWidth: 1,
          axisLabelVisible: false,
        });
        lines.push(line);
      } catch (err) { console.warn('[plan_overlay] price line err:', err); }
      return style;
    };

    // Entry
    if (Number.isFinite(entryPrice) && entryPrice > 0) {
      const style = draw(entryPrice, isFilled ? 'entryFill' : 'entryPlan');
      const qtyStr = qty ? ` ${qty}` : '';
      labels.push({
        price: entryPrice,
        text: `${statusTag} ${dirTag} ${_fmtPrice(entryPrice)}${qtyStr}`,
        bg: style.bg, fg: style.fg,
      });
    }
    // Stop
    if (Number.isFinite(stopPrice) && stopPrice > 0) {
      const style = draw(stopPrice, isFilled ? 'stopFill' : 'stopPlan');
      labels.push({
        price: stopPrice,
        text: `${statusTag}止损 ${_fmtPrice(stopPrice)}`,
        bg: style.bg, fg: style.fg,
      });
    }
    // Target
    if (Number.isFinite(tpPrice) && tpPrice > 0) {
      const style = draw(tpPrice, isFilled ? 'tpFill' : 'tpPlan');
      labels.push({
        price: tpPrice,
        text: `${statusTag}止盈 ${_fmtPrice(tpPrice)}`,
        bg: style.bg, fg: style.fg,
      });
    }

    if (lines.length > 0 || labels.length > 0) {
      byCond.set(c.conditional_id, { lines, labels });
    }
  }

  _byKey.set(key, byCond);
  _lastPullMs = Date.now();

  _positionLabels();
  _startRafLoop();

  // Enrich probe with the actually-rendered rows
  try {
    const dump = [];
    for (const [k, m] of _byKey) {
      for (const [cid, rec] of m) {
        dump.push({ key: k, cond_id: cid, line_count: rec.lines.length, label_count: rec.labels.length });
      }
    }
    if (window.__planOverlayProbe) window.__planOverlayProbe.rows = dump;
  } catch {}
}

/** Start a polling timer that refreshes markers every N seconds.
 *
 *  Event-driven refresh (chart.js subscribes to 'conditionals.changed')
 *  handles the fast path — user places/cancels, watcher replans, server
 *  pushes an event → overlay redraws within one tick.
 *
 *  This poll is just a fallback for edge cases where the event bus
 *  missed something (stream disconnect, watcher crash, etc.). 60s is
 *  plenty — previous 10s added ~6 Bitget calls/min per open chart and
 *  contributed to 429s on 2026-04-23 00:48 LA. */
export function startPlanOverlayPoll(chart, candleSeries, intervalMs = 60000) {
  if (_pollTimer) clearInterval(_pollTimer);
  _candleSeriesRef = candleSeries;
  if (chart) _chartRef = chart;
  _pollTimer = setInterval(() => {
    refreshPlanOverlay(chart, candleSeries).catch(() => {});
  }, intervalMs);
}

export function stopPlanOverlayPoll() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

/** Force-refresh (e.g. when user just placed/cancelled an order). */
export function forcePlanOverlayRefresh(chart, candleSeries) {
  // Invalidate account cache so the next refresh picks up changes fast
  _accountCache = { ts: 0, data: null };
  return refreshPlanOverlay(chart, candleSeries);
}

export function setPlanOverlayVisible(on) {
  _visible = !!on;
  if (!on) clearPlanOverlay();
}
