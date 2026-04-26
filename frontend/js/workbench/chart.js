// frontend/js/workbench/chart.js — minimal LightweightCharts wrapper using services+state

import { $ } from '../util/dom.js';
import { marketState, setCandles, setHistoryMeta, setHistoryMode, setPrecision, setScale } from '../state/market.js';
import { strategyState, setStrategyLayerVisible } from '../state/strategy.js';
import { publish, subscribe } from '../util/events.js';
import * as marketSvc from '../services/market.js';
import { inferPrecision, formatPrice } from '../util/format.js';
import { markBoot } from '../ui/boot_status.js';
import { drawMAOverlays, toggleMAOverlays as toggleMA, computeOverlaysFromCandles } from './ma_overlay.js';
import { applyIndicators } from './indicators/indicator_controller.js';
import { drawWyckoffOverlay, clearWyckoffOverlay, toggleWyckoff, getWyckoffMarkers } from './wyckoff_overlay.js';
import { startTickerWS, setTickerSymbol, stopTickerWS } from './ws_ticker.js';
import { clearHorizontalSRZones } from './patterns.js';
import { clearTrendlineOverlay } from './overlays/trendline_overlay.js';
import { clearSignalOverlay } from './overlays/signal_overlay.js';
import { clearOrderOverlay } from './overlays/order_overlay.js';
import {
  refreshPlanOverlay, clearPlanOverlay,
  startPlanOverlayPoll, stopPlanOverlayPoll,
} from './overlays/plan_order_overlay.js';
import { initManualTrendlineController, refreshManualDrawings, renderManualLines } from './drawings/manual_trendline_controller.js';

let chart = null;
let candleSeries = null;
let volumeSeries = null;
let liveTimer = null;
let strategyLayerPanel = null;
let chartModePanel = null;
let chartLoadSeq = 0;
let _lastFitKey = null;  // tracks last symbol/interval we fitContent'd for
let _lastFullReloadTs = 0;
// Right-side empty area shown BEYOND the last candle so trendlines that
// extend into the future have canvas room. 2026-04-20 iterations:
//   48  → 120 (too much; user said chart stretched weird)
//   120 → 60  (balance — ~5h on 5m, ~15h on 15m, ~60h on 1h).
//   2026-04-22: flat 60 still stretched 4h/1d charts to look "not
//   like TradingView" (candles squished into left half, vast empty
//   right side). TradingView default rightOffset is ~10 bars. We
//   need SOME future room for trendline extensions, so scale by TF:
//   fewer absolute bars on higher TFs where each bar is many hours.
const FUTURE_DRAW_BARS_BY_TF = {
  '1m': 45, '3m': 35, '5m': 30, '15m': 24, '30m': 20,
  '1h': 16, '2h': 14, '4h': 12, '6h': 10, '12h': 10,
  '1d': 10, '1w': 8,
};
const FUTURE_DRAW_BARS_DEFAULT = 20;
function futureDrawBarsFor(tf) {
  return FUTURE_DRAW_BARS_BY_TF[(tf || '').toLowerCase()] ?? FUTURE_DRAW_BARS_DEFAULT;
}
// Kept for backwards compat where the flat constant is referenced;
// most call sites should now use futureDrawBarsFor(tf) instead.
const FUTURE_DRAW_BARS = FUTURE_DRAW_BARS_DEFAULT;

// Lazy backfill state. Kept at module level so the visible-range
// subscriber can see the current candle buffer without a closure over
// loadCurrent. Cleared on every loadCurrent() call so old buffers from
// a previous symbol/TF don't leak into the new view.
let _backfillCandles = [];   // current full candle array (ref we prepend to)
let _backfillVolume = [];    // companion volume array
let _backfillInFlight = false;
let _backfillExhausted = false;  // stop after a backfill returns 0 rows
let _backfillSymbol = null;
let _backfillInterval = null;

export function initChart(containerId = 'chart-container') {
  const el = $('#' + containerId);
  if (!el) {
    console.error('[chart] container not found:', containerId);
    return;
  }

  el.innerHTML = '<div class="chart-skeleton"><div class="spinner"></div><div>Loading chart...</div></div>';

  if (typeof LightweightCharts === 'undefined') {
    el.innerHTML = '<div class="chart-skeleton error"><div>Chart library failed to load</div><div class="muted">Check network</div></div>';
    console.error('[chart] LightweightCharts library not loaded');
    return;
  }

  el.innerHTML = '';

  chart = LightweightCharts.createChart(el, {
    width: el.clientWidth,
    height: el.clientHeight,
    layout: { background: { color: '#0a0e17' }, textColor: '#e0e6ed' },
    // Watermark: personal brand in the bottom-left, semi-transparent so
    // it doesn't fight with candle rendering. User 2026-04-22: 承砚.
    watermark: {
      visible: true,
      text: '承砚',
      fontSize: 28,
      fontFamily: "'PingFang SC','Microsoft YaHei','Hiragino Sans GB',system-ui,sans-serif",
      fontStyle: 'bold',
      color: 'rgba(255,255,255,0.08)',
      horzAlign: 'left',
      vertAlign: 'bottom',
    },
    grid: { vertLines: { color: '#1a2035' }, horzLines: { color: '#1a2035' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: {
      borderColor: '#2a3548',
      autoScale: true,
      mode: LightweightCharts.PriceScaleMode.Logarithmic,  // log by default
      scaleMargins: { top: 0.05, bottom: 0.05 },
    },
    timeScale: {
      borderColor: '#2a3548',
      timeVisible: true,
      secondsVisible: false,
      rightOffset: FUTURE_DRAW_BARS,
      // Reserve space for future-time labels in the axis. Without this,
      // the axis only renders labels for existing candles, leaving the
      // rightOffset area visually blank ("没有未来的日期时间" 2026-04-20).
      fixLeftEdge: false,
      fixRightEdge: false,
    },
    // Pan/zoom: explicit so a freezeChart() during draw mode that failed
    // to thaw can't leave the chart undraggable across a reload. Defaults
    // are all `true` but we name them so the intent is unambiguous.
    handleScroll: {
      mouseWheel: true,
      pressedMouseMove: true,
      horzTouchDrag: true,
      vertTouchDrag: true,
    },
    handleScale: {
      axisPressedMouseMove: true,
      mouseWheel: true,
      pinch: true,
    },
  });

  candleSeries = chart.addCandlestickSeries({
    upColor: '#00e676',
    downColor: '#ff1744',
    borderVisible: false,
    wickUpColor: '#00e676',
    wickDownColor: '#ff1744',
  });

  volumeSeries = chart.addHistogramSeries({
    color: '#2a3548',
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
    scaleMargins: { top: 0.8, bottom: 0 },
    // Without these, the volume bar's last value (e.g. "199.44K") shows
    // up as a highlighted price-line label on the MAIN price axis between
    // two adjacent price ticks — because lightweight-charts treats
    // histogram's current value the same as a candle's close price. User
    // caught it 2026-04-20 on RAVEUSDT where "199.44K" appeared between
    // 0.50 and 0.49 on the Y-axis.
    lastValueVisible: false,
    priceLineVisible: false,
  });

  // Resize chart on window resize AND container resize (fixes zoom disappear bug)
  const resizeChart = () => {
    if (!chart || !el) return;
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (w > 0 && h > 0) {
      chart.applyOptions({ width: w, height: h });
    }
  };
  window.addEventListener('resize', resizeChart);
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(resizeChart).observe(el);
  }

  // strategy layer panel removed — all auto-strategy drawing is disabled
  ensureChartModePanel(el.parentElement || el);
  initManualTrendlineController(chart, el.parentElement || el);
  applyScaleMode();

  // Measure + prediction tools. Both need the chart + candle series for
  // data-space anchoring of their overlays.
  try {
    import('./drawings/measure_tool.js').then((mod) => {
      mod.initMeasureTool(chart, candleSeries, el.parentElement || el);
    }).catch((err) => console.warn('[measure] init err:', err));
    import('./drawings/prediction_tool.js').then((mod) => {
      mod.initPredictionTool(chart, candleSeries, el.parentElement || el);
    }).catch((err) => console.warn('[predict] init err:', err));
  } catch (err) { console.warn('[chart] tool init err:', err); }

  // Keyboard: M key toggles measure tool (TradingView parity).
  document.addEventListener('keydown', (ev) => {
    if (ev.key?.toLowerCase() !== 'm') return;
    // Ignore when user is typing in any input/textarea.
    const t = ev.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    ev.preventDefault();
    import('./drawings/measure_tool.js').then((mod) => {
      // Toggle: if we're in measure mode already, exit; else enter.
      // The module tracks its own _active flag via publish('measure.mode', ...)
      mod.enterMeasureMode('rect');
    });
  });

  // Plan/position overlay refreshes when ANY conditionals state change
  // ripples through the system (user places / cancels / replan fires,
  // server pushes 'conditionals.changed'). Event is the primary refresh
  // path — instant response to user gestures. Poll (60s, see below) is
  // a fallback for when events are missed (stream disconnect, etc.).
  subscribe('conditionals.changed', () => {
    if (!candleSeries) return;
    refreshPlanOverlay(chart, candleSeries).catch(() => {});
  });
  // Apply X-axis timezone per user preference (localStorage: v2.chart.tz.v1)
  _applyTickMarkFormatter();

  // OHLC hover tooltip (Bitget-style info strip). User 2026-04-21:
  // "滑过去的时候,它并不能显示它到底是赚了还是多少 — 没有 information"
  _initOhlcHover(el);

  // Lazy backfill: when the user scrolls so the visible range's leftmost
  // logical index is within the first ~10 bars of loaded data, fetch
  // another ~500 bars older and prepend them. TradingView-style.
  try {
    chart.timeScale().subscribeVisibleLogicalRangeChange?.((range) => {
      if (!range || _backfillInFlight || _backfillExhausted) return;
      if (_backfillCandles.length === 0) return;
      // range.from is a logical index (can be fractional, negative when
      // scrolled left of bar 0). Trigger when within 10 bars of the left.
      if (range.from > 10) return;
      void _doBackfill();
    });
  } catch {}

  // 2026-04-20: symbol/TF switch must clear old series data BEFORE the
  // async fetch starts. Otherwise the live WS ticker (which switches
  // subscriptions synchronously) can push prices for the NEW symbol
  // into the OLD candle history, blowing up the chart's y-axis. Seen
  // when flipping HYPE → ETH: ETH's $2317 live tick got pasted onto
  // HYPE's $30-$45 historical candles → giant green spike.
  //
  // 2026-04-23: ALSO swap drawings to the new symbol synchronously, so
  // SVG overlay doesn't try to render OLD-symbol lines on NEW-symbol
  // chart's price grid (caused the "lines flash off/on" bug user reported
  // on XAU). preSwapDrawingsForSymbol hits an in-memory cache keyed by
  // (symbol, TF) — instant swap; refreshManualDrawings refines from
  // server right after via loadCurrent().
  const clearForSymbolSwitch = (newSymbol, newInterval) => {
    try {
      candleSeries?.setData([]);
      volumeSeries?.setData([]);
    } catch {}
    // Also zero out lastCandles so onTickerTick doesn't compute
    // high/low vs a stale bar from the previous symbol.
    try { setCandles([]); } catch {}
    // Synchronously swap drawings to NEW symbol's cache (or empty if
    // never visited). This eliminates the orphan-render flicker.
    try {
      const sym = newSymbol || marketState.currentSymbol;
      const tf = newInterval || marketState.currentInterval;
      if (sym && tf) {
        // Lazy-import so we don't introduce a circular dep at top.
        import('./drawings/manual_trendline_controller.js')
          .then((mod) => mod.preSwapDrawingsForSymbol(sym, tf))
          .catch(() => {});
      }
    } catch {}
  };
  // 2026-04-25: coalesce symbol+interval changes that fire within the
  // same microtask. Bug surfaced when user clicked a row in 我的手画线
  // panel: handler did `setSymbol(X); setIntervalTF(Y);` back-to-back
  // → two loadCurrent() calls raced; first one returned `{stale:true}`
  // because its loadSeq was superseded; second sometimes also got
  // superseded; chart stayed empty with no candles even though price
  // scale showed the new symbol's range. Now we wait one microtask
  // (queueMicrotask) before firing the actual reload, so a setSymbol
  // immediately followed by setIntervalTF only triggers ONE fetch.
  let _coalesceScheduled = false;
  const _coalesceReload = () => {
    if (_coalesceScheduled) return;
    _coalesceScheduled = true;
    queueMicrotask(() => {
      _coalesceScheduled = false;
      clearForSymbolSwitch(marketState.currentSymbol, marketState.currentInterval);
      void loadCurrent(true).catch((err) => console.warn('[chart] coalesced reload failed:', err));
    });
  };
  subscribe('market.symbol.changed', _coalesceReload);
  subscribe('market.interval.changed', _coalesceReload);
  subscribe('market.history_mode.changed', () => {
    syncChartModePanel();
    void loadCurrent(true).catch((err) => console.warn('[chart] history mode refresh failed:', err));
  });
  subscribe('market.history_meta.changed', () => {
    syncChartModePanel();
    updateHeader(marketState.currentSymbol, marketState.currentInterval, marketState.lastCandles.at(-1)?.close);
  });
  subscribe('market.scale.changed', () => {
    applyScaleMode();
    syncChartModePanel();
    updateHeader(marketState.currentSymbol, marketState.currentInterval, marketState.lastCandles.at(-1)?.close);
  });
  subscribe('strategy.snapshot.updated', () => renderStrategyOverlays());
  subscribe('strategy.layers.changed', () => {
    syncStrategyLayerPanel();
    renderStrategyOverlays();
  });
  subscribe('drawings.updated', () => renderStrategyOverlays());
  subscribe('drawings.viewMode', () => renderStrategyOverlays());

  // Trade markers from execution panel (buy/sell arrows)
  let execMarkers = [];
  subscribe('execution.trade.markers', (markers) => {
    execMarkers = markers || [];
    applyExecMarkers();
  });
  subscribe('execution.strategy.deselected', () => {
    execMarkers = [];
    applyExecMarkers();
  });

  function applyExecMarkers() {
    if (!candleSeries) return;
    if (execMarkers.length === 0) return;
    try {
      // Merge with existing signal markers instead of replacing
      const existing = [];
      try { const cur = candleSeries.markers?.() || []; existing.push(...cur); } catch {}
      const merged = [...existing, ...execMarkers].sort((a, b) => a.time - b.time);
      candleSeries.setMarkers(merged);
    } catch (err) {
      console.warn('[chart] exec markers failed:', err);
    }
  }

  return chart;
}

export async function loadCurrent(forcePatterns = false) {
  const { currentSymbol, currentInterval } = marketState;
  const loadSeq = ++chartLoadSeq;
  if (!candleSeries) {
    throw new Error('candleSeries not ready');
  }

  try {
    // 2026-04-25: per CLAUDE.md "When in doubt about a default value":
    // **Chart initial-fetch SIZE is bounded by render budget, NOT data
    // depth.** Lightweight-charts is smooth up to ~5000 bars; beyond
    // that, candle painting + indicator recompute + marker placement
    // visibly stutters. Older bars are NOT lost — they lazy-load via
    // the existing `/api/ohlcv/backfill` path when the user scrolls
    // toward the left edge.
    //
    // Why the previous "1m=30d" rule had to die: 30d of 1m = 43,200
    // bars = 15MB JSON. The API took ~25s to serialize, the browser
    // froze for ~10s parsing it, and the chart took another ~5s to
    // mount the indicator series across all 43K bars. User clicked
    // "1m" and the page hung for 30+ seconds. Strategy.py's deep
    // history is a separate fetch path with its own caching — it
    // does NOT need to share the chart's initial-window budget.
    //
    // The target is ~5000 bars per TF for smooth render on commodity
    // laptops (HYPE 1m → 3.5d, 4h → 833d, 1d → 5000d ≈ all history).
    // Each entry below is computed as: max bars / bars-per-day.
    const tfDays = {
      '1m': 4,        // 5760 bars
      '3m': 12,       // 5760 bars
      '5m': 18,       // 5184 bars
      '15m': 60,      // 5760 bars
      '30m': 100,     // 4800 bars
      '1h': 210,      // 5040 bars
      '2h': 420,      // 5040 bars
      '4h': 730,      // 4380 bars
      '6h': 1100,     // 4400 bars
      '12h': 2500,    // 5000 bars
      '1d': 5000,     // 5000 bars (effectively all history)
      '1w': 3650,     // ~520 bars (full Bitget retention)
    };
    const days = tfDays[currentInterval] || 180;

    // OPTIMISTIC SWAP: if we have a recent cached result for this
    // (symbol, interval, days), paint it IMMEDIATELY without waiting
    // for the network. The subsequent awaited fetch either returns
    // the same cached data (service cache hit) or refreshes with live
    // bars. Either way the user sees an instant chart swap on TF
    // button click, and the live tail catches up a tick later.
    const cachedPeek = marketSvc.peekOhlcvCache(currentSymbol, currentInterval, days, marketState.historyMode);
    if (cachedPeek && Array.isArray(cachedPeek.candles) && cachedPeek.candles.length > 0) {
      const instantCandles = cachedPeek.candles.map((c) => ({
        time: typeof c.time === 'string' ? Math.floor(new Date(c.time).getTime() / 1000) : c.time,
        open: Number(c.open),
        high: Number(c.high),
        low: Number(c.low),
        close: Number(c.close),
      }));
      try { candleSeries.setData(instantCandles); } catch {}
    }

    const data = await marketSvc.getOhlcv(currentSymbol, currentInterval, days, null, marketState.historyMode);
    if (!isChartLoadCurrent(loadSeq, currentSymbol, currentInterval)) {
      return { ok: false, stale: true };
    }

    const rawCandles = data.candles || [];
    if (!Array.isArray(rawCandles) || rawCandles.length === 0) {
      throw new Error(`No candles for ${currentSymbol} ${currentInterval}`);
    }

    const candles = rawCandles.map((c) => ({
      time: typeof c.time === 'string' ? Math.floor(new Date(c.time).getTime() / 1000) : c.time,
      open: Number(c.open),
      high: Number(c.high),
      low: Number(c.low),
      close: Number(c.close),
    }));

    const rawVolume = data.volume || [];
    const volumes = rawVolume.map((v, i) => {
      const c = rawCandles[i];
      const isUp = c && Number(c.close) >= Number(c.open);
      return {
        time: typeof v.time === 'string' ? Math.floor(new Date(v.time).getTime() / 1000) : v.time,
        value: Number(v.value || 0),
        color: isUp ? 'rgba(0,230,118,0.4)' : 'rgba(255,23,68,0.4)',
      };
    });

    candleSeries.setData(candles);
    if (volumes.length > 0) volumeSeries.setData(volumes);

    // Reset + prime the lazy backfill buffer for this symbol/TF.
    _backfillCandles = candles.slice();
    _backfillVolume = volumes.slice();
    _backfillInFlight = false;
    _backfillExhausted = false;
    _backfillSymbol = currentSymbol;
    _backfillInterval = currentInterval;

    // ── Verifiable load report ─────────────────────────────────
    const _firstT = candles[0]?.time;
    const _lastT  = candles[candles.length - 1]?.time;
    const _spanD  = candles.length ? ((_lastT - _firstT) / 86400).toFixed(1) : 0;
    const _firstD = _firstT ? new Date(_firstT * 1000).toISOString().slice(0, 10) : '?';
    const _lastDS = _lastT  ? new Date(_lastT  * 1000).toISOString().slice(0, 10) : '?';
    console.log(
      `%c[chart] LOADED ${currentSymbol} ${currentInterval} — ${candles.length} bars · span ${_spanD}d · ${_firstD} → ${_lastDS}`,
      'background:#0d4a2a;color:#00e676;padding:2px 6px;font-weight:bold'
    );

    // ── Viewport invariant (2026-04-25, root-cause fix) ─────────
    //
    // INVARIANT: after a successful candle load, the user's visible
    //   range must INTERSECT the loaded data's time range. If the
    //   previous viewport (preserved from prior visit / user scroll)
    //   no longer intersects today's data, force re-fit.
    //
    // Why this isn't "another patch":
    //   The previous code only set viewport on FIRST visit per (sym,TF).
    //   It assumed prior viewport state was always valid for new data.
    //   That assumption was UNTESTED + UNENFORCED — it failed silently
    //   when:
    //     - user scrolled left to a date now outside the new data
    //     - days param expanded (e.g. tfDays cap removal 04-24)
    //     - IndexedDB cache returned an older subset
    //   Black-screen chart was the symptom; the absent invariant was
    //   the cause. The fix is the invariant, not a "fitContent on load".
    const ts = chart.timeScale();
    const dataFrom = candles[0]?.time;
    const dataTo = candles[candles.length - 1]?.time;
    let viewportStale = false;
    try {
      const vis = ts.getVisibleRange();
      if (!vis) {
        viewportStale = true;
      } else if (typeof dataFrom === 'number' && typeof dataTo === 'number') {
        // Intersect check: either end of viewport is outside the data span.
        // 1-bar slack so a viewport ending exactly on the last bar is fine.
        const barDurApprox = candles.length >= 2 ? (candles[1].time - candles[0].time) : 3600;
        const slack = barDurApprox;
        if (vis.to < dataFrom - slack || vis.from > dataTo + slack) {
          viewportStale = true;
        }
      }
    } catch { viewportStale = true; }

    const fitKey = `${currentSymbol}:${currentInterval}`;
    const isFirstFit = _lastFitKey !== fitKey;
    const needFit = isFirstFit || viewportStale;
    if (viewportStale && !isFirstFit) {
      console.log(
        `%c[chart] VIEWPORT STALE — re-fitting (data: ${new Date(dataFrom*1000).toISOString().slice(0,10)} → ${new Date(dataTo*1000).toISOString().slice(0,10)})`,
        'background:#7f1d1d;color:#fff;padding:2px 6px;font-weight:bold',
      );
    }
    if (needFit) {
      _lastFitKey = fitKey;
      const VISIBLE_BARS = 200;
      const totalBars = candles.length;
      const barDur = totalBars >= 2 ? candles[1].time - candles[0].time : 3600;
      // TF-aware future bars — 4h/1d with a flat 60-bar tail looked
      // stretched (candles squeezed into left half). See FUTURE_DRAW_BARS_BY_TF.
      const tfFutureBars = futureDrawBarsFor(currentInterval);
      try { ts.applyOptions({ rightOffset: tfFutureBars }); } catch {}

      const applyViewport = () => {
        try {
          if (totalBars > VISIBLE_BARS) {
            const fromTime = candles[totalBars - VISIBLE_BARS].time;
            const toTime   = candles[totalBars - 1].time + barDur * tfFutureBars;
            ts.setVisibleRange({ from: fromTime, to: toTime });
            // Verify it stuck
            const got = ts.getVisibleRange();
            const okFrom = got && Math.abs(got.from - fromTime) < barDur * 5;
            console.log(
              `%c[chart] viewport set → from ${new Date(fromTime*1000).toISOString().slice(0,10)} to ${new Date(toTime*1000).toISOString().slice(0,10)}  (got: ${got ? new Date(got.from*1000).toISOString().slice(0,10)+' → '+new Date(got.to*1000).toISOString().slice(0,10) : 'null'})  stuck=${okFrom}`,
              'background:#0a2540;color:#60a5fa;padding:2px 6px'
            );
            return okFrom;
          } else {
            ts.fitContent();
            return true;
          }
        } catch (e) {
          console.warn('[chart] viewport set err', e);
          return false;
        }
      };

      // Try 4 times across animation frames in case lightweight-charts
      // re-applies its own layout after setData. Each retry double-checks
      // and re-sets if the previous set didn't stick.
      let attempt = 0;
      const tryViewport = () => {
        attempt++;
        const ok = applyViewport();
        if (!ok && attempt < 4) {
          requestAnimationFrame(tryViewport);
        } else if (!ok) {
          console.warn('[chart] viewport never stuck after 4 attempts, giving up');
        }
      };
      requestAnimationFrame(tryViewport);
    }

    setCandles(candles);
    _lastFullReloadTs = Date.now() / 1000;
    setHistoryMeta({
      historyMode: data.historyMode || marketState.historyMode,
      loadedBarCount: data.loadedBarCount ?? candles.length,
      earliestLoadedTimestamp: data.earliestLoadedTimestamp ?? candles[0]?.time ?? null,
      latestLoadedTimestamp: data.latestLoadedTimestamp ?? candles[candles.length - 1]?.time ?? null,
      listingStartTimestamp: data.listingStartTimestamp ?? candles[0]?.time ?? null,
      isFullHistory: Boolean(data.isFullHistory),
      isTruncated: Boolean(data.isTruncated),
      truncationReason: data.truncationReason || '',
    });
    const lastPrice = candles[candles.length - 1].close;
    const precision = data.pricePrecision ?? inferPrecision(lastPrice);
    setPrecision(precision);
    // Apply exchange-provided tick precision to the candle series so Y-axis
    // labels and crosshair readouts match Bitget exactly (e.g. 0.50234 not
    // 0.50 for RAVEUSDT where tickSz=1e-5). Default lightweight-charts
    // precision is 2 decimals, which hides sub-cent detail for sub-$1 coins.
    if (candleSeries && typeof precision === 'number' && precision >= 0 && precision <= 10) {
      try {
        candleSeries.applyOptions({
          priceFormat: {
            type: 'price',
            precision,
            minMove: Math.pow(10, -precision),
          },
        });
      } catch (err) {
        console.warn('[chart] applyOptions priceFormat failed:', err);
      }
    }

    // Prefer client-computed overlays — saves a round of backend CPU
    // and shrinks the /api/ohlcv payload by ~40%. Falls back to server
    // overlays if candles are too short to compute MA55.
    const overlays = computeOverlaysFromCandles(candles);
    const useOverlays = Object.keys(overlays).length > 0 ? overlays : data.overlays;

    // Unified indicator dispatch (2026-04-20). Replaces the direct
    // drawMAOverlays + drawWyckoffOverlay calls; applyIndicators reads
    // the user's visibility preferences from the indicator panel's
    // localStorage and draws only what's turned on.
    try {
      window.__chartRef = chart;
      await applyIndicators(chart, candleSeries, candles, useOverlays);
    } catch (err) {
      console.warn('[chart] applyIndicators failed:', err);
      // Last-resort: still draw MA so chart isn't empty.
      if (useOverlays) {
        const candleTimes = candles.map((c) => c.time);
        drawMAOverlays(chart, useOverlays, candleTimes);
      }
    }
    // Wyckoff markers (Spring/UTAD shapes on candles) if Wyckoff on
    if (window.__wyckoffEnabled) {
      try {
        const wMarkers = getWyckoffMarkers();
        if (wMarkers.length > 0 && candleSeries) {
          const existing = candleSeries.markers ? candleSeries.markers() : [];
          candleSeries.setMarkers([...existing, ...wMarkers].sort((a, b) => a.time - b.time));
        }
      } catch {}
    }

    updateHeader(currentSymbol, currentInterval, lastPrice);
    // Loaded silently — no console spam
    publish('chart.load.succeeded', {
      symbol: currentSymbol,
      interval: currentInterval,
      loadSeq,
    });

    void refreshManualDrawings(currentSymbol, currentInterval).catch((err) => console.warn('[drawings] refresh failed:', err));
    // Entry / SL / TP price-line markers for every triggered plan +
    // live position on this symbol+TF. Added 2026-04-23 — user spent
    // days flying blind without seeing where stop/target actually sit.
    try {
      clearPlanOverlay();
      void refreshPlanOverlay(chart, candleSeries, {
        symbol: currentSymbol, interval: currentInterval,
      }).catch(() => {});
      // P0 2026-04-23: 10s → 60s fallback. Instant refresh comes from
      // the 'conditionals.changed' subscription above; the poll is only
      // a safety net for missed events (stream disconnect, etc.).
      startPlanOverlayPoll(chart, candleSeries, 60000);
    } catch (err) { console.warn('[plan_overlay] wire err:', err); }
    // 2026-04-25: OKX/Bitget-style trade execution markers (▲/▼ at entry
    // + circle at SL/TP exits). Symbol-scoped, TF-agnostic — fills show
    // on every chart of that symbol because the timestamp maps to the
    // bar containing it.
    try {
      const { initTradeMarkers, refreshTradeMarkers, startTradeMarkersAutoRefresh } =
        await import('./overlays/trade_markers_overlay.js');
      initTradeMarkers(candleSeries);
      // Pass currentInterval so the overlay can bucket entry fills by
      // bar (one marker per (bar, side), not one per fill). Switching
      // TF re-runs this with the new interval, so the bucketing
      // refreshes naturally.
      void refreshTradeMarkers(currentSymbol, currentInterval).catch(() => {});
      startTradeMarkersAutoRefresh();
    } catch (err) { console.warn('[trade_markers] wire err:', err); }
    markBoot('patterns', 'ok', 'manual-only mode');

    return {
      ok: true,
      stale: false,
      symbol: currentSymbol,
      interval: currentInterval,
    };
  } catch (err) {
    if (!isChartLoadCurrent(loadSeq, currentSymbol, currentInterval)) {
      return { ok: false, stale: true };
    }
    // AbortError is an expected side-effect of rapid symbol/TF
    // switching — we abort an in-flight fetch when the user selects
    // a new symbol before the previous fetch resolves. Don't spam
    // console.error for the normal cancel path; just warn + return.
    const isAbort = err?.name === 'AbortError'
      || /abort/i.test(err?.message || '');
    if (isAbort) {
      console.warn('[chart] previous load aborted (user switched symbol/TF)');
      return { ok: false, aborted: true };
    }
    setHistoryMeta(null);
    renderStrategyOverlays();
    console.error('[chart] load failed:', err);
    throw err;
  }
}

function isChartLoadCurrent(loadSeq, symbol, interval) {
  return (
    loadSeq === chartLoadSeq
    && symbol === marketState.currentSymbol
    && interval === marketState.currentInterval
  );
}

function renderStrategyOverlays() {
  // Auto-detected trendlines / signals / orders / horizontal SR zones are
  // disabled — user wants a clean chart with only lines he draws himself.
  if (!chart || !candleSeries) return;
  try { clearTrendlineOverlay(chart); } catch {}
  try { clearSignalOverlay(candleSeries); } catch {}
  try { clearOrderOverlay(chart); } catch {}
  try { clearHorizontalSRZones(chart); } catch {}
  renderManualLines();
}

function ensureStrategyLayerPanel(container) {
  if (!container || strategyLayerPanel) return;
  strategyLayerPanel = document.createElement('div');
  strategyLayerPanel.className = 'strategy-layer-panel';
  strategyLayerPanel.innerHTML = `
    <div class="strategy-layer-title">Strategy Layers</div>
    <div class="strategy-layer-grid">
      <label class="strategy-layer-option"><input type="checkbox" data-layer="primaryTrendlines" checked /> Primary Lines</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="debugTrendlines" /> Debug Lines</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="confirmingTouches" checked /> Confirming Touches</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="barTouches" /> Bar Touches</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="projectedLine" checked /> Projection</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="signalMarkers" checked /> Signals</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="collapsedInvalidations" checked /> Invalidations</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="orderMarkers" /> Orders</label>
    </div>
    <div class="strategy-layer-meta">Backend-driven strategy overlay</div>
  `;

  strategyLayerPanel.addEventListener('change', (event) => {
    const input = event.target;
    if (!(input instanceof HTMLInputElement)) return;
    const layer = input.dataset.layer;
    if (!layer) return;
    setStrategyLayerVisible(layer, input.checked);
  });

  container.appendChild(strategyLayerPanel);
  syncStrategyLayerPanel();
}

function syncStrategyLayerPanel() {
  if (!strategyLayerPanel) return;
  const inputs = strategyLayerPanel.querySelectorAll('input[data-layer]');
  inputs.forEach((input) => {
    const layer = input.dataset.layer;
    if (!layer) return;
    input.checked = !!strategyState.layerVisibility[layer];
  });
}

function ensureChartModePanel(container) {
  if (!container || chartModePanel) return;
  chartModePanel = document.createElement('div');
  chartModePanel.className = 'chart-mode-panel';
  chartModePanel.innerHTML = `
    <div class="chart-mode-title">Chart Mode</div>
    <div class="chart-mode-actions">
      <button class="btn chart-mode-btn" data-history-mode="fast_window">Fast</button>
      <button class="btn chart-mode-btn" data-history-mode="full_history">Full History</button>
    </div>
    <div class="chart-mode-actions">
      <button class="btn chart-scale-btn" data-scale-mode="linear">Linear</button>
      <button class="btn chart-scale-btn" data-scale-mode="log">Log</button>
    </div>
    <div class="chart-mode-meta" id="chart-mode-meta">Loading chart window...</div>
  `;

  chartModePanel.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    const historyMode = target.dataset.historyMode;
    if (historyMode) {
      setHistoryMode(historyMode);
      return;
    }
    const scaleMode = target.dataset.scaleMode;
    if (scaleMode) {
      setScale(scaleMode);
    }
  });

  container.appendChild(chartModePanel);
  syncChartModePanel();
}

function syncChartModePanel() {
  if (!chartModePanel) return;
  const historyButtons = chartModePanel.querySelectorAll('[data-history-mode]');
  historyButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.historyMode === marketState.historyMode);
  });
  const scaleButtons = chartModePanel.querySelectorAll('[data-scale-mode]');
  scaleButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.scaleMode === marketState.currentScale);
  });
  const meta = chartModePanel.querySelector('#chart-mode-meta');
  if (meta) {
    meta.textContent = formatHistoryMeta();
  }
}

function formatHistoryMeta() {
  const meta = marketState.historyMeta;
  if (!meta) {
    return 'No history metadata loaded yet.';
  }
  const range = `${formatUnixDate(meta.earliestLoadedTimestamp)} -> ${formatUnixDate(meta.latestLoadedTimestamp)}`;
  const base = `${meta.historyMode === 'full_history' ? 'Full history' : 'Fast window'} | ${meta.loadedBarCount ?? 0} bars | ${range}`;
  if (meta.isTruncated) {
    return `${base} | truncated (${meta.truncationReason || 'fast_window'}) | listing start ${formatUnixDate(meta.listingStartTimestamp)}`;
  }
  return `${base} | listing start ${formatUnixDate(meta.listingStartTimestamp)}`;
}

function formatUnixDate(timestamp) {
  if (!timestamp) return '-';
  const date = new Date(Number(timestamp) * 1000);
  return Number.isNaN(date.getTime()) ? '-' : date.toISOString().slice(0, 10);
}

function applyScaleMode() {
  if (!chart) return;
  if (typeof chart.priceScale !== 'function') return;
  const priceScale = chart.priceScale('right');
  if (!priceScale || typeof priceScale.applyOptions !== 'function') return;
  priceScale.applyOptions({
    mode: marketState.currentScale === 'log'
      ? LightweightCharts.PriceScaleMode.Logarithmic
      : LightweightCharts.PriceScaleMode.Normal,
  });
}

export function toggleMAOverlays() {
  return toggleMA();
}

export function toggleWyckoffOverlay() {
  return toggleWyckoff();
}

// Chart timezone for X-axis labels. User 2026-04-21: "K线时间基本上无所
// 谓, 你可以让我选择是 洛杉矶/国内/UTC".
//
// Default CHANGED to 'utc' 2026-04-23 — user asked every time reference
// (chart axis, bar boundaries, line projection server-side, % change
// baseline) to share ONE time base with no shift. Mixing UTC (% change
// is UTC 00:00) with local X-axis ('la' = UTC-7) meant "today's visible
// bar" on the chart was offset 7-8 hours from the UTC bar the server
// used for line projection, so the user's visual line reading didn't
// match what the server placed against.
//
// Migration: existing users who never touched the setting auto-switch
// to 'utc' once. Explicit post-migration choices (including switching
// back to 'la') persist.
const TZ_LS_KEY = 'v2.chart.tz.v1';
const TZ_MIGRATION_KEY = 'v2.chart.tz.utc-default.v1';
function _loadChartTz() {
  try {
    if (!localStorage.getItem(TZ_MIGRATION_KEY)) {
      localStorage.setItem(TZ_LS_KEY, 'utc');
      localStorage.setItem(TZ_MIGRATION_KEY, '1');
      return 'utc';
    }
    return localStorage.getItem(TZ_LS_KEY) || 'utc';
  } catch { return 'utc'; }
}
function _saveChartTz(tz) {
  try { localStorage.setItem(TZ_LS_KEY, tz); } catch {}
}
function _tzOffsetHours(tz) {
  if (tz === 'utc') return 0;
  if (tz === 'bj' || tz === 'cn') return 8;
  if (tz === 'la') {
    // DST-aware per reviewer S3. Nov-Mar = PST (-8), Mar-Nov = PDT (-7).
    // Use Intl to ask what LA's UTC offset is at RIGHT NOW. This still
    // isn't perfect for historical candles that crossed a DST boundary
    // (we apply ONE offset to the whole chart) but is correct for
    // the current moment's labels — which is what the user reads.
    try {
      const now = new Date();
      const laStr = now.toLocaleString('en-US', {
        timeZone: 'America/Los_Angeles',
        hour12: false, year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      });
      const m = laStr.match(/(\d+)\/(\d+)\/(\d+),?\s+(\d+):(\d+):(\d+)/);
      if (m) {
        const laLocal = Date.UTC(+m[3], +m[1] - 1, +m[2], +m[4], +m[5], +m[6]);
        const offsetMs = laLocal - now.getTime();
        return Math.round(offsetMs / 3600000);   // -7 in PDT, -8 in PST
      }
    } catch {}
    return -7;   // fallback
  }
  return 0;
}
function _tzLabel(tz) {
  if (tz === 'utc') return 'UTC';
  if (tz === 'bj') return '国内 UTC+8';
  if (tz === 'la') return '洛杉矶 UTC-7';
  return tz;
}
let _currentTz = _loadChartTz();

function _applyTickMarkFormatter() {
  if (!chart || typeof chart.applyOptions !== 'function') return;
  const offsetSec = _tzOffsetHours(_currentTz) * 3600;
  chart.applyOptions({
    localization: {
      timeFormatter: (time) => {
        // time is Unix seconds. Shift and format. Show H:mm for intraday,
        // MMM DD for date boundaries — matches lightweight-charts default style.
        const d = new Date((Number(time) + offsetSec) * 1000);
        const Y = d.getUTCFullYear();
        const M = String(d.getUTCMonth() + 1).padStart(2, '0');
        const D = String(d.getUTCDate()).padStart(2, '0');
        const h = String(d.getUTCHours()).padStart(2, '0');
        const m = String(d.getUTCMinutes()).padStart(2, '0');
        return `${Y}-${M}-${D} ${h}:${m}`;
      },
    },
    timeScale: {
      tickMarkFormatter: (time /* seconds */, tickMarkType, locale) => {
        const d = new Date((Number(time) + offsetSec) * 1000);
        const hr = d.getUTCHours();
        const mn = d.getUTCMinutes();
        // New-day boundary: show date
        if (hr === 0 && mn === 0) {
          const day = d.getUTCDate();
          const mon = d.getUTCMonth();
          const monStr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][mon];
          return `${day} ${monStr} '${String(d.getUTCFullYear()).slice(-2)}`;
        }
        return `${String(hr).padStart(2, '0')}:${String(mn).padStart(2, '0')}`;
      },
    },
  });
}

export function setChartTimezone(tz) {
  if (!['utc', 'bj', 'la'].includes(tz)) return;
  _currentTz = tz;
  _saveChartTz(tz);
  _applyTickMarkFormatter();
  // Force re-render of current labels
  try { chart?.timeScale()?.fitContent(); } catch {}
}

export function getChartTimezone() { return _currentTz; }

/**
 * Re-enable TradingView-style "auto fit" mode: price axis auto-scales
 * to visible bars, and time axis fits all candles including future
 * right-offset. User 2026-04-22: wanted an explicit button for this
 * because lightweight-charts disables autoScale as soon as you drag
 * the price axis, and there was no way back without reloading.
 */
export function resetChartViewport() {
  if (!chart) return;
  try {
    chart.priceScale('right').applyOptions({ autoScale: true });
  } catch {}
  try {
    chart.timeScale().fitContent();
  } catch {}
  // Also re-apply the TF-aware rightOffset (fitContent can sometimes
  // collapse the future-area margin).
  try {
    const tfBars = futureDrawBarsFor(marketState?.currentInterval);
    chart.timeScale().applyOptions({ rightOffset: tfBars });
  } catch {}
}

// Expose for external UI
if (typeof window !== 'undefined') {
  window.setChartTimezone = setChartTimezone;
  window.getChartTimezone = getChartTimezone;
}

// ─── UTC-day change calculation ───────────────────────────────
// % change since UTC 00:00 today (NOT a rolling 24h window). User
// 2026-04-21: "涨跌幅时间也是 UTC0, 就是 UTC 时间".
function _utc0TodayMs() {
  const now = new Date();
  return Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
}
function _computeUtcDayChange(candles, currentPrice) {
  if (!Array.isArray(candles) || candles.length === 0 || !currentPrice) return null;
  const utc0 = Math.floor(_utc0TodayMs() / 1000);
  // Find first candle whose time >= utc0 — its OPEN is our baseline.
  let baseline = null;
  for (const c of candles) {
    const t = typeof c.time === 'number' ? c.time : Math.floor(new Date(c.time).getTime() / 1000);
    if (t >= utc0) {
      baseline = Number(c.open ?? c.o ?? c.close);
      break;
    }
  }
  if (baseline == null || !isFinite(baseline) || baseline === 0) return null;
  return ((currentPrice - baseline) / baseline) * 100;
}

// ─────────────────────────────────────────────────────────────
// OHLC hover tooltip — Bitget-style info strip near chart top-left.
// Subscribes to crosshair move; reads the candle at the hovered X.
// ─────────────────────────────────────────────────────────────
let _ohlcHoverEl = null;

function _fmtPriceHover(p) {
  if (!isFinite(p)) return '—';
  if (p >= 1000) return p.toFixed(2);
  if (p >= 100) return p.toFixed(3);
  if (p >= 1) return p.toFixed(4);
  if (p >= 0.01) return p.toFixed(5);
  return p.toFixed(7);
}

function _initOhlcHover(chartEl) {
  if (!chart || !chartEl) return;
  // Host the floating strip INSIDE the chart container so it's clipped
  // to the chart panel (not overlapping sidebars).
  _ohlcHoverEl = document.createElement('div');
  _ohlcHoverEl.className = 'chart-ohlc-hover';
  _ohlcHoverEl.style.display = 'none';
  chartEl.appendChild(_ohlcHoverEl);

  try {
    chart.subscribeCrosshairMove((param) => {
      if (!_ohlcHoverEl) return;
      if (!param || !param.time || !param.seriesData) {
        // Mouse left the chart or no data — hide
        _ohlcHoverEl.style.display = 'none';
        return;
      }
      const candle = param.seriesData.get(candleSeries);
      if (!candle) {
        _ohlcHoverEl.style.display = 'none';
        return;
      }
      const { open, high, low, close } = candle;
      const delta = close - open;
      const pct = open > 0 ? (delta / open) * 100 : 0;
      const clsDelta = delta >= 0 ? 'ohlc-up' : 'ohlc-dn';
      const sign = delta >= 0 ? '+' : '';

      // Volume — lookup via _backfillCandles for this time
      let volStr = '';
      if (Array.isArray(_backfillCandles) && _backfillCandles.length > 0) {
        const t = Number(param.time);
        // Binary search for matching candle
        const match = _backfillCandles.find((c) => Number(c.time) === t);
        if (match && match.volume != null) {
          const v = Number(match.volume);
          if (isFinite(v) && v > 0) {
            const vs = v >= 1e6 ? (v / 1e6).toFixed(2) + 'M'
                     : v >= 1e3 ? (v / 1e3).toFixed(2) + 'K'
                     : v.toFixed(2);
            volStr = `<span class="ohlc-label">成交量</span><span class="ohlc-val">${vs}</span>`;
          }
        }
      }

      _ohlcHoverEl.innerHTML = `
        <span class="ohlc-label">开</span><span class="ohlc-val">${_fmtPriceHover(open)}</span>
        <span class="ohlc-label">高</span><span class="ohlc-val">${_fmtPriceHover(high)}</span>
        <span class="ohlc-label">低</span><span class="ohlc-val">${_fmtPriceHover(low)}</span>
        <span class="ohlc-label">收</span><span class="ohlc-val ${clsDelta}">${_fmtPriceHover(close)}</span>
        <span class="ohlc-val ${clsDelta}">${sign}${delta.toFixed(Math.abs(delta) < 1 ? 5 : 3)}</span>
        <span class="ohlc-val ${clsDelta}">(${sign}${pct.toFixed(2)}%)</span>
        ${volStr}
      `;
      _ohlcHoverEl.style.display = '';
    });
  } catch (err) {
    console.warn('[chart] ohlc hover init failed', err);
  }
}

function updateHeader(symbol, interval, price) {
  const header = $('#chart-header-v2');
  if (!header) return;
  const historyModeLabel = marketState.historyMode === 'full_history' ? 'FULL' : 'FAST';
  const scaleLabel = marketState.currentScale === 'log' ? 'LOG' : 'LIN';
  // UTC-day change (since UTC 00:00 today)
  const candles = _backfillCandles.length > 0 ? _backfillCandles : [];
  const utcChg = _computeUtcDayChange(candles, price);
  const chgTxt = utcChg != null
    ? ` · <span style="color:${utcChg >= 0 ? '#00e676' : '#ff5252'}">${utcChg >= 0 ? '+' : ''}${utcChg.toFixed(2)}% UTC</span>`
    : '';
  const tzTxt = ` · <span class="chart-tz-picker" style="cursor:pointer;color:#94a3b8;text-decoration:underline" title="点击切换时区">${_tzLabel(_currentTz)}</span>`;
  header.innerHTML = `${symbol} · ${interval} · ${historyModeLabel}/${scaleLabel} · $${formatPrice(price)}${chgTxt}${tzTxt}`;
  // Wire timezone picker click once
  const picker = header.querySelector('.chart-tz-picker');
  if (picker && !picker._wired) {
    picker._wired = true;
    picker.addEventListener('click', (ev) => {
      ev.stopPropagation();
      // Simple cycle: la → bj → utc → la
      const next = _currentTz === 'la' ? 'bj' : _currentTz === 'bj' ? 'utc' : 'la';
      setChartTimezone(next);
      // Re-render header
      updateHeader(symbol, interval, price);
    });
  }
}

export function startLiveUpdates(intervalMs = 10000) {
  stopLiveUpdates();
  // Full OHLCV reload on a slow cadence — this handles new bars rolling
  // over and keeps historical data honest.
  liveTimer = setInterval(() => {
    if (!shouldReloadFullCandles()) return;
    void loadCurrent().catch((err) => console.warn('[chart] live update failed:', err));
  }, intervalMs);
  // Direct Bitget WebSocket ticker — tick-level price updates, no polling.
  startTickerWS(marketState.currentSymbol, onTickerTick);
  // Re-subscribe when symbol changes
  subscribe('market.symbol.changed', (sym) => setTickerSymbol(sym));
  // 2026-04-23: IndexedDB flow — when the service fires a background
  // delta refresh after an instant IDB paint, repaint the chart so the
  // newest bar catches up. loadCurrent() hits the now-warm hot cache
  // (no server round-trip); it's cheap.
  subscribe('ohlcv.delta', (evt) => {
    if (!evt || evt.symbol !== marketState.currentSymbol) return;
    if (evt.interval !== marketState.currentInterval) return;
    void loadCurrent().catch((err) => console.warn('[chart] delta repaint failed:', err));
  });
}

export function stopLiveUpdates() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  stopTickerWS();
}

function shouldReloadFullCandles() {
  const now = Date.now() / 1000;
  const candles = marketState.lastCandles || [];
  if (!candles.length || !_lastFullReloadTs) return true;
  if (now - _lastFullReloadTs >= 300) return true;
  const last = candles[candles.length - 1];
  const barSec = barIntervalSec();
  return now >= Number(last.time) + barSec + 5;
}

function onTickerTick(tick) {
  if (!tick || tick.symbol !== marketState.currentSymbol) return;
  const price = tick.markPrice || tick.lastPrice;
  if (!isFinite(price) || price <= 0) return;

  // Header price
  updateHeader(marketState.currentSymbol, marketState.currentInterval, price);

  // Update the last candle's close in place — tradingview-style tick update.
  if (candleSeries && marketState.lastCandles?.length) {
    const last = marketState.lastCandles[marketState.lastCandles.length - 1];
    const updated = {
      time: last.time,
      open: last.open,
      high: Math.max(last.high, price),
      low: Math.min(last.low, price),
      close: price,
    };
    try { candleSeries.update(updated); } catch {}
    last.close = updated.close;
    last.high = updated.high;
    last.low = updated.low;
  }
}

export function getChart() { return chart; }
export function getCandleSeries() { return candleSeries; }

/**
 * Lazy backfill: fetch ~500 older bars and prepend to the chart.
 * Called when the user scrolls the visible range near the leftmost
 * loaded bar. Guarded against concurrent calls and exhaustion.
 */
async function _doBackfill() {
  if (_backfillInFlight || _backfillExhausted) return;
  if (_backfillCandles.length === 0) return;
  const sym = _backfillSymbol;
  const ivl = _backfillInterval;
  if (!sym || !ivl) return;
  const earliest = _backfillCandles[0].time;
  if (!earliest) return;

  _backfillInFlight = true;
  try {
    const resp = await marketSvc.getOhlcvBackfill(sym, ivl, earliest, 300);
    // Guard: ensure user hasn't switched symbol/TF mid-fetch
    if (sym !== marketState.currentSymbol || ivl !== marketState.currentInterval) {
      return;
    }
    const older = Array.isArray(resp?.candles) ? resp.candles : [];
    if (older.length === 0) {
      _backfillExhausted = true;
      return;
    }
    // Normalize + dedupe — backfill may return a bar that's already in
    // our buffer if Bitget returned inclusive-of-boundary.
    const newCandles = older.map((c) => ({
      time: typeof c.time === 'string' ? Math.floor(new Date(c.time).getTime() / 1000) : c.time,
      open: Number(c.open),
      high: Number(c.high),
      low: Number(c.low),
      close: Number(c.close),
    }));
    const newVolume = (resp.volume || []).map((v) => ({
      time: typeof v.time === 'string' ? Math.floor(new Date(v.time).getTime() / 1000) : v.time,
      value: Number(v.value || 0),
      color: v.color || 'rgba(0,230,118,0.4)',
    }));
    const seen = new Set(_backfillCandles.map((c) => c.time));
    const uniqCandles = newCandles.filter((c) => !seen.has(c.time));
    const uniqVolume = newVolume.filter((v) => !seen.has(v.time));
    if (uniqCandles.length === 0) {
      _backfillExhausted = true;
      return;
    }
    _backfillCandles = [...uniqCandles, ..._backfillCandles];
    _backfillVolume = [...uniqVolume, ..._backfillVolume];
    // Prepending via setData triggers a reflow; preserve visible range
    // so the user's viewport doesn't jump.
    const preserved = chart.timeScale().getVisibleRange?.();
    candleSeries.setData(_backfillCandles);
    if (_backfillVolume.length > 0) volumeSeries.setData(_backfillVolume);
    if (preserved) {
      try { chart.timeScale().setVisibleRange(preserved); } catch {}
    }
    // Re-compute overlays client-side so the MA lines extend over the new bars
    try {
      const overlays = computeOverlaysFromCandles(_backfillCandles);
      if (Object.keys(overlays).length > 0) {
        drawMAOverlays(chart, overlays, _backfillCandles.map((c) => c.time));
      }
    } catch {}
    console.log(`[chart] backfilled ${uniqCandles.length} older bars (total now ${_backfillCandles.length})`);
  } catch (err) {
    // Any failure (including 500 from backend when Bitget has no older
    // history) must mark exhausted — otherwise every subsequent scroll-
    // left fires another doomed request, which is how we'd flood the
    // server with 500s on a freshly listed coin.
    console.warn('[chart] backfill failed, marking exhausted:', err?.message || err);
    _backfillExhausted = true;
  } finally {
    _backfillInFlight = false;
  }
}
