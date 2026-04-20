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
const FUTURE_DRAW_BARS = 60;

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

  subscribe('market.symbol.changed', () => {
    void loadCurrent(true).catch((err) => console.warn('[chart] symbol refresh failed:', err));
  });
  subscribe('market.interval.changed', () => {
    void loadCurrent(true).catch((err) => console.warn('[chart] interval refresh failed:', err));
  });
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
    // History windows per TF — targeting ~500-1000 initial bars per
    // call, similar to TradingView's initial viewport. Scroll-back
    // fetches additional history on demand (separate code path, TODO).
    // Each TF capped so one fetch = one Bitget page (max 500 bars),
    // so the backend finishes in ~1 request, not 9 paginated ones.
    //   1m  × 0.35d ≈ 504
    //   3m  × 1d    ≈ 480
    //   5m  × 2d    ≈ 576
    //   15m × 5d    ≈ 480
    //   30m × 10d   ≈ 480
    //   1h  × 21d   ≈ 504
    //   2h  × 42d   ≈ 504
    //   4h  × 84d   ≈ 504
    //   6h  × 125d  ≈ 500
    //   12h × 250d  ≈ 500
    //   1d  × 500d  = 500
    //   1w  × 3500d ≈ 500
    const tfDays = {
      '1m': 1, '3m': 1, '5m': 2, '15m': 5,
      '30m': 10, '1h': 21, '2h': 42,
      '4h': 84, '6h': 125, '12h': 250,
      '1d': 500, '1w': 3500,
    };
    const days = tfDays[currentInterval] || 21;

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

    // ── Viewport (only on FIRST load per symbol/tf) ────────────
    const fitKey = `${currentSymbol}:${currentInterval}`;
    if (_lastFitKey !== fitKey) {
      _lastFitKey = fitKey;
      const VISIBLE_BARS = 200;
      const ts = chart.timeScale();
      const totalBars = candles.length;
      const barDur = totalBars >= 2 ? candles[1].time - candles[0].time : 3600;

      const applyViewport = () => {
        try {
          if (totalBars > VISIBLE_BARS) {
            const fromTime = candles[totalBars - VISIBLE_BARS].time;
            const toTime   = candles[totalBars - 1].time + barDur * FUTURE_DRAW_BARS;
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

function updateHeader(symbol, interval, price) {
  const header = $('#chart-header-v2');
  if (header) {
    const historyModeLabel = marketState.historyMode === 'full_history' ? 'FULL' : 'FAST';
    const scaleLabel = marketState.currentScale === 'log' ? 'LOG' : 'LIN';
    header.textContent = `${symbol} · ${interval} · ${historyModeLabel}/${scaleLabel} · $${formatPrice(price)}`;
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
    console.warn('[chart] backfill failed', err);
  } finally {
    _backfillInFlight = false;
  }
}
