// frontend/js/workbench/chart.js — minimal LightweightCharts wrapper using services+state

import { $ } from '../util/dom.js';
import { marketState, setCandles, setHistoryMeta, setHistoryMode, setPrecision, setScale } from '../state/market.js';
import { strategyState, setStrategyConfig, setStrategyError, setStrategySnapshot, clearStrategySnapshot, clearStrategyReplay, getCurrentStrategySnapshot, setStrategyLayerVisible } from '../state/strategy.js';
import { drawingsState } from '../state/drawings.js';
import { publish, subscribe } from '../util/events.js';
import * as marketSvc from '../services/market.js';
import * as patternsSvc from '../services/patterns.js';
import * as strategySvc from '../services/strategy.js';
import { inferPrecision, formatPrice } from '../util/format.js';
import { markBoot } from '../ui/boot_status.js';
import { drawMAOverlays, toggleMAOverlays as toggleMA } from './ma_overlay.js';
import { drawPatterns, drawZones, drawHorizontalSRZones, clearHorizontalSRZones, clearPatternLines } from './patterns.js';
import { clearTrendlineOverlay, drawTrendlineOverlay } from './overlays/trendline_overlay.js';
import { clearSignalOverlay, drawSignalOverlay } from './overlays/signal_overlay.js';
import { clearOrderOverlay, drawOrderOverlay } from './overlays/order_overlay.js';
import { initManualTrendlineController, refreshManualDrawings, renderManualLines, getSuppressedAutoLineIds } from './drawings/manual_trendline_controller.js';

let chart = null;
let candleSeries = null;
let volumeSeries = null;
let liveTimer = null;
let strategyLayerPanel = null;
let chartModePanel = null;
let strategyRequestSeq = 0;
let patternRequestSeq = 0;
let lastPatternKey = null;
let lastStrategyConfigKey = null;
let chartLoadSeq = 0;
let overlayScheduleSeq = 0;
let overlayTimer = null;
let strategyAbortController = null;
let patternAbortController = null;
const OVERLAY_REQUEST_TIMEOUT_MS = 30000;

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
      autoScale: true,        // Y-axis auto-fits visible candles (like TradingView)
      scaleMargins: { top: 0.05, bottom: 0.05 },
    },
    timeScale: { borderColor: '#2a3548', timeVisible: true, secondsVisible: false },
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

  ensureStrategyLayerPanel(el.parentElement || el);
  ensureChartModePanel(el.parentElement || el);
  initManualTrendlineController(chart, el.parentElement || el);
  applyScaleMode();

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
    cancelDeferredOverlayLoad();
    abortOverlayRequests();

    // Smart data loading: short timeframes load less data to stay fast
    const tfDays = { '1m': 7, '3m': 14, '5m': 30, '15m': 60, '1h': 180, '4h': 730, '1d': 1095, '1w': 1095 };
    const days = tfDays[currentInterval] || 90;
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
    chart.timeScale().fitContent();

    setCandles(candles);
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
    setPrecision(data.pricePrecision ?? inferPrecision(lastPrice));

    if (data.overlays) {
      const candleTimes = candles.map((c) => c.time);
      drawMAOverlays(chart, data.overlays, candleTimes);
    }

    updateHeader(currentSymbol, currentInterval, lastPrice);
    // Loaded silently — no console spam
    publish('chart.load.succeeded', {
      symbol: currentSymbol,
      interval: currentInterval,
      loadSeq,
    });

    const patternKey = `${currentSymbol}:${currentInterval}`;
    const shouldLoadPatterns = forcePatterns || lastPatternKey !== patternKey;
    if (shouldLoadPatterns) {
      lastPatternKey = patternKey;
    }
    scheduleDeferredOverlayLoad({
      loadSeq,
      symbol: currentSymbol,
      interval: currentInterval,
      shouldLoadPatterns,
    });
    void refreshManualDrawings(currentSymbol, currentInterval).catch((err) => console.warn('[drawings] refresh failed:', err));

    return {
      ok: true,
      stale: false,
      symbol: currentSymbol,
      interval: currentInterval,
      shouldLoadPatterns,
    };
  } catch (err) {
    if (!isChartLoadCurrent(loadSeq, currentSymbol, currentInterval)) {
      return { ok: false, stale: true };
    }
    clearPatternLines(chart);
    clearStrategySnapshot();
    clearStrategyReplay();
    setStrategyError(err?.message || String(err));
    setHistoryMeta(null);
    renderStrategyOverlays();
    console.error('[chart] load failed:', err);
    throw err;
  }
}

async function loadPatterns(symbol, interval, overlayContext = null) {
  const requestId = ++patternRequestSeq;
  // Don't abort in-flight — requestId check drops stale responses.
  patternAbortController = new AbortController();
  try {
    const data = await patternsSvc.getPatterns(symbol, interval, 90, 'full', null, {
      timeout: OVERLAY_REQUEST_TIMEOUT_MS,
      signal: patternAbortController.signal,
    });
    if (requestId !== patternRequestSeq || !isOverlayContextCurrent(overlayContext)) {
      return { ok: false, stale: true };
    }
    // Pattern lines disabled — strategy overlay already draws S/R from snapshot.
    // Drawing both layers creates duplicate noise on the chart.
    clearPatternLines(chart);
    return { ok: true, stale: false };
  } catch (err) {
    if (requestId !== patternRequestSeq || err?.name === 'AbortError' || !isOverlayContextCurrent(overlayContext)) {
      return { ok: false, stale: true };
    }
    console.warn('[patterns] load failed:', err);
    throw err;
  }
}

async function loadStrategy(symbol, interval, overlayContext = null) {
  const requestId = ++strategyRequestSeq;
  // Don't abort in-flight requests — stale responses are filtered by the
  // requestId check below. Aborting causes ERR_ABORTED cascades when
  // multiple overlay loads are scheduled in rapid succession at boot.
  strategyAbortController = new AbortController();
  try {
    const configKey = `${symbol}:${interval}`;
    const shouldFetchConfig = lastStrategyConfigKey !== configKey;
    const [config, snapshotEnvelope] = await Promise.all([
      shouldFetchConfig
        ? strategySvc.getStrategyConfig(symbol, interval, {
          timeout: OVERLAY_REQUEST_TIMEOUT_MS,
          signal: strategyAbortController.signal,
        })
        : Promise.resolve(strategyState.config),
      strategySvc.getStrategySnapshot(
        symbol,
        interval,
        { analysisBars: 500 },
        {
          timeout: OVERLAY_REQUEST_TIMEOUT_MS,
          signal: strategyAbortController.signal,
        },
      ),
    ]);
    if (requestId !== strategyRequestSeq || !isOverlayContextCurrent(overlayContext)) {
      return { ok: false, stale: true };
    }
    if (shouldFetchConfig && config) {
      lastStrategyConfigKey = configKey;
      setStrategyConfig(config);
    }
    setStrategySnapshot(snapshotEnvelope);
    clearStrategyReplay();
    setStrategyError(null);
    renderStrategyOverlays();
    return { ok: true, stale: false };
  } catch (err) {
    if (requestId !== strategyRequestSeq || err?.name === 'AbortError' || !isOverlayContextCurrent(overlayContext)) {
      return { ok: false, stale: true };
    }
    console.warn('[strategy] load failed:', err);
    clearStrategySnapshot();
    setStrategyError(err?.message || String(err));
    renderStrategyOverlays();
    throw err;
  }
}

function scheduleDeferredOverlayLoad(overlayContext) {
  overlayScheduleSeq += 1;
  const scheduledContext = {
    ...overlayContext,
    overlaySeq: overlayScheduleSeq,
  };
  cancelDeferredOverlayLoad();
  markBoot('patterns', 'pending', `loading ${overlayContext.symbol} ${overlayContext.interval}`);
  overlayTimer = setTimeout(() => {
    overlayTimer = null;
    if (!isOverlayContextCurrent(scheduledContext)) return;
    void loadDeferredOverlays(scheduledContext);
  }, 100);
}

async function loadDeferredOverlays(overlayContext) {
  try {
    const results = await Promise.allSettled([
      loadStrategy(overlayContext.symbol, overlayContext.interval, overlayContext),
      overlayContext.shouldLoadPatterns
        ? loadPatterns(overlayContext.symbol, overlayContext.interval, overlayContext)
        : Promise.resolve({ ok: true, stale: false, skipped: true }),
    ]);

    if (!isOverlayContextCurrent(overlayContext)) return;

    const blockingFailure = results.find((result) => result.status === 'rejected');
    if (blockingFailure?.status === 'rejected') {
      markBoot('patterns', 'error', blockingFailure.reason?.message || String(blockingFailure.reason));
      return;
    }

    const settledValues = results
      .filter((result) => result.status === 'fulfilled')
      .map((result) => result.value);
    const activeValues = settledValues.filter((value) => !value?.stale);
    if (activeValues.some((value) => value?.ok)) {
      markBoot('patterns', 'ok', `${overlayContext.symbol} ${overlayContext.interval} loaded`);
    }
  } catch (err) {
    if (!isOverlayContextCurrent(overlayContext)) return;
    markBoot('patterns', 'error', err?.message || String(err));
  }
}

function cancelDeferredOverlayLoad() {
  if (overlayTimer) {
    clearTimeout(overlayTimer);
    overlayTimer = null;
  }
}

function abortOverlayRequests() {
  // No-op: stale responses are dropped by the requestId check inside
  // each loader. Explicit aborts cause ERR_ABORTED cascades at boot
  // when loadCurrent fires multiple times from subscribed events.
}

function isChartLoadCurrent(loadSeq, symbol, interval) {
  return (
    loadSeq === chartLoadSeq
    && symbol === marketState.currentSymbol
    && interval === marketState.currentInterval
  );
}

function isOverlayContextCurrent(overlayContext) {
  if (!overlayContext) return true;
  return (
    overlayContext.loadSeq === chartLoadSeq
    && overlayContext.overlaySeq === overlayScheduleSeq
    && overlayContext.symbol === marketState.currentSymbol
    && overlayContext.interval === marketState.currentInterval
  );
}

function renderStrategyOverlays() {
  const snapshot = getCurrentStrategySnapshot();
  if (!chart || !candleSeries) return;
  const viewMode = drawingsState.viewMode || 'mixed';
  if (!snapshot) {
    clearTrendlineOverlay(chart);
    clearSignalOverlay(candleSeries);
    clearOrderOverlay(chart);
    renderManualLines();
    return;
  }

  const suppressedAutoLineIds = getSuppressedAutoLineIds();
  const filteredSnapshot = {
    ...snapshot,
    candidate_lines: viewMode === 'manual_only'
      ? []
      : (snapshot.candidate_lines || []).filter((line) => !suppressedAutoLineIds.has(line.line_id)),
    active_lines: viewMode === 'manual_only'
      ? []
      : (snapshot.active_lines || []).filter((line) => !suppressedAutoLineIds.has(line.line_id)),
    touch_points: viewMode === 'manual_only'
      ? []
      : (snapshot.touch_points || []).filter((point) => !suppressedAutoLineIds.has(point.line_id)),
    signals: viewMode === 'manual_only'
      ? []
      : (snapshot.signals || []).filter((signal) => !suppressedAutoLineIds.has(signal.line_id)),
    invalidations: viewMode === 'manual_only'
      ? []
      : (snapshot.invalidations || []).filter((item) => !suppressedAutoLineIds.has(item.line_id)),
  };

  const candleTimes = (marketState.lastCandles || []).map((c) => c.time);
  drawTrendlineOverlay(chart, filteredSnapshot, strategyState.layerVisibility, candleTimes);
  drawSignalOverlay(candleSeries, filteredSnapshot, strategyState.layerVisibility);
  drawOrderOverlay(chart, filteredSnapshot, strategyState.layerVisibility);

  // Draw horizontal S/R zones — show top zones by strength, max 3 per side
  const srZones = (snapshot.horizontal_zones || [])
    .filter((z) => z.strength > 25)
    .sort((a, b) => b.strength - a.strength);
  const supportZones = srZones.filter((z) => z.side === 'support').slice(0, 3);
  const resistanceZones = srZones.filter((z) => z.side === 'resistance').slice(0, 3);
  const filteredZones = [...supportZones, ...resistanceZones];
  if (filteredZones.length > 0) {
    drawHorizontalSRZones(chart, candleSeries, filteredZones);
  } else {
    clearHorizontalSRZones(chart);
  }

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
  liveTimer = setInterval(() => {
    void loadCurrent().catch((err) => console.warn('[chart] live update failed:', err));
  }, intervalMs);
}

export function stopLiveUpdates() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
}

export function getChart() { return chart; }
