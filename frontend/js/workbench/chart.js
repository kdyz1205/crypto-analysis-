// frontend/js/workbench/chart.js — minimal LightweightCharts wrapper using services+state

import { $ } from '../util/dom.js';
import { marketState, setCandles, setPrecision } from '../state/market.js';
import { strategyState, setStrategyConfig, setStrategyError, setStrategySnapshot, clearStrategySnapshot, clearStrategyReplay, getCurrentStrategySnapshot, setStrategyLayerVisible } from '../state/strategy.js';
import { subscribe } from '../util/events.js';
import * as marketSvc from '../services/market.js';
import * as patternsSvc from '../services/patterns.js';
import * as strategySvc from '../services/strategy.js';
import { inferPrecision, formatPrice } from '../util/format.js';
import { markBoot } from '../ui/boot_status.js';
import { drawMAOverlays, toggleMAOverlays as toggleMA } from './ma_overlay.js';
import { drawPatterns, drawZones, clearPatternLines } from './patterns.js';
import { clearTrendlineOverlay, drawTrendlineOverlay } from './overlays/trendline_overlay.js';
import { clearSignalOverlay, drawSignalOverlay } from './overlays/signal_overlay.js';
import { clearOrderOverlay, drawOrderOverlay } from './overlays/order_overlay.js';

let chart = null;
let candleSeries = null;
let volumeSeries = null;
let liveTimer = null;
let strategyLayerPanel = null;
let strategyRequestSeq = 0;
let patternRequestSeq = 0;
let lastPatternKey = null;
let lastStrategyConfigKey = null;
let chartLoadSeq = 0;
let overlayScheduleSeq = 0;
let overlayTimer = null;
let strategyAbortController = null;
let patternAbortController = null;
const OVERLAY_REQUEST_TIMEOUT_MS = 4000;

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
    rightPriceScale: { borderColor: '#2a3548' },
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

  window.addEventListener('resize', () => {
    if (!chart || !el) return;
    chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
  });

  ensureStrategyLayerPanel(el.parentElement || el);

  subscribe('market.symbol.changed', () => {
    void loadCurrent(true).catch((err) => console.warn('[chart] symbol refresh failed:', err));
  });
  subscribe('market.interval.changed', () => {
    void loadCurrent(true).catch((err) => console.warn('[chart] interval refresh failed:', err));
  });
  subscribe('strategy.snapshot.updated', () => renderStrategyOverlays());
  subscribe('strategy.layers.changed', () => {
    syncStrategyLayerPanel();
    renderStrategyOverlays();
  });

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

    const data = await marketSvc.getOhlcv(currentSymbol, currentInterval, 365);
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
    const lastPrice = candles[candles.length - 1].close;
    setPrecision(data.pricePrecision ?? inferPrecision(lastPrice));

    if (data.overlays) {
      const candleTimes = candles.map((c) => c.time);
      drawMAOverlays(chart, data.overlays, candleTimes);
    }

    updateHeader(currentSymbol, currentInterval, lastPrice);
    console.log(`[chart] loaded ${candles.length} candles for ${currentSymbol} ${currentInterval}`);

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
    renderStrategyOverlays();
    console.error('[chart] load failed:', err);
    throw err;
  }
}

async function loadPatterns(symbol, interval, overlayContext = null) {
  const requestId = ++patternRequestSeq;
  patternAbortController?.abort();
  patternAbortController = new AbortController();
  try {
    const data = await patternsSvc.getPatterns(symbol, interval, 90, 'full', null, {
      timeout: OVERLAY_REQUEST_TIMEOUT_MS,
      signal: patternAbortController.signal,
    });
    if (requestId !== patternRequestSeq || !isOverlayContextCurrent(overlayContext)) {
      return { ok: false, stale: true };
    }
    clearPatternLines(chart);
    drawPatterns(chart, data.supportLines || [], data.resistanceLines || [], 8);
    drawZones(chart, data.consolidationZones || []);
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
  strategyAbortController?.abort();
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
  }, 2000);
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
  strategyAbortController?.abort();
  patternAbortController?.abort();
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
  if (!snapshot) {
    clearTrendlineOverlay(chart);
    clearSignalOverlay(candleSeries);
    clearOrderOverlay(chart);
    return;
  }

  drawTrendlineOverlay(chart, snapshot, strategyState.layerVisibility);
  drawSignalOverlay(candleSeries, snapshot, strategyState.layerVisibility);
  drawOrderOverlay(chart, snapshot, strategyState.layerVisibility);
}

function ensureStrategyLayerPanel(container) {
  if (!container || strategyLayerPanel) return;
  strategyLayerPanel = document.createElement('div');
  strategyLayerPanel.className = 'strategy-layer-panel';
  strategyLayerPanel.innerHTML = `
    <div class="strategy-layer-title">Strategy Layers</div>
    <div class="strategy-layer-grid">
      <label class="strategy-layer-option"><input type="checkbox" data-layer="trendlines" checked /> Trendlines</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="touchMarkers" checked /> Touches</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="projectedLine" checked /> Projection</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="signalMarkers" checked /> Signals</label>
      <label class="strategy-layer-option"><input type="checkbox" data-layer="invalidationMarkers" checked /> Invalidations</label>
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

export function toggleMAOverlays() {
  return toggleMA();
}

function updateHeader(symbol, interval, price) {
  const header = $('#chart-header-v2');
  if (header) {
    header.textContent = `${symbol} · ${interval} · $${formatPrice(price)}`;
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
