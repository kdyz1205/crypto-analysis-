// frontend/js/workbench/chart.js — minimal LightweightCharts wrapper using services+state

import { $ } from '../util/dom.js';
import { marketState, setCandles, setPrecision, setIntervalTF, setSymbol } from '../state/market.js';
import { subscribe } from '../util/events.js';
import * as marketSvc from '../services/market.js';
import * as patternsSvc from '../services/patterns.js';
import { inferPrecision, formatPrice } from '../util/format.js';
import { drawMAOverlays, toggleMAOverlays as toggleMA } from './ma_overlay.js';
import { drawPatterns, drawZones, clearPatternLines } from './patterns.js';

let chart = null;
let candleSeries = null;
let volumeSeries = null;
let liveTimer = null;

export function initChart(containerId = 'chart-container') {
  const el = $('#' + containerId);
  if (!el) {
    console.error('[chart] container not found:', containerId);
    return;
  }
  if (typeof LightweightCharts === 'undefined') {
    console.error('[chart] LightweightCharts library not loaded');
    return;
  }

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

  // React to state changes
  subscribe('market.symbol.changed', () => loadCurrent());
  subscribe('market.interval.changed', () => loadCurrent());

  return chart;
}

export async function loadCurrent() {
  const { currentSymbol, currentInterval } = marketState;
  if (!candleSeries) return;

  try {
    // API returns: { candles: [{time,open,high,low,close}], volume: [{time,value}], overlays, pricePrecision }
    const data = await marketSvc.getOhlcv(currentSymbol, currentInterval, 365);

    const rawCandles = data.candles || [];
    if (!Array.isArray(rawCandles) || rawCandles.length === 0) {
      console.warn('[chart] empty candles for', currentSymbol, currentInterval);
      return;
    }

    // LightweightCharts expects time as number (unix seconds) or 'YYYY-MM-DD' string
    const candles = rawCandles.map((c) => ({
      time: typeof c.time === 'string' ? Math.floor(new Date(c.time).getTime() / 1000) : c.time,
      open: Number(c.open),
      high: Number(c.high),
      low: Number(c.low),
      close: Number(c.close),
    }));

    // Volume data: aligned with candles, color based on close vs open
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

    // MA ribbon + BB overlays from same response
    if (data.overlays) {
      const candleTimes = candles.map((c) => c.time);
      drawMAOverlays(chart, data.overlays, candleTimes);
    }

    // Load S/R patterns in parallel
    loadPatterns(currentSymbol, currentInterval);

    updateHeader(currentSymbol, currentInterval, lastPrice);
    console.log(`[chart] loaded ${candles.length} candles for ${currentSymbol} ${currentInterval}`);
  } catch (err) {
    console.error('[chart] load failed:', err);
  }
}

async function loadPatterns(symbol, interval) {
  try {
    const data = await patternsSvc.getPatterns(symbol, interval, 90, 'full');
    clearPatternLines(chart);
    drawPatterns(chart, data.supportLines || [], data.resistanceLines || [], 8);
    drawZones(chart, data.consolidationZones || []);
  } catch (err) {
    console.warn('[patterns] load failed:', err);
  }
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
  liveTimer = setInterval(() => loadCurrent(), intervalMs);
}

export function stopLiveUpdates() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
}

export function getChart() { return chart; }
