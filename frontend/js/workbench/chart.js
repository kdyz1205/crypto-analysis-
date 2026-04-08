// frontend/js/workbench/chart.js — minimal LightweightCharts wrapper using services+state

import { $ } from '../util/dom.js';
import { marketState, setCandles, setPrecision, setIntervalTF, setSymbol } from '../state/market.js';
import { subscribe } from '../util/events.js';
import * as marketSvc from '../services/market.js';
import { inferPrecision, formatPrice } from '../util/format.js';

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
    const data = await marketSvc.getOhlcv(currentSymbol, currentInterval, 365);
    const ohlcv = data.ohlcv || data;
    if (!Array.isArray(ohlcv) || ohlcv.length === 0) {
      console.warn('[chart] empty data');
      return;
    }

    const candles = ohlcv.map((c) => ({
      time: c.time || c[0],
      open: c.open ?? c[1],
      high: c.high ?? c[2],
      low: c.low ?? c[3],
      close: c.close ?? c[4],
    }));
    const volumes = ohlcv.map((c) => ({
      time: c.time || c[0],
      value: c.volume ?? c[5] ?? 0,
      color: (c.close ?? c[4]) >= (c.open ?? c[1]) ? 'rgba(0,230,118,0.3)' : 'rgba(255,23,68,0.3)',
    }));

    candleSeries.setData(candles);
    volumeSeries.setData(volumes);
    chart.timeScale().fitContent();

    setCandles(candles);
    const lastPrice = candles[candles.length - 1].close;
    setPrecision(inferPrecision(lastPrice));

    // Update header
    updateHeader(currentSymbol, currentInterval, lastPrice);
  } catch (err) {
    console.error('[chart] load failed:', err);
  }
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
