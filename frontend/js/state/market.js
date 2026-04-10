// frontend/js/state/market.js
import { publish } from '../util/events.js';

export const marketState = {
  currentSymbol: 'HYPEUSDT',
  currentInterval: '4h',
  allSymbols: [],
  lastCandles: [],
  pricePrecision: null,
  historyMode: 'fast_window',
  historyMeta: null,
  currentScale: 'linear',  // 'linear' | 'log'
  magnetMode: 'weak',      // 'weak' | 'strong'
  replayEndTime: null,
  liveUpdateInterval: null,
};

export function setSymbol(sym) {
  if (!sym || marketState.currentSymbol === sym) return;
  marketState.currentSymbol = sym;
  publish('market.symbol.changed', sym);
}

export function setIntervalTF(iv) {
  if (!iv || marketState.currentInterval === iv) return;
  marketState.currentInterval = iv;
  publish('market.interval.changed', iv);
}

export function setScale(scale) {
  if (marketState.currentScale === scale) return;
  marketState.currentScale = scale;
  publish('market.scale.changed', scale);
}

export function setHistoryMode(mode) {
  if (!mode || marketState.historyMode === mode) return;
  marketState.historyMode = mode;
  publish('market.history_mode.changed', mode);
}

export function setHistoryMeta(meta) {
  marketState.historyMeta = meta || null;
  publish('market.history_meta.changed', marketState.historyMeta);
}

export function setMagnet(mode) {
  if (marketState.magnetMode === mode) return;
  marketState.magnetMode = mode;
  publish('market.magnet.changed', mode);
}

export function setReplayEnd(ts) {
  marketState.replayEndTime = ts;
  publish('market.replay.changed', ts);
}

export function setAllSymbols(list) {
  marketState.allSymbols = list || [];
  publish('market.symbols.loaded', marketState.allSymbols);
}

export function setCandles(candles) {
  marketState.lastCandles = candles || [];
}

export function setPrecision(p) {
  marketState.pricePrecision = p;
}
