// frontend/js/workbench/indicators/indicator_math.js
//
// Pure math for client-side indicator computation. All functions take
// an array of candle objects {time, open, high, low, close, volume} and
// return series of {time, value} pairs aligned with the input length
// (nulls for warmup bars). Kept framework-free so they're trivial to
// unit-test and reuse.

export function computeSMA(values, period) {
  const out = new Array(values.length).fill(null);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}

export function computeEMA(values, period) {
  const out = new Array(values.length).fill(null);
  const k = 2 / (period + 1);
  let prev = null;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (prev == null) {
      // Seed with SMA of first `period` values.
      if (i >= period - 1) {
        let s = 0;
        for (let j = i - period + 1; j <= i; j++) s += values[j];
        prev = s / period;
        out[i] = prev;
      }
    } else {
      prev = v * k + prev * (1 - k);
      out[i] = prev;
    }
  }
  return out;
}

// RSI (14) using Wilder smoothing.
export function computeRSI(closes, period = 14) {
  const n = closes.length;
  const out = new Array(n).fill(null);
  if (n < period + 1) return out;
  let gainSum = 0;
  let lossSum = 0;
  for (let i = 1; i <= period; i++) {
    const d = closes[i] - closes[i - 1];
    if (d >= 0) gainSum += d; else lossSum -= d;
  }
  let avgGain = gainSum / period;
  let avgLoss = lossSum / period;
  out[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  for (let i = period + 1; i < n; i++) {
    const d = closes[i] - closes[i - 1];
    const g = d > 0 ? d : 0;
    const l = d < 0 ? -d : 0;
    avgGain = (avgGain * (period - 1) + g) / period;
    avgLoss = (avgLoss * (period - 1) + l) / period;
    out[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return out;
}

// MACD: fast EMA - slow EMA, signal = EMA(macd, signalPeriod), hist = macd - signal.
export function computeMACD(closes, fast = 12, slow = 26, signalPeriod = 9) {
  const fastEma = computeEMA(closes, fast);
  const slowEma = computeEMA(closes, slow);
  const macd = closes.map((_, i) =>
    fastEma[i] != null && slowEma[i] != null ? fastEma[i] - slowEma[i] : null
  );
  // Signal = EMA of macd (skipping nulls)
  const firstValid = macd.findIndex((v) => v != null);
  const signal = new Array(closes.length).fill(null);
  if (firstValid >= 0) {
    const sliced = macd.slice(firstValid).map((v) => v ?? 0);
    const sig = computeEMA(sliced, signalPeriod);
    for (let i = 0; i < sig.length; i++) signal[i + firstValid] = sig[i];
  }
  const hist = macd.map((v, i) =>
    v != null && signal[i] != null ? v - signal[i] : null
  );
  return { macd, signal, hist };
}

// Volume MA — straight SMA on the volume series.
export function computeVolumeMA(candles, period = 20) {
  const vols = candles.map((c) => Number(c.volume) || 0);
  return computeSMA(vols, period);
}

// ─── Factor signals (boolean per bar) ──────────────────────────
// These are binary conditions (e.g. "RSI < 30") used for markers on
// the chart. Each returns an array of booleans aligned with candles.

export function factorRSIOversold(candles, period = 14, threshold = 30) {
  const rsi = computeRSI(candles.map((c) => c.close), period);
  return rsi.map((v) => v != null && v < threshold);
}

export function factorRSIOverbought(candles, period = 14, threshold = 70) {
  const rsi = computeRSI(candles.map((c) => c.close), period);
  return rsi.map((v) => v != null && v > threshold);
}

export function factorVolumeSurge(candles, period = 20, multiple = 2.0) {
  const vols = candles.map((c) => Number(c.volume) || 0);
  const avg = computeSMA(vols, period);
  return vols.map((v, i) => avg[i] != null && avg[i] > 0 && v > avg[i] * multiple);
}

export function factorMACDBullCross(candles, fast = 12, slow = 26, signal = 9) {
  const closes = candles.map((c) => c.close);
  const { macd, signal: sig } = computeMACD(closes, fast, slow, signal);
  const out = new Array(candles.length).fill(false);
  for (let i = 1; i < candles.length; i++) {
    if (macd[i] == null || sig[i] == null || macd[i - 1] == null || sig[i - 1] == null) continue;
    if (macd[i - 1] <= sig[i - 1] && macd[i] > sig[i]) out[i] = true;
  }
  return out;
}

export function factorMACDBearCross(candles, fast = 12, slow = 26, signal = 9) {
  const closes = candles.map((c) => c.close);
  const { macd, signal: sig } = computeMACD(closes, fast, slow, signal);
  const out = new Array(candles.length).fill(false);
  for (let i = 1; i < candles.length; i++) {
    if (macd[i] == null || sig[i] == null || macd[i - 1] == null || sig[i - 1] == null) continue;
    if (macd[i - 1] >= sig[i - 1] && macd[i] < sig[i]) out[i] = true;
  }
  return out;
}

// ATR (Wilder). Returns the volatility envelope — useful for eyeballing
// whether a candle is "big" or "small" for its regime.
export function computeATR(candles, period = 14) {
  const n = candles.length;
  const tr = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    const c = candles[i];
    if (i === 0) { tr[i] = c.high - c.low; continue; }
    const prevClose = candles[i - 1].close;
    tr[i] = Math.max(
      c.high - c.low,
      Math.abs(c.high - prevClose),
      Math.abs(c.low - prevClose),
    );
  }
  const out = new Array(n).fill(null);
  if (n < period) return out;
  let sum = 0;
  for (let i = 0; i < period; i++) sum += tr[i];
  out[period - 1] = sum / period;
  for (let i = period; i < n; i++) {
    out[i] = (out[i - 1] * (period - 1) + tr[i]) / period;
  }
  return out;
}
