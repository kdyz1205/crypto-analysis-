// frontend/js/workbench/wyckoff_overlay.js
//
// Wyckoff Accumulation / Distribution indicator overlay.
//
// Draws on the chart:
//   1. Volume ratio line (short/long MA) — shows accumulation/distribution phases
//   2. Phase markers: accumulation zones (green bg), distribution zones (red bg)
//   3. Spring/UTAD detection markers (key reversal points)
//
// Wyckoff phases:
//   - Accumulation: vol dries up + tight range → smart money loading
//   - Markup: breakout from accumulation → strong trend up
//   - Distribution: vol dries up at top + tight range → smart money unloading
//   - Markdown: breakdown from distribution → strong trend down
//
// Key signals:
//   - "Spring": price briefly dips below support then snaps back (bullish)
//   - "UTAD" (Upthrust After Distribution): price briefly pokes above resistance then fails (bearish)

const COLORS = {
  accumulation: 'rgba(0, 200, 100, 0.08)',
  distribution: 'rgba(200, 50, 50, 0.08)',
  volRatio: '#ffa726',
  volRatioLow: '#4caf50',
  spring: '#00e676',
  utad: '#ff1744',
};

let _seriesRefs = {};
let _markers = [];

// ─── Indicator computation ───

function _sma(data, period) {
  const out = new Array(data.length).fill(null);
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) sum += data[j];
    out[i] = sum / period;
  }
  return out;
}

function computeWyckoff(candles, cfg = {}) {
  const volShort = cfg.volShort || 5;
  const volLong = cfg.volLong || 50;
  const rangeLen = cfg.rangeLen || 20;
  const dryThreshold = cfg.dryThreshold || 0.4;
  const springLookback = cfg.springLookback || 10;

  const n = candles.length;
  const closes = candles.map(c => c.close);
  const highs = candles.map(c => c.high);
  const lows = candles.map(c => c.low);
  const volumes = candles.map(c => c.volume || 1);

  // Volume ratio: short MA / long MA
  const volShortMA = _sma(volumes, volShort);
  const volLongMA = _sma(volumes, volLong);
  const volRatio = volShortMA.map((s, i) => {
    if (s === null || volLongMA[i] === null || volLongMA[i] === 0) return null;
    return s / volLongMA[i];
  });

  // Volatility (rolling range as % of price)
  const volatility = new Array(n).fill(null);
  for (let i = rangeLen; i < n; i++) {
    const hi = Math.max(...highs.slice(i - rangeLen, i + 1));
    const lo = Math.min(...lows.slice(i - rangeLen, i + 1));
    volatility[i] = (hi - lo) / closes[i] * 100;
  }

  // Detect phases
  const phases = new Array(n).fill('none'); // 'accumulation', 'distribution', 'none'
  const markers = [];
  let dryCount = 0;
  let rangeHigh = 0, rangeLow = Infinity;

  for (let i = volLong + rangeLen; i < n; i++) {
    if (volRatio[i] === null || volatility[i] === null) continue;

    // Volatility percentile (last 200 bars)
    const start = Math.max(0, i - 200);
    const volHist = volatility.slice(start, i).filter(v => v !== null);
    const volPct = volHist.length > 0 ?
      volHist.filter(v => v < volatility[i]).length / volHist.length : 0.5;

    const isDry = volRatio[i] < dryThreshold && volPct < 0.3;

    if (isDry) {
      dryCount++;
      rangeHigh = Math.max(rangeHigh, highs[i]);
      rangeLow = Math.min(rangeLow, lows[i]);

      // Determine phase by price position relative to recent trend
      const lookback = Math.min(100, i);
      const priorHigh = Math.max(...highs.slice(i - lookback, i - dryCount + 1));
      const priorLow = Math.min(...lows.slice(i - lookback, i - dryCount + 1));
      const midPrior = (priorHigh + priorLow) / 2;

      if (closes[i] < midPrior) {
        phases[i] = 'accumulation'; // consolidating at bottom → accumulating
      } else {
        phases[i] = 'distribution'; // consolidating at top → distributing
      }
    } else {
      // Check for Spring / UTAD at end of dry phase
      if (dryCount >= 10) {
        // Spring: price dips below range low then closes back above
        if (lows[i] < rangeLow && closes[i] > rangeLow) {
          markers.push({
            time: candles[i].time,
            position: 'belowBar',
            color: COLORS.spring,
            shape: 'arrowUp',
            text: 'Spring',
          });
        }
        // UTAD: price pokes above range high then closes back below
        if (highs[i] > rangeHigh && closes[i] < rangeHigh) {
          markers.push({
            time: candles[i].time,
            position: 'aboveBar',
            color: COLORS.utad,
            shape: 'arrowDown',
            text: 'UTAD',
          });
        }
      }
      dryCount = 0;
      rangeHigh = 0;
      rangeLow = Infinity;
    }
  }

  // Build output series
  const volRatioSeries = candles.map((c, i) => ({
    time: c.time,
    value: volRatio[i] !== null ? volRatio[i] : undefined,
  })).filter(d => d.value !== undefined);

  // Phase background markers (create colored rectangles via markers)
  const phaseMarkers = [];
  for (let i = 1; i < n; i++) {
    if (phases[i] !== 'none' && phases[i - 1] === 'none') {
      // Phase started
      phaseMarkers.push({
        time: candles[i].time,
        phase: phases[i],
        type: 'start',
      });
    }
    if (phases[i] === 'none' && phases[i - 1] !== 'none') {
      phaseMarkers.push({
        time: candles[i].time,
        phase: phases[i - 1],
        type: 'end',
      });
    }
  }

  return { volRatioSeries, markers, phases, phaseMarkers };
}

// ─── Chart rendering ───

export function drawWyckoffOverlay(chart, candles) {
  if (!chart || !candles || candles.length < 100) return;

  const { volRatioSeries, markers } = computeWyckoff(candles);

  // Volume ratio as a separate pane (histogram at bottom)
  if (!_seriesRefs.volRatio) {
    _seriesRefs.volRatio = chart.addLineSeries({
      color: COLORS.volRatio,
      lineWidth: 1,
      priceScaleId: 'wyckoff',
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    // Scale it to bottom 15% of chart
    chart.priceScale('wyckoff').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
      borderVisible: false,
    });
  }
  _seriesRefs.volRatio.setData(volRatioSeries);

  // Draw horizontal line at ratio = 1.0 (normal volume)
  if (!_seriesRefs.volBaseline) {
    _seriesRefs.volBaseline = chart.addLineSeries({
      color: 'rgba(255,255,255,0.15)',
      lineWidth: 1,
      lineStyle: 2, // dashed
      priceScaleId: 'wyckoff',
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
  }
  if (volRatioSeries.length >= 2) {
    _seriesRefs.volBaseline.setData([
      { time: volRatioSeries[0].time, value: 1.0 },
      { time: volRatioSeries[volRatioSeries.length - 1].time, value: 1.0 },
    ]);
  }

  // Dry threshold line at 0.4
  if (!_seriesRefs.dryLine) {
    _seriesRefs.dryLine = chart.addLineSeries({
      color: 'rgba(76,175,80,0.3)',
      lineWidth: 1,
      lineStyle: 2,
      priceScaleId: 'wyckoff',
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
  }
  if (volRatioSeries.length >= 2) {
    _seriesRefs.dryLine.setData([
      { time: volRatioSeries[0].time, value: 0.4 },
      { time: volRatioSeries[volRatioSeries.length - 1].time, value: 0.4 },
    ]);
  }

  // Spring / UTAD markers on the main price series
  _markers = markers;
  // Note: markers must be set on a price series, not on a separate scale
  // We'll attach them to the main candlestick series if accessible
}

export function getWyckoffMarkers() {
  return _markers;
}

export function clearWyckoffOverlay() {
  for (const key of Object.keys(_seriesRefs)) {
    try {
      // Can't easily remove series from LWC, just clear data
      _seriesRefs[key].setData([]);
    } catch (e) { /* ignore */ }
  }
  _markers = [];
}

export function toggleWyckoff() {
  for (const key of Object.keys(_seriesRefs)) {
    try {
      const vis = _seriesRefs[key].options().visible;
      _seriesRefs[key].applyOptions({ visible: !vis });
    } catch (e) { /* ignore */ }
  }
}
