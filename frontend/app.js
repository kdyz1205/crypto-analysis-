// ── API Base URL (empty = same origin, set for cross-origin deployment) ──
const API_BASE = window.__API_BASE || '';

// ── Global State ──
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let currentSymbol = 'HYPEUSDT';
let currentInterval = '4h';
let allSymbols = [];
let currentScale = 'linear'; // 'linear' or 'log'
let magnetMode = 'weak';     // 'weak' or 'strong'
let lastCandles = [];        // full candle array for magnet snapping
let pricePrecision = null;   // Price precision from exchange
let liveUpdateInterval = null; // Interval ID for live updates

// V6 MA overlay series
let maOverlaySeries = {};    // { ma5, ma8, ema21, ma55, bb_upper, bb_lower }
let maOverlayVisible = true; // toggle MA visibility

// Pattern overlay state
let patternLineSeries = [];  // LineSeries objects for trendlines + zones
let userDrawnLines = [];    // { type: 'trend'|'horizontal', t1, v1, t2, v2 } for re-draw after API refresh
let drawingMode = null;     // 'trend' | 'horizontal' | null
let pendingDrawPoint = null; // { time, value } when drawing trend line (first point)
let previewLineSeries = null; // preview line while dragging
let srVisible = true;
let backtestMarkers = [];    // Store current backtest markers for re-apply after chart reload
let maxSRLines = 0;          // 0 = show all
let lastCandle = null;
let rawPatternData = null;
let patternStatsData = null; // current-vs-history API response
let selectedLineIndex = null; // index into patternStatsData.current (S/R line)
let similarLineIndices = []; // indices of other lines in same class (for highlight)
let srLineSegments = [];     // { t1, v1, t2, v2, indexInCurrent, lineType } for hit-test
let srSeriesRefs = [];       // LineSeries refs by indexInCurrent (for style-only updates)
let srSeriesTypes = [];      // 'support' | 'resistance' by indexInCurrent
let patternResponseCache = new Map(); // key=params string, value=pattern response
const PATTERN_CACHE_MAX_SIZE = 50; // Evict oldest entries beyond this limit
let toolMode = null; // null | 'recognize' | 'draw' | 'assist'
const N_FUTURE_BARS = 4; // Trendline extends into future per spec

// ── MA Overlay Drawing ──
const MA_COLORS = {
    ma5:      { color: '#ffeb3b', width: 1 },   // yellow, thin
    ma8:      { color: '#ff9800', width: 1 },   // orange, thin
    ema21:    { color: '#2196f3', width: 2 },   // blue, medium
    ma55:     { color: '#e91e63', width: 2 },   // pink, medium
    bb_upper: { color: 'rgba(156,39,176,0.5)', width: 1, dash: true },  // purple dashed
    bb_lower: { color: 'rgba(156,39,176,0.5)', width: 1, dash: true },  // purple dashed
};

function drawMAOverlays(overlays) {
    if (!chart || !maOverlayVisible) return;

    // Remove old series
    for (const key of Object.keys(maOverlaySeries)) {
        try { chart.removeSeries(maOverlaySeries[key]); } catch (e) {}
    }
    maOverlaySeries = {};

    if (!overlays || Object.keys(overlays).length === 0) return;

    // Update MA legend text
    const legendEl = document.getElementById('ma-legend');
    if (legendEl) {
        const labels = { ma5: 'MA5', ma8: 'MA8', ema21: 'EMA21', ma55: 'MA55', bb_upper: 'BB↑', bb_lower: 'BB↓' };
        let html = '';
        for (const [key, arr] of Object.entries(overlays)) {
            if (!arr || arr.length === 0) continue;
            const last = arr[arr.length - 1];
            const style = MA_COLORS[key] || { color: '#888' };
            const lbl = labels[key] || key;
            html += `<span style="color:${style.color};margin-right:8px;">${lbl} ${last.value.toFixed(2)}</span> `;
        }
        legendEl.innerHTML = html;
    }

    for (const [key, data] of Object.entries(overlays)) {
        if (!data || data.length === 0) continue;
        const style = MA_COLORS[key] || { color: '#888', width: 1 };
        const series = chart.addLineSeries({
            color: style.color,
            lineWidth: style.width,
            lineStyle: style.dash ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });
        series.setData(data);
        maOverlaySeries[key] = series;
    }
}

function toggleMAOverlays() {
    maOverlayVisible = !maOverlayVisible;
    if (!maOverlayVisible) {
        for (const key of Object.keys(maOverlaySeries)) {
            try { chart.removeSeries(maOverlaySeries[key]); } catch (e) {}
        }
        maOverlaySeries = {};
    }
    // Will redraw on next data load or call loadData()
    const btn = document.getElementById('ma-toggle-btn');
    if (btn) btn.textContent = maOverlayVisible ? 'MA ✓' : 'MA ✗';
}

// ── Chart Initialization ──
function initChart() {
    const container = document.getElementById('chart-container');

    chart = LightweightCharts.createChart(container, {
        layout: {
            background: { color: '#191919' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#252a32' },
            horzLines: { color: '#252a32' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {
                labelBackgroundColor: '#2962ff',
            },
            horzLine: {
                labelBackgroundColor: '#2962ff',
            },
        },
        timeScale: {
            borderColor: '#363a45',
            timeVisible: true,
            secondsVisible: false,
            rightBarOffset: 0,
        },
        rightPriceScale: {
            borderColor: '#363a45',
            mode: LightweightCharts.PriceScaleMode.Normal,
        },
    });

    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderDownColor: '#ef5350',
        borderUpColor: '#26a69a',
        wickDownColor: '#ef5350',
        wickUpColor: '#26a69a',
    });

    volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
    });

    chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
    });

    // Crosshair OHLC legend
    chart.subscribeCrosshairMove(updateOHLCLegend);

    // Resize
    const ro = new ResizeObserver(() => {
        chart.applyOptions({
            width: container.clientWidth,
            height: container.clientHeight,
        });
    });
    ro.observe(container);

    container.addEventListener('click', onChartClick);
    container.addEventListener('mousemove', onChartMouseMove);
    container.addEventListener('mouseleave', onChartMouseLeave);
}

function getTimePriceFromEvent(event) {
    if (!chart || !candleSeries) return null;
    const container = document.getElementById('chart-container');
    const rect = container.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const time = chart.timeScale().coordinateToTime(x);
    if (time === null) return null;
    const price = candleSeries.coordinateToPrice(y);
    if (price === undefined || price === null) return null;
    const barIndex = timeToBarIndex(time);
    return { time, value: price, barIndex };
}

/** Chart-native: time -> bar_index (integer). Uses lastCandles. Binary search for O(log n). */
function timeToBarIndex(time) {
    if (!lastCandles?.length) return 0;
    const t = Number(time);
    let lo = 0, hi = lastCandles.length - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const midTime = Number(lastCandles[mid].time);
        if (midTime === t) return mid;
        if (midTime < t) lo = mid + 1;
        else hi = mid - 1;
    }
    // Return nearest
    if (lo >= lastCandles.length) return lastCandles.length - 1;
    if (lo === 0) return 0;
    return (t - Number(lastCandles[lo - 1].time)) < (Number(lastCandles[lo].time) - t) ? lo - 1 : lo;
}

/** Chart-native: bar_index -> time for rendering. */
function barIndexToTime(barIndex) {
    if (!lastCandles?.length) return barIndex;
    const i = Math.max(0, Math.min(barIndex, lastCandles.length - 1));
    return lastCandles[i].time;
}

/** Distance from point (time, value) to line segment (t1,v1)-(t2,v2) in price space; time outside segment adds penalty. */
function distanceToSegment(time, value, t1, v1, t2, v2) {
    const tMin = Math.min(t1, t2);
    const tMax = Math.max(t1, t2);
    const dt = tMax - tMin;
    const extend = dt ? Math.max(0, dt * 0.2) : 0;
    const tLo = tMin - extend;
    const tHi = tMax + extend;
    if (time < tLo || time > tHi) return Infinity;
    const priceAt = dt ? v1 + (v2 - v1) * (time - t1) / (t2 - t1) : v1;
    const priceDist = Math.abs(value - priceAt);
    const range = Math.abs(v2 - v1) || 1;
    return priceDist / range;
}

/** Return index into srLineSegments of line hit by (time, value), or null. */
function hitTestSRLine(time, value) {
    if (!srLineSegments.length) return null;
    // Dynamic threshold: use 0.5% of current price so hit-test works across all price scales
    const priceThreshold = value > 0 ? value * 0.005 : 0.05;
    let best = null;
    let bestDist = Infinity;
    for (const seg of srLineSegments) {
        // Compute price distance at the given time
        const dt = seg.t2 - seg.t1;
        const extend = dt ? Math.max(0, dt * 0.2) : 0;
        if (time < seg.t1 - extend || time > seg.t2 + extend) continue;
        const priceAt = dt ? seg.v1 + (seg.v2 - seg.v1) * (time - seg.t1) / dt : seg.v1;
        const dist = Math.abs(value - priceAt);
        if (dist < priceThreshold && dist < bestDist) {
            bestDist = dist;
            best = seg.indexInCurrent;
        }
    }
    return best;
}

function isRecognitionOverlayMode() {
    return toolMode === 'recognize' || toolMode === 'assist';
}

function onChartClick(event) {
    const tp = getTimePriceFromEvent(event);
    if (!tp) return;

    // DRAW mode: two-click for trendline (spec: first click P1, second click P2; no drag)
    if (toolMode === 'draw' && drawingMode) {
        if (drawingMode === 'trend') {
            if (!pendingDrawPoint) {
                pendingDrawPoint = tp; // First click: P1, enter DRAW_IN_PROGRESS; disable pan per spec
                if (chart) chart.applyOptions({ handleScroll: { mouseWheel: false, pressedMouseMove: false, horzTouchDrag: false, vertTouchDrag: false } });
                return;
            }
            // Second click: P2, finalize
            const x1 = pendingDrawPoint.barIndex ?? timeToBarIndex(pendingDrawPoint.time);
            const x2 = tp.barIndex ?? timeToBarIndex(tp.time);
            const y1 = pendingDrawPoint.value;
            const y2 = tp.value;
            const dBar = Math.abs(x2 - x1);
            if (dBar === 0) {
                pendingDrawPoint = null;
                removePreviewLine();
                return;
            }
            const dPrice = y2 - y1;
            const slope = dPrice / dBar;
            const t1 = barIndexToTime(x1);
            const t2 = barIndexToTime(x2);
            const xEnd = Math.min(Math.max(x1, x2) + N_FUTURE_BARS, (lastCandles?.length ?? 1) - 1);
            const tEnd = barIndexToTime(xEnd);
            const line = { type: 'trend', x1, y1, x2, y2, t1, t2, dBar, dPrice, slope };
            userDrawnLines.push(line);
            const series = chart.addLineSeries({
                color: 'rgba(41, 98, 255, 0.9)',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            series.setData([
                { time: t1, value: y1 },
                { time: tEnd, value: y1 + slope * (xEnd - x1) },
            ]);
            patternLineSeries.push(series);
            removePreviewLine();
            pendingDrawPoint = null;
            if (chart) chart.applyOptions({ handleScroll: true });
            onDrawLineFinalized(line);
            return;
        }
        if (drawingMode === 'horizontal') {
            // Spec: single click defines y, line extends along x into future
            const range = chart.timeScale().getVisibleRange();
            if (!range) return;
            const barIdx = tp.barIndex ?? timeToBarIndex(tp.time);
            const xEnd = Math.min(barIdx + N_FUTURE_BARS, (lastCandles?.length ?? 1) - 1);
            const t1 = barIndexToTime(barIdx);
            const t2 = barIndexToTime(xEnd);
            const line = { type: 'horizontal', x1: barIdx, y1: tp.value, x2: xEnd, t1, t2, dBar: xEnd - barIdx, dPrice: 0, slope: 0 };
            userDrawnLines.push(line);
            const series = chart.addLineSeries({
                color: 'rgba(255, 193, 7, 0.9)',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            series.setData([{ time: t1, value: tp.value }, { time: t2, value: tp.value }]);
            patternLineSeries.push(series);
            removePreviewLine();
            pendingDrawPoint = null;
            onDrawLineFinalized(line);
            return;
        }
    }

    // RECOGNIZE mode: click on recognition line to select (legacy)
    if (!drawingMode && isRecognitionOverlayMode()) {
        const hit = hitTestSRLine(tp.time, tp.value);
        if (hit !== null && patternStatsData?.current?.length) {
            if (selectedLineIndex === hit) {
                selectedLineIndex = null;
                similarLineIndices = [];
            } else {
                selectedLineIndex = hit;
                const cur = patternStatsData.current[hit];
                similarLineIndices = cur?.similar_line_indices ?? [];
            }
            updatePatternStatsSelectedUI();
            applySRSelectionStyles();
        } else if (selectedLineIndex != null) {
            selectedLineIndex = null;
            similarLineIndices = [];
            updatePatternStatsSelectedUI();
            applySRSelectionStyles();
        }
    }
}

function removePreviewLine() {
    if (previewLineSeries && chart) {
        try { chart.removeSeries(previewLineSeries); } catch (_) {}
        previewLineSeries = null;
    }
}

let _mouseMoveRAF = null;
function onChartMouseMove(event) {
    if (_mouseMoveRAF) return;
    _mouseMoveRAF = requestAnimationFrame(() => {
        _mouseMoveRAF = null;
        if (!chart || !candleSeries) return;
        const tp = getTimePriceFromEvent(event);
        if (!tp) return;

        // DRAW mode: preview line from P1 to cursor (chart pan disabled during draw per spec)
        if (!drawingMode) return;

        if (drawingMode === 'trend' && pendingDrawPoint) {
            if (!previewLineSeries) {
                previewLineSeries = chart.addLineSeries({
                    color: 'rgba(41, 98, 255, 0.6)',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Solid,
                    crosshairMarkerVisible: false,
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
            }
            previewLineSeries.setData([
                { time: pendingDrawPoint.time, value: pendingDrawPoint.value },
                { time: tp.time, value: tp.value },
            ]);
            return;
        }

        // No mousemove preview for horizontal (single-click only)
    });
}

function onDrawLineFinalized(line) {
    const selEl = document.getElementById('pattern-stats-selected');
    if (selEl) {
        selEl.innerHTML = `Δbar=${line.dBar} slope=${line.slope.toFixed(6)}`;
        selEl.classList.remove('hidden');
    }
    lastDrawnLineForAssist = line;
    assistSimilarData = null;
    fetchAssistSimilar(line).then((data) => {
        assistSimilarData = data;
        updateAssistPanelNotification();
    });
}

let lastDrawnLineForAssist = null;
let assistSimilarData = null;

function updateAssistPanelNotification() {
    const selEl = document.getElementById('pattern-stats-selected');
    if (!selEl) return;
    if (!lastDrawnLineForAssist) {
        selEl.innerHTML = 'Draw a line in Draw mode to compare with history.';
        selEl.classList.remove('hidden');
        return;
    }
    if (assistSimilarData && assistSimilarData.count > 0) {
        selEl.innerHTML = `Similar: ${assistSimilarData.count} found. <button type="button" id="btn-show-similar" class="assist-btn">Show Similar Patterns</button>`;
        selEl.querySelector('#btn-show-similar')?.addEventListener('click', showSimilarPatterns);
    } else if (assistSimilarData) {
        selEl.innerHTML = 'No similar historical structures found.';
    } else {
        selEl.innerHTML = 'Checking for similar historical structures...';
    }
    selEl.classList.remove('hidden');
}

function showSimilarPatterns() {
    if (!assistSimilarData?.lines?.length) return;
    similarLinesAssistLayer = assistSimilarData.lines;
    visibleSimilarIndices = new Set(similarLinesAssistLayer.map((_, i) => i));
    renderSimilarLinesList();
    scheduleDrawPatterns();
}

function toggleSimilarLineVisibility(index) {
    if (visibleSimilarIndices.has(index)) {
        visibleSimilarIndices.delete(index);
    } else {
        visibleSimilarIndices.add(index);
    }
    scheduleDrawPatterns();
}

function renderSimilarLinesList() {
    const container = document.getElementById('similar-lines-list');
    if (!container) return;
    if (!similarLinesAssistLayer.length) {
        container.innerHTML = '';
        container.classList.add('hidden');
        return;
    }
    container.classList.remove('hidden');
    container.innerHTML = '<div class="similar-lines-title">相似历史线 (点击显示/隐藏)</div>';
    similarLinesAssistLayer.forEach((seg, i) => {
        const sim = seg.similarity != null ? seg.similarity.toFixed(3) : '—';
        const ok = seg.success ? '成功' : '失败';
        const visible = visibleSimilarIndices.has(i);
        const row = document.createElement('div');
        row.className = 'similar-line-row' + (visible ? ' visible' : '');
        row.dataset.index = i;
        row.innerHTML = `<span class="similar-line-num">${i + 1}.</span> 相似 ${sim} · ${ok}`;
        row.title = visible ? '点击隐藏' : '点击显示';
        row.addEventListener('click', () => {
            toggleSimilarLineVisibility(i);
            renderSimilarLinesList();
        });
        container.appendChild(row);
    });
}

let similarLinesAssistLayer = [];
/** Indices into similarLinesAssistLayer that are currently shown on chart; click list item to toggle */
let visibleSimilarIndices = new Set();

async function fetchAssistSimilar(line) {
    try {
        const params = new URLSearchParams({
            symbol: currentSymbol,
            interval: currentInterval,
            days: getDaysForInterval(currentInterval),
            x1: line.x1,
            y1: line.y1,
            x2: line.x2,
            y2: line.y2,
        });
        const resp = await fetch(`${API_BASE}/api/pattern-stats/line-similar?${params}`);
        if (!resp.ok) return { count: 0, lines: [] };
        const data = await resp.json();
        return { count: data.count ?? 0, lines: data.lines ?? [] };
    } catch {
        return { count: 0, lines: [] };
    }
}

function onChartMouseLeave() {
    if (pendingDrawPoint) {
        removePreviewLine();
        pendingDrawPoint = null;
        if (chart) chart.applyOptions({ handleScroll: true });
    }
}

function showDrawnLineStats(slope, periodBars) {
    const selEl = document.getElementById('pattern-stats-selected');
    if (!selEl) return;
    selEl.classList.remove('hidden');
    if (periodBars != null) {
        selEl.innerHTML = `手绘线：斜率 <span class="line-stats">${slope.toFixed(6)}</span>，周期 <span class="line-stats">${periodBars} bar</span>`;
    } else {
        selEl.innerHTML = `手绘线：斜率 <span class="line-stats">${slope.toFixed(6)}</span>`;
    }
    selEl.classList.add('highlight');
}

// ── Price Scale Mode ──
function setScaleMode(mode) {
    if (!chart) return;
    currentScale = mode;

    const priceScale = chart.priceScale('right');
    if (mode === 'log') {
        priceScale.applyOptions({ mode: LightweightCharts.PriceScaleMode.Logarithmic });
    } else {
        priceScale.applyOptions({ mode: LightweightCharts.PriceScaleMode.Normal });
    }

    const linearBtn = document.querySelector('#chart-scale-toggle .scale-btn[data-scale="linear"]');
    const logBtn = document.querySelector('#chart-scale-toggle .scale-btn[data-scale="log"]');
    if (linearBtn && logBtn) {
        linearBtn.classList.toggle('active', mode === 'linear');
        logBtn.classList.toggle('active', mode === 'log');
    }
}

// ── Price Formatting (OKX-style precision) ──
function inferPrecision(price) {
    if (price >= 1000) return 2;
    if (price >= 1) return 4;
    if (price >= 0.01) return 6;
    return 8;
}

function formatPrice(value) {
    const p = pricePrecision !== null && pricePrecision !== undefined ? pricePrecision : inferPrecision(value);
    return value.toFixed(p);
}

// ── TradingView-style chart header (on chart) and toolbar price ──
function updateChartHeader() {
    const symEl = document.getElementById('chart-symbol');
    const intEl = document.getElementById('chart-interval');
    const priceEl = document.getElementById('chart-price');
    const changeEl = document.getElementById('chart-change');
    if (!symEl || !intEl || !priceEl || !changeEl) return;

    symEl.textContent = formatTicker(currentSymbol);
    intEl.textContent = currentInterval.toUpperCase();

    if (!lastCandle) {
        priceEl.textContent = '—';
        changeEl.textContent = '—';
        changeEl.className = 'chart-change';
        return;
    }

    const c = lastCandle.close;
    const o = lastCandle.open;
    const pct = o !== 0 ? ((c - o) / o) * 100 : 0;
    const up = c >= o;

    priceEl.textContent = formatPrice(c);
    priceEl.style.color = up ? '#26a69a' : '#ef5350';
    changeEl.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
    changeEl.className = 'chart-change ' + (up ? 'change-up' : 'change-down');
}

function updateToolbarPrice() {
    const priceEl = document.getElementById('toolbar-price');
    const changeEl = document.getElementById('toolbar-change');
    if (!priceEl || !changeEl) return;

    if (!lastCandle) {
        priceEl.textContent = '—';
        changeEl.textContent = '';
        changeEl.className = 'toolbar-change';
        return;
    }

    const c = lastCandle.close;
    const o = lastCandle.open;
    const pct = o !== 0 ? ((c - o) / o) * 100 : 0;
    const up = c >= o;

    priceEl.textContent = formatPrice(c);
    changeEl.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
    changeEl.className = 'toolbar-change ' + (up ? 'change-up' : 'change-down');
}

// ── OHLC Legend ──
function updateOHLCLegend(param) {
    const legend = document.getElementById('ohlc-legend');

    if (!param || !param.time || !param.seriesData || !param.seriesData.has(candleSeries)) {
        return;
    }

    const d = param.seriesData.get(candleSeries);
    if (!d || d.open === undefined) return;

    const color = d.close >= d.open ? '#26a69a' : '#ef5350';

    legend.innerHTML =
        `<span><span class="ohlc-label">O</span> <span style="color:${color}">${formatPrice(d.open)}</span></span>` +
        `<span><span class="ohlc-label">H</span> <span style="color:${color}">${formatPrice(d.high)}</span></span>` +
        `<span><span class="ohlc-label">L</span> <span style="color:${color}">${formatPrice(d.low)}</span></span>` +
        `<span><span class="ohlc-label">C</span> <span style="color:${color}">${formatPrice(d.close)}</span></span>`;
}

// ── Helpers ──
function getDaysForInterval(interval) {
    // Data priority: fast (less data, quick load), balanced, deep (full history)
    const priority = document.getElementById('data-priority-select')?.value || 'balanced';
    const profiles = {
        fast:     { '1m': 1, '5m': 3,  '15m': 7,  '1h': 30,  '4h': 90,   '1d': 365 },
        balanced: { '1m': 2, '5m': 7,  '15m': 21, '1h': 90,  '4h': 365,  '1d': 365 * 3 },
        deep:     { '1m': 2, '5m': 7,  '15m': 21, '1h': 180, '4h': 365 * 2, '1d': 365 * 5 },
    };
    return (profiles[priority] || profiles.balanced)[interval] || 180;
}

function buildParams() {
    const params = new URLSearchParams({
        symbol: currentSymbol,
        interval: currentInterval,
        days: getDaysForInterval(currentInterval),
    });

    const replayToggle = document.getElementById('replay-toggle');
    if (replayToggle?.checked) {
        const endTime = getReplayEndTime();
        if (endTime) {
            params.set('end_time', endTime);
        }
    }

    return params;
}

/** Params for /api/patterns including mode: recognizing (patterns only) | assist (trendlines only) | full */
function buildPatternParams() {
    const params = buildParams();
    const mode = toolMode === 'recognize' ? 'recognizing' : (toolMode === 'assist' ? 'assist' : 'full');
    params.set('mode', mode);
    return params;
}

function getReplayEndTime() {
    const dateVal = document.getElementById('replay-date').value;
    if (!dateVal) return null;
    const hour = parseInt(document.getElementById('replay-hour').value) || 0;
    const activeBtn = document.querySelector('#tz-group .tz-btn.active');
    const tzOffset = activeBtn ? parseInt(activeBtn.dataset.offset) : 0;

    // Build a UTC date from the user's local-in-timezone selection
    const utcDate = new Date(`${dateVal}T${String(hour).padStart(2, '0')}:00:00Z`);
    utcDate.setUTCHours(utcDate.getUTCHours() - tzOffset);

    return utcDate.toISOString().slice(0, 16);
}

// ── Data Loading ──
const LOAD_DATA_TIMEOUT_MS = 30000; // 30s timeout (was 90s — too long)
let _loadDataInFlight = false;
let _loadDataAbortController = null;
async function loadData(isLiveUpdate = false) {
    if (_loadDataInFlight && isLiveUpdate) return;
    if (_loadDataAbortController) _loadDataAbortController.abort();
    _loadDataAbortController = new AbortController();
    _loadDataInFlight = true;
    if (!isLiveUpdate) showLoading(true);

    const params = buildParams();
    const signal = _loadDataAbortController.signal;
    const timeoutId = setTimeout(() => {
        if (_loadDataAbortController) _loadDataAbortController.abort();
    }, LOAD_DATA_TIMEOUT_MS);

    try {
        const resp = await fetch(`${API_BASE}/api/ohlcv?${params}`, { signal });
        if (!resp.ok) {
            let errMsg = 'Unknown error';
            try {
                const err = await resp.json();
                errMsg = err.detail || err.message || `HTTP ${resp.status}: ${resp.statusText}`;
            } catch {
                errMsg = `HTTP ${resp.status}: ${resp.statusText}`;
            }
            console.error('Chart fetch error:', errMsg, 'URL:', `/api/ohlcv?${params}`);
            if (!isLiveUpdate) {
                console.warn(`Data error: ${errMsg} — will retry in 3s`);
                setTimeout(() => loadData(), 3000);
            }
            return;
        }
        if (signal.aborted) return;

        const data = await resp.json();
        window._loadRetryCount = 0; // reset retry counter on success

        if (data.pricePrecision !== undefined) {
            pricePrecision = data.pricePrecision;
        }
        const candlesIn = data.candles || [];
        const precision = pricePrecision !== null && pricePrecision !== undefined
            ? pricePrecision
            : (candlesIn.length > 0 ? inferPrecision(candlesIn[candlesIn.length - 1].close) : 2);
        const minMove = precision === 0 ? 1 : Math.pow(10, -precision);
        candleSeries.applyOptions({
            priceFormat: {
                type: 'price',
                precision: precision,
                minMove: minMove,
            },
        });
        
        lastCandles = data.candles || [];
        candleSeries.setData(lastCandles);
        volumeSeries.setData(data.volume || []);
        if (backtestMarkers.length > 0) candleSeries.setMarkers(backtestMarkers);

        // Draw V6 MA overlays
        drawMAOverlays(data.overlays || {});

        if (lastCandles.length > 0) {
            lastCandle = lastCandles[lastCandles.length - 1];
            const last = lastCandle;
            const legend = document.getElementById('ohlc-legend');
            const color = last.close >= last.open ? '#26a69a' : '#ef5350';
            legend.innerHTML =
                `<span><span class="ohlc-label">O</span> <span style="color:${color}">${formatPrice(last.open)}</span></span>` +
                `<span><span class="ohlc-label">H</span> <span style="color:${color}">${formatPrice(last.high)}</span></span>` +
                `<span><span class="ohlc-label">L</span> <span style="color:${color}">${formatPrice(last.low)}</span></span>` +
                `<span><span class="ohlc-label">C</span> <span style="color:${color}">${formatPrice(last.close)}</span></span>`;
            updateChartHeader();
            updateToolbarPrice();
        }

        // Pattern overlay is opt-in via Pattern tab only.
        if (!isRecognitionOverlayMode()) {
            rawPatternData = null;
            patternStatsData = null;
            selectedLineIndex = null;
            similarLineIndices = [];
        }
        if (!isLiveUpdate) {
            chart.timeScale().fitContent();
        }
        // Patterns load in background so timeframe switch shows candles immediately
        if (isRecognitionOverlayMode() && !isLiveUpdate) {
            loadPatterns(buildPatternParams().toString()).then(() => {
                drawAllPatterns();
            });
        }
        drawAllPatterns();
    } catch (e) {
        if (e.name === 'AbortError') {
            if (!isLiveUpdate) {
                showLoading(false);
                console.warn('Load timeout — will retry in 3s');
                setTimeout(() => loadData(), 3000);
            }
            return;
        }
        if (!isLiveUpdate) {
            console.error('Data load error:', e);
            showLoading(false);
            // Auto-retry on network error (server may still be starting)
            if (!window._loadRetryCount) window._loadRetryCount = 0;
            window._loadRetryCount++;
            if (window._loadRetryCount <= 5) {
                console.warn(`Network error, retry ${window._loadRetryCount}/5 in 2s...`);
                setTimeout(() => loadData(), 2000);
            } else {
                console.error('Max retries reached. Server may be down.');
                window._loadRetryCount = 0;
            }
        }
    } finally {
        clearTimeout(timeoutId);
        _loadDataInFlight = false;
        if (!isLiveUpdate) {
            showLoading(false);
        }
    }
}

// ── Live Updates ──
function startLiveUpdates() {
    stopLiveUpdates();
    
    // Determine update interval based on current timeframe
    const intervalMs = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    };
    
    const updateInterval = intervalMs[currentInterval] || 60000;
    liveUpdateInterval = setInterval(() => loadData(true), updateInterval);
}

function stopLiveUpdates() {
    if (liveUpdateInterval) {
        clearInterval(liveUpdateInterval);
        liveUpdateInterval = null;
    }
}

// ── Pattern Overlay ──
function clearPatterns() {
    for (const series of patternLineSeries) {
        if (chart) chart.removeSeries(series);
    }
    patternLineSeries = [];
    srSeriesRefs = [];
    srSeriesTypes = [];
    srLineSegments = [];
    // Clear trend indicator
    const box = document.getElementById('trend-indicator');
    if (box) box.classList.add('hidden');
}

async function loadPatterns(existingParams) {
    rawPatternData = null;

    try {
        if (patternResponseCache.has(existingParams)) {
            rawPatternData = patternResponseCache.get(existingParams);
            if (rawPatternData) drawAllPatterns();
            return;
        }
        const resp = await fetch(`${API_BASE}/api/patterns?${existingParams}`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
            console.error('Pattern API error:', err);
            // Don't show alert for pattern errors, just log
            return;
        }

        rawPatternData = await resp.json();
        // Evict oldest cache entries to prevent memory leak
        if (patternResponseCache.size >= PATTERN_CACHE_MAX_SIZE) {
            const firstKey = patternResponseCache.keys().next().value;
            patternResponseCache.delete(firstKey);
        }
        patternResponseCache.set(existingParams, rawPatternData);
        console.log('Pattern data received:', {
            hasSupport: !!rawPatternData?.supportLines,
            hasResistance: !!rawPatternData?.resistanceLines,
            supportCount: rawPatternData?.supportLines?.length || 0,
            resistanceCount: rawPatternData?.resistanceLines?.length || 0
        });
        if (rawPatternData) {
            drawAllPatterns();
            if (toolMode === 'recognize') updateRecognizePanelContent();
        }
    } catch (e) {
        console.error('Pattern load failed:', e);
    }
}

function updateRecognizePanelContent() {
    const el = document.getElementById('pattern-stats-content');
    if (!el || toolMode !== 'recognize') return;
    const patterns = rawPatternData?.patterns || [];
    if (patterns.length === 0) {
        el.innerHTML = '无识别形态（三角形、头肩、通道等）';
        return;
    }
    const types = [...new Set(patterns.map(function (p) { return p.pattern_type || p.type || '\u5f62\u6001'; }))];
    el.innerHTML = '\u8bc6\u5224\u5230 <strong>' + patterns.length + '</strong> \u4e2a\u5f62\u6001\uff1a' + types.join('\u3001');
}

async function fetchPatternStats() {
    const el = document.getElementById('pattern-stats-content');
    const selEl = document.getElementById('pattern-stats-selected');
    if (!el) return;
    el.textContent = '…';
    selectedLineIndex = null;
    similarLineIndices = [];
    patternStatsData = null;
    updatePatternStatsSelectedUI();
    try {
        const params = new URLSearchParams({
            symbol: currentSymbol,
            interval: currentInterval,
            days: getDaysForInterval(currentInterval),
        });
        const resp = await fetch(`${API_BASE}/api/pattern-stats/current-vs-history?${params}`);
        if (!resp.ok) {
            el.textContent = '—';
            return;
        }
        const data = await resp.json();
        patternStatsData = data;
        const current = data.current || [];
        const nLines = current.length;
        const similarCount = data.overall_similar_count ?? 0;
        const rate = data.overall_success_rate_pct;

        if (nLines === 0) {
            el.textContent = '当前无趋势线';
            if (selEl) selEl.classList.add('hidden');
            return;
        }
        const supportCount = current.filter(c => (c.feature && c.feature.line_type === 'support')).length;
        const resistCount = nLines - supportCount;
        const lineDesc = [supportCount && `${supportCount} 支撑`, resistCount && `${resistCount} 阻力`].filter(Boolean).join('，') || `${nLines} 条`;
        let html = `当前：${lineDesc}<br>`;
        if (similarCount > 0 && rate != null) {
            html += `历史同类：${similarCount} 样本，成功率 <strong>${rate}%</strong>`;
        } else {
            html += '历史同类：无相似样本';
        }
        el.innerHTML = html;
        if (selEl) selEl.classList.remove('hidden');
    } catch (e) {
        console.error('Pattern stats fetch failed:', e);
        el.textContent = '—';
    }
}

function updatePatternStatsSelectedUI() {
    const selEl = document.getElementById('pattern-stats-selected');
    if (!selEl) return;
    if (!patternStatsData?.current?.length) {
        selEl.classList.add('hidden');
        return;
    }
    selEl.classList.remove('hidden');
    if (selectedLineIndex == null) {
        selEl.innerHTML = '点击支撑/阻力线查看该线历史表现';
        selEl.classList.remove('highlight');
        return;
    }
    const cur = patternStatsData.current[selectedLineIndex];
    const similar = cur?.similar_count ?? 0;
    const rate = cur?.success_rate_pct;
    const similarLines = cur?.similar_line_indices?.length ?? 0;
    let html = `该线：<span class="line-stats">${similar} 样本，成功率 ${rate != null ? rate + '%' : '—'}</span>`;
    if (similarLines > 0) {
        html += `<br><span class="line-stats">共 ${similarLines} 条相似线已高亮</span>`;
    }
    selEl.innerHTML = html;
    selEl.classList.add('highlight');
}

let _drawPatternsRAF = null;
function scheduleDrawPatterns() {
    if (_drawPatternsRAF) return;
    _drawPatternsRAF = requestAnimationFrame(() => {
        _drawPatternsRAF = null;
        drawAllPatterns();
    });
}

function drawAllPatterns() {
    if (!chart) return;
    const visibleRange = chart.timeScale().getVisibleLogicalRange();

    clearPatterns();
    if (isRecognitionOverlayMode() && rawPatternData) {
        // Recognizing: only structured patterns (triangles, etc.); no S/R trendlines
        if (toolMode === 'recognize' && rawPatternData.patterns?.length) {
            drawStructuredPatterns(rawPatternData.patterns);
        }
        // Assist: only extended trendlines; no pattern shapes, no consolidation (clean separation from Recognizing)
        if (toolMode === 'assist') {
            if (srVisible && (rawPatternData.supportLines?.length || rawPatternData.resistanceLines?.length)) {
                drawTrendlines(rawPatternData.supportLines || [], rawPatternData.resistanceLines || []);
            }
        }
        // Full (e.g. from /api/chart): draw both trendlines and consolidation when present
        if (toolMode !== 'recognize' && toolMode !== 'assist') {
            if (srVisible && (rawPatternData.supportLines?.length || rawPatternData.resistanceLines?.length)) {
                drawTrendlines(rawPatternData.supportLines || [], rawPatternData.resistanceLines || []);
            }
            drawConsolidationZones(rawPatternData.consolidationZones || []);
        }
        updateTrendIndicator(rawPatternData.trendLabel, rawPatternData.trendSlope);
    }
    drawUserDrawnLines();
    drawAssistSimilarLayer();

    if (visibleRange) {
        chart.timeScale().setVisibleLogicalRange(visibleRange);
    }
}

/** Draw only recognized patterns (e.g. triangle = 2 lines connecting). No S/R lines. */
function drawStructuredPatterns(patterns) {
    if (!chart || !patterns.length) return;
    for (const p of patterns) {
        const color = (p.breakout_bias === 'bullish' ? '#4caf50' : (p.breakout_bias === 'bearish' ? '#f44336' : '#2196f3'));
        // Pattern = connected lines (e.g. triangle: upper + lower boundary)
        if (p.high_points?.length >= 2) {
            const series = chart.addLineSeries({
                color: color,
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            series.setData(p.high_points.map(pt => ({ time: pt.time, value: pt.value })));
            patternLineSeries.push(series);
        }
        if (p.low_points?.length >= 2) {
            const series = chart.addLineSeries({
                color: color,
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            series.setData(p.low_points.map(pt => ({ time: pt.time, value: pt.value })));
            patternLineSeries.push(series);
        }
    }
}

function drawAssistSimilarLayer() {
    if (!chart || toolMode !== 'assist' || !similarLinesAssistLayer?.length) return;
    similarLinesAssistLayer.forEach((seg, i) => {
        if (!visibleSimilarIndices.has(i)) return;
        const t1 = seg.t1 ?? barIndexToTime(seg.x1);
        const t2 = seg.t2 ?? barIndexToTime(seg.x2);
        const v1 = seg.v1 ?? seg.y1;
        const v2 = seg.v2 ?? seg.y2;
        const series = chart.addLineSeries({
            color: 'rgba(255, 152, 0, 0.6)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });
        series.setData([{ time: t1, value: v1 }, { time: t2, value: v2 }]);
        patternLineSeries.push(series);
    });
}

// ── Magnet Helpers ──
function findNearestCandle(time) {
    if (!lastCandles || lastCandles.length === 0) return null;
    // Binary search for O(log n) instead of O(n)
    const idx = timeToBarIndex(time);
    return lastCandles[idx] || lastCandles[lastCandles.length - 1];
}

function snapPriceToCandle(time, price, mode) {
    const candle = findNearestCandle(time);
    if (!candle) return price;

    const candidates = [candle.open, candle.high, candle.low, candle.close];
    let best = candidates[0];
    let bestDiff = Math.abs(price - best);
    for (let i = 1; i < candidates.length; i++) {
        const diff = Math.abs(price - candidates[i]);
        if (diff < bestDiff) {
            bestDiff = diff;
            best = candidates[i];
        }
    }

    const diffPct = Math.abs(best - price) / (Math.abs(price) || 1);

    // Weak magnet: only snap when已经很接近 K 线价位
    if (mode === 'weak') {
        const threshold = 0.002; // 0.2%
        if (diffPct > threshold) return price;
    }

    // Strong magnet: 始终吸附到最近的 OHLC
    return best;
}

function applyMagnetToLine(line) {
    if (!lastCandles.length) return line;
    const y1 = snapPriceToCandle(line.x1, line.y1, magnetMode);
    const y2 = snapPriceToCandle(line.x2, line.y2, magnetMode);
    return { ...line, y1, y2 };
}

/**
 * Filter lines to keep only the `maxCount` closest to `targetPrice`,
 * measured by the line's y-intercept at the last candle's time.
 */
function filterLinesByProximity(lines, targetPrice, maxCount) {
    if (maxCount <= 0 || lines.length <= maxCount) return lines;
    if (!lastCandle) return lines.slice(0, maxCount);

    const lastTime = lastCandle.time;
    const scored = lines.map(line => {
        const dx = line.x2 - line.x1;
        const slopePerSec = dx !== 0 ? (line.y2 - line.y1) / dx : 0;
        const yAtCurrent = line.y1 + slopePerSec * (lastTime - line.x1);
        return { line, distance: Math.abs(yAtCurrent - targetPrice) };
    });

    scored.sort((a, b) => a.distance - b.distance);
    return scored.slice(0, maxCount).map(s => s.line);
}

function drawTrendlines(supportLines, resistanceLines) {
    if (!supportLines) supportLines = [];
    if (!resistanceLines) resistanceLines = [];
    if (!chart) return;

    srLineSegments = [];
    srSeriesRefs = [];
    srSeriesTypes = [];
    let indexInCurrent = 0;

    // Filter by max lines if set
    let filteredSupport = supportLines;
    let filteredResistance = resistanceLines;

    if (maxSRLines > 0 && lastCandle) {
        filteredSupport = filterLinesByProximity(supportLines, lastCandle.low, maxSRLines);
        filteredResistance = filterLinesByProximity(resistanceLines, lastCandle.high, maxSRLines);
    }

    const isSelected = (i) => selectedLineIndex === i;
    const isSimilar = (i) => similarLineIndices.includes(i);

    // 支撑线：连接局部低点
    for (const line of filteredSupport) {
        const t1 = Number(line.x1);
        const t2 = Number(line.x2);
        const v1 = Number(line.y1);
        const v2 = Number(line.y2);
        if (Number.isFinite(t1) && Number.isFinite(t2) && Number.isFinite(v1) && Number.isFinite(v2)) {
            srLineSegments.push({ t1, v1, t2, v2, indexInCurrent, lineType: 'support' });
            let color = 'rgba(38, 166, 154, 0.85)';
            if (isSelected(indexInCurrent)) color = 'rgba(41, 98, 255, 1)';
            else if (isSimilar(indexInCurrent)) color = 'rgba(255, 193, 7, 0.95)';
            const series = chart.addLineSeries({
                color,
                lineWidth: isSelected(indexInCurrent) ? 3 : 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            series.setData([
                { time: t1, value: v1 },
                { time: t2, value: v2 },
            ]);
            patternLineSeries.push(series);
            srSeriesRefs[indexInCurrent] = series;
            srSeriesTypes[indexInCurrent] = 'support';
            indexInCurrent++;
        }
    }

    // 阻力线：连接局部高点
    for (const line of filteredResistance) {
        const t1 = Number(line.x1);
        const t2 = Number(line.x2);
        const v1 = Number(line.y1);
        const v2 = Number(line.y2);
        if (Number.isFinite(t1) && Number.isFinite(t2) && Number.isFinite(v1) && Number.isFinite(v2)) {
            srLineSegments.push({ t1, v1, t2, v2, indexInCurrent, lineType: 'resistance' });
            let color = 'rgba(239, 83, 80, 0.85)';
            if (isSelected(indexInCurrent)) color = 'rgba(41, 98, 255, 1)';
            else if (isSimilar(indexInCurrent)) color = 'rgba(255, 193, 7, 0.95)';
            const series = chart.addLineSeries({
                color,
                lineWidth: isSelected(indexInCurrent) ? 3 : 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            series.setData([
                { time: t1, value: v1 },
                { time: t2, value: v2 },
            ]);
            patternLineSeries.push(series);
            srSeriesRefs[indexInCurrent] = series;
            srSeriesTypes[indexInCurrent] = 'resistance';
            indexInCurrent++;
        }
    }
}

function applySRSelectionStyles() {
    if (!srSeriesRefs.length) return;
    for (let i = 0; i < srSeriesRefs.length; i++) {
        const s = srSeriesRefs[i];
        if (!s) continue;
        const type = srSeriesTypes[i] || 'support';
        let color = type === 'support' ? 'rgba(38, 166, 154, 0.85)' : 'rgba(239, 83, 80, 0.85)';
        let lineWidth = 2;
        if (selectedLineIndex === i) {
            color = 'rgba(41, 98, 255, 1)';
            lineWidth = 3;
        } else if (similarLineIndices.includes(i)) {
            color = 'rgba(255, 193, 7, 0.95)';
        }
        s.applyOptions({ color, lineWidth });
    }
}

function drawConsolidationZones(zones) {
    for (const zone of zones) {
        // Top boundary
        const topSeries = chart.addLineSeries({
            color: 'rgba(255, 152, 0, 0.35)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });
        topSeries.setData([
            { time: zone.startTime, value: zone.priceHigh },
            { time: zone.endTime, value: zone.priceHigh },
        ]);
        patternLineSeries.push(topSeries);

        // Bottom boundary
        const botSeries = chart.addLineSeries({
            color: 'rgba(255, 152, 0, 0.35)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });
        botSeries.setData([
            { time: zone.startTime, value: zone.priceLow },
            { time: zone.endTime, value: zone.priceLow },
        ]);
        patternLineSeries.push(botSeries);
    }
}

function drawUserDrawnLines() {
    if (!chart || !candleSeries) return;
    for (const line of userDrawnLines) {
        const opts = {
            color: line.type === 'horizontal' ? 'rgba(255, 193, 7, 0.9)' : 'rgba(41, 98, 255, 0.9)',
            lineWidth: 2,
            lineStyle: line.type === 'horizontal' ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        };
        const series = chart.addLineSeries(opts);
        if (line.type === 'horizontal') {
            const t1 = line.t1 ?? barIndexToTime(line.x1);
            const t2 = line.t2 ?? barIndexToTime(line.x2);
            const v = line.y1 ?? line.v1;
            series.setData([{ time: t1, value: v }, { time: t2, value: v }]);
        } else {
            const t1 = line.t1 ?? barIndexToTime(line.x1);
            const y1 = line.y1 ?? line.v1;
            const y2 = line.y2 ?? line.v2;
            const xEnd = Math.min(Math.max(line.x1 ?? 0, line.x2 ?? 0) + N_FUTURE_BARS, (lastCandles?.length ?? 1) - 1);
            const tEnd = barIndexToTime(xEnd);
            const slope = line.slope ?? ((y2 - y1) / (Math.abs((line.x2 ?? 0) - (line.x1 ?? 0)) || 1));
            const yEnd = y1 + slope * (xEnd - (line.x1 ?? 0));
            series.setData([{ time: t1, value: y1 }, { time: tEnd, value: yEnd }]);
        }
        patternLineSeries.push(series);
    }
}

function updateTrendIndicator(label, slope) {
    const box = document.getElementById('trend-indicator');
    if (!box) return;

    box.classList.remove('hidden', 'trend-up', 'trend-down', 'trend-sideways');

    const slopeVal = (typeof slope === 'number' && !isNaN(slope)) ? slope : 0;
    const slopeStr = `${slopeVal > 0 ? '+' : ''}${slopeVal.toFixed(2)}%`;

    if (label === 'UPTREND') {
        box.classList.add('trend-up');
        box.textContent = `UPTREND (${slopeStr})`;
    } else if (label === 'DOWNTREND') {
        box.classList.add('trend-down');
        box.textContent = `DOWNTREND (${slopeStr})`;
    } else {
        box.classList.add('trend-sideways');
        box.textContent = `SIDEWAYS (${slopeStr})`;
    }
}

function showLoading(show) {
    const el = document.getElementById('loading');
    if (!el) return;
    if (show) {
        el.classList.remove('loading-hidden');
    } else {
        el.classList.add('loading-hidden');
    }
}

// ── Backtest ──
function isBacktestPanelOpen() {
    return document.getElementById('backtest-panel') && !document.getElementById('backtest-panel').classList.contains('hidden');
}

function closeBacktestPanel() {
    document.getElementById('backtest-panel')?.classList.add('hidden');
}

async function runBacktest() {
    const panel = document.getElementById('backtest-panel');
    const content = document.getElementById('backtest-content');
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
    } else {
        closeBacktestPanel();
        return;
    }

    content.innerHTML = '<div class="backtest-loading">Running backtest...</div>';
    const days = getDaysForInterval(currentInterval);
    const params = new URLSearchParams({ symbol: currentSymbol, interval: currentInterval, days });

    try {
        const resp = await fetch(`${API_BASE}/api/backtest?${params}`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Backtest failed');
        }
        const data = await resp.json();

        let html = `
            <div class="backtest-summary">
                <div class="backtest-row"><span>Total Trades</span><strong>${data.total_trades}</strong></div>
                <div class="backtest-row"><span>Wins / Losses</span><strong>${data.wins} / ${data.losses}</strong></div>
                <div class="backtest-row"><span>Win Rate</span><strong>${data.win_rate}%</strong></div>
                <div class="backtest-row"><span>Total PnL</span><strong class="${data.total_pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg'}">${data.total_pnl_pct}%</strong></div>
                <div class="backtest-row"><span>Avg Win</span><strong class="pnl-pos">${data.avg_win}%</strong></div>
                <div class="backtest-row"><span>Avg Loss</span><strong class="pnl-neg">${data.avg_loss}%</strong></div>
            </div>
            <p class="backtest-strategy">${data.strategy || ''}</p>
        `;
        if (data.trades && data.trades.length > 0) {
            html += '<div class="backtest-trades"><strong>Trades</strong><ul>';
            data.trades.slice(0, 15).forEach(t => {
                const cls = t.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg';
                html += `<li>${t.exit_reason}: <span class="${cls}">${t.pnl_pct.toFixed(2)}%</span></li>`;
            });
            if (data.trades.length > 15) html += `<li>... +${data.trades.length - 15} more</li>`;
            html += '</ul></div>';
        }
        content.innerHTML = html;

        // Build markers for chart: entry = arrowUp/belowBar, exit = arrowDown/aboveBar (or circle)
        backtestMarkers = [];
        if (data.trades && lastCandles.length > 0) {
            for (const t of data.trades) {
                backtestMarkers.push({
                    time: t.entry_time,
                    position: 'belowBar',
                    color: '#26a69a',
                    shape: 'arrowUp',
                    text: 'B',
                });
                backtestMarkers.push({
                    time: t.exit_time,
                    position: 'aboveBar',
                    color: t.exit_reason === 'TP' ? '#26a69a' : (t.exit_reason === 'SL' ? '#ef5350' : '#787b86'),
                    shape: t.exit_reason === 'TP' ? 'arrowDown' : 'circle',
                    text: t.exit_reason,
                });
            }
            candleSeries.setMarkers(backtestMarkers);
        }
    } catch (e) {
        content.innerHTML = `<div class="backtest-error">Error: ${e.message}</div>`;
        backtestMarkers = [];
        candleSeries.setMarkers([]);
    }
}

async function runOptimize() {
    const panel = document.getElementById('backtest-panel');
    const content = document.getElementById('backtest-content');
    panel.classList.remove('hidden');

    content.innerHTML = '<div class="backtest-loading">Optimizing strategy (gradient-free)...</div>';
    const days = getDaysForInterval(currentInterval);
    const params = new URLSearchParams({
        symbol: currentSymbol,
        interval: currentInterval,
        days,
        objective: 'total_pnl',
        maxiter: '80',
        method: 'L-BFGS-B',
    });

    try {
        const resp = await fetch(`${API_BASE}/api/backtest/optimize?${params}`, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Optimize failed');
        }
        const data = await resp.json();
        const res = data.best_result;
        const prm = data.best_params || {};
        const opt = data.optimization || {};

        let html = `
            <div class="backtest-summary">
                <div class="backtest-row"><span>Total Trades</span><strong>${res.total_trades}</strong></div>
                <div class="backtest-row"><span>Wins / Losses</span><strong>${res.wins} / ${res.losses}</strong></div>
                <div class="backtest-row"><span>Win Rate</span><strong>${res.win_rate}%</strong></div>
                <div class="backtest-row"><span>Total PnL</span><strong class="${res.total_pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg'}">${res.total_pnl_pct}%</strong></div>
                <div class="backtest-row"><span>Avg Win</span><strong class="pnl-pos">${res.avg_win}%</strong></div>
                <div class="backtest-row"><span>Avg Loss</span><strong class="pnl-neg">${res.avg_loss}%</strong></div>
            </div>
            <p class="backtest-strategy">${res.strategy || ''}</p>
            <div class="backtest-params-block">
                <strong>Optimized params</strong>
                <pre class="backtest-params-json">${JSON.stringify(prm, null, 2)}</pre>
                <span class="backtest-optimization-meta">${opt.success ? 'Converged' : 'Stopped'}: ${opt.message || ''}</span>
            </div>
        `;
        if (res.trades && res.trades.length > 0) {
            html += '<div class="backtest-trades"><strong>Trades</strong><ul>';
            res.trades.slice(0, 15).forEach(t => {
                const cls = t.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg';
                html += `<li>${t.exit_reason}: <span class="${cls}">${t.pnl_pct.toFixed(2)}%</span></li>`;
            });
            if (res.trades.length > 15) html += `<li>... +${res.trades.length - 15} more</li>`;
            html += '</ul></div>';
        }
        content.innerHTML = html;

        backtestMarkers = [];
        if (res.trades && lastCandles.length > 0) {
            for (const t of res.trades) {
                backtestMarkers.push({
                    time: t.entry_time,
                    position: 'belowBar',
                    color: '#26a69a',
                    shape: 'arrowUp',
                    text: 'B',
                });
                backtestMarkers.push({
                    time: t.exit_time,
                    position: 'aboveBar',
                    color: t.exit_reason === 'TP' ? '#26a69a' : (t.exit_reason === 'SL' ? '#ef5350' : '#787b86'),
                    shape: t.exit_reason === 'TP' ? 'arrowDown' : 'circle',
                    text: t.exit_reason,
                });
            }
            candleSeries.setMarkers(backtestMarkers);
        }
    } catch (e) {
        content.innerHTML = `<div class="backtest-error">Error: ${e.message}</div>`;
        backtestMarkers = [];
        candleSeries.setMarkers([]);
    }
}

document.getElementById('backtest-btn')?.addEventListener('click', runBacktest);
document.getElementById('optimize-btn')?.addEventListener('click', runOptimize);
document.getElementById('backtest-close')?.addEventListener('click', (e) => {
    e.preventDefault();
    closeBacktestPanel();
});

// ── Ticker Selector ──
const SYMBOLS_FETCH_TIMEOUT_MS = 15000; // 15s so page doesn't hang if OKX symbols API is slow
async function loadSymbols() {
    const ac = new AbortController();
    const timeoutId = setTimeout(() => ac.abort(), SYMBOLS_FETCH_TIMEOUT_MS);
    try {
        const resp = await fetch(`${API_BASE}/api/symbols`, { signal: ac.signal });
        clearTimeout(timeoutId);
        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
        }
        allSymbols = await resp.json();
        if (!Array.isArray(allSymbols) || allSymbols.length === 0) {
            console.warn('No symbols returned from API');
            allSymbols = ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT'];
        }
        renderTickerList(allSymbols);
    } catch (e) {
        clearTimeout(timeoutId);
        console.error('Failed to load symbols:', e);
        allSymbols = ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT'];
        renderTickerList(allSymbols);
    }
}

function renderTickerList(symbols) {
    const list = document.getElementById('ticker-list');
    if (!list) return;
    
    // Handle both string arrays and object arrays (for OKX with metadata)
    const symbolList = symbols.map(s => {
        if (typeof s === 'string') return s;
        if (typeof s === 'object' && s.symbol) return s.symbol;
        return String(s);
    });
    
    list.innerHTML = symbolList.map(s =>
        `<div class="ticker-item${s === currentSymbol ? ' active' : ''}" data-symbol="${s}">${formatTicker(s)}</div>`
    ).join('');
}

function formatTicker(s) {
    if (s.endsWith('USDT')) {
        return s.slice(0, -4) + ' / USDT';
    }
    return s;
}

function updateViewTabsUI() {
    const recognizingBtn = document.getElementById('tab-recognizing');
    const drawBtn = document.getElementById('tab-draw');
    const assistBtn = document.getElementById('tab-assist');
    const agentBtn = document.getElementById('tab-agent');
    const chatBtn = document.getElementById('tab-chat');
    if (recognizingBtn) recognizingBtn.classList.toggle('active', toolMode === 'recognize');
    if (drawBtn) drawBtn.classList.toggle('active', toolMode === 'draw');
    if (assistBtn) assistBtn.classList.toggle('active', toolMode === 'assist');
    if (agentBtn) agentBtn.classList.toggle('active', agentPanelOpen);
    if (chatBtn) chatBtn.classList.toggle('active', chatPanelOpen);
}

let _setToolModeVersion = 0;
async function setToolMode(nextMode) {
    const myVersion = ++_setToolModeVersion;
    // clicking the same tab toggles mode off
    toolMode = (toolMode === nextMode) ? null : nextMode;
    updateViewTabsUI();
    const contentEl = document.getElementById('pattern-stats-content');
    const selectedEl = document.getElementById('pattern-stats-selected');

    drawingMode = toolMode === 'draw' ? (drawingMode || 'trend') : null;
    updateDrawingModeUI();

    if (toolMode === 'draw') {
        if (contentEl) contentEl.textContent = 'Draw mode: click two points for trendline, one for horizontal';
        updateAssistPanelNotification();
        scheduleDrawPatterns();
        return;
    }

    if (!isRecognitionOverlayMode()) {
        rawPatternData = null;
        patternStatsData = null;
        selectedLineIndex = null;
        similarLineIndices = [];
        similarLinesAssistLayer = [];
        visibleSimilarIndices = new Set();
        const listEl = document.getElementById('similar-lines-list');
        if (listEl) { listEl.innerHTML = ''; listEl.classList.add('hidden'); }
        if (contentEl) contentEl.textContent = 'Choose Recognizing or Assist mode to load patterns';
        if (selectedEl) selectedEl.classList.add('hidden');
        scheduleDrawPatterns();
        return;
    }

    if (toolMode === 'assist') {
        if (contentEl) contentEl.textContent = 'Assist: 趋势线（延伸）. 与 Recognizing 分离，仅显示辅助线.';
        updateAssistPanelNotification();
        rawPatternData = null;
        scheduleDrawPatterns();
        await loadPatterns(buildPatternParams().toString());
        if (myVersion !== _setToolModeVersion) return; // stale
        scheduleDrawPatterns();
        if (contentEl && rawPatternData) {
            const s = rawPatternData.supportLines?.length || 0;
            const r = rawPatternData.resistanceLines?.length || 0;
            if (s || r) contentEl.textContent = `Assist: ${s} 支撑、${r} 阻力（延伸）`;
        }
        return;
    }
    if (contentEl) contentEl.textContent = 'Recognizing: loading patterns...';
    await loadPatterns(buildPatternParams().toString());
    if (myVersion !== _setToolModeVersion) return; // stale
    if (toolMode === 'recognize') updateRecognizePanelContent();
    await fetchPatternStats();
}

function selectTicker(symbol) {
    currentSymbol = symbol;
    lastCandle = null;
    patternResponseCache.clear(); // Clear cache on symbol change to prevent stale data
    document.getElementById('current-ticker').textContent = formatTicker(symbol);
    document.getElementById('ticker-dropdown').classList.add('hidden');
    updateChartHeader();
    updateToolbarPrice();
    document.getElementById('ticker-search').value = '';
    document.title = `${formatTicker(symbol)} - Crypto TA`;
    stopLiveUpdates();
    loadData();
    startLiveUpdates();
}

// Ticker dropdown toggle
document.getElementById('ticker-selector').addEventListener('click', (e) => {
    const dropdown = document.getElementById('ticker-dropdown');
    dropdown.classList.toggle('hidden');
    if (!dropdown.classList.contains('hidden')) {
        document.getElementById('ticker-search').focus();
        renderTickerList(allSymbols);
    }
});

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    const selector = document.getElementById('ticker-selector');
    if (!selector.contains(e.target)) {
        document.getElementById('ticker-dropdown').classList.add('hidden');
    }
});

// Stop propagation on dropdown to prevent closing
document.getElementById('ticker-dropdown').addEventListener('click', (e) => {
    e.stopPropagation();
});

// Ticker search (debounced)
let _tickerSearchTimer = null;
document.getElementById('ticker-search').addEventListener('input', (e) => {
    clearTimeout(_tickerSearchTimer);
    _tickerSearchTimer = setTimeout(() => {
        const query = e.target.value.toUpperCase().trim();
        const filtered = query
            ? allSymbols.filter(s => s.includes(query))
            : allSymbols;
        renderTickerList(filtered);
    }, 150);
});

// Ticker item click (delegated)
document.getElementById('ticker-list').addEventListener('click', (e) => {
    const item = e.target.closest('.ticker-item');
    if (item) {
        selectTicker(item.dataset.symbol);
    }
});

// ── Timeframe Selector ──
document.querySelectorAll('.tf-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentInterval = btn.dataset.tf;
        updateChartHeader();
        stopLiveUpdates();
        patternResponseCache.clear();
        loadData().then(() => startLiveUpdates());
    });
});

// ── View Tabs (mode tabs; click again to deactivate) ──
document.getElementById('tab-recognizing')?.addEventListener('click', async () => {
    await setToolMode('recognize');
});

document.getElementById('tab-draw')?.addEventListener('click', async () => {
    await setToolMode('draw');
});

document.getElementById('tab-assist')?.addEventListener('click', async () => {
    await setToolMode('assist');
});

// ── S/R Toggle ──
document.getElementById('sr-toggle').addEventListener('change', (e) => {
    srVisible = e.target.checked;
    scheduleDrawPatterns();
});

// ── Max Lines ──
document.getElementById('max-lines-select').addEventListener('change', (e) => {
    maxSRLines = parseInt(e.target.value);
    scheduleDrawPatterns();
});

// ── Replay Controls ──
function initReplayPanel() {
    // Populate hour dropdown (00:00 – 23:00)
    const hourSelect = document.getElementById('replay-hour');
    for (let h = 0; h < 24; h++) {
        const opt = document.createElement('option');
        opt.value = h;
        opt.textContent = String(h).padStart(2, '0') + ':00';
        hourSelect.appendChild(opt);
    }

    // Default date to today
    const today = new Date().toISOString().slice(0, 10);
    document.getElementById('replay-date').value = today;
}

document.getElementById('replay-toggle').addEventListener('change', (e) => {
    const panel = document.getElementById('replay-panel');
    panel.classList.toggle('hidden', !e.target.checked);
    loadData();
});

document.getElementById('replay-date').addEventListener('change', () => {
    if (document.getElementById('replay-toggle').checked) loadData();
});

document.getElementById('replay-hour').addEventListener('change', () => {
    if (document.getElementById('replay-toggle').checked) loadData();
});

// Timezone toggle buttons
document.getElementById('tz-group').addEventListener('click', (e) => {
    const btn = e.target.closest('.tz-btn');
    if (!btn) return;
    document.querySelectorAll('#tz-group .tz-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (document.getElementById('replay-toggle').checked) loadData();
});

// ── Keyboard Shortcuts ──
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.getElementById('ticker-dropdown').classList.add('hidden');
    }
});

// Scale mode toggle (bottom-right)
document.getElementById('chart-scale-toggle').addEventListener('click', (e) => {
    const btn = e.target.closest('.scale-btn');
    if (!btn) return;
    const mode = btn.dataset.scale === 'log' ? 'log' : 'linear';
    setScaleMode(mode);
});

// Magnet mode toggle (bottom-right)
document.getElementById('chart-magnet-toggle').addEventListener('click', (e) => {
    const btn = e.target.closest('.magnet-btn');
    if (!btn) return;
    magnetMode = btn.dataset.magnet === 'strong' ? 'strong' : 'weak';

    const weakBtn = document.querySelector('#chart-magnet-toggle .magnet-btn[data-magnet="weak"]');
    const strongBtn = document.querySelector('#chart-magnet-toggle .magnet-btn[data-magnet="strong"]');
    if (weakBtn && strongBtn) {
        weakBtn.classList.toggle('active', magnetMode === 'weak');
        strongBtn.classList.toggle('active', magnetMode === 'strong');
    }

    // 重新按当前磁铁模式绘制所有趋势线
    scheduleDrawPatterns();
});

// ── Drawing toolbar ──
function updateDrawingModeUI() {
    removePreviewLine();
    pendingDrawPoint = null;
    const chartArea = document.getElementById('chart-area');
    chartArea.classList.toggle('drawing-mode', !!drawingMode);
    document.getElementById('draw-trend').classList.toggle('active', drawingMode === 'trend');
    document.getElementById('draw-horizontal').classList.toggle('active', drawingMode === 'horizontal');
}

document.getElementById('draw-trend').addEventListener('click', () => {
    drawingMode = drawingMode === 'trend' ? null : 'trend';
    pendingDrawPoint = null;
    updateDrawingModeUI();
});

document.getElementById('draw-horizontal').addEventListener('click', () => {
    drawingMode = drawingMode === 'horizontal' ? null : 'horizontal';
    pendingDrawPoint = null;
    updateDrawingModeUI();
});

document.getElementById('draw-clear').addEventListener('click', () => {
    userDrawnLines = [];
    pendingDrawPoint = null;
    drawingMode = null;
    updateDrawingModeUI();
    scheduleDrawPatterns();
});

// ── AI Chat Panel ──
let chatPanelOpen = false;
let chatSending = false;

function _adjustChartForPanels() {
    // Shrink chart area when side panels are open so they don't overlap
    const chartArea = document.getElementById('chart-area');
    if (!chartArea) return;
    const CHAT_W = 440, AGENT_W = 420;
    let rightOffset = 0;
    if (chatPanelOpen) rightOffset = CHAT_W;
    else if (agentPanelOpen) rightOffset = AGENT_W;
    chartArea.style.right = rightOffset + 'px';
    // ResizeObserver on #chart-container handles chart.resize automatically
}

function toggleChatPanel() {
    chatPanelOpen = !chatPanelOpen;
    if (chatPanelOpen) agentPanelOpen = false; // close agent when chat opens
    const panel = document.getElementById('chat-panel');
    const agentPanel = document.getElementById('agent-panel');
    if (panel) panel.classList.toggle('hidden', !chatPanelOpen);
    if (agentPanel) agentPanel.classList.toggle('hidden', !agentPanelOpen);
    if (agentPollTimer) { clearInterval(agentPollTimer); agentPollTimer = null; }
    updateViewTabsUI();
    _adjustChartForPanels();
    if (chatPanelOpen) {
        document.getElementById('chat-input')?.focus();
    }
}

async function sendChatMessage(text) {
    if (!text?.trim() || chatSending) return;

    chatSending = true;
    const messagesEl = document.getElementById('chat-messages');
    const sendBtn = document.getElementById('chat-send-btn');
    const inputEl = document.getElementById('chat-input');

    // Remove welcome message
    const welcome = messagesEl.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-msg user';
    userMsg.textContent = text;
    messagesEl.appendChild(userMsg);

    // Add thinking indicator
    const thinkMsg = document.createElement('div');
    thinkMsg.className = 'chat-msg thinking';
    thinkMsg.textContent = 'Thinking...';
    messagesEl.appendChild(thinkMsg);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendBtn.disabled = true;

    const model = document.getElementById('chat-model-select')?.value || 'claude-sonnet';

    try {
        const resp = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                session_id: 'default',
                model: model,
            }),
        });

        thinkMsg.remove();

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();

        // Add assistant message
        const assistMsg = document.createElement('div');
        assistMsg.className = 'chat-msg assistant';

        // Show tool badges if tools were called
        let toolBadgesHtml = '';
        if (data.tool_calls?.length) {
            toolBadgesHtml = data.tool_calls.map(tc =>
                `<span class="tool-badge">${tc.tool}</span>`
            ).join('') + '<br>';
        }

        // Simple markdown rendering (bold, code, pre)
        let reply = data.reply || '';
        reply = reply.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        reply = reply.replace(/`([^`]+)`/g, '<code>$1</code>');
        reply = reply.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        assistMsg.innerHTML = toolBadgesHtml + reply;
        messagesEl.appendChild(assistMsg);

    } catch (e) {
        thinkMsg.remove();
        const errMsg = document.createElement('div');
        errMsg.className = 'chat-msg assistant';
        errMsg.style.color = '#ef5350';
        errMsg.textContent = `Error: ${e.message}`;
        messagesEl.appendChild(errMsg);
    }

    messagesEl.scrollTop = messagesEl.scrollHeight;
    chatSending = false;
    sendBtn.disabled = false;
    inputEl.focus();
}

// Chat event handlers
document.getElementById('tab-chat')?.addEventListener('click', () => toggleChatPanel());
document.getElementById('chat-close')?.addEventListener('click', () => toggleChatPanel());

document.getElementById('chat-send-btn')?.addEventListener('click', () => {
    sendChatMessage(document.getElementById('chat-input')?.value);
});

document.getElementById('chat-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage(e.target.value);
    }
});

// Auto-resize textarea
document.getElementById('chat-input')?.addEventListener('input', (e) => {
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
});

// Suggestion buttons
document.getElementById('chat-messages')?.addEventListener('click', (e) => {
    const btn = e.target.closest('.chat-suggestion');
    if (btn) {
        sendChatMessage(btn.dataset.msg);
    }
});

// Clear chat
document.getElementById('chat-clear-btn')?.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/chat/clear?session_id=default`, { method: 'POST' });
    const messagesEl = document.getElementById('chat-messages');
    messagesEl.innerHTML = `
        <div class="chat-welcome">
            <p>Chat cleared. Ask me anything!</p>
            <div class="chat-suggestions">
                <button class="chat-suggestion" data-msg="分析一下 BTC 现在的走势">分析 BTC 走势</button>
                <button class="chat-suggestion" data-msg="HYPE 的支撑阻力在哪里？">HYPE 支撑阻力</button>
                <button class="chat-suggestion" data-msg="帮我跑一下 ETH 4h 的回测">ETH 回测</button>
            </div>
        </div>`;
});

// ── Agent Dashboard ──
let agentPanelOpen = false;
let agentPollTimer = null;

function toggleAgentPanel() {
    agentPanelOpen = !agentPanelOpen;
    if (agentPanelOpen) chatPanelOpen = false; // close chat when agent opens
    const panel = document.getElementById('agent-panel');
    const chatPanel = document.getElementById('chat-panel');
    if (panel) panel.classList.toggle('hidden', !agentPanelOpen);
    if (chatPanel) chatPanel.classList.toggle('hidden', !chatPanelOpen);
    updateViewTabsUI();
    _adjustChartForPanels();
    if (agentPanelOpen) {
        refreshAgentStatus();
        refreshOKXStatus();
        agentPollTimer = setInterval(refreshAgentStatus, 5000);
    } else {
        if (agentPollTimer) { clearInterval(agentPollTimer); agentPollTimer = null; }
    }
}

let _agentPollInFlight = false;
let _lastOKXBalanceFetch = 0;
async function refreshAgentStatus() {
    if (_agentPollInFlight) return;
    _agentPollInFlight = true;
    try {
        const res = await fetch(`${API_BASE}/api/agent/status`);
        if (!res.ok) return;
        const d = await res.json();
        const $ = id => document.getElementById(id);
        const mode = (d.mode || 'paper').toLowerCase();
        const isLive = mode === 'live';

        // Hidden compat field
        const modeEl = $('agent-mode');
        if (modeEl) modeEl.textContent = mode.toUpperCase();

        // Header mode badge
        const headerBadge = $('agent-header-mode');
        if (headerBadge) {
            headerBadge.textContent = isLive ? 'LIVE' : 'PAPER';
            headerBadge.classList.toggle('live', isLive);
        }
        // Header dot color (green=paper, red=live)
        const headerDot = document.querySelector('.agent-header-dot');
        if (headerDot) {
            headerDot.style.background = isLive ? 'var(--accent-red)' : 'var(--accent-green)';
            headerDot.style.boxShadow = isLive ? '0 0 8px var(--accent-red)' : '0 0 8px var(--accent-green)';
        }

        // Mode banner
        const banner = $('mode-banner');
        const bannerText = $('mode-banner-text');
        const paperBtn = $('mode-switch-paper');
        const liveBtn = $('mode-switch-live');
        if (banner) {
            banner.classList.toggle('live-mode', isLive);
        }
        if (bannerText) {
            bannerText.textContent = isLive ? 'LIVE TRADING — Real Money' : 'PAPER MODE — Simulated';
        }
        if (paperBtn) { paperBtn.classList.toggle('active', !isLive); }
        if (liveBtn) { liveBtn.classList.toggle('active', isLive); }

        // Status
        const statusEl = $('agent-status');
        if (statusEl) {
            if (d.running) {
                statusEl.textContent = 'RUNNING';
                statusEl.className = 'agent-stat-value status-running';
            } else if (d.emergency_shutdown) {
                statusEl.textContent = 'SHUTDOWN';
                statusEl.className = 'agent-stat-value status-shutdown';
            } else {
                statusEl.textContent = 'STOPPED';
                statusEl.className = 'agent-stat-value status-stopped';
            }
        }

        // Gen / Cycle combined
        const genEl = $('agent-gen');
        const cycle = d.harness?.cycle || 0;
        if (genEl) genEl.textContent = `${d.generation ?? 0} / ${cycle}`;
        const cycleEl = $('agent-cycle');
        if (cycleEl) cycleEl.textContent = cycle;

        // Account hero: Equity — show OKX balance in live mode
        const equityEl = $('agent-equity');
        const sourceEl = $('agent-equity-source');
        if (isLive) {
            // Fetch OKX balance every 15s max
            const now = Date.now();
            if (now - _lastOKXBalanceFetch > 15000) {
                _lastOKXBalanceFetch = now;
                try {
                    const okxRes = await fetch(`${API_BASE}/api/agent/okx-status`);
                    if (okxRes.ok) {
                        const okxData = await okxRes.json();
                        if (okxData.balance?.total_equity != null) {
                            if (equityEl) equityEl.textContent = `$${okxData.balance.total_equity.toFixed(2)}`;
                            if (sourceEl) sourceEl.textContent = `OKX Live | USDT: $${(okxData.balance.usdt_available ?? 0).toFixed(2)}`;
                        } else {
                            if (equityEl) equityEl.textContent = d.equity != null ? `$${d.equity.toFixed(2)}` : '—';
                            if (sourceEl) sourceEl.textContent = 'OKX Live (balance unavailable)';
                        }
                    }
                } catch (_) {
                    if (equityEl && d.equity != null) equityEl.textContent = `$${d.equity.toFixed(2)}`;
                    if (sourceEl) sourceEl.textContent = 'OKX Live (offline)';
                }
            }
        } else {
            if (equityEl) equityEl.textContent = d.equity != null ? `$${d.equity.toFixed(2)}` : '—';
            if (sourceEl) sourceEl.textContent = 'Paper Account';
        }

        // Cash
        const cashEl = $('agent-cash');
        if (cashEl) cashEl.textContent = d.cash != null ? `$${d.cash.toFixed(2)}` : '—';

        // Hero PnL
        const pnl = d.total_pnl_usd ?? d.total_pnl;
        const pnlEl = $('agent-pnl');
        if (pnlEl && pnl != null) {
            const sign = pnl >= 0 ? '+' : '';
            pnlEl.textContent = `${sign}$${pnl.toFixed(2)}`;
            pnlEl.className = `hero-stat-value ${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
        } else if (pnlEl) { pnlEl.textContent = '$0.00'; }

        // Hero Win Rate
        const wrEl = $('agent-winrate');
        if (wrEl) wrEl.textContent = d.win_rate != null ? `${d.win_rate.toFixed(1)}%` : '0.0%';

        // Hero Trades
        const tradesEl = $('agent-trades');
        if (tradesEl) tradesEl.textContent = d.total_trades ?? '0';

        // Hero Daily PnL
        const dailyEl = $('agent-daily-pnl');
        if (dailyEl && d.daily_pnl != null) {
            const sign = d.daily_pnl >= 0 ? '+' : '';
            dailyEl.textContent = `${sign}$${d.daily_pnl.toFixed(2)}`;
            dailyEl.className = `hero-stat-value ${d.daily_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
        } else if (dailyEl) { dailyEl.textContent = '$0.00'; }

        // Harness: regime, phase, lessons
        if (d.harness) {
            const h = d.harness;
            const regimeEl = $('agent-regime');
            if (regimeEl) {
                regimeEl.textContent = (h.market_regime || 'unknown').toUpperCase();
                regimeEl.className = `agent-stat-value regime-badge regime-${h.market_regime || 'unknown'}`;
            }
            const phaseEl = $('agent-phase');
            if (phaseEl) phaseEl.textContent = (d.cycle_phase || 'idle').toUpperCase();

            // Lessons display
            const lessonsCountEl = $('lessons-count');
            if (lessonsCountEl) lessonsCountEl.textContent = `${h.total_lessons || 0} lessons`;
            const lessonsList = $('lessons-list');
            if (lessonsList && h.recent && h.recent.length > 0) {
                lessonsList.innerHTML = h.recent.slice().reverse().map(l =>
                    `<div class="lesson-entry ${l.category}"><strong>[${l.category}]</strong> ${l.lesson} <span style="color:#555;font-size:9px;">${l.symbol}</span></div>`
                ).join('');
            } else if (lessonsList && (!h.recent || h.recent.length === 0)) {
                lessonsList.textContent = 'No lessons yet — agent learns from trades automatically.';
            }
        }

        // Positions (API returns {symbol: {side, size, entry_price, unrealized_pnl}})
        const posEl = $('agent-positions');
        const posArr = d.positions ? (Array.isArray(d.positions) ? d.positions : Object.values(d.positions)) : [];
        if (posArr.length > 0) {
            posEl.innerHTML = `<table><tr><th>Symbol</th><th>Side</th><th>Size</th><th>Entry</th><th>PnL%</th></tr>` +
                posArr.map(p => {
                    const pnlCls = (p.unrealized_pnl ?? 0) >= 0 ? 'pnl-positive' : 'pnl-negative';
                    return `<tr><td>${p.symbol}</td><td class="${p.side === 'long' ? 'pnl-positive' : 'pnl-negative'}">${(p.side || '—').toUpperCase()}</td><td>$${p.size}</td><td>${p.entry_price?.toFixed?.(2) ?? p.entry_price ?? '—'}</td><td class="${pnlCls}">${(p.unrealized_pnl ?? 0).toFixed(2)}%</td></tr>`;
                }).join('') + `</table>`;
        } else { posEl.textContent = 'None'; }

        // Recent trades
        const trEl = $('agent-recent-trades');
        if (d.recent_trades && d.recent_trades.length > 0) {
            trEl.innerHTML = `<table><tr><th>Symbol</th><th>Side</th><th>PnL%</th><th>PnL$</th><th>Reason</th></tr>` +
                d.recent_trades.slice(-10).reverse().map(t => {
                    const pnlVal = t.pnl_usd ?? t.pnl ?? 0;
                    const pnlPct = t.pnl_pct ?? 0;
                    const pnlCls = pnlVal >= 0 ? 'pnl-positive' : 'pnl-negative';
                    return `<tr><td>${t.symbol}</td><td>${t.side}</td><td class="${pnlCls}">${pnlPct.toFixed(2)}%</td><td class="${pnlCls}">$${pnlVal.toFixed(2)}</td><td>${t.reason || '—'}</td></tr>`;
                }).join('') + `</table>`;
        } else { trEl.textContent = 'None'; }

        // Current signals (from status.last_signals)
        const sigEl = $('agent-signals');
        const sigs = d.last_signals || {};
        const sigEntries = Object.entries(sigs).filter(([, v]) => v && v.action);
        if (sigEntries.length > 0) {
            sigEl.innerHTML = `<table><tr><th>Symbol</th><th>Signal</th><th>Conf</th><th>Status</th><th>Reason</th></tr>` +
                sigEntries.map(([sym, s]) => {
                    const cls = s.action === 'long' ? 'pnl-positive' : s.action === 'short' ? 'pnl-negative' : '';
                    const blocked = s.blocked ? '<span style="color:#ef5350;font-weight:600">BLOCKED</span>' : '<span style="color:#26a69a">PASS</span>';
                    const blockInfo = s.block_reasons?.length ? `<div style="color:#ef5350;font-size:9px;margin-top:2px">${s.block_reasons.join('; ')}</div>` : '';
                    return `<tr><td>${sym}</td><td class="${cls}">${(s.action || '—').toUpperCase()}</td><td>${(s.confidence ?? 0).toFixed(2)}</td><td>${blocked}</td><td style="font-size:10px;max-width:200px;overflow:hidden;text-overflow:ellipsis">${s.reason || '—'}${blockInfo}</td></tr>`;
                }).join('') + `</table>`;
        } else {
            sigEl.textContent = d.running ? 'Scanning...' : 'Agent stopped — no signals';
        }

        // Sync config fields from server state (skip if user is editing)
        const cfgTf = $('cfg-timeframe');
        if (cfgTf && d.signal_interval && document.activeElement !== cfgTf) cfgTf.value = d.signal_interval;
        const cfgSym = $('cfg-symbols');
        if (cfgSym && d.watch_symbols && document.activeElement !== cfgSym) cfgSym.value = d.watch_symbols.join(',');

        // Sync risk limits from server
        if (d.risk_limits) {
            const rl = d.risk_limits;
            const rlEl = (id, val) => { const el = $(id); if (el && !document.activeElement?.id?.startsWith('rl-')) el.value = val; };
            rlEl('rl-max-pos-pct', (rl.max_position_pct * 100).toFixed(1));
            rlEl('rl-max-exp-pct', (rl.max_total_exposure_pct * 100).toFixed(0));
            rlEl('rl-max-daily-loss', (rl.max_daily_loss_pct * 100).toFixed(1));
            rlEl('rl-max-dd', (rl.max_drawdown_pct * 100).toFixed(0));
            rlEl('rl-max-positions', rl.max_positions);
            rlEl('rl-cooldown', rl.cooldown_seconds);
        }

        // Strategy params (V6) — editable (only rebuild if user is NOT editing)
        $('agent-params-gen').textContent = d.generation ?? '0';
        const paramsEl = $('agent-params');
        const userEditingParams = paramsEl && paramsEl.querySelector('.agent-param-input:focus');
        if (d.strategy_params && typeof d.strategy_params === 'object' && !userEditingParams) {
            // Check if params actually changed to avoid unnecessary DOM rebuild
            const newParamsKey = JSON.stringify(d.strategy_params);
            if (paramsEl._lastParamsKey !== newParamsKey) {
                paramsEl._lastParamsKey = newParamsKey;
                const paramLabels = {
                    'ma5_len': 'MA5', 'ma8_len': 'MA8', 'ema21_len': 'EMA21', 'ma55_len': 'MA55',
                    'bb_length': 'BB Len', 'bb_std_dev': 'BB Std',
                    'dist_ma5_ma8': 'Dist 5-8', 'dist_ma8_ema21': 'Dist 8-21', 'dist_ema21_ma55': 'Dist 21-55',
                    'slope_len': 'Slope Len', 'slope_threshold': 'Slope Thr', 'atr_period': 'ATR',
                };
                const paramStep = {
                    'ma5_len': 1, 'ma8_len': 1, 'ema21_len': 1, 'ma55_len': 1,
                    'bb_length': 1, 'bb_std_dev': 0.1,
                    'dist_ma5_ma8': 0.1, 'dist_ma8_ema21': 0.1, 'dist_ema21_ma55': 0.1,
                    'slope_len': 1, 'slope_threshold': 0.01, 'atr_period': 1,
                };
                paramsEl.innerHTML = Object.entries(d.strategy_params).map(([k, v]) => {
                    const label = paramLabels[k] || k;
                    const val = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v;
                    const step = paramStep[k] || 0.1;
                    return `<div class="agent-param-item">
                        <span class="agent-param-key">${label}</span>
                        <input type="number" class="agent-param-input" data-param="${k}" value="${val}" step="${step}" style="width:60px;background:#252a32;color:#d1d4dc;border:1px solid #363a45;border-radius:3px;text-align:right;font-size:11px;padding:1px 4px;">
                    </div>`;
                }).join('') + '<button id="save-params-btn" style="margin-top:6px;padding:3px 12px;background:#2962ff;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;">Save Params</button>';

                // Attach save handler (use event delegation to avoid stale listeners)
                document.getElementById('save-params-btn')?.addEventListener('click', async () => {
                    const inputs = paramsEl.querySelectorAll('.agent-param-input');
                    const updates = {};
                    inputs.forEach(inp => {
                        updates[inp.dataset.param] = parseFloat(inp.value);
                    });
                    const btn = document.getElementById('save-params-btn');
                    if (btn) { btn.textContent = 'Saving...'; btn.disabled = true; }
                    try {
                        const resp = await fetch(`${API_BASE}/api/agent/strategy-params`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(updates),
                        });
                        const data = await resp.json();
                        if (data.ok) {
                            if (btn) { btn.textContent = 'Saved ✓'; btn.style.background = '#26a69a'; btn.disabled = false; setTimeout(() => { btn.textContent = 'Save Params'; btn.style.background = '#2962ff'; }, 2000); }
                            // Force cache invalidation so next poll picks up new values
                            paramsEl._lastParamsKey = null;
                        } else {
                            if (btn) { btn.textContent = 'Failed'; btn.style.background = '#ef5350'; btn.disabled = false; }
                        }
                    } catch (e) {
                        console.error('Save params error:', e);
                        if (btn) { btn.textContent = 'Error'; btn.style.background = '#ef5350'; btn.disabled = false; }
                    }
                });
            }
        }

        // Fetch & display logs (only when section is expanded)
        const logsBody = $('agent-logs-body');
        if (logsBody && logsBody.style.display !== 'none') {
            try {
                const logRes = await fetch(`${API_BASE}/api/agent/logs?limit=30`);
                if (logRes.ok) {
                    const logData = await logRes.json();
                    const logEl = $('agent-logs');
                    if (logEl && logData.logs) {
                        if (logData.logs.length === 0) {
                            logEl.innerHTML = '<div class="log-line log-info" style="opacity:0.5">Agent not started — click Start to begin V6 strategy</div>';
                        } else {
                            logEl.innerHTML = logData.logs.slice(-30).reverse().map(l => {
                                const msg = l.msg || '';
                                const cls = msg.includes('Error') || msg.includes('error') || msg.includes('FAIL') || msg.includes('SL:') ? 'log-error' :
                                            msg.includes('Opened') || msg.includes('TP:') ? 'log-success' :
                                            msg.includes('[Agent]') ? 'log-agent' :
                                            msg.includes('[OKX]') ? 'log-okx' :
                                            msg.includes('[Data]') ? 'log-data' : 'log-info';
                                const clean = msg.replace(/^\[Agent\]\s*/, '🤖 ').replace(/^\[OKX\]\s*/, '📡 ').replace(/^\[Data\]\s*/, '📊 ');
                                return `<div class="log-line ${cls}"><span class="log-time">${l.time}</span> ${clean}</div>`;
                            }).join('');
                        }
                        logEl.scrollTop = 0;
                    }
                }
            } catch (_) {}
        }

        // Self-healer status (only when section is expanded)
        const healerBody = $('healer-body');
        if (healerBody && healerBody.style.display !== 'none') {
            try {
                const hr = await fetch(`${API_BASE}/api/healer/status`);
                if (hr.ok) {
                    const hd = await hr.json();
                    const badge = $('healer-status-badge');
                    if (badge) { badge.textContent = hd.running ? 'ACTIVE' : 'OFF'; badge.className = `healer-badge${hd.running ? '' : ' inactive'}`; }
                    const fc = $('healer-fix-count'); if (fc) fc.textContent = hd.fix_count ?? 0;
                    const ai = $('healer-ai-status');
                    if (ai) { ai.textContent = hd.has_ai ? 'ON' : 'NO KEY'; ai.style.color = hd.has_ai ? '#26a69a' : '#ef5350'; }
                    const errEl = $('healer-recent-errors');
                    if (errEl) errEl.textContent = (hd.recent_errors || '(no errors)').slice(-400);
                }
            } catch (_) {}
        }
    } catch (e) {
        console.warn('Agent status fetch failed:', e);
    } finally {
        _agentPollInFlight = false;
    }
}

// Agent button handlers
document.getElementById('tab-agent')?.addEventListener('click', () => toggleAgentPanel());
document.getElementById('agent-close')?.addEventListener('click', () => toggleAgentPanel());

// Start/Stop/Revive with loading states
document.getElementById('agent-start-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('agent-start-btn');
    if (btn) { btn.textContent = '...'; btn.disabled = true; }
    try { await fetch(`${API_BASE}/api/agent/start`, { method: 'POST' }); } catch (_) {}
    if (btn) { btn.textContent = 'Start'; btn.disabled = false; }
    refreshAgentStatus();
});
document.getElementById('agent-stop-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('agent-stop-btn');
    if (btn) { btn.textContent = '...'; btn.disabled = true; }
    try { await fetch(`${API_BASE}/api/agent/stop`, { method: 'POST' }); } catch (_) {}
    if (btn) { btn.textContent = 'Stop'; btn.disabled = false; }
    refreshAgentStatus();
});
document.getElementById('agent-revive-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('agent-revive-btn');
    if (btn) { btn.textContent = '...'; btn.disabled = true; }
    try { await fetch(`${API_BASE}/api/agent/revive`, { method: 'POST' }); } catch (_) {}
    if (btn) { btn.textContent = 'Revive'; btn.disabled = false; }
    refreshAgentStatus();
});

// Mode switch buttons
document.getElementById('mode-switch-paper')?.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/agent/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: 'paper' }) });
    _lastOKXBalanceFetch = 0; // force refresh
    refreshAgentStatus();
});
document.getElementById('mode-switch-live')?.addEventListener('click', async () => {
    if (!confirm('Switch to LIVE trading? This will use real money via OKX API.\n\nMake sure your API keys are configured.')) return;
    const resp = await fetch(`${API_BASE}/api/agent/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: 'live' }) });
    const data = await resp.json();
    if (!data.ok) {
        alert('Cannot switch to live: ' + (data.reason || 'Unknown error'));
    }
    _lastOKXBalanceFetch = 0; // force refresh
    refreshAgentStatus();
});

// Scan Now button — trigger fresh signal scan
document.getElementById('scan-now-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('scan-now-btn');
    const sigEl = document.getElementById('agent-signals');
    if (btn) { btn.textContent = 'Scanning...'; btn.disabled = true; }
    if (sigEl) sigEl.innerHTML = '<div style="color:#787b86">Scanning all symbols...</div>';
    try {
        const resp = await fetch(`${API_BASE}/api/agent/signals`);
        const data = await resp.json();
        const entries = Object.entries(data.signals || {}).filter(([, v]) => v && v.action);
        if (entries.length > 0) {
            sigEl.innerHTML = `<table><tr><th>Symbol</th><th>Signal</th><th>Conf</th><th>Reason</th></tr>` +
                entries.map(([sym, s]) => {
                    const cls = s.action === 'long' ? 'pnl-positive' : s.action === 'short' ? 'pnl-negative' : '';
                    return `<tr><td>${sym}</td><td class="${cls}">${(s.action || '—').toUpperCase()}</td><td>${(s.confidence ?? 0).toFixed(2)}</td><td style="font-size:10px;max-width:200px">${s.reason || '—'}</td></tr>`;
                }).join('') + `</table>`;
        } else {
            sigEl.textContent = 'No signals detected across watched symbols';
        }
    } catch (e) {
        if (sigEl) sigEl.textContent = 'Scan failed: ' + e.message;
    }
    if (btn) { btn.textContent = 'Scan Now'; btn.disabled = false; }
});
document.getElementById('healer-trigger-btn')?.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/healer/trigger`, { method: 'POST' });
    setTimeout(refreshAgentStatus, 2000);
});

// ── OKX Trading Config ──
document.getElementById('okx-save-keys-btn')?.addEventListener('click', async () => {
    const apiKey = document.getElementById('okx-api-key')?.value?.trim();
    const secret = document.getElementById('okx-secret')?.value?.trim();
    const passphrase = document.getElementById('okx-passphrase')?.value?.trim();
    const msgEl = document.getElementById('okx-status-msg');
    if (!apiKey || !secret || !passphrase) {
        if (msgEl) msgEl.textContent = 'Please fill all 3 fields';
        return;
    }
    if (msgEl) msgEl.textContent = 'Verifying keys...';
    try {
        const resp = await fetch(`${API_BASE}/api/agent/okx-keys`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: apiKey, secret, passphrase }),
        });
        const data = await resp.json();
        if (data.ok) {
            if (msgEl) msgEl.innerHTML = '<span style="color:#26a69a">Keys verified! Balance: $' + (data.balance?.total_equity?.toFixed(2) ?? '—') + '</span>';
            const liveBtn = document.getElementById('okx-go-live-btn');
            if (liveBtn) liveBtn.disabled = false;
        } else {
            if (msgEl) msgEl.innerHTML = '<span style="color:#ef5350">Failed: ' + (data.reason || 'unknown') + '</span>';
        }
    } catch (e) {
        if (msgEl) msgEl.innerHTML = '<span style="color:#ef5350">Error: ' + e.message + '</span>';
    }
    refreshOKXStatus();
});

// (OKX Go Live / Paper buttons removed — use top mode banner instead)

// ── Strategy Config ──
// Toggle symbols input visibility based on watch mode
document.getElementById('cfg-watch-mode')?.addEventListener('change', (e) => {
    const row = document.getElementById('cfg-symbols-row');
    if (row) row.style.display = e.target.value === 'manual' ? 'flex' : 'none';
});

document.getElementById('cfg-apply-btn')?.addEventListener('click', async () => {
    const msgEl = document.getElementById('cfg-status-msg');
    const timeframe = document.getElementById('cfg-timeframe')?.value;
    const watchMode = document.getElementById('cfg-watch-mode')?.value || 'manual';
    const symbolsRaw = document.getElementById('cfg-symbols')?.value?.trim();
    const tick = parseInt(document.getElementById('cfg-tick')?.value || '60');
    const posPct = parseFloat(document.getElementById('cfg-pos-pct')?.value || '5');
    const maxPos = parseInt(document.getElementById('cfg-max-pos')?.value || '3');

    const body = { timeframe, tick_interval: tick, max_position_pct: posPct, max_positions: maxPos };

    if (watchMode === 'manual') {
        body.symbols = symbolsRaw ? symbolsRaw.split(',').map(s => s.trim()).filter(Boolean) : null;
    } else {
        // top5, top10, top20
        body.top_volume = parseInt(watchMode.replace('top', ''));
    }

    try {
        if (msgEl) msgEl.textContent = 'Applying...';
        const resp = await fetch(`${API_BASE}/api/agent/strategy-config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (data.ok && data.changes?.length > 0) {
            if (msgEl) msgEl.innerHTML = `<span style="color:#26a69a">Applied: ${data.changes.join(', ')}</span>`;
            // Update symbols field with server's actual watch list
            if (data.watch_symbols) {
                const symEl = document.getElementById('cfg-symbols');
                if (symEl) symEl.value = data.watch_symbols.join(',');
            }
        } else {
            if (msgEl) msgEl.innerHTML = '<span style="color:#787b86">No changes</span>';
        }
    } catch (e) {
        if (msgEl) msgEl.innerHTML = `<span style="color:#ef5350">Error: ${e.message}</span>`;
    }
    refreshAgentStatus();
});

// Risk limits save handler
document.getElementById('rl-save-btn')?.addEventListener('click', async () => {
    const msgEl = document.getElementById('rl-status-msg');
    const body = {
        max_position_pct: parseFloat(document.getElementById('rl-max-pos-pct')?.value || '5'),
        max_total_exposure_pct: parseFloat(document.getElementById('rl-max-exp-pct')?.value || '15'),
        max_daily_loss_pct: parseFloat(document.getElementById('rl-max-daily-loss')?.value || '2'),
        max_drawdown_pct: parseFloat(document.getElementById('rl-max-dd')?.value || '5'),
        max_positions: parseInt(document.getElementById('rl-max-positions')?.value || '3'),
        cooldown_seconds: parseInt(document.getElementById('rl-cooldown')?.value || '3600'),
    };
    try {
        if (msgEl) msgEl.textContent = 'Saving...';
        const resp = await fetch(`${API_BASE}/api/agent/risk-limits`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (data.ok && data.changes?.length > 0) {
            if (msgEl) msgEl.innerHTML = `<span style="color:#26a69a">Saved: ${data.changes.join(', ')}</span>`;
            setTimeout(() => { if (msgEl) msgEl.textContent = ''; }, 3000);
        } else {
            if (msgEl) msgEl.innerHTML = '<span style="color:#787b86">No changes</span>';
        }
    } catch (e) {
        if (msgEl) msgEl.innerHTML = `<span style="color:#ef5350">Error: ${e.message}</span>`;
    }
});

async function refreshOKXStatus() {
    try {
        const resp = await fetch(`${API_BASE}/api/agent/okx-status`);
        if (!resp.ok) return;
        const data = await resp.json();
        const badge = document.getElementById('okx-status-badge');
        const balInfo = document.getElementById('okx-balance-info');
        if (badge) {
            if (data.has_keys && data.balance) {
                badge.textContent = data.mode === 'live' ? 'LIVE' : 'READY';
                badge.style.background = data.mode === 'live' ? '#ef5350' : '#26a69a';
            } else if (data.has_keys) {
                badge.textContent = 'KEY ERR';
                badge.style.background = '#ff9800';
            } else {
                badge.textContent = 'NO KEY';
                badge.style.background = '#787b86';
            }
        }
        if (balInfo && data.balance) {
            balInfo.textContent = 'Equity: $' + (data.balance.total_equity?.toFixed(2) || '—') + ' | USDT: $' + (data.balance.usdt_available?.toFixed(2) || '—');
        }
        // Update agent equity display with real balance in live mode
        if (data.mode === 'live' && data.balance?.total_equity) {
            const eqEl = document.getElementById('agent-equity');
            if (eqEl) eqEl.textContent = `$${data.balance.total_equity.toFixed(2)} (LIVE)`;
        }
    } catch (_) {}
}

// (Data priority removed — auto-adaptive)

// ── Refresh button ──
document.getElementById('data-refresh-btn')?.addEventListener('click', () => {
    loadData();
});

// (Quick command and Layout presets removed — not useful, caused refresh loops)

// ── Strategy presets ──
document.getElementById('strategy-select')?.addEventListener('change', async (e) => {
    const presets = {
        default:     { dist_ma5_ma8: 1.5, dist_ma8_ema21: 2.5, dist_ema21_ma55: 4.0, slope_threshold: 0.1, bb_std_dev: 2.5 },
        momentum:    { dist_ma5_ma8: 2.0, dist_ma8_ema21: 3.5, dist_ema21_ma55: 5.0, slope_threshold: 0.05, bb_std_dev: 2.0 },
        mean_revert: { dist_ma5_ma8: 1.0, dist_ma8_ema21: 1.5, dist_ema21_ma55: 3.0, slope_threshold: 0.15, bb_std_dev: 3.0 },
        defensive:   { dist_ma5_ma8: 1.0, dist_ma8_ema21: 2.0, dist_ema21_ma55: 3.5, slope_threshold: 0.2, bb_std_dev: 2.5 },
    };
    const params = presets[e.target.value];
    if (!params) return;
    try {
        await fetch(`${API_BASE}/api/agent/strategy-params`, {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(params)
        });
    } catch (_) {}
});

// ── Strategy Presets (save/load/delete) ──
async function loadPresetList() {
    try {
        const res = await fetch(`${API_BASE}/api/agent/strategy-presets`);
        const data = await res.json();
        const sel = document.getElementById('preset-select');
        if (!sel || !data.presets) return;
        sel.innerHTML = Object.keys(data.presets).map(name =>
            `<option value="${name}">${name}</option>`
        ).join('');
    } catch (_) {}
}
loadPresetList();

document.getElementById('preset-load-btn')?.addEventListener('click', async () => {
    const name = document.getElementById('preset-select')?.value;
    if (!name) return;
    const msg = document.getElementById('preset-status-msg');
    try {
        const res = await fetch(`${API_BASE}/api/agent/strategy-presets/load`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (data.ok) {
            if (msg) msg.textContent = `Loaded "${name}"`;
            refreshAgentStatus();
        } else {
            if (msg) msg.textContent = data.reason || 'Failed';
        }
    } catch (e) { if (msg) msg.textContent = 'Error: ' + e.message; }
});

document.getElementById('preset-save-btn')?.addEventListener('click', async () => {
    const nameInput = document.getElementById('preset-name-input');
    const name = nameInput?.value?.trim();
    if (!name) { alert('Enter a preset name'); return; }
    const msg = document.getElementById('preset-status-msg');
    try {
        const res = await fetch(`${API_BASE}/api/agent/strategy-presets/save`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (data.ok) {
            if (msg) msg.textContent = `Saved "${name}"`;
            nameInput.value = '';
            loadPresetList();
        }
    } catch (e) { if (msg) msg.textContent = 'Error: ' + e.message; }
});

document.getElementById('preset-delete-btn')?.addEventListener('click', async () => {
    const name = document.getElementById('preset-select')?.value;
    if (!name || !confirm(`Delete preset "${name}"?`)) return;
    const msg = document.getElementById('preset-status-msg');
    try {
        await fetch(`${API_BASE}/api/agent/strategy-presets/delete`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name })
        });
        if (msg) msg.textContent = `Deleted "${name}"`;
        loadPresetList();
    } catch (e) { if (msg) msg.textContent = 'Error: ' + e.message; }
});

// ── Position Sizer (in Agent Dashboard, uses agent equity) ──
function calcPositionSize() {
    const entry = parseFloat(document.getElementById('ps-entry')?.value);
    const stop = parseFloat(document.getElementById('ps-stop')?.value);
    const riskPct = parseFloat(document.getElementById('ps-risk-pct')?.value);
    const out = document.getElementById('ps-result');
    if (!out) return;
    // Get equity from agent status display
    const eqEl = document.getElementById('agent-equity');
    const equity = eqEl ? parseFloat(eqEl.textContent.replace(/[^0-9.-]/g, '')) : NaN;
    if (isNaN(entry) || isNaN(stop) || isNaN(riskPct) || entry === stop) {
        out.textContent = 'Position: — | Risk: —';
        return;
    }
    const bal = isNaN(equity) ? 10000 : equity;
    const riskUsd = bal * riskPct / 100;
    const distPct = Math.abs(entry - stop) / entry * 100;
    const posSize = riskUsd / (Math.abs(entry - stop) / entry);
    const contracts = posSize / entry;
    out.textContent = `Size: $${posSize.toFixed(2)} (${contracts.toFixed(4)}) | Risk: $${riskUsd.toFixed(2)} (${distPct.toFixed(2)}%) | Equity: $${bal.toFixed(0)}`;
}
['ps-entry', 'ps-stop', 'ps-risk-pct'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', calcPositionSize);
});

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    setScaleMode(currentScale);
    magnetMode = 'weak';
    updateChartHeader();
    updateToolbarPrice();
    initReplayPanel();
    updateViewTabsUI();
    const contentEl = document.getElementById('pattern-stats-content');
    if (contentEl) contentEl.textContent = '选择 Recognizing / Assist 模式后显示形态';
    loadSymbols();
    loadData().then(() => startLiveUpdates());
    document.title = `${formatTicker(currentSymbol)} - Crypto TA`;

    // Cleanup intervals on page unload to prevent memory leaks
    window.addEventListener('beforeunload', () => {
        if (agentPollTimer) { clearInterval(agentPollTimer); agentPollTimer = null; }
        if (liveUpdateInterval) { clearInterval(liveUpdateInterval); liveUpdateInterval = null; }
    });

    // ── Collapsible sections ──
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => {
            const targetId = header.dataset.target;
            const body = document.getElementById(targetId);
            if (!body) return;
            const isOpen = body.style.display !== 'none';
            body.style.display = isOpen ? 'none' : 'block';
            const arrow = header.querySelector('.collapse-arrow');
            if (arrow) arrow.textContent = isOpen ? '▶' : '▼';
        });
    });
});
