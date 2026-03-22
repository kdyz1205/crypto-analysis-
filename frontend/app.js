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

/** Chart-native: time -> bar_index (integer). Uses lastCandles. */
function timeToBarIndex(time) {
    if (!lastCandles?.length) return 0;
    let best = 0;
    let bestDiff = Infinity;
    for (let i = 0; i < lastCandles.length; i++) {
        const d = Math.abs(Number(lastCandles[i].time) - Number(time));
        if (d < bestDiff) { bestDiff = d; best = i; }
    }
    return best;
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

function onChartMouseMove(event) {
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
    drawAllPatterns();
}

function toggleSimilarLineVisibility(index) {
    if (visibleSimilarIndices.has(index)) {
        visibleSimilarIndices.delete(index);
    } else {
        visibleSimilarIndices.add(index);
    }
    drawAllPatterns();
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
    if (interval === '1m') return 2;
    if (['5m', '15m'].includes(interval)) return 90;
    if (interval === '2h') return 120;
    if (interval === '1d') return 90;   // fewer days = faster 4h<->1d switch
    return 180;  // 1h, 4h: balance between history and speed
}

function buildParams() {
    const params = new URLSearchParams({
        symbol: currentSymbol,
        interval: currentInterval,
        days: getDaysForInterval(currentInterval),
    });

    const replayToggle = document.getElementById('replay-toggle');
    if (replayToggle.checked) {
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
const LOAD_DATA_TIMEOUT_MS = 90000; // 90s for API-only first load (OKX pagination can be slow)
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
                setTimeout(() => fetchPatternStats(), 0);
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
        chart.removeSeries(series);
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

function drawAllPatterns() {
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
    let nearest = lastCandles[0];
    let minDt = Math.abs(time - nearest.time);
    for (let i = 1; i < lastCandles.length; i++) {
        const c = lastCandles[i];
        const dt = Math.abs(time - c.time);
        if (dt < minDt) {
            minDt = dt;
            nearest = c;
        }
    }
    return nearest;
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

    if (label === 'UPTREND') {
        box.classList.add('trend-up');
        box.textContent = `UPTREND (${slope > 0 ? '+' : ''}${slope.toFixed(2)}%)`;
    } else if (label === 'DOWNTREND') {
        box.classList.add('trend-down');
        box.textContent = `DOWNTREND (${slope > 0 ? '+' : ''}${slope.toFixed(2)}%)`;
    } else {
        box.classList.add('trend-sideways');
        box.textContent = `SIDEWAYS (${slope > 0 ? '+' : ''}${slope.toFixed(2)}%)`;
    }
}

function showLoading(show) {
    const el = document.getElementById('loading');
    if (!el) return;
    el.classList.toggle('loading-hidden', !show);
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

async function setToolMode(nextMode) {
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
        drawAllPatterns();
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
        drawAllPatterns();
        return;
    }

    if (toolMode === 'assist') {
        if (contentEl) contentEl.textContent = 'Assist: 趋势线（延伸）. 与 Recognizing 分离，仅显示辅助线.';
        updateAssistPanelNotification();
        rawPatternData = null;
        drawAllPatterns();
        await loadPatterns(buildPatternParams().toString());
        drawAllPatterns();
        if (contentEl && rawPatternData) {
            const s = rawPatternData.supportLines?.length || 0;
            const r = rawPatternData.resistanceLines?.length || 0;
            if (s || r) contentEl.textContent = `Assist: ${s} 支撑、${r} 阻力（延伸）`;
        }
        return;
    }
    if (contentEl) contentEl.textContent = 'Recognizing: loading patterns...';
    await loadPatterns(buildPatternParams().toString());
    if (toolMode === 'recognize') updateRecognizePanelContent();
    await fetchPatternStats();
}

function selectTicker(symbol) {
    currentSymbol = symbol;
    lastCandle = null;
    document.getElementById('current-ticker').textContent = formatTicker(symbol);
    document.getElementById('ticker-dropdown').classList.add('hidden');
    updateChartHeader();
    updateToolbarPrice();
    document.getElementById('ticker-search').value = '';
    document.title = `${formatTicker(symbol)} - Crypto TA`;
    stopLiveUpdates();
    loadData();
    // startLiveUpdates();
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

// Ticker search
document.getElementById('ticker-search').addEventListener('input', (e) => {
    const query = e.target.value.toUpperCase().trim();
    const filtered = query
        ? allSymbols.filter(s => s.includes(query))
        : allSymbols;
    renderTickerList(filtered);
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
        loadData();
    });
});

// ── View Tabs (mode tabs; click again to deactivate) ──
document.getElementById('tab-recognizing').addEventListener('click', async () => {
    await setToolMode('recognize');
});

document.getElementById('tab-draw').addEventListener('click', async () => {
    await setToolMode('draw');
});

document.getElementById('tab-assist').addEventListener('click', async () => {
    await setToolMode('assist');
});

// ── S/R Toggle ──
document.getElementById('sr-toggle').addEventListener('change', (e) => {
    srVisible = e.target.checked;
    drawAllPatterns();
});

// ── Max Lines ──
document.getElementById('max-lines-select').addEventListener('change', (e) => {
    maxSRLines = parseInt(e.target.value);
    drawAllPatterns();
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
    drawAllPatterns();
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
    drawAllPatterns();
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
    // Let TradingView chart know to resize
    if (chart) setTimeout(() => chart.resize(chartArea.clientWidth, chartArea.clientHeight), 250);
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
        agentPollTimer = setInterval(refreshAgentStatus, 5000);
    } else {
        if (agentPollTimer) { clearInterval(agentPollTimer); agentPollTimer = null; }
    }
}

async function refreshAgentStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/agent/status`);
        if (!res.ok) return;
        const d = await res.json();
        const $ = id => document.getElementById(id);
        $('agent-mode').textContent = (d.mode || '—').toUpperCase();
        const statusEl = $('agent-status');
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
        $('agent-gen').textContent = d.generation ?? '—';
        $('agent-equity').textContent = d.equity != null ? `$${d.equity.toFixed(2)}` : '—';
        $('agent-cash').textContent = d.cash != null ? `$${d.cash.toFixed(2)}` : '—';

        const pnl = d.total_pnl;
        const pnlEl = $('agent-pnl');
        if (pnl != null) {
            pnlEl.textContent = `$${pnl.toFixed(2)}`;
            pnlEl.className = `agent-stat-value ${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
        } else { pnlEl.textContent = '—'; }

        $('agent-winrate').textContent = d.win_rate != null ? `${d.win_rate.toFixed(1)}%` : '—';
        $('agent-trades').textContent = d.total_trades ?? '—';

        const dailyEl = $('agent-daily-pnl');
        if (d.daily_pnl != null) {
            dailyEl.textContent = `$${d.daily_pnl.toFixed(2)}`;
            dailyEl.className = `agent-stat-value ${d.daily_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
        } else { dailyEl.textContent = '—'; }

        // Positions
        const posEl = $('agent-positions');
        if (d.positions && d.positions.length > 0) {
            posEl.innerHTML = `<table><tr><th>Symbol</th><th>Side</th><th>Size</th><th>Entry</th><th>SL</th></tr>` +
                d.positions.map(p => `<tr><td>${p.symbol}</td><td>${p.side}</td><td>${p.size}</td><td>${p.entry_price?.toFixed(2) ?? '—'}</td><td>${p.sl_price?.toFixed(2) ?? '—'}</td></tr>`).join('') +
                `</table>`;
        } else { posEl.textContent = 'None'; }

        // Recent trades
        const trEl = $('agent-recent-trades');
        if (d.recent_trades && d.recent_trades.length > 0) {
            trEl.innerHTML = `<table><tr><th>Symbol</th><th>Side</th><th>PnL</th><th>Time</th></tr>` +
                d.recent_trades.slice(-10).reverse().map(t => {
                    const pnlCls = (t.pnl ?? 0) >= 0 ? 'pnl-positive' : 'pnl-negative';
                    const time = t.exit_time ? new Date(t.exit_time * 1000).toLocaleString() : '—';
                    return `<tr><td>${t.symbol}</td><td>${t.side}</td><td class="${pnlCls}">$${(t.pnl ?? 0).toFixed(2)}</td><td>${time}</td></tr>`;
                }).join('') + `</table>`;
        } else { trEl.textContent = 'None'; }

        // Strategy params
        $('agent-params-gen').textContent = d.generation ?? '0';
        const paramsEl = $('agent-params');
        if (d.strategy_params && typeof d.strategy_params === 'object') {
            paramsEl.innerHTML = Object.entries(d.strategy_params).map(([k, v]) =>
                `<div class="agent-param-item"><span class="agent-param-key">${k}</span><span class="agent-param-val">${typeof v === 'number' ? v.toFixed(2) : v}</span></div>`
            ).join('');
        }

        // Self-healer status
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
    } catch (e) {
        console.warn('Agent status fetch failed:', e);
    }
}

// Agent button handlers
document.getElementById('tab-agent')?.addEventListener('click', () => toggleAgentPanel());
document.getElementById('agent-close')?.addEventListener('click', () => toggleAgentPanel());
document.getElementById('agent-start-btn')?.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/agent/start`, { method: 'POST' });
    refreshAgentStatus();
});
document.getElementById('agent-stop-btn')?.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/agent/stop`, { method: 'POST' });
    refreshAgentStatus();
});
document.getElementById('agent-revive-btn')?.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/agent/revive`, { method: 'POST' });
    refreshAgentStatus();
});
document.getElementById('healer-trigger-btn')?.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/healer/trigger`, { method: 'POST' });
    setTimeout(refreshAgentStatus, 2000);
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
    loadData();
    // startLiveUpdates(); // 关闭自动定时刷新，避免「一直刷新」；需要时可再开启
    document.title = `${formatTicker(currentSymbol)} - Crypto TA`;

    // Cleanup intervals on page unload to prevent memory leaks
    window.addEventListener('beforeunload', () => {
        if (agentPollTimer) { clearInterval(agentPollTimer); agentPollTimer = null; }
        if (liveUpdateInterval) { clearInterval(liveUpdateInterval); liveUpdateInterval = null; }
    });
});
