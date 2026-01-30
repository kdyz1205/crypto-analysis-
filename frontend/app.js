// ── Global State ──
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let currentSymbol = 'HYPEUSDT';
let currentInterval = '1h';
let allSymbols = [];

// Pattern overlay state
let patternLineSeries = [];  // LineSeries objects for trendlines + zones
let srVisible = true;
let maxSRLines = 0;          // 0 = show all
let lastCandle = null;
let rawPatternData = null;

// ── Chart Initialization ──
function initChart() {
    const container = document.getElementById('chart-container');

    chart = LightweightCharts.createChart(container, {
        layout: {
            background: { color: '#131722' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#1e222d' },
            horzLines: { color: '#1e222d' },
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
        rightPriceScale: {
            borderColor: '#2a2e39',
        },
        timeScale: {
            borderColor: '#2a2e39',
            timeVisible: true,
            secondsVisible: false,
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
    const fmt = (v) => {
        if (v >= 1000) return v.toFixed(2);
        if (v >= 1) return v.toFixed(4);
        return v.toFixed(6);
    };

    legend.innerHTML =
        `<span><span class="ohlc-label">O</span> <span style="color:${color}">${fmt(d.open)}</span></span>` +
        `<span><span class="ohlc-label">H</span> <span style="color:${color}">${fmt(d.high)}</span></span>` +
        `<span><span class="ohlc-label">L</span> <span style="color:${color}">${fmt(d.low)}</span></span>` +
        `<span><span class="ohlc-label">C</span> <span style="color:${color}">${fmt(d.close)}</span></span>`;
}

// ── Helpers ──
function getDaysForInterval(interval) {
    return ['5m', '15m'].includes(interval) ? 7 : 60;
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
async function loadData() {
    showLoading(true);

    const params = buildParams();

    try {
        const resp = await fetch(`/api/ohlcv?${params}`);
        if (!resp.ok) {
            const err = await resp.json();
            alert(`Error: ${err.detail || 'Failed to load data'}`);
            return;
        }

        const data = await resp.json();
        candleSeries.setData(data.candles);
        volumeSeries.setData(data.volume);

        // Store last candle and update legend
        if (data.candles.length > 0) {
            lastCandle = data.candles[data.candles.length - 1];
            const last = lastCandle;
            const legend = document.getElementById('ohlc-legend');
            const color = last.close >= last.open ? '#26a69a' : '#ef5350';
            const fmt = (v) => {
                if (v >= 1000) return v.toFixed(2);
                if (v >= 1) return v.toFixed(4);
                return v.toFixed(6);
            };
            legend.innerHTML =
                `<span><span class="ohlc-label">O</span> <span style="color:${color}">${fmt(last.open)}</span></span>` +
                `<span><span class="ohlc-label">H</span> <span style="color:${color}">${fmt(last.high)}</span></span>` +
                `<span><span class="ohlc-label">L</span> <span style="color:${color}">${fmt(last.low)}</span></span>` +
                `<span><span class="ohlc-label">C</span> <span style="color:${color}">${fmt(last.close)}</span></span>`;
        }

        chart.timeScale().fitContent();

        // Load patterns after candle data is set
        loadPatterns(params);
    } catch (e) {
        alert(`Network error: ${e.message}`);
    } finally {
        showLoading(false);
    }
}

// ── Pattern Overlay ──
function clearPatterns() {
    for (const series of patternLineSeries) {
        chart.removeSeries(series);
    }
    patternLineSeries = [];
    // Clear trend indicator
    const box = document.getElementById('trend-indicator');
    if (box) box.classList.add('hidden');
}

async function loadPatterns(existingParams) {
    clearPatterns();
    rawPatternData = null;

    try {
        const resp = await fetch(`/api/patterns?${existingParams}`);
        if (!resp.ok) return;

        rawPatternData = await resp.json();
        drawAllPatterns();
    } catch (e) {
        console.warn('Pattern load failed:', e);
    }
}

function drawAllPatterns() {
    // Preserve the current visible range so extended lines don't cause zoom changes
    const visibleRange = chart.timeScale().getVisibleLogicalRange();

    clearPatterns();
    if (!rawPatternData) return;

    if (srVisible) {
        drawTrendlines(rawPatternData.supportLines, rawPatternData.resistanceLines);
    }
    drawConsolidationZones(rawPatternData.consolidationZones);
    updateTrendIndicator(rawPatternData.trendLabel, rawPatternData.trendSlope);

    // Restore visible range
    if (visibleRange) {
        chart.timeScale().setVisibleLogicalRange(visibleRange);
    }
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
    // Filter by max lines if set
    let filteredSupport = supportLines;
    let filteredResistance = resistanceLines;

    if (maxSRLines > 0 && lastCandle) {
        filteredSupport = filterLinesByProximity(supportLines, lastCandle.low, maxSRLines);
        filteredResistance = filterLinesByProximity(resistanceLines, lastCandle.high, maxSRLines);
    }

    // Support lines: green
    for (const line of filteredSupport) {
        const series = chart.addLineSeries({
            color: 'rgba(38, 166, 154, 0.7)',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });
        series.setData([
            { time: line.x1, value: line.y1 },
            { time: line.x2, value: line.y2 },
        ]);
        patternLineSeries.push(series);
    }

    // Resistance lines: red
    for (const line of filteredResistance) {
        const series = chart.addLineSeries({
            color: 'rgba(239, 83, 80, 0.7)',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });
        series.setData([
            { time: line.x1, value: line.y1 },
            { time: line.x2, value: line.y2 },
        ]);
        patternLineSeries.push(series);
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
    document.getElementById('loading').classList.toggle('hidden', !show);
}

// ── Ticker Selector ──
async function loadSymbols() {
    try {
        const resp = await fetch('/api/symbols');
        allSymbols = await resp.json();
        renderTickerList(allSymbols);
    } catch (e) {
        console.error('Failed to load symbols:', e);
    }
}

function renderTickerList(symbols) {
    const list = document.getElementById('ticker-list');
    list.innerHTML = symbols.map(s =>
        `<div class="ticker-item${s === currentSymbol ? ' active' : ''}" data-symbol="${s}">${formatTicker(s)}</div>`
    ).join('');
}

function formatTicker(s) {
    if (s.endsWith('USDT')) {
        return s.slice(0, -4) + ' / USDT';
    }
    return s;
}

function selectTicker(symbol) {
    currentSymbol = symbol;
    document.getElementById('current-ticker').textContent = formatTicker(symbol);
    document.getElementById('ticker-dropdown').classList.add('hidden');
    document.getElementById('ticker-search').value = '';
    document.title = `${formatTicker(symbol)} - Crypto TA`;
    loadData();
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
        loadData();
    });
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

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initReplayPanel();
    loadSymbols();
    loadData();
    document.title = `${formatTicker(currentSymbol)} - Crypto TA`;
});
