// ── Global State ──
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let currentSymbol = 'HYPEUSDT';
let currentInterval = '1h';
let allSymbols = [];

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

// ── Data Loading ──
async function loadData() {
    showLoading(true);

    const params = new URLSearchParams({
        symbol: currentSymbol,
        interval: currentInterval,
        days: 30,
    });

    const replayToggle = document.getElementById('replay-toggle');
    const replayTime = document.getElementById('replay-time');
    if (replayToggle.checked && replayTime.value) {
        params.set('end_time', replayTime.value);
    }

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

        // Update OHLC legend with last candle
        if (data.candles.length > 0) {
            const last = data.candles[data.candles.length - 1];
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
    } catch (e) {
        alert(`Network error: ${e.message}`);
    } finally {
        showLoading(false);
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
    // HYPEUSDT -> HYPE / USDT
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

// ── Replay Controls ──
document.getElementById('replay-toggle').addEventListener('change', (e) => {
    const timeInput = document.getElementById('replay-time');
    timeInput.classList.toggle('hidden', !e.target.checked);
    if (!e.target.checked) {
        loadData();
    }
});

document.getElementById('replay-time').addEventListener('change', () => {
    loadData();
});

// ── Keyboard Shortcuts ──
document.addEventListener('keydown', (e) => {
    // Escape closes dropdown
    if (e.key === 'Escape') {
        document.getElementById('ticker-dropdown').classList.add('hidden');
    }
});

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    loadSymbols();
    loadData();
    document.title = `${formatTicker(currentSymbol)} - Crypto TA`;
});
