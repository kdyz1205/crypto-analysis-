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
let lastVolumeBars = [];
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
let dataPriorityMode = 'fast'; // fast | balanced | deep
let tradeJournal = [];
let activeJournalIndex = -1;
const ohlcvResponseCache = new Map(); // key: query string => { ts, data }
const OHLCV_CACHE_MAX = 30;
let selectedUserLineIndex = null;
const prefetchInFlight = new Set();
let _previewFramePending = false;
let _lastFitContentKey = null;

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
            visible: true,
            minimumWidth: 72,
            scaleMargins: { top: 0.08, bottom: 0.12 },
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

function getTimePriceFromCoords(clientX, clientY) {
    if (!chart || !candleSeries) return null;
    const container = document.getElementById('chart-container');
    const rect = container.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
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
    const t = Number(time);
    let lo = 0;
    let hi = lastCandles.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (Number(lastCandles[mid].time) < t) lo = mid + 1;
        else hi = mid;
    }
    const right = lo;
    const left = Math.max(0, right - 1);
    return Math.abs(Number(lastCandles[right]?.time ?? t) - t) < Math.abs(Number(lastCandles[left]?.time ?? t) - t) ? right : left;
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

function hitTestUserLine(time, value) {
    if (!userDrawnLines.length) return null;
    const priceThreshold = value > 0 ? value * 0.004 : 0.05;
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < userDrawnLines.length; i++) {
        const line = userDrawnLines[i];
        const x1 = line.t1 ?? barIndexToTime(line.x1);
        const x2 = line.t2 ?? barIndexToTime(line.x2);
        const y1 = line.y1;
        const y2 = line.type === 'horizontal' ? line.y1 : line.y2;
        const dt = x2 - x1;
        const extend = dt ? Math.max(0, Math.abs(dt) * 0.2) : 0;
        if (time < Math.min(x1, x2) - extend || time > Math.max(x1, x2) + extend) continue;
        const priceAt = dt ? y1 + (y2 - y1) * (time - x1) / dt : y1;
        const dist = Math.abs(value - priceAt);
        if (dist < priceThreshold && dist < bestDist) {
            best = i;
            bestDist = dist;
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
            const t1 = barIndexToTime(x1);
            const t2 = barIndexToTime(x2);
            const xEnd = Math.min(Math.max(x1, x2) + N_FUTURE_BARS, (lastCandles?.length ?? 1) - 1);
            const tEnd = barIndexToTime(xEnd);
            const snapped = applyMagnetToLine({ x1, y1, x2, y2, t1, t2 });
            const dPrice = snapped.y2 - snapped.y1;
            const slope = dPrice / dBar;
            const line = { type: 'trend', x1, y1: snapped.y1, x2, y2: snapped.y2, t1, t2, dBar, dPrice, slope };
            userDrawnLines.push(line);
            selectedUserLineIndex = userDrawnLines.length - 1;
            const series = chart.addLineSeries({
                color: 'rgba(41, 98, 255, 0.9)',
                priceScaleId: '',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            series.setData([
                { time: t1, value: line.y1 },
                { time: tEnd, value: line.y1 + line.slope * (xEnd - x1) },
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
            const snapped = applyMagnetToLine({ x1: barIdx, y1: tp.value, x2: xEnd, y2: tp.value, t1, t2 });
            const line = { type: 'horizontal', x1: barIdx, y1: snapped.y1, x2: xEnd, t1, t2, dBar: xEnd - barIdx, dPrice: 0, slope: 0 };
            userDrawnLines.push(line);
            selectedUserLineIndex = userDrawnLines.length - 1;
            const series = chart.addLineSeries({
                color: 'rgba(255, 193, 7, 0.9)',
                priceScaleId: '',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            series.setData([{ time: t1, value: line.y1 }, { time: t2, value: line.y1 }]);
            patternLineSeries.push(series);
            removePreviewLine();
            pendingDrawPoint = null;
            onDrawLineFinalized(line);
            return;
        }
    }

    // RECOGNIZE mode: click on recognition line to select (legacy)
    if (toolMode === 'draw' && !drawingMode) {
        const hitUser = hitTestUserLine(tp.time, tp.value);
        selectedUserLineIndex = hitUser;
        drawAllPatterns();
        if (hitUser !== null) {
            showStatus(`已选中第 ${hitUser + 1} 条手绘线，按 Delete 删除`, 'success', 1300);
        }
        return;
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
        const { clientX, clientY } = event;
        if (_previewFramePending) return;
        _previewFramePending = true;
        requestAnimationFrame(() => {
            _previewFramePending = false;
            const latest = getTimePriceFromCoords(clientX, clientY);
            if (!latest || !pendingDrawPoint) return;
            const snappedY = snapPriceToCandle(latest.time, latest.value, magnetMode);
            if (!previewLineSeries) {
                previewLineSeries = chart.addLineSeries({
                    color: 'rgba(41, 98, 255, 0.6)',
                    priceScaleId: '',
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Solid,
                    crosshairMarkerVisible: false,
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
            }
            previewLineSeries.setData([
                { time: pendingDrawPoint.time, value: pendingDrawPoint.value },
                { time: latest.time, value: snappedY },
            ]);
        });
        return;
    }

    // No mousemove preview for horizontal (single-click only)
}

function sanitizeCandles(candles) {
    if (!Array.isArray(candles)) return [];
    const out = candles.filter((c) =>
        c && Number.isFinite(Number(c.time)) &&
        Number.isFinite(Number(c.open)) &&
        Number.isFinite(Number(c.high)) &&
        Number.isFinite(Number(c.low)) &&
        Number.isFinite(Number(c.close))
    ).map((c) => ({
        time: Number(c.time),
        open: Number(c.open),
        high: Number(c.high),
        low: Number(c.low),
        close: Number(c.close),
    }));
    out.sort((a, b) => a.time - b.time);
    return out;
}

function sanitizeVolume(volume) {
    if (!Array.isArray(volume)) return [];
    return volume.filter((v) =>
        v && Number.isFinite(Number(v.time)) && Number.isFinite(Number(v.value))
    ).map((v) => ({
        time: Number(v.time),
        value: Number(v.value),
        color: v.color,
    }));
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
let statusBannerTimer = null;

function normalizeSymbolEntry(entry) {
    if (typeof entry === 'string') return entry.toUpperCase().replace('/', '');
    if (entry && typeof entry === 'object') {
        const raw = entry.symbol || entry.instId || entry.id || '';
        return String(raw).toUpperCase().replace('/', '');
    }
    return '';
}

function showStatus(message, type = 'info', timeoutMs = 3500) {
    const el = document.getElementById('status-banner');
    if (!el) return;
    el.textContent = message;
    el.classList.remove('hidden', 'status-error', 'status-success');
    if (type === 'error') el.classList.add('status-error');
    if (type === 'success') el.classList.add('status-success');
    if (statusBannerTimer) clearTimeout(statusBannerTimer);
    if (timeoutMs > 0) {
        statusBannerTimer = setTimeout(() => {
            el.classList.add('hidden');
        }, timeoutMs);
    }
}

function updateChecklistProgress() {
    const ids = ['check-trend', 'check-level', 'check-risk'];
    const done = ids.filter((id) => document.getElementById(id)?.checked).length;
    const el = document.getElementById('check-progress');
    if (el) el.textContent = `${done}/3 complete`;
    updateAICoachSummary();
}

function computeRiskPosition() {
    const balance = parseFloat(document.getElementById('risk-balance')?.value);
    const riskPct = parseFloat(document.getElementById('risk-percent')?.value);
    const entry = parseFloat(document.getElementById('risk-entry')?.value);
    const stop = parseFloat(document.getElementById('risk-stop')?.value);
    const out = document.getElementById('risk-output');
    if (!out) return;
    if (![balance, riskPct, entry, stop].every((v) => Number.isFinite(v) && v > 0)) {
        out.textContent = '建议仓位：—';
        updateAICoachSummary();
        return;
    }
    const perUnitRisk = Math.abs(entry - stop);
    if (perUnitRisk <= 0) {
        out.textContent = '建议仓位：止损不能等于入场价';
        updateAICoachSummary();
        return;
    }
    const riskAmount = balance * (riskPct / 100);
    const qty = riskAmount / perUnitRisk;
    const notional = qty * entry;
    out.textContent = `建议仓位：${qty.toFixed(4)} (${notional.toFixed(2)} USDT 名义仓位)`;
    updateAICoachSummary();
}

function persistJournal() {
    localStorage.setItem('tradeJournal', JSON.stringify(tradeJournal.slice(0, 300)));
}

function renderJournal() {
    const list = document.getElementById('journal-list');
    if (!list) return;
    if (!tradeJournal.length) {
        list.textContent = '暂无记录';
        return;
    }
    list.innerHTML = '';
    tradeJournal.forEach((t, i) => {
        const row = document.createElement('div');
        row.className = 'journal-row' + (i === activeJournalIndex ? ' active' : '');
        row.textContent = `${new Date(t.ts).toLocaleString()} ${t.symbol} ${t.interval} E:${t.entry} S:${t.stop}`;
        row.addEventListener('click', () => replayJournalAt(i));
        list.appendChild(row);
    });
}

function replayJournalAt(index) {
    if (index < 0 || index >= tradeJournal.length) return;
    activeJournalIndex = index;
    const t = tradeJournal[index];
    currentInterval = t.interval;
    selectTicker(t.symbol);
    document.querySelectorAll('.tf-btn').forEach((b) => b.classList.toggle('active', b.dataset.tf === currentInterval));
    document.getElementById('risk-entry').value = t.entry;
    document.getElementById('risk-stop').value = t.stop;
    computeRiskPosition();
    renderJournal();
    showStatus(`已回放第 ${index + 1} 条交易记录`, 'success', 1300);
}

function saveCurrentTradeToJournal() {
    const entry = parseFloat(document.getElementById('risk-entry')?.value);
    const stop = parseFloat(document.getElementById('risk-stop')?.value);
    if (!Number.isFinite(entry) || !Number.isFinite(stop) || entry <= 0 || stop <= 0) {
        showStatus('保存失败：请先输入有效的入场价和止损价。', 'error', 2200);
        return;
    }
    tradeJournal.unshift({
        ts: Date.now(),
        symbol: currentSymbol,
        interval: currentInterval,
        entry,
        stop,
        lastClose: lastCandle?.close ?? null,
    });
    if (tradeJournal.length > 300) tradeJournal = tradeJournal.slice(0, 300);
    activeJournalIndex = 0;
    persistJournal();
    renderJournal();
    showStatus('已保存到交易日志。', 'success', 1200);
}

async function applyLayoutPreset(preset) {
    if (preset === 'scalper') {
        currentInterval = '5m';
        maxSRLines = 3;
        dataPriorityMode = 'fast';
        await setToolMode('assist');
    } else if (preset === 'swing') {
        currentInterval = '4h';
        maxSRLines = 5;
        dataPriorityMode = 'balanced';
        await setToolMode('recognize');
    } else if (preset === 'position') {
        currentInterval = '1d';
        maxSRLines = 10;
        dataPriorityMode = 'deep';
        await setToolMode('recognize');
    } else {
        return;
    }
    document.getElementById('data-priority-select').value = dataPriorityMode;
    document.getElementById('max-lines-select').value = String(maxSRLines);
    document.querySelectorAll('.tf-btn').forEach((b) => b.classList.toggle('active', b.dataset.tf === currentInterval));
    showStatus(`已应用 ${preset} 布局`, 'success', 1400);
    patternResponseCache.clear();
    scheduleLoadData();
}

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
        const resp = await fetch(`/api/pattern-stats/line-similar?${params}`);
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
    const profiles = {
        fast: { '1m': 1, '5m': 30, '15m': 45, '1h': 90, '4h': 120, '1d': 60 },
        balanced: { '1m': 2, '5m': 90, '15m': 90, '1h': 180, '4h': 180, '1d': 120 },
        deep: { '1m': 3, '5m': 180, '15m': 240, '1h': 365, '4h': 365, '1d': 365 },
    };
    const profile = profiles[dataPriorityMode] || profiles.fast;
    return profile[interval] || profile['1h'];
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

function setCachedOHLCV(key, value) {
    if (!key || !value) return;
    if (ohlcvResponseCache.size >= OHLCV_CACHE_MAX) {
        const firstKey = ohlcvResponseCache.keys().next().value;
        ohlcvResponseCache.delete(firstKey);
    }
    ohlcvResponseCache.set(key, { ts: Date.now(), data: value });
}

function getCacheTTLms(interval) {
    const ttl = {
        '5m': 45 * 1000,
        '15m': 90 * 1000,
        '1h': 3 * 60 * 1000,
        '4h': 10 * 60 * 1000,
        '1d': 30 * 60 * 1000,
    };
    return ttl[interval] || 60 * 1000;
}

function getCachedOHLCV(key, interval) {
    const item = ohlcvResponseCache.get(key);
    if (!item?.data || !item?.ts) return null;
    if (Date.now() - item.ts > getCacheTTLms(interval)) {
        ohlcvResponseCache.delete(key);
        return null;
    }
    return item.data;
}

function prefetchAdjacentIntervals() {
    const order = ['5m', '15m', '1h', '4h', '1d'];
    const idx = order.indexOf(currentInterval);
    if (idx < 0) return;
    const targets = [order[idx - 1], order[idx + 1]].filter(Boolean);
    targets.forEach((tf) => {
        const params = new URLSearchParams({
            symbol: currentSymbol,
            interval: tf,
            days: getDaysForInterval(tf),
        });
        const key = params.toString();
        if (getCachedOHLCV(key, tf)) return;
        if (prefetchInFlight.has(key)) return;
        prefetchInFlight.add(key);
        fetch(`/api/ohlcv?${params}`)
            .then((r) => (r.ok ? r.json() : null))
            .then((d) => {
                if (d) setCachedOHLCV(key, d);
            })
            .catch(() => {})
            .finally(() => prefetchInFlight.delete(key));
    });
}

function updateAICoachSummary() {
    const el = document.getElementById('ai-coach-summary');
    if (!el) return;
    const checks = ['check-trend', 'check-level', 'check-risk'].map((id) => !!document.getElementById(id)?.checked);
    const checksDone = checks.filter(Boolean).length;
    const trend = rawPatternData?.trendLabel || 'UNKNOWN';
    const patterns = rawPatternData?.patterns?.length || 0;
    const statsRate = patternStatsData?.overall_success_rate_pct;
    const riskText = document.getElementById('risk-output')?.textContent || '建议仓位：—';
    const bias = trend === 'UPTREND' ? '偏多' : (trend === 'DOWNTREND' ? '偏空' : '震荡');
    const conviction = checksDone >= 3 && (statsRate ?? 0) >= 55 ? '高' : (checksDone >= 2 ? '中' : '低');
    el.textContent =
        `Bias: ${bias} | Conviction: ${conviction}\n` +
        `Trend: ${trend}, Patterns: ${patterns}, 历史同类胜率: ${statsRate ?? '—'}%\n` +
        `Checklist: ${checksDone}/3\n${riskText}`;
}

async function applyQuickCommand(rawCmd) {
    const cmd = (rawCmd || '').trim().toLowerCase();
    if (!cmd) return;
    if (cmd === 'help') {
        showStatus('示例: btc 5m assist / swing / replay on / priority deep', 'info', 3200);
        return;
    }
    if (cmd === 'refresh' || cmd === 'reload') {
        loadData(false, true);
        showStatus('已强制刷新数据。', 'success', 1200);
        return;
    }
    if (cmd === 'swing' || cmd === 'scalper' || cmd === 'position') {
        document.getElementById('layout-select').value = cmd;
        await applyLayoutPreset(cmd);
        return;
    }
    if (cmd.startsWith('priority ')) {
        const mode = cmd.split(' ')[1];
        if (['fast', 'balanced', 'deep'].includes(mode)) {
            dataPriorityMode = mode;
            document.getElementById('data-priority-select').value = mode;
            scheduleLoadData();
            return;
        }
    }
    if (cmd === 'replay on' || cmd === 'replay off') {
        const on = cmd.endsWith('on');
        const toggle = document.getElementById('replay-toggle');
        toggle.checked = on;
        document.getElementById('replay-panel').classList.toggle('hidden', !on);
        scheduleLoadData();
        return;
    }
    const parts = cmd.split(/\s+/);
    const reserved = new Set([
        'assist', 'recognize', 'draw', 'replay', 'on', 'off',
        'priority', 'fast', 'balanced', 'deep', 'swing', 'scalper', 'position', 'help',
    ]);
    const maybeSymbol = parts.find((p) => (/^[a-z0-9]{2,14}$/.test(p) || /^[a-z0-9]{2,14}usdt$/.test(p)) && !reserved.has(p));
    const maybeTf = parts.find((p) => ['5m', '15m', '1h', '4h', '1d'].includes(p));
    const maybeMode = parts.find((p) => ['assist', 'recognize', 'draw'].includes(p));
    if (maybeSymbol) {
        const sym = maybeSymbol.toUpperCase().endsWith('USDT') ? maybeSymbol.toUpperCase() : `${maybeSymbol.toUpperCase()}USDT`;
        const changed = selectTicker(sym, true);
        if (!changed) return;
    }
    if (maybeTf) currentInterval = maybeTf;
    if (maybeMode) await setToolMode(maybeMode === 'recognize' ? 'recognize' : maybeMode);
    document.querySelectorAll('.tf-btn').forEach((b) => b.classList.toggle('active', b.dataset.tf === currentInterval));
    scheduleLoadData();
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
let _loadDataDebounceTimer = null;

function scheduleLoadData(forceRefresh = false, delayMs = 120) {
    if (_loadDataDebounceTimer) clearTimeout(_loadDataDebounceTimer);
    _loadDataDebounceTimer = setTimeout(() => {
        loadData(false, forceRefresh);
        _loadDataDebounceTimer = null;
    }, delayMs);
}

async function loadData(isLiveUpdate = false, forceRefresh = false) {
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
        const cacheKey = params.toString();
        if (forceRefresh) ohlcvResponseCache.delete(cacheKey);
        let data = getCachedOHLCV(cacheKey, currentInterval);
        if (!data) {
            const resp = await fetch(`/api/ohlcv?${params}`, { signal });
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
                    showStatus(`加载失败：${errMsg}`, 'error', 7000);
                }
                return;
            }
            if (signal.aborted) return;
            data = await resp.json();
            setCachedOHLCV(cacheKey, data);
        }
        if (!data) return;
        if (signal.aborted) return;

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
        
        const incomingCandles = sanitizeCandles(data.candles || []);
        const incomingVolume = sanitizeVolume(data.volume || []);
        if (incomingCandles.length === 0 && lastCandles.length > 0 && !forceRefresh) {
            showStatus(`未返回K线数据，保留上次图表（${currentSymbol} ${currentInterval}）`, 'error', 2600);
            if (lastVolumeBars.length > 0) volumeSeries.setData(lastVolumeBars);
        } else {
            lastCandles = incomingCandles;
            lastVolumeBars = incomingVolume;
            candleSeries.setData(lastCandles);
            volumeSeries.setData(lastVolumeBars);
        }
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
            const fitKey = `${currentSymbol}|${currentInterval}|${document.getElementById('replay-toggle')?.checked ? 'replay' : 'live'}`;
            if (fitKey !== _lastFitContentKey || forceRefresh) {
                chart.timeScale().fitContent();
                _lastFitContentKey = fitKey;
            }
        }
        // Patterns load in background so timeframe switch shows candles immediately
        if (isRecognitionOverlayMode() && !isLiveUpdate) {
            loadPatterns(buildPatternParams().toString()).then(() => {
                drawAllPatterns();
                setTimeout(() => fetchPatternStats(), 0);
            });
        }
        drawAllPatterns();
        prefetchAdjacentIntervals();
        updateAICoachSummary();
    } catch (e) {
        if (e.name === 'AbortError') {
            if (!isLiveUpdate) {
                showLoading(false);
                showStatus('加载超时（约 90 秒）。API 拉取较慢，可稍后重试。', 'error', 7000);
            }
            return;
        }
        if (!isLiveUpdate) {
            console.error('Data load error:', e);
            showLoading(false);
            showStatus(`网络错误：${e.message}`, 'error', 7000);
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
        const resp = await fetch(`/api/patterns?${existingParams}`);
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
        const resp = await fetch(`/api/pattern-stats/current-vs-history?${params}`);
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
            updateAICoachSummary();
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
        updateAICoachSummary();
    } catch (e) {
        console.error('Pattern stats fetch failed:', e);
        el.textContent = '—';
        updateAICoachSummary();
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
                priceScaleId: '',
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
                priceScaleId: '',
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
            priceScaleId: '',
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
    return lastCandles[timeToBarIndex(time)] || lastCandles[lastCandles.length - 1];
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
    const t1 = line.t1 ?? barIndexToTime(line.x1);
    const t2 = line.t2 ?? barIndexToTime(line.x2);
    const y1 = snapPriceToCandle(t1, line.y1, magnetMode);
    const y2 = snapPriceToCandle(t2, line.y2, magnetMode);
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
                priceScaleId: '',
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
                priceScaleId: '',
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
            priceScaleId: '',
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
            priceScaleId: '',
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
    for (let idx = 0; idx < userDrawnLines.length; idx++) {
        const line = userDrawnLines[idx];
        const isSelected = idx === selectedUserLineIndex;
        const opts = {
            color: isSelected ? 'rgba(255, 87, 34, 0.95)' : (line.type === 'horizontal' ? 'rgba(255, 193, 7, 0.9)' : 'rgba(41, 98, 255, 0.9)'),
            priceScaleId: '',
            lineWidth: isSelected ? 3 : 2,
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

function getStrategyParamsByPreset() {
    const preset = document.getElementById('strategy-select')?.value || 'default';
    if (preset === 'momentum') {
        return { mfi_period: 14, ma_fast: 9, ma_slow: 34, atr_sl_mult: 2.2, bb_period: 20, bb_std: 2.0 };
    }
    if (preset === 'mean_revert') {
        return { mfi_period: 21, ma_fast: 20, ma_slow: 55, atr_sl_mult: 1.4, bb_period: 26, bb_std: 2.4 };
    }
    if (preset === 'defensive') {
        return { mfi_period: 10, ma_fast: 12, ma_slow: 55, atr_sl_mult: 1.1, bb_period: 18, bb_std: 1.8 };
    }
    return {};
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
    const strategyParams = getStrategyParamsByPreset();
    Object.entries(strategyParams).forEach(([k, v]) => params.set(k, String(v)));

    try {
        const resp = await fetch(`/api/backtest?${params}`);
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
    const strategyParams = getStrategyParamsByPreset();
    Object.entries(strategyParams).forEach(([k, v]) => params.set(k, String(v)));

    try {
        const resp = await fetch(`/api/backtest/optimize?${params}`, { method: 'POST' });
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
document.getElementById('data-refresh-btn')?.addEventListener('click', () => {
    loadData(false, true);
    showStatus('已强制刷新数据。', 'success', 1200);
});
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
        const resp = await fetch('/api/symbols?include_extended=false', { signal: ac.signal });
        clearTimeout(timeoutId);
        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
        }
        const rawSymbols = await resp.json();
        allSymbols = Array.isArray(rawSymbols)
            ? rawSymbols.map(normalizeSymbolEntry).filter(Boolean)
            : [];
        if (allSymbols.length === 0) {
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
    const symbolList = symbols.map(normalizeSymbolEntry).filter(Boolean);
    
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
    if (recognizingBtn) recognizingBtn.classList.toggle('active', toolMode === 'recognize');
    if (drawBtn) drawBtn.classList.toggle('active', toolMode === 'draw');
    if (assistBtn) assistBtn.classList.toggle('active', toolMode === 'assist');
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

function selectTicker(symbol, skipLoad = false) {
    if (Array.isArray(allSymbols) && allSymbols.length > 0 && !allSymbols.includes(symbol)) {
        showStatus(`未找到交易对 ${symbol}`, 'error', 2200);
        return false;
    }
    currentSymbol = symbol;
    lastCandle = null;
    document.getElementById('current-ticker').textContent = formatTicker(symbol);
    document.getElementById('ticker-dropdown').classList.add('hidden');
    updateChartHeader();
    updateToolbarPrice();
    document.getElementById('ticker-search').value = '';
    document.title = `${formatTicker(symbol)} - Crypto TA`;
    stopLiveUpdates();
    if (!skipLoad) scheduleLoadData();
    // startLiveUpdates();
    return true;
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
        ? allSymbols.filter(s => normalizeSymbolEntry(s).includes(query))
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
        scheduleLoadData();
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

document.getElementById('layout-select').addEventListener('change', async (e) => {
    await applyLayoutPreset(e.target.value);
});

document.getElementById('data-priority-select').addEventListener('change', (e) => {
    dataPriorityMode = e.target.value || 'fast';
    patternResponseCache.clear();
    showStatus(`数据策略切换为 ${dataPriorityMode}`, 'success', 1200);
    scheduleLoadData();
});

document.getElementById('strategy-select').addEventListener('change', () => {
    showStatus(`策略模板切换为 ${document.getElementById('strategy-select').value}`, 'success', 1200);
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
    scheduleLoadData();
});

document.getElementById('replay-date').addEventListener('change', () => {
    if (document.getElementById('replay-toggle').checked) scheduleLoadData();
});

document.getElementById('replay-hour').addEventListener('change', () => {
    if (document.getElementById('replay-toggle').checked) scheduleLoadData();
});

// Timezone toggle buttons
document.getElementById('tz-group').addEventListener('click', (e) => {
    const btn = e.target.closest('.tz-btn');
    if (!btn) return;
    document.querySelectorAll('#tz-group .tz-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (document.getElementById('replay-toggle').checked) scheduleLoadData();
});

// ── Keyboard Shortcuts ──
document.addEventListener('keydown', (e) => {
    const activeTag = (document.activeElement?.tagName || '').toLowerCase();
    const isTyping = activeTag === 'input' || activeTag === 'textarea';

    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        undoLastDrawnLine();
        return;
    }

    if (e.key === 'Escape') {
        document.getElementById('ticker-dropdown').classList.add('hidden');
        cancelCurrentDrawing();
        return;
    }
    if (isTyping) return;

    if (e.key === 'Delete' && selectedUserLineIndex != null) {
        userDrawnLines.splice(selectedUserLineIndex, 1);
        selectedUserLineIndex = null;
        drawAllPatterns();
        showStatus('已删除选中的手绘线。', 'success', 1200);
        return;
    }

    if (e.key === '/') {
        e.preventDefault();
        document.getElementById('quick-command')?.focus();
        return;
    }

    if (e.shiftKey && e.key.toLowerCase() === 'r') {
        loadData(false, true);
        showStatus('已强制刷新数据。', 'success', 1100);
        return;
    }

    if (e.key.toLowerCase() === 'r') {
        setToolMode('recognize');
        showStatus('切换到 Recognizing 模式。', 'success', 1100);
    } else if (e.key.toLowerCase() === 'a') {
        setToolMode('assist');
        showStatus('切换到 Assist 模式。', 'success', 1100);
    } else if (e.key.toLowerCase() === 'd') {
        setToolMode('draw');
        showStatus('切换到 Draw 模式。', 'success', 1100);
    } else if (e.key.toLowerCase() === 't' && toolMode === 'draw') {
        drawingMode = 'trend';
        updateDrawingModeUI();
        showStatus('绘图工具：趋势线。', 'success', 1000);
    } else if (e.key.toLowerCase() === 'h' && toolMode === 'draw') {
        drawingMode = 'horizontal';
        updateDrawingModeUI();
        showStatus('绘图工具：水平线。', 'success', 1000);
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

function cancelCurrentDrawing() {
    pendingDrawPoint = null;
    removePreviewLine();
    if (chart) chart.applyOptions({ handleScroll: true });
}

function undoLastDrawnLine() {
    if (!userDrawnLines.length) {
        showStatus('没有可撤销的手绘线。', 'info', 1800);
        return;
    }
    userDrawnLines.pop();
    if (selectedUserLineIndex != null && selectedUserLineIndex >= userDrawnLines.length) {
        selectedUserLineIndex = null;
    }
    cancelCurrentDrawing();
    drawAllPatterns();
    showStatus('已撤销上一条手绘线。', 'success', 1200);
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

document.getElementById('draw-undo').addEventListener('click', () => {
    undoLastDrawnLine();
});

document.getElementById('draw-clear').addEventListener('click', () => {
    userDrawnLines = [];
    selectedUserLineIndex = null;
    cancelCurrentDrawing();
    drawingMode = null;
    updateDrawingModeUI();
    drawAllPatterns();
    showStatus('已清除全部手绘线。', 'success', 1400);
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
    if (contentEl) contentEl.textContent = 'Recognizing 模式已默认开启，正在加载形态…';
    const savedJournal = localStorage.getItem('tradeJournal');
    try {
        tradeJournal = savedJournal ? JSON.parse(savedJournal) : [];
        if (!Array.isArray(tradeJournal)) tradeJournal = [];
    } catch {
        tradeJournal = [];
    }
    renderJournal();
    updateChecklistProgress();
    computeRiskPosition();
    ['check-trend', 'check-level', 'check-risk'].forEach((id) => {
        document.getElementById(id)?.addEventListener('change', updateChecklistProgress);
    });
    ['risk-balance', 'risk-percent', 'risk-entry', 'risk-stop'].forEach((id) => {
        document.getElementById(id)?.addEventListener('input', computeRiskPosition);
    });
    document.getElementById('save-journal')?.addEventListener('click', saveCurrentTradeToJournal);
    document.getElementById('journal-prev')?.addEventListener('click', () => replayJournalAt(Math.max(0, activeJournalIndex - 1)));
    document.getElementById('journal-next')?.addEventListener('click', () => replayJournalAt(Math.min(tradeJournal.length - 1, activeJournalIndex + 1)));
    document.getElementById('journal-clear')?.addEventListener('click', () => {
        tradeJournal = [];
        activeJournalIndex = -1;
        persistJournal();
        renderJournal();
        showStatus('交易日志已清空。', 'success', 1200);
    });
    document.getElementById('ai-refresh')?.addEventListener('click', updateAICoachSummary);
    const quickInput = document.getElementById('quick-command');
    quickInput?.addEventListener('keydown', async (e) => {
        if (e.key === 'Enter') {
            await applyQuickCommand(quickInput.value);
            quickInput.value = '';
            quickInput.blur();
        }
    });
    loadSymbols();
    setToolMode('recognize').then(() => loadData());
    // startLiveUpdates(); // 关闭自动定时刷新，避免「一直刷新」；需要时可再开启
    document.title = `${formatTicker(currentSymbol)} - Crypto TA`;
});
