// frontend/js/main.js — clean boot: UI shell first, data async

import { initChart, loadCurrent, startLiveUpdates, toggleMAOverlays, getChart, getCandleSeries } from './workbench/chart.js';
import { initManualTrendlineController, refreshManualDrawings } from './workbench/drawings/manual_trendline_controller.js';
import { initDrawTool } from './workbench/drawings/draw_tool.js';
import { initDrawToolbar } from './workbench/drawings/draw_toolbar.js';
import { initTicker } from './workbench/ticker.js';
import { initTimeframe } from './workbench/timeframe.js';
import { initDecisionRail, refreshDecisionRail } from './workbench/decision_rail.js';
import { initConditionalPanel } from './workbench/conditional_panel.js';
import { initExecutionPanel, togglePanel as toggleExec } from './execution/panel.js';
import { connectStream } from './services/stream.js';
import { initBootStatus, markBoot } from './ui/boot_status.js';
import { $, on, esc } from './util/dom.js';
import { subscribe } from './util/events.js';
import { setScale, marketState } from './state/market.js';

function boot() {
  console.log('[main] Trading OS booting...');
  let liveUpdatesStarted = false;
  let railRefreshed = false;
  let streamConnected = false;

  const afterChartLoad = () => {
    markBoot('chart', 'ok', 'loaded');
    if (!liveUpdatesStarted) { startLiveUpdates(30000); liveUpdatesStarted = true; }
    if (!railRefreshed) {
      railRefreshed = true;
      setTimeout(() => refreshDecisionRail().then(() => markBoot('rail', 'ok', 'loaded')).catch(() => {}), 200);
    }
    if (!streamConnected) {
      streamConnected = true;
      setTimeout(() => { try { connectStream(); } catch {} }, 300);
    }
    try { void refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval); } catch {}
  };

  initBootStatus();

  try { initChart('chart-container'); markBoot('chart', 'pending', 'loading'); } catch (err) { markBoot('chart', 'error', err.message); }
  try {
    const chartObj = getChart();
    const containerEl = document.getElementById('chart-container');
    if (chartObj && containerEl) initManualTrendlineController(chartObj, containerEl);
  } catch (err) { console.warn('[boot] manual drawing:', err); }
  // Drawing tool (TradingView-style): rubber-band preview + floating toolbar.
  // Must come after chart init so candleSeries is available.
  try {
    const chartObj = getChart();
    const cs = getCandleSeries();
    const containerEl = document.getElementById('chart-container');
    if (chartObj && cs && containerEl) {
      initDrawTool(chartObj, cs, containerEl);
      initDrawToolbar(containerEl);
    }
  } catch (err) { console.warn('[boot] draw tool:', err); }
  try { initTimeframe('#v2-tf-group'); } catch {}
  try { initDecisionRail(); } catch {}
  try {
    const condHost = document.getElementById('v2-cond-rail');
    if (condHost) initConditionalPanel(condHost);
  } catch (err) { console.warn('[boot] conditional panel:', err); }
  try { initExecutionPanel(); } catch {}

  wireHeaderButtons();
  subscribe('chart.load.succeeded', afterChartLoad);

  initTicker('v2-symbol-select')
    .then(() => markBoot('symbols', 'ok', 'loaded'))
    .catch((err) => console.error('[boot] ticker:', err));

  loadCurrent(true).catch((err) => {
    markBoot('chart', 'error', err.message);
    try { connectStream(); streamConnected = true; } catch {}
  });

  console.log('[main] Trading OS ready');
}

let _previousView = 'market';

function wireHeaderButtons() {
  // View navigation
  const nav = $('#v2-nav');
  if (nav) {
    nav.addEventListener('click', async (e) => {
      const btn = e.target.closest('.v2-nav-btn');
      if (!btn) return;
      const view = btn.dataset.view;
      // Tear down market-only timers when leaving the market view
      if (_previousView === 'market' && view !== 'market') {
        try {
          const { stopDecisionRail } = await import('./workbench/decision_rail.js');
          stopDecisionRail();
        } catch {}
      }
      _previousView = view;
      // Switch active nav button
      nav.querySelectorAll('.v2-nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // Switch active view
      document.querySelectorAll('.v2-view').forEach(v => v.classList.remove('active'));
      const target = document.querySelector(`.v2-view[data-view="${view}"]`);
      if (target) target.classList.add('active');
      // Load view content
      loadView(view);
    });
  }

  // Cross-view handoff: open Factory with pre-filled pattern from chart
  window.__openFactoryWithPattern = (payload) => {
    // Stash payload for the factory loader to pick up
    window.__factoryPrefill = payload;
    // Switch to factory tab
    nav?.querySelectorAll('.v2-nav-btn').forEach(b => b.classList.remove('active'));
    nav?.querySelector('[data-view="factory"]')?.classList.add('active');
    document.querySelectorAll('.v2-view').forEach(v => v.classList.remove('active'));
    document.querySelector('.v2-view[data-view="factory"]')?.classList.add('active');
    loadView('factory');
  };

  on('#v2-ma-toggle', 'click', () => {
    const visible = toggleMAOverlays();
    $('#v2-ma-toggle')?.classList.toggle('active', visible);
  });

  const scaleToggle = $('#v2-scale-toggle');
  if (scaleToggle) {
    scaleToggle.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-scale]');
      if (!btn) return;
      setScale(btn.dataset.scale);
      scaleToggle.querySelectorAll('.v2-scale-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  }

  const setCombatMode = (on) => {
    document.body.classList.toggle('combat-mode', on);
    window.dispatchEvent(new Event('resize'));
  };

  on('#v2-combat-btn', 'click', () => setCombatMode(!document.body.classList.contains('combat-mode')));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'F11' || (e.ctrlKey && e.key === '.')) {
      e.preventDefault();
      setCombatMode(!document.body.classList.contains('combat-mode'));
    }
    if (e.key === 'Escape' && document.body.classList.contains('combat-mode')) {
      if (e.target?.tagName !== 'INPUT' && e.target?.tagName !== 'TEXTAREA') setCombatMode(false);
    }
  });
}

// ── View Loaders ────────────────────────────────────────────────────────

import { loadDashboard, loadFactory, loadLeaderboard, loadFactors, loadLive, loadMonitor } from './views.js';

const _viewLoaders = {
  dashboard: loadDashboard, factory: loadFactory, leaderboard: loadLeaderboard,
  factors: loadFactors, live: loadLive, monitor: loadMonitor,
};
const _viewLoaded = new Set(); // track which views have been loaded at least once

async function loadView(view) {
  if (view === 'market') return;
  const el = $(`#view-${view}`);
  if (!el) return;
  const loader = _viewLoaders[view];
  if (!loader) return;

  // First load: show skeleton. Subsequent loads: keep old content, refresh in background.
  if (!_viewLoaded.has(view)) {
    el.innerHTML = skeletonFor(view);
  } else {
    // Add subtle loading indicator without wiping content
    el.style.opacity = '0.7';
  }

  try {
    await loader(el);
    _viewLoaded.add(view);
  } catch (e) {
    // Only show error if we have nothing to show
    if (!_viewLoaded.has(view)) {
      el.innerHTML = `<p style="color:var(--v2-red);padding:20px">加载失败: ${esc(e?.message || String(e))}</p>`;
    }
  } finally {
    el.style.opacity = '';
  }
}

function skeletonFor(view) {
  const sk = `<div class="skel-line" style="width:60%"></div>`;
  const skw = `<div class="skel-line" style="width:40%"></div>`;
  const card = `<div class="skel-card"></div>`;
  const row = `<div class="skel-row"></div>`;
  if (view === 'dashboard') return `
    <div style="padding:20px">
      ${sk}<div style="height:12px"></div>
      <div class="dash-cards">${card}${card}${card}${card}${card}</div>
      <div class="dash-grid"><div>${row}${row}${row}</div><div>${row}${row}${row}</div></div>
    </div>`;
  if (view === 'leaderboard' || view === 'factors') return `
    <div style="padding:20px">
      ${sk}<div style="height:12px"></div>
      <div class="view-stats">${card}${card}${card}${card}</div>
      ${row}${row}${row}${row}${row}${row}
    </div>`;
  return `<div style="padding:20px">${sk}<div style="height:12px"></div>${row}${row}${row}${row}</div>`;
}


document.addEventListener('DOMContentLoaded', boot);
