// frontend/js/main.js — clean boot: UI shell first, data async

import { initChart, loadCurrent, startLiveUpdates, toggleMAOverlays } from './workbench/chart.js';
import { initTicker } from './workbench/ticker.js';
import { initTimeframe } from './workbench/timeframe.js';
import { initDecisionRail, refreshDecisionRail } from './workbench/decision_rail.js';
import { initExecutionPanel, togglePanel as toggleExec } from './execution/panel.js';
import { connectStream } from './services/stream.js';
import { initBootStatus, markBoot } from './ui/boot_status.js';
import { $, on } from './util/dom.js';
import { subscribe } from './util/events.js';
import { setScale } from './state/market.js';

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
  };

  initBootStatus();

  try { initChart('chart-container'); markBoot('chart', 'pending', 'loading'); } catch (err) { markBoot('chart', 'error', err.message); }
  try { initTimeframe('#v2-tf-group'); } catch {}
  try { initDecisionRail(); } catch {}
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

function wireHeaderButtons() {
  on('#v2-exec-toggle', 'click', () => toggleExec());
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

document.addEventListener('DOMContentLoaded', boot);
