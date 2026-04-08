// frontend/js/main.js — Phase 2 module entry point for v2 UI

import { initChart, loadCurrent, startLiveUpdates } from './workbench/chart.js';
import { initTicker } from './workbench/ticker.js';
import { initTimeframe } from './workbench/timeframe.js';
import { initDecisionRail } from './workbench/decision_rail.js';
import { initExecutionPanel, togglePanel as toggleExec } from './execution/panel.js';
import { $, on } from './util/dom.js';

async function boot() {
  console.log('[main] Phase 2 v2 booting...');

  initChart('chart-container');
  await initTicker('v2-symbol-select');
  initTimeframe('#v2-tf-group');
  await loadCurrent();
  startLiveUpdates(10000);

  initDecisionRail();
  initExecutionPanel();

  on('#v2-exec-toggle', 'click', () => toggleExec());

  console.log('[main] Phase 2 v2 ready');
}

document.addEventListener('DOMContentLoaded', boot);
