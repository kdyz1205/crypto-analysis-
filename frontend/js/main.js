// frontend/js/main.js — Phase 2 + 4 module entry point

import { initChart, loadCurrent, startLiveUpdates } from './workbench/chart.js';
import { initTicker } from './workbench/ticker.js';
import { initTimeframe } from './workbench/timeframe.js';
import { initDecisionRail } from './workbench/decision_rail.js';
import { initExecutionPanel, togglePanel as toggleExec } from './execution/panel.js';
import { initCommandPalette, openPalette } from './command_palette/palette.js';
import { initGlassbox } from './control/glassbox.js';
import { connectStream } from './services/stream.js';
import { $, on } from './util/dom.js';
import { subscribe } from './util/events.js';

async function boot() {
  console.log('[main] Trading OS v2 booting...');

  // Workbench
  initChart('chart-container');
  await initTicker('v2-symbol-select');
  initTimeframe('#v2-tf-group');
  await loadCurrent();
  startLiveUpdates(10000);

  initDecisionRail();

  // Execution Center
  initExecutionPanel();

  // Phase 4: Command palette
  initCommandPalette();

  // Phase 4: Glassbox (timeline) — consumes SSE events
  initGlassbox();

  // Phase 3: Connect SSE stream
  connectStream();

  // Wire header buttons
  on('#v2-exec-toggle', 'click', () => toggleExec());
  on('#v2-cmdk-btn', 'click', () => openPalette());
  on('#v2-combat-btn', 'click', () => {
    document.body.classList.toggle('combat-mode');
    window.dispatchEvent(new Event('resize'));
  });

  // F11 / Ctrl+. for combat mode
  document.addEventListener('keydown', (e) => {
    if (e.key === 'F11' || (e.ctrlKey && e.key === '.')) {
      e.preventDefault();
      document.body.classList.toggle('combat-mode');
      window.dispatchEvent(new Event('resize'));
    }
  });

  // Show a visual flash when we receive SSE events (debug)
  subscribe('connected', () => console.log('[stream] connection confirmed'));

  console.log('[main] Trading OS v2 ready');
}

document.addEventListener('DOMContentLoaded', boot);
