// frontend/js/main.js — Phase 2 + 4 module entry point

import { initChart, loadCurrent, startLiveUpdates, toggleMAOverlays } from './workbench/chart.js';
import { initTicker } from './workbench/ticker.js';
import { initTimeframe } from './workbench/timeframe.js';
import { initDecisionRail } from './workbench/decision_rail.js';
import { initExecutionPanel, togglePanel as toggleExec } from './execution/panel.js';
import { initCommandPalette, openPalette } from './command_palette/palette.js';
import { initGlassbox } from './control/glassbox.js';
import { initChatDock, openChatDock } from './control/chat.js';
import { initResearchDrawer, openResearchDrawer } from './research/drawer.js';
import { connectStream } from './services/stream.js';
import { $, on } from './util/dom.js';
import { subscribe } from './util/events.js';
import { setChatDock, setResearchDrawer, uiState } from './state/ui.js';

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

  // Research Drawer (backtest, MA ribbon)
  initResearchDrawer();

  // Chat Dock (AI assistant with memory + scheduling)
  initChatDock();

  // Phase 4: Command palette
  initCommandPalette();

  // Phase 4: Glassbox (timeline) — consumes SSE events
  initGlassbox();

  // Phase 3: Connect SSE stream
  connectStream();

  // Wire header buttons
  on('#v2-exec-toggle', 'click', () => toggleExec());
  on('#v2-cmdk-btn', 'click', () => openPalette());
  on('#v2-research-btn', 'click', () => setResearchDrawer(!uiState.researchDrawerOpen));
  on('#v2-chat-btn', 'click', () => setChatDock(!uiState.chatDockOpen));
  on('#v2-ma-toggle', 'click', () => {
    const visible = toggleMAOverlays();
    $('#v2-ma-toggle')?.classList.toggle('active', visible);
  });

  // Combat mode toggle with guaranteed-visible exit button
  const setCombatMode = (enabled) => {
    document.body.classList.toggle('combat-mode', enabled);
    window.dispatchEvent(new Event('resize'));
    const exitBtn = $('#v2-combat-exit');
    if (exitBtn) exitBtn.classList.toggle('hidden', !enabled);
  };

  on('#v2-combat-btn', 'click', () => {
    setCombatMode(!document.body.classList.contains('combat-mode'));
  });

  // Floating exit button (visible only in combat mode)
  if (!$('#v2-combat-exit')) {
    const exitBtn = document.createElement('button');
    exitBtn.id = 'v2-combat-exit';
    exitBtn.className = 'combat-exit-btn hidden';
    exitBtn.innerHTML = '✕ Exit Combat (Esc)';
    exitBtn.title = 'Exit combat mode';
    document.body.appendChild(exitBtn);
    exitBtn.addEventListener('click', () => setCombatMode(false));
  }

  // Keyboard: F11 / Ctrl+. toggle, Esc exit
  document.addEventListener('keydown', (e) => {
    if (e.key === 'F11' || (e.ctrlKey && e.key === '.')) {
      e.preventDefault();
      setCombatMode(!document.body.classList.contains('combat-mode'));
    }
    // Esc always exits combat mode (if not in an input)
    if (e.key === 'Escape' && document.body.classList.contains('combat-mode')) {
      const t = e.target;
      if (!t || (t.tagName !== 'INPUT' && t.tagName !== 'TEXTAREA')) {
        setCombatMode(false);
      }
    }
  });

  // Show a visual flash when we receive SSE events (debug)
  subscribe('connected', () => console.log('[stream] connection confirmed'));

  console.log('[main] Trading OS v2 ready');
}

document.addEventListener('DOMContentLoaded', boot);
