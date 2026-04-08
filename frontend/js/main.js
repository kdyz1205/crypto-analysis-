// frontend/js/main.js — non-blocking boot: UI shell first, data async

import { initChart, loadCurrent, startLiveUpdates, toggleMAOverlays } from './workbench/chart.js';
import { initTicker } from './workbench/ticker.js';
import { initTimeframe } from './workbench/timeframe.js';
import { initDecisionRail } from './workbench/decision_rail.js';
import { initExecutionPanel, togglePanel as toggleExec } from './execution/panel.js';
import { initCommandPalette, openPalette } from './command_palette/palette.js';
import { initGlassbox } from './control/glassbox.js';
import { initChatDock } from './control/chat.js';
import { initResearchDrawer } from './research/drawer.js';
import { connectStream } from './services/stream.js';
import { initBootStatus, markBoot } from './ui/boot_status.js';
import { $, on } from './util/dom.js';
import { subscribe } from './util/events.js';
import { setChatDock, setResearchDrawer, uiState } from './state/ui.js';

/**
 * Boot philosophy: NEVER block the UI shell on a data fetch.
 * Every init function is fire-and-forget. Each component shows its own
 * skeleton/loading state until its data arrives. Failures are isolated.
 */
function boot() {
  console.log('[main] Trading OS v2 booting (non-blocking)...');

  // Boot status indicator (render immediately — no dependencies)
  initBootStatus();

  // ── Synchronous UI shell (everything that doesn't need network) ──
  // These functions set up DOM containers, event listeners, and skeletons.
  // No awaits — they return instantly.
  try {
    initChart('chart-container');          // creates LightweightCharts shell
    markBoot('chart', 'pending', 'creating chart');
  } catch (err) { markBoot('chart', 'error', err.message); console.error('[boot] chart init failed:', err); }

  try {
    initTimeframe('#v2-tf-group');          // sync button wiring
  } catch (err) { console.error('[boot] timeframe init failed:', err); }

  try {
    initDecisionRail();                     // renders loading cards first
    markBoot('rail', 'pending', 'loading cards');
  } catch (err) { markBoot('rail', 'error', err.message); }

  try {
    initExecutionPanel();                   // builds panel shell, hidden
    markBoot('exec', 'ok', 'panel ready');
  } catch (err) { markBoot('exec', 'error', err.message); }

  try { initResearchDrawer(); } catch (err) { console.error('[boot] research drawer failed:', err); }
  try { initChatDock(); } catch (err) { console.error('[boot] chat dock failed:', err); }
  try { initCommandPalette(); } catch (err) { console.error('[boot] palette failed:', err); }
  try { initGlassbox(); } catch (err) { console.error('[boot] glassbox failed:', err); }

  // ── Wire header buttons (sync, no network) ──
  wireHeaderButtons();

  // ── Fire async data loads in parallel — non-blocking ──

  // 1. Load symbol list (fills the ticker dropdown)
  initTicker('v2-symbol-select')
    .then(() => markBoot('symbols', 'ok', 'loaded'))
    .catch((err) => { markBoot('symbols', 'error', err.message); console.error('[boot] ticker:', err); });

  // 2. Load initial chart data
  loadCurrent(true)
    .then(() => {
      markBoot('chart', 'ok', 'data loaded');
      markBoot('patterns', 'ok', 'loaded');
      startLiveUpdates(30000);
    })
    .catch((err) => { markBoot('chart', 'error', err.message); console.error('[boot] chart data:', err); });

  // 3. Connect SSE stream (long-lived, don't treat as loading)
  try {
    connectStream();
    markBoot('stream', 'ok', 'connected');
    subscribe('connected', () => markBoot('stream', 'ok', 'alive'));
  } catch (err) {
    markBoot('stream', 'error', err.message);
  }

  console.log('[main] Trading OS v2 shell mounted (data loading in background)');
}

function wireHeaderButtons() {
  on('#v2-exec-toggle', 'click', () => toggleExec());
  on('#v2-cmdk-btn', 'click', () => openPalette());
  on('#v2-research-btn', 'click', () => setResearchDrawer(!uiState.researchDrawerOpen));
  on('#v2-chat-btn', 'click', () => setChatDock(!uiState.chatDockOpen));
  on('#v2-ma-toggle', 'click', () => {
    const visible = toggleMAOverlays();
    $('#v2-ma-toggle')?.classList.toggle('active', visible);
  });

  // Combat mode toggle with visible exit button
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
    if (e.key === 'Escape' && document.body.classList.contains('combat-mode')) {
      const t = e.target;
      if (!t || (t.tagName !== 'INPUT' && t.tagName !== 'TEXTAREA')) {
        setCombatMode(false);
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', boot);
