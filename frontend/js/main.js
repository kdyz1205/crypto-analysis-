// frontend/js/main.js - non-blocking boot: UI shell first, data async

import { initChart, loadCurrent, startLiveUpdates, toggleMAOverlays } from './workbench/chart.js';
import { initTicker } from './workbench/ticker.js';
import { initTimeframe } from './workbench/timeframe.js';
import { initDecisionRail, refreshDecisionRail } from './workbench/decision_rail.js';
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
import { setScale } from './state/market.js';

/**
 * Boot philosophy: NEVER block the UI shell on a data fetch.
 * Every init function is fire-and-forget. Each component shows its own
 * skeleton/loading state until its data arrives. Failures are isolated.
 */
function boot() {
  console.log('[main] Trading OS v2 booting (non-blocking)...');
  let liveUpdatesStarted = false;
  let railRefreshScheduled = false;
  let streamConnectScheduled = false;
  let streamConnectedSubscribed = false;

  const ensureStreamConnectedSubscription = () => {
    if (streamConnectedSubscribed) return;
    subscribe('connected', () => markBoot('stream', 'ok', 'alive'));
    streamConnectedSubscribed = true;
  };

  const ensureChartBootFollowups = () => {
    markBoot('chart', 'ok', 'data loaded');

    if (!liveUpdatesStarted) {
      startLiveUpdates(30000);
      liveUpdatesStarted = true;
    }

    if (!railRefreshScheduled) {
      railRefreshScheduled = true;
      setTimeout(() => {
        refreshDecisionRail()
          .then(() => markBoot('rail', 'ok', 'loaded'))
          .catch((err) => {
            markBoot('rail', 'error', err.message);
            console.error('[boot] decision rail:', err);
          });
      }, 200);
    }

    if (!streamConnectScheduled) {
      streamConnectScheduled = true;
      setTimeout(() => {
        try {
          ensureStreamConnectedSubscription();
          connectStream();
          markBoot('stream', 'pending', 'connecting');
        } catch (err) {
          markBoot('stream', 'error', err.message);
        }
      }, 300);
    }
  };

  // Boot status indicator (render immediately - no dependencies)
  initBootStatus();

  // Synchronous UI shell (everything that does not need network).
  try {
    initChart('chart-container');
    markBoot('chart', 'pending', 'creating chart');
  } catch (err) {
    markBoot('chart', 'error', err.message);
    console.error('[boot] chart init failed:', err);
  }

  try {
    initTimeframe('#v2-tf-group');
  } catch (err) {
    console.error('[boot] timeframe init failed:', err);
  }

  try {
    initDecisionRail();
    markBoot('rail', 'pending', 'loading cards');
  } catch (err) {
    markBoot('rail', 'error', err.message);
  }

  try {
    initExecutionPanel();
    markBoot('exec', 'ok', 'panel ready');
  } catch (err) {
    markBoot('exec', 'error', err.message);
  }

  try { initResearchDrawer(); } catch (err) { console.error('[boot] research drawer failed:', err); }
  try { initChatDock(); } catch (err) { console.error('[boot] chat dock failed:', err); }
  try { initCommandPalette(); } catch (err) { console.error('[boot] palette failed:', err); }
  try { initGlassbox(); } catch (err) { console.error('[boot] glassbox failed:', err); }

  wireHeaderButtons();
  subscribe('chart.load.succeeded', ensureChartBootFollowups);

  initTicker('v2-symbol-select')
    .then(() => markBoot('symbols', 'ok', 'loaded'))
    .catch((err) => {
      markBoot('symbols', 'error', err.message);
      console.error('[boot] ticker:', err);
    });

  // Load initial chart data first. Strategy overlays and execution state
  // are more useful than opening the long-lived SSE stream immediately.
  loadCurrent(true).catch((err) => {
    markBoot('chart', 'error', err.message);
    markBoot('patterns', 'error', 'chart load failed');
    console.error('[boot] chart data:', err);
    try {
      ensureStreamConnectedSubscription();
      connectStream();
      markBoot('stream', 'pending', 'connecting');
      streamConnectScheduled = true;
    } catch (streamErr) {
      markBoot('stream', 'error', streamErr.message);
    }
  });

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

  // Log/Linear scale toggle
  const scaleToggle = $('#v2-scale-toggle');
  if (scaleToggle) {
    scaleToggle.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-scale]');
      if (!btn) return;
      const scale = btn.dataset.scale;
      setScale(scale);
      scaleToggle.querySelectorAll('.v2-scale-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  }

  const setCombatMode = (enabled) => {
    document.body.classList.toggle('combat-mode', enabled);
    window.dispatchEvent(new Event('resize'));
    const exitBtn = $('#v2-combat-exit');
    if (exitBtn) exitBtn.classList.toggle('hidden', !enabled);
  };

  on('#v2-combat-btn', 'click', () => {
    setCombatMode(!document.body.classList.contains('combat-mode'));
  });

  if (!$('#v2-combat-exit')) {
    const exitBtn = document.createElement('button');
    exitBtn.id = 'v2-combat-exit';
    exitBtn.className = 'combat-exit-btn hidden';
    exitBtn.innerHTML = '&#x2715; Exit Combat (Esc)';
    exitBtn.title = 'Exit combat mode';
    document.body.appendChild(exitBtn);
    exitBtn.addEventListener('click', () => setCombatMode(false));
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'F11' || (e.ctrlKey && e.key === '.')) {
      e.preventDefault();
      setCombatMode(!document.body.classList.contains('combat-mode'));
    }
    if (e.key === 'Escape' && document.body.classList.contains('combat-mode')) {
      const target = e.target;
      if (!target || (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA')) {
        setCombatMode(false);
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', boot);
