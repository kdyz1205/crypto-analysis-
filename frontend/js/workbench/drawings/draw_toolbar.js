// frontend/js/workbench/drawings/draw_toolbar.js
//
// A floating toolbar that sits on top of the chart canvas. Contains the
// drawing tool buttons (support line, resistance line, + future tools).
// Replaces the old #manual-drawing-panel sidebar.
//
// The buttons directly drive draw_tool.startDrawing(toolName) — no sidebar
// state machine in between. Status feedback comes from subscribing to
// drawtool.mode events.

import { startDrawing, cancelDrawing, getDrawState } from './draw_tool.js';
import { subscribe } from '../../util/events.js';
import * as drawingsSvc from '../../services/drawings.js';
import { marketState } from '../../state/market.js';

let toolbar = null;
let statusEl = null;

export function initDrawToolbar(containerEl) {
  if (toolbar) return toolbar;
  toolbar = document.createElement('div');
  toolbar.className = 'draw-toolbar';
  toolbar.innerHTML = `
    <button class="dtb-btn dtb-support"   data-tool="trendline-support"   title="画支撑线 (S)">
      <span class="dtb-icon">╱</span><span>支撑</span>
    </button>
    <button class="dtb-btn dtb-resistance" data-tool="trendline-resistance" title="画阻力线 (R)">
      <span class="dtb-icon">╲</span><span>阻力</span>
    </button>
    <button class="dtb-btn dtb-cancel"    data-tool="cancel"              title="取消 (Esc)">✕</button>
    <button class="dtb-btn dtb-clear"     data-tool="clear"               title="清空当前 symbol/tf 的所有手画线">清空</button>
    <div class="dtb-status"></div>
  `;
  containerEl.appendChild(toolbar);
  statusEl = toolbar.querySelector('.dtb-status');

  toolbar.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-tool]');
    if (!btn) return;
    const tool = btn.dataset.tool;
    if (tool === 'cancel') {
      cancelDrawing('toolbar_cancel');
      return;
    }
    if (tool === 'clear') {
      if (!confirm(`清空 ${marketState.currentSymbol} ${marketState.currentInterval} 的所有手画线?`)) return;
      drawingsSvc
        .clearManualDrawings(marketState.currentSymbol, marketState.currentInterval)
        .then(() => {
          // trigger refresh via the existing controller
          import('./manual_trendline_controller.js').then((mod) => {
            mod.refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
          });
        })
        .catch((err) => alert(`清空失败: ${err?.message || err}`));
      return;
    }
    startDrawing(tool);
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // Only when NOT typing in an input
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable)) return;
    if (e.key === 's' || e.key === 'S') startDrawing('trendline-support');
    else if (e.key === 'r' || e.key === 'R') startDrawing('trendline-resistance');
  });

  subscribe('drawtool.mode', onModeChange);

  updateStatus('idle', null);
  return toolbar;
}

function onModeChange(detail) {
  const { tool, phase, anchor, reason } = detail || {};
  updateStatus(phase, tool, reason);

  // Toggle active class on the button matching the current tool
  toolbar.querySelectorAll('.dtb-btn[data-tool]').forEach((b) => {
    b.classList.toggle('is-active', b.dataset.tool === tool);
  });
}

function updateStatus(phase, tool, reason) {
  if (!statusEl) return;
  if (phase === 'idle') {
    statusEl.textContent = '按 S 画支撑 · R 画阻力 · Esc 取消';
    statusEl.className = 'dtb-status';
    return;
  }
  if (phase === 'picking_first') {
    statusEl.textContent = `点击图上任意位置放置第 1 个锚点`;
    statusEl.className = 'dtb-status dtb-status-active';
    return;
  }
  if (phase === 'picking_second') {
    statusEl.textContent = `移动鼠标预览,再点一次固定线`;
    statusEl.className = 'dtb-status dtb-status-active';
    return;
  }
}
