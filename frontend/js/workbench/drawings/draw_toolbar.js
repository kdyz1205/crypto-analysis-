// frontend/js/workbench/drawings/draw_toolbar.js
//
// A floating toolbar that sits on top of the chart canvas. Contains the
// drawing tool buttons (support line, resistance line, + future tools).
// Replaces the old #manual-drawing-panel sidebar.
//
// The buttons directly drive draw_tool.startDrawing(toolName) — no sidebar
// state machine in between. Status feedback comes from subscribing to
// drawtool.mode events.

import { startTrendlineTool } from './chart_drawing.js';
import * as drawingsSvc from '../../services/drawings.js';
import { marketState } from '../../state/market.js';
import { subscribe } from '../../util/events.js';
import {
  clearMultiSelected,
  setDrawingsError,
  setManualDrawings,
  setSelectedManualLine,
} from '../../state/drawings.js';

let toolbar = null;
let statusEl = null;
let trendlineBtn = null;

export function initDrawToolbar(containerEl) {
  if (toolbar) return toolbar;
  toolbar = document.createElement('div');
  toolbar.className = 'draw-toolbar';
  toolbar.innerHTML = `
    <button class="dtb-btn dtb-trendline" data-tool="trendline" title="画线 (T)">
      <span class="dtb-icon">╱</span><span>画线</span>
    </button>
    <button class="dtb-btn dtb-cancel" data-tool="cancel" title="取消 (Esc)">✕</button>
    <button class="dtb-btn dtb-clear"  data-tool="clear"  title="清空当前 symbol/tf 的所有线">清空</button>
    <div class="dtb-status"></div>
  `;
  containerEl.appendChild(toolbar);
  statusEl = toolbar.querySelector('.dtb-status');
  trendlineBtn = toolbar.querySelector('[data-tool="trendline"]');

  toolbar.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-tool]');
    if (!btn) return;
    const tool = btn.dataset.tool;
    if (tool === 'cancel') {
      // chart_drawing handles Esc itself via document keydown
      const ev = new KeyboardEvent('keydown', { key: 'Escape' });
      document.dispatchEvent(ev);
      return;
    }
    if (tool === 'clear') {
      if (!confirm(`清空 ${marketState.currentSymbol} ${marketState.currentInterval} 的所有手画线?`)) return;
      drawingsSvc
        .clearManualDrawings(marketState.currentSymbol, marketState.currentInterval)
        .then(() => {
          setSelectedManualLine(null);
          clearMultiSelected();
          setManualDrawings([]);
          setDrawingsError(null);
          import('./manual_trendline_controller.js').then((mod) => {
            mod.refreshManualDrawings(marketState.currentSymbol, marketState.currentInterval);
          });
        })
        .catch((err) => alert(`清空失败: ${err?.message || err}`));  // SAFE: alert() renders text, not HTML
      return;
    }
    startTrendlineTool();
  });

  // (T-shortcut also handled inside chart_drawing.js itself)
  subscribe('drawtool.mode', (mode) => {
    const phase = mode?.phase || 'idle';
    updateStatus(phase, mode?.state || null, mode?.reason || null);
    trendlineBtn?.classList.toggle('is-active', phase === 'picking_first' || phase === 'picking_second');
  });

  updateStatus('idle', null);
  return toolbar;
}

function updateStatus(phase, tool, reason) {
  if (!statusEl) return;
  if (phase === 'idle') {
    statusEl.textContent = '按 T 画线 · Esc 取消';
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
