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
import { resetChartViewport } from '../chart.js';
import * as drawingsSvc from '../../services/drawings.js';
import { marketState } from '../../state/market.js';
import { subscribe } from '../../util/events.js';
import {
  clearMultiSelected,
  setDrawingsError,
  setManualDrawings,
  setSelectedManualLine,
} from '../../state/drawings.js';
import { enterMeasureMode, exitMeasureMode, clearAllMeasurements } from './measure_tool.js';
import { enterPredictionMode, exitPredictionMode, clearAllPredictions } from './prediction_tool.js';
import { uiConfirm } from '../../util/ui_confirm.js';

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
    <button class="dtb-btn dtb-measure" data-tool="measure" title="测量 · 价格/日期范围 (M)">
      <span class="dtb-icon">📏</span><span>测量</span>
    </button>
    <button class="dtb-btn dtb-predict-long" data-tool="predict-long" title="多头仓位预测">
      <span class="dtb-icon">📈</span><span>多头</span>
    </button>
    <button class="dtb-btn dtb-predict-short" data-tool="predict-short" title="空头仓位预测">
      <span class="dtb-icon">📉</span><span>空头</span>
    </button>
    <button class="dtb-btn dtb-cancel" data-tool="cancel" title="取消 (Esc)">✕</button>
    <button class="dtb-btn dtb-clear"  data-tool="clear"  title="清空当前 symbol/tf 的所有线">清空</button>
    <button class="dtb-btn dtb-autofit" data-tool="autofit" title="自动 (调整数据适于屏幕) · R" aria-label="自动适应屏幕">
      <span class="dtb-icon">⇱⇲</span><span>自动</span>
    </button>
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
      // P0 2026-04-23: native confirm() replaced with uiConfirm so ESC
      // (which also exits draw-tool mode) can't interact with the
      // dismiss flow, and so the destructive action has a red-styled
      // explicit button rather than a grey OK.
      const sym = marketState.currentSymbol || '—';
      const tf = marketState.currentInterval || '—';
      uiConfirm({
        title: '清空所有手画线?',
        message: `${sym} ${tf} 上的所有手画线将被删除。这个操作无法撤销。`,
        confirmLabel: '清空',
        cancelLabel: '取消',
        destructive: true,
      }).then((ok) => {
        if (!ok) return;
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
      });
      return;
    }
    if (tool === 'trendline') {
      // Toggle: if the button is already armed (is-active), treat the click
      // as a cancel instead of re-arming. This kills the class of bugs where
      // the state machine thinks it's idle but the user still sees "draw
      // mode" active — pressing the button again always escapes cleanly.
      if (btn.classList.contains('is-active')) {
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        return;
      }
      startTrendlineTool();
      return;
    }
    if (tool === 'autofit') {
      resetChartViewport();
      // Brief visual pulse on the button so the user sees it fired
      btn.classList.add('is-pulse');
      setTimeout(() => btn.classList.remove('is-pulse'), 300);
      return;
    }
    if (tool === 'measure') {
      // Toggle: if already armed, exit. Otherwise enter rect mode.
      if (btn.classList.contains('is-active')) {
        exitMeasureMode();
      } else {
        // Exit any other armed tool first
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        enterMeasureMode('rect');
      }
      return;
    }
    if (tool === 'predict-long' || tool === 'predict-short') {
      const dir = tool === 'predict-long' ? 'long' : 'short';
      if (btn.classList.contains('is-active')) {
        exitPredictionMode();
      } else {
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
        enterPredictionMode(dir);
      }
      return;
    }
    startTrendlineTool();
  });

  // Subscribe to measure/predict mode events to toggle button active state
  subscribe('measure.mode', (evt) => {
    const btn = toolbar.querySelector('[data-tool="measure"]');
    btn?.classList.toggle('is-active', !!evt?.active);
    if (evt?.active) updateStatus('measuring', null);
  });
  subscribe('predict.mode', (evt) => {
    const lBtn = toolbar.querySelector('[data-tool="predict-long"]');
    const sBtn = toolbar.querySelector('[data-tool="predict-short"]');
    lBtn?.classList.toggle('is-active', !!evt?.active && evt?.direction === 'long');
    sBtn?.classList.toggle('is-active', !!evt?.active && evt?.direction === 'short');
    if (evt?.active) updateStatus('predicting', evt.direction);
  });
  subscribe('predict.phase', (evt) => {
    if (evt?.phase === 'awaiting_tp') {
      if (statusEl) {
        statusEl.textContent = '入场已定 → 再点一下图上选 TP (SL 自动放在入场反向 1%)';
        statusEl.className = 'dtb-status dtb-status-active';
      }
    }
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
    statusEl.textContent = '按 T 画线 · M 测量 · Esc 取消';
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
  if (phase === 'measuring') {
    statusEl.textContent = '点击+拖动选定范围 · Esc 退出';
    statusEl.className = 'dtb-status dtb-status-active';
    return;
  }
  if (phase === 'predicting') {
    const dir = tool === 'short' ? '空头' : '多头';
    statusEl.textContent = `${dir}预测: 点一下图上选择入场价`;
    statusEl.className = 'dtb-status dtb-status-active';
    return;
  }
}
