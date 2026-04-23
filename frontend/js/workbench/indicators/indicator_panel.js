// frontend/js/workbench/indicators/indicator_panel.js
//
// Floating dropdown panel shown when the user clicks the "指标" button.
// Lists the current indicators with visibility / delete / ⚙ config,
// plus an "+ 添加指标" dropdown that uses INDICATOR_CATALOG.
//
// 2026-04-23: ⚙ gear per row opens an inline config editor. Currently
// wired for MA Ribbon (lines: period, type SMA/EMA, color, width) and
// simple period-based indicators (RSI, ATR, Volume MA, BB). Config
// changes persist to localStorage and trigger a chart re-render via
// the onChange callback indicator_panel was opened with.

import {
  getIndicators, setVisible, removeIndicator, addIndicator,
  updateConfig, renameIndicator,
  subscribe, INDICATOR_CATALOG,
} from './indicator_controller.js';
import { RIBBON_PRESETS, DEFAULT_RIBBON_LINES } from '../ma_overlay.js';
import { esc } from '../../util/dom.js';

let _panel = null;
let _cleanup = null;
// indicator id -> true when its config row is expanded
const _expanded = new Set();

export function toggleIndicatorPanel(anchorEl, onChange) {
  if (_panel) {
    closePanel();
    return;
  }
  openPanel(anchorEl, onChange);
}

function openPanel(anchorEl, onChange) {
  closePanel();
  _panel = document.createElement('div');
  _panel.className = 'ind-panel';
  _panel.innerHTML = render();
  document.body.appendChild(_panel);
  injectStyles();
  positionPanel(anchorEl);

  const unsub = subscribe(() => {
    if (_panel) _panel.innerHTML = render();
    wire(onChange);
    if (typeof onChange === 'function') onChange();
  });

  const onDocClick = (ev) => {
    if (!_panel) return;
    if (_panel.contains(ev.target)) return;
    if (anchorEl && anchorEl.contains(ev.target)) return;
    closePanel();
  };
  const onKey = (ev) => {
    if (ev.key === 'Escape') closePanel();
  };
  setTimeout(() => {
    document.addEventListener('mousedown', onDocClick);
  }, 0);
  document.addEventListener('keydown', onKey);

  _cleanup = () => {
    unsub();
    document.removeEventListener('mousedown', onDocClick);
    document.removeEventListener('keydown', onKey);
  };

  wire(onChange);
}

function closePanel() {
  if (_cleanup) { try { _cleanup(); } catch {} _cleanup = null; }
  if (_panel && _panel.parentNode) _panel.parentNode.removeChild(_panel);
  _panel = null;
}

function positionPanel(anchorEl) {
  if (!_panel) return;
  const r = anchorEl?.getBoundingClientRect?.();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  _panel.style.position = 'fixed';
  if (r) {
    const width = 340; // wider to fit config editors
    _panel.style.width = `${width}px`;
    let left = r.right - width;
    if (left < 8) left = 8;
    _panel.style.left = `${left}px`;
    _panel.style.top = `${r.bottom + 4}px`;
  } else {
    _panel.style.width = '340px';
    _panel.style.right = '16px';
    _panel.style.top = '60px';
  }
  setTimeout(() => {
    if (!_panel) return;
    const rect = _panel.getBoundingClientRect();
    if (rect.bottom > vh - 8) {
      _panel.style.top = `${Math.max(8, vh - rect.height - 8)}px`;
    }
    if (rect.right > vw - 8) {
      _panel.style.left = `${vw - rect.width - 8}px`;
    }
  }, 0);
}

function render() {
  const list = getIndicators();
  const rows = list.map((ind) => {
    const eye = ind.visible ? '👁' : '—';
    const eyeColor = ind.visible ? '#38bdf8' : '#5a6577';
    const hasEditor = _hasEditor(ind.type);
    const open = _expanded.has(ind.id);
    const gearTitle = hasEditor ? (open ? '收起设置' : '打开设置') : '此指标暂无可配置项';
    return `
      <div class="ind-row" data-id="${esc(ind.id)}">
        <div class="ind-row-head">
          <button class="ind-eye" data-action="toggle" title="${ind.visible ? '隐藏' : '显示'}" style="color:${eyeColor}">${eye}</button>
          <span class="ind-name" data-action="rename" title="点击重命名">${esc(ind.name)}</span>
          <button class="ind-gear ${hasEditor ? '' : 'disabled'}" data-action="gear"
                  title="${esc(gearTitle)}" ${hasEditor ? '' : 'disabled'}>⚙</button>
          <button class="ind-del" data-action="remove" title="删除">✕</button>
        </div>
        ${hasEditor && open ? `<div class="ind-config">${renderEditor(ind)}</div>` : ''}
      </div>
    `;
  }).join('') || '<div class="ind-empty">还没添加指标</div>';

  const catalogHtml = INDICATOR_CATALOG.map((cat) => `
    <div class="ind-cat">
      <div class="ind-cat-head">${esc(cat.category)}</div>
      ${cat.items.map((it) => `
        <button class="ind-add-item" data-type="${esc(it.type)}" data-name="${esc(it.name)}">
          + ${esc(it.name)}
        </button>
      `).join('')}
    </div>
  `).join('');

  return `
    <div class="ind-head">
      <span>指标</span>
      <button class="ind-close" data-action="close" title="关闭">✕</button>
    </div>
    <div class="ind-list">${rows}</div>
    <div class="ind-footer">
      <button class="ind-add-btn" data-action="toggle-catalog">+ 添加指标</button>
      <div class="ind-catalog" data-catalog hidden>${catalogHtml}</div>
    </div>
  `;
}

function _hasEditor(type) {
  return ['ma_ribbon', 'bb', 'rsi', 'atr', 'volume_ma', 'macd'].includes(type);
}

// ─── Editor renderers per indicator type ─────────────────────────

function renderEditor(ind) {
  if (ind.type === 'ma_ribbon') return renderMARibbonEditor(ind);
  if (ind.type === 'bb') return renderBBEditor(ind);
  if (ind.type === 'macd') return renderMACDEditor(ind);
  if (ind.type === 'rsi' || ind.type === 'atr' || ind.type === 'volume_ma') {
    return renderPeriodEditor(ind);
  }
  return '';
}

function renderMARibbonEditor(ind) {
  const lines = Array.isArray(ind.config?.lines) && ind.config.lines.length > 0
    ? ind.config.lines
    : DEFAULT_RIBBON_LINES;

  const presetOptions = RIBBON_PRESETS.map(
    (p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`
  ).join('');

  const lineRows = lines.map((ln, i) => `
    <div class="cfg-line-row" data-line-idx="${i}">
      <span class="cfg-line-num">${i + 1}</span>
      <input type="number" class="cfg-period" data-f="period"
             value="${Number(ln.period) || 0}" min="1" max="500" step="1"
             title="周期" />
      <select class="cfg-type" data-f="type" title="类型">
        <option value="sma" ${ln.type === 'sma' ? 'selected' : ''}>SMA</option>
        <option value="ema" ${ln.type === 'ema' ? 'selected' : ''}>EMA</option>
      </select>
      <input type="color" class="cfg-color" data-f="color"
             value="${esc(ln.color || '#ffeb3b')}" title="颜色" />
      <select class="cfg-width" data-f="width" title="线宽">
        <option value="1" ${Number(ln.width) === 1 ? 'selected' : ''}>1</option>
        <option value="2" ${Number(ln.width) === 2 ? 'selected' : ''}>2</option>
        <option value="3" ${Number(ln.width) === 3 ? 'selected' : ''}>3</option>
        <option value="4" ${Number(ln.width) === 4 ? 'selected' : ''}>4</option>
      </select>
      <button class="cfg-line-del" data-action="line-del" title="删除此线"
              ${lines.length <= 1 ? 'disabled' : ''}>✕</button>
    </div>
  `).join('');

  return `
    <div class="cfg-block">
      <div class="cfg-row">
        <label class="cfg-label">预设</label>
        <select class="cfg-preset" data-action="preset">
          <option value="">— 选预设自动填入 —</option>
          ${presetOptions}
        </select>
      </div>
      <div class="cfg-lines">
        <div class="cfg-lines-head">
          <span style="width:18px"></span>
          <span>周期</span>
          <span>类型</span>
          <span>颜色</span>
          <span>宽</span>
          <span></span>
        </div>
        ${lineRows}
      </div>
      <div class="cfg-actions">
        <button class="cfg-btn" data-action="line-add">+ 加一条线</button>
        <button class="cfg-btn" data-action="reset-default">重置默认</button>
      </div>
      <div class="cfg-hint">
        记忆参考: 4h 冠军 R=[8,21,55,89] · 真正最佳 R=[5,13,34,89]
      </div>
    </div>
  `;
}

function renderBBEditor(ind) {
  const period = Number(ind.config?.period || 20);
  const stddev = Number(ind.config?.stddev || 2);
  return `
    <div class="cfg-block">
      <div class="cfg-row">
        <label class="cfg-label">周期</label>
        <input type="number" class="cfg-num" data-f="period" value="${period}" min="2" max="200" step="1" />
      </div>
      <div class="cfg-row">
        <label class="cfg-label">标准差倍数</label>
        <input type="number" class="cfg-num" data-f="stddev" value="${stddev}" min="0.5" max="5" step="0.1" />
      </div>
    </div>
  `;
}

function renderMACDEditor(ind) {
  const fast = Number(ind.config?.fast || 12);
  const slow = Number(ind.config?.slow || 26);
  const signal = Number(ind.config?.signal || 9);
  return `
    <div class="cfg-block">
      <div class="cfg-row">
        <label class="cfg-label">Fast</label>
        <input type="number" class="cfg-num" data-f="fast" value="${fast}" min="2" max="100" step="1" />
      </div>
      <div class="cfg-row">
        <label class="cfg-label">Slow</label>
        <input type="number" class="cfg-num" data-f="slow" value="${slow}" min="3" max="200" step="1" />
      </div>
      <div class="cfg-row">
        <label class="cfg-label">Signal</label>
        <input type="number" class="cfg-num" data-f="signal" value="${signal}" min="1" max="100" step="1" />
      </div>
    </div>
  `;
}

function renderPeriodEditor(ind) {
  const period = Number(ind.config?.period || 14);
  return `
    <div class="cfg-block">
      <div class="cfg-row">
        <label class="cfg-label">周期</label>
        <input type="number" class="cfg-num" data-f="period" value="${period}" min="2" max="500" step="1" />
      </div>
    </div>
  `;
}

// ─── Wiring ──────────────────────────────────────────────────────

function wire(onChange) {
  if (!_panel) return;

  // Row-level actions
  _panel.querySelectorAll('.ind-row').forEach((row) => {
    const id = row.dataset.id;
    row.querySelector('[data-action="toggle"]')?.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const inds = getIndicators();
      const cur = inds.find((x) => x.id === id);
      if (cur) setVisible(id, !cur.visible);
    });
    row.querySelector('[data-action="remove"]')?.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (!confirm('确认删除此指标?')) return;
      _expanded.delete(id);
      removeIndicator(id);
    });
    row.querySelector('[data-action="gear"]')?.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (_expanded.has(id)) _expanded.delete(id); else _expanded.add(id);
      if (_panel) _panel.innerHTML = render();
      wire(onChange);
    });
    row.querySelector('[data-action="rename"]')?.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const cur = getIndicators().find((x) => x.id === id);
      if (!cur) return;
      const next = prompt('重命名', cur.name);
      if (next && next.trim() && next !== cur.name) renameIndicator(id, next.trim());
    });

    // Editor wiring (only if expanded)
    const cfgEl = row.querySelector('.ind-config');
    if (!cfgEl) return;
    const ind = getIndicators().find((x) => x.id === id);
    if (!ind) return;

    if (ind.type === 'ma_ribbon') wireMARibbon(cfgEl, id, ind, onChange);
    else wireSimpleEditor(cfgEl, id, ind, onChange);
  });

  _panel.querySelector('[data-action="close"]')?.addEventListener('click', (ev) => {
    ev.stopPropagation();
    closePanel();
  });
  const togBtn = _panel.querySelector('[data-action="toggle-catalog"]');
  const catEl = _panel.querySelector('[data-catalog]');
  togBtn?.addEventListener('click', (ev) => {
    ev.stopPropagation();
    if (catEl) catEl.hidden = !catEl.hidden;
  });
  _panel.querySelectorAll('.ind-add-item').forEach((btn) => {
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const type = btn.dataset.type;
      const name = btn.dataset.name;
      let cfg = {};
      for (const cat of INDICATOR_CATALOG) {
        const hit = cat.items.find((i) => i.type === type);
        if (hit?.defaultConfig) cfg = JSON.parse(JSON.stringify(hit.defaultConfig));
      }
      const newId = addIndicator(type, name, cfg);
      _expanded.add(newId); // auto-expand newly added so user sees config
      if (catEl) catEl.hidden = true;
    });
  });
}

function wireMARibbon(cfgEl, id, ind, onChange) {
  // Preset dropdown
  cfgEl.querySelector('[data-action="preset"]')?.addEventListener('change', (ev) => {
    ev.stopPropagation();
    const presetId = ev.target.value;
    if (!presetId) return;
    const p = RIBBON_PRESETS.find((x) => x.id === presetId);
    if (!p) return;
    updateConfig(id, { lines: JSON.parse(JSON.stringify(p.lines)) });
  });

  // Per-line fields
  cfgEl.querySelectorAll('.cfg-line-row').forEach((lineRow) => {
    const idx = Number(lineRow.dataset.lineIdx);
    lineRow.querySelectorAll('[data-f]').forEach((input) => {
      const commit = () => {
        const field = input.dataset.f;
        const raw = input.value;
        const lines = JSON.parse(JSON.stringify(ind.config?.lines || DEFAULT_RIBBON_LINES));
        if (!lines[idx]) return;
        if (field === 'period' || field === 'width') {
          const n = Math.max(1, Math.floor(Number(raw) || 0));
          if (!n) return;
          lines[idx][field] = n;
        } else if (field === 'type') {
          lines[idx].type = raw === 'ema' ? 'ema' : 'sma';
        } else if (field === 'color') {
          lines[idx].color = raw || '#ffeb3b';
        }
        updateConfig(id, { lines });
      };
      input.addEventListener('change', commit);
      // color input fires 'input' on every slider move — debounce to
      // avoid re-rendering 100 times per drag.
      if (input.type === 'color') {
        let t = null;
        input.addEventListener('input', () => {
          if (t) clearTimeout(t);
          t = setTimeout(commit, 150);
        });
      }
    });
    lineRow.querySelector('[data-action="line-del"]')?.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const lines = JSON.parse(JSON.stringify(ind.config?.lines || DEFAULT_RIBBON_LINES));
      if (lines.length <= 1) return;
      lines.splice(idx, 1);
      updateConfig(id, { lines });
    });
  });

  cfgEl.querySelector('[data-action="line-add"]')?.addEventListener('click', (ev) => {
    ev.stopPropagation();
    const lines = JSON.parse(JSON.stringify(ind.config?.lines || DEFAULT_RIBBON_LINES));
    const last = lines[lines.length - 1] || { period: 21, type: 'sma', color: '#888', width: 1 };
    lines.push({
      period: Math.min(500, Number(last.period || 21) * 2),
      type: last.type,
      color: '#9e9e9e',
      width: 1,
    });
    updateConfig(id, { lines });
  });

  cfgEl.querySelector('[data-action="reset-default"]')?.addEventListener('click', (ev) => {
    ev.stopPropagation();
    updateConfig(id, { lines: JSON.parse(JSON.stringify(DEFAULT_RIBBON_LINES)) });
  });
}

function wireSimpleEditor(cfgEl, id, ind, onChange) {
  cfgEl.querySelectorAll('[data-f]').forEach((input) => {
    const commit = () => {
      const field = input.dataset.f;
      const next = Number(input.value);
      if (!Number.isFinite(next) || next <= 0) return;
      updateConfig(id, { [field]: next });
    };
    input.addEventListener('change', commit);
  });
}

function injectStyles() {
  if (document.getElementById('ind-panel-styles')) return;
  const s = document.createElement('style');
  s.id = 'ind-panel-styles';
  s.textContent = `
    .ind-panel {
      background: #0e141f;
      border: 1px solid #2a3548;
      border-radius: 8px;
      color: #d8dde8;
      font-size: 12px;
      z-index: 10000;
      box-shadow: 0 12px 32px rgba(0,0,0,0.55);
      overflow: hidden;
      max-height: 660px;
      display: flex; flex-direction: column;
    }
    .ind-head {
      padding: 8px 12px;
      border-bottom: 1px solid #1d2537;
      font-weight: 600; font-size: 12px;
      display: flex; justify-content: space-between; align-items: center;
      background: #11182a;
    }
    .ind-close {
      background: none; border: none; color: #6b7889;
      cursor: pointer; font-size: 14px; line-height: 1;
    }
    .ind-close:hover { color: #e8edf5; }
    .ind-list {
      max-height: 440px;
      overflow-y: auto;
    }
    .ind-row {
      border-bottom: 1px solid #141a26;
    }
    .ind-row-head {
      display: flex; align-items: center; gap: 8px;
      padding: 6px 12px;
    }
    .ind-row-head:hover { background: #161e2f; }
    .ind-eye {
      background: none; border: none; cursor: pointer;
      font-size: 14px; width: 18px; padding: 0;
    }
    .ind-name {
      flex: 1; font-size: 11.5px; cursor: text;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .ind-name:hover { color: #38bdf8; }
    .ind-gear {
      background: none; border: none; cursor: pointer;
      color: #94a3b8; font-size: 13px; padding: 2px 6px;
    }
    .ind-gear:hover:not(.disabled) { color: #38bdf8; }
    .ind-gear.disabled { opacity: 0.3; cursor: not-allowed; }
    .ind-del {
      background: none; border: none; cursor: pointer;
      color: #6b7889; font-size: 12px; padding: 2px 6px;
    }
    .ind-del:hover { color: #ff5252; }
    .ind-empty {
      padding: 16px 12px; text-align: center;
      color: #6b7889; font-size: 11px;
    }
    .ind-footer {
      padding: 8px 10px;
      border-top: 1px solid #1d2537;
      background: #101728;
    }
    .ind-add-btn {
      width: 100%; padding: 6px 10px;
      background: #1d2537; border: 1px solid #2a3548;
      color: #e8edf5; font-size: 12px;
      border-radius: 4px; cursor: pointer;
    }
    .ind-add-btn:hover { background: #24304a; }
    .ind-catalog {
      margin-top: 6px; padding: 6px;
      background: #0a0f1a; border: 1px solid #1d2537;
      border-radius: 4px;
      max-height: 220px; overflow-y: auto;
    }
    .ind-cat-head {
      font-size: 10px; color: #7a8599; font-weight: 600;
      padding: 4px 6px; letter-spacing: 0.04em;
    }
    .ind-add-item {
      display: block; width: 100%; text-align: left;
      background: none; border: none; color: #d8dde8;
      padding: 5px 10px; font-size: 11.5px; cursor: pointer;
      border-radius: 3px;
    }
    .ind-add-item:hover { background: #1d2537; color: #38bdf8; }

    /* ─── Config editor (expanded under each row) ─────────── */
    .ind-config {
      padding: 8px 12px 10px 34px;
      background: #0a0f1a;
      border-top: 1px solid #141a26;
    }
    .cfg-block { display: flex; flex-direction: column; gap: 6px; }
    .cfg-row {
      display: flex; align-items: center; gap: 8px;
    }
    .cfg-label {
      width: 72px; font-size: 11px; color: #94a3b8;
    }
    .cfg-num, .cfg-preset {
      flex: 1; min-width: 0;
      background: #101728; border: 1px solid #24304a;
      color: #e8edf5; font-size: 11px;
      padding: 3px 6px; border-radius: 3px;
      font-family: inherit;
    }
    .cfg-num:focus, .cfg-preset:focus {
      outline: none; border-color: #38bdf8;
    }

    /* MA Ribbon line table */
    .cfg-lines {
      display: flex; flex-direction: column; gap: 4px;
      margin-top: 2px;
    }
    .cfg-lines-head {
      display: grid;
      grid-template-columns: 18px 62px 56px 32px 42px 20px;
      gap: 6px; align-items: center;
      font-size: 10px; color: #7a8599;
      padding: 0 0 2px 0;
    }
    .cfg-line-row {
      display: grid;
      grid-template-columns: 18px 62px 56px 32px 42px 20px;
      gap: 6px; align-items: center;
    }
    .cfg-line-num {
      font-size: 10px; color: #7a8599;
      text-align: center;
    }
    .cfg-period, .cfg-width, .cfg-type {
      background: #101728; border: 1px solid #24304a;
      color: #e8edf5; font-size: 11px;
      padding: 3px 4px; border-radius: 3px;
      font-family: inherit;
      min-width: 0; width: 100%; box-sizing: border-box;
    }
    .cfg-color {
      background: transparent; border: 1px solid #24304a;
      width: 32px; height: 22px; padding: 0;
      border-radius: 3px; cursor: pointer;
    }
    .cfg-color::-webkit-color-swatch-wrapper { padding: 2px; }
    .cfg-color::-webkit-color-swatch { border: none; border-radius: 2px; }
    .cfg-line-del {
      background: none; border: none; color: #6b7889;
      cursor: pointer; font-size: 11px; padding: 0;
    }
    .cfg-line-del:hover:not(:disabled) { color: #ff5252; }
    .cfg-line-del:disabled { opacity: 0.3; cursor: not-allowed; }

    .cfg-actions {
      display: flex; gap: 6px; margin-top: 4px;
    }
    .cfg-btn {
      flex: 1;
      background: #1d2537; border: 1px solid #2a3548;
      color: #d8dde8; font-size: 11px;
      padding: 4px 8px; border-radius: 3px;
      cursor: pointer;
    }
    .cfg-btn:hover { background: #24304a; color: #e8edf5; }

    .cfg-hint {
      margin-top: 4px; font-size: 10px; color: #6b7889;
      font-style: italic;
    }
  `;
  document.head.appendChild(s);
}
