// frontend/js/workbench/indicators/indicator_panel.js
//
// Floating dropdown panel shown when the user clicks the "指标" button.
// Lists the current indicators with visibility / delete controls, plus
// an "+ 添加指标" dropdown that uses INDICATOR_CATALOG to offer new ones.

import {
  getIndicators, setVisible, removeIndicator, addIndicator,
  subscribe, INDICATOR_CATALOG,
} from './indicator_controller.js';
import { esc } from '../../util/dom.js';

let _panel = null;
let _cleanup = null;

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
    // Align right edge of panel with right edge of anchor
    const width = 280;
    _panel.style.width = `${width}px`;
    let left = r.right - width;
    if (left < 8) left = 8;
    _panel.style.left = `${left}px`;
    _panel.style.top = `${r.bottom + 4}px`;
  } else {
    _panel.style.width = '280px';
    _panel.style.right = '16px';
    _panel.style.top = '60px';
  }
  // Viewport clamp
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
    return `
      <div class="ind-row" data-id="${esc(ind.id)}">
        <button class="ind-eye" data-action="toggle" title="${ind.visible ? '隐藏' : '显示'}" style="color:${eyeColor}">${eye}</button>
        <span class="ind-name">${esc(ind.name)}</span>
        <button class="ind-del" data-action="remove" title="删除">✕</button>
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

function wire(onChange) {
  if (!_panel) return;
  _panel.querySelectorAll('[data-action="toggle"]').forEach((btn) => {
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const row = btn.closest('.ind-row');
      if (!row) return;
      const id = row.dataset.id;
      const inds = getIndicators();
      const cur = inds.find((x) => x.id === id);
      if (cur) setVisible(id, !cur.visible);
    });
  });
  _panel.querySelectorAll('[data-action="remove"]').forEach((btn) => {
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const row = btn.closest('.ind-row');
      if (!row) return;
      removeIndicator(row.dataset.id);
    });
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
      // Look up defaultConfig from catalog
      let cfg = {};
      for (const cat of INDICATOR_CATALOG) {
        const hit = cat.items.find((i) => i.type === type);
        if (hit?.defaultConfig) cfg = { ...hit.defaultConfig };
      }
      addIndicator(type, name, cfg);
      if (catEl) catEl.hidden = true;
    });
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
      max-height: 560px;
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
      max-height: 280px;
      overflow-y: auto;
    }
    .ind-row {
      display: flex; align-items: center; gap: 8px;
      padding: 6px 12px;
      border-bottom: 1px solid #141a26;
    }
    .ind-row:hover { background: #161e2f; }
    .ind-eye {
      background: none; border: none; cursor: pointer;
      font-size: 14px; width: 18px; padding: 0;
    }
    .ind-name { flex: 1; font-size: 11.5px; }
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
  `;
  document.head.appendChild(s);
}
