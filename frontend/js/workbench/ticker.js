// frontend/js/workbench/ticker.js — searchable symbol picker (combobox)
//
// Replaces the native <select> with a text input + filtered dropdown.
// Typing characters narrows the list in-place. Arrow keys + enter pick.
// Click outside or Escape closes.

import { $, on } from '../util/dom.js';
import { marketState, setSymbol, setAllSymbols } from '../state/market.js';
import * as marketSvc from '../services/market.js';

const PINNED = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'HYPEUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT'];

let _allSymbols = [];
let _input = null;
let _dropdown = null;
let _combo = null;
let _activeIdx = -1;       // keyboard-highlighted row
let _currentList = [];     // currently displayed filtered symbols

export async function initTicker(comboId = 'v2-symbol-combo') {
  try {
    const symbols = await marketSvc.getSymbols();
    _allSymbols = Array.isArray(symbols) ? symbols : [];
    setAllSymbols(_allSymbols);

    _combo = $('#' + comboId);
    _input = $('#v2-symbol-input');
    _dropdown = $('#v2-symbol-dropdown');
    if (!_combo || !_input || !_dropdown) return;

    _input.value = marketState.currentSymbol || '';

    on(_input, 'focus', () => {
      _input.select();
      openDropdown('');
    });
    on(_input, 'input', () => openDropdown(_input.value));
    on(_input, 'keydown', onKey);
    on(_input, 'blur', () => {
      // Delay so click events on dropdown rows still fire first
      setTimeout(() => {
        closeDropdown();
        // Restore to current symbol if user typed garbage and blurred
        if (!_allSymbols.includes(_input.value.toUpperCase())) {
          _input.value = marketState.currentSymbol || '';
        }
      }, 150);
    });

    // Click outside dismisses
    document.addEventListener('mousedown', (ev) => {
      if (!_combo) return;
      if (!_combo.contains(ev.target)) closeDropdown();
    });
  } catch (err) {
    console.error('[ticker] failed to load symbols:', err);
  }
}

function openDropdown(query) {
  if (!_dropdown) return;
  const q = (query || '').trim().toUpperCase();
  const filtered = q
    ? _allSymbols.filter((s) => s.includes(q))
    : _allSymbols.slice();

  // Reorder: pinned first (if they match), then everything else
  const pinned = PINNED.filter((s) => filtered.includes(s));
  const rest = filtered.filter((s) => !pinned.includes(s));
  _currentList = [...pinned, ...rest];
  _activeIdx = _currentList.length > 0 ? 0 : -1;

  renderDropdown();
  _dropdown.hidden = false;
}

function renderDropdown() {
  if (!_dropdown) return;
  if (_currentList.length === 0) {
    _dropdown.innerHTML = `<div class="v2-symbol-empty">没有匹配</div>`;
    return;
  }
  const pinnedSet = new Set(PINNED);
  let html = '';
  let seenPinnedGroup = false;
  let seenAllGroup = false;
  _currentList.forEach((s, i) => {
    const isPinned = pinnedSet.has(s);
    if (isPinned && !seenPinnedGroup) {
      html += `<div class="v2-symbol-group">Popular</div>`;
      seenPinnedGroup = true;
    }
    if (!isPinned && !seenAllGroup) {
      html += `<div class="v2-symbol-group">All</div>`;
      seenAllGroup = true;
    }
    const activeCls = i === _activeIdx ? 'is-active' : '';
    html += `<div class="v2-symbol-row ${activeCls}" data-sym="${s}" data-idx="${i}">${s}</div>`;
  });
  _dropdown.innerHTML = html;
  // Bind clicks on rows (fires before blur thanks to 150ms timeout above)
  _dropdown.querySelectorAll('[data-sym]').forEach((el) => {
    el.addEventListener('mousedown', (ev) => {
      ev.preventDefault();  // keep focus on input
      pick(el.dataset.sym);
    });
    el.addEventListener('mouseenter', () => {
      const idx = Number(el.dataset.idx);
      if (!isNaN(idx)) { _activeIdx = idx; updateActiveRow(); }
    });
  });
}

function updateActiveRow() {
  if (!_dropdown) return;
  _dropdown.querySelectorAll('[data-idx]').forEach((el) => {
    const idx = Number(el.dataset.idx);
    el.classList.toggle('is-active', idx === _activeIdx);
  });
  // Scroll active into view
  const el = _dropdown.querySelector('.v2-symbol-row.is-active');
  if (el && typeof el.scrollIntoView === 'function') {
    el.scrollIntoView({ block: 'nearest' });
  }
}

function onKey(ev) {
  if (_dropdown?.hidden) {
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
      ev.preventDefault();
      openDropdown(_input.value);
    }
    return;
  }
  if (ev.key === 'ArrowDown') {
    ev.preventDefault();
    if (_currentList.length === 0) return;
    _activeIdx = (_activeIdx + 1) % _currentList.length;
    updateActiveRow();
  } else if (ev.key === 'ArrowUp') {
    ev.preventDefault();
    if (_currentList.length === 0) return;
    _activeIdx = (_activeIdx - 1 + _currentList.length) % _currentList.length;
    updateActiveRow();
  } else if (ev.key === 'Enter') {
    ev.preventDefault();
    if (_activeIdx >= 0 && _activeIdx < _currentList.length) {
      pick(_currentList[_activeIdx]);
    }
  } else if (ev.key === 'Escape') {
    ev.preventDefault();
    closeDropdown();
    _input.value = marketState.currentSymbol || '';
    _input.blur();
  }
}

function pick(sym) {
  if (!sym) return;
  _input.value = sym;
  setSymbol(sym);
  closeDropdown();
  _input.blur();
}

function closeDropdown() {
  if (_dropdown) _dropdown.hidden = true;
}
