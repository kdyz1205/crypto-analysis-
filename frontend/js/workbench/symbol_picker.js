// frontend/js/workbench/symbol_picker.js
//
// Sortable 4-column symbol picker (Bitget-style).
//
// Columns (match user spec 2026-04-20):
//   - 币种 (symbol)
//   - 交易量 (24h USDT volume)
//   - 最新价 (last price)
//   - 24小时涨跌幅 (24h change %)
//
// Behaviour:
//   - Click empty input → top-200 rows by current sort (default: volume desc)
//   - Type text → filter by symbol prefix, fall back to substring
//   - Click column header → toggle sort (asc/desc), arrow indicator
//   - Click row or Enter → pick symbol, dispatch market.symbol.changed
//   - Arrow up/down → keyboard nav
//   - Portal to <body> so dropdown escapes header overflow:hidden
//
// Backend:
//   /api/symbols/extended → [{symbol, last_price, change24h, volume_usdt}]

import { $, on, esc } from '../util/dom.js';
import { marketState, setSymbol, setAllSymbols } from '../state/market.js';
import * as marketSvc from '../services/market.js';

const TOP_N_DEFAULT = 200;
const DROPDOWN_MAX_ROWS = 300;

// Row shape: { symbol, last_price, change24h, volume_usdt }
let _rows = [];                  // full dataset from /api/symbols/extended
let _allSymbols = [];            // flat symbol list (back-compat for marketState)
let _input = null;
let _dropdown = null;
let _combo = null;
let _activeIdx = -1;
let _currentList = [];
let _isOpen = false;
let _outsideHandler = null;
let _sortKey = 'volume_usdt';    // current sort column
let _sortDir = 'desc';           // 'asc' | 'desc'

export async function initSymbolPicker(comboId = 'v2-symbol-combo') {
  const loadLegacyFallback = async (reason) => {
    console.warn(`[symbol_picker] using /api/symbols fallback (${reason})`);
    try {
      const syms = await marketSvc.getSymbols();
      _allSymbols = Array.isArray(syms) ? syms : [];
      _rows = _allSymbols.map((s) => ({ symbol: s, last_price: 0, change24h: 0, volume_usdt: 0 }));
      setAllSymbols(_allSymbols);
    } catch (err2) {
      console.error('[symbol_picker] legacy fallback also failed:', err2);
      _rows = [];
      _allSymbols = [];
    }
  };

  try {
    const data = await marketSvc.getSymbolsExtended(TOP_N_DEFAULT);
    const rows = Array.isArray(data) ? data : [];
    // /api/symbols/extended returns [] with HTTP 200 when Bitget's ticker
    // fetch hiccups (see server/routers/market.py). Treat empty-but-ok
    // the same as thrown — user's picker must never render zero symbols.
    // Caught by Codex review 2026-04-20.
    if (rows.length === 0) {
      await loadLegacyFallback('extended returned empty');
    } else {
      _rows = rows;
      _allSymbols = _rows.map((r) => r.symbol);
      setAllSymbols(_allSymbols);
    }
  } catch (err) {
    console.error('[symbol_picker] extended load threw:', err);
    await loadLegacyFallback('extended threw');
  }

  _combo = $('#' + comboId);
  _input = $('#v2-symbol-input');
  _dropdown = $('#v2-symbol-dropdown');
  if (!_combo || !_input || !_dropdown) {
    console.warn('[symbol_picker] DOM not found:', comboId);
    return;
  }

  if (_dropdown.parentNode !== document.body) {
    document.body.appendChild(_dropdown);
  }
  injectStyles();

  _input.value = marketState.currentSymbol || '';

  on(_input, 'focus', () => {
    _input.select();
    open('');
  });
  on(_input, 'input', () => open(_input.value));
  on(_input, 'keydown', onKey);
  on(_input, 'blur', () => {
    setTimeout(() => {
      close();
      const typed = (_input.value || '').toUpperCase();
      if (typed && !_allSymbols.includes(typed)) {
        _input.value = marketState.currentSymbol || '';
      }
    }, 150);
  });

  _outsideHandler = (ev) => {
    if (!_isOpen) return;
    if (_combo && _combo.contains(ev.target)) return;
    if (_dropdown && _dropdown.contains(ev.target)) return;
    close();
  };
  document.addEventListener('mousedown', _outsideHandler);

  window.addEventListener('resize', () => { if (_isOpen) positionDropdown(); });
  window.addEventListener('scroll', () => { if (_isOpen) positionDropdown(); }, true);
}

// ───────── open / filter / sort ─────────

function open(query) {
  if (!_dropdown) return;
  const q = (query || '').trim().toUpperCase();

  let filtered;
  if (!q) {
    filtered = [..._rows];
  } else {
    const prefix = _rows.filter((r) => r.symbol.startsWith(q));
    filtered = prefix.length > 0
      ? prefix
      : _rows.filter((r) => r.symbol.includes(q));
  }

  filtered = applySort(filtered);
  _currentList = filtered.slice(0, DROPDOWN_MAX_ROWS);
  _activeIdx = _currentList.length > 0 ? 0 : -1;
  render();
  positionDropdown();
  _dropdown.hidden = false;
  _isOpen = true;
}

function applySort(list) {
  const key = _sortKey;
  const dir = _sortDir === 'asc' ? 1 : -1;
  const copy = [...list];
  copy.sort((a, b) => {
    if (key === 'symbol') {
      return dir * a.symbol.localeCompare(b.symbol);
    }
    const av = Number(a[key]) || 0;
    const bv = Number(b[key]) || 0;
    return dir * (av - bv);
  });
  return copy;
}

function cycleSort(key) {
  if (_sortKey === key) {
    _sortDir = _sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    _sortKey = key;
    // Sensible default direction per column
    _sortDir = key === 'symbol' ? 'asc' : 'desc';
  }
  open(_input.value);
}

function positionDropdown() {
  if (!_dropdown || !_input) return;
  const r = _input.getBoundingClientRect();
  _dropdown.style.position = 'fixed';
  _dropdown.style.top = `${r.bottom}px`;
  _dropdown.style.left = `${r.left}px`;
  _dropdown.style.width = `${Math.max(r.width, 520)}px`;
}

function sortIndicator(key) {
  if (_sortKey !== key) return '<span class="sp-sort sp-sort-idle">⇅</span>';
  return _sortDir === 'asc'
    ? '<span class="sp-sort sp-sort-active">▲</span>'
    : '<span class="sp-sort sp-sort-active">▼</span>';
}

function fmtVolume(v) {
  const n = Number(v) || 0;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
}

function fmtPrice(p) {
  const n = Number(p) || 0;
  if (n === 0) return '—';
  if (n >= 1000) return n.toFixed(2);
  if (n >= 10) return n.toFixed(3);
  if (n >= 1) return n.toFixed(4);
  if (n >= 0.01) return n.toFixed(5);
  return n.toFixed(7);
}

function fmtChange(c) {
  const n = Number(c) || 0;
  const pct = n * 100;
  const sign = pct > 0 ? '+' : '';
  return `${sign}${pct.toFixed(2)}%`;
}

function render() {
  if (!_dropdown) return;

  const header = `
    <div class="sp-head">
      <div class="sp-col sp-col-sym sp-h" data-sort="symbol">币种 ${sortIndicator('symbol')}</div>
      <div class="sp-col sp-col-vol sp-h" data-sort="volume_usdt">交易量 ${sortIndicator('volume_usdt')}</div>
      <div class="sp-col sp-col-price sp-h" data-sort="last_price">最新价 ${sortIndicator('last_price')}</div>
      <div class="sp-col sp-col-chg sp-h" data-sort="change24h">24小时涨跌幅 ${sortIndicator('change24h')}</div>
    </div>
  `;

  if (_currentList.length === 0) {
    _dropdown.innerHTML = header + `<div class="sp-empty">(no match)</div>`;
    wireHeader();
    return;
  }

  const rows = _currentList.map((r, i) => {
    const activeCls = i === _activeIdx ? 'is-active' : '';
    const chgCls = r.change24h > 0 ? 'sp-up' : r.change24h < 0 ? 'sp-dn' : '';
    return `
      <div class="sp-row ${activeCls}" data-sym="${esc(r.symbol)}" data-idx="${i}">
        <div class="sp-col sp-col-sym">${esc(r.symbol)}</div>
        <div class="sp-col sp-col-vol">${fmtVolume(r.volume_usdt)}</div>
        <div class="sp-col sp-col-price">${fmtPrice(r.last_price)}</div>
        <div class="sp-col sp-col-chg ${chgCls}">${fmtChange(r.change24h)}</div>
      </div>`;
  }).join('');

  _dropdown.innerHTML = header + `<div class="sp-body">${rows}</div>`;

  wireHeader();
  _dropdown.querySelectorAll('[data-sym]').forEach((el) => {
    el.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      pick(el.dataset.sym);
    });
    el.addEventListener('mouseenter', () => {
      const idx = Number(el.dataset.idx);
      if (!Number.isNaN(idx)) {
        _activeIdx = idx;
        updateActiveRow();
      }
    });
  });
}

function wireHeader() {
  _dropdown.querySelectorAll('.sp-h[data-sort]').forEach((el) => {
    el.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      cycleSort(el.dataset.sort);
    });
  });
}

function updateActiveRow() {
  if (!_dropdown) return;
  _dropdown.querySelectorAll('[data-idx]').forEach((el) => {
    const idx = Number(el.dataset.idx);
    el.classList.toggle('is-active', idx === _activeIdx);
  });
  const el = _dropdown.querySelector('.sp-row.is-active');
  if (el && typeof el.scrollIntoView === 'function') {
    el.scrollIntoView({ block: 'nearest' });
  }
}

// ───────── keyboard ─────────

function onKey(ev) {
  if (!_isOpen) {
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
      ev.preventDefault();
      open(_input.value);
    }
    if (ev.key === 'Enter') {
      const typed = (_input.value || '').trim().toUpperCase();
      if (typed && _allSymbols.includes(typed)) {
        ev.preventDefault();
        pick(typed);
      }
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
      pick(_currentList[_activeIdx].symbol);
      return;
    }
    const typed = (_input.value || '').trim().toUpperCase();
    if (typed && _allSymbols.includes(typed)) {
      pick(typed);
    }
  } else if (ev.key === 'Escape') {
    ev.preventDefault();
    close();
    _input.value = marketState.currentSymbol || '';
    _input.blur();
  }
}

// ───────── pick / close ─────────

function pick(sym) {
  if (!sym) return;
  _input.value = sym;
  setSymbol(sym);
  close();
  _input.blur();
}

function close() {
  if (_dropdown) _dropdown.hidden = true;
  _isOpen = false;
}

// ───────── styles ─────────

function injectStyles() {
  if (document.getElementById('sp-styles')) return;
  const el = document.createElement('style');
  el.id = 'sp-styles';
  el.textContent = `
    #v2-symbol-dropdown {
      background: #0e141f;
      border: 1px solid #2a3548;
      border-radius: 6px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.6);
      color: #d8dde8;
      font-size: 12px;
      max-height: 520px;
      overflow: hidden;
      z-index: 10000;
    }
    .sp-head {
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr 1fr;
      background: #161e2f;
      border-bottom: 1px solid #2a3548;
      font-size: 11px;
      font-weight: 600;
      color: #8a95a6;
      position: sticky; top: 0; z-index: 1;
    }
    .sp-head .sp-col { padding: 8px 10px; user-select: none; }
    .sp-head .sp-h { cursor: pointer; display: flex; align-items: center; gap: 4px; }
    .sp-head .sp-h:hover { color: #d8dde8; }
    .sp-sort { font-size: 9px; line-height: 1; }
    .sp-sort-idle { opacity: 0.35; }
    .sp-sort-active { color: #38bdf8; opacity: 1; }
    .sp-body {
      max-height: 460px;
      overflow-y: auto;
    }
    .sp-row {
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr 1fr;
      border-bottom: 1px solid #141a26;
      cursor: pointer;
    }
    .sp-row:hover, .sp-row.is-active {
      background: #1d2537;
    }
    .sp-row .sp-col {
      padding: 6px 10px;
      font-variant-numeric: tabular-nums;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .sp-col-sym { font-weight: 600; color: #e8edf5; }
    .sp-col-vol { color: #a3b0c4; text-align: right; }
    .sp-col-price { color: #e8edf5; text-align: right; }
    .sp-col-chg { text-align: right; font-weight: 600; }
    .sp-up { color: #26a69a; }
    .sp-dn { color: #ef5350; }
    .sp-empty { padding: 12px; text-align: center; color: #6b7889; }
  `;
  document.head.appendChild(el);
}

// Exposed for tests / debug
export function _debugState() {
  return {
    total: _rows.length,
    listed: _currentList.length,
    isOpen: _isOpen,
    activeIdx: _activeIdx,
    current: _input?.value,
    sortKey: _sortKey,
    sortDir: _sortDir,
  };
}
