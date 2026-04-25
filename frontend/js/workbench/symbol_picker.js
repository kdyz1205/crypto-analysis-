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
import { subscribe } from '../util/events.js';

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

// ──────────────────────────────────────────────────────────────
// 2026-04-23: Bitget-style category tabs + user-owned favourites.
//
// Tabs:
//   全部 | 收藏 | 贵金属 | 商品 | 股票 | ETF
//
// 收藏 (favourites) is user-defined, persisted in localStorage key
// `v2.picker.favorites`. Each row in the dropdown has a ☆ / ★ toggle.
// Seed defaults = the old hardcoded 核心资产 list so the first-time user
// isn't staring at an empty favourites tab.
// ──────────────────────────────────────────────────────────────
const FAVORITES_LS_KEY = 'v2.picker.favorites';
const DEFAULT_FAVORITES = ['BTCUSDT','ETHUSDT','SOLUSDT','HYPEUSDT','ZECUSDT','BNBUSDT','XRPUSDT','DOGEUSDT'];

function loadFavorites() {
  try {
    const raw = localStorage.getItem(FAVORITES_LS_KEY);
    if (!raw) return new Set(DEFAULT_FAVORITES);
    const arr = JSON.parse(raw);
    return new Set(Array.isArray(arr) ? arr.map(String) : DEFAULT_FAVORITES);
  } catch { return new Set(DEFAULT_FAVORITES); }
}

function saveFavorites(set) {
  try { localStorage.setItem(FAVORITES_LS_KEY, JSON.stringify([...set])); } catch {}
}

let _favorites = loadFavorites();

function toggleFavorite(sym) {
  if (_favorites.has(sym)) _favorites.delete(sym);
  else _favorites.add(sym);
  saveFavorites(_favorites);
}

const CATEGORY_TABS = [
  { id: 'all',       label: '全部',    pred: () => true },
  { id: 'favorite',  label: '收藏 ★',  pred: (r) => _favorites.has(r.symbol) },
  { id: 'precious',  label: '贵金属',  pred: (r) => ['XAUUSDT','XAGUSDT','XPTUSDT','XPDUSDT'].includes(r.symbol) },
  { id: 'commodity', label: '商品',    pred: (r) => r.category === 'commodity' },
  { id: 'stock',     label: '股票',    pred: (r) => r.category === 'stock' },
  { id: 'index',     label: 'ETF',     pred: (r) => r.category === 'index' },
];
const CATEGORY_LS_KEY = 'v2.picker.active_category';
let _activeCategoryId = (() => {
  try { return localStorage.getItem(CATEGORY_LS_KEY) || 'all'; } catch { return 'all'; }
})();

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

  // 2026-04-25: subscribe to market.symbol.changed so the search box
  // ALWAYS shows the active symbol — including changes made elsewhere
  // (e.g. clicking a row in 我的手画线 panel, command palette, deep
  // link). Bug surfaced when user clicked XAUUSDT in the side panel
  // and the top search box still showed HYPEUSDT, with the chart half-
  // rendered for XAU. Per PRINCIPLES.md P14, fix at the source: any
  // UI element displaying current symbol MUST react to the bus event.
  subscribe('market.symbol.changed', (sym) => {
    if (!_input) return;
    const next = String(sym || marketState.currentSymbol || '');
    // Avoid clobbering the user's in-progress typing — only sync if the
    // user isn't actively focused on the input. They could be searching
    // for a different symbol when an event fires (e.g. ws_ticker
    // reconnect re-publishing).
    if (document.activeElement !== _input) {
      _input.value = next;
    }
  });

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

  // Apply active filter preset (if backend screen endpoint returned matches)
  filtered = _applyFilterToList(filtered);
  filtered = applySort(filtered);
  _currentList = filtered.slice(0, DROPDOWN_MAX_ROWS);
  _activeIdx = _currentList.length > 0 ? 0 : -1;
  render();
  positionDropdown();
  _dropdown.hidden = false;
  _isOpen = true;

  // Kick off filter refresh in background so next render has fresh
  // screener results. Non-blocking.
  void applyActiveFilter().then(() => {
    if (_isOpen) {
      const refilt = applySort(_applyFilterToList(_rows.filter((r) => !q || r.symbol.includes(q))));
      _currentList = refilt.slice(0, DROPDOWN_MAX_ROWS);
      _activeIdx = _currentList.length > 0 ? 0 : -1;
      render();
    }
  });
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
  _dropdown.style.top = `${r.bottom + 2}px`;
  _dropdown.style.left = `${r.left}px`;
  // Narrower — 420 instead of 520. Four columns still readable because
  // volume / price are K-formatted and change24h is compact (+x.xx%).
  _dropdown.style.width = `${Math.max(r.width, 420)}px`;
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

// ──────────────────────────────────────────────────────────────
// Filter presets (user-defined screener rules — TradingView-like)
// localStorage key: v2.picker.filters.v1
//   { active_id, presets: [{ id, name, rules: [...] }] }
// Each rule matches the symbol's ma/ema/slope/volume state.
// For now only a small set of scannable rules — backend does the
// actual evaluation via /api/symbols/screen (to be added next turn).
// ──────────────────────────────────────────────────────────────
const FILTERS_LS_KEY = 'v2.picker.filters.v1';

function loadFilterPresets() {
  try {
    const raw = localStorage.getItem(FILTERS_LS_KEY);
    if (!raw) return _seedFilterPresets();
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed?.presets) || parsed.presets.length === 0) return _seedFilterPresets();
    return parsed;
  } catch {
    return _seedFilterPresets();
  }
}
function _seedFilterPresets() {
  // 3 sensible defaults matching user's TradingView screenshots:
  const seed = {
    active_id: 'all',
    presets: [
      { id: 'all',        name: '全部',              rules: [] },
      { id: 'bull_4h',    name: '4h EMA21>MA55 多头', rules: [{ tf: '4h', kind: 'ema_above', fast: 21, slow: 55 }] },
      { id: 'bull_1h',    name: '1h 多头周期',         rules: [{ tf: '1h', kind: 'ema_above', fast: 8, slow: 21 }] },
      { id: 'weekly_bull',name: '周线 EMA21>MA55 多头',rules: [{ tf: '1w', kind: 'ema_above', fast: 21, slow: 55 }] },
    ],
  };
  try { localStorage.setItem(FILTERS_LS_KEY, JSON.stringify(seed)); } catch {}
  return seed;
}
function saveFilterPresets(state) {
  try { localStorage.setItem(FILTERS_LS_KEY, JSON.stringify(state)); } catch {}
}

function renderFilterBar() {
  const state = loadFilterPresets();
  const chips = state.presets.map((p) => {
    const active = p.id === state.active_id;
    return `<div class="sp-filter-chip ${active ? 'is-active' : ''}" data-filter-id="${esc(p.id)}">${esc(p.name)}</div>`;
  }).join('');
  return `
    <div class="sp-filter-bar">
      ${chips}
      <div class="sp-filter-chip sp-filter-add" data-filter-id="__configure__" title="管理 filter">⚙</div>
    </div>
  `;
}

function render() {
  if (!_dropdown) return;

  const catBar = renderCategoryBar();
  const filterBar = renderFilterBar();
  const header = `
    <div class="sp-head">
      <div class="sp-col sp-col-sym sp-h" data-sort="symbol">币种 ${sortIndicator('symbol')}</div>
      <div class="sp-col sp-col-vol sp-h" data-sort="volume_usdt">量 ${sortIndicator('volume_usdt')}</div>
      <div class="sp-col sp-col-price sp-h" data-sort="last_price">价 ${sortIndicator('last_price')}</div>
      <div class="sp-col sp-col-chg sp-h" data-sort="change24h">24h ${sortIndicator('change24h')}</div>
    </div>
  `;

  if (_currentList.length === 0) {
    _dropdown.innerHTML = catBar + filterBar + header + `<div class="sp-empty">(no match)</div>`;
    wireCategoryBar();
    wireHeader();
    wireFilterBar();
    return;
  }

  const rows = _currentList.map((r, i) => {
    const activeCls = i === _activeIdx ? 'is-active' : '';
    const chgCls = r.change24h > 0 ? 'sp-up' : r.change24h < 0 ? 'sp-dn' : '';
    const favCls = _favorites.has(r.symbol) ? 'is-fav' : '';
    const favChar = _favorites.has(r.symbol) ? '★' : '☆';
    return `
      <div class="sp-row ${activeCls}" data-sym="${esc(r.symbol)}" data-idx="${i}">
        <div class="sp-col sp-col-sym">
          <span class="sp-fav ${favCls}" data-fav-sym="${esc(r.symbol)}" title="收藏/取消收藏">${favChar}</span>
          ${esc(r.symbol)}
        </div>
        <div class="sp-col sp-col-vol">${fmtVolume(r.volume_usdt)}</div>
        <div class="sp-col sp-col-price">${fmtPrice(r.last_price)}</div>
        <div class="sp-col sp-col-chg ${chgCls}">${fmtChange(r.change24h)}</div>
      </div>`;
  }).join('');

  _dropdown.innerHTML = catBar + filterBar + header + `<div class="sp-body">${rows}</div>`;

  wireCategoryBar();
  wireHeader();
  wireFilterBar();
  wireFavoriteButtons();
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

function wireFilterBar() {
  _dropdown.querySelectorAll('[data-filter-id]').forEach((el) => {
    el.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      const id = el.dataset.filterId;
      if (id === '__configure__') {
        openFilterConfigureModal();
        return;
      }
      const state = loadFilterPresets();
      state.active_id = id;
      saveFilterPresets(state);
      applyActiveFilter();
      render();
    });
  });
}

let _matchedSymbols = null;   // null = no filter active; Set otherwise

async function applyActiveFilter() {
  const state = loadFilterPresets();
  const active = state.presets.find((p) => p.id === state.active_id);
  if (!active || active.rules.length === 0) {
    _matchedSymbols = null;    // no filter → all pass
    return;
  }
  try {
    const { fetchJson } = await import('../util/fetch.js');
    const resp = await fetchJson('/api/symbols/screen', {
      method: 'POST',
      body: { rules: active.rules },
      timeout: 20000,
    });
    _matchedSymbols = new Set((resp.matched || []).map((s) => String(s).toUpperCase()));
  } catch (err) {
    console.warn('[picker] filter /screen endpoint not ready', err);
    _matchedSymbols = null;    // fallback: no filter applied
  }
}

function _applyFilterToList(list) {
  // Category tab first, then screener filter.
  const tab = CATEGORY_TABS.find((t) => t.id === _activeCategoryId) || CATEGORY_TABS[0];
  let out = tab.id === 'all' ? list : list.filter(tab.pred);
  if (_matchedSymbols) {
    out = out.filter((r) => _matchedSymbols.has(String(r.symbol || '').toUpperCase()));
  }
  return out;
}

function renderCategoryBar() {
  const chips = CATEGORY_TABS.map((t) => {
    const active = t.id === _activeCategoryId;
    return `<div class="sp-cat-chip ${active ? 'is-active' : ''}" data-cat-id="${esc(t.id)}">${esc(t.label)}</div>`;
  }).join('');
  return `<div class="sp-cat-bar">${chips}</div>`;
}

function wireCategoryBar() {
  if (!_dropdown) return;
  _dropdown.querySelectorAll('[data-cat-id]').forEach((el) => {
    // Same mousedown-stops-blur trick as other chips so clicking doesn't
    // close the dropdown before the handler fires.
    el.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      _activeCategoryId = el.dataset.catId;
      try { localStorage.setItem(CATEGORY_LS_KEY, _activeCategoryId); } catch {}
      open(_input.value);
    });
  });
}

function wireFavoriteButtons() {
  if (!_dropdown) return;
  _dropdown.querySelectorAll('[data-fav-sym]').forEach((el) => {
    el.addEventListener('mousedown', (ev) => {
      // Stop the row-click (pick symbol) AND the outside-click (close).
      ev.preventDefault();
      ev.stopPropagation();
      const sym = el.dataset.favSym;
      toggleFavorite(sym);
      // Re-render so the star + the 收藏 tab both update in-place.
      open(_input.value);
    });
  });
}

function openFilterConfigureModal() {
  // Lightweight inline editor — reuses prompt() for now; a full UI
  // can come later if user wants. Shows filter list + delete + add.
  const state = loadFilterPresets();
  const menu = state.presets.map((p, i) => `${i}. ${p.name} (${p.rules.length} 条规则)`).join('\n');
  const action = prompt(
    '管理 filter:\n' + menu + '\n\n' +
    '输入:\n' +
    '  数字 → 删除该 filter\n' +
    '  + 名字|TF|kind|fast|slow  → 新建 (例: + 15m_cross|15m|ema_above|5|21)\n' +
    '  空 → 取消',
    ''
  );
  if (!action) return;
  if (/^\d+$/.test(action.trim())) {
    const idx = parseInt(action.trim(), 10);
    if (idx >= 0 && idx < state.presets.length) {
      const removed = state.presets.splice(idx, 1)[0];
      if (state.active_id === removed.id) state.active_id = 'all';
      saveFilterPresets(state);
      render();
    }
    return;
  }
  if (action.startsWith('+')) {
    const parts = action.slice(1).trim().split('|').map((s) => s.trim());
    if (parts.length < 5) { alert('格式: + 名字|TF|kind|fast|slow'); return; }
    const [name, tf, kind, fast, slow] = parts;
    const id = 'f_' + Date.now().toString(36);
    state.presets.push({
      id, name,
      rules: [{ tf, kind, fast: Number(fast), slow: Number(slow) }],
    });
    saveFilterPresets(state);
    render();
  }
}

function wireHeader() {
  _dropdown.querySelectorAll('.sp-h[data-sort]').forEach((el) => {
    // Bug 2026-04-20: mousedown → cycleSort → render() replaces
    // innerHTML synchronously, so the element the user clicked is
    // DETACHED from the DOM before the document-level outside-click
    // handler runs. That handler then sees a target no longer inside
    // _dropdown and closes the picker. Fixed by:
    //  (a) stopping propagation so the document handler never fires
    //  (b) using `click` (fires after mouseup) instead of mousedown,
    //      so the input's 150ms blur timeout is already waiting and
    //      close() from blur is benign
    el.addEventListener('mousedown', (ev) => {
      ev.preventDefault();  // don't steal focus from the input
      ev.stopPropagation();
    });
    el.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
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
      border-radius: 8px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.55);
      color: #d8dde8;
      font-size: 11.5px;
      max-height: 440px;
      overflow: hidden;
      z-index: 10000;
    }
    .sp-head {
      display: grid;
      grid-template-columns: 1.3fr 1fr 1fr 0.9fr;
      background: #11182a;
      border-bottom: 1px solid #1e2638;
      font-size: 10.5px;
      font-weight: 600;
      color: #7a8599;
      letter-spacing: 0.02em;
      position: sticky; top: 0; z-index: 1;
    }
    .sp-head .sp-col { padding: 6px 10px; user-select: none; }
    .sp-head .sp-h { cursor: pointer; display: flex; align-items: center; gap: 4px; transition: color 0.12s; }
    .sp-head .sp-h:hover { color: #d8dde8; }
    .sp-sort { font-size: 8.5px; line-height: 1; }
    .sp-sort-idle { opacity: 0.3; }
    .sp-sort-active { color: #38bdf8; opacity: 1; }
    .sp-body {
      max-height: 390px;
      overflow-y: auto;
    }
    .sp-row {
      display: grid;
      grid-template-columns: 1.3fr 1fr 1fr 0.9fr;
      border-bottom: 1px solid #121827;
      cursor: pointer;
      transition: background 0.08s;
    }
    .sp-row:hover, .sp-row.is-active {
      background: #1a2335;
    }
    .sp-row .sp-col {
      padding: 4.5px 10px;
      font-variant-numeric: tabular-nums;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .sp-col-sym { font-weight: 600; color: #e8edf5; }
    .sp-col-vol { color: #94a1b7; text-align: right; }
    .sp-col-price { color: #e8edf5; text-align: right; }
    .sp-col-chg { text-align: right; font-weight: 600; font-size: 11px; }
    .sp-up { color: #26a69a; }
    .sp-dn { color: #ef5350; }
    .sp-empty { padding: 12px; text-align: center; color: #6b7889; font-size: 11px; }

    /* Filter bar — TradingView-like chips above the symbol table.
       User 2026-04-22: "把交易量/最新价/涨跌幅 做小, 单独挪 filter
       配置区出来". Chips are compact, click switches filter, ⚙ opens
       inline configure modal. */
    .sp-filter-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      padding: 6px 8px;
      background: #0b121f;
      border-bottom: 1px solid #1e2638;
    }
    .sp-filter-chip {
      padding: 3px 8px;
      font-size: 10px;
      font-weight: 500;
      background: #141c2d;
      color: #94a1b7;
      border: 1px solid #1e2638;
      border-radius: 12px;
      cursor: pointer;
      user-select: none;
      transition: all 0.1s;
      white-space: nowrap;
    }
    .sp-filter-chip:hover {
      background: #1a2335;
      color: #d8dde8;
      border-color: #38bdf8;
    }
    .sp-filter-chip.is-active {
      background: #1e3a5c;
      color: #38bdf8;
      border-color: #38bdf8;
      font-weight: 600;
    }
    .sp-filter-chip.sp-filter-add {
      margin-left: auto;
      color: #6b7889;
      background: transparent;
    }
    .sp-filter-chip.sp-filter-add:hover {
      color: #38bdf8;
      background: #141c2d;
    }

    /* 2026-04-23 Bitget-style category tabs — sits above the preset
       screener filter bar. Category tabs are horizontal, slightly taller
       than filter chips, underline-highlight for the active one. */
    .sp-cat-bar {
      display: flex;
      gap: 2px;
      padding: 4px 8px 0;
      background: #0b121f;
      border-bottom: 1px solid #111826;
      overflow-x: auto;
    }
    .sp-cat-chip {
      padding: 6px 12px;
      font-size: 11px;
      font-weight: 500;
      color: #94a1b7;
      border-bottom: 2px solid transparent;
      cursor: pointer;
      user-select: none;
      transition: color 0.12s, border-color 0.12s;
      white-space: nowrap;
    }
    .sp-cat-chip:hover {
      color: #d8dde8;
    }
    .sp-cat-chip.is-active {
      color: #e8edf5;
      border-bottom-color: #38bdf8;
      font-weight: 600;
    }

    /* 2026-04-23 收藏夹 star — single-click toggle on each row. Hollow ☆
       = not favourited, solid ★ amber = favourited. Sits before the
       symbol text so the row still reads naturally. */
    .sp-fav {
      display: inline-block;
      width: 14px;
      font-size: 12px;
      color: #4a5568;
      cursor: pointer;
      margin-right: 4px;
      user-select: none;
      transition: color 0.12s, transform 0.12s;
    }
    .sp-fav:hover {
      transform: scale(1.15);
      color: #fbbf24;
    }
    .sp-fav.is-fav {
      color: #fbbf24;
    }
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
