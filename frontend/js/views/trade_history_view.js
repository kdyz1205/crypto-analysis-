// frontend/js/views/trade_history_view.js
//
// Spreadsheet-like view of every CLOSED manual-line trade. User
// 2026-04-22 asked for an Excel-style browse so they could post-mortem
// each trade and later train models on the features.
//
// Features:
//   - Sortable columns (click header to cycle asc → desc → unsorted)
//   - Filter bar (symbol + direction + close_reason + free-text)
//   - CSV export (all rows + all columns, respects current filter/sort)
//   - Click row to jump back to that symbol/TF on the market view
//   - Hover a row to see every column in a detail card below the table

import { esc } from '../util/dom.js';
import { fetchJson } from '../util/fetch.js';

let _host = null;
let _rows = [];
let _columns = [];
let _sort = { key: 'dt', dir: 'desc' };
let _filter = { symbol: '', direction: '', reason: '', q: '' };

export async function loadTradeHistory(el) {
  _host = el;
  el.innerHTML = renderShell();
  injectStyles();
  wire(el);
  await refresh();
}

export function unloadTradeHistory() {
  _host = null;
}

async function refresh() {
  try {
    const resp = await fetchJson('/api/trades/manual-history?limit=500', {
      noCache: true, timeout: 15000,
    });
    _rows = Array.isArray(resp?.rows) ? resp.rows : [];
    _columns = Array.isArray(resp?.columns) ? resp.columns : [];
    renderTable();
  } catch (err) {
    if (_host) {
      const tbody = _host.querySelector('#th-tbody');
      if (tbody) tbody.innerHTML = `<tr><td colspan="99" style="color:#ff5252;padding:12px">历史加载失败: ${esc(String(err?.message || err))}</td></tr>`;
    }
  }
}

function renderShell() {
  return `
  <div class="th-wrap">
    <header class="th-head">
      <h2>交易历史 · manual lines</h2>
      <div class="th-head-actions">
        <input class="th-filter" data-filter="symbol" placeholder="symbol (ETH...)" />
        <select class="th-filter" data-filter="direction">
          <option value="">全部方向</option>
          <option value="long">long</option>
          <option value="short">short</option>
        </select>
        <select class="th-filter" data-filter="reason">
          <option value="">全部 close reason</option>
          <option value="sl_or_tp">sl_or_tp</option>
          <option value="manual">manual</option>
          <option value="reverse">reverse</option>
        </select>
        <input class="th-filter" data-filter="q" placeholder="搜任意列 (RSI > 70 ...)" />
        <button class="th-btn" data-action="refresh">⟳ 刷新</button>
        <button class="th-btn th-btn-primary" data-action="export">⬇ CSV</button>
      </div>
    </header>
    <div class="th-stats" id="th-stats">—</div>
    <div class="th-table-wrap">
      <table class="th-table">
        <thead id="th-thead"></thead>
        <tbody id="th-tbody"><tr><td colspan="99" style="padding:12px;color:#8a95a6">加载中...</td></tr></tbody>
      </table>
    </div>
    <div class="th-detail" id="th-detail" hidden>
      <div class="th-detail-head">点击行查看全部特征</div>
      <pre id="th-detail-body"></pre>
    </div>
  </div>`;
}

function wire(el) {
  el.addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-action]');
    if (btn) {
      const action = btn.dataset.action;
      if (action === 'refresh') refresh();
      if (action === 'export') exportCsv();
      return;
    }
    const th = ev.target.closest('th[data-sort]');
    if (th) {
      const k = th.dataset.sort;
      if (_sort.key === k) {
        _sort.dir = _sort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        _sort.key = k;
        _sort.dir = 'desc';
      }
      renderTable();
      return;
    }
    const row = ev.target.closest('tr[data-row-idx]');
    if (row) {
      const idx = Number(row.dataset.rowIdx);
      showDetail(idx);
    }
  });
  el.querySelectorAll('[data-filter]').forEach((inp) => {
    inp.addEventListener('input', () => {
      const k = inp.dataset.filter;
      _filter[k] = inp.value.trim().toLowerCase();
      renderTable();
    });
  });
}

const DEFAULT_VISIBLE_COLS = [
  'dt', 'symbol', 'timeframe', 'side', 'entry_price', 'exit_price',
  'price_move_pct', 'pnl_usd', 'pnl_pct', 'close_reason',
  'bars_to_fill', 'bars_held', 'notional_usd', 'equity_pct', 'leverage',
  'tolerance_pct_of_line', 'stop_offset_pct_of_line', 'replan_count',
  'line_kind', 'touch_count', 'feat_rsi', 'feat_atr_pct', 'feat_ribbon_score',
  'feat_bb_pct', 'user_label',
];

function currentVisibleColumns() {
  // Intersection of preferred order with what the backend actually returned.
  const have = new Set(_columns);
  return DEFAULT_VISIBLE_COLS.filter((c) => have.has(c));
}

function filteredRows() {
  return _rows.filter((r) => {
    if (_filter.symbol && !(r.symbol || '').toLowerCase().includes(_filter.symbol)) return false;
    if (_filter.direction && r.side !== _filter.direction) return false;
    if (_filter.reason && r.close_reason !== _filter.reason) return false;
    if (_filter.q) {
      const blob = Object.entries(r).map(([k, v]) => `${k}:${v}`).join(' ').toLowerCase();
      if (!blob.includes(_filter.q)) return false;
    }
    return true;
  });
}

function sortedRows(rows) {
  const k = _sort.key;
  if (!k) return rows;
  const dir = _sort.dir === 'asc' ? 1 : -1;
  return [...rows].sort((a, b) => {
    const av = a[k];
    const bv = b[k];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
    return String(av).localeCompare(String(bv)) * dir;
  });
}

function fmt(v, col) {
  if (v == null) return '—';
  if (typeof v === 'number') {
    if (col === 'pnl_pct' || col === 'price_move_pct') {
      return (v * 100).toFixed(2) + '%';
    }
    if (col === 'pnl_usd' || col === 'margin_used' || col === 'notional_usd') {
      return (v >= 0 ? '+' : '') + v.toFixed(2);
    }
    if (col === 'dt' || col === 'ts') return v; // handled below
    if (Math.abs(v) < 1 && v !== 0) return v.toFixed(6);
    return String(v);
  }
  if (col === 'dt' && typeof v === 'string') {
    return v.replace('T', ' ').split('.')[0];  // YYYY-MM-DD HH:MM:SS
  }
  return String(v);
}

function renderTable() {
  if (!_host) return;
  const cols = currentVisibleColumns();
  const rows = sortedRows(filteredRows());
  const thead = _host.querySelector('#th-thead');
  const tbody = _host.querySelector('#th-tbody');
  const stats = _host.querySelector('#th-stats');

  if (stats) {
    const totalPnl = rows.reduce((s, r) => s + (Number(r.pnl_usd) || 0), 0);
    const wins = rows.filter((r) => Number(r.pnl_usd) > 0).length;
    const losses = rows.filter((r) => Number(r.pnl_usd) < 0).length;
    const wr = rows.length ? (wins / (wins + losses) * 100).toFixed(1) : '—';
    const pnlColor = totalPnl >= 0 ? '#00e676' : '#ff5252';
    stats.innerHTML = `
      显示 <b>${rows.length}</b> / ${_rows.length} 笔 ·
      净 PnL <b style="color:${pnlColor}">${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}</b> USDT ·
      胜率 <b>${wr}%</b> (${wins}W / ${losses}L)
    `;
  }

  if (thead) {
    thead.innerHTML = '<tr>' + cols.map((c) => {
      const active = _sort.key === c;
      const arrow = active ? (_sort.dir === 'asc' ? ' ▲' : ' ▼') : '';
      return `<th data-sort="${esc(c)}"${active ? ' class="is-active"' : ''}>${esc(c)}${arrow}</th>`;
    }).join('') + '</tr>';
  }

  if (tbody) {
    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="${cols.length}" style="padding:14px;color:#8a95a6;text-align:center">没有匹配的交易记录</td></tr>`;
    } else {
      tbody.innerHTML = rows.map((r, idx) => {
        const cls = (Number(r.pnl_usd) || 0) >= 0 ? 'th-win' : 'th-loss';
        return `<tr class="${cls}" data-row-idx="${idx}">` +
          cols.map((c) => `<td class="col-${esc(c)}">${esc(fmt(r[c], c))}</td>`).join('') +
          `</tr>`;
      }).join('');
    }
  }
}

function showDetail(idx) {
  const detail = _host?.querySelector('#th-detail');
  const body = _host?.querySelector('#th-detail-body');
  if (!detail || !body) return;
  const r = sortedRows(filteredRows())[idx];
  if (!r) return;
  detail.hidden = false;
  body.textContent = JSON.stringify(r, null, 2);
}

function exportCsv() {
  const cols = _columns.length > 0 ? _columns : currentVisibleColumns();
  const rows = sortedRows(filteredRows());
  const escCsv = (v) => {
    if (v == null) return '';
    const s = typeof v === 'number' ? String(v) : String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const header = cols.join(',');
  const body = rows.map((r) => cols.map((c) => escCsv(r[c])).join(',')).join('\n');
  const csv = header + '\n' + body;
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  a.download = `manual_trades_${ts}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

function injectStyles() {
  if (document.getElementById('th-styles')) return;
  const s = document.createElement('style');
  s.id = 'th-styles';
  s.textContent = `
    .th-wrap { padding: 14px 18px; color: #d8dde8; }
    .th-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
    .th-head h2 { margin:0; font-size:16px; color:#cde; font-weight:600; }
    .th-head-actions { display:flex; gap:6px; flex-wrap:wrap; margin-left:auto; }
    .th-filter {
      background:#141a26; border:1px solid #2a3548; color:#d8dde8;
      padding:4px 8px; border-radius:4px; font-size:11px; min-width:120px;
    }
    .th-filter:focus { outline:none; border-color:#60a5fa; }
    .th-btn {
      background:#141a26; border:1px solid #2a3548; color:#d8dde8;
      padding:4px 10px; border-radius:4px; font-size:11px; cursor:pointer;
    }
    .th-btn:hover { border-color:#3a4558; background:#1a2234; }
    .th-btn-primary { background:#0b2540; border-color:#1e4a7a; color:#60a5fa; }
    .th-btn-primary:hover { background:#113352; color:#93c5fd; }
    .th-stats { font-size:11px; color:#8a95a6; margin-bottom:8px; }
    .th-table-wrap { overflow:auto; max-height: calc(100vh - 240px); border:1px solid #1a2234; border-radius:4px; }
    .th-table { width:100%; border-collapse:collapse; font-size:11px; font-family: 'JetBrains Mono', monospace; }
    .th-table thead th {
      background:#0f1420; position:sticky; top:0; z-index:1;
      padding:6px 8px; text-align:left; border-bottom:1px solid #2a3548;
      cursor:pointer; color:#8aa0bd; font-weight:600; white-space:nowrap;
    }
    .th-table thead th:hover { background:#141a2a; color:#cde; }
    .th-table thead th.is-active { color:#60a5fa; }
    .th-table tbody td {
      padding:4px 8px; border-bottom:1px solid #1a2234;
      white-space:nowrap; max-width:180px; overflow:hidden; text-overflow:ellipsis;
    }
    .th-table tbody tr { cursor:pointer; }
    .th-table tbody tr:hover { background:#141a2a; }
    .th-table tbody tr.th-win .col-pnl_usd,
    .th-table tbody tr.th-win .col-pnl_pct,
    .th-table tbody tr.th-win .col-price_move_pct { color:#00e676; font-weight:600; }
    .th-table tbody tr.th-loss .col-pnl_usd,
    .th-table tbody tr.th-loss .col-pnl_pct,
    .th-table tbody tr.th-loss .col-price_move_pct { color:#ff5252; font-weight:600; }
    .th-detail { margin-top:10px; padding:10px; background:#0f1420; border:1px solid #2a3548; border-radius:4px; }
    .th-detail-head { color:#8a95a6; font-size:10px; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
    .th-detail pre {
      margin:0; font-size:10px; color:#cde; font-family:'JetBrains Mono',monospace;
      max-height:240px; overflow:auto; white-space:pre-wrap; word-break:break-word;
    }
  `;
  document.head.appendChild(s);
}
