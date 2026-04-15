// frontend/js/views/runner_view.js
//
// Dashboard for the MA Ribbon + Trendline Bounce live runner.
//
// Sections:
//   1. Status bar       — status pill + start/stop/kick buttons
//   2. Config form      — editable (top_n, tf, max_pos, notional, lev, strategies)
//   3. Live positions   — symbol, side, entry, mark, unrealized PnL, strategy
//   4. Session stats    — scans, signals, orders, rejects, free slots
//   5. Recent signals   — last 20 (including rejected ones)
//
// Polls /api/mar-bb/state + /api/live-execution/account every 3 seconds.
// Data freshness indicator in the status bar.

import { esc } from '../util/dom.js';
import { fetchJson } from '../util/fetch.js';

let _pollTimer = null;
let _historyTimer = null;
let _host = null;
let _lastState = null;
let _lastAccount = null;
let _lastHistory = null;
let _historyDays = 30;

export async function loadRunner(el) {
  _host = el;
  stopPolling();
  el.innerHTML = renderShell();
  injectStyles();
  wire(el);
  await refresh();
  await refreshHistory();
  _pollTimer = setInterval(refresh, 3000);
  _historyTimer = setInterval(refreshHistory, 30000);   // history reloads every 30s
}

export function unloadRunner() {
  stopPolling();
  _host = null;
}

function stopPolling() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
  if (_historyTimer) { clearInterval(_historyTimer); _historyTimer = null; }
}

async function refreshHistory() {
  if (!_host) return;
  try {
    const resp = await fetchJson(`/api/mar-bb/history?days=${_historyDays}`, {
      noCache: true, timeout: 30000,
    }).catch(() => null);
    _lastHistory = resp?.stats || null;
    renderHistory();
  } catch (err) {
    console.warn('[runner_view] history err', err);
  }
}

async function refresh() {
  if (!_host) return;
  try {
    const [stateResp, acctResp] = await Promise.all([
      fetchJson('/api/mar-bb/state', { noCache: true }).catch(() => null),
      fetchJson('/api/live-execution/account?mode=live', { noCache: true }).catch(() => null),
    ]);
    _lastState = stateResp?.state || null;
    _lastAccount = acctResp || null;
    render();
  } catch (err) {
    console.warn('[runner_view] refresh err', err);
  }
}

function render() {
  if (!_host) return;
  const s = _lastState || {};
  const a = _lastAccount || {};

  const statusEl = _host.querySelector('#rn-status-pill');
  if (statusEl) {
    const status = s.status || 'idle';
    statusEl.textContent = status;
    statusEl.className = `rn-status-pill rn-status-${status}`;
  }

  const metaEl = _host.querySelector('#rn-meta');
  if (metaEl) {
    const lastScan = s.last_scan_ts
      ? new Date(s.last_scan_ts * 1000).toLocaleTimeString()
      : '—';
    const freshness = s.last_scan_ts
      ? Math.max(0, Math.floor(Date.now() / 1000 - s.last_scan_ts))
      : '—';
    metaEl.innerHTML =
      `scans: <b>${s.scans_completed ?? 0}</b> · ` +
      `dur: <b>${s.last_scan_duration_s ?? 0}s</b> · ` +
      `last: <b>${lastScan}</b> (<span class="rn-age">${freshness}s ago</span>)`;
  }

  // Config form — only fill on first render or when user isn't editing
  const cfgRoot = _host.querySelector('#rn-config');
  if (cfgRoot && !cfgRoot.dataset.dirty) {
    const cfg = s.config || {};
    _host.querySelector('[name=top_n]').value = cfg.top_n ?? 100;
    _host.querySelector('[name=timeframe]').value = cfg.timeframe ?? '1h';
    _host.querySelector('[name=scan_interval_s]').value = cfg.scan_interval_s ?? 60;
    _host.querySelector('[name=notional_usd]').value = cfg.notional_usd ?? 12;
    _host.querySelector('[name=leverage]').value = cfg.leverage ?? 5;
    _host.querySelector('[name=max_concurrent_positions]').value = cfg.max_concurrent_positions ?? 5;
    const strats = cfg.strategies || [];
    _host.querySelector('[name=strat_mar_bb]').checked = strats.includes('mar_bb');
    _host.querySelector('[name=strat_trendline]').checked = strats.includes('trendline');
    _host.querySelector('[name=dry_run]').checked = !!cfg.dry_run;
  }

  // Stats row
  const statsEl = _host.querySelector('#rn-stats');
  if (statsEl) {
    const maxPos = s.config?.max_concurrent_positions ?? 5;
    const held = (a.positions || []).length;
    const free = Math.max(0, maxPos - held);
    statsEl.innerHTML = `
      <div class="rn-stat">
        <div class="rn-stat-label">账户</div>
        <div class="rn-stat-value">$${Number(a.total_equity || 0).toFixed(2)}</div>
        <div class="rn-stat-sub">可用 $${Number(a.usdt_available || 0).toFixed(2)}</div>
      </div>
      <div class="rn-stat">
        <div class="rn-stat-label">仓位</div>
        <div class="rn-stat-value">${held} / ${maxPos}</div>
        <div class="rn-stat-sub">${free} 空闲</div>
      </div>
      <div class="rn-stat">
        <div class="rn-stat-label">识别信号</div>
        <div class="rn-stat-value">${s.signals_detected ?? 0}</div>
        <div class="rn-stat-sub">成交 ${s.orders_submitted ?? 0}</div>
      </div>
      <div class="rn-stat">
        <div class="rn-stat-label">被拒</div>
        <div class="rn-stat-value ${(s.orders_rejected||0) > 0 ? 'is-bad' : ''}">${s.orders_rejected ?? 0}</div>
        <div class="rn-stat-sub">${s.last_error ? '最后: ' + truncate(s.last_error, 28) : ''}</div>
      </div>
    `;
  }

  // Positions table
  const posEl = _host.querySelector('#rn-positions');
  if (posEl) {
    const positions = a.positions || [];
    if (!positions.length) {
      posEl.innerHTML = `<div class="rn-empty">无持仓</div>`;
    } else {
      const rows = positions.map(p => {
        const sym = p.symbol || '?';
        const side = (p.holdSide || '').toLowerCase();
        const size = Number(p.total || 0);
        const avg = Number(p.averageOpenPrice || 0);
        const mark = Number(p.markPrice || 0);
        const notional = size * mark;
        const pnl = side === 'long'
          ? (mark - avg) * size
          : (avg - mark) * size;
        const pnlPct = avg > 0 ? ((pnl / (size * avg)) * 100) : 0;
        const cls = pnl > 0 ? 'is-up' : (pnl < 0 ? 'is-dn' : '');
        return `
          <tr>
            <td class="rn-pos-sym">${esc(sym)}</td>
            <td class="rn-pos-side rn-side-${side}">${esc(side).toUpperCase()}</td>
            <td>${size.toFixed(size < 1 ? 4 : 2)}</td>
            <td>${avg.toFixed(avg < 10 ? 4 : 2)}</td>
            <td>${mark.toFixed(mark < 10 ? 4 : 2)}</td>
            <td>$${notional.toFixed(2)}</td>
            <td class="${cls}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(4)} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)</td>
            <td>
              <button class="rn-btn-sm rn-btn-close" data-action="close-pos" data-symbol="${esc(sym)}" data-side="${esc(side)}" data-size="${size}">平仓</button>
            </td>
          </tr>
        `;
      }).join('');
      posEl.innerHTML = `
        <table class="rn-table">
          <thead>
            <tr>
              <th>Symbol</th><th>方向</th><th>数量</th><th>均价</th><th>Mark</th>
              <th>名义</th><th>PnL</th><th></th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }
  }

  // Recent signals feed
  const sigEl = _host.querySelector('#rn-signals');
  if (sigEl) {
    const signals = s.recent_signals || [];
    if (!signals.length) {
      sigEl.innerHTML = `<div class="rn-empty">等待信号…</div>`;
    } else {
      sigEl.innerHTML = signals.map(sg => {
        const ts = sg.ts ? new Date(sg.ts * 1000).toLocaleTimeString() : '';
        const strat = sg.strategy || '?';
        const dir = sg.direction || '?';
        const dirCls = dir === 'long' ? 'is-up' : 'is-dn';
        return `
          <div class="rn-signal-row">
            <span class="rn-sig-ts">${esc(ts)}</span>
            <span class="rn-sig-strat rn-strat-${esc(strat)}">${esc(strat)}</span>
            <span class="rn-sig-sym">${esc(sg.symbol || '')}</span>
            <span class="rn-sig-dir ${dirCls}">${esc(dir).toUpperCase()}</span>
            <span class="rn-sig-price">@ ${Number(sg.entry || 0).toFixed(sg.entry < 10 ? 4 : 2)}</span>
            <span class="rn-sig-sl">sl ${Number(sg.stop || 0).toFixed(sg.stop < 10 ? 4 : 2)}</span>
            <span class="rn-sig-tp">tp ${Number(sg.tp || 0).toFixed(sg.tp < 10 ? 4 : 2)}</span>
          </div>
        `;
      }).join('');
    }
  }
}

function renderHistory() {
  if (!_host) return;
  const h = _lastHistory;
  if (!h) return;
  const o = h.overall || {};
  const byStrat = h.by_strategy || {};
  const daily = h.daily || [];
  const weekly = h.weekly || [];
  const bySym = h.by_symbol || [];

  const hostEl = _host.querySelector('#rn-history');
  if (!hostEl) return;

  const netCls = (o.net_pnl || 0) >= 0 ? 'is-up' : 'is-dn';
  const fmt = (n) => (n >= 0 ? '+' : '') + '$' + Number(n || 0).toFixed(2);

  // ─── Overall header row ───
  let html = `
    <div class="rn-hist-head">
      <div class="rn-hist-range">
        <label>窗口:</label>
        <select id="rn-hist-days">
          ${[7, 14, 30, 60, 90].map(d =>
            `<option value="${d}"${d === _historyDays ? ' selected' : ''}>${d} 天</option>`).join('')}
        </select>
      </div>
      <div class="rn-hist-kpi-row">
        <div class="rn-hist-kpi">
          <div class="rn-hist-kpi-label">总交易</div>
          <div class="rn-hist-kpi-value">${o.trades || 0}</div>
          <div class="rn-hist-kpi-sub">${o.wins || 0}W / ${o.losses || 0}L</div>
        </div>
        <div class="rn-hist-kpi">
          <div class="rn-hist-kpi-label">胜率</div>
          <div class="rn-hist-kpi-value">${(o.winrate || 0).toFixed(1)}%</div>
          <div class="rn-hist-kpi-sub">平均赢 ${fmt(o.avg_win)} · 亏 ${fmt(o.avg_loss)}</div>
        </div>
        <div class="rn-hist-kpi">
          <div class="rn-hist-kpi-label">毛盈亏</div>
          <div class="rn-hist-kpi-value">${fmt(o.total_pnl)}</div>
          <div class="rn-hist-kpi-sub">手续费 $${Number(o.total_fees || 0).toFixed(3)}</div>
        </div>
        <div class="rn-hist-kpi">
          <div class="rn-hist-kpi-label">净盈亏</div>
          <div class="rn-hist-kpi-value ${netCls}">${fmt(o.net_pnl)}</div>
          <div class="rn-hist-kpi-sub">${
            o.best ? `最好 ${o.best.symbol} ${fmt(o.best.pnl)}` : ''
          }</div>
        </div>
      </div>
    </div>
  `;

  // ─── PnL cumulative curve (SVG sparkline) ───
  if (daily.length >= 2) {
    html += renderSparkline(daily);
  } else if (daily.length === 1) {
    html += `<div class="rn-empty">只有 1 天数据,无法画曲线</div>`;
  }

  // ─── By strategy ───
  if (Object.keys(byStrat).length > 0) {
    html += `<div class="rn-hist-section-title">按策略</div>`;
    html += `<table class="rn-table rn-hist-strat-table"><thead>
      <tr>
        <th>策略</th><th>交易数</th><th>胜率</th>
        <th>毛盈亏</th><th>手续费</th><th>净盈亏</th>
      </tr></thead><tbody>`;
    for (const [name, v] of Object.entries(byStrat)) {
      const strNetCls = (v.net_pnl || 0) >= 0 ? 'is-up' : 'is-dn';
      html += `
        <tr>
          <td><span class="rn-sig-strat rn-strat-${esc(name)}">${esc(name)}</span></td>
          <td>${v.trades}</td>
          <td>${(v.winrate || 0).toFixed(1)}%</td>
          <td>${fmt(v.total_pnl)}</td>
          <td>$${Number(v.total_fees || 0).toFixed(3)}</td>
          <td class="${strNetCls}">${fmt(v.net_pnl)}</td>
        </tr>
      `;
    }
    html += `</tbody></table>`;
  }

  // ─── Weekly rollup ───
  if (weekly.length > 0) {
    html += `<div class="rn-hist-section-title">周</div>`;
    html += `<table class="rn-table"><thead>
      <tr><th>Week</th><th>开始日</th><th>交易数</th><th>胜</th><th>毛</th><th>费</th><th>净</th></tr>
    </thead><tbody>`;
    for (const w of weekly.slice(-12)) {
      const wCls = (w.net || 0) >= 0 ? 'is-up' : 'is-dn';
      html += `
        <tr>
          <td>${esc(w.week)}</td>
          <td>${esc(w.week_start)}</td>
          <td>${w.trades}</td>
          <td>${w.wins}</td>
          <td>${fmt(w.pnl)}</td>
          <td>$${Number(w.fee).toFixed(3)}</td>
          <td class="${wCls}">${fmt(w.net)}</td>
        </tr>
      `;
    }
    html += `</tbody></table>`;
  }

  // ─── Per-symbol top 10 ───
  if (bySym.length > 0) {
    html += `<div class="rn-hist-section-title">按 symbol (Top 10)</div>`;
    html += `<table class="rn-table"><thead>
      <tr><th>Symbol</th><th>交易数</th><th>胜率</th><th>毛</th><th>费</th><th>净</th></tr>
    </thead><tbody>`;
    for (const s of bySym) {
      const sCls = (s.net || 0) >= 0 ? 'is-up' : 'is-dn';
      html += `
        <tr>
          <td class="rn-pos-sym">${esc(s.symbol)}</td>
          <td>${s.trades}</td>
          <td>${(s.winrate || 0).toFixed(1)}%</td>
          <td>${fmt(s.pnl)}</td>
          <td>$${Number(s.fee).toFixed(3)}</td>
          <td class="${sCls}">${fmt(s.net)}</td>
        </tr>
      `;
    }
    html += `</tbody></table>`;
  }

  hostEl.innerHTML = html;

  // Wire window-size selector
  const daysSel = hostEl.querySelector('#rn-hist-days');
  if (daysSel) {
    daysSel.addEventListener('change', async () => {
      _historyDays = Number(daysSel.value) || 30;
      await refreshHistory();
    });
  }
}

function renderSparkline(daily) {
  const W = 720, H = 120, PAD = 8;
  const cumVals = daily.map(d => d.cumulative || 0);
  const minY = Math.min(0, ...cumVals);
  const maxY = Math.max(0, ...cumVals);
  const rangeY = (maxY - minY) || 1;
  const n = daily.length;
  const stepX = (W - PAD * 2) / Math.max(1, n - 1);
  const points = daily.map((d, i) => {
    const x = PAD + i * stepX;
    const y = PAD + (H - PAD * 2) * (1 - (d.cumulative - minY) / rangeY);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  // zero line
  const zeroY = PAD + (H - PAD * 2) * (1 - (0 - minY) / rangeY);
  const finalCum = cumVals[cumVals.length - 1] || 0;
  const lineColor = finalCum >= 0 ? '#00e676' : '#ff5252';
  const fillColor = finalCum >= 0 ? 'rgba(0,230,118,0.12)' : 'rgba(255,82,82,0.12)';

  // Build area path under the line, closed to zero line
  const firstX = PAD;
  const lastX = PAD + (n - 1) * stepX;
  const areaPath = `M ${firstX},${zeroY} L ${points.split(' ').map((p, i) => {
    const [x, y] = p.split(',');
    return `${x},${y}`;
  }).join(' L ')} L ${lastX},${zeroY} Z`;

  return `
    <div class="rn-spark">
      <div class="rn-spark-title">累积净盈亏曲线 (${daily.length} 天)</div>
      <svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
        <line x1="${PAD}" y1="${zeroY}" x2="${W - PAD}" y2="${zeroY}"
              stroke="#2a3548" stroke-width="1" stroke-dasharray="2,3"/>
        <path d="${areaPath}" fill="${fillColor}" stroke="none"/>
        <polyline points="${points}" fill="none" stroke="${lineColor}" stroke-width="2"/>
      </svg>
      <div class="rn-spark-axis">
        <span>${daily[0].date}</span>
        <span>最终: ${finalCum >= 0 ? '+' : ''}$${finalCum.toFixed(2)}</span>
        <span>${daily[daily.length - 1].date}</span>
      </div>
    </div>
  `;
}

function wire(el) {
  // Start / Apply Config (hot-update if running)
  el.querySelector('#rn-btn-start').addEventListener('click', async () => {
    const cfg = readConfig(el);
    try {
      const running = _lastState?.status === 'running';
      if (running) {
        // Hot-update: config merges into the running loop, no restart
        const resp = await fetchJson('/api/mar-bb/update-config', {
          method: 'POST', body: cfg,
        });
        if (!resp?.ok) throw new Error(resp?.reason || 'update failed');
        const keys = (resp.updated_keys || []).join(', ');
        flashToast(el, `已实时更新: ${keys || '(无变化)'}`);
      } else {
        // Cold start: initialize the runner loop
        await fetchJson('/api/mar-bb/start', { method: 'POST', body: cfg });
        flashToast(el, '已启动 runner');
      }
      el.querySelector('#rn-config').removeAttribute('data-dirty');
      await refresh();
    } catch (err) {
      alert(`配置失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
    }
  });
  el.querySelector('#rn-btn-stop').addEventListener('click', async () => {
    if (!confirm('停止 runner? 已开仓位不会被平,但不再识别新信号。')) return;
    try {
      await fetchJson('/api/mar-bb/stop', { method: 'POST' });
      await refresh();
    } catch (err) {
      alert(`停止失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
    }
  });
  el.querySelector('#rn-btn-kick').addEventListener('click', async () => {
    el.querySelector('#rn-btn-kick').disabled = true;
    try {
      await fetchJson('/api/mar-bb/kick', { method: 'POST', timeout: 180000 });
      await refresh();
    } catch (err) {
      alert(`扫描失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
    } finally {
      el.querySelector('#rn-btn-kick').disabled = false;
    }
  });

  // Track config edits so refresh doesn't overwrite mid-type
  el.querySelector('#rn-config').addEventListener('input', () => {
    el.querySelector('#rn-config').dataset.dirty = '1';
  });
  el.querySelector('#rn-config').addEventListener('change', () => {
    el.querySelector('#rn-config').dataset.dirty = '1';
  });

  // Position close (delegated)
  el.querySelector('#rn-positions').addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-action=close-pos]');
    if (!btn) return;
    const sym = btn.dataset.symbol;
    const side = btn.dataset.side;
    if (!confirm(`平仓 ${sym} ${side}?`)) return;
    btn.disabled = true;
    btn.textContent = '平仓中...';
    try {
      // Use live-execution close endpoint
      const resp = await fetchJson('/api/live-execution/close', {
        method: 'POST',
        body: { symbol: sym, side: side, mode: 'live' },
        timeout: 15000,
      });
      if (!resp?.ok) throw new Error(resp?.reason || resp?.error || 'close failed');
      await refresh();
    } catch (err) {
      alert(`平仓失败: ${err?.message || err}`);  // SAFE: alert() renders text, not HTML
      btn.disabled = false;
      btn.textContent = '平仓';
    }
  });
}

function readConfig(el) {
  return {
    top_n: Number(el.querySelector('[name=top_n]').value) || 100,
    timeframe: el.querySelector('[name=timeframe]').value,
    scan_interval_s: Number(el.querySelector('[name=scan_interval_s]').value) || 60,
    notional_usd: Number(el.querySelector('[name=notional_usd]').value) || 12,
    leverage: Number(el.querySelector('[name=leverage]').value) || 5,
    max_concurrent_positions: Number(el.querySelector('[name=max_concurrent_positions]').value) || 5,
    dry_run: el.querySelector('[name=dry_run]').checked,
    strategies: [
      el.querySelector('[name=strat_mar_bb]').checked ? 'mar_bb' : null,
      el.querySelector('[name=strat_trendline]').checked ? 'trendline' : null,
    ].filter(Boolean),
  };
}

function renderShell() {
  return `
    <div class="rn-root">
      <div class="rn-header">
        <div class="rn-title">
          <span>策略 Runner</span>
          <span class="rn-status-pill rn-status-idle" id="rn-status-pill">idle</span>
        </div>
        <div class="rn-actions">
          <button class="rn-btn rn-btn-primary" id="rn-btn-start">启动 / 应用配置</button>
          <button class="rn-btn rn-btn-kick" id="rn-btn-kick">立即扫描</button>
          <button class="rn-btn rn-btn-danger" id="rn-btn-stop">停止</button>
        </div>
      </div>
      <div class="rn-meta" id="rn-meta"></div>

      <div class="rn-section">
        <div class="rn-section-head">配置</div>
        <form id="rn-config" class="rn-config" onsubmit="return false">
          <label class="rn-field">
            <span>策略</span>
            <div class="rn-strat-checks">
              <label><input type="checkbox" name="strat_mar_bb"/> MA Ribbon</label>
              <label><input type="checkbox" name="strat_trendline"/> Trendline</label>
            </div>
          </label>
          <label class="rn-field">
            <span>扫描 symbol 数</span>
            <input type="number" name="top_n" min="1" max="200" />
          </label>
          <label class="rn-field">
            <span>Timeframe</span>
            <select name="timeframe">
              <option value="15m">15m</option>
              <option value="1h">1h</option>
              <option value="4h">4h</option>
              <option value="1d">1d</option>
            </select>
          </label>
          <label class="rn-field">
            <span>扫描间隔 (秒)</span>
            <input type="number" name="scan_interval_s" min="10" max="3600" />
          </label>
          <label class="rn-field">
            <span>每单名义 ($)</span>
            <input type="number" name="notional_usd" min="5" step="1" />
          </label>
          <label class="rn-field">
            <span>杠杆</span>
            <input type="number" name="leverage" min="1" max="50" />
          </label>
          <label class="rn-field">
            <span>最大并发</span>
            <input type="number" name="max_concurrent_positions" min="1" max="20" />
          </label>
          <label class="rn-field rn-field-check">
            <input type="checkbox" name="dry_run"/>
            <span>Dry run (只 log 信号,不实际下单)</span>
          </label>
        </form>
      </div>

      <div class="rn-section">
        <div class="rn-section-head">统计</div>
        <div class="rn-stats-grid" id="rn-stats"></div>
      </div>

      <div class="rn-section">
        <div class="rn-section-head">实时仓位</div>
        <div id="rn-positions" class="rn-positions"></div>
      </div>

      <div class="rn-section">
        <div class="rn-section-head">最近信号 (最多 20 条)</div>
        <div id="rn-signals" class="rn-signals"></div>
      </div>

      <div class="rn-section">
        <div class="rn-section-head">历史成交统计</div>
        <div id="rn-history"></div>
      </div>
    </div>
  `;
}

function truncate(s, n) {
  s = String(s || '');
  return s.length > n ? s.slice(0, n) + '…' : s;
}

function flashToast(host, msg) {
  let t = host.querySelector('.rn-toast');
  if (!t) {
    t = document.createElement('div');
    t.className = 'rn-toast';
    host.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add('rn-toast-visible');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('rn-toast-visible'), 2500);
}

function injectStyles() {
  if (document.getElementById('rn-runner-styles')) return;
  const style = document.createElement('style');
  style.id = 'rn-runner-styles';
  style.textContent = `
    .rn-root { padding: 20px 24px; color: #d8dde8; font-size: 13px; max-width: 1400px; }
    .rn-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .rn-title { display: flex; align-items: center; gap: 12px; font-size: 18px; font-weight: 700; }
    .rn-status-pill {
      padding: 3px 12px; border-radius: 999px; font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.06em;
    }
    .rn-status-running { background: rgba(0, 230, 118, 0.18); color: #00e676; }
    .rn-status-stopped { background: rgba(255, 82, 82, 0.15); color: #ff5252; }
    .rn-status-idle    { background: rgba(138, 149, 166, 0.2); color: #8a95a6; }
    .rn-status-error   { background: rgba(255, 152, 0, 0.22); color: #ff9800; }
    .rn-actions { display: flex; gap: 8px; }
    .rn-btn {
      background: #1d2537; border: 1px solid #2a3548; color: #d8dde8;
      padding: 7px 18px; border-radius: 4px; cursor: pointer;
      font-size: 12px; font-weight: 600;
    }
    .rn-btn:hover { background: #24304a; }
    .rn-btn-primary { background: #0284c7; border-color: #0284c7; color: white; }
    .rn-btn-primary:hover { background: #0369a1; }
    .rn-btn-kick { background: #14532d; border-color: #166534; color: #86efac; }
    .rn-btn-kick:hover { background: #166534; }
    .rn-btn-danger { background: #450a0a; border-color: #7f1d1d; color: #fca5a5; }
    .rn-btn-danger:hover { background: #7f1d1d; }
    .rn-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .rn-btn-sm { padding: 3px 10px; font-size: 11px; }
    .rn-btn-close { background: #450a0a; border: 1px solid #7f1d1d; color: #fca5a5;
                    border-radius: 3px; cursor: pointer; }
    .rn-btn-close:hover { background: #7f1d1d; }

    .rn-meta { color: #8a95a6; font-size: 11px; margin-bottom: 20px; }
    .rn-meta b { color: #d8dde8; }
    .rn-age { color: #60a5fa; }

    .rn-section {
      background: #0e141f;
      border: 1px solid #1d2537;
      border-radius: 6px;
      padding: 14px 18px;
      margin-bottom: 14px;
    }
    .rn-section-head {
      font-size: 11px; font-weight: 700; color: #38bdf8;
      text-transform: uppercase; letter-spacing: 0.06em;
      margin-bottom: 12px;
    }

    .rn-config {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px 16px;
    }
    .rn-field { display: flex; flex-direction: column; gap: 4px; }
    .rn-field > span { font-size: 11px; color: #8a95a6; font-weight: 600; }
    .rn-field input[type=number], .rn-field select {
      background: #141a26; border: 1px solid #2a3548; color: #d8dde8;
      padding: 6px 10px; border-radius: 4px; font-size: 13px; width: 100%;
      box-sizing: border-box;
    }
    .rn-field-check {
      flex-direction: row; align-items: center; gap: 6px;
      grid-column: span 2;
    }
    .rn-strat-checks { display: flex; gap: 12px; align-items: center; }
    .rn-strat-checks label { display: flex; align-items: center; gap: 4px; cursor: pointer; }

    .rn-stats-grid {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    }
    .rn-stat {
      background: #141a26; border: 1px solid #1d2537; border-radius: 4px;
      padding: 12px 14px;
    }
    .rn-stat-label { font-size: 10px; color: #8a95a6; text-transform: uppercase; }
    .rn-stat-value { font-size: 20px; font-weight: 700; color: #e8edf5; margin: 4px 0; }
    .rn-stat-value.is-bad { color: #ff5252; }
    .rn-stat-sub { font-size: 11px; color: #6b7889; }

    .rn-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .rn-table th { text-align: left; color: #6b7889; font-weight: 600;
                   padding: 8px 10px; border-bottom: 1px solid #1d2537; }
    .rn-table td { padding: 10px; border-bottom: 1px solid #141a26; }
    .rn-pos-sym { font-weight: 700; color: #d8dde8; }
    .rn-pos-side { font-weight: 700; font-size: 11px; }
    .rn-side-long  { color: #00e676; }
    .rn-side-short { color: #ff5252; }
    .is-up { color: #00e676; }
    .is-dn { color: #ff5252; }
    .rn-empty { color: #6b7889; font-size: 12px; padding: 16px; text-align: center; }

    .rn-signals { max-height: 400px; overflow-y: auto; }
    .rn-signal-row {
      display: grid;
      grid-template-columns: 85px 70px 100px 60px 100px 100px 100px;
      gap: 8px; padding: 7px 4px; font-size: 11px;
      border-bottom: 1px solid #141a26;
      font-family: ui-monospace, Menlo, monospace;
    }
    .rn-sig-ts   { color: #6b7889; }
    .rn-sig-strat {
      font-weight: 700; font-size: 10px; text-transform: uppercase;
      padding: 2px 6px; border-radius: 3px; text-align: center;
    }
    .rn-strat-mar_bb     { background: rgba(56, 189, 248, 0.18); color: #38bdf8; }
    .rn-strat-trendline  { background: rgba(251, 191, 36, 0.18); color: #fbbf24; }
    .rn-sig-sym  { color: #d8dde8; font-weight: 600; }
    .rn-sig-dir  { font-weight: 700; }
    .rn-sig-price, .rn-sig-sl, .rn-sig-tp { color: #8a95a6; }

    /* History section */
    .rn-hist-head {
      display: flex; justify-content: space-between; align-items: flex-start;
      gap: 20px; margin-bottom: 14px;
    }
    .rn-hist-range { display: flex; align-items: center; gap: 6px; }
    .rn-hist-range label { font-size: 11px; color: #8a95a6; }
    .rn-hist-range select {
      background: #141a26; border: 1px solid #2a3548; color: #d8dde8;
      padding: 4px 8px; border-radius: 3px; font-size: 12px;
    }
    .rn-hist-kpi-row {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; flex: 1;
    }
    .rn-hist-kpi {
      background: #141a26; border: 1px solid #1d2537; border-radius: 4px;
      padding: 10px 12px;
    }
    .rn-hist-kpi-label { font-size: 10px; color: #8a95a6; text-transform: uppercase; }
    .rn-hist-kpi-value { font-size: 18px; font-weight: 700; color: #e8edf5; margin: 3px 0; }
    .rn-hist-kpi-sub { font-size: 10px; color: #6b7889; }

    .rn-hist-section-title {
      font-size: 10px; font-weight: 700; color: #8a95a6;
      text-transform: uppercase; letter-spacing: 0.06em;
      margin: 18px 0 8px;
    }
    .rn-hist-strat-table td:first-child { padding: 6px 10px; }

    .rn-spark {
      background: #141a26; border: 1px solid #1d2537; border-radius: 4px;
      padding: 14px; margin: 16px 0;
    }
    .rn-spark-title { font-size: 11px; color: #8a95a6; margin-bottom: 8px; }
    .rn-spark svg { display: block; width: 100%; height: auto; max-width: 100%; }
    .rn-spark-axis {
      display: flex; justify-content: space-between;
      font-size: 10px; color: #6b7889; margin-top: 6px;
      font-family: ui-monospace, Menlo, monospace;
    }

    /* Toast */
    .rn-toast {
      position: fixed; bottom: 28px; right: 28px;
      background: #0284c7; color: white;
      padding: 10px 18px; border-radius: 4px;
      font-size: 12px; font-weight: 600;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
      opacity: 0; transform: translateY(10px);
      transition: opacity 180ms, transform 180ms;
      pointer-events: none; z-index: 2000;
    }
    .rn-toast-visible { opacity: 1; transform: translateY(0); }
  `;
  document.head.appendChild(style);
}
