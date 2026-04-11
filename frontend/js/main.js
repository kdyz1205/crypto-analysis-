// frontend/js/main.js — clean boot: UI shell first, data async

import { initChart, loadCurrent, startLiveUpdates, toggleMAOverlays } from './workbench/chart.js';
import { initTicker } from './workbench/ticker.js';
import { initTimeframe } from './workbench/timeframe.js';
import { initDecisionRail, refreshDecisionRail } from './workbench/decision_rail.js';
import { initExecutionPanel, togglePanel as toggleExec } from './execution/panel.js';
import { connectStream } from './services/stream.js';
import { initBootStatus, markBoot } from './ui/boot_status.js';
import { $, on } from './util/dom.js';
import { subscribe } from './util/events.js';
import { setScale } from './state/market.js';

function boot() {
  console.log('[main] Trading OS booting...');
  let liveUpdatesStarted = false;
  let railRefreshed = false;
  let streamConnected = false;

  const afterChartLoad = () => {
    markBoot('chart', 'ok', 'loaded');
    if (!liveUpdatesStarted) { startLiveUpdates(30000); liveUpdatesStarted = true; }
    if (!railRefreshed) {
      railRefreshed = true;
      setTimeout(() => refreshDecisionRail().then(() => markBoot('rail', 'ok', 'loaded')).catch(() => {}), 200);
    }
    if (!streamConnected) {
      streamConnected = true;
      setTimeout(() => { try { connectStream(); } catch {} }, 300);
    }
  };

  initBootStatus();

  try { initChart('chart-container'); markBoot('chart', 'pending', 'loading'); } catch (err) { markBoot('chart', 'error', err.message); }
  try { initTimeframe('#v2-tf-group'); } catch {}
  try { initDecisionRail(); } catch {}
  try { initExecutionPanel(); } catch {}

  wireHeaderButtons();
  subscribe('chart.load.succeeded', afterChartLoad);

  initTicker('v2-symbol-select')
    .then(() => markBoot('symbols', 'ok', 'loaded'))
    .catch((err) => console.error('[boot] ticker:', err));

  loadCurrent(true).catch((err) => {
    markBoot('chart', 'error', err.message);
    try { connectStream(); streamConnected = true; } catch {}
  });

  console.log('[main] Trading OS ready');
}

function wireHeaderButtons() {
  // View navigation
  const nav = $('#v2-nav');
  if (nav) {
    nav.addEventListener('click', (e) => {
      const btn = e.target.closest('.v2-nav-btn');
      if (!btn) return;
      const view = btn.dataset.view;
      // Switch active nav button
      nav.querySelectorAll('.v2-nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // Switch active view
      document.querySelectorAll('.v2-view').forEach(v => v.classList.remove('active'));
      const target = document.querySelector(`.v2-view[data-view="${view}"]`);
      if (target) target.classList.add('active');
      // Load view content
      loadView(view);
    });
  }

  on('#v2-ma-toggle', 'click', () => {
    const visible = toggleMAOverlays();
    $('#v2-ma-toggle')?.classList.toggle('active', visible);
  });

  const scaleToggle = $('#v2-scale-toggle');
  if (scaleToggle) {
    scaleToggle.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-scale]');
      if (!btn) return;
      setScale(btn.dataset.scale);
      scaleToggle.querySelectorAll('.v2-scale-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  }

  const setCombatMode = (on) => {
    document.body.classList.toggle('combat-mode', on);
    window.dispatchEvent(new Event('resize'));
  };

  on('#v2-combat-btn', 'click', () => setCombatMode(!document.body.classList.contains('combat-mode')));

  document.addEventListener('keydown', (e) => {
    if (e.key === 'F11' || (e.ctrlKey && e.key === '.')) {
      e.preventDefault();
      setCombatMode(!document.body.classList.contains('combat-mode'));
    }
    if (e.key === 'Escape' && document.body.classList.contains('combat-mode')) {
      if (e.target?.tagName !== 'INPUT' && e.target?.tagName !== 'TEXTAREA') setCombatMode(false);
    }
  });
}

// ── View Loaders ────────────────────────────────────────────────────────

import { fetchJson } from './util/fetch.js';
import { setHtml } from './util/dom.js';
import { formatUsd } from './util/format.js';

const viewLoaded = {};

async function loadView(view) {
  if (view === 'market') return; // market is always loaded

  const container = $(`#view-${view}`);
  if (!container) return;

  if (view === 'factory') loadFactory(container);
  else if (view === 'simulate') loadSimulate(container);
  else if (view === 'leaderboard') loadLeaderboard(container);
  else if (view === 'live') loadLive(container);
  else if (view === 'monitor') loadMonitor(container);
}

async function loadFactory(el) {
  try {
    const { templates } = await fetchJson('/api/runtime/catalog');
    const risk = { low: '🟢低', medium: '🟡中', high: '🔴高' };
    setHtml(el, `
      <div class="view-header"><h2>策略工厂</h2><p class="view-desc">创建和管理策略模板</p></div>
      <div class="view-grid">
        <div class="view-section">
          <h3>📚 策略模板 (${templates.length})</h3>
          ${templates.map(t => `
            <div class="factory-card">
              <div class="factory-card-header">
                <span class="factory-card-name">${t.name}</span>
                <span class="factory-card-risk">${risk[t.risk_level] || ''}</span>
              </div>
              <p class="factory-card-desc">${t.description}</p>
              <div class="factory-card-meta">
                <span>周期: ${t.supported_timeframes.join(', ')}</span>
                <span>RR: ${t.default_params?.rr_target || '动态'}</span>
              </div>
            </div>
          `).join('')}
        </div>
        <div class="view-section">
          <h3>🧪 因子组合器</h3>
          <p class="view-desc">选择因子 → 组合成策略 → 提交模拟</p>
          <div class="factory-composer">
            <div class="composer-group">
              <label>策略类型</label>
              <select id="factory-type">
                <option value="sr_full">S/R 全模式</option>
                <option value="sr_reversal">S/R 反转</option>
                <option value="ma_ribbon">均线开花</option>
                <option value="hft_mm">做市</option>
                <option value="hft_imbalance">盘口失衡</option>
              </select>
            </div>
            <div class="composer-group">
              <label>币种</label>
              <input id="factory-symbol" value="BTCUSDT" placeholder="BTCUSDT" />
            </div>
            <div class="composer-group">
              <label>周期</label>
              <select id="factory-tf">
                <option value="5m">5m</option><option value="15m">15m</option>
                <option value="1h">1h</option><option value="4h" selected>4h</option>
                <option value="1d">1d</option>
              </select>
            </div>
            <div class="composer-group">
              <label>模拟资金</label>
              <input id="factory-equity" type="number" value="10000" />
            </div>
            <button class="view-btn view-btn-primary" onclick="document.dispatchEvent(new CustomEvent('factory-submit'))">提交到模拟中心</button>
          </div>
        </div>
      </div>
    `);
  } catch { setHtml(el, '<p>加载失败</p>'); }
}

async function loadSimulate(el) {
  try {
    const { instances } = await fetchJson('/api/runtime/instances');
    const running = instances.filter(i => i.status.runtime_state === 'running');
    const paper = instances.filter(i => i.config.live_mode !== 'live');
    setHtml(el, `
      <div class="view-header"><h2>模拟中心</h2><p class="view-desc">批量运行策略实验</p></div>
      <div class="view-stats">
        <div class="view-stat"><div class="view-stat-value">${instances.length}</div><div class="view-stat-label">总策略</div></div>
        <div class="view-stat"><div class="view-stat-value">${running.length}</div><div class="view-stat-label">运行中</div></div>
        <div class="view-stat"><div class="view-stat-value">${paper.length}</div><div class="view-stat-label">模拟</div></div>
      </div>
      <div class="view-list">
        ${instances.map(i => {
          const c = i.config, s = i.status;
          const pnl = s.paper_state?.account?.realized_pnl || 0;
          const state = s.runtime_state;
          const stCls = state === 'running' ? 'state-running' : state === 'blocked' ? 'state-blocked' : 'state-stopped';
          return `<div class="sim-row">
            <span class="sim-symbol">${c.symbol}</span>
            <span class="sim-tf">${c.timeframe}</span>
            <span class="sim-state ${stCls}">${state}</span>
            <span class="sim-pnl ${pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${pnl >= 0 ? '+' : ''}${formatUsd(pnl)}</span>
            <span class="sim-bar">bar ${s.last_processed_bar || '-'}</span>
          </div>`;
        }).join('')}
      </div>
    `);
  } catch { setHtml(el, '<p>加载失败</p>'); }
}

async function loadLeaderboard(el) {
  try {
    const data = await fetchJson('/api/runtime/leaderboard?limit=20');
    const lb = data.leaderboard || [];
    setHtml(el, `
      <div class="view-header"><h2>排行榜</h2><p class="view-desc">Gen ${data.generation} | 已测试 ${data.total_tested} | 盈利 ${data.total_profitable}</p></div>
      ${lb.length === 0 ? '<p class="view-desc">暂无结果。运行 python run_evolution.py 启动进化引擎。</p>' : ''}
      <div class="view-list">
        ${lb.map((v, i) => `<div class="lb-row">
          <span class="lb-rank">#${i + 1}</span>
          <span class="lb-sym">${v.symbol} ${v.timeframe}</span>
          <span class="lb-name">${v.name}</span>
          <span class="${v.total_return_pct >= 0 ? 'pnl-pos' : 'pnl-neg'}">${v.total_return_pct >= 0 ? '+' : ''}${v.total_return_pct}%</span>
          <span class="lb-meta">WR ${v.win_rate}% | S ${v.sharpe_ratio} | DD ${v.max_drawdown_pct}%</span>
          <button class="view-btn view-btn-sm" onclick="alert('复制到实盘库 — 即将实现')">加入实盘</button>
        </div>`).join('')}
      </div>
    `);
  } catch { setHtml(el, '<p>加载失败</p>'); }
}

async function loadLive(el) {
  try {
    const { instances } = await fetchJson('/api/runtime/instances');
    const live = instances.filter(i => i.config.live_mode === 'live');
    let accountHtml = '';
    try {
      const acc = await fetchJson('/api/live-execution/account?mode=live');
      if (acc.ok) accountHtml = `<div class="view-stats">
        <div class="view-stat"><div class="view-stat-value">${formatUsd(acc.total_equity)}</div><div class="view-stat-label">权益</div></div>
        <div class="view-stat"><div class="view-stat-value">${formatUsd(acc.usdt_available)}</div><div class="view-stat-label">可用</div></div>
        <div class="view-stat"><div class="view-stat-value">${acc.positions?.length || 0}</div><div class="view-stat-label">持仓</div></div>
      </div>`;
    } catch {}
    setHtml(el, `
      <div class="view-header"><h2>实盘策略库</h2><p class="view-desc">Bitget 实盘 | ${live.length} 个策略</p></div>
      ${accountHtml}
      <div class="view-list">
        ${live.length === 0 ? '<p class="view-desc">暂无实盘策略。从排行榜选择优胜策略加入。</p>' : ''}
        ${live.map(i => {
          const c = i.config, s = i.status;
          return `<div class="live-row">
            <span class="live-symbol">${c.symbol}</span>
            <span class="live-tf">${c.timeframe}</span>
            <span class="sim-state ${s.runtime_state === 'running' ? 'state-running' : 'state-stopped'}">${s.runtime_state}</span>
            <span class="live-bar">bar ${s.last_processed_bar || '-'}</span>
            <span class="live-submit">${c.auto_live_submit ? '🟢自动下单' : '⚪仅预览'}</span>
          </div>`;
        }).join('')}
      </div>
    `);
  } catch { setHtml(el, '<p>加载失败</p>'); }
}

async function loadMonitor(el) {
  try {
    const { instances } = await fetchJson('/api/runtime/instances');
    const running = instances.filter(i => i.status.runtime_state === 'running');
    setHtml(el, `
      <div class="view-header"><h2>运行监控</h2><p class="view-desc">${running.length} 个策略活跃</p></div>
      <div class="view-list">
        ${running.map(i => {
          const c = i.config, s = i.status;
          const pnl = s.paper_state?.account?.realized_pnl || 0;
          const isLive = c.live_mode === 'live';
          return `<div class="monitor-row">
            <span class="monitor-badge ${isLive ? 'badge-live' : 'badge-paper'}">${isLive ? '实盘' : '模拟'}</span>
            <span class="monitor-sym">${c.symbol} ${c.timeframe}</span>
            <span class="monitor-bar">bar ${s.last_processed_bar || '-'}</span>
            <span class="${pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${pnl >= 0 ? '+' : ''}${formatUsd(pnl)}</span>
            <span class="monitor-update">${s.last_tick_at ? '活跃' : '等待'}</span>
          </div>`;
        }).join('')}
      </div>
    `);
  } catch { setHtml(el, '<p>加载失败</p>'); }
}

document.addEventListener('DOMContentLoaded', boot);
