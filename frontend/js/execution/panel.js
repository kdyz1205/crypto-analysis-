// frontend/js/execution/panel.js — v5 correct architecture
// Flow: 策略库(browse) → 因子组合(compose) → 排行榜(rank) → 运行中(monitor)

import { $, setHtml, on, show, hide } from '../util/dom.js';
import { publish } from '../util/events.js';
import { marketState, setSymbol, setIntervalTF } from '../state/market.js';
import * as paperSvc from '../services/paper_execution.js';
import * as runtimeSvc from '../services/runtime.js';
import * as liveSvc from '../services/live_execution.js';
import { formatUsd, pnlColorClass } from '../util/format.js';

const POLL_MS = 15000;
let pollTimer = null, panelOpen = false, accountMode = 'paper';
let expandedId = null, expandedCatalogId = null, showInactive = false;
let catalog = [], catalogLoaded = false, leaderboard = [];
let paperState = null, liveAccount = null, liveStatus = null, instances = [];
let liveStatusCacheTime = 0, liveAccountCacheTime = 0, renderQueued = false;
const CACHE_TTL = 30000;

// ── Public ──────────────────────────────────────────────────────────────
export function initExecutionPanel() { buildShell(); wireEvents(); }
export function openPanel() { panelOpen = true; show('#v2-execution-panel'); refreshAll(); startPolling(); }
export function closePanel() { panelOpen = false; hide('#v2-execution-panel'); stopPolling(); }
export function togglePanel() { panelOpen ? closePanel() : openPanel(); }

// ── Shell ───────────────────────────────────────────────────────────────
function buildShell() {
  const root = $('#v2-execution-panel');
  if (!root) return;
  setHtml(root, `
    <div class="exec-header"><h3>交易面板</h3><button class="exec-close" id="v2-exec-close">&times;</button></div>
    <div class="exec-account-switcher">
      <button class="exec-acct-btn active" data-mode="paper">模拟</button>
      <button class="exec-acct-btn" data-mode="live">Bitget</button>
    </div>
    <div class="exec-body" id="exec-body"><div class="exec-loading">加载中...</div></div>`);
}

// ── Events ──────────────────────────────────────────────────────────────
function wireEvents() {
  on('#v2-exec-close', 'click', () => closePanel());
  const root = $('#v2-execution-panel');
  if (!root) return;

  // Coin search in factor composer
  root.addEventListener('input', (e) => {
    if (!e.target.matches('[data-role="coin-filter"]')) return;
    const q = e.target.value.toUpperCase().trim();
    if (q.length < 2) return;
    const box = e.target.closest('.exec-composer')?.querySelector('.exec-coin-chips');
    if (!box) return;
    const have = new Set([...box.querySelectorAll('[data-coin]')].map(cb => cb.dataset.coin));
    (marketState.allSymbols || []).filter(c => c.includes(q) && !have.has(c)).slice(0, 20).forEach(coin => {
      const lbl = document.createElement('label');
      lbl.className = 'exec-chip';
      lbl.innerHTML = `<input type="checkbox" data-coin="${coin}" checked/> ${coin.replace('USDT','')}`;
      box.appendChild(lbl);
    });
  });

  root.addEventListener('click', (e) => {
    const t = e.target;

    // Account switch
    const ab = t.closest('.exec-acct-btn');
    if (ab) { accountMode = ab.dataset.mode; root.querySelectorAll('.exec-acct-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === accountMode)); qRender(); refreshAll(); return; }

    // Expand strategy card
    const hdr = t.closest('.exec-strategy-header');
    if (hdr && !t.closest('button')) {
      const id = hdr.closest('.exec-strategy-card')?.dataset?.instanceId;
      if (!id) return;
      expandedId = expandedId === id ? null : id;
      if (expandedId) syncChart(id);
      else publish('execution.strategy.deselected');
      qRender(); return;
    }

    // Expand catalog item (read-only detail)
    const catHdr = t.closest('.exec-catalog-header');
    if (catHdr) {
      const tid = catHdr.dataset.templateId;
      expandedCatalogId = expandedCatalogId === tid ? null : tid;
      qRender(); return;
    }

    // Toggle inactive
    if (t.closest('[data-action="toggle-inactive"]')) { showInactive = !showInactive; qRender(); return; }

    // Select/deselect all coins
    if (t.closest('[data-action="select-all-coins"]')) { t.closest('.exec-composer')?.querySelectorAll('[data-coin]').forEach(cb => cb.checked = true); return; }
    if (t.closest('[data-action="deselect-all-coins"]')) { t.closest('.exec-composer')?.querySelectorAll('[data-coin]').forEach(cb => cb.checked = false); return; }

    // Submit to backtest / paper
    const submitBtn = t.closest('[data-action="submit-backtest"], [data-action="submit-paper"]');
    if (submitBtn) { handleComposerSubmit(submitBtn); return; }

    // Copy from leaderboard
    const copyBtn = t.closest('[data-action="copy-variant"]');
    if (copyBtn) {
      const vid = copyBtn.dataset.variantId;
      const eq = prompt('模拟资金 (USDT):', '10000');
      if (!eq) return;
      copyBtn.disabled = true; copyBtn.textContent = '...';
      runtimeSvc.copyVariant(vid).then(() => refreshAll()).catch(log).finally(() => { copyBtn.disabled = false; copyBtn.textContent = '复制'; });
      return;
    }

    // Go live
    if (t.closest('[data-action="go-live"]')) {
      const id = t.closest('[data-action="go-live"]').dataset.instanceId;
      if (!confirm('确定切换到实盘？将使用 Bitget 真实资金。')) return;
      runtimeSvc.updateRuntimeInstance(id, { live_mode: 'live' }).then(() => refreshAll()).catch(log);
      return;
    }

    // Delete
    if (t.closest('[data-action="delete"]')) {
      if (!confirm('删除？')) return;
      runtimeSvc.deleteRuntimeInstance(t.closest('[data-action="delete"]').dataset.instanceId).then(() => refreshAll()).catch(log);
      return;
    }

    // Start/Stop
    const actBtn = t.closest('[data-action="start"], [data-action="stop"]');
    if (actBtn) {
      const fn = actBtn.dataset.action === 'start' ? runtimeSvc.startRuntimeInstance : runtimeSvc.stopRuntimeInstance;
      actBtn.disabled = true;
      fn(actBtn.dataset.instanceId).then(() => refreshAll()).catch(log).finally(() => { actBtn.disabled = false; });
      return;
    }
  });
}

function handleComposerSubmit(btn) {
  const composer = btn.closest('.exec-composer');
  if (!composer) return;
  const tid = composer.dataset.templateId;
  const coins = [...composer.querySelectorAll('[data-coin]:checked')].map(cb => cb.dataset.coin);
  const tfs = [...composer.querySelectorAll('[data-tf]:checked')].map(cb => cb.dataset.tf);
  const eq = parseFloat(composer.querySelector('[data-field="equity"]')?.value || '10000');
  if (!coins.length || !tfs.length) { alert('请选择币种和周期'); return; }
  btn.disabled = true;
  const total = coins.length * tfs.length;
  let done = 0;
  btn.textContent = `0/${total}...`;
  (async () => {
    for (const sym of coins) {
      for (const tf of tfs) {
        try { await runtimeSvc.launchFromCatalog(tid, sym, tf, 'disabled', eq); } catch {}
        btn.textContent = `${++done}/${total}...`;
      }
    }
    btn.disabled = false; btn.textContent = btn.dataset.action === 'submit-paper' ? '开始模拟' : '提交回测';
    refreshAll();
  })();
}

function log(e) { console.warn('[exec]', e); }

// ── Data ────────────────────────────────────────────────────────────────
function refreshAll() {
  // Core data — always fetch
  let changed = false;
  paperSvc.getPaperExecutionState().then(s => { paperState = s; changed = true; }).catch(() => {});
  runtimeSvc.getRuntimeInstances().then(r => { instances = r?.instances || []; changed = true; }).catch(() => {});
  // Leaderboard — only fetch every 3rd poll (evolution runs slowly)
  if (!this._pollCount) this._pollCount = 0;
  if (++this._pollCount % 3 === 0) {
    runtimeSvc.getLeaderboard(10).then(r => { leaderboard = r?.leaderboard || []; }).catch(() => {});
  }
  // Single render after a short delay to batch all responses
  setTimeout(() => { if (changed || true) qRender(); }, 300);
  if (!catalogLoaded) loadCatalog();
  if (accountMode === 'live') {
    const now = Date.now();
    if (now - liveStatusCacheTime > CACHE_TTL) liveSvc.getLiveExecutionStatus().then(s => { liveStatus = s; liveStatusCacheTime = Date.now(); qRender(); }).catch(() => {});
    if (now - liveAccountCacheTime > CACHE_TTL) liveSvc.getLiveAccount('live').then(a => { liveAccount = a; liveAccountCacheTime = Date.now(); qRender(); }).catch(() => {});
  }
}
async function loadCatalog() { catalogLoaded = true; try { catalog = (await runtimeSvc.getStrategyCatalog())?.templates || []; } catch {} qRender(); }
function startPolling() { stopPolling(); pollTimer = setInterval(() => { if (panelOpen) refreshAll(); }, POLL_MS); }
function stopPolling() { if (pollTimer) clearInterval(pollTimer); pollTimer = null; }
function qRender() { if (renderQueued) return; renderQueued = true; requestAnimationFrame(() => { renderQueued = false; render(); }); }

// ── Chart Sync ──────────────────────────────────────────────────────────
function syncChart(id) {
  const inst = instances.find(i => i.config?.instance_id === id);
  if (!inst?.config) return;
  // Set interval silently first, then trigger symbol change (which loads chart)
  marketState.currentInterval = inst.config.timeframe;
  marketState.currentSymbol = '';
  setSymbol(inst.config.symbol);
  const ps = inst.status?.paper_state;
  if (!ps) { publish('execution.trade.markers', []); return; }
  const markers = [];
  for (const pos of [...(ps.open_positions || []), ...(ps.recent_closed_positions || [])]) {
    if (pos.opened_at_ts) markers.push({ time: toUnix(pos.opened_at_ts), position: pos.direction==='short'?'aboveBar':'belowBar', color: pos.direction==='long'?'#00e676':'#ff1744', shape: pos.direction==='long'?'arrowUp':'arrowDown', text: pos.direction==='long'?'买':'卖' });
    if (pos.closed_at_ts) markers.push({ time: toUnix(pos.closed_at_ts), position: pos.direction==='short'?'belowBar':'aboveBar', color: (pos.realized_pnl??0)>=0?'#00e676':'#ff1744', shape:'square', text:(pos.realized_pnl??0)>=0?'盈':'损' });
  }
  markers.sort((a,b) => a.time - b.time);
  publish('execution.trade.markers', markers);
}
function toUnix(ts) { return typeof ts==='number'?(ts>1e12?Math.floor(ts/1000):ts):Math.floor(new Date(ts).getTime()/1000); }

// ── Render ──────────────────────────────────────────────────────────────
function render() {
  const body = $('#exec-body');
  if (!body) return;
  setHtml(body, [
    renderOverview(),
    renderRunning(),
    renderCatalog(),
    renderComposer(),
    renderLeaderboard(),
    renderInactive(),
  ].join(''));
}

// ── 1. Overview ─────────────────────────────────────────────────────────
function renderOverview() {
  if (accountMode === 'live') {
    if (!liveStatus?.api_key_ready) return sec('概览', 'Bitget', '<div class="exec-note">连接中...</div>');
    const ah = liveAccount?.ok ? `<div class="exec-overview-compact">
      <div class="exec-ov-item"><span class="exec-ov-label">权益</span><span class="exec-ov-value">${formatUsd(liveAccount.total_equity??0)}</span></div>
      <div class="exec-ov-item"><span class="exec-ov-label">可用</span><span class="exec-ov-value">${formatUsd(liveAccount.usdt_available??0)}</span></div></div>` : '';
    return sec('概览', 'Bitget', `<div class="exec-overview-compact">
      <div class="exec-ov-item"><span class="exec-ov-label">状态</span><span class="exec-ov-value" style="color:var(--v2-green)">已连接</span></div>
      <div class="exec-ov-item"><span class="exec-ov-label">模式</span><span class="exec-ov-value">${liveStatus?.default_mode||'live'}</span></div></div>${ah}`);
  }
  const a = paperState?.account;
  if (!a) return sec('概览', '模拟', '<div class="exec-note">加载中...</div>');
  const running = instances.filter(i => i.status?.runtime_state==='running').length;
  return sec('概览', '模拟', `<div class="exec-overview-compact">
    <div class="exec-ov-item"><span class="exec-ov-label">权益</span><span class="exec-ov-value">${formatUsd(a.equity??10000)}</span></div>
    <div class="exec-ov-item"><span class="exec-ov-label">盈亏</span><span class="exec-ov-value ${pnlColorClass(a.realized_pnl??0)}">${fmtPnl(a.realized_pnl??0)}</span></div>
    <div class="exec-ov-item"><span class="exec-ov-label">运行</span><span class="exec-ov-value">${running}</span></div>
    <div class="exec-ov-item"><span class="exec-ov-label">总数</span><span class="exec-ov-value">${instances.length}</span></div></div>`);
}

// ── 2. Running ──────────────────────────────────────────────────────────
function renderRunning() {
  const running = instances.filter(i => i.status?.runtime_state === 'running');
  if (!running.length) return '';
  return sec('运行中', `(${running.length})`, running.map(renderCard).join(''));
}

// ── 3. Strategy Catalog (read-only, click to see details) ───────────────
function renderCatalog() {
  if (!catalog.length) return '';
  const riskLabel = {low:'低',medium:'中',high:'高'};
  const catLabel = {trend:'趋势',reversal:'反转',breakout:'突破',scalp:'剥头皮'};
  const items = catalog.map(t => {
    const expanded = expandedCatalogId === t.template_id;
    const triggers = (t.default_trigger_modes || []).map(m => ({pre_limit:'限价预挂',rejection:'反转拒绝',failed_breakout:'假突破回收',retest:'突破回测'}[m]||m)).join(', ');
    return `<div class="exec-catalog-item">
      <div class="exec-catalog-header" data-template-id="${esc(t.template_id)}">
        <span class="exec-catalog-name">${esc(t.name)}</span>
        <span style="color:var(--v2-muted);font-size:10px">${catLabel[t.category]||''} | 风险${riskLabel[t.risk_level]||''}</span>
        <span style="margin-left:auto;font-size:10px">${expanded?'▾':'▸'}</span>
      </div>
      ${expanded ? `<div class="exec-catalog-detail">
        <div class="exec-catalog-desc">${esc(t.description)}</div>
        <div class="exec-detail-row"><span>触发模式</span><span>${triggers}</span></div>
        <div class="exec-detail-row"><span>支持周期</span><span>${(t.supported_timeframes||[]).join(', ')}</span></div>
        <div class="exec-detail-row"><span>默认 RR</span><span>${t.default_params?.rr_target || '2.0'}</span></div>
        <div class="exec-detail-row"><span>默认风险</span><span>${((t.default_params?.risk_per_trade||0.003)*100).toFixed(1)}%</span></div>
      </div>` : ''}
    </div>`;
  }).join('');
  return sec('策略库', `(${catalog.length})`, items);
}

// ── 4. Factor Composer (create custom strategies) ───────────────────────
function renderComposer() {
  const topCoins = ['BTCUSDT','ETHUSDT','SOLUSDT','HYPEUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','SUIUSDT','PEPEUSDT','TAOUSDT','BNBUSDT','AVAXUSDT'];
  const tfs = ['5m','15m','1h','4h','1d'];
  // Use the first catalog template as base (sr_full is most versatile)
  const baseTid = 'sr_full';
  return sec('因子组合', '', `
    <div class="exec-composer" data-template-id="${baseTid}">
      <div class="exec-multi-label">策略类型</div>
      <div class="exec-multi-chips">
        <label class="exec-chip"><input type="checkbox" data-trigger="rejection" checked/> 反转拒绝</label>
        <label class="exec-chip"><input type="checkbox" data-trigger="failed_breakout" checked/> 假突破回收</label>
        <label class="exec-chip"><input type="checkbox" data-trigger="retest" checked/> 突破回测</label>
        <label class="exec-chip"><input type="checkbox" data-trigger="pre_limit"/> 限价预挂</label>
      </div>
      <div class="exec-multi-label">币种 <input class="exec-coin-search" data-role="coin-filter" placeholder="搜索添加..." style="width:80px;margin-left:6px;padding:2px 6px;font-size:10px;background:var(--v2-bg);border:1px solid var(--v2-border);color:var(--v2-text);border-radius:3px"/>
        <button data-action="select-all-coins" style="margin-left:4px;font-size:9px;cursor:pointer;color:var(--v2-primary);background:none;border:none">全选</button>
        <button data-action="deselect-all-coins" style="font-size:9px;cursor:pointer;color:var(--v2-muted);background:none;border:none">清空</button>
      </div>
      <div class="exec-multi-chips exec-coin-chips">${topCoins.map(c => `<label class="exec-chip"><input type="checkbox" data-coin="${c}"/> ${c.replace('USDT','')}</label>`).join('')}</div>
      <div class="exec-multi-label">周期</div>
      <div class="exec-multi-chips">${tfs.map(tf => `<label class="exec-chip"><input type="checkbox" data-tf="${tf}" ${['1h','4h'].includes(tf)?'checked':''}/> ${tf}</label>`).join('')}</div>
      <div class="exec-catalog-form" style="margin-top:8px">
        <span style="font-size:11px;color:var(--v2-muted)">资金</span>
        <input data-field="equity" value="10000" style="width:60px" type="number"/>
        <button class="exec-btn exec-btn-sm exec-btn-primary" data-action="submit-paper" data-template-id="${baseTid}">开始模拟</button>
      </div>
    </div>`);
}

// ── 5. Leaderboard ──────────────────────────────────────────────────────
function renderLeaderboard() {
  if (!leaderboard.length) return sec('进化排行榜', '', '<div class="exec-note" style="font-size:10px">引擎未运行或暂无结果。终端运行 python run_evolution.py 启动</div>');
  const rows = leaderboard.map((v, i) => `<div class="exec-lb-row">
    <span class="exec-lb-rank">#${i+1}</span>
    <span class="exec-lb-sym">${esc(v.symbol)} ${esc(v.timeframe)}</span>
    <span class="${v.total_return_pct>=0?'pnl-pos':'pnl-neg'}">${v.total_return_pct>=0?'+':''}${v.total_return_pct}%</span>
    <span style="color:var(--v2-muted);font-size:10px">WR${v.win_rate}% S${v.sharpe_ratio}</span>
    <button class="exec-btn exec-btn-sm exec-btn-primary" data-action="copy-variant" data-variant-id="${esc(v.variant_id)}">复制</button>
  </div>`).join('');
  return sec('进化排行榜', '', rows);
}

// ── 6. Inactive ─────────────────────────────────────────────────────────
function renderInactive() {
  const inactive = instances.filter(i => i.status?.runtime_state !== 'running');
  if (!inactive.length) return '';
  return `<div class="exec-section"><div class="exec-group-label exec-group-toggle" data-action="toggle-inactive">已停止 (${inactive.length}) ${showInactive?'▾':'▸'}</div>${showInactive?inactive.map(renderCard).join(''):''}</div>`;
}

// ── Card ─────────────────────────────────────────────────────────────────
function renderCard(inst) {
  const c = inst.config||{}, s = inst.status||{}, id = c.instance_id||'';
  const state = s.runtime_state||'stopped';
  const isLive = c.live_mode === 'live';
  const pnl = s.paper_state?.account?.realized_pnl;
  const expanded = expandedId === id;
  return `<div class="exec-strategy-card ${expanded?'is-expanded':''}" data-instance-id="${esc(id)}">
    <div class="exec-strategy-header">
      <span class="exec-strategy-symbol">${esc(c.symbol||'?')}</span>
      <span class="exec-strategy-tf">${esc(c.timeframe||'?')}</span>
      ${isLive?'<span style="color:#ff1744;font-size:10px;font-weight:700">实盘</span>':''}
      ${!isLive&&pnl!=null?`<span class="exec-strategy-pnl ${pnlColorClass(pnl)}">${fmtPnl(pnl)}</span>`:''}
      <span class="exec-strategy-state ${{running:'exec-state-running',stopped:'exec-state-stopped',blocked:'exec-state-blocked'}[state]||''}">${{running:'运行中',stopped:'停止',blocked:'阻止'}[state]||state}</span>
    </div>
    ${expanded ? renderDetail(inst) : ''}
    <div class="exec-strategy-actions">
      ${state==='running'?`<button class="exec-btn exec-btn-sm" data-action="stop" data-instance-id="${esc(id)}">停止</button>`:`<button class="exec-btn exec-btn-sm exec-btn-primary" data-action="start" data-instance-id="${esc(id)}">启动</button>`}
      ${!isLive?`<button class="exec-btn exec-btn-sm" data-action="go-live" data-instance-id="${esc(id)}" style="color:var(--v2-red)">转实盘</button>`:''}
      ${state!=='running'?`<button class="exec-btn exec-btn-sm exec-btn-danger" data-action="delete" data-instance-id="${esc(id)}">删除</button>`:''}
    </div></div>`;
}

function renderDetail(inst) {
  const c=inst.config||{},s=inst.status||{},ps=s.paper_state,sc=c.strategy_config||{};
  const isLive=c.live_mode==='live';
  const triggers=(Array.isArray(sc.enabled_trigger_modes)?sc.enabled_trigger_modes:[]).map(m=>({pre_limit:'限价预挂',rejection:'反转拒绝',failed_breakout:'假突破回收',retest:'突破回测'}[m]||m)).join(' + ');
  let h=`<div class="exec-strategy-detail">
    <div class="exec-detail-row"><span>策略</span><span>${triggers||'S/R'}</span></div>
    <div class="exec-detail-row"><span>模式</span><span>${isLive?'🔴 实盘':'📋 模拟'}</span></div>
    <div class="exec-detail-row"><span>Bar</span><span>${s.last_processed_bar??'-'}</span></div>
    <div class="exec-detail-row"><span>更新</span><span>${rel(s.last_tick_at)}</span></div>`;
  if(s.last_error) h+=`<div class="exec-strategy-error" style="white-space:normal;margin:4px 0">${esc(s.last_error).slice(0,100)}</div>`;
  if(ps){
    const a=ps.account||{},eq=isLive&&liveAccount?.ok?liveAccount.total_equity:(a.equity??10000);
    h+=`<div class="exec-detail-row"><span>${isLive?'实盘权益':'模拟权益'}</span><span>${formatUsd(eq)}</span></div>
      <div class="exec-detail-row"><span>盈亏</span><span class="${pnlColorClass(a.realized_pnl??0)}">${fmtPnl(a.realized_pnl??0)}</span></div>`;
    for(const pos of(ps.open_positions||[])) h+=`<div class="exec-trade-mini"><span class="exec-dir-${pos.direction}">${pos.direction==='long'?'多':'空'}</span> ${formatUsd(pos.entry_price)} <span class="${pnlColorClass(pos.unrealized_pnl??0)}">${fmtPnl(pos.unrealized_pnl??0)}</span></div>`;
    const closed=ps.recent_closed_positions||[];
    if(closed.length){h+='<div class="exec-detail-subtitle">最近交易</div>';
      for(const pos of closed.slice(-3).reverse()){const r={tp_hit:'盈',sl_hit:'损',expired:'期'}[pos.exit_reason]||'';
        h+=`<div class="exec-trade-mini"><span class="exec-dir-${pos.direction}">${pos.direction==='long'?'多':'空'}</span> ${formatUsd(pos.entry_price)}→${formatUsd(pos.exit_price??0)} <span class="${pnlColorClass(pos.realized_pnl??0)}">${fmtPnl(pos.realized_pnl??0)}</span> ${r}</div>`;}}
  }
  return h+'</div>';
}

// ── Helpers ─────────────────────────────────────────────────────────────
function sec(title, badge, content) { return `<div class="exec-section"><div class="exec-section-title">${title} <small style="color:var(--v2-muted)">${badge}</small></div>${content}</div>`; }
function esc(s) { const d=document.createElement('div'); d.textContent=String(s??''); return d.innerHTML; }
function fmtPnl(v) { return (v>=0?'+':'')+formatUsd(v); }
function rel(ts) { if(!ts)return'-'; try{const d=Math.floor((Date.now()-(typeof ts==='number'?ts*1000:new Date(ts).getTime()))/1000); if(d<60)return`${d}秒前`;if(d<3600)return`${Math.floor(d/60)}分前`;if(d<86400)return`${Math.floor(d/3600)}时前`;return`${Math.floor(d/86400)}天前`;}catch{return'-';} }
