/**
 * View Loaders — product-grade pages for Strategy OS
 * Each view is a self-contained async function that fetches data and renders.
 */

import { fetchJson } from './util/fetch.js';
import { formatUsd, pnlColorClass } from './util/format.js';

const T = {pre_limit:'限价预挂',rejection:'反转拒绝',failed_breakout:'假突破回收',retest:'突破回测'};
const R = {low:'🟢低',medium:'🟡中',high:'🔴高'};
const S = {running:'运行中',stopped:'已完成',blocked:'已阻止'};
const SC = {running:'state-run',stopped:'state-stop',blocked:'state-block'};
const pnl = v => (v>=0?'+':'')+formatUsd(v);
const rel = ts => {if(!ts)return'-';try{const d=Math.floor((Date.now()-(typeof ts==='number'?ts*1000:new Date(ts).getTime()))/1000);if(d<60)return d+'秒前';if(d<3600)return Math.floor(d/60)+'分前';if(d<86400)return Math.floor(d/3600)+'时前';return Math.floor(d/86400)+'天前';}catch{return'-';}};

// ── FACTORY ─────────────────────────────────────────────────────────────
export async function loadFactory(el) {
  const [cat, inst] = await Promise.all([
    fetchJson('/api/runtime/catalog').catch(()=>({})),
    fetchJson('/api/runtime/instances').catch(()=>({})),
  ]);
  const templates = cat.templates || [];
  const myStrats = (inst.instances || []).slice(-10).reverse();

  el.innerHTML = `
    <div class="view-header">
      <h2>策略工厂</h2>
      <p class="view-desc">从模板创建或自定义组合策略 → 送入模拟中心测试</p>
    </div>
    <div class="view-grid">
      <!-- LEFT: Templates + My Strategies -->
      <div>
        <div class="view-section">
          <div class="section-tabs">
            <button class="stab active" data-tab="tpl">系统模板 (${templates.length})</button>
            <button class="stab" data-tab="my">我的策略 (${myStrats.length})</button>
          </div>
          <div class="stab-content active" data-tab="tpl">
            ${templates.map(t => `
              <div class="fcard">
                <div class="fcard-top">
                  <span class="fcard-name">${t.name}</span>
                  <span class="fcard-risk">${R[t.risk_level]||''}</span>
                </div>
                <p class="fcard-desc">${t.description}</p>
                <div class="fcard-tags">
                  ${(t.default_trigger_modes||[]).map(m=>`<span class="tag">${T[m]||m}</span>`).join('')}
                </div>
                <div class="fcard-meta">
                  <span>周期: ${t.supported_timeframes.join(' ')}</span>
                  <span>RR: ${t.default_params?.rr_target||'动态'}</span>
                </div>
              </div>
            `).join('')}
          </div>
          <div class="stab-content" data-tab="my">
            ${myStrats.length === 0 ? '<div class="empty-state">暂无自定义策略<br><small>在右侧组合器创建</small></div>' : ''}
            ${myStrats.map(i => {
              const c=i.config, s=i.status, p=s.paper_state?.account?.realized_pnl||0;
              return `<div class="fcard fcard-my">
                <div class="fcard-top">
                  <span class="fcard-name">${c.symbol} ${c.timeframe}</span>
                  <span class="${pnlColorClass(p)}">${pnl(p)}</span>
                </div>
                <div class="fcard-meta"><span>${c.label||'-'}</span><span>${S[s.runtime_state]||s.runtime_state}</span></div>
              </div>`;
            }).join('')}
          </div>
        </div>
      </div>

      <!-- RIGHT: Strategy Builder -->
      <div>
        <div class="view-section">
          <h3>🧪 策略构建器</h3>
          <div class="builder">
            <div class="builder-row">
              <label>策略名称</label>
              <input id="b-name" placeholder="我的S/R策略" />
            </div>
            <div class="builder-row">
              <label>逻辑骨架</label>
              <select id="b-type">
                <option value="sr_full">S/R 全模式 (反转+假突破+回测)</option>
                <option value="sr_reversal">S/R 反转 (拒绝入场)</option>
                <option value="sr_retest">突破回测 (趋势接法)</option>
                <option value="ma_ribbon">均线开花 (趋势跟随)</option>
                <option value="hft_mm">做市偏斜 (高频)</option>
                <option value="hft_imbalance">盘口失衡 (高频)</option>
              </select>
            </div>
            <div class="builder-row">
              <label>因子条件</label>
              <div class="builder-chips">
                <label class="bchip"><input type="checkbox" checked /> RSI 过滤</label>
                <label class="bchip"><input type="checkbox" /> ADX 趋势</label>
                <label class="bchip"><input type="checkbox" /> BB 压缩</label>
                <label class="bchip"><input type="checkbox" /> 成交量确认</label>
                <label class="bchip"><input type="checkbox" /> EMA 排列</label>
                <label class="bchip"><input type="checkbox" /> MACD 交叉</label>
              </div>
            </div>
            <div class="builder-row">
              <label>入场方式</label>
              <div class="builder-chips">
                <label class="bchip"><input type="checkbox" checked /> 反转拒绝</label>
                <label class="bchip"><input type="checkbox" checked /> 假突破回收</label>
                <label class="bchip"><input type="checkbox" /> 限价预挂</label>
                <label class="bchip"><input type="checkbox" /> 突破回测</label>
              </div>
            </div>
            <div class="builder-2col">
              <div class="builder-row"><label>币种</label><input id="b-sym" value="BTCUSDT" /></div>
              <div class="builder-row"><label>周期</label>
                <select id="b-tf"><option>5m</option><option>15m</option><option selected>1h</option><option>4h</option><option>1d</option></select>
              </div>
            </div>
            <div class="builder-2col">
              <div class="builder-row"><label>模拟资金</label><input id="b-eq" type="number" value="10000" /></div>
              <div class="builder-row"><label>风控</label>
                <select><option>固定1%风险</option><option>结构止损</option><option>ATR止损</option></select>
              </div>
            </div>
            <div class="builder-actions">
              <button class="view-btn">保存草稿</button>
              <button class="view-btn view-btn-primary" id="b-submit">提交到模拟中心 →</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Tab switching
  el.querySelectorAll('.stab').forEach(btn => {
    btn.onclick = () => {
      el.querySelectorAll('.stab').forEach(b=>b.classList.remove('active'));
      el.querySelectorAll('.stab-content').forEach(c=>c.classList.remove('active'));
      btn.classList.add('active');
      el.querySelector(`.stab-content[data-tab="${btn.dataset.tab}"]`)?.classList.add('active');
    };
  });

  // Submit
  el.querySelector('#b-submit')?.addEventListener('click', async () => {
    const type = el.querySelector('#b-type')?.value || 'sr_full';
    const sym = el.querySelector('#b-sym')?.value || 'BTCUSDT';
    const tf = el.querySelector('#b-tf')?.value || '4h';
    const eq = el.querySelector('#b-eq')?.value || '10000';
    try {
      await fetchJson(`/api/runtime/catalog/${type}/launch?symbol=${sym}&timeframe=${tf}&live_mode=disabled&starting_equity=${eq}`, {method:'POST'});
      alert(`策略已提交: ${sym} ${tf} ${type}`);
    } catch (e) { alert('提交失败: ' + e.message); }
  });
}

// ── SIMULATE ────────────────────────────────────────────────────────────
export async function loadSimulate(el) {
  const { instances } = await fetchJson('/api/runtime/instances');
  const all = instances || [];
  const paper = all.filter(i => i.config.live_mode !== 'live');
  const running = paper.filter(i => i.status.runtime_state === 'running');
  const finished = paper.filter(i => i.status.runtime_state !== 'running');
  const profitable = paper.filter(i => (i.status.paper_state?.account?.realized_pnl||0) > 0);
  const totalPnl = paper.reduce((s,i) => s + (i.status.paper_state?.account?.realized_pnl||0), 0);

  // Group by label prefix (strategy type)
  const groups = {};
  paper.forEach(i => {
    const key = i.config.label?.split('-')[0] || i.config.symbol;
    if (!groups[key]) groups[key] = [];
    groups[key].push(i);
  });

  el.innerHTML = `
    <div class="view-header">
      <h2>模拟中心</h2>
      <p class="view-desc">批量策略实验 — ${paper.length} 个模拟策略</p>
    </div>
    <div class="view-stats">
      <div class="view-stat"><div class="view-stat-value">${running.length}</div><div class="view-stat-label">运行中</div></div>
      <div class="view-stat"><div class="view-stat-value">${finished.length}</div><div class="view-stat-label">已完成</div></div>
      <div class="view-stat"><div class="view-stat-value">${profitable.length}</div><div class="view-stat-label">盈利</div></div>
      <div class="view-stat"><div class="view-stat-value ${pnlColorClass(totalPnl)}">${pnl(totalPnl)}</div><div class="view-stat-label">总盈亏</div></div>
    </div>

    ${Object.entries(groups).map(([name, items]) => {
      const gPnl = items.reduce((s,i) => s + (i.status.paper_state?.account?.realized_pnl||0), 0);
      const gRun = items.filter(i => i.status.runtime_state === 'running').length;
      return `
        <div class="sim-group">
          <div class="sim-group-header">
            <span class="sim-group-name">${name}</span>
            <span class="sim-group-count">${items.length} 策略 | ${gRun} 运行</span>
            <span class="${pnlColorClass(gPnl)}">${pnl(gPnl)}</span>
          </div>
          ${items.slice(0,8).map(i => {
            const c=i.config, s=i.status;
            const p = s.paper_state?.account?.realized_pnl||0;
            const trades = s.paper_state?.account?.closed_trade_count||0;
            const wr = trades > 0 ? '—' : '0%';
            return `<div class="sim-row">
              <span class="sim-symbol">${c.symbol}</span>
              <span class="sim-tf">${c.timeframe}</span>
              <span class="${SC[s.runtime_state]||''}">${S[s.runtime_state]||s.runtime_state}</span>
              <span class="${pnlColorClass(p)}" style="min-width:60px">${pnl(p)}</span>
              <span class="sim-meta">${trades}笔 | bar ${s.last_processed_bar||'-'}</span>
            </div>`;
          }).join('')}
          ${items.length > 8 ? `<div class="sim-more">... 还有 ${items.length-8} 个</div>` : ''}
        </div>`;
    }).join('')}
  `;
}

// ── LEADERBOARD ─────────────────────────────────────────────────────────
export async function loadLeaderboard(el) {
  const data = await fetchJson('/api/runtime/leaderboard?limit=20');
  const lb = data.leaderboard || [];

  el.innerHTML = `
    <div class="view-header">
      <h2>排行榜</h2>
      <p class="view-desc">Gen ${data.generation||0} | 已测试 ${data.total_tested||0} | 盈利 ${data.total_profitable||0}</p>
    </div>

    <!-- Filters -->
    <div class="lb-filters">
      <select class="lb-filter"><option>全部来源</option><option>AI生成</option><option>手动</option></select>
      <select class="lb-filter"><option>全部周期</option><option>1h</option><option>4h</option><option>1d</option></select>
      <select class="lb-filter"><option>按综合分</option><option>按收益率</option><option>按胜率</option><option>按Sharpe</option></select>
    </div>

    ${lb.length === 0 ? `
      <div class="empty-state">
        <div class="empty-icon">🧬</div>
        <div>暂无排行数据</div>
        <small>终端运行 python run_evolution.py 启动进化引擎</small>
      </div>
    ` : ''}

    <div class="view-list">
      ${lb.map((v,i) => {
        const triggers = (v.trigger_modes||[]).map(m=>T[m]||m).join(' + ') || v.name;
        const factors = (v.factors||[]).join(', ') || '基础S/R';
        return `<div class="lb-card">
          <div class="lb-card-top">
            <span class="lb-rank">${['🥇','🥈','🥉'][i]||'#'+(i+1)}</span>
            <span class="lb-card-sym">${v.symbol} ${v.timeframe}</span>
            <span class="${v.total_return_pct>=0?'pnl-pos':'pnl-neg'}" style="font-size:14px;font-weight:700">${v.total_return_pct>=0?'+':''}${v.total_return_pct}%</span>
          </div>
          <div class="lb-card-strategy">${triggers}</div>
          <div class="lb-card-factors">因子: ${factors}</div>
          <div class="lb-card-stats">
            <span>WR ${v.win_rate}%</span>
            <span>Sharpe ${v.sharpe_ratio}</span>
            <span>DD ${v.max_drawdown_pct}%</span>
            <span>${v.total_trades}笔</span>
            <span>PF ${v.profit_factor}</span>
          </div>
          <div class="lb-card-actions">
            <button class="view-btn view-btn-sm">查看详情</button>
            <button class="view-btn view-btn-sm view-btn-primary">加入实盘库</button>
          </div>
        </div>`;
      }).join('')}
    </div>
  `;
}

// ── LIVE LIBRARY ────────────────────────────────────────────────────────
export async function loadLive(el) {
  const [instRes, accRes, statusRes] = await Promise.allSettled([
    fetchJson('/api/runtime/instances'),
    fetchJson('/api/live-execution/account?mode=live'),
    fetchJson('/api/live-execution/status'),
  ]);
  const instances = instRes.status==='fulfilled' ? instRes.value?.instances||[] : [];
  const acc = accRes.status==='fulfilled' ? accRes.value : null;
  const status = statusRes.status==='fulfilled' ? statusRes.value?.status : null;
  const live = instances.filter(i => i.config.live_mode === 'live');
  const liveRunning = live.filter(i => i.status.runtime_state === 'running');

  el.innerHTML = `
    <div class="view-header">
      <h2>实盘策略库</h2>
      <p class="view-desc">Bitget ${status?.api_key_ready ? '🟢 已连接' : '🔴 未连接'} | ${live.length} 个部署策略</p>
    </div>

    <div class="view-stats">
      <div class="view-stat"><div class="view-stat-value">${acc?.ok ? formatUsd(acc.total_equity) : '-'}</div><div class="view-stat-label">总权益</div></div>
      <div class="view-stat"><div class="view-stat-value">${acc?.ok ? formatUsd(acc.usdt_available) : '-'}</div><div class="view-stat-label">可用资金</div></div>
      <div class="view-stat"><div class="view-stat-value">${acc?.ok ? acc.positions?.length||0 : '-'}</div><div class="view-stat-label">持仓</div></div>
      <div class="view-stat"><div class="view-stat-value">${liveRunning.length}</div><div class="view-stat-label">活跃策略</div></div>
    </div>

    ${live.length === 0 ? `
      <div class="empty-state">
        <div class="empty-icon">📦</div>
        <div>暂无实盘策略</div>
        <small>从排行榜选择优胜策略 → 加入实盘库 → 配置资金后启动</small>
      </div>
    ` : ''}

    <div class="live-cards">
      ${live.map(i => {
        const c=i.config, s=i.status;
        const p = s.paper_state?.account?.realized_pnl||0;
        const eq = s.paper_state?.account?.equity||c.paper_config?.starting_equity||0;
        const bar = s.last_processed_bar||'-';
        const triggers = (c.strategy_config?.enabled_trigger_modes||[]).map(m=>T[m]||m).join('+');
        return `<div class="deploy-card">
          <div class="deploy-card-header">
            <span class="deploy-card-sym">${c.symbol} ${c.timeframe}</span>
            <span class="${SC[s.runtime_state]||''}">${S[s.runtime_state]||s.runtime_state}</span>
          </div>
          <div class="deploy-card-label">${c.label||triggers||'S/R策略'}</div>
          <div class="deploy-card-stats">
            <div><span class="deploy-stat-label">分配资金</span><span class="deploy-stat-value">${formatUsd(eq)}</span></div>
            <div><span class="deploy-stat-label">盈亏</span><span class="deploy-stat-value ${pnlColorClass(p)}">${pnl(p)}</span></div>
            <div><span class="deploy-stat-label">自动下单</span><span class="deploy-stat-value">${c.auto_live_submit?'🟢是':'⚪否'}</span></div>
            <div><span class="deploy-stat-label">Bar</span><span class="deploy-stat-value">${bar}</span></div>
          </div>
          <div class="deploy-card-actions">
            <button class="view-btn view-btn-sm ${s.runtime_state==='running'?'':'view-btn-primary'}">${s.runtime_state==='running'?'暂停':'启动'}</button>
            <button class="view-btn view-btn-sm">编辑</button>
            <button class="view-btn view-btn-sm" style="color:var(--v2-red)">移除</button>
          </div>
        </div>`;
      }).join('')}
    </div>
  `;
}

// ── MONITOR ─────────────────────────────────────────────────────────────
export async function loadMonitor(el) {
  const [instRes, statusRes] = await Promise.allSettled([
    fetchJson('/api/runtime/instances'),
    fetchJson('/api/live-execution/status'),
  ]);
  const instances = instRes.status==='fulfilled' ? instRes.value?.instances||[] : [];
  const status = statusRes.status==='fulfilled' ? statusRes.value?.status : null;
  const running = instances.filter(i => i.status.runtime_state === 'running');
  const liveRunning = running.filter(i => i.config.live_mode === 'live');
  const paperRunning = running.filter(i => i.config.live_mode !== 'live');

  el.innerHTML = `
    <div class="view-header">
      <h2>运行监控</h2>
      <p class="view-desc">实时策略状态和系统健康</p>
    </div>

    <!-- System Status -->
    <div class="monitor-status-bar">
      <div class="monitor-status-item"><span class="monitor-dot dot-green"></span> 系统在线</div>
      <div class="monitor-status-item"><span class="monitor-dot ${status?.api_key_ready?'dot-green':'dot-red'}"></span> Bitget ${status?.api_key_ready?'已连接':'未连接'}</div>
      <div class="monitor-status-item"><span class="monitor-dot ${running.length>0?'dot-green':'dot-gray'}"></span> ${running.length} 个策略活跃</div>
    </div>

    <div class="view-stats">
      <div class="view-stat"><div class="view-stat-value">${liveRunning.length}</div><div class="view-stat-label">实盘活跃</div></div>
      <div class="view-stat"><div class="view-stat-value">${paperRunning.length}</div><div class="view-stat-label">模拟活跃</div></div>
      <div class="view-stat"><div class="view-stat-value">0</div><div class="view-stat-label">今日订单</div></div>
      <div class="view-stat"><div class="view-stat-value">0%</div><div class="view-stat-label">风险占用</div></div>
    </div>

    ${running.length === 0 ? `
      <div class="empty-state">
        <div class="empty-icon">📡</div>
        <div>暂无活跃策略</div>
        <small>在策略工厂创建策略 → 模拟中心测试 → 实盘库部署</small>
      </div>
    ` : `
      <!-- Live strategies first -->
      ${liveRunning.length > 0 ? `<h3 style="margin:16px 0 8px;font-size:13px;color:var(--v2-red)">🔴 实盘运行</h3>` : ''}
      ${liveRunning.map(i => monitorRow(i)).join('')}

      ${paperRunning.length > 0 ? `<h3 style="margin:16px 0 8px;font-size:13px;color:var(--v2-primary)">📋 模拟运行</h3>` : ''}
      ${paperRunning.map(i => monitorRow(i)).join('')}
    `}

    <!-- Recent Events -->
    <div class="monitor-section" style="margin-top:20px">
      <h3 style="font-size:13px;color:var(--v2-muted);margin-bottom:8px">最近事件</h3>
      <div class="monitor-log">
        <div class="log-entry">系统启动</div>
        <div class="log-entry">${running.length} 个策略恢复运行</div>
        <div class="log-entry">Bitget API ${status?.api_key_ready?'连接成功':'未连接'}</div>
      </div>
    </div>
  `;
}

function monitorRow(i) {
  const c=i.config, s=i.status;
  const p = s.paper_state?.account?.realized_pnl||0;
  const isLive = c.live_mode === 'live';
  return `<div class="monitor-row">
    <span class="monitor-badge ${isLive?'badge-live':'badge-paper'}">${isLive?'实盘':'模拟'}</span>
    <span class="monitor-sym">${c.symbol} ${c.timeframe}</span>
    <span class="monitor-bar">bar ${s.last_processed_bar||'-'}</span>
    <span class="${pnlColorClass(p)}" style="min-width:60px;font-weight:600">${pnl(p)}</span>
    <span style="color:var(--v2-green);font-size:10px">${rel(s.last_tick_at)}</span>
  </div>`;
}
