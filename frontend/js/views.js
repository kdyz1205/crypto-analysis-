/**
 * View Loaders — Trading OS Research Console
 * Each page answers ONE question. Debug info in expand-only areas.
 */

import { fetchJson, invalidateCachePrefix } from './util/fetch.js';
import { formatUsd, pnlColorClass } from './util/format.js';
import { esc } from './util/dom.js';

async function doAction(btn, url, method, parentEl, reloader) {
  const orig = btn.textContent;
  btn.disabled = true; btn.textContent = orig + '...';
  try {
    await fetchJson(url, { method, noCache: true });
    invalidateCachePrefix('/api/tools/live');
    if (reloader && parentEl) await reloader(parentEl);
  } catch(e) {
    btn.textContent = '失败';
    setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 1500);
    return;
  }
  btn.disabled = false; btn.textContent = orig;
}

const T = {pre_limit:'限价预挂',rejection:'反转拒绝',failed_breakout:'假突破回收',retest:'突破回测'};
const pnlV = v => (v>=0?'+':'')+v;
const pnlU = v => (v>=0?'+':'')+formatUsd(v);
const rel = ts => {if(!ts)return'-';try{const d=Math.floor((Date.now()-(typeof ts==='number'?ts*1000:new Date(ts).getTime()))/1000);if(d<60)return d+'秒前';if(d<3600)return Math.floor(d/60)+'分前';if(d<86400)return Math.floor(d/3600)+'时前';return Math.floor(d/86400)+'天前';}catch(e){return'-';}};
const kpi = (label, value, cls='') => `<div class="kpi"><div class="kpi-val ${cls}">${value}</div><div class="kpi-label">${label}</div></div>`;
const badge = (text, type='') => `<span class="badge badge-${type}">${text}</span>`;
const expandBtn = (id) => `<button class="expand-btn" data-expand="${id}">详情</button>`;
const expandArea = (id, html) => `<div class="expand-area hidden" id="${id}">${html}</div>`;

// Shared helpers
const errorBanner = (msg) => `<div class="card-alert" style="margin-bottom:12px">⚠ ${msg}</div>`;
const loadingCard = (title) => `<div class="card"><div class="card-title">${title}</div><div class="c-muted" style="font-size:12px">加载中...</div></div>`;

// Safely unwrap an API response — handles both {ok,data,meta} and legacy direct returns
function unwrap(resp, path = null) {
  if (!resp || typeof resp !== 'object') return null;
  if (resp.ok === false) return null;
  const d = resp.data !== undefined ? resp.data : resp;
  if (!path) return d;
  const parts = path.split('.');
  let cur = d;
  for (const p of parts) {
    if (cur == null) return null;
    cur = cur[p];
  }
  return cur;
}

// Extract error message from any API response shape
function apiError(resp) {
  if (!resp) return null;
  if (resp.error) return String(resp.error);
  if (resp.detail) return String(resp.detail);
  return null;
}

// Parallel fetch with per-URL error capture
async function fetchMany(urls) {
  const results = await Promise.allSettled(urls.map(u => fetchJson(u).catch(e => ({ok:false,error:e.message}))));
  return results.map(r => r.status === 'fulfilled' ? r.value : {ok:false,error:r.reason?.message||'network error'});
}

// Shared tab switcher — binds click handlers once per container
function wireTabs(container) {
  container.querySelectorAll('.stab').forEach(btn => {
    // SAFE: fresh DOM per render, handler does not accumulate
    btn.onclick = () => {
      container.querySelectorAll('.stab').forEach(b => b.classList.remove('active'));
      container.querySelectorAll('.stab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      container.querySelector(`.stab-content[data-tab="${btn.dataset.tab}"]`)?.classList.add('active');
    };
  });
}

// ═══════════════════════════════════════════════════════════════════════
// DASHBOARD — "系统在变强吗？引擎活着吗？结果值得信吗？"
// ═══════════════════════════════════════════════════════════════════════
export async function loadDashboard(el) {
  let raw = null;
  let fetchErr = null;
  try {
    raw = await fetchJson('/api/tools/dashboard');
  } catch (e) {
    fetchErr = e.message || 'API error';
  }
  const d = unwrap(raw) || {};
  const p = d.performance || {};
  const st = d.state || {};
  const trend = d.trend || [];
  const topF = d.top_factors_used || [];
  const errDist = d.error_distribution || [];
  const fc = d.factor_counts || {};
  const audit = d.recent_audit || [];

  // Health digest — closed-loop daily summary
  let digest = null;
  try {
    const dg = await fetchJson('/api/tools/health-digest?hours=24');
    digest = dg?.data || null;
  } catch(e) {}

  if (fetchErr && !d.state) {
    el.innerHTML = `<div class="page-title">仪表盘</div>${errorBanner('Dashboard API 无法访问: ' + fetchErr)}<div class="c-muted">请检查 /api/tools/dashboard 是否正常</div>`;
    return;
  }
  const isAlive = st.worker_status === 'running' || st.worker_status === 'idle';
  const hasErrors = d.total_failures > 0;

  // Health digest card (closed-loop view)
  const digestCard = digest ? `
    <div class="card" style="margin-bottom:16px;border-left:3px solid var(--v2-primary)">
      <div class="card-title">闭环健康日报 <span class="c-muted" style="font-weight:400;font-size:10px">过去 24h</span></div>
      <div style="font-size:13px;margin:4px 0">${digest.summary_line || '-'}</div>
      <div class="two-col" style="margin-top:8px;grid-template-columns:1fr 1fr">
        <div class="card-kv">
          <div><span>新 pattern 样本</span><span>${digest.research_volume?.new_live_scans_24h || 0}</span></div>
          <div><span>Pattern 数据库</span><span>${digest.research_volume?.pattern_dbs || 0} 个</span></div>
          <div><span>总 pattern 数</span><span>${(digest.research_volume?.total_patterns || 0).toLocaleString()}</span></div>
          <div><span>24h 平仓</span><span>${digest.trade_volume?.closed_last_24h || 0}</span></div>
          <div><span>24h 胜率</span><span class="${(digest.trade_volume?.win_rate||0)>=0.5?'c-green':'c-red'}">${((digest.trade_volume?.win_rate || 0)*100).toFixed(0)}%</span></div>
          <div><span>24h 平均 EV</span><span class="${(digest.trade_volume?.avg_return_atr||0)>=0?'c-green':'c-red'}">${(digest.trade_volume?.avg_return_atr||0) >= 0 ? '+' : ''}${digest.trade_volume?.avg_return_atr || 0} ATR</span></div>
        </div>
        <div>
          ${(digest.rule_performance?.top_performers || []).length > 0 ? `
            <div class="c-muted" style="font-size:10px;margin-bottom:4px">最佳规则 (按 lifetime EV)</div>
            ${digest.rule_performance.top_performers.slice(0,3).map(r => `
              <div class="mini-row"><span class="c-primary">${r.rule_id}</span><span class="c-green">+${r.lifetime_ev} ATR (${r.lifetime_count} 次)</span></div>
            `).join('')}
          ` : '<div class="c-muted" style="font-size:11px;padding:4px">暂无已验证规则</div>'}
          ${(digest.rule_performance?.degrading || []).length > 0 ? `
            <div class="c-muted" style="font-size:10px;margin:8px 0 4px">⚠ 漂移中的规则</div>
            ${digest.rule_performance.degrading.slice(0,3).map(r => `
              <div class="mini-row"><span class="c-red">${r.rule_id}</span><span class="c-red">漂移 ${r.drift} ATR</span></div>
            `).join('')}
          ` : ''}
        </div>
      </div>
      ${(digest.alerts || []).length > 0 ? `
        <div style="margin-top:8px;padding-top:8px;border-top:1px dashed var(--v2-border)">
          ${digest.alerts.map(a => `
            <div class="mini-row" style="font-size:11px">
              <span class="${a.severity==='warning'?'c-red':'c-muted'}">${a.severity==='warning'?'⚠':'ℹ'} ${a.type}</span>
              <span>${esc(a.message)}</span>
            </div>
          `).join('')}
        </div>
      ` : ''}
    </div>
  ` : '';

  el.innerHTML = `
    <div class="page-title">仪表盘</div>

    ${digestCard}

    <!-- ROW 1: 4 强 KPI -->
    <div class="kpi-row">
      ${kpi('研究表现', `${p.avg_return>=0?'+':''}${p.avg_return||0}%`, p.avg_return>=0?'c-green':'c-red')}
      ${kpi('引擎状态', isAlive ? 'Gen '+st.current_generation : '离线', isAlive?'c-green':'c-red')}
      ${kpi('可部署策略', p.deployable||0, (p.deployable||0)>0?'c-green':'')}
      ${kpi('失败/错误', d.total_failures||0, hasErrors?'c-red':'c-muted')}
    </div>

    <!-- ROW 2: 研究摘要 + 引擎详情 -->
    <div class="two-col">
      <div class="card">
        <div class="card-title">研究表现</div>
        <div class="card-kv">
          <div><span>平均收益</span><span class="${p.avg_return>=0?'c-green':'c-red'}">${pnlV(p.avg_return||0)}%</span></div>
          <div><span>最佳收益</span><span class="c-green">+${p.best_return||0}%</span></div>
          <div><span>平均胜率</span><span>${p.avg_win_rate||0}%</span></div>
          <div><span>平均 Sharpe</span><span>${p.avg_sharpe||0}</span></div>
          <div><span>平均 RR</span><span>${p.avg_rr||0}</span></div>
          <div><span>最大回撤</span><span class="c-red">-${p.max_drawdown||0}%</span></div>
          <div><span>最佳综合分</span><span>${p.best_score||0}</span></div>
          <div><span>策略生成</span><span>${st.total_strategies_generated||0}</span></div>
          <div><span>回测结果</span><span>${st.total_results_produced||0}</span></div>
          <div><span>盈利结果</span><span class="c-green">${st.total_profitable||0}</span></div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">引擎 & 因子</div>
        <div class="card-kv">
          <div><span>状态</span><span class="${isAlive?'c-green':'c-red'}">${st.worker_status||'offline'}</span></div>
          <div><span>当前代</span><span>Gen ${st.current_generation||0}</span></div>
          <div><span>上次运行</span><span>${rel(st.last_run_at)}</span></div>
          <div><span>核心因子</span><span>${fc.core||0}</span></div>
          <div><span>候选因子</span><span>${fc.candidate||0}</span></div>
          <div><span>已验证因子</span><span class="c-green">${fc.validated||0}</span></div>
          <div><span>已测试因子</span><span>${fc.tested||0}</span></div>
          <div><span>排行榜</span><span>${d.leaderboard_size||0} 条</span></div>
        </div>
        ${st.last_error ? `<div class="card-alert">最近错误: ${esc(st.last_error)}</div>` : ''}
      </div>
    </div>

    <!-- ROW 3: 趋势 + 高价值因子 + 错误 -->
    <div class="three-col">
      <div class="card">
        <div class="card-title">进化趋势</div>
        ${trend.length === 0 ? '<div class="c-muted" style="font-size:12px">暂无数据</div>' : `
          <div class="trend-bars">
            ${trend.map(t => {
              const maxS = Math.max(...trend.map(x=>x.best_score||0), 0.01);
              const h = Math.max(6, ((t.avg_score||0) / maxS) * 50);
              const col = (t.avg_return||0) > 0 ? 'var(--v2-green)' : 'var(--v2-red)';
              return `<div class="trend-bar" title="Gen${t.generation}\nscore=${t.avg_score}\nret=${t.avg_return}%" style="height:${h}px;background:${col}"></div>`;
            }).join('')}
          </div>
        `}
      </div>

      <div class="card">
        <div class="card-title">高分因子</div>
        ${topF.length === 0 ? '<div class="c-muted" style="font-size:12px">暂无数据</div>' : `
          ${topF.slice(0,6).map(f => `
            <div class="mini-row"><span class="mono">${f.factor_id}</span><span class="c-muted">${f.usage_count}x</span></div>
          `).join('')}
        `}
      </div>

      <div class="card">
        <div class="card-title">错误分布</div>
        ${errDist.length === 0 ? '<div class="c-muted" style="font-size:12px">无错误</div>' : `
          ${errDist.map(e => `
            <div class="mini-row"><span class="c-red">${e.stage}</span><span>${e.count}</span></div>
          `).join('')}
        `}
      </div>
    </div>

    <!-- ROW 4: 最近操作 -->
    <div class="card" style="margin-top:12px">
      <div class="card-title">最近操作</div>
      <div class="log-list">
        ${audit.slice(-8).reverse().map(a => {
          const ts = a.timestamp ? new Date(a.timestamp * 1000).toLocaleTimeString() : '';
          return `<div class="log-row"><span class="c-muted">${ts}</span><span class="c-primary">[${a.actor}]</span> ${a.action} <span class="c-muted">${a.object_type}:${(a.object_id||'').slice(0,8)}</span></div>`;
        }).join('')}
      </div>
    </div>
  `;
}

// ═══════════════════════════════════════════════════════════════════════
// FACTORY — "怎么构建一个可测试的策略？"
// ═══════════════════════════════════════════════════════════════════════
export async function loadFactory(el) {
  const [cat, draftsRes, topVolRes] = await Promise.all([
    fetchJson('/api/runtime/catalog').catch(()=>({})),
    fetchJson('/api/tools/drafts?limit=10').catch(()=>({})),
    fetchJson('/api/top-volume?limit=30').catch(()=>({})),
  ]);
  const templates = cat.templates || [];
  const myDrafts = (draftsRes.data || []).slice(0, 10);
  const topVol = topVolRes.symbols || [];

  // Collapsible section helper
  const section = (id, title, content, open=false) => `
    <div class="builder-module ${open?'open':''}" data-module="${id}">
      <div class="builder-module-header">${title}</div>
      <div class="builder-module-body">${content}</div>
    </div>`;

  // Dynamic symbols: top volume coins from exchange
  const top10 = topVol.slice(0, 10);
  const top20 = topVol.slice(0, 20);
  const symOptions = topVol.length > 0 ? topVol : ['BTCUSDT','ETHUSDT','SOLUSDT','HYPEUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','SUIUSDT','PEPEUSDT'];
  const tfOptions = ['5m','15m','1h','4h','1d'];
  let conditionCount = 1;

  el.innerHTML = `
    <div class="page-title">策略工厂</div>

    <!-- Pattern → Strategy Auto-Generator -->
    <div class="card" style="margin-bottom:12px;border:1px solid var(--v2-primary)">
      <div class="card-title">⚡ 从结构自动生成</div>
      <div class="c-muted" style="font-size:11px;margin-bottom:10px">
        输入一条 2-touch 趋势线 → Pattern Engine 查历史相似结构 → 自动决策策略类型 + 参数
      </div>
      <div class="builder-row2">
        <div class="builder-section"><label>币种</label>
          <select id="pg-sym">${(topVol.slice(0,15).length?topVol.slice(0,15):['BTCUSDT','ETHUSDT','SOLUSDT']).map(s=>`<option>${s}</option>`).join('')}</select>
        </div>
        <div class="builder-section"><label>周期</label>
          <select id="pg-tf"><option>1h</option><option selected>4h</option></select>
        </div>
      </div>
      <div class="builder-row2">
        <div class="builder-section"><label>锚点1 bar_idx</label><input id="pg-a1i" type="number" value="100"/></div>
        <div class="builder-section"><label>锚点1 价格</label><input id="pg-a1p" type="number" step="any" value="3000"/></div>
      </div>
      <div class="builder-row2">
        <div class="builder-section"><label>锚点2 bar_idx</label><input id="pg-a2i" type="number" value="150"/></div>
        <div class="builder-section"><label>锚点2 价格</label><input id="pg-a2p" type="number" step="any" value="3100"/></div>
      </div>
      <div class="builder-row2">
        <div class="builder-section"><label>方向</label>
          <select id="pg-side"><option value="support">支撑</option><option value="resistance">阻力</option></select>
        </div>
        <div class="builder-section"><label>样本数 K</label><input id="pg-k" type="number" value="30"/></div>
      </div>
      <button class="btn-primary" id="pg-run" style="margin-top:8px">分析并生成策略</button>
      <div id="pg-result" style="margin-top:12px"></div>
    </div>

    <div class="two-col" style="grid-template-columns:280px 1fr">
      <!-- LEFT: Templates + Drafts -->
      <div>
        <div class="card">
          <div class="section-tabs">
            <button class="stab active" data-tab="tpl">模板 (${templates.length})</button>
            <button class="stab" data-tab="drafts">草稿 (${myDrafts.length})</button>
          </div>
          <div class="stab-content active" data-tab="tpl">
            ${templates.map(t => `
              <div class="list-item">
                <div class="list-item-top"><strong>${t.name}</strong><span class="badge badge-${t.risk_level}">${t.risk_level}</span></div>
                <div class="c-muted" style="font-size:10px">${t.description}</div>
                <div class="tag-row" style="margin-top:2px">${(t.default_trigger_modes||[]).map(m=>`<span class="tag">${T[m]||m}</span>`).join('')}</div>
              </div>
            `).join('')}
          </div>
          <div class="stab-content" data-tab="drafts">
            ${myDrafts.length===0 ? '<div class="empty">暂无草稿</div>' : ''}
            ${myDrafts.map(d => `<div class="list-item"><div class="list-item-top"><strong>${d.name||d.id?.slice(0,8)}</strong><span class="badge badge-muted">${d.status}</span></div><div class="c-muted" style="font-size:10px">${(d.symbols||[]).join(' ')} ${(d.timeframes||[]).join(' ')}</div></div>`).join('')}
          </div>
        </div>
      </div>

      <!-- RIGHT: Strategy Builder (modular) -->
      <div>
        <div class="card">
          <div class="card-title">策略构建器</div>
          <div class="builder-section"><label>策略名称</label><input id="b-name" placeholder="我的S/R反转策略" style="width:100%"/></div>

          ${section('market', '1. 市场范围', `
            <div class="builder-section"><label>币种（多选）</label>
              <div class="chip-group">${symOptions.map(s=>`<label class="chip"><input type="checkbox" name="b-sym" value="${s}" ${s==='BTCUSDT'?'checked':''}/> ${s.replace('USDT','')}</label>`).join('')}</div>
            </div>
            <div class="builder-row2" style="margin-top:8px">
              <div class="builder-section"><label>主周期</label><select id="b-main-tf">${tfOptions.map(t=>`<option ${t==='4h'?'selected':''}>${t}</option>`).join('')}</select></div>
              <div class="builder-section"><label>确认周期</label><select id="b-confirm-tf"><option value="">不使用</option>${tfOptions.map(t=>`<option>${t}</option>`).join('')}</select></div>
            </div>
            <div class="builder-section"><label>入场周期</label><select id="b-entry-tf"><option value="">同主周期</option>${tfOptions.map(t=>`<option>${t}</option>`).join('')}</select></div>
          `, true)}

          ${section('logic', '2. 逻辑骨架', `
            <div class="chip-group">
              <label class="chip"><input type="checkbox" name="b-logic" value="reversal" checked/> 反转</label>
              <label class="chip"><input type="checkbox" name="b-logic" value="breakout"/> 突破</label>
              <label class="chip"><input type="checkbox" name="b-logic" value="retest"/> 回测</label>
              <label class="chip"><input type="checkbox" name="b-logic" value="trend"/> 趋势</label>
              <label class="chip"><input type="checkbox" name="b-logic" value="scalp"/> 剥头皮</label>
            </div>
            <div class="builder-section" style="margin-top:6px"><label>组合方式</label>
              <select id="b-logic-combine"><option value="OR">OR - 满足任一</option><option value="AND">AND - 全部满足</option></select>
            </div>
          `)}

          ${section('conditions', '3. 条件系统', `
            <div id="cond-list">
              <div class="condition-card" data-idx="0">
                <div class="builder-row2">
                  <div class="builder-section"><label>指标</label><select name="cond-ind"><option>rsi</option><option>adx</option><option>bb_width</option><option>volume_ratio</option><option>ema_alignment</option><option>macd</option><option>stoch</option><option>cci</option><option>mfi</option></select></div>
                  <div class="builder-section"><label>条件</label><select name="cond-op"><option value="lt">&lt; 小于</option><option value="gt">&gt; 大于</option><option value="between">区间</option></select></div>
                </div>
                <div class="builder-row2">
                  <div class="builder-section"><label>阈值</label><input name="cond-thr" type="number" value="30" step="1"/></div>
                  <div class="builder-section"><label>参数(period)</label><input name="cond-period" type="number" value="14" step="1"/></div>
                </div>
              </div>
            </div>
            <button class="btn-secondary btn-sm" id="add-cond" style="margin-top:6px">+ 添加条件</button>
            <div class="builder-section" style="margin-top:6px"><label>条件组合</label>
              <select id="b-cond-combine"><option value="AND">AND - 全部满足</option><option value="OR">OR - 满足任一</option></select>
            </div>
          `)}

          ${section('entry', '4. 入场规则', `
            <div class="chip-group">
              <label class="chip"><input type="checkbox" name="b-entry" value="rejection" checked/> 反转拒绝</label>
              <label class="chip"><input type="checkbox" name="b-entry" value="failed_breakout" checked/> 假突破回收</label>
              <label class="chip"><input type="checkbox" name="b-entry" value="pre_limit"/> 限价预挂</label>
              <label class="chip"><input type="checkbox" name="b-entry" value="retest"/> 突破回测</label>
            </div>
            <div class="builder-section" style="margin-top:6px"><label>限价偏移</label><input id="b-offset" type="number" value="0" step="0.1" style="width:80px"/> %</div>
          `)}

          ${section('exit', '5. 出场规则', `
            <div class="builder-row2">
              <div class="builder-section"><label>止损方式</label><select id="b-stop"><option value="structure">结构止损</option><option value="fixed_pct">固定%</option><option value="atr">ATR止损</option></select></div>
              <div class="builder-section"><label>止损值</label><input id="b-stop-val" type="number" value="1.5" step="0.1"/></div>
            </div>
            <div class="builder-row2">
              <div class="builder-section"><label>止盈方式</label><select id="b-tp"><option value="rr">盈亏比</option><option value="fixed_pct">固定%</option><option value="trailing">跟踪止盈</option></select></div>
              <div class="builder-section"><label>止盈值</label><input id="b-tp-val" type="number" value="2.0" step="0.1"/></div>
            </div>
          `)}

          ${section('risk', '6. 风控', `
            <div class="builder-row2">
              <div class="builder-section"><label>每笔风险 %</label><input id="b-risk" type="number" value="1" step="0.1"/></div>
              <div class="builder-section"><label>最大并发</label><input id="b-maxpos" type="number" value="3" step="1"/></div>
            </div>
            <div class="builder-row2">
              <div class="builder-section"><label>最大日亏 %</label><input id="b-dailyloss" type="number" value="5" step="0.5"/></div>
              <div class="builder-section"><label>最大回撤 %</label><input id="b-maxdd" type="number" value="10" step="1"/></div>
            </div>
            <div class="builder-row2">
              <div class="builder-section"><label>最大连亏</label><input id="b-maxloss" type="number" value="5" step="1"/></div>
              <div class="builder-section"><label>回撤自动暂停</label><select id="b-autopause"><option value="true">是</option><option value="false">否</option></select></div>
            </div>
          `)}

          <div style="display:flex;gap:8px;margin-top:16px">
            <button class="btn-secondary" id="b-save">保存草稿</button>
            <button class="btn-primary" id="b-submit">提交模拟 →</button>
          </div>
        </div>
      </div>
    </div>
  `;

  // Module collapse/expand
  el.querySelectorAll('.builder-module-header').forEach(h => {
    // SAFE: fresh DOM per render, handler does not accumulate
    h.onclick = () => h.parentElement.classList.toggle('open');
  });

  // Tab switching
  el.querySelectorAll('.stab').forEach(btn => {
    // SAFE: fresh DOM per render, handler does not accumulate
    btn.onclick = () => {
      el.querySelectorAll('.stab').forEach(b=>b.classList.remove('active'));
      el.querySelectorAll('.stab-content').forEach(c=>c.classList.remove('active'));
      btn.classList.add('active');
      el.querySelector(`.stab-content[data-tab="${btn.dataset.tab}"]`)?.classList.add('active');
    };
  });

  // Add condition
  el.querySelector('#add-cond')?.addEventListener('click', () => {
    conditionCount++;
    const card = document.createElement('div');
    card.className = 'condition-card';
    card.dataset.idx = conditionCount;
    card.innerHTML = `
      <div class="builder-row2">
        <div class="builder-section"><label>指标</label><select name="cond-ind"><option>rsi</option><option>adx</option><option>bb_width</option><option>volume_ratio</option><option>ema_alignment</option><option>macd</option><option>stoch</option></select></div>
        <div class="builder-section"><label>条件</label><select name="cond-op"><option value="lt">&lt;</option><option value="gt">&gt;</option><option value="between">区间</option></select></div>
      </div>
      <div class="builder-row2">
        <div class="builder-section"><label>阈值</label><input name="cond-thr" type="number" value="50" step="1"/></div>
        <div class="builder-section"><label>参数</label><input name="cond-period" type="number" value="14" step="1"/></div>
      </div>
      <button class="btn-sm btn-danger js-cond-remove">删除</button>
    `;
    // Wire the remove button via addEventListener (Round 5/10 #19: no
    // inline onclick — XSS-shaped + breaks under strict CSP).
    const removeBtn = card.querySelector('.js-cond-remove');
    if (removeBtn) {
      removeBtn.addEventListener('click', () => card.remove());
    }
    el.querySelector('#cond-list')?.appendChild(card);
  });

  // Collect config and submit
  const collectConfig = () => {
    const syms = [...el.querySelectorAll('[name="b-sym"]:checked')].map(c=>c.value);
    const logics = [...el.querySelectorAll('[name="b-logic"]:checked')].map(c=>c.value);
    const entries = [...el.querySelectorAll('[name="b-entry"]:checked')].map(c=>c.value);

    // Conditions
    const conditions = [];
    el.querySelectorAll('.condition-card').forEach(card => {
      conditions.push({
        indicator: card.querySelector('[name="cond-ind"]')?.value || 'rsi',
        condition: card.querySelector('[name="cond-op"]')?.value || 'lt',
        threshold: parseFloat(card.querySelector('[name="cond-thr"]')?.value || '30'),
        params: { period: parseInt(card.querySelector('[name="cond-period"]')?.value || '14') },
        enabled: true,
      });
    });

    return {
      name: el.querySelector('#b-name')?.value || 'Untitled',
      source: 'manual',
      config: {
        market: {
          symbols: syms.length ? syms : ['BTCUSDT'],
          main_tf: el.querySelector('#b-main-tf')?.value || '4h',
          confirm_tf: el.querySelector('#b-confirm-tf')?.value || '',
          entry_tf: el.querySelector('#b-entry-tf')?.value || '',
        },
        logic_tags: logics.length ? logics : ['reversal'],
        logic_combine: el.querySelector('#b-logic-combine')?.value || 'OR',
        conditions,
        entry: {
          modes: entries.length ? entries : ['rejection'],
          logic_combine: 'OR',
          offset_pct: parseFloat(el.querySelector('#b-offset')?.value || '0'),
        },
        exit: {
          stop_type: el.querySelector('#b-stop')?.value || 'structure',
          stop_pct: parseFloat(el.querySelector('#b-stop-val')?.value || '1.5'),
          stop_atr_mult: 1.5,
          tp_type: el.querySelector('#b-tp')?.value || 'rr',
          rr_target: parseFloat(el.querySelector('#b-tp-val')?.value || '2.0'),
        },
        risk: {
          risk_per_trade: parseFloat(el.querySelector('#b-risk')?.value || '1') / 100,
          max_concurrent: parseInt(el.querySelector('#b-maxpos')?.value || '3'),
          max_daily_loss_pct: parseFloat(el.querySelector('#b-dailyloss')?.value || '5'),
          max_drawdown_pct: parseFloat(el.querySelector('#b-maxdd')?.value || '10'),
          max_consecutive_losses: parseInt(el.querySelector('#b-maxloss')?.value || '5'),
          auto_pause_on_dd: el.querySelector('#b-autopause')?.value === 'true',
        },
      },
    };
  };

  const submitStrategy = async (btn, status) => {
    const payload = collectConfig();
    if (!payload.name || payload.name === 'Untitled') {
      payload.name = `Strategy-${new Date().toISOString().slice(5,16).replace(/[T:]/g,'')}`;
    }
    const orig = btn.textContent;
    btn.disabled = true; btn.textContent = orig + '...';
    try {
      const resp = await fetchJson('/api/tools/strategies/create', { method: 'POST', body: payload, noCache: true });
      if (resp.ok) {
        btn.textContent = '已创建'; invalidateCachePrefix('/api/tools/drafts');
        setTimeout(()=>{btn.textContent=orig;btn.disabled=false;}, 1500);
      } else {
        btn.textContent = resp.error || '失败';
        setTimeout(()=>{btn.textContent=orig;btn.disabled=false;}, 1500);
      }
    } catch(e) {
      btn.textContent = '失败'; setTimeout(()=>{btn.textContent=orig;btn.disabled=false;}, 1500);
    }
  };

  el.querySelector('#b-save')?.addEventListener('click', function() { submitStrategy(this, 'draft'); });
  el.querySelector('#b-submit')?.addEventListener('click', function() { submitStrategy(this, 'pending_simulation'); });

  // If prefill payload exists (navigated from market page), populate and auto-run
  if (window.__factoryPrefill) {
    const p = window.__factoryPrefill;
    window.__factoryPrefill = null;
    const symSel = el.querySelector('#pg-sym');
    if (symSel) {
      let opt = [...symSel.options].find(o => o.value === p.symbol);
      if (!opt) {
        opt = new Option(p.symbol, p.symbol);
        symSel.add(opt);
      }
      symSel.value = p.symbol;
    }
    const tfSel = el.querySelector('#pg-tf');
    if (tfSel) {
      let tfOpt = [...tfSel.options].find(o => o.value === p.timeframe);
      if (!tfOpt) {
        tfOpt = new Option(p.timeframe, p.timeframe);
        tfSel.add(tfOpt);
      }
      tfSel.value = p.timeframe;
    }
    el.querySelector('#pg-a1i').value = p.anchor1_idx;
    el.querySelector('#pg-a1p').value = p.anchor1_price;
    el.querySelector('#pg-a2i').value = p.anchor2_idx;
    el.querySelector('#pg-a2p').value = p.anchor2_price;
    el.querySelector('#pg-side').value = p.side;
    // Auto-run after a short delay
    setTimeout(() => el.querySelector('#pg-run')?.click(), 300);
  }

  // Pattern Generator handler
  el.querySelector('#pg-run')?.addEventListener('click', async function() {
    const btn = this;
    const orig = btn.textContent;
    btn.disabled = true; btn.textContent = '分析中...';
    const slot = el.querySelector('#pg-result');
    slot.innerHTML = '<div class="c-muted">正在查询历史相似结构...</div>';
    try {
      const params = new URLSearchParams({
        symbol: el.querySelector('#pg-sym').value,
        timeframe: el.querySelector('#pg-tf').value,
        anchor1_idx: el.querySelector('#pg-a1i').value,
        anchor1_price: el.querySelector('#pg-a1p').value,
        anchor2_idx: el.querySelector('#pg-a2i').value,
        anchor2_price: el.querySelector('#pg-a2p').value,
        side: el.querySelector('#pg-side').value,
        k: el.querySelector('#pg-k').value,
        both_variants: 'true',
      });
      const resp = await fetchJson('/api/tools/patterns/recommend-strategies?' + params.toString(), { timeout: 30000, noCache: true });
      if (!resp.ok) {
        slot.innerHTML = `<div class="c-red">失败: ${esc(resp.error || resp.detail || 'unknown')}</div>`;
      } else {
        slot.innerHTML = renderPatternResult(resp.data);
        slot.querySelectorAll('[data-create-draft]').forEach(b => {
          // SAFE: fresh DOM per render, handler does not accumulate
          b.onclick = async function() {
            const draftJson = this.dataset.createDraft;
            const draft = JSON.parse(decodeURIComponent(draftJson));
            this.disabled = true; this.textContent = '创建中...';
            try {
              const r = await fetchJson('/api/tools/patterns/create-from-recommendation', {
                method: 'POST', body: { draft }, noCache: true,
              });
              if (r.ok) {
                this.textContent = '已创建 ✓';
                invalidateCachePrefix('/api/tools/drafts');
              } else {
                this.textContent = '失败'; this.disabled = false;
              }
            } catch(e) { this.textContent = '错误'; this.disabled = false; }
          };
        });
      }
    } catch (e) {
      slot.innerHTML = `<div class="c-red">错误: ${esc(e?.message || String(e))}</div>`;
    }
    btn.disabled = false; btn.textContent = orig;
  });
}

function renderPatternResult(data) {
  const dec = data.decision;
  const match = data.match;
  const stats = match.stats;
  const anomaly = match.anomaly || {};
  const decIcon = {
    'reversal': '↩️', 'breakout': '🚀', 'failed_breakout': '🔄',
    'no_trade': '❌', 'watch_only': '👀',
  }[dec.decision] || '•';
  const decClass = {
    'reversal': 'c-green', 'breakout': 'c-primary', 'failed_breakout': '',
    'no_trade': 'c-red', 'watch_only': 'c-muted',
  }[dec.decision] || '';

  let html = `
    <div class="card" style="background:rgba(42,53,72,0.2)">
      <div style="font-size:14px;font-weight:700;margin-bottom:6px">
        ${decIcon} 决策: <span class="${decClass}">${dec.decision.toUpperCase()}</span>
      </div>
      <div class="c-muted" style="font-size:11px;margin-bottom:8px">${esc(dec.reason)}</div>
      <div class="lb-metrics" style="font-size:11px">
        <span>样本 ${stats.sample_size}</span>
        <span>反弹 ${(stats.p_bounce*100).toFixed(0)}%</span>
        <span>破位 ${(stats.p_break*100).toFixed(0)}%</span>
        <span>EV ${stats.expected_value >= 0 ? '+' : ''}${stats.expected_value}ATR</span>
        <span>可信度 ${(stats.confidence*100).toFixed(0)}%</span>
        <span>稳定性 ${stats.overfit_flag}</span>
      </div>
      ${anomaly.is_anomaly ? '<div class="card-alert" style="margin-top:6px">⚠️ 异常结构 — 历史数据覆盖不足</div>' : ''}
    </div>
  `;

  if (data.drafts && data.drafts.length > 0) {
    html += '<div style="margin-top:10px;font-size:12px;font-weight:600">自动生成的策略草案:</div>';
    data.drafts.forEach(draft => {
      const cfg = draft.config;
      const draftParam = encodeURIComponent(JSON.stringify(draft));
      const approved = draft.approved;
      html += `
        <div class="list-item" style="background:rgba(42,53,72,0.1);border-radius:6px;padding:10px;margin-top:8px">
          <div class="list-item-top">
            <strong>${draft.name}</strong>
            ${approved ? '<span class="badge badge-green">验证通过</span>' : '<span class="badge badge-warn">验证未通过</span>'}
          </div>
          <div class="c-muted" style="font-size:10px;margin:2px 0">${draft.variant} | ${draft.decision_type}</div>
          <div class="lb-metrics" style="font-size:10px">
            <span>逻辑 ${cfg.logic_tags.join('+')}</span>
            <span>入场 ${cfg.entry.modes.join('/')}</span>
            <span>RR ${cfg.exit.rr_target}</span>
            <span>止损 ${cfg.exit.stop_type}</span>
            <span>风险 ${(cfg.risk.risk_per_trade*100).toFixed(2)}%</span>
          </div>
          <div class="c-muted" style="font-size:10px;margin-top:4px">验证: ${esc(draft.validation.reason)}</div>
          ${approved ? `<button class="btn-primary btn-sm" style="margin-top:6px" data-create-draft="${draftParam}">采纳此草案 → 创建</button>` : ''}
        </div>
      `;
    });
  } else {
    html += '<div class="c-muted" style="margin-top:10px;font-size:11px">不生成策略草案 — 历史数据不支持。</div>';
  }
  return html;
}

// ═══════════════════════════════════════════════════════════════════════
// LEADERBOARD — "哪些策略值得部署？为什么？"
// ═══════════════════════════════════════════════════════════════════════
export async function loadLeaderboard(el) {
  let lb = [], meta = {}, fetchErr = null;
  try {
    const resp = await fetchJson('/api/tools/leaderboard?limit=30');
    if (resp && resp.ok !== false) {
      lb = Array.isArray(resp.data) ? resp.data : (resp.entries || []);
      meta = resp.meta || {};
    } else {
      fetchErr = resp?.error || 'empty response';
    }
  } catch(e) {
    fetchErr = e.message || 'network error';
    // Fallback to evolution engine leaderboard (older format)
    try {
      const resp2 = await fetchJson('/api/runtime/leaderboard?limit=20');
      lb = resp2?.leaderboard || [];
      meta = { generation: resp2?.generation, total_profitable: resp2?.total_profitable };
      fetchErr = null;
    } catch {}
  }
  // Normalize field names — evolution uses total_return_pct, tools uses return_pct
  lb = lb.map(v => ({
    ...v,
    return_pct: v.return_pct ?? v.total_return_pct ?? 0,
    trade_count: v.trade_count ?? v.total_trades ?? 0,
    factor_ids: v.factor_ids ?? v.factors ?? [],
    id: v.id || v.variant_id || '',
  }));

  // Classify
  const deployable = lb.filter(v => v.deployment_eligible);
  const topRecent = [...lb].sort((a,b) => (b.generation||0) - (a.generation||0)).slice(0,10);
  const topStable = [...lb].filter(v => (v.max_drawdown_pct||99) < 5).sort((a,b) => (b.score||0) - (a.score||0));

  el.innerHTML = `
    <div class="page-title">排行榜 <span class="c-muted" style="font-size:13px;font-weight:400">Gen ${meta.generation||0} | ${lb.length} 策略 | ${meta.total_profitable||0} 盈利</span></div>
    ${fetchErr ? errorBanner('API 异常: ' + fetchErr) : ''}

    <div class="kpi-row">
      ${kpi('总策略', lb.length)}
      ${kpi('可部署', deployable.length, deployable.length>0?'c-green':'')}
      ${kpi('盈利', meta.total_profitable||0, 'c-green')}
      ${kpi('总回测', meta.total_results||0)}
    </div>

    <!-- Layer tabs -->
    <div class="section-tabs">
      <button class="stab active" data-tab="all">全部 (${lb.length})</button>
      <button class="stab" data-tab="deploy">可部署 (${deployable.length})</button>
      <button class="stab" data-tab="recent">最近 (${topRecent.length})</button>
      <button class="stab" data-tab="stable">低回撤 (${topStable.length})</button>
    </div>

    <div class="stab-content active" data-tab="all">${renderLbList(lb, el)}</div>
    <div class="stab-content" data-tab="deploy">${renderLbList(deployable, el)}</div>
    <div class="stab-content" data-tab="recent">${renderLbList(topRecent, el)}</div>
    <div class="stab-content" data-tab="stable">${renderLbList(topStable, el)}</div>
  `;

  // Tab switching
  el.querySelectorAll('.stab').forEach(btn => {
    // SAFE: fresh DOM per render, handler does not accumulate
    btn.onclick = () => {
      el.querySelectorAll('.stab').forEach(b=>b.classList.remove('active'));
      el.querySelectorAll('.stab-content').forEach(c=>c.classList.remove('active'));
      btn.classList.add('active');
      el.querySelector(`.stab-content[data-tab="${btn.dataset.tab}"]`)?.classList.add('active');
    };
  });

  // Expand toggle
  el.addEventListener('click', e => {
    const eb = e.target.closest('.expand-btn');
    if (eb) { document.getElementById(eb.dataset.expand)?.classList.toggle('hidden'); return; }
    const ab = e.target.closest('[data-action="add-live"]');
    if (ab) { doAction(ab, `/api/tools/live-drafts?entry_id=${ab.dataset.entry}&capital=15&risk_per_trade=0.01`, 'POST', el, loadLeaderboard); }
  });
}

function renderLbList(items) {
  if (!items.length) return '<div class="empty">暂无数据</div>';
  return items.map((v, i) => {
    const ret = v.return_pct ?? v.total_return_pct ?? 0;
    const trades = v.trade_count ?? v.total_trades ?? 0;
    const triggers = (v.trigger_modes||[]).map(m=>T[m]||m).join('+') || '-';
    const factors = (v.factor_ids||v.factors||[]).join(', ') || '-';
    const eligible = v.deployment_eligible;
    const lowSample = trades < 10;
    const eid = 'lb-' + (v.id||i);
    return `<div class="lb-card">
      <div class="lb-card-main">
        <span class="lb-rank">${i<3?['🥇','🥈','🥉'][i]:'#'+(i+1)}</span>
        <div class="lb-card-info">
          <div class="lb-card-top">
            <strong>${v.symbol} ${v.timeframe}</strong>
            <span class="${ret>=0?'c-green':'c-red'}" style="font-size:15px;font-weight:700">${ret>=0?'+':''}${ret}%</span>
          </div>
          <div class="lb-metrics">
            <span>WR ${v.win_rate}%</span>
            <span>Sharpe ${v.sharpe_ratio}</span>
            <span>DD -${v.max_drawdown_pct}%</span>
            <span>PF ${v.profit_factor}</span>
            <span>${trades}笔</span>
          </div>
          <div class="lb-tags">
            ${eligible ? badge('可部署','green') : ''}
            ${lowSample ? badge('样本不足','warn') : ''}
            ${v.generation ? badge('Gen'+v.generation,'muted') : ''}
            <span class="c-muted" style="font-size:10px">${v.strategy_name||''}</span>
          </div>
        </div>
        <div class="lb-actions">
          ${eligible ? `<button class="btn-primary btn-sm" data-action="add-live" data-entry="${v.id}">加入实盘库</button>` : ''}
          ${expandBtn(eid)}
        </div>
      </div>
      ${expandArea(eid, `
        <div class="card-kv" style="font-size:11px">
          <div><span>策略</span><span class="mono">${triggers}</span></div>
          <div><span>因子</span><span class="mono">${factors}</span></div>
          <div><span>strategy_id</span><span class="mono">${v.strategy_id||'-'}</span></div>
          <div><span>batch_id</span><span class="mono">${v.batch_id||'-'}</span></div>
          <div><span>sim_job</span><span class="mono">${v.simulation_job_id||'-'}</span></div>
          <div><span>来源</span><span>${v.source||'-'}</span></div>
        </div>
      `)}
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════════════════
// FACTORS — "哪些因子真的有用？哪些该晋升？哪些该淘汰？"
// ═══════════════════════════════════════════════════════════════════════
export async function loadFactors(el) {
  const [rankRes, coreRes, candRes, valRes] = await Promise.allSettled([
    fetchJson('/api/tools/factor-rankings'),
    fetchJson('/api/tools/factors?stage=core'),
    fetchJson('/api/tools/factors?stage=candidate'),
    fetchJson('/api/tools/factors?stage=validated'),
  ]);
  const rankings = (rankRes.status==='fulfilled' ? rankRes.value : {}).data || [];
  const coreM = (coreRes.status==='fulfilled' ? coreRes.value : {}).meta || {};
  const candM = (candRes.status==='fulfilled' ? candRes.value : {}).meta || {};
  const valM = (valRes.status==='fulfilled' ? valRes.value : {}).meta || {};
  const valFactors = (valRes.status==='fulfilled' ? valRes.value : {}).data || [];
  const candFactors = (candRes.status==='fulfilled' ? candRes.value : {}).data || [];

  const trendIcon = t => t==='improving'?'↑':t==='declining'?'↓':'→';
  const trendCls = t => t==='improving'?'c-green':t==='declining'?'c-red':'c-muted';

  el.innerHTML = `
    <div class="page-title">因子面板</div>

    <div class="kpi-row">
      ${kpi('核心', coreM.count||0)}
      ${kpi('候选', candM.count||0)}
      ${kpi('已验证', valM.count||0, (valM.count||0)>0?'c-green':'')}
      ${kpi('已测试', rankings.length)}
    </div>

    <div class="section-tabs">
      <button class="stab active" data-tab="ranking">排行</button>
      <button class="stab" data-tab="lifecycle">生命周期</button>
      <button class="stab" data-tab="candidates">候选 (${candM.count||0})</button>
    </div>

    <!-- Rankings tab -->
    <div class="stab-content active" data-tab="ranking">
      ${rankings.length === 0 ? '<div class="empty">暂无测试数据，运行 python run_agent.py</div>' : `
        <div class="factor-table">
          <div class="factor-header"><span style="flex:2">因子</span><span>分类</span><span>测试</span><span>交易</span><span>平均分</span><span>趋势</span><span>最佳</span></div>
          ${rankings.map((f,i) => `
            <div class="factor-row">
              <span style="flex:2"><strong>${i<3?['🥇','🥈','🥉'][i]:'#'+(i+1)}</strong> ${f.name||f.id}</span>
              <span class="tag">${f.category}</span>
              <span>${f.test_count}</span>
              <span>${f.total_trades}</span>
              <span style="font-weight:600;color:${f.avg_score>=0.5?'var(--v2-green)':f.avg_score>=0.3?'inherit':'var(--v2-red)'}">${f.avg_score}</span>
              <span class="${trendCls(f.score_trend)}">${trendIcon(f.score_trend)}</span>
              <span class="c-muted" style="font-size:10px">${f.best_symbol||'-'}</span>
            </div>
          `).join('')}
        </div>
      `}
    </div>

    <!-- Lifecycle tab -->
    <div class="stab-content" data-tab="lifecycle">
      <div class="card" style="margin-bottom:12px">
        <div class="card-title">已验证因子 (${valFactors.length})</div>
        ${valFactors.length===0 ? '<div class="c-muted" style="font-size:12px">暂无已验证因子</div>' : `
          ${valFactors.map(f => `
            <div class="list-item">
              <div class="list-item-top"><strong>${f.name||f.id}</strong>${badge('validated','green')}</div>
              <div class="c-muted" style="font-size:11px">测试 ${f.test_count||0} 次 | 平均分 ${f.avg_score||0} | 交易 ${f.total_trades||0}</div>
              ${f.promoted_at ? `<div class="c-muted" style="font-size:10px">晋升于 ${rel(f.promoted_at)}</div>` : ''}
            </div>
          `).join('')}
        `}
      </div>
      <div class="card">
        <div class="card-title">晋升路径</div>
        <div class="c-muted" style="font-size:12px;padding:8px 0">
          candidate → 测试 3+ 组合 → 10+ 交易 → 分数 ≥ 0.3 → validated → core
        </div>
      </div>
    </div>

    <!-- Candidates tab -->
    <div class="stab-content" data-tab="candidates">
      ${candFactors.length===0 ? '<div class="empty">暂无候选因子</div>' : `
        ${candFactors.map(f => `
          <div class="list-item">
            <div class="list-item-top"><strong>${f.name||f.id}</strong>${badge(f.source||'','muted')}</div>
            <div class="c-muted" style="font-size:11px">
              测试 ${f.test_count||0} 次 | 平均分 ${(f.avg_score||0).toFixed(3)} | 交易 ${f.total_trades||0}
              ${(f.test_count||0)>=3 && (f.total_trades||0)>=10 && (f.avg_score||0)>=0.3 ? badge('可晋升','green') : badge('不满足条件','warn')}
            </div>
          </div>
        `).join('')}
      `}
    </div>
  `;

  el.querySelectorAll('.stab').forEach(btn => {
    // SAFE: fresh DOM per render, handler does not accumulate
    btn.onclick = () => {
      el.querySelectorAll('.stab').forEach(b=>b.classList.remove('active'));
      el.querySelectorAll('.stab-content').forEach(c=>c.classList.remove('active'));
      btn.classList.add('active');
      el.querySelector(`.stab-content[data-tab="${btn.dataset.tab}"]`)?.classList.add('active');
    };
  });
}

// ═══════════════════════════════════════════════════════════════════════
// LIVE LIBRARY — "我的钱在哪？策略跑得怎样？需要我做什么？"
// ═══════════════════════════════════════════════════════════════════════
export async function loadLive(el) {
  // Fast local data first
  const [draftsRes, instancesRes] = await Promise.allSettled([
    fetchJson('/api/tools/live-drafts'),
    fetchJson('/api/tools/live-instances'),
  ]);
  const drafts = (draftsRes.status==='fulfilled' ? draftsRes.value : {}).data || [];
  const instances = (instancesRes.status==='fulfilled' ? instancesRes.value : {}).data || [];
  // Account fetch: strict 5s timeout — don't block page on Bitget API
  let acc = null;
  let accError = null;
  try {
    acc = await fetchJson('/api/live-execution/account?mode=live', { timeout: 5000 });
  } catch (e) {
    accError = e.message || 'Bitget 连接超时';
  }
  const running = instances.filter(i => i.running_status === 'running');
  const pending = drafts.filter(d => ['draft','pending_approval','approved'].includes(d.status));
  const totalCap = running.reduce((s,i) => s + (i.allocated_capital||0), 0);
  // Sum SIMULATED P&L (pattern_virtual_pnl) and REAL P&L
  // (realized_pnl_usd) separately. Round 10 #25: do NOT fall back to
  // the deprecated unified field that may carry stale values. Run
  // scripts/migrate_pnl_fields.py to migrate legacy instances.
  const totalPnlSim = instances.reduce((s,i) => s + (i.pattern_virtual_pnl ?? 0), 0);
  const totalPnlReal = instances.reduce((s,i) => s + (i.realized_pnl_usd ?? 0), 0);
  const totalPnl = totalPnlSim;  // backward-compat alias

  el.innerHTML = `
    <div class="page-title">实盘管理</div>

    <div class="kpi-row">
      ${kpi('账户权益', acc?.ok ? formatUsd(acc.total_equity) : '连接中...', acc?.ok?'':'c-muted')}
      ${kpi('策略运行中', running.length, running.length>0?'c-green':'c-muted')}
      ${kpi('实盘盈亏', pnlU(totalPnlReal), totalPnlReal>=0?'c-green':'c-red')}
      ${kpi('模拟盈亏 [虚拟]', pnlU(totalPnlSim), totalPnlSim>=0?'c-green':'c-red')}
      ${kpi('等待审批', pending.length, pending.length>0?'c-primary':'')}
    </div>

    <!-- 待审批区 -->
    <div class="card" style="margin-bottom:12px">
      <div class="card-title">待处理 (${pending.length})</div>
      ${pending.length === 0 ? '<div class="empty">暂无待处理策略 — 去排行榜选择表现好的策略加入</div>' : `
        ${pending.map(d => {
          const canApprove = d.status === 'draft' || d.status === 'pending_approval';
          const canDeploy = d.status === 'approved';
          const capOk = (d.capital_allocation||0) <= 20;
          const guardHint = !capOk ? `单策略上限$20，当前$${d.capital_allocation}` : '风控检查通过';
          return `<div class="list-item">
            <div class="list-item-top">
              <strong>${(d.symbols||[])[0]||'?'} ${(d.timeframes||[])[0]||'?'}</strong>
              ${d.status==='approved' ? badge('已通过审批，可部署','green')
                : d.status==='pending_approval' ? badge('审批被拒，需修改','warn')
                : badge('待审批','muted')}
            </div>
            <div class="lb-metrics" style="margin:4px 0">
              <span>投入 ${formatUsd(d.capital_allocation||0)}</span>
              <span>每笔风险 ${((d.risk_per_trade||0)*100).toFixed(1)}%</span>
              <span>${d.auto_submit?'自动下单':'仅预览'}</span>
            </div>
            ${(d.source_pattern_id || d.decision_rule) ? `
              <div style="background:rgba(42,53,72,0.15);padding:6px 8px;border-radius:4px;margin:4px 0;font-size:10px">
                <div><span class="c-muted">来源:</span> Pattern Engine | <span class="c-muted">规则:</span> <strong>${d.decision_rule || '-'}</strong> | <span class="c-muted">决策:</span> ${d.pattern_decision || '-'}</div>
                <div style="margin-top:2px"><span class="c-muted">预期 EV:</span> <strong class="${(d.pattern_ev||0)>0?'c-green':'c-red'}">${d.pattern_ev >= 0 ? '+' : ''}${d.pattern_ev || 0} ATR</strong></div>
                <div class="c-muted" style="margin-top:2px;font-family:monospace;font-size:9px">pattern: ${(d.source_pattern_id||'-').slice(0,12)}</div>
              </div>
            ` : ''}
            <div style="font-size:11px;color:${capOk?'var(--v2-green)':'var(--v2-red)'};margin:4px 0">
              风控: ${guardHint}
            </div>
            <div style="display:flex;gap:6px;margin-top:6px;align-items:center">
              ${canApprove ? `<button class="btn-primary btn-sm" data-action="approve-confirm" data-id="${d.id}" data-cap="${d.capital_allocation}">审批通过 →</button>` : ''}
              ${canDeploy ? `<button class="btn-primary btn-sm" data-action="deploy" data-url="/api/tools/live-drafts/${d.id}/deploy">启动策略</button>` : ''}
              <button class="btn-sm btn-danger" data-action="delete" data-url="/api/tools/live-drafts/${d.id}" data-method="DELETE">删除</button>
            </div>
            <div class="approve-confirm hidden" id="approve-${d.id}" style="margin-top:8px;padding:8px;background:rgba(42,53,72,0.2);border-radius:6px">
              <div style="font-size:12px;margin-bottom:6px">确认审批这个策略？</div>
              <div style="font-size:11px;color:var(--v2-muted);margin-bottom:6px">
                系统会检查: 单策略资金上限($20) · 总部署资金上限($50) · 最大同时运行数(5)
              </div>
              <div style="display:flex;gap:6px">
                <button class="btn-primary btn-sm" data-action="approve" data-url="/api/tools/live-drafts/${d.id}/approve">确认审批</button>
                <button class="btn-sm" data-action="cancel-approve" data-target="approve-${d.id}">取消</button>
              </div>
            </div>
          </div>`;
        }).join('')}
      `}
    </div>

    <!-- 运行中策略 -->
    <div class="card">
      <div class="card-title">运行中策略 (${instances.length})</div>
      ${instances.length === 0 ? '<div class="empty">暂无运行策略</div>' : `
        ${instances.map(i => {
          // Pattern-virtual P&L (simulation) vs realized USD (real exchange).
          // Total real trades is what counts. If 0 trades → no real P&L.
          const sim_v = i.pattern_virtual_pnl ?? 0;
          const real_v = i.realized_pnl_usd ?? 0;
          const trades = i.total_trades || 0;
          const dd = i.current_drawdown||0;
          const statusText = {running:'正常运行',paused:'已暂停',stopped:'已停止',retired:'已退役',error:'异常'}[i.running_status] || i.running_status;
          const statusType = i.running_status==='running'?'green':i.running_status==='error'?'red':'muted';
          const hasLineage = i.source_pattern_id || i.decision_rule;
          const realized = i.running_status === 'stopped' && i.outcome_written_back;
          // Display: real if any real trades happened, otherwise virtual badge
          const pnlCell = trades > 0
            ? `<span class="${pnlColorClass(real_v)} exec-pnl-real">实盘 ${pnlU(real_v)}</span>`
            : `<span class="${pnlColorClass(sim_v)}"><span class="exec-pnl-virtual">[虚拟]</span> ${pnlU(sim_v)}</span>`;
          return `<div class="list-item">
            <div class="list-item-top">
              <strong>${i.symbol || '策略'} ${i.timeframe || ''}</strong>
              ${badge(statusText, statusType)}
              ${i.pattern_decision ? badge(i.pattern_decision, 'muted') : ''}
            </div>
            <div class="lb-metrics" style="margin:4px 0">
              <span>投入 ${formatUsd(i.allocated_capital||0)}</span>
              ${pnlCell}
              <span>回撤 ${dd.toFixed(1)}%</span>
              <span>${trades} 笔</span>
              <span>运行 ${rel(i.started_at)}</span>
            </div>
            ${hasLineage ? `
              <div style="background:rgba(42,53,72,0.15);padding:6px 8px;border-radius:4px;margin:4px 0;font-size:10px">
                <div><span class="c-muted">预期 EV:</span> <strong>${i.pattern_ev_expected || '-'}</strong> ATR | <span class="c-muted">规则:</span> <strong>${i.decision_rule || '-'}</strong></div>
                ${realized ? `<div style="margin-top:2px"><span class="c-muted">实际 EV:</span> <strong class="${(i.realized_return_atr||0) > 0 ? 'c-green' : 'c-red'}">${i.realized_return_atr >= 0 ? '+' : ''}${i.realized_return_atr}</strong> ATR | <span class="c-muted">结果:</span> ${i.outcome_class || '-'} ${i.outcome_success ? '✓' : '✗'}</div>` : ''}
                <div class="c-muted" style="margin-top:2px;font-family:monospace;font-size:9px">pattern: ${(i.source_pattern_id || '-').slice(0,12)} | instance: ${(i.id || '').slice(0,8)}</div>
              </div>
            ` : ''}
            ${i.error ? `<div class="card-alert" style="margin:4px 0">${esc(i.error)}</div>` : ''}
            <div style="display:flex;gap:6px;margin-top:6px">
              ${i.running_status==='running' ? `<button class="btn-sm" data-action="pause" data-url="/api/tools/live-instances/${i.id}/pause">暂停策略</button>` : ''}
              ${i.running_status==='paused' ? `<button class="btn-primary btn-sm" data-action="resume" data-url="/api/tools/live-instances/${i.id}/resume">恢复运行</button>` : ''}
              ${['running','paused'].includes(i.running_status) ? `<button class="btn-sm btn-danger" data-action="stop" data-url="/api/tools/live-instances/${i.id}/stop">停止策略 (触发回写)</button>` : ''}
              ${i.running_status==='stopped' ? `<button class="btn-sm" data-action="retire" data-url="/api/tools/live-instances/${i.id}/retire">归档</button>` : ''}
            </div>
          </div>`;
        }).join('')}
      `}
    </div>
  `;

  // Event delegation
  el.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;

    // Show approve confirmation
    if (action === 'approve-confirm') {
      const panel = el.querySelector('#approve-' + btn.dataset.id);
      if (panel) panel.classList.remove('hidden');
      return;
    }
    if (action === 'cancel-approve') {
      el.querySelector('#' + btn.dataset.target)?.classList.add('hidden');
      return;
    }

    // All other actions
    await doAction(btn, btn.dataset.url, btn.dataset.method || 'POST', el, loadLive);
  });
}

// ═══════════════════════════════════════════════════════════════════════
// MONITOR — "系统正常吗？我需要做什么？"
// ═══════════════════════════════════════════════════════════════════════

// Translate system events to trader-friendly descriptions
const auditTranslate = {
  'strategy_generated': '新策略已生成',
  'strategy_status_changed': '策略状态变更',
  'simulation_started': '回测开始',
  'simulation_completed': '回测完成',
  'simulation_failures_recorded': '回测失败记录',
  'leaderboard_updated': '排行榜更新',
  'factor_created': '新因子创建',
  'factor_promoted': '因子晋升',
  'live_draft_created': '新部署草稿',
  'live_draft_approved': '部署审批通过',
  'live_draft_deleted': '草稿已删除',
  'live_instance_created': '策略已部署',
  'live_instance_paused': '策略已暂停',
  'live_instance_running': '策略恢复运行',
  'live_instance_stopped': '策略已停止',
  'live_instance_retired': '策略已归档',
  'pattern_driven_draft': '模式驱动草案',
  'writeback_failed': '回写失败',
  'agent_started': '研究引擎启动',
  'agent_stopped': '研究引擎停止',
  'generation_completed': '研究轮次完成',
  'generation_failed': '研究轮次失败',
};
const failStageTranslate = {
  'draft_load': '策略文件加载失败',
  'draft_parse': '策略格式错误',
  'market_data': '行情数据获取失败',
  'insufficient_trades': '交易样本不足',
  'backtest_runtime': '回测引擎异常',
  'snapshot_build': '市场分析异常',
};

export async function loadMonitor(el) {
  const [auditRes, failRes, stateRes, summRes, ruleRes] = await Promise.allSettled([
    fetchJson('/api/tools/audit?limit=50'),
    fetchJson('/api/tools/failures?limit=20'),
    fetchJson('/api/tools/agent-state'),
    fetchJson('/api/tools/failures/summary'),
    fetchJson('/api/tools/patterns/rule-effectiveness'),
  ]);
  const audit = (auditRes.status==='fulfilled' ? auditRes.value : {}).data || [];
  const failures = (failRes.status==='fulfilled' ? failRes.value : {}).data || [];
  const st = (stateRes.status==='fulfilled' ? stateRes.value : {}).data || {};
  const summary = (summRes.status==='fulfilled' ? summRes.value : {}).data || {};
  const ruleStats = (ruleRes.status==='fulfilled' ? ruleRes.value : {}).data || [];

  // Additional pattern engine data (live outcomes + batch build progress)
  let liveOutcomes = [];
  let batchProgress = null;
  try {
    const lo = await fetchJson('/api/tools/patterns/live-outcomes?limit=30');
    liveOutcomes = lo?.data || [];
  } catch(e) {}
  try {
    const bp = await fetchJson('/api/tools/patterns/batch-progress');
    batchProgress = bp?.data || null;
  } catch(e) {}

  // Extract writeback events from audit log (legacy format)
  const writebacks = audit.filter(a => a.action === 'pattern_writeback_completed' || (a.action === 'live_instance_stopped' && a.details && a.details.writeback));
  const engineOk = st.worker_status === 'idle' || st.worker_status === 'running';
  const hasErrors = (summary.total || 0) > 0;
  const needsAction = !!st.last_error || (summary.total || 0) > 10;

  el.innerHTML = `
    <div class="page-title">系统监控 ${needsAction ? badge('需要处理','warn') : badge('一切正常','green')}</div>

    <div class="kpi-row">
      ${kpi('研究引擎', engineOk ? '正常运行' : '已停止', engineOk ? 'c-green' : 'c-red')}
      ${kpi('已完成轮次', st.current_generation || 0)}
      ${kpi('最近运行', rel(st.last_run_at))}
      ${kpi('异常数量', summary.total || 0, hasErrors ? 'c-red' : 'c-muted')}
    </div>

    ${st.last_error ? `<div class="card-alert" style="margin-bottom:12px">最近错误: ${esc(st.last_error)}</div>` : ''}

    <div class="two-col">
      <div>
        <!-- 系统状态 -->
        <div class="card" style="margin-bottom:12px">
          <div class="card-title">研究引擎</div>
          <div class="card-kv">
            <div><span>状态</span><span class="${engineOk ? 'c-green' : 'c-red'}">${engineOk ? '正在研究新策略' : '引擎未运行'}</span></div>
            <div><span>已完成</span><span>${st.current_generation || 0} 轮研究</span></div>
            <div><span>生成策略</span><span>${st.total_strategies_generated || 0} 个</span></div>
            <div><span>回测结果</span><span>${st.total_results_produced || 0} 个</span></div>
            <div><span>盈利结果</span><span class="c-green">${st.total_profitable || 0} 个</span></div>
          </div>
        </div>

        <!-- 异常摘要 -->
        ${hasErrors ? `
          <div class="card">
            <div class="card-title">异常摘要</div>
            ${(summary.by_stage || []).map(s => `
              <div class="mini-row">
                <span class="${s.count > 5 ? 'c-red' : 'c-muted'}">${failStageTranslate[s.stage] || s.stage}</span>
                <span>${s.count} 次</span>
              </div>
            `).join('')}
            <div class="c-muted" style="font-size:10px;margin-top:6px">
              ${(summary.by_stage || []).some(s => s.stage === 'backtest_runtime') ? '⚠️ 回测引擎异常需关注' : '大部分是数据类异常，可忽略'}
            </div>
          </div>
        ` : ''}
      </div>

      <!-- 最近动态 + 规则表现 -->
      <div>
        <div class="card">
          <div class="section-tabs">
            <button class="stab active" data-tab="rules">规则表现 (${ruleStats.length})</button>
            <button class="stab" data-tab="outcomes">实盘结果 (${liveOutcomes.length})</button>
            <button class="stab" data-tab="writebacks">闭环回写 (${writebacks.length})</button>
            <button class="stab" data-tab="dbbuild">数据库建库 ${batchProgress ? `(${batchProgress.completed_jobs}/${batchProgress.total_jobs})` : ''}</button>
            <button class="stab" data-tab="activity">最近动态 (${audit.length})</button>
            <button class="stab" data-tab="errors">异常 (${failures.length})</button>
          </div>

          <!-- Live Outcomes Tab -->
          <div class="stab-content" data-tab="outcomes">
            ${liveOutcomes.length === 0 ? '<div class="c-muted" style="padding:8px">暂无实盘结果 — 需要实例停止后触发回写</div>' : ''}
            ${liveOutcomes.map(o => {
              const cls = o.outcome_class || '-';
              const confirmed = o.pattern_prediction_confirmed ? '✓' : '✗';
              const ret = o.realized_return_atr || 0;
              const retCls = ret > 0 ? 'pnl-pos' : 'pnl-neg';
              return `<div class="list-item" style="padding:8px 0;font-size:11px">
                <div class="list-item-top">
                  <strong>${o.symbol} ${o.timeframe}</strong>
                  <span class="${retCls}"><strong>${ret >= 0 ? '+' : ''}${ret} ATR</strong></span>
                </div>
                <div class="c-muted" style="font-size:10px;margin-top:2px">
                  ${cls} | 预测验证: ${confirmed} | 规则: ${o.origin_decision_rule || '-'} | ${o.bars_held || 0} bars
                </div>
                <div class="c-muted" style="font-size:9px;font-family:monospace">
                  instance: ${(o.live_instance_id||'').slice(0,8)} | pattern: ${(o.origin_pattern_id||'').slice(0,12)}
                </div>
              </div>`;
            }).join('')}
          </div>

          <!-- DB Build Progress Tab -->
          <div class="stab-content" data-tab="dbbuild">
            ${!batchProgress ? '<div class="c-muted" style="padding:8px">暂无建库任务</div>' : `
              <div class="card-kv">
                <div><span>状态</span><span class="${batchProgress.status==='running'?'c-primary':batchProgress.status==='completed'?'c-green':'c-muted'}">${batchProgress.status}${batchProgress.is_running ? ' (运行中)' : ''}</span></div>
                <div><span>进度</span><span>${batchProgress.completed_jobs}/${batchProgress.total_jobs}</span></div>
                <div><span>失败</span><span class="${batchProgress.failed_jobs>0?'c-red':''}">${batchProgress.failed_jobs}</span></div>
                <div><span>累计模式</span><span class="c-green">${batchProgress.total_patterns?.toLocaleString() || 0}</span></div>
                ${batchProgress.current_job ? `<div><span>当前</span><span class="c-primary">${batchProgress.current_job}</span></div>` : ''}
                ${batchProgress.started_at ? `<div><span>开始时间</span><span>${rel(batchProgress.started_at)}</span></div>` : ''}
              </div>
              ${batchProgress.status === 'running' ? `
                <div style="background:rgba(42,53,72,0.2);height:6px;border-radius:3px;margin:8px 0;overflow:hidden">
                  <div style="background:var(--v2-primary);height:100%;width:${Math.round((batchProgress.completed_jobs/Math.max(batchProgress.total_jobs,1))*100)}%;transition:width .3s"></div>
                </div>
              ` : ''}
              ${(batchProgress.results||[]).slice(-5).reverse().map(r => `
                <div class="mini-row"><span class="c-muted">${r.symbol_timeframe}</span><span>${r.patterns} patterns</span></div>
              `).join('')}
              ${(batchProgress.errors||[]).length > 0 ? `
                <div class="card-alert" style="margin-top:6px">错误 ${batchProgress.errors.length} 条: ${esc(batchProgress.errors.slice(-2).map(e => e.job + ': ' + (e.error||'').slice(0,50)).join(', '))}</div>
              ` : ''}
              <button class="btn-secondary btn-sm" style="margin-top:8px" data-action="start-batch-build">重新建库 (20x4)</button>
              <button class="btn-secondary btn-sm" style="margin-top:8px" data-action="process-recompute">处理重算标记</button>
            `}
          </div>

          <!-- Rule Effectiveness Tab -->
          <div class="stab-content active" data-tab="rules">
            ${ruleStats.length === 0 ? '<div class="c-muted" style="padding:8px">暂无规则统计 — 需要至少一个实盘实例停止后才有数据</div>' : ''}
            ${ruleStats.map(r => {
              const evCls = r.live_expected_value >= 0.5 ? 'pnl-pos' : r.live_expected_value >= 0 ? '' : 'pnl-neg';
              const wrCls = r.live_win_rate >= 0.6 ? 'pnl-pos' : r.live_win_rate >= 0.4 ? '' : 'pnl-neg';
              return `<div class="list-item" style="padding:8px 0">
                <div class="list-item-top">
                  <strong>${r.rule_id}</strong>
                  <span class="badge badge-muted">${r.decision_type}</span>
                </div>
                <div class="lb-metrics" style="font-size:10px">
                  <span>实盘次数 <strong>${r.live_count}</strong></span>
                  <span class="${wrCls}">胜率 ${(r.live_win_rate*100).toFixed(0)}%</span>
                  <span>平均收益 ${r.live_avg_return_atr}ATR</span>
                  <span>平均回撤 ${r.live_avg_drawdown_atr}ATR</span>
                  <span class="${evCls}"><strong>实盘EV ${r.live_expected_value >= 0 ? '+' : ''}${r.live_expected_value}ATR</strong></span>
                </div>
              </div>`;
            }).join('')}
          </div>

          <!-- Writeback History Tab -->
          <div class="stab-content" data-tab="writebacks">
            ${writebacks.length === 0 ? '<div class="c-muted" style="padding:8px">暂无闭环回写记录</div>' : ''}
            ${writebacks.slice(-20).reverse().map(a => {
              const ts = a.timestamp ? new Date(a.timestamp * 1000).toLocaleString() : '';
              const wb = a.details?.writeback || {};
              const ruleUpdate = wb.rule_update || {};
              return `<div class="list-item" style="padding:8px 0;font-size:11px">
                <div class="list-item-top">
                  <strong>实例 ${(a.object_id||'').slice(0,8)}</strong>
                  <span class="c-muted">${ts}</span>
                </div>
                <div class="c-muted" style="font-size:10px;margin-top:2px">
                  ${wb.symbol || ''} ${wb.timeframe || ''} | 决策: ${wb.decision_type || '-'} | 规则: ${wb.rule_id || '-'}
                </div>
                ${ruleUpdate.live_count ? `<div class="c-primary" style="font-size:10px">→ ${ruleUpdate.rule_id}: 累计 ${ruleUpdate.live_count} 次, EV ${ruleUpdate.live_ev}</div>` : ''}
                ${wb.pattern_writeback?.ok ? `<div class="c-green" style="font-size:10px">→ pattern DB 已更新</div>` : wb.pattern_writeback?.error ? `<div class="c-red" style="font-size:10px">pattern DB 错误: ${esc(wb.pattern_writeback.error)}</div>` : ''}
              </div>`;
            }).join('')}
          </div>

          <div class="stab-content" data-tab="activity">
            <div class="log-list">
              ${audit.slice(-30).reverse().map(a => {
                const ts = a.timestamp ? new Date(a.timestamp * 1000).toLocaleTimeString() : '';
                const desc = auditTranslate[a.action] || a.action;
                const isImportant = ['live_instance_created','live_draft_approved','live_instance_stopped','generation_failed','factor_promoted'].includes(a.action);
                return `<div class="log-row${isImportant ? ' c-primary' : ''}"><span class="c-muted">${ts}</span> ${desc}</div>`;
              }).join('')}
            </div>
          </div>
          <div class="stab-content" data-tab="errors">
            <div class="log-list">
              ${failures.length === 0 ? '<div class="c-muted" style="padding:8px">暂无异常</div>' : ''}
              ${failures.map(f => {
                const stageDesc = failStageTranslate[f.stage] || f.stage;
                return `<div class="log-row"><span class="c-red">${esc(stageDesc)}</span> <span class="c-muted">${esc(f.symbol || '')} | ${esc(f.error || '')}</span></div>`;
              }).join('')}
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  el.querySelectorAll('.stab').forEach(btn => {
    // SAFE: fresh DOM per render, handler does not accumulate
    btn.onclick = () => {
      el.querySelectorAll('.stab').forEach(b => b.classList.remove('active'));
      el.querySelectorAll('.stab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      el.querySelector(`.stab-content[data-tab="${btn.dataset.tab}"]`)?.classList.add('active');
    };
  });

  // Action buttons in dbbuild tab
  el.querySelector('[data-action="start-batch-build"]')?.addEventListener('click', async function() {
    const b = this;
    const orig = b.textContent;
    b.disabled = true; b.textContent = '启动中...';
    try {
      await fetchJson('/api/tools/patterns/batch-build', {
        method: 'POST',
        body: {
          symbols: ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","HYPEUSDT","DOGEUSDT","PEPEUSDT","ADAUSDT","TAOUSDT","SUIUSDT","LINKUSDT","AVAXUSDT","BNBUSDT","DOTUSDT","ATOMUSDT","NEARUSDT","UNIUSDT","AAVEUSDT","OPUSDT","ARBUSDT"],
          timeframes: ["15m","1h","4h","1d"],
          days: 730
        },
        noCache: true,
      });
      b.textContent = '已启动';
      setTimeout(() => loadMonitor(el), 2000);
    } catch(e) {
      b.textContent = '失败';
    } finally {
      // Always restore the button after 2s, regardless of success/failure.
      // Without this, a successful batch-build leaves the button stuck on
      // "已启动" forever, so the user can't restart.
      setTimeout(() => { b.textContent = orig; b.disabled = false; }, 2000);
    }
  });

  // Auto-scroll any audit log lists to the bottom on render so the most
  // recent entries are visible without manual scrolling.
  el.querySelectorAll('.log-list, [data-tab="activity"] .stab-content, [data-tab="errors"] .stab-content')
    .forEach(node => { try { node.scrollTop = node.scrollHeight; } catch {} });

  el.querySelector('[data-action="process-recompute"]')?.addEventListener('click', async function() {
    const b = this;
    b.disabled = true; b.textContent = '处理中...';
    try {
      const r = await fetchJson('/api/tools/patterns/process-recompute', { method: 'POST', noCache: true });
      b.textContent = `已处理 ${r.data?.processed || 0}`;
      setTimeout(() => { b.textContent = '处理重算标记'; b.disabled = false; }, 2000);
    } catch(e) {
      b.textContent = '失败'; b.disabled = false;
    }
  });
}

function monitorRow(i) {
  const c=i.config, s=i.status;
  const acct = s.paper_state?.account || {};
  const isLive = c.live_mode === 'live';
  // Show live → real P&L, paper → simulated [虚拟] P&L
  // SAFE: legacy alias fallback for old PaperAccountSummary (sim only)
  const simP = acct.realized_pnl_sim ?? acct.realized_pnl ?? 0;
  const realP = acct.realized_pnl_usd ?? 0;
  const p = isLive ? realP : simP;
  const pnlBadge = isLive
    ? `<span class="${pnlColorClass(p)} exec-pnl-real">${pnlU(p)}</span>`
    : `<span class="${pnlColorClass(p)}"><span class="exec-pnl-virtual">[虚拟]</span> ${pnlU(p)}</span>`;
  return `<div class="monitor-row">
    <span class="monitor-badge ${isLive?'badge-live':'badge-paper'}">${isLive?'实盘':'模拟'}</span>
    <span>${c.symbol} ${c.timeframe}</span>
    <span>bar ${s.last_processed_bar||'-'}</span>
    ${pnlBadge}
    <span class="c-muted" style="font-size:10px">${rel(s.last_tick_at)}</span>
  </div>`;
}

// Keep loadSimulate for backward compat (not in nav)
export async function loadSimulate(el) { el.innerHTML = '<div class="empty">已合并到仪表盘</div>'; }
