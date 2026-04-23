// frontend/js/workbench/decision_rail.js
// 4 cards: Market State · Current Setup · Risk Gate · Trade Candidate

import { $, setHtml, esc } from '../util/dom.js';
import { marketState } from '../state/market.js';
import { subscribe } from '../util/events.js';
import { fetchJson } from '../util/fetch.js';

let pollTimer = null;
// Round 11 P0-3: cold-cache snapshot can take 4-5 s (evolved detector +
// data fetch). The previous 4000 ms was shorter than the API itself, so
// every first-load decision rail call timed out and showed "加载中".
const DECISION_RAIL_TIMEOUT_MS = 30000;

async function fetchAll(symbol, interval) {
  // P0 2026-04-23: noCache:true per PRINCIPLES.md P12. This rail polls
  // every 30s for user's real-money decision making; the 30s default
  // fetch cache would show data up to 60s old (30s rail poll + 30s
  // fetch cache) — user would be looking at stale S/R and risk state.
  const [structure, risk, snapshot] = await Promise.all([
    fetchJson(`/api/market/structure-summary?symbol=${symbol}&interval=${interval}`, { timeout: DECISION_RAIL_TIMEOUT_MS, noCache: true }).catch(() => null),
    fetchJson('/api/agent/risk-state', { timeout: DECISION_RAIL_TIMEOUT_MS, noCache: true }).catch(() => null),
    fetchJson(`/api/strategy/snapshot?symbol=${symbol}&interval=${interval}&analysis_bars=200`, { timeout: DECISION_RAIL_TIMEOUT_MS, noCache: true }).catch(() => null),
  ]);
  return { structure, risk, snapshot };
}

function cardMarketState(structure) {
  if (!structure || structure.error) {
    return `<div class="dr-card"><h3>市场状态</h3><p class="muted">${esc(structure?.error || '加载中...')}</p></div>`;
  }
  const trendLabel = structure.trend_label === 'UPTREND' ? '上升趋势' : structure.trend_label === 'DOWNTREND' ? '下降趋势' : '震荡';
  const trendClass = structure.trend_label === 'UPTREND' ? 'pnl-pos' : structure.trend_label === 'DOWNTREND' ? 'pnl-neg' : '';
  return `
    <div class="dr-card">
      <h3>市场状态</h3>
      <div class="dr-row">
        <span class="dr-label">趋势</span>
        <span class="dr-value ${trendClass}"><strong>${trendLabel}</strong> ${structure.trend_slope_pct > 0 ? '+' : ''}${structure.trend_slope_pct}%</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">MA 排列</span>
        <span class="dr-value">${{'BULL_ORDERED':'多头排列','BEAR_ORDERED':'空头排列','MIXED':'混合','NEUTRAL':'中性'}[structure.ma_alignment] || structure.ma_alignment}</span>
      </div>
      <div class="dr-row">
        <span class="dr-label">Ribbon</span>
        <span class="dr-value">${structure.ribbon_score}/${structure.ribbon_max}</span>
      </div>
      ${structure.nearest_support ? `
        <div class="dr-row">
          <span class="dr-label">最近支撑</span>
          <span class="dr-value pnl-pos">$${structure.nearest_support} (${structure.distance_to_support_pct}%)</span>
        </div>
      ` : ''}
      ${structure.nearest_resistance ? `
        <div class="dr-row">
          <span class="dr-label">最近阻力</span>
          <span class="dr-value pnl-neg">$${structure.nearest_resistance} (+${structure.distance_to_resistance_pct}%)</span>
        </div>
      ` : ''}
    </div>
  `;
}

function cardRegime(snapshot) {
  const regime = snapshot?.snapshot?.market_regime;
  if (!regime) return '';

  const regimeLabel = { trending: '趋势', ranging: '震荡', compressed: '压缩', breakout: '突破' }[regime.regime] || regime.regime;
  const dirLabel = { up: '上行', down: '下行', neutral: '中性' }[regime.trend_direction] || '-';
  const volLabel = { expanding: '扩张', normal: '正常', compressed: '压缩' }[regime.volatility_state] || '-';
  const dirClass = regime.trend_direction === 'up' ? 'pnl-pos' : regime.trend_direction === 'down' ? 'pnl-neg' : '';

  return `
    <div class="dr-card">
      <h3>市场政权</h3>
      <div class="dr-row"><span class="dr-label">政权</span><span class="dr-value"><strong>${regimeLabel}</strong></span></div>
      <div class="dr-row"><span class="dr-label">方向</span><span class="dr-value ${dirClass}">${dirLabel} (${(regime.trend_strength * 100).toFixed(0)}%)</span></div>
      <div class="dr-row"><span class="dr-label">ADX</span><span class="dr-value">${regime.adx}</span></div>
      <div class="dr-row"><span class="dr-label">波动</span><span class="dr-value">${volLabel} (${regime.volatility_ratio.toFixed(2)}x)</span></div>
      <div class="dr-row"><span class="dr-label">结构</span><span class="dr-value">${(regime.structure_score * 100).toFixed(0)}%</span></div>
      <div class="dr-row"><span class="dr-label">区间度</span><span class="dr-value">${(regime.range_score * 100).toFixed(0)}%</span></div>
    </div>
  `;
}

function cardSRZones(snapshot, currentPrice) {
  const allZones = snapshot?.snapshot?.horizontal_zones || [];
  const signals = snapshot?.snapshot?.signals || [];
  // Filter out stale zones far from current price
  const price = currentPrice || 0;
  const zones = allZones.filter(z => {
    if (price > 0 && z.price_center > 0) {
      const distPct = Math.abs(z.price_center - price) / price * 100;
      if (distPct > 8) return false;
    }
    return z.strength >= 25;
  }).sort((a, b) => b.strength - a.strength);

  if (!zones.length) {
    return `<div class="dr-card"><h3>S/R 区域</h3><p class="muted">暂无检测到区域</p></div>`;
  }

  const sup = zones.filter(z => z.side === 'support').slice(0, 2);
  const res = zones.filter(z => z.side === 'resistance').slice(0, 2);

  const zoneRow = (z) => {
    const dist = price > 0 ? ((z.price_center - price) / price * 100).toFixed(1) : '?';
    const color = z.side === 'support' ? 'pnl-pos' : 'pnl-neg';
    const sc = z.strength_components || {};
    const scoreDetail = [
      sc.touch_score != null ? `触${(sc.touch_score * 100).toFixed(0)}` : '',
      sc.reaction_score != null ? `反${(sc.reaction_score * 100).toFixed(0)}` : '',
      sc.trend_score != null ? `势${(sc.trend_score * 100).toFixed(0)}` : '',
      sc.volume_failure_score != null ? `量${(sc.volume_failure_score * 100).toFixed(0)}` : '',
    ].filter(Boolean).join(' ');
    return `
      <div class="dr-row">
        <span class="dr-label ${color}">${z.side === 'support' ? '支撑' : '阻力'} ${z.touches}t <small>${Math.round(z.strength)}</small></span>
        <span class="dr-value">$${z.price_center.toFixed(2)} <small>(${dist}%)</small></span>
      </div>
      ${scoreDetail ? `<div class="dr-row"><span class="dr-label muted" style="font-size:9px">${scoreDetail}</span></div>` : ''}
    `;
  };

  const sigRows = signals.slice(0, 2).map(s => `
    <div class="dr-row" style="background:rgba(255,235,59,0.08);margin:2px 0;padding:2px 4px;border-radius:3px;">
      <span class="dr-label" style="color:#fbbf24">${s.direction === 'long' ? '做多' : '做空'}</span>
      <span class="dr-value" style="color:#fbbf24">$${s.entry_price.toFixed(2)} RR=${s.risk_reward.toFixed(1)}</span>
    </div>
  `).join('');

  return `
    <div class="dr-card">
      <h3>S/R 区域 <small style="color:var(--v2-muted)">${zones.length}个</small></h3>
      ${res.map(zoneRow).join('')}
      ${sup.map(zoneRow).join('')}
      ${sigRows || '<div class="dr-row"><span class="dr-label muted">等待入场信号...</span></div>'}
    </div>
  `;
}

function cardRiskGate(risk) {
  if (!risk) return `<div class="dr-card"><h3>风控</h3><p class="muted">加载中...</p></div>`;
  const stateLabel = { NORMAL: '正常', WATCH: '观察', COOLDOWN: '冷却', HALTED: '停止' }[risk.state] || risk.state;
  const stateClass = { NORMAL: 'pnl-pos', HALTED: 'pnl-neg' }[risk.state] || '';

  const m = risk.meters || {};
  const miniMeter = (label, obj) => {
    if (!obj) return '';
    const pct = obj.max > 0 ? Math.min(100, (obj.current / obj.max) * 100) : 0;
    const cls = pct < 50 ? 'green' : pct < 80 ? 'yellow' : 'red';
    return `
      <div class="dr-meter ${cls}">
        <div class="dr-meter-label">${label}: ${obj.current}${obj.unit} / ${obj.max}${obj.unit}</div>
        <div class="dr-meter-bar"><div class="dr-meter-fill" style="width:${pct}%"></div></div>
      </div>
    `;
  };

  return `
    <div class="dr-card">
      <h3>风控</h3>
      <div class="dr-row">
        <span class="dr-label">状态</span>
        <span class="dr-value ${stateClass}"><strong>${stateLabel}</strong></span>
      </div>
      ${risk.state_reason ? `<div class="dr-reason">${risk.state_reason}</div>` : ''}
      ${miniMeter('敞口', m.exposure)}
      ${miniMeter('日亏损', m.daily_loss)}
      ${miniMeter('回撤', m.drawdown)}
      ${miniMeter('持仓', m.positions)}
      ${risk.cooldown_remaining_sec > 0 ? `
        <div class="dr-row">
          <span class="dr-label">冷却中</span>
          <span class="dr-value">${risk.cooldown_remaining_sec}秒</span>
        </div>
      ` : ''}
    </div>
  `;
}

async function cardPatternStats(snapshot, symbol, interval, selectedLineId = null) {
  // Find the best candidate line and query the pattern engine for historical stats
  const lines = [
    ...(snapshot?.snapshot?.active_lines || []),
    ...(snapshot?.snapshot?.candidate_lines || []),
  ].filter(l => !l.is_invalidated && !l.invalidation_reason && (l.display_class === 'primary' || l.display_class === 'secondary'));

  if (lines.length === 0) {
    return '<div class="dr-card"><h3>历史概率</h3><p class="muted">暂无候选趋势线</p></div>';
  }

  // Pick highest-priority line (or selected)
  lines.sort((a, b) => {
    const stateA = a.state === 'confirmed' ? 0 : 1;
    const stateB = b.state === 'confirmed' ? 0 : 1;
    if (stateA !== stateB) return stateA - stateB;
    return (a.display_rank || 999) - (b.display_rank || 999);
  });
  const line = (selectedLineId && lines.find(l => l.line_id === selectedLineId)) || lines[0];

  // Build line selector options
  const lineOptions = lines.map((l, i) => {
    const label = `${l.side === 'support' ? '支撑' : '阻力'} #${i+1} (${l.state === 'confirmed' ? '3点' : '2点'}, ${l.confirming_touch_count||2}触)`;
    return `<option value="${l.line_id}" ${l.line_id === line.line_id ? 'selected' : ''}>${label}</option>`;
  }).join('');

  try {
    const params = new URLSearchParams({
      symbol,
      timeframe: interval,
      anchor1_idx: String(line.anchor_indices?.[0] || 0),
      anchor1_price: String(line.anchor_prices?.[0] || line.price_start),
      anchor2_idx: String(line.anchor_indices?.[1] || 0),
      anchor2_price: String(line.anchor_prices?.[1] || line.price_end),
      side: line.side,
      k: '30',
    });
    const resp = await fetchJson(`/api/tools/patterns/match?${params}`, { timeout: 8000 }).catch((e) => ({ok:false, _netErr: e.message}));
    if (!resp?.ok || !resp.data?.stats) {
      const err = resp?._netErr || resp?.error || '';
      // Check if any DB exists at all
      let dbHint = '此币对样本库未建立';
      try {
        const dbs = await fetchJson('/api/tools/patterns/database');
        const names = (dbs?.data || []).map(x => x.symbol_timeframe);
        const expected = `${symbol}_${interval}`;
        if (!names.includes(expected)) {
          dbHint = `${symbol} ${interval} 样本库未建 — 已有 ${names.length} 个数据库。可在监控页看建库进度或运行 patterns/batch-build`;
        }
      } catch {}
      return `<div class="dr-card"><h3>历史概率</h3><p class="muted">${dbHint}</p>${err ? `<p class="muted" style="font-size:10px">${err}</p>` : ''}</div>`;
    }
    const s = resp.data.stats;
    const anomaly = resp.data.anomaly || {};
    if (s.sample_size === 0) {
      return '<div class="dr-card"><h3>历史概率</h3><p class="muted">数据库存在但当前结构无相似样本</p></div>';
    }

    const evClass = s.expected_value > 0 ? 'pnl-pos' : 'pnl-neg';
    const lineTypeZh = line.side === 'support' ? '支撑' : '阻力';
    const stateLabel = line.state === 'confirmed' ? '已确认3点' : '2点假设';

    // Trustworthiness badge
    const trustMap = { high: ['高可信','pnl-pos'], medium: ['中可信',''], low: ['低可信','pnl-neg'], none: ['无数据','muted'] };
    const [trustLabel, trustCls] = trustMap[s.trustworthiness] || trustMap.none;

    // Overfit flag
    const overfitMap = {
      'stable': ['稳定',''], 'overfit_detected': ['过拟合','pnl-neg'],
      'high_variance': ['高方差','pnl-neg'], 'improving': ['改善中','pnl-pos'],
      'insufficient_samples': ['样本不足','muted'],
    };
    const [ofLabel, ofCls] = overfitMap[s.overfit_flag] || ['-',''];

    const sb = s.split_breakdown || {};
    const splitRows = ['train','val','test'].map(k => {
      const st = sb[k] || {n:0};
      if (st.n === 0) return '';
      const cls = (st.ev || 0) > 0 ? 'pnl-pos' : 'pnl-neg';
      return `<div class="dr-row" style="font-size:10px"><span class="dr-label muted">${k} (${st.n})</span><span class="dr-value ${cls}">EV ${st.ev > 0 ? '+' : ''}${st.ev}</span></div>`;
    }).join('');

    return `
      <div class="dr-card">
        <h3>历史概率 <small style="color:var(--v2-muted);font-weight:normal">${s.sample_size}样本</small></h3>
        ${lines.length > 1 ? `
          <div class="dr-row">
            <span class="dr-label">选中线</span>
            <select id="dr-line-selector" style="flex:1;background:var(--v2-bg);border:1px solid var(--v2-border);color:var(--v2-text);font-size:11px;padding:2px 4px;border-radius:3px">
              ${lineOptions}
            </select>
          </div>
        ` : ''}
        <div class="dr-row"><span class="dr-label">对象</span><span class="dr-value">${lineTypeZh} ${stateLabel}</span></div>
        <div class="dr-row"><span class="dr-label">可信度</span><span class="dr-value ${trustCls}"><strong>${trustLabel}</strong> (${(s.confidence*100).toFixed(0)}%)</span></div>
        <div class="dr-row"><span class="dr-label">稳定性</span><span class="dr-value ${ofCls}">${ofLabel}</span></div>
        ${anomaly.is_anomaly ? `<div class="dr-row"><span class="dr-label" style="color:#fbbf24">⚠ 异常结构</span><span class="dr-value muted">最近距离 ${anomaly.nearest_distance}</span></div>` : ''}
        <div class="dr-row"><span class="dr-label">反弹概率</span><span class="dr-value ${s.p_bounce >= 0.5 ? 'pnl-pos' : ''}">${(s.p_bounce*100).toFixed(0)}%</span></div>
        <div class="dr-row"><span class="dr-label">破位概率</span><span class="dr-value ${s.p_break >= 0.5 ? 'pnl-neg' : ''}">${(s.p_break*100).toFixed(0)}%</span></div>
        <div class="dr-row"><span class="dr-label">假突破</span><span class="dr-value">${(s.p_fake_break*100).toFixed(0)}%</span></div>
        <div class="dr-row"><span class="dr-label">平均收益/回撤</span><span class="dr-value">+${s.avg_return_atr}/-${s.avg_drawdown_atr} ATR</span></div>
        <div class="dr-row" style="border-top:1px solid var(--v2-border);padding-top:4px;margin-top:4px">
          <span class="dr-label"><strong>期望值</strong></span>
          <span class="dr-value ${evClass}"><strong>${s.expected_value > 0 ? '+' : ''}${s.expected_value} ATR</strong></span>
        </div>
        ${splitRows ? `<div style="margin-top:4px;padding-top:4px;border-top:1px dashed var(--v2-border)">${splitRows}</div>` : ''}
        <button class="btn-primary btn-sm dr-gen-strategy" style="margin-top:8px;width:100%"
                data-symbol="${symbol}"
                data-tf="${interval}"
                data-a1-idx="${line.anchor_indices?.[0] || 0}"
                data-a1-price="${line.anchor_prices?.[0] || line.price_start}"
                data-a2-idx="${line.anchor_indices?.[1] || 0}"
                data-a2-price="${line.anchor_prices?.[1] || line.price_end}"
                data-side="${line.side}">
          → 生成策略草案
        </button>
      </div>
    `;
  } catch (err) {
    return `<div class="dr-card"><h3>历史概率</h3><p class="muted">${esc(err?.message || String(err))}</p></div>`;
  }
}

async function render() {
  const container = $('#v2-decision-rail');
  if (!container) return;

  try {
    const { structure, risk, snapshot } = await fetchAll(marketState.currentSymbol, marketState.currentInterval);
    const lastPrice = structure?.last_price || marketState.lastCandles?.at?.(-1)?.close || 0;

    // Render immediately with core cards, then async fill pattern stats
    const core = `
      ${cardMarketState(structure)}
      ${cardRegime(snapshot)}
      ${cardSRZones(snapshot, lastPrice)}
      <div id="dr-pattern-slot"></div>
      ${cardRiskGate(risk)}
    `;
    setHtml(container, core);

    // Event delegation for the rail (set once per render, not per line).
    // Avoids the indefinite-growth problem of re-binding onchange/onclick
    // every refresh — and replaces the inline onclick=JSON.stringify XSS
    // vector for the "→ 生成策略草案" buttons.
    if (!container._delegationWired) {
      container.addEventListener('click', (e) => {
        const btn = e.target.closest('.dr-gen-strategy');
        if (btn && typeof window.__openFactoryWithPattern === 'function') {
          window.__openFactoryWithPattern({
            symbol: btn.dataset.symbol,
            timeframe: btn.dataset.tf,
            anchor1_idx: Number(btn.dataset.a1Idx) || 0,
            anchor1_price: Number(btn.dataset.a1Price) || 0,
            anchor2_idx: Number(btn.dataset.a2Idx) || 0,
            anchor2_price: Number(btn.dataset.a2Price) || 0,
            side: btn.dataset.side,
          });
        }
      });
      container.addEventListener('change', (e) => {
        if (e.target.id === 'dr-line-selector') {
          refreshPatternCard(e.target.value);
        }
      });
      container._delegationWired = true;
    }

    // Async pattern stats, with line selector support
    const refreshPatternCard = async (selectedLineId = null) => {
      const html = await cardPatternStats(snapshot, marketState.currentSymbol, marketState.currentInterval, selectedLineId);
      const existing = container.querySelector('.dr-card h3')?.textContent?.includes('历史概率')
        ? [...container.querySelectorAll('.dr-card')].find(c => c.querySelector('h3')?.textContent?.includes('历史概率'))
        : null;
      if (existing) {
        existing.outerHTML = html;
      } else {
        const slot = container.querySelector('#dr-pattern-slot');
        if (slot) slot.outerHTML = html;
      }
      // Selector change is now handled by event delegation above — no
      // per-render onchange wiring (which leaked listeners).
    };
    refreshPatternCard();
  } catch (err) {
    console.error('[decision-rail] render failed:', err);
    setHtml(container, `<div class="dr-card"><h3>市场分析</h3><p class="muted">加载失败: ${esc(err?.message || String(err))}</p></div>`);
  }
}

let lastRender = 0;
const MIN_RENDER_GAP = 12000; // 12s minimum between renders

async function safeRender() {
  const now = Date.now();
  if (now - lastRender < MIN_RENDER_GAP) return;
  lastRender = now;
  await render();
}

function paintLoading() {
  const container = $('#v2-decision-rail');
  if (!container) return;
  const loadingCard = (title) => `
    <div class="dr-card dr-loading">
      <h3>${title}</h3>
      <div class="dr-skel-bar"></div>
      <div class="dr-skel-bar short"></div>
      <div class="dr-skel-bar"></div>
    </div>
  `;
  container.innerHTML = [
    loadingCard('市场状态'),
    loadingCard('S/R 区域'),
    loadingCard('风控'),
  ].join('');
}

export function initDecisionRail() {
  paintLoading();  // instant skeleton
  pollTimer = setInterval(safeRender, 30000);
  subscribe('market.symbol.changed', safeRender);
  subscribe('market.interval.changed', safeRender);
  // Round 11 P0-1: kick off the FIRST render immediately. Without this,
  // the rail sits at "加载中..." for the full 30 s until the interval
  // fires for the first time. (Symbol/interval-change subscriptions also
  // fire on first chart load but only AFTER candles are ready, which
  // is also slow.)
  void safeRender();
}

export function stopDecisionRail() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

export async function refreshDecisionRail() {
  await safeRender();
}
