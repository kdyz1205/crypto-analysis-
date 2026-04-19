// frontend/js/workbench/drawings/trade_plan_modal.js
//
// "Create Trade Plan from Line" modal.
//
// Opened by:
//   - right-click menu "Create Trade Plan"
//   - panel "+" button
//
// Emits POST /api/drawings/manual/place-line-order when "submit to exchange"
// is enabled. The backend places a Bitget plan order and the watcher moves
// that exchange-side order as the line projection changes. If submission is
// disabled, it falls back to a local watcher-only conditional.
//
// Leverage math (cross-margin):
//   notional        = account_equity * leverage
//   account_risk%   = stop_distance_% * leverage
//   account_reward% = account_risk% * rr_target
//
// Fields are persisted to localStorage so the next line opens with the
// last values pre-filled (per user Q16).

import { getLiveAccount } from '../../services/live_execution.js';
import { createConditional, placeLineOrder } from '../../services/conditionals.js';
import { esc } from '../../util/dom.js';

const LS_KEY = 'v2.tradeplan.defaults.v1';

const DEFAULTS = {
  direction: 'short',           // support→long, resistance→short — auto-set on open
  order_kind: 'bounce',         // bounce | breakout
  buffer_pct: 0.05,             // entry offset; also full line-stop risk
  stop_pct: 0.05,               // line-break confirmation beyond the line
  rr_target: 2.0,
  leverage: 10,
  notional_usd: 100,            // only used if leverage=0
  exchange_mode: 'live',        // paper | live
  submit_to_exchange: true,
  // auto-reverse
  reverse_enabled: true,
  reverse_buffer_pct: 0.05,
  reverse_stop_pct: 0.05,
  reverse_rr_target: 2.0,
};

function loadDefaults() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { ...DEFAULTS };
    const saved = JSON.parse(raw);
    return { ...DEFAULTS, ...saved };
  } catch {
    return { ...DEFAULTS };
  }
}

function saveDefaults(values) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(values));
  } catch {}
}

let _modal = null;
let _equityCache = { ts: 0, equity: 0, mode: '' };
const EQUITY_CACHE_MS = 60_000;

async function fetchEquity(mode) {
  // Cache 10s — account endpoint hits Bitget.
  const now = Date.now();
  if (_equityCache.mode === mode && now - _equityCache.ts < EQUITY_CACHE_MS) {
    return _equityCache.equity;
  }
  try {
    const resp = await getLiveAccount(mode, 5000);
    const equity = Number(
      resp?.equity ?? resp?.account?.equity ?? resp?.usdtEquity ?? resp?.totalEquity ?? 0,
    );
    _equityCache = { ts: now, equity, mode };
    return equity;
  } catch {
    return 0;
  }
}

function getCachedEquity(mode) {
  const now = Date.now();
  if (_equityCache.mode === mode && now - _equityCache.ts < EQUITY_CACHE_MS) {
    return Number(_equityCache.equity) || 0;
  }
  return 0;
}

/**
 * Open the modal.
 * @param {object} line  Manual line object with side, symbol, timeframe, manual_line_id
 * @returns {Promise<object|null>}  The created conditional, or null on cancel
 */
export function openTradePlanModal(line) {
  return new Promise((resolve) => {
    closeExisting();
    const defaults = loadDefaults();

    // Auto-direction: support→long, resistance→short (user can flip)
    const autoDir = line?.side === 'support' ? 'long' : 'short';
    const values = { ...defaults, direction: autoDir };

    _modal = document.createElement('div');
    _modal.className = 'tp-modal-backdrop';
    _modal.innerHTML = renderShell(line, values);
    document.body.appendChild(_modal);
    injectStyles();

    // Wire events
    const root = _modal;
    const $ = (sel) => root.querySelector(sel);
    const $$ = (sel) => Array.from(root.querySelectorAll(sel));

    const readForm = () => ({
      direction: $('[name=direction]').value,
      order_kind: $('[name=order_kind]').value,
      buffer_pct: Number($('[name=buffer_pct]').value) || 0,
      stop_pct: Number($('[name=stop_pct]').value) || 0,
      rr_target: Number($('[name=rr_target]').value) || 0,
      leverage: Number($('[name=leverage]').value) || 0,
      notional_usd: Number($('[name=notional_usd]').value) || 0,
      exchange_mode: $('[name=exchange_mode]').value,
      submit_to_exchange: $('[name=submit_to_exchange]').checked,
      reverse_enabled: $('[name=reverse_enabled]').checked,
      reverse_buffer_pct: Number($('[name=reverse_buffer_pct]').value) || 0,
      reverse_stop_pct: Number($('[name=reverse_stop_pct]').value) || 0,
      reverse_rr_target: Number($('[name=reverse_rr_target]').value) || 0,
    });

    const refreshLivePreview = async () => {
      const v = readForm();
      const mode = v.exchange_mode;
      const equity = await fetchEquity(mode);
      const accountDisplay = equity > 0 ? `$${equity.toFixed(2)}` : '—';

      let notional, riskUsd, rewardUsd, riskPctAccount, rewardPctAccount;
      if (v.leverage > 0) {
        notional = equity * v.leverage;
        riskPctAccount = v.stop_pct * v.leverage;
        rewardPctAccount = riskPctAccount * v.rr_target;
        riskUsd = equity * (riskPctAccount / 100);
        rewardUsd = equity * (rewardPctAccount / 100);
      } else {
        notional = v.notional_usd;
        riskUsd = notional * (v.stop_pct / 100);
        rewardUsd = riskUsd * v.rr_target;
        riskPctAccount = equity > 0 ? (riskUsd / equity) * 100 : 0;
        rewardPctAccount = equity > 0 ? (rewardUsd / equity) * 100 : 0;
      }

      $('#tp-preview-equity').textContent = accountDisplay;
      $('#tp-preview-notional').textContent = `$${notional.toFixed(2)}`;
      $('#tp-preview-risk').textContent =
        `$${riskUsd.toFixed(2)}  (${riskPctAccount.toFixed(2)}% of account)`;
      $('#tp-preview-reward').textContent =
        `$${rewardUsd.toFixed(2)}  (${rewardPctAccount.toFixed(2)}% of account)`;
    };

    refreshLivePreview();

    root.addEventListener('input', refreshLivePreview);
    root.addEventListener('change', refreshLivePreview);

    $('#tp-cancel').addEventListener('click', () => {
      closeExisting();
      resolve(null);
    });
    _modal.addEventListener('click', (e) => {
      if (e.target === _modal) {
        closeExisting();
        resolve(null);
      }
    });

    $('#tp-confirm').addEventListener('click', async () => {
      const btn = $('#tp-confirm');
      btn.disabled = true;
      btn.textContent = '提交中...';
      try {
        const v = readForm();
        saveDefaults(v);

        // Compute size_usdt. If user set a leverage, we derive notional
        // from (equity × leverage); otherwise use the raw notional_usd.
        let size_usdt = v.notional_usd;
        if (v.leverage > 0) {
          const cachedEquity = getCachedEquity(v.exchange_mode);
          if (cachedEquity > 0) {
            size_usdt = cachedEquity * v.leverage;
          } else {
            try {
              const equity = await fetchEquity(v.exchange_mode);
              if (equity > 0) size_usdt = equity * v.leverage;
            } catch {}
          }
        }
        if (!size_usdt || size_usdt <= 0) {
          throw new Error('size_usdt 计算失败 — 填一个 notional 或确保账户有余额');
        }

        // Place a real Bitget plan order. The backend stores it as a
        // triggered conditional so the watcher can cancel+replace it as
        // the sloped line projection moves.
        const payload = {
          manual_line_id: line.manual_line_id,
          direction: v.direction,
          kind: v.order_kind === 'breakout' ? 'break' : 'bounce',
          tolerance_pct: v.buffer_pct,
          stop_offset_pct: v.stop_pct,
          size_usdt,
          leverage: v.leverage > 0 ? Math.round(v.leverage) : 1,
          mode: v.exchange_mode === 'paper' ? 'demo' : 'live',
          rr_target: v.rr_target,
          reverse_enabled: v.reverse_enabled,
          reverse_entry_offset_pct: v.reverse_buffer_pct,
          reverse_stop_offset_pct: v.reverse_stop_pct,
          reverse_rr_target: v.reverse_rr_target,
          reverse_leverage: v.leverage > 0 ? Math.round(v.leverage) : null,
        };

        const resp = v.submit_to_exchange
          ? await placeLineOrder(payload)
          : await createConditional({
              manual_line_id: line.manual_line_id,
              trigger: {
                tolerance_atr: 0.2,
                poll_seconds: 0,
                max_age_seconds: 0,
                max_distance_atr: 0,
                break_threshold_atr: 0.5,
              },
              order: {
                direction: v.direction,
                order_kind: v.order_kind === 'breakout' ? 'breakout' : 'bounce',
                tolerance_pct_of_line: v.buffer_pct,
                stop_offset_pct_of_line: v.stop_pct,
                rr_target: v.rr_target,
                leverage: null,
                notional_usd: size_usdt,
                submit_to_exchange: false,
                exchange_mode: 'paper',
                reverse_enabled: v.reverse_enabled,
                reverse_entry_offset_pct: v.reverse_buffer_pct,
                reverse_stop_offset_pct: v.reverse_stop_pct,
                reverse_rr_target: v.reverse_rr_target,
                reverse_leverage: null,
              },
              pattern_stats: {},
            });
        if (!resp?.ok) {
          const err = resp?.error || resp?.detail || resp?.reason || 'create failed';
          throw new Error(err);
        }
        closeExisting();
        resolve(resp);
      } catch (err) {
        btn.disabled = false;
        btn.textContent = '确认挂单';
        $('#tp-error').textContent = `创建失败: ${err?.message || err}`;
      }
    });

    // Esc closes
    const onKey = (e) => {
      if (e.key === 'Escape') {
        document.removeEventListener('keydown', onKey);
        closeExisting();
        resolve(null);
      }
    };
    document.addEventListener('keydown', onKey);
  });
}

function closeExisting() {
  if (_modal && _modal.parentNode) _modal.parentNode.removeChild(_modal);
  _modal = null;
}

function renderShell(line, v) {
  const sym = esc(line?.symbol || '');
  const tf = esc(line?.timeframe || '');
  const side = esc(line?.side || '');
  const id = esc(line?.manual_line_id || '');
  return `
    <div class="tp-modal">
      <div class="tp-head">
        <div class="tp-title">Trade Plan — <span class="tp-sym">${sym}</span> ${tf}</div>
        <div class="tp-sub">line: ${side} · ${id}</div>
      </div>

      <div class="tp-body">
        <div class="tp-row">
          <label class="tp-label">方向</label>
          <select name="direction">
            <option value="long" ${v.direction === 'long' ? 'selected' : ''}>做多 Long</option>
            <option value="short" ${v.direction === 'short' ? 'selected' : ''}>做空 Short</option>
          </select>
          <label class="tp-label">入场模式</label>
          <select name="order_kind">
            <option value="bounce" ${v.order_kind === 'bounce' ? 'selected' : ''}>触碰 Bounce</option>
            <option value="breakout" ${v.order_kind === 'breakout' ? 'selected' : ''}>突破 Breakout</option>
          </select>
        </div>

        <div class="tp-row">
          <label class="tp-label">入场 Buffer %</label>
          <input type="number" name="buffer_pct" value="${v.buffer_pct}" step="0.01" min="0"/>
          <label class="tp-label">止损</label>
          <input type="number" name="stop_pct" value="${v.stop_pct}" step="0.01" min="0"/>
          <label class="tp-label">RR</label>
          <input type="number" name="rr_target" value="${v.rr_target}" step="0.1" min="0"/>
        </div>

        <div class="tp-row">
          <label class="tp-label">杠杆</label>
          <input type="number" name="leverage" value="${v.leverage}" step="1" min="0"/>
          <label class="tp-label">名义 USDT (leverage=0 时)</label>
          <input type="number" name="notional_usd" value="${v.notional_usd}" step="10" min="0"/>
          <label class="tp-label">账户</label>
          <select name="exchange_mode">
            <option value="live" ${v.exchange_mode === 'live' ? 'selected' : ''}>Live</option>
            <option value="paper" ${v.exchange_mode === 'paper' ? 'selected' : ''}>Paper</option>
          </select>
        </div>

        <div class="tp-row tp-row-check">
          <label><input type="checkbox" name="submit_to_exchange" ${v.submit_to_exchange ? 'checked' : ''}/> 提交到交易所</label>
        </div>

        <div class="tp-preview">
          <div class="tp-preview-row"><span>账户余额</span><span id="tp-preview-equity">—</span></div>
          <div class="tp-preview-row"><span>仓位名义</span><span id="tp-preview-notional">—</span></div>
          <div class="tp-preview-row tp-risk"><span>止损风险</span><span id="tp-preview-risk">—</span></div>
          <div class="tp-preview-row tp-reward"><span>止盈目标</span><span id="tp-preview-reward">—</span></div>
        </div>

        <div class="tp-section-head">自动反手 (止损后同一条线反向挂单)</div>
        <div class="tp-row tp-row-check">
          <label><input type="checkbox" name="reverse_enabled" ${v.reverse_enabled ? 'checked' : ''}/> 启用自动反手</label>
        </div>
        <div class="tp-row">
          <label class="tp-label">反手 Buffer %</label>
          <input type="number" name="reverse_buffer_pct" value="${v.reverse_buffer_pct}" step="0.01" min="0"/>
          <label class="tp-label">反手止损</label>
          <input type="number" name="reverse_stop_pct" value="${v.reverse_stop_pct}" step="0.01" min="0"/>
          <label class="tp-label">反手 RR</label>
          <input type="number" name="reverse_rr_target" value="${v.reverse_rr_target}" step="0.1" min="0"/>
        </div>

        <div class="tp-error" id="tp-error"></div>
      </div>

      <div class="tp-foot">
        <button class="tp-btn" id="tp-cancel">取消</button>
        <button class="tp-btn tp-btn-primary" id="tp-confirm">确认挂单</button>
      </div>
    </div>
  `;
}

function injectStyles() {
  if (document.getElementById('tp-modal-styles')) return;
  const style = document.createElement('style');
  style.id = 'tp-modal-styles';
  style.textContent = `
    .tp-modal-backdrop {
      position: fixed; inset: 0; background: rgba(5,8,14,0.82);
      display: flex; align-items: center; justify-content: center;
      z-index: 10000; backdrop-filter: blur(4px);
    }
    .tp-modal {
      background: #0e141f; border: 1px solid #2a3548;
      border-radius: 8px; width: 640px; max-width: 95vw;
      max-height: 92vh; overflow: auto;
      color: #d8dde8; font-size: 13px;
      box-shadow: 0 12px 48px rgba(0,0,0,0.7);
    }
    .tp-head { padding: 16px 20px; border-bottom: 1px solid #1d2537; }
    .tp-title { font-size: 15px; font-weight: 700; color: #e8edf5; }
    .tp-title .tp-sym { color: #38bdf8; }
    .tp-sub { font-size: 11px; color: #6b7889; margin-top: 4px; }
    .tp-body { padding: 16px 20px; }
    .tp-row {
      display: grid; grid-template-columns: 1fr auto 1fr auto 1fr auto;
      gap: 8px 12px; align-items: center; margin-bottom: 12px;
    }
    .tp-row-check { display: block; }
    .tp-row-check label { cursor: pointer; }
    .tp-label { font-size: 11px; color: #8a95a6; font-weight: 600; }
    .tp-row input[type=number], .tp-row select {
      background: #141a26; border: 1px solid #2a3548; color: #d8dde8;
      padding: 6px 10px; border-radius: 4px; font-size: 13px;
      width: 100%; box-sizing: border-box;
    }
    .tp-row input[type=number]:focus, .tp-row select:focus {
      outline: none; border-color: #38bdf8;
    }
    .tp-static {
      background: #141a26; border: 1px solid #2a3548; color: #fbbf24;
      padding: 6px 10px; border-radius: 4px; font-size: 12px;
      min-width: 82px; box-sizing: border-box; text-align: center;
    }
    .tp-preview {
      background: #141a26; border: 1px solid #1d2537;
      border-radius: 6px; padding: 12px 16px; margin: 16px 0;
    }
    .tp-preview-row {
      display: flex; justify-content: space-between;
      padding: 4px 0; font-size: 12px;
    }
    .tp-preview-row.tp-risk span:last-child { color: #ff5252; font-weight: 700; }
    .tp-preview-row.tp-reward span:last-child { color: #00e676; font-weight: 700; }
    .tp-section-head {
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      color: #38bdf8; letter-spacing: 0.06em;
      margin: 20px 0 10px; padding-top: 12px;
      border-top: 1px solid #1d2537;
    }
    .tp-error { color: #ff5252; font-size: 12px; margin-top: 8px; min-height: 16px; }
    .tp-foot {
      padding: 14px 20px; border-top: 1px solid #1d2537;
      display: flex; justify-content: flex-end; gap: 10px;
    }
    .tp-btn {
      background: #1d2537; border: 1px solid #2a3548; color: #d8dde8;
      padding: 8px 20px; border-radius: 4px; cursor: pointer;
      font-size: 13px; font-weight: 600;
    }
    .tp-btn:hover { background: #24304a; }
    .tp-btn-primary { background: #0284c7; border-color: #0284c7; color: white; }
    .tp-btn-primary:hover { background: #0369a1; }
    .tp-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  `;
  document.head.appendChild(style);
}
