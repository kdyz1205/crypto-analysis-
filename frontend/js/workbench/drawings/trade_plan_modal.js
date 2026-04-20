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
const SETUPS_KEY = 'v2.tradeplan.setups.v1';

// ────────────────────────────────────────────────────────────────
// Setup presets (reusable parameter bundles)
// ────────────────────────────────────────────────────────────────
// Schema in localStorage:
//   { active_id: string, setups: [{ id, name, config: {...} }] }
// Each setup.config carries the same shape as DEFAULTS (minus `direction`
// because that's derived from the line side at open time).
// First use: seeds one "默认 Default" setup from DEFAULTS.

function loadSetups() {
  try {
    const raw = localStorage.getItem(SETUPS_KEY);
    if (!raw) return _seedSetups();
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed?.setups) || parsed.setups.length === 0) return _seedSetups();
    return parsed;
  } catch {
    return _seedSetups();
  }
}

function _seedSetups() {
  const seed = {
    active_id: 'default',
    setups: [
      { id: 'default', name: '默认 Default', config: { ...DEFAULTS } },
    ],
  };
  try { localStorage.setItem(SETUPS_KEY, JSON.stringify(seed)); } catch {}
  return seed;
}

function saveSetups(state) {
  try { localStorage.setItem(SETUPS_KEY, JSON.stringify(state)); } catch {}
}

function getActiveSetup(state) {
  return state.setups.find((s) => s.id === state.active_id) || state.setups[0];
}

function _genId() {
  return 'setup_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 6);
}

const DEFAULTS = {
  direction: 'short',           // support→long, resistance→short — auto-set on open
  order_kind: 'bounce',         // bounce | breakout
  // User guidance 2026-04-20: "I want 0.01 / 0.02 / 0.03 so it fully
  // crosses the line before firing". 0.03% = confirmed break, not noise.
  buffer_pct: 0.03,             // entry buffer beyond the line (%)
  stop_pct: 0.1,                // line-break stop (%)
  rr_target: 5.0,
  leverage: 10,                 // sent to Bitget via set-leverage API
  // Two sizing modes (user spec 2026-04-20 part 2):
  //   'notional_usd' : fixed USDT value, `notional_usd` is the size
  //   'equity_pct'   : percentage of account equity, notional =
  //                    equity * (equity_pct / 100) * leverage
  size_mode: 'notional_usd',
  notional_usd: 30,             // used when size_mode='notional_usd'
  equity_pct: 10,               // used when size_mode='equity_pct' (% of equity)
  exchange_mode: 'live',        // paper | live
  submit_to_exchange: true,
  // Auto-reverse: trigger IMMEDIATELY at the stop price (no entry buffer),
  // then a tight reverse stop in case price oscillates back.
  reverse_enabled: true,
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
let _equityCache = { ts: 0, equity: 0, mode: '', error: '' };
const EQUITY_CACHE_MS = 10_000;  // header uses same data, 10s is fine here

async function fetchEquity(mode) {
  const now = Date.now();
  if (_equityCache.mode === mode && now - _equityCache.ts < EQUITY_CACHE_MS && !_equityCache.error) {
    return _equityCache.equity;
  }
  try {
    const resp = await getLiveAccount(mode, 5000);
    // Backend returns snake_case: total_equity / usdt_available.
    // Legacy camelCase fallbacks kept for defensive compatibility, but
    // total_equity is the source of truth (matches views.js, runner_view.js,
    // execution/panel.js, conditional_panel.js).
    const equity = Number(
      resp?.total_equity
        ?? resp?.usdt_available
        ?? resp?.account?.total_equity
        ?? resp?.equity
        ?? resp?.account?.equity
        ?? 0,
    );
    _equityCache = { ts: now, equity, mode, error: '' };
    return equity;
  } catch (err) {
    const msg = err?.message || String(err);
    console.warn('[trade_plan_modal] fetchEquity failed:', msg);
    // P10: errors must be visible. Surface the failure via _equityCache.error
    // so refreshLivePreview can render "—" with an explanation.
    _equityCache = { ts: now, equity: 0, mode, error: msg };
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
/**
 * @param {object} line
 * @param {object} [options]
 * @param {string} [options.activeSetupId]  Open with this setup pre-selected.
 */
export function openTradePlanModal(line, options = {}) {
  return new Promise((resolve) => {
    closeExisting();
    // Load the ACTIVE setup's config (not the legacy LS_KEY defaults).
    // Setups system supersedes "last used" persistence: now the user
    // picks from named bundles instead of carrying whatever they last
    // typed into the next line.
    let setupsState = loadSetups();
    if (options.activeSetupId
        && setupsState.setups.some((s) => s.id === options.activeSetupId)) {
      setupsState.active_id = options.activeSetupId;
      saveSetups(setupsState);
    }
    const activeSetup = getActiveSetup(setupsState);
    const defaults = { ...DEFAULTS, ...(activeSetup.config || {}) };

    // Auto-direction: support→long, resistance→short (user can flip)
    const autoDir = line?.side === 'support' ? 'long' : 'short';
    const values = { ...defaults, direction: autoDir };

    _modal = document.createElement('div');
    _modal.className = 'tp-modal-backdrop';
    _modal.innerHTML = renderShell(line, values, setupsState);
    document.body.appendChild(_modal);
    injectStyles();

    // Wire events
    const root = _modal;
    const $ = (sel) => root.querySelector(sel);
    const $$ = (sel) => Array.from(root.querySelectorAll(sel));

    // ─── Setup preset management ──────────────────────────────
    const applySetupToForm = (config) => {
      if (!config) return;
      const numericFields = [
        'buffer_pct', 'stop_pct', 'rr_target', 'leverage',
        'notional_usd', 'equity_pct',
        'reverse_stop_pct', 'reverse_rr_target',
      ];
      numericFields.forEach((k) => {
        const el = $(`[name=${k}]`);
        if (el != null && config[k] != null) el.value = config[k];
      });
      if (config.order_kind != null) $('[name=order_kind]').value = config.order_kind;
      if (config.exchange_mode != null) $('[name=exchange_mode]').value = config.exchange_mode;
      if (config.size_mode != null) $('[name=size_mode]').value = config.size_mode;
      if (config.submit_to_exchange != null) $('[name=submit_to_exchange]').checked = !!config.submit_to_exchange;
      if (config.reverse_enabled != null) $('[name=reverse_enabled]').checked = !!config.reverse_enabled;
      // Don't override direction — that came from the line side.
      applySizeModeVisibility();
      refreshLivePreview();
    };

    const refreshSetupDropdown = () => {
      const sel = $('#tp-setup-select');
      if (!sel) return;
      sel.innerHTML = setupsState.setups
        .map((s) => `<option value="${esc(s.id)}" ${s.id === setupsState.active_id ? 'selected' : ''}>${esc(s.name)}</option>`)
        .join('') + `<option value="__new__">+ 新建 setup...</option>`;
    };

    $('#tp-setup-select')?.addEventListener('change', (ev) => {
      const v = ev.target.value;
      if (v === '__new__') {
        const name = prompt('新 setup 名字:', `Setup ${setupsState.setups.length + 1}`);
        if (!name) { ev.target.value = setupsState.active_id; return; }
        const id = _genId();
        setupsState.setups.push({ id, name, config: readForm() });
        setupsState.active_id = id;
        saveSetups(setupsState);
        refreshSetupDropdown();
      } else {
        setupsState.active_id = v;
        const s = getActiveSetup(setupsState);
        applySetupToForm(s.config);
        saveSetups(setupsState);
      }
    });

    $('#tp-setup-save')?.addEventListener('click', () => {
      const s = getActiveSetup(setupsState);
      s.config = readForm();
      saveSetups(setupsState);
      flashToast('已保存到当前 setup');
    });

    $('#tp-setup-rename')?.addEventListener('click', () => {
      const s = getActiveSetup(setupsState);
      const name = prompt('重命名 setup:', s.name);
      if (!name) return;
      s.name = name;
      saveSetups(setupsState);
      refreshSetupDropdown();
    });

    $('#tp-setup-delete')?.addEventListener('click', () => {
      if (setupsState.setups.length <= 1) { flashToast('至少保留 1 个 setup'); return; }
      const s = getActiveSetup(setupsState);
      if (!confirm(`删除 setup "${s.name}"?`)) return;
      setupsState.setups = setupsState.setups.filter((x) => x.id !== s.id);
      setupsState.active_id = setupsState.setups[0].id;
      saveSetups(setupsState);
      applySetupToForm(setupsState.setups[0].config);
      refreshSetupDropdown();
    });

    function flashToast(msg) {
      const el = $('#tp-setup-toast');
      if (!el) return;
      el.textContent = msg;
      el.style.opacity = '1';
      setTimeout(() => { el.style.opacity = '0'; }, 1400);
    }

    const readForm = () => ({
      direction: $('[name=direction]').value,
      order_kind: $('[name=order_kind]').value,
      buffer_pct: Number($('[name=buffer_pct]').value) || 0,
      stop_pct: Number($('[name=stop_pct]').value) || 0,
      rr_target: Number($('[name=rr_target]').value) || 0,
      leverage: Number($('[name=leverage]').value) || 0,
      size_mode: $('[name=size_mode]').value,
      notional_usd: Number($('[name=notional_usd]').value) || 0,
      equity_pct: Number($('[name=equity_pct]').value) || 0,
      exchange_mode: $('[name=exchange_mode]').value,
      submit_to_exchange: $('[name=submit_to_exchange]').checked,
      reverse_enabled: $('[name=reverse_enabled]').checked,
      // Auto-reverse has NO buffer — it triggers at the stop price. See
      // user spec 2026-04-20: "止损了你马上就做" = fire immediately, no
      // distance-based entry offset.
      reverse_buffer_pct: 0,
      reverse_stop_pct: Number($('[name=reverse_stop_pct]').value) || 0,
      reverse_rr_target: Number($('[name=reverse_rr_target]').value) || 0,
    });

    // Toggle visibility of the notional_usd vs equity_pct input row
    const applySizeModeVisibility = () => {
      const mode = $('[name=size_mode]')?.value || 'notional_usd';
      const usdRow = $('#tp-size-usd-group');
      const pctRow = $('#tp-size-pct-group');
      if (usdRow) usdRow.style.display = mode === 'notional_usd' ? '' : 'none';
      if (pctRow) pctRow.style.display = mode === 'equity_pct' ? '' : 'none';
    };

    const refreshLivePreview = async () => {
      const v = readForm();
      const mode = v.exchange_mode;
      const equity = await fetchEquity(mode);
      const accountErr = _equityCache.error;
      const accountDisplay = equity > 0
        ? `$${equity.toFixed(2)}`
        : (accountErr ? `— 账户数据加载失败: ${accountErr}` : '— (账户余额为 0)');

      // Size calculation respects size_mode:
      //   'notional_usd' — fixed USDT value (user's specified position size)
      //   'equity_pct'   — % of equity * leverage (scales with account)
      // Both interpretations yield a "notional" number in USDT;
      // margin on Bitget = notional / leverage.
      let notional = 0;
      if (v.size_mode === 'equity_pct') {
        notional = equity > 0 && v.leverage > 0 && v.equity_pct > 0
          ? equity * (v.equity_pct / 100) * v.leverage
          : 0;
      } else {
        // Default: fixed USDT notional
        notional = v.notional_usd > 0
          ? v.notional_usd
          : (equity > 0 && v.leverage > 0 ? equity * v.leverage : 0);
      }
      const riskUsd = notional * (v.stop_pct / 100);
      const rewardUsd = riskUsd * v.rr_target;
      const riskPctAccount = equity > 0 ? (riskUsd / equity) * 100 : 0;
      const rewardPctAccount = equity > 0 ? (rewardUsd / equity) * 100 : 0;
      const marginUsd = v.leverage > 0 ? notional / v.leverage : notional;

      const eqEl = $('#tp-preview-equity');
      eqEl.textContent = accountDisplay;
      eqEl.style.color = equity > 0 ? '' : '#ff9800';
      $('#tp-preview-notional').textContent = `$${notional.toFixed(2)} (保证金 $${marginUsd.toFixed(2)})`;
      $('#tp-preview-risk').textContent =
        `$${riskUsd.toFixed(2)}  (${riskPctAccount.toFixed(2)}% of account)`;
      $('#tp-preview-reward').textContent =
        `$${rewardUsd.toFixed(2)}  (${rewardPctAccount.toFixed(2)}% of account)`;
    };

    applySizeModeVisibility();
    refreshLivePreview();

    root.addEventListener('input', refreshLivePreview);
    root.addEventListener('change', (ev) => {
      if (ev.target?.name === 'size_mode') applySizeModeVisibility();
      refreshLivePreview();
    });

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

        // Resolve size_usdt from the active size_mode.
        let size_usdt = 0;
        if (v.size_mode === 'equity_pct') {
          const cachedEquity = getCachedEquity(v.exchange_mode);
          const equity = cachedEquity > 0
            ? cachedEquity
            : await fetchEquity(v.exchange_mode).catch(() => 0);
          if (!equity || equity <= 0) {
            throw new Error('账户余额无法读取,无法用百分比计算仓位。改用 USDT 模式或检查 API key。');
          }
          if (!v.equity_pct || v.equity_pct <= 0) {
            throw new Error('账户百分比必须 > 0');
          }
          size_usdt = equity * (v.equity_pct / 100) * (v.leverage || 1);
        } else {
          size_usdt = v.notional_usd;
          if (!size_usdt || size_usdt <= 0) {
            // Fallback: try full equity × leverage so very-small accounts
            // with notional=0 still get SOME order.
            const cachedEquity = getCachedEquity(v.exchange_mode);
            const equity = cachedEquity > 0
              ? cachedEquity
              : await fetchEquity(v.exchange_mode).catch(() => 0);
            if (equity > 0 && v.leverage > 0) size_usdt = equity * v.leverage;
          }
          if (!size_usdt || size_usdt <= 0) {
            throw new Error('notional_usd 为空且账户余额未读到 — 填一个数额或切换到百分比模式');
          }
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

function renderShell(line, v, setupsState) {
  const sym = esc(line?.symbol || '');
  const tf = esc(line?.timeframe || '');
  const side = esc(line?.side || '');
  const id = esc(line?.manual_line_id || '');
  const setupsOpts = (setupsState?.setups || [])
    .map((s) => `<option value="${esc(s.id)}" ${s.id === setupsState.active_id ? 'selected' : ''}>${esc(s.name)}</option>`)
    .join('') + `<option value="__new__">+ 新建 setup...</option>`;
  return `
    <div class="tp-modal">
      <div class="tp-head">
        <div class="tp-title">Trade Plan — <span class="tp-sym">${sym}</span> ${tf}</div>
        <div class="tp-sub">line: ${side} · ${id}</div>
        <div class="tp-setup-bar">
          <label class="tp-setup-label">Setup:</label>
          <select id="tp-setup-select" class="tp-setup-select">${setupsOpts}</select>
          <button type="button" id="tp-setup-save" class="tp-setup-btn" title="把当前表单值保存到选中的 setup">保存</button>
          <button type="button" id="tp-setup-rename" class="tp-setup-btn" title="重命名当前 setup">重命名</button>
          <button type="button" id="tp-setup-delete" class="tp-setup-btn tp-setup-btn-danger" title="删除当前 setup">删除</button>
          <span id="tp-setup-toast" class="tp-setup-toast"></span>
        </div>
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
          <label class="tp-label" title="触发价离线的距离(%)。typical: 0.01-0.1">入场 Buffer %</label>
          <input type="number" name="buffer_pct" value="${v.buffer_pct}" step="0.01" min="0"/>
          <label class="tp-label" title="止损离线的距离(%)">止损 %</label>
          <input type="number" name="stop_pct" value="${v.stop_pct}" step="0.01" min="0"/>
          <label class="tp-label">RR</label>
          <input type="number" name="rr_target" value="${v.rr_target}" step="0.1" min="0"/>
        </div>

        <div class="tp-row">
          <label class="tp-label" title="通过 set-leverage API 设到 Bitget 账户,无需手动开">Bitget 杠杆</label>
          <input type="number" name="leverage" value="${v.leverage}" step="1" min="1" max="125"/>
          <label class="tp-label">仓位大小</label>
          <select name="size_mode" style="min-width:110px">
            <option value="notional_usd" ${v.size_mode === 'notional_usd' ? 'selected' : ''}>USDT 固定值</option>
            <option value="equity_pct" ${v.size_mode === 'equity_pct' ? 'selected' : ''}>账户 %</option>
          </select>
          <label class="tp-label">账户</label>
          <select name="exchange_mode">
            <option value="live" ${v.exchange_mode === 'live' ? 'selected' : ''}>Live</option>
            <option value="paper" ${v.exchange_mode === 'paper' ? 'selected' : ''}>Paper</option>
          </select>
        </div>

        <div class="tp-row" id="tp-size-usd-group">
          <label class="tp-label" title="固定 USDT 名义(仓位大小)。保证金 = 名义 ÷ 杠杆">名义 USDT</label>
          <input type="number" name="notional_usd" value="${v.notional_usd}" step="5" min="0"/>
          <span class="tp-hint" style="color:#6b7889;font-size:11px;grid-column:span 4">
            例: 30 USDT × 10x 杠杆 = 仓位 300U / 保证金 30U
          </span>
        </div>

        <div class="tp-row" id="tp-size-pct-group" style="display:none">
          <label class="tp-label" title="equity × pct% × leverage 得到名义">账户 %</label>
          <input type="number" name="equity_pct" value="${v.equity_pct}" step="0.5" min="0" max="100"/>
          <span class="tp-hint" style="color:#6b7889;font-size:11px;grid-column:span 4">
            名义 = 权益 × 百分比 × 杠杆;按账户规模自动缩放
          </span>
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

        <div class="tp-section-head">自动反手 (止损触发后立即反向挂单)</div>
        <div class="tp-row tp-row-check">
          <label><input type="checkbox" name="reverse_enabled" ${v.reverse_enabled ? 'checked' : ''}/> 启用自动反手</label>
          <span class="tp-hint" style="margin-left:12px;color:#6b7889;font-size:11px">
            反手在止损价直接触发(无 buffer),反手 stop 一般 0.03-0.1%。
          </span>
        </div>
        <div class="tp-row">
          <label class="tp-label" title="反手止损距新入场的距离(%)。一般很小:0.03-0.1">反手止损 %</label>
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
    .tp-head { padding: 14px 20px 12px; border-bottom: 1px solid #1d2537; }
    .tp-setup-bar {
      display: flex; align-items: center; gap: 6px;
      margin-top: 10px; flex-wrap: wrap;
    }
    .tp-setup-label {
      font-size: 11px; color: #8a95a6; font-weight: 600;
    }
    .tp-setup-select {
      background: #141a26; border: 1px solid #2a3548; color: #e8edf5;
      padding: 5px 8px; border-radius: 4px; font-size: 12px;
      min-width: 160px; cursor: pointer;
    }
    .tp-setup-select:focus { outline: none; border-color: #38bdf8; }
    .tp-setup-btn {
      background: #1d2537; border: 1px solid #2a3548; color: #d8dde8;
      padding: 4px 10px; border-radius: 4px; font-size: 11px;
      cursor: pointer;
    }
    .tp-setup-btn:hover { background: #24304a; }
    .tp-setup-btn-danger { color: #ff9090; }
    .tp-setup-btn-danger:hover { background: #3a1e22; }
    .tp-setup-toast {
      margin-left: 6px; font-size: 11px; color: #00e676;
      opacity: 0; transition: opacity 0.2s;
    }
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


// ────────────────────────────────────────────────────────────────
// Setup picker popup — small floating menu shown before full modal
// ────────────────────────────────────────────────────────────────
// User flow (2026-04-20 spec): right-click line → "创建交易计划" →
// this lightweight picker appears at the click point with a vertical
// list of setups and "+ 新建 setup" at the bottom. Clicking a setup
// opens the full modal PRE-FILLED with that setup's values — 1-2
// clicks to place an order for the common case.

let _picker = null;
let _pickerCleanup = null;

export function openSetupPickerPopup(line, clickX = null, clickY = null) {
  return new Promise((resolve) => {
    // If a previous picker is still open, tear it down first.
    // IMPORTANT: don't reference inner handlers here (TDZ trap) — use
    // the previously-captured cleanup closure instead.
    if (_pickerCleanup) { try { _pickerCleanup(); } catch {} _pickerCleanup = null; }
    const setupsState = loadSetups();
    const x = Number.isFinite(clickX) ? clickX : window.innerWidth / 2;
    const y = Number.isFinite(clickY) ? clickY : window.innerHeight / 2;

    _picker = document.createElement('div');
    _picker.className = 'tp-picker';
    Object.assign(_picker.style, {
      position: 'fixed',
      left: `${x}px`,
      top: `${y}px`,
      background: '#0e141f',
      border: '1px solid #2a3548',
      borderRadius: '6px',
      padding: '4px 0',
      minWidth: '220px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      zIndex: '10000',
      fontSize: '12px',
      color: '#d8dde8',
    });

    const setupRows = setupsState.setups.map((s) => {
      const isActive = s.id === setupsState.active_id;
      const star = isActive ? '★ ' : '  ';
      return `<div class="tp-pick-row" data-id="${esc(s.id)}" style="padding:8px 14px;cursor:pointer;${isActive ? 'color:#38bdf8;font-weight:600' : ''}">
        ${star}${esc(s.name)}
      </div>`;
    }).join('');

    _picker.innerHTML = `
      <div style="padding:6px 14px;color:#8a95a6;font-size:11px;font-weight:600;border-bottom:1px solid #1d2537">选一个 setup</div>
      ${setupRows}
      <div style="height:1px;background:#1d2537;margin:4px 0"></div>
      <div class="tp-pick-row" data-action="new" style="padding:8px 14px;cursor:pointer;color:#00e676">+ 新建 setup / 配置...</div>
      <div class="tp-pick-row" data-action="manage" style="padding:6px 14px;cursor:pointer;color:#8a95a6;font-size:11px">⚙ 管理现有 setup (打开完整编辑器)</div>
    `;

    // Hover highlight
    _picker.querySelectorAll('.tp-pick-row').forEach((el) => {
      el.addEventListener('mouseenter', () => { el.style.background = '#1d2537'; });
      el.addEventListener('mouseleave', () => { el.style.background = 'transparent'; });
    });

    document.body.appendChild(_picker);

    // Viewport clamp so it doesn't get cut off at bottom/right
    try {
      const rect = _picker.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const pad = 8;
      let nx = x, ny = y;
      if (nx + rect.width + pad > vw) nx = vw - rect.width - pad;
      if (ny + rect.height + pad > vh) ny = vh - rect.height - pad;
      if (nx < pad) nx = pad;
      if (ny < pad) ny = pad;
      _picker.style.left = `${nx}px`;
      _picker.style.top = `${ny}px`;
    } catch {}

    // Define cleanup BEFORE handlers so the handlers can reference it
    // via closure without hitting a TDZ. `_pickerCleanup` is the
    // module-level handle used by a subsequent openSetupPickerPopup
    // call to tear this picker down cleanly.
    const onDocClick = (ev) => {
      if (_picker && !_picker.contains(ev.target)) {
        cleanup();
        resolve(null);
      }
    };
    const onKey = (ev) => {
      if (ev.key === 'Escape') {
        cleanup();
        resolve(null);
      }
    };
    const cleanup = () => {
      if (_picker && _picker.parentNode) _picker.parentNode.removeChild(_picker);
      _picker = null;
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
      _pickerCleanup = null;
    };
    _pickerCleanup = cleanup;

    _picker.addEventListener('click', async (ev) => {
      const row = ev.target.closest('.tp-pick-row');
      if (!row) return;
      const action = row.dataset.action;
      const id = row.dataset.id;
      cleanup();
      if (action === 'new' || action === 'manage') {
        const result = await openTradePlanModal(line);
        resolve(result);
      } else if (id) {
        const result = await openTradePlanModal(line, { activeSetupId: id });
        resolve(result);
      } else {
        resolve(null);
      }
    });

    // Defer attachment so the current click's mousedown (which opened
    // this picker) doesn't immediately close it.
    setTimeout(() => {
      document.addEventListener('mousedown', onDocClick);
    }, 0);
    document.addEventListener('keydown', onKey);
  });
}

