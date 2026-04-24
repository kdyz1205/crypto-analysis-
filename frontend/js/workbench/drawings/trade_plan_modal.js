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
import { getTradePlanSetups, saveTradePlanSetups } from '../../services/trade_plan_setups.js';
import { esc } from '../../util/dom.js';
import { drawingsState } from '../../state/drawings.js';

const LS_KEY = 'v2.tradeplan.defaults.v1';
const SETUPS_KEY = 'v2.tradeplan.setups.v1';
const SETUPS_BACKUP_KEY = 'v2.tradeplan.setups.backup.v1';

// ────────────────────────────────────────────────────────────────
// Setup presets (reusable parameter bundles)
// ────────────────────────────────────────────────────────────────
// Schema in localStorage:
//   { active_id: string, setups: [{ id, name, config: {...} }] }
// Each setup.config carries the same shape as DEFAULTS (minus `direction`
// because that's derived from the line side at open time).
// First use: seeds one "默认 Default" setup from DEFAULTS.

function _normalizeSetupsState(parsed) {
  if (!Array.isArray(parsed?.setups) || parsed.setups.length === 0) return null;
  const setups = parsed.setups
    .filter((s) => s && typeof s === 'object' && s.id && s.config && typeof s.config === 'object')
    .map((s) => ({
      id: String(s.id),
      name: String(s.name || s.id),
      config: { ...s.config },
      custom_name: !!s.custom_name,
    }));
  if (!setups.length) return null;
  const active = setups.some((s) => s.id === parsed?.active_id)
    ? String(parsed.active_id)
    : setups[0].id;
  return { active_id: active, setups };
}

function loadSetupsLocal() {
  try {
    const raw = localStorage.getItem(SETUPS_KEY);
    if (!raw) return null;
    return _normalizeSetupsState(JSON.parse(raw));
  } catch {
    return null;
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
  try { localStorage.setItem(SETUPS_BACKUP_KEY, JSON.stringify(seed)); } catch {}
  return seed;
}

function saveSetups(state) {
  const normalized = _normalizeSetupsState(state) || _seedSetups();
  try { localStorage.setItem(SETUPS_KEY, JSON.stringify(normalized)); } catch {}
  try { localStorage.setItem(SETUPS_BACKUP_KEY, JSON.stringify(normalized)); } catch {}
  void saveTradePlanSetups(normalized).catch((err) => {
    console.warn('[trade_plan_modal] server setup backup save failed', err);
  });
}

async function loadSetupsState() {
  const local = loadSetupsLocal();
  try {
    const resp = await getTradePlanSetups();
    const remote = _normalizeSetupsState(resp?.state);
    if (remote) {
      try { localStorage.setItem(SETUPS_KEY, JSON.stringify(remote)); } catch {}
      try { localStorage.setItem(SETUPS_BACKUP_KEY, JSON.stringify(remote)); } catch {}
      return remote;
    }
  } catch (err) {
    console.warn('[trade_plan_modal] server setup load failed', err);
  }
  if (local) {
    void saveTradePlanSetups(local).catch((err) => {
      console.warn('[trade_plan_modal] server setup backfill failed', err);
    });
    return local;
  }
  try {
    const rawBackup = localStorage.getItem(SETUPS_BACKUP_KEY);
    const backup = rawBackup ? _normalizeSetupsState(JSON.parse(rawBackup)) : null;
    if (backup) {
      try { localStorage.setItem(SETUPS_KEY, JSON.stringify(backup)); } catch {}
      void saveTradePlanSetups(backup).catch((err) => {
        console.warn('[trade_plan_modal] server backup restore save failed', err);
      });
      return backup;
    }
  } catch {}
  return _seedSetups();
}

function getActiveSetup(state) {
  return state.setups.find((s) => s.id === state.active_id) || state.setups[0];
}

function _genId() {
  return 'setup_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 6);
}

// Build a descriptive name from a config so the dropdown label
// always reflects the latest saved values. User report 2026-04-21:
// "为什么 setup 永远无法保存" — actual cause was the name wasn't
// regenerated after save, so even though config WAS persisted, the
// dropdown showed stale text and user assumed save had failed.
//
// Examples:
//   "1%损 0.1%buffer RR5 做多 $30U 10x"
//   "0.5%损 0.3%buffer RR3 做空 3%仓 5x"
//   "0.5%损 0.05%buffer RR5 做多 $50U 20x 反手"
function _summarizeConfig(cfg) {
  if (!cfg) return '';
  const parts = [];
  if (cfg.stop_pct != null) parts.push(`${cfg.stop_pct}%损`);
  if (cfg.buffer_pct != null) parts.push(`${cfg.buffer_pct}%buffer`);
  if (cfg.rr_target != null) parts.push(`RR${cfg.rr_target}`);
  // 2026-04-21 user spec: direction + reverse are per-trade decisions,
  // NOT part of the saved setup. Setup = strategy parameters only.
  if (cfg.size_mode === 'equity_pct' && cfg.equity_pct != null) {
    parts.push(`${cfg.equity_pct}%仓`);
  } else if (cfg.notional_usd != null) {
    parts.push(`$${cfg.notional_usd}U`);
  }
  if (cfg.leverage != null) parts.push(`${cfg.leverage}x`);
  return parts.join(' ');
}

// When persisting a setup, strip out the per-trade fields (direction,
// reverse_enabled) so they don't become sticky. User 2026-04-21:
// "config的多空它并不会保存, 是直接用这个[快捷popup]上面去多空".
function _stripPerTradeFields(config) {
  if (!config) return config;
  const { direction, reverse_enabled, ...rest } = config;
  return rest;
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
  // Three sizing modes:
  //   'notional_usd' : fixed USDT value, `notional_usd` is the stake;
  //                    actual notional = stake * leverage
  //   'equity_pct'   : percentage of account equity * leverage
  //                    (notional = equity * equity_pct/100 * leverage)
  //   'risk_pct'     : user sets % of equity willing to LOSE per trade,
  //                    system computes notional = equity * risk_pct /
  //                    (buffer + stop + taker_fee*2). 2026-04-22 user:
  //                    "愿意亏损一个固定 % 就够了,懒得算 lev/pct/usd"
  size_mode: 'notional_usd',
  notional_usd: 30,             // used when size_mode='notional_usd'
  equity_pct: 10,               // used when size_mode='equity_pct' (% of equity)
  risk_pct: 3,                  // used when size_mode='risk_pct' (% willing to lose)
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
// Two-level cache: fresh fetch succeeds → freshTs + equity. Fetch fails →
// we PRESERVE the last good equity so the popup/modal can fall back to it.
// User 2026-04-22: popup showed "$0 notional (pending...)" because a
// transient /account abort zeroed out the cache, blocking order placement
// even though the sidebar was still showing real equity from its own poll.
let _equityCache = {
  freshTs: 0,        // last time we GOT a successful fetch
  staleTs: 0,        // last time we TRIED (success or fail)
  equity: 0,         // last good value (kept across failures)
  mode: '',
  error: '',
};
const EQUITY_FRESH_MS = 10_000;   // within this: no re-fetch
const EQUITY_STALE_MS = 5 * 60_000;  // last good remains usable for 5 min

async function fetchEquity(mode) {
  const now = Date.now();
  if (_equityCache.mode === mode && now - _equityCache.freshTs < EQUITY_FRESH_MS && !_equityCache.error) {
    return _equityCache.equity;
  }
  try {
    // 15s timeout: /account fans out to 5 sequential Bitget REST calls.
    const resp = await getLiveAccount(mode, 15000);
    const equity = Number(
      resp?.total_equity
        ?? resp?.usdt_available
        ?? resp?.account?.total_equity
        ?? resp?.equity
        ?? resp?.account?.equity
        ?? 0,
    );
    _equityCache = {
      freshTs: now, staleTs: now, equity, mode, error: '',
    };
    return equity;
  } catch (err) {
    const msg = err?.message || String(err);
    console.warn('[trade_plan_modal] fetchEquity failed:', msg);
    // CRITICAL: do NOT zero the equity. Keep last good. The popup's
    // size_usdt calculation falls back to this via getCachedEquity().
    const prevEquity = (_equityCache.mode === mode) ? _equityCache.equity : 0;
    const prevFreshTs = (_equityCache.mode === mode) ? _equityCache.freshTs : 0;
    _equityCache = {
      freshTs: prevFreshTs,
      staleTs: now,
      equity: prevEquity,
      mode,
      error: msg,
    };
    return prevEquity;  // return last good instead of 0 so caller can proceed
  }
}

function getCachedEquity(mode) {
  const now = Date.now();
  if (_equityCache.mode === mode && _equityCache.equity > 0
      && now - _equityCache.freshTs < EQUITY_STALE_MS) {
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
  return new Promise(async (resolve) => {
    closeExisting();
    // Load the ACTIVE setup's config (not the legacy LS_KEY defaults).
    // Setups system supersedes "last used" persistence: now the user
    // picks from named bundles instead of carrying whatever they last
    // typed into the next line.
    let setupsState = await loadSetupsState();
    if (options.activeSetupId
        && setupsState.setups.some((s) => s.id === options.activeSetupId)) {
      setupsState.active_id = options.activeSetupId;
      saveSetups(setupsState);
    }
    const activeSetup = getActiveSetup(setupsState);
    const defaults = { ...DEFAULTS, ...(activeSetup.config || {}) };

    // Direction resolution (2026-04-21 rewrite): direction is a PER-TRADE
    // decision, NOT a setup field. Always derive from line side initially;
    // user can flip in the modal but that choice doesn't persist to setup.
    //   support → long   (bounce long)
    //   resistance → short (bounce short)
    // Quick-trade popup is the primary direction picker for setup-based
    // flows. Full modal still shows a Direction selector for one-off
    // orders, but its value is stripped before saving to setup.
    const initialDir = line?.side === 'support' ? 'long' : 'short';
    const values = { ...defaults, direction: initialDir, reverse_enabled: false };

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
        'notional_usd', 'equity_pct', 'risk_pct',
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
      // 2026-04-21: direction + reverse_enabled are per-trade, NOT part
      // of the setup. Switching setups in the modal does NOT reset these
      // — the user's choice in the Direction/Reverse fields is preserved.
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
        const formSnap = _stripPerTradeFields(readForm());
        const autoName = _summarizeConfig(formSnap);
        const name = prompt('新 setup 名字 (留空用自动命名):', autoName || `Setup ${setupsState.setups.length + 1}`);
        if (name === null) { ev.target.value = setupsState.active_id; return; }
        const id = _genId();
        const finalName = name.trim() || autoName || `Setup ${setupsState.setups.length + 1}`;
        setupsState.setups.push({
          id, name: finalName,
          custom_name: !!name.trim(),   // if user typed a name, lock it
          config: formSnap,
        });
        setupsState.active_id = id;
        saveSetups(setupsState);
        refreshSetupDropdown();
        flashToast(`✓ 新 setup "${finalName}"`);
      } else {
        setupsState.active_id = v;
        const s = getActiveSetup(setupsState);
        applySetupToForm(s.config);
        saveSetups(setupsState);
      }
    });

    $('#tp-setup-save')?.addEventListener('click', () => {
      const s = getActiveSetup(setupsState);
      if (!s) { flashToast('❌ 未找到活动 setup'); return; }
      // Strip direction + reverse_enabled — those are per-trade, not setup.
      const newConfig = _stripPerTradeFields(readForm());
      s.config = newConfig;
      // Auto-regenerate the name to reflect the saved values — unless
      // the user has explicitly locked it via rename (custom_name=true).
      // Without this, the dropdown kept showing the OLD name after save,
      // making it look like save never worked. User report 2026-04-21.
      if (!s.custom_name) {
        const autoName = _summarizeConfig(newConfig);
        if (autoName) s.name = autoName;
      }
      saveSetups(setupsState);
      refreshSetupDropdown();
      flashToast(`✓ 已保存 "${s.name}"`);
    });

    $('#tp-setup-rename')?.addEventListener('click', () => {
      const s = getActiveSetup(setupsState);
      if (!s) return;
      const name = prompt('重命名 setup (留空恢复自动命名):', s.name);
      if (name === null) return;   // cancelled
      if (name.trim() === '') {
        // Empty = re-enable auto-naming from config
        s.name = _summarizeConfig(s.config) || '默认 Default';
        s.custom_name = false;
      } else {
        s.name = name;
        s.custom_name = true;
      }
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
      risk_pct: Number($('[name=risk_pct]')?.value) || 0,
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

    // Toggle visibility of the 3 mutually-exclusive size input rows
    const applySizeModeVisibility = () => {
      const mode = $('[name=size_mode]')?.value || 'notional_usd';
      const usdRow = $('#tp-size-usd-group');
      const pctRow = $('#tp-size-pct-group');
      const riskRow = $('#tp-size-risk-group');
      if (usdRow) usdRow.style.display = mode === 'notional_usd' ? '' : 'none';
      if (pctRow) pctRow.style.display = mode === 'equity_pct' ? '' : 'none';
      if (riskRow) riskRow.style.display = mode === 'risk_pct' ? '' : 'none';
    };

    const refreshLivePreview = async () => {
      const v = readForm();
      const mode = v.exchange_mode;

      // ── Raw-price block: line_at_now + entry + SL + TP ──────────
      // User 2026-04-23 ZEC incident: server placed at a different
      // price than user read visually. These raw numbers go to Bitget
      // if "确认挂单" is clicked — user needs to see them BEFORE.
      // Uses the SAME log projection as chart_drawing.js + server
      // conditionals.py:_project_manual_line_price so all three agree.
      {
        const drawing = (drawingsState.lines || []).find(
          (l) => l && l.manual_line_id === line.manual_line_id
        );
        const lineNowEl = $('#tp-preview-line-now');
        const entryEl = $('#tp-preview-entry-price');
        const slEl = $('#tp-preview-sl-price');
        const tpEl = $('#tp-preview-tp-price');
        const fmt = (p) => {
          const n = Number(p);
          if (!Number.isFinite(n) || n <= 0) return '—';
          if (n >= 1000) return n.toFixed(2);
          if (n >= 100) return n.toFixed(3);
          if (n >= 10) return n.toFixed(3);
          if (n >= 1) return n.toFixed(4);
          if (n >= 0.1) return n.toFixed(5);
          if (n >= 0.01) return n.toFixed(6);
          return n.toFixed(7);
        };
        if (drawing && drawing.t_end > drawing.t_start
            && drawing.price_start > 0 && drawing.price_end > 0) {
          const nowS = Math.floor(Date.now() / 1000);
          const r = (nowS - drawing.t_start) / (drawing.t_end - drawing.t_start);
          const lineNow = Math.exp(
            Math.log(drawing.price_start)
            + r * (Math.log(drawing.price_end) - Math.log(drawing.price_start))
          );
          const bufPct = Math.abs(v.buffer_pct || 0) / 100;
          const stopPct = Math.abs(v.stop_pct || 0) / 100;
          const isLong = v.direction === 'long';
          // LONG: entry above line, SL below line.
          // SHORT: entry below line, SL above line.
          const entryP = isLong ? lineNow * (1 + bufPct) : lineNow * (1 - bufPct);
          const slP = isLong ? lineNow * (1 - stopPct) : lineNow * (1 + stopPct);
          const slDist = Math.abs(entryP - slP);
          const slDistPct = entryP > 0 ? (slDist / entryP) * 100 : 0;
          const tpP = isLong
            ? entryP + slDist * (v.rr_target || 0)
            : entryP - slDist * (v.rr_target || 0);
          if (lineNowEl) lineNowEl.textContent = fmt(lineNow);
          if (entryEl) {
            entryEl.textContent =
              `${fmt(entryP)}  (${v.buffer_pct || 0}% ${isLong ? '上' : '下'}, 距 line ${fmt(Math.abs(entryP - lineNow))})`;
          }
          if (slEl) {
            slEl.textContent =
              `${fmt(slP)}  (距 entry ${fmt(slDist)} = ${slDistPct.toFixed(3)}%)`;
          }
          if (tpEl) {
            tpEl.textContent = `${fmt(tpP)}  (RR ${(v.rr_target || 0).toFixed(1)})`;
          }
        } else {
          if (lineNowEl) lineNowEl.textContent = drawing ? '— (端点无效)' : '— (找不到线)';
          if (entryEl) entryEl.textContent = '—';
          if (slEl) slEl.textContent = '—';
          if (tpEl) tpEl.textContent = '—';
        }
      }

      const equity = await fetchEquity(mode);
      const accountErr = _equityCache.error;
      // Friendlier message — user doesn't care about "signal is aborted",
      // they care whether they can still place the order (YES — notional
      // is independent of the equity fetch).
      const isAbort = accountErr && /abort|timeout|signal/i.test(accountErr);
      const accountDisplay = equity > 0
        ? `$${equity.toFixed(2)}`
        : (isAbort
            ? '— Bitget API 慢,稍后再读(不影响挂单)'
            : accountErr
              ? `— 账户数据加载失败: ${accountErr}`
              : '— (账户余额为 0)');

      // Size calculation respects size_mode:
      //   'notional_usd' — the USDT number the user stakes = MARGIN,
      //                    position = stake × leverage. Matches the UI
      //                    hint "30 USDT × 10x = 仓位 300U / 保证金 30U".
      //   'equity_pct'   — % of equity × leverage (scales with account)
      //
      // User 2026-04-21: placed $30 USDT 10x and got 2849 BAS qty ($30
      // notional) instead of expected $300 notional. Root cause: code
      // was treating the USDT input as raw notional instead of margin,
      // contradicting the UI hint. Fixed 2026-04-21.
      // Compute buffer+stop first because risk_pct mode needs it.
      const buffer = Math.abs(v.buffer_pct || 0);
      const stop = Math.abs(v.stop_pct || 0);
      const totalStopPct = buffer + stop;

      // Read taker fee early too (needed by risk_pct formula).
      let _previewTakerFeePct = 0.06;
      try {
        const _ov = parseFloat(localStorage.getItem('v2.trade.takerFeePct') || '');
        if (Number.isFinite(_ov) && _ov > 0) _previewTakerFeePct = _ov;
      } catch {}
      const totalCostPct = totalStopPct + (_previewTakerFeePct * 2);

      let notional = 0;
      if (v.size_mode === 'equity_pct') {
        notional = equity > 0 && v.leverage > 0 && v.equity_pct > 0
          ? equity * (v.equity_pct / 100) * v.leverage
          : 0;
      } else if (v.size_mode === 'risk_pct') {
        // Risk-based sizing: willingLoss$ = notional * totalCostPct/100
        // => notional = equity * risk_pct / totalCostPct
        // (leverage ignored here: user said "Bitget 那边账户级设置,UI 杠杆没用")
        notional = equity > 0 && v.risk_pct > 0 && totalCostPct > 0
          ? equity * (v.risk_pct / 100) / (totalCostPct / 100)
          : 0;
      } else {
        // User enters STAKE (margin), actual position = stake × leverage
        const lev = v.leverage > 0 ? v.leverage : 1;
        notional = v.notional_usd > 0
          ? v.notional_usd * lev
          : (equity > 0 && v.leverage > 0 ? equity * v.leverage : 0);
      }

      // 2026-04-21 user: "Bitget 收 taker fee,开仓 + 平仓两次。你这
      // 边没算,所以数字和 Bitget 对不上。加上去让显示一致。"
      // Bitget USDT-M Futures taker fee = 0.06% (VIP 0 default).
      // Users with VIP/discount can override via localStorage key
      // v2.trade.takerFeePct (e.g. set to 0.048 for VIP 1).
      let takerFeePct = 0.06;
      try {
        const override = parseFloat(localStorage.getItem('v2.trade.takerFeePct') || '');
        if (Number.isFinite(override) && override > 0) takerFeePct = override;
      } catch {}
      // We pay taker fee on ENTRY (market trigger fills as taker) and
      // again on EXIT (stop or TP — both market orders). Both are on
      // notional at the fill price. Approximation: use the same
      // notional for both legs (exit notional differs by stop/tp move
      // but the % difference is <1% — not worth complicating).
      const feePctTotal = takerFeePct * 2;                 // open + close
      const feeUsd = notional * (feePctTotal / 100);

      const riskPriceUsd = notional * (totalStopPct / 100);
      const riskUsdNet = riskPriceUsd + feeUsd;            // lose price move + both fees
      const rewardPriceUsd = riskPriceUsd * v.rr_target;
      const rewardUsdNet = rewardPriceUsd - feeUsd;        // gain price move - both fees

      const riskPctAccount = equity > 0 ? (riskUsdNet / equity) * 100 : 0;
      const rewardPctAccount = equity > 0 ? (rewardUsdNet / equity) * 100 : 0;
      // Effective RR after fees (the true risk:reward from the user's
      // perspective, not the theoretical price-distance RR).
      const effectiveRr = riskUsdNet > 0 ? rewardUsdNet / riskUsdNet : 0;
      const marginUsd = v.leverage > 0 ? notional / v.leverage : notional;

      const eqEl = $('#tp-preview-equity');
      eqEl.textContent = accountDisplay;
      eqEl.style.color = equity > 0 ? '' : '#ff9800';
      $('#tp-preview-notional').textContent = `$${notional.toFixed(2)} (保证金 $${marginUsd.toFixed(2)})`;

      // Show the stop-distance breakdown so the user can see at a glance
      // WHY the risk is what it is.
      const stopEl = $('#tp-preview-stop-dist');
      if (stopEl) {
        stopEl.style.color = '';
        stopEl.textContent =
          `${totalStopPct.toFixed(2)}% `
          + `(${buffer.toFixed(2)}% buffer + ${stop.toFixed(2)}% 过线)`;
      }
      // Fee row — show the pair total so user can sanity-check against Bitget
      const feeEl = $('#tp-preview-fee');
      if (feeEl) {
        feeEl.textContent =
          `$${feeUsd.toFixed(2)}  (${takerFeePct.toFixed(3)}% × 2, 开+平)`;
      }
      // Risk/reward: show NET (after fees) with price-only as secondary
      $('#tp-preview-risk').textContent =
        `$${riskUsdNet.toFixed(2)}  (${riskPctAccount.toFixed(2)}% of account)`
        + `  [价差 $${riskPriceUsd.toFixed(2)} + 手续费 $${feeUsd.toFixed(2)}]`;
      $('#tp-preview-reward').textContent =
        `$${rewardUsdNet.toFixed(2)}  (${rewardPctAccount.toFixed(2)}% of account)`
        + `  [价差 $${rewardPriceUsd.toFixed(2)} - 手续费 $${feeUsd.toFixed(2)}]`;
      // Effective RR row (real RR after fees eat into both sides)
      const rrEl = $('#tp-preview-effective-rr');
      if (rrEl) {
        const rrColor = effectiveRr >= 2 ? '#00e676' : (effectiveRr >= 1.3 ? '#fbbf24' : '#ff5252');
        rrEl.style.color = rrColor;
        rrEl.textContent =
          `${effectiveRr.toFixed(2)} : 1  (目标 ${(v.rr_target || 0).toFixed(1)}:1 毛)`;
      }
    };

    applySizeModeVisibility();
    refreshLivePreview();

    // Debounce 120ms so typing "0.61" doesn't fire 4 refresh passes with
    // intermediate render flicker. User report 2026-04-21: "preview 闪来
    // 闪去". Preview math is pure client-side except for equity (cached
    // 10s), so the debounce hides transient flashes without hurting UX.
    let _refreshTimer = null;
    const debouncedRefresh = () => {
      if (_refreshTimer) clearTimeout(_refreshTimer);
      _refreshTimer = setTimeout(() => {
        _refreshTimer = null;
        refreshLivePreview();
      }, 120);
    };

    root.addEventListener('input', debouncedRefresh);
    root.addEventListener('change', (ev) => {
      if (ev.target?.name === 'size_mode') applySizeModeVisibility();
      refreshLivePreview();
    });

    // 2026-04-21 user spec: 取消 = "don't place order" but AUTO-SAVE
    // the current form to the active setup first, so reopening the
    // modal restores exactly what the user was configuring. No lost
    // work. User: "点取消, 它只是说取消这个挂单, 那保存的细节已经保存好了".
    const _autoSaveAndClose = () => {
      try {
        const s = getActiveSetup(setupsState);
        if (s) {
          // Strip direction + reverse_enabled — per-trade, not setup.
          const newConfig = _stripPerTradeFields(readForm());
          s.config = newConfig;
          if (!s.custom_name) {
            const autoName = _summarizeConfig(newConfig);
            if (autoName) s.name = autoName;
          }
          saveSetups(setupsState);
        }
      } catch (err) {
        console.warn('[trade_plan_modal] auto-save on cancel failed', err);
      }
      closeExisting();
      resolve(null);
    };
    $('#tp-cancel').addEventListener('click', _autoSaveAndClose);

    // Also hook the × (close) button if it exists.
    const closeBtn = $('#tp-close');
    if (closeBtn) closeBtn.addEventListener('click', _autoSaveAndClose);

    // 2026-04-21 user request: modal should NOT close when clicking
    // outside the box. Only × or 取消 closes it. This prevents the
    // classic "I was configuring and accidentally lost my progress"
    // when switching browser tabs / clicking empty space.
    // Backdrop click handler intentionally REMOVED.
    // ESC still closes (keyboard shortcut, intentional). ESC also
    // auto-saves so the form state isn't lost.
    const _escHandler = (e) => {
      if (e.key === 'Escape') {
        _autoSaveAndClose();
      }
    };
    document.addEventListener('keydown', _escHandler);
    // Clean up ESC listener when modal closes (store on _modal so
    // closeExisting can pick it up).
    _modal._escHandler = _escHandler;

    $('#tp-confirm').addEventListener('click', async () => {
      const btn = $('#tp-confirm');
      btn.disabled = true;
      btn.textContent = '提交中...';
      try {
        const v = readForm();
        saveDefaults(v);

        // Resolve size_usdt from the active size_mode. NOTE:
        // notional_usd input = MARGIN/stake, backend expects actual
        // NOTIONAL (= stake × leverage). Matches the UI preview +
        // hint "30 USDT × 10x = 仓位 300U / 保证金 30U".
        let size_usdt = 0;
        const lev = v.leverage > 0 ? v.leverage : 1;
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
          size_usdt = equity * (v.equity_pct / 100) * lev;
        } else if (v.size_mode === 'risk_pct') {
          // Risk-based: size_usdt = equity * risk_pct / (buffer+stop+fee*2)
          const cachedEquity = getCachedEquity(v.exchange_mode);
          const equity = cachedEquity > 0
            ? cachedEquity
            : await fetchEquity(v.exchange_mode).catch(() => 0);
          if (!equity || equity <= 0) {
            throw new Error('账户余额无法读取 — 风险%模式必须拿到 equity 才能反推仓位。稍等片刻重试。');
          }
          if (!v.risk_pct || v.risk_pct <= 0) {
            throw new Error('愿意亏损% 必须 > 0');
          }
          const _buffer = Math.abs(v.buffer_pct || 0);
          const _stop = Math.abs(v.stop_pct || 0);
          let _tfee = 0.06;
          try {
            const _ov = parseFloat(localStorage.getItem('v2.trade.takerFeePct') || '');
            if (Number.isFinite(_ov) && _ov > 0) _tfee = _ov;
          } catch {}
          const _totalCostPct = _buffer + _stop + (_tfee * 2);
          if (_totalCostPct <= 0) {
            throw new Error('buffer + stop + 费用 必须 > 0,否则无法反推仓位');
          }
          size_usdt = equity * (v.risk_pct / 100) / (_totalCostPct / 100);
        } else {
          // v.notional_usd is the STAKE; multiply by leverage for actual position
          size_usdt = (Number(v.notional_usd) || 0) * lev;
          if (!size_usdt || size_usdt <= 0) {
            // Fallback: try full equity × leverage so very-small accounts
            // with notional=0 still get SOME order.
            const cachedEquity = getCachedEquity(v.exchange_mode);
            const equity = cachedEquity > 0
              ? cachedEquity
              : await fetchEquity(v.exchange_mode).catch(() => 0);
            if (equity > 0 && v.leverage > 0) size_usdt = equity * lev;
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
          // Forward equity_pct so backend can dynamically resize qty
          // as account equity changes. User 2026-04-22 spec.
          equity_pct: v.size_mode === 'equity_pct' && v.equity_pct > 0 ? v.equity_pct : null,
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
        // Clear feedback on success so the user doesn't wonder "did it
        // work?" and re-submit. Previously modal closed silently/instantly
        // → user on slow network saw nothing → hit confirm again →
        // double-order. (2026-04-21: HYPE position fill had 2 duplicate
        // SL + 2 duplicate TP plans from exactly this pattern.)
        btn.textContent = '✓ 已挂单';
        btn.style.background = '#00e676';
        btn.style.color = '#000';
        const oid = resp.exchange_order_id || resp.conditional_id || '';
        if (oid) {
          $('#tp-error').style.color = '#00e676';
          $('#tp-error').textContent = `✓ 成功: ${String(oid).slice(-10)}`;
        }
        setTimeout(() => {
          closeExisting();
          resolve(resp);
        }, 900);
      } catch (err) {
        $('#tp-error').style.color = '';
        const msg = String(err?.message || err || '');
        if (/abort|timeout|signal/i.test(msg)) {
          $('#tp-error').style.color = '#fbbf24';
          $('#tp-error').textContent = '网络超时: 订单可能已经到 Bitget。先去 Bitget app 或右侧挂单栏确认,不要直接重复点击。';
          btn.textContent = '状态未知';
        } else {
          btn.disabled = false;
          btn.textContent = '确认挂单';
          $('#tp-error').textContent = `创建失败: ${msg}`;
        }
      }
    });

    // ESC is already wired via _escHandler earlier in this function
    // (stored on _modal._escHandler for cleanup). Redundant second
    // listener removed 2026-04-21 per code-reviewer S6.
  });
}

function closeExisting() {
  if (_modal) {
    if (_modal._escHandler) {
      try { document.removeEventListener('keydown', _modal._escHandler); } catch {}
    }
    if (_modal.parentNode) _modal.parentNode.removeChild(_modal);
  }
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
          <select name="size_mode" style="min-width:130px">
            <option value="notional_usd" ${v.size_mode === 'notional_usd' ? 'selected' : ''}>USDT 固定值</option>
            <option value="equity_pct" ${v.size_mode === 'equity_pct' ? 'selected' : ''}>账户 %</option>
            <option value="risk_pct" ${v.size_mode === 'risk_pct' ? 'selected' : ''}>风险 %(自动)</option>
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

        <div class="tp-row" id="tp-size-risk-group" style="display:none">
          <label class="tp-label" title="愿意这一单最多损失占账户的百分比。名义自动反推,不用手动算杠杆">愿意亏损 %</label>
          <input type="number" name="risk_pct" value="${v.risk_pct}" step="0.5" min="0.1" max="50"/>
          <span class="tp-hint" style="color:#6b7889;font-size:11px;grid-column:span 4">
            名义 = 权益 × 亏损% ÷ (buffer + stop + 手续费×2)% · 杠杆字段不参与计算
          </span>
        </div>

        <div class="tp-row tp-row-check">
          <label><input type="checkbox" name="submit_to_exchange" ${v.submit_to_exchange ? 'checked' : ''}/> 提交到交易所</label>
        </div>

        <div class="tp-preview tp-preview-prices">
          <div class="tp-preview-header">📊 挂单前核对 · 服务端计算,即将写入 Bitget</div>
          <div class="tp-preview-row tp-price-line"><span>line 现在值 (log @ now)</span><span id="tp-preview-line-now">—</span></div>
          <div class="tp-preview-row tp-price-entry"><span>入场价</span><span id="tp-preview-entry-price">—</span></div>
          <div class="tp-preview-row tp-price-sl"><span>止损价</span><span id="tp-preview-sl-price">—</span></div>
          <div class="tp-preview-row tp-price-tp"><span>止盈价</span><span id="tp-preview-tp-price">—</span></div>
          <div class="tp-preview-row tp-price-note"><span></span><span id="tp-preview-price-note" style="color:#8ac4ff;font-size:10px">视觉读数和 line 值不符请点 "取消" 调整参数或重画线</span></div>
        </div>

        <div class="tp-preview">
          <div class="tp-preview-row"><span>账户余额</span><span id="tp-preview-equity">—</span></div>
          <div class="tp-preview-row"><span>仓位名义</span><span id="tp-preview-notional">—</span></div>
          <div class="tp-preview-row tp-breakdown"><span>止损距离</span><span id="tp-preview-stop-dist">—</span></div>
          <div class="tp-preview-row tp-fee" title="Bitget taker fee 0.06% × 2 (开仓 + 平仓)"><span>手续费 (开+平)</span><span id="tp-preview-fee">—</span></div>
          <div class="tp-preview-row tp-risk"><span>止损风险 (含费)</span><span id="tp-preview-risk">—</span></div>
          <div class="tp-preview-row tp-reward"><span>止盈目标 (扣费)</span><span id="tp-preview-reward">—</span></div>
          <div class="tp-preview-row tp-effective-rr" title="实际盈亏比 = (目标价差 - 手续费) / (止损价差 + 手续费)"><span>实际 RR</span><span id="tp-preview-effective-rr">—</span></div>
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
    /* Raw-price preview section — distinct style so user's eye lands on it */
    .tp-preview.tp-preview-prices {
      background: #0d1624; border: 1px solid #2a3f5f;
      box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.15) inset;
    }
    .tp-preview-header {
      font-size: 11px; font-weight: 700; color: #60a5fa;
      letter-spacing: 0.03em; margin-bottom: 6px;
      padding-bottom: 6px; border-bottom: 1px solid #1d2537;
    }
    .tp-preview-row.tp-price-line span:last-child { color: #fbbf24; font-weight: 700; }
    .tp-preview-row.tp-price-entry span:last-child { color: #60a5fa; font-weight: 700; }
    .tp-preview-row.tp-price-sl span:last-child { color: #ef4444; font-weight: 700; }
    .tp-preview-row.tp-price-tp span:last-child { color: #22c55e; font-weight: 700; }
    .tp-preview-row.tp-price-note span:last-child { font-weight: 400 !important; }
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
  return new Promise(async (resolve) => {
    // If a previous picker is still open, tear it down first.
    // IMPORTANT: don't reference inner handlers here (TDZ trap) — use
    // the previously-captured cleanup closure instead.
    if (_pickerCleanup) { try { _pickerCleanup(); } catch {} _pickerCleanup = null; }
    const setupsState = await loadSetupsState();
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


// ─────────────────────────────────────────────────────────────
// Quick-trade popup (⚡ 交易)
// ─────────────────────────────────────────────────────────────
// User 2026-04-21 spec: after drawing a line, clicking "交易" in the
// context menu opens this small popup RIGHT NEXT TO the menu. It has:
//   - Direction toggle: 做多 / 做空 (pill)
//   - Reverse toggle:   自动反手 on/off
//   - Setup dropdown:   vertical list of saved setups
//   Click a setup row = PLACE ORDER IMMEDIATELY with that setup's
//   config + the user's chosen direction + reverse flag.
//
// Goal: line → Bitget order in ≤3 clicks (right-click line, hit 交易,
// click a setup). No full modal. "完整配置" escape link to open the
// normal editor if a setup isn't pre-saved.

let _quickPopup = null;
let _quickPopupCleanup = null;

export function openQuickTradePopup(line, clickX = null, clickY = null) {
  return new Promise(async (resolve) => {
    if (_quickPopupCleanup) { try { _quickPopupCleanup(); } catch {} _quickPopupCleanup = null; }

    const setupsState = await loadSetupsState();
    const x = Number.isFinite(clickX) ? clickX : window.innerWidth / 2;
    const y = Number.isFinite(clickY) ? clickY : window.innerHeight / 2;

    // Direction + reverse are per-trade only (not in setup). Require
    // an explicit direction pick before any real-money quick order.
    let direction = null;
    let reverseEnabled = false;

    // Warm equity cache in background so each setup row's $ risk /
    // reward shows real numbers for equity_pct setups. Non-blocking —
    // the first render shows "—" for amounts and a re-render fires
    // once the fetch resolves.
    const _equityModes = new Set();
    for (const s of setupsState.setups) {
      const m = s?.config?.exchange_mode === 'paper' ? 'demo' : (s?.config?.exchange_mode || 'live');
      _equityModes.add(m);
    }
    Promise.all([..._equityModes].map((m) => fetchEquity(m).catch(() => 0))).then(() => {
      if (_quickPopup) renderBody();
    });

    _quickPopup = document.createElement('div');
    _quickPopup.className = 'tp-quick-popup';
    Object.assign(_quickPopup.style, {
      position: 'fixed',
      left: `${x}px`,
      top: `${y}px`,
      background: '#0e141f',
      border: '1px solid #2a3548',
      borderRadius: '8px',
      padding: '10px',
      minWidth: '280px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.7)',
      // Bug 2026-04-22 (user: "还是会有这种重叠的可视化bug"): lightweight-charts
      // price-axis labels (40.983 / EMA21 / MA5 / MA55) were bleeding through
      // the popup because they render inside the chart's canvas with an
      // implicit stacking priority. Max int z-index + `isolation:isolate`
      // forces this popup into its own topmost stacking context so the
      // chart's canvas labels render below it.
      zIndex: '2147483647',
      isolation: 'isolate',
      fontSize: '12px',
      color: '#d8dde8',
    });

    function renderBody() {
      const longCls = direction === 'long' ? 'background:#0b4d1f;color:#00e676;font-weight:600' : 'background:#1a2133;color:#8a95a6';
      const shortCls = direction === 'short' ? 'background:#4d0b0b;color:#ff5252;font-weight:600' : 'background:#1a2133;color:#8a95a6';
      const revCls = reverseEnabled ? 'background:#3b2a0b;color:#fbbf24;font-weight:600' : 'background:#1a2133;color:#8a95a6';

      // Per-setup fee-aware P&L preview. User 2026-04-21: "加上 taker
      // fee 两次,让这边和 Bitget 的数字一致". Taker fee default 0.06%
      // (overrideable via localStorage v2.trade.takerFeePct).
      let takerFeePct = 0.06;
      try {
        const ov = parseFloat(localStorage.getItem('v2.trade.takerFeePct') || '');
        if (Number.isFinite(ov) && ov > 0) takerFeePct = ov;
      } catch {}

      const setupRows = setupsState.setups.map((s) => {
        const c = s.config || {};
        const buffer = Math.abs(Number(c.buffer_pct) || 0);
        const stop = Math.abs(Number(c.stop_pct) || 0);
        const totalPct = buffer + stop;
        const rr = Number(c.rr_target) || 2;
        const lev = Number(c.leverage) > 0 ? Number(c.leverage) : 1;
        // Bug fix 2026-04-22 (user: "挂了 $300 但 plan 说 $1396"): previously
        // the popup's preview AND the real place path used
        //     if (size_mode === 'equity_pct' && equity_pct > 0) use equity_pct
        //     else use notional_usd
        // If a setup had equity_pct saved but size_mode was stale / absent,
        // it fell to the notional_usd branch and shipped a tiny order.
        // New policy: equity_pct > 0 ALWAYS wins. notional_usd is legacy
        // fallback only when equity_pct is unset.
        let notional = 0;
        let sizeSrc = ''; // for preview label — tells user which formula ran
        // Priority: risk_pct > equity_pct > notional_usd.
        // risk_pct mode REQUIRES equity fetch. Leverage is ignored.
        if (c.size_mode === 'risk_pct' && Number(c.risk_pct) > 0) {
          const mode = c.exchange_mode === 'paper' ? 'demo' : (c.exchange_mode || 'live');
          const eq = getCachedEquity(mode);
          const totalCostPct = totalPct + (takerFeePct * 2);
          if (eq > 0 && totalCostPct > 0) {
            notional = eq * (Number(c.risk_pct) / 100) / (totalCostPct / 100);
            sizeSrc = `$${eq.toFixed(0)} x ${c.risk_pct}% / ${totalCostPct.toFixed(3)}%`;
          } else {
            sizeSrc = eq <= 0 ? 'pending equity...' : 'buffer+stop+fee must > 0';
          }
        } else if (Number(c.equity_pct) > 0) {
          const mode = c.exchange_mode === 'paper' ? 'demo' : (c.exchange_mode || 'live');
          const eq = getCachedEquity(mode);
          notional = eq > 0 ? eq * (Number(c.equity_pct) / 100) * lev : 0;
          sizeSrc = eq > 0
            ? `$${eq.toFixed(0)} x ${c.equity_pct}% x ${lev}x`
            : `pending equity...`;
        } else {
          notional = (Number(c.notional_usd) || 0) * lev;
          sizeSrc = `$${(Number(c.notional_usd) || 0).toFixed(0)} x ${lev}x`;
        }
        const feeUsd = notional * (takerFeePct * 2) / 100;
        const riskPriceUsd = notional * (totalPct / 100);
        const riskNet = riskPriceUsd + feeUsd;
        const rewardNet = riskPriceUsd * rr - feeUsd;
        const effRr = riskNet > 0 ? (rewardNet / riskNet) : 0;
        const effColor = effRr >= 2 ? '#00e676' : (effRr >= 1.3 ? '#fbbf24' : '#ff5252');
        // Preview line now ALWAYS shows real computed notional up front so
        // user can catch "wait, that's wrong" before clicking.
        const notionalLine = `<div style="color:#38bdf8;font-size:10px;margin-top:2px;font-weight:600">` +
          `≈ $${notional.toFixed(0)} notional <span style="color:#6b7889;font-weight:400">(${sizeSrc})</span>` +
          `</div>`;
        const pnlLine = notional > 0
          ? `<div style="color:#6b7889;font-size:10px;margin-top:2px">` +
              `风险 $${riskNet.toFixed(2)} · 目标 $${rewardNet.toFixed(2)} · ` +
              `<span style="color:${effColor}">RR ${effRr.toFixed(2)}</span> · 费 $${feeUsd.toFixed(2)}` +
            `</div>`
          : '';
        return `
          <div class="qt-setup-row" data-setup-id="${esc(s.id)}" style="padding:8px 10px;cursor:pointer;border-radius:5px;margin:2px 0;background:#16202f;">
            <div style="color:#d8dde8;font-size:12px">${esc(s.name)}</div>
            ${notionalLine}
            ${pnlLine}
          </div>
        `;
      }).join('');
      _quickPopup.innerHTML = `
        <div style="color:#8a95a6;font-size:11px;font-weight:600;margin-bottom:8px">
          ⚡ 快捷挂单 · ${esc(line.symbol)} ${esc(line.timeframe)}
        </div>

        <div style="display:flex;gap:6px;margin-bottom:6px">
          <div class="qt-dir-btn" data-dir="long" style="${longCls};flex:1;text-align:center;padding:6px;border-radius:5px;cursor:pointer">做多 LONG</div>
          <div class="qt-dir-btn" data-dir="short" style="${shortCls};flex:1;text-align:center;padding:6px;border-radius:5px;cursor:pointer">做空 SHORT</div>
        </div>

        <div class="qt-dir-hint" style="color:#fbbf24;font-size:10px;margin-bottom:6px;text-align:center;${direction ? 'display:none' : ''}">
          ⚠ 先选方向再点 setup
        </div>

        <div class="qt-rev-btn" style="${revCls};text-align:center;padding:6px;border-radius:5px;cursor:pointer;margin-bottom:10px">
          ${reverseEnabled ? '✓ 自动反手 ON' : '自动反手 OFF'}
        </div>

        <div style="color:#8a95a6;font-size:10px;padding:0 2px 4px 2px;text-transform:uppercase;letter-spacing:0.5px">Setup (点击立即挂单)</div>
        <div style="max-height:220px;overflow-y:auto;">${setupRows || '<div style="color:#8a95a6;padding:10px;font-size:11px">没有 setup，请先在完整配置里创建</div>'}</div>

        <div style="height:1px;background:#1d2537;margin:8px 0 6px 0"></div>
        <div class="qt-full-cfg" style="padding:6px 4px;cursor:pointer;color:#38bdf8;font-size:11px">⚙ 需要改参数? → 打开完整配置</div>
        <div id="qt-status" style="color:#8a95a6;font-size:11px;margin-top:6px;min-height:14px"></div>
      `;

      // Hover highlight for setup rows
      _quickPopup.querySelectorAll('.qt-setup-row').forEach((el) => {
        el.addEventListener('mouseenter', () => { el.style.background = '#223049'; });
        el.addEventListener('mouseleave', () => { el.style.background = '#16202f'; });
      });
    }

    renderBody();
    document.body.appendChild(_quickPopup);

    // Viewport clamp
    try {
      const rect = _quickPopup.getBoundingClientRect();
      const vw = window.innerWidth, vh = window.innerHeight, pad = 8;
      let nx = x, ny = y;
      if (nx + rect.width + pad > vw) nx = vw - rect.width - pad;
      if (ny + rect.height + pad > vh) ny = vh - rect.height - pad;
      if (nx < pad) nx = pad;
      if (ny < pad) ny = pad;
      _quickPopup.style.left = `${nx}px`;
      _quickPopup.style.top = `${ny}px`;
    } catch {}

    const onDocClick = (ev) => {
      if (_quickPopup && !_quickPopup.contains(ev.target)) {
        cleanup();
        resolve(null);
      }
    };
    const onKey = (ev) => {
      if (ev.key === 'Escape') { cleanup(); resolve(null); }
    };
    const cleanup = () => {
      if (_quickPopup && _quickPopup.parentNode) _quickPopup.parentNode.removeChild(_quickPopup);
      _quickPopup = null;
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
      _quickPopupCleanup = null;
    };
    _quickPopupCleanup = cleanup;

    const setStatus = (msg, color = '#8a95a6') => {
      const s = _quickPopup?.querySelector('#qt-status');
      if (s) { s.textContent = msg; s.style.color = color; }
    };

    // Hard click lock. User 2026-04-21: clicked setup 3 times when the
    // first click appeared to do nothing (request was in-flight but no
    // visible feedback). Result: 3 duplicate orders on Bitget. Keep the
    // guard, but don't dim/blur the whole popup: user 2026-04-22 reported
    // the "faded shadow" made it look broken.
    let _placing = false;
    const _applyPlacingState = (on) => {
      _placing = on;
      if (_quickPopup) {
        _quickPopup.dataset.placing = on ? '1' : '0';
        _quickPopup.style.borderColor = on ? '#38bdf8' : '#2a3548';
      }
    };

    // Click handler
    _quickPopup.addEventListener('click', async (ev) => {
      if (_placing) {
        // Defense-in-depth: keep double-clicks from posting duplicate
        // orders while leaving the popup visually readable.
        ev.preventDefault();
        ev.stopPropagation();
        return;
      }
      const dirBtn = ev.target.closest('.qt-dir-btn');
      const revBtn = ev.target.closest('.qt-rev-btn');
      const setupRow = ev.target.closest('.qt-setup-row');
      const fullCfg = ev.target.closest('.qt-full-cfg');

      if (dirBtn) {
        direction = dirBtn.dataset.dir;
        renderBody();
        return;
      }
      if (revBtn) {
        reverseEnabled = !reverseEnabled;
        renderBody();
        return;
      }
      if (fullCfg) {
        cleanup();
        const result = await openTradePlanModal(line);
        resolve(result);
        return;
      }
      if (setupRow) {
        if (!direction) {
          setStatus('⚠ 请先选方向 (做多 LONG / 做空 SHORT)', '#fbbf24');
          return;
        }
        const id = setupRow.dataset.setupId;
        const s = setupsState.setups.find((x) => x.id === id);
        if (!s) { setStatus('setup 找不到', '#ff5252'); return; }
        const cfg = s.config || {};
        // LOCK IMMEDIATELY — before any await, before any network call.
        _applyPlacingState(true);
        setStatus(`⏳ 挂单中 "${s.name}"… (最多 30s)`, '#38bdf8');  // SAFE: setStatus uses textContent
        // Compute size_usdt. Same convention as full modal:
        //   notional_usd = STAKE (margin), actual notional = stake × leverage
        //   equity_pct mode = equity × pct × leverage
        const lev = Number(cfg.leverage) > 0 ? Number(cfg.leverage) : 1;
        let size_usdt = 0;
        // User 2026-04-22: popup kept saying "signal is aborted without
        // reason → 暂停挂单" because a transient /account abort produced
        // equity=0 AND our cache had been cleared. Fix: try fresh fetch,
        // fall back to getCachedEquity (now keeps last good for 5 min).
        // Only fail LOUD if BOTH fresh fetch AND stale cache unusable.
        const _resolveEquity = async () => {
          const mode = cfg.exchange_mode === 'paper' ? 'demo' : (cfg.exchange_mode || 'live');
          try {
            const { getLiveAccount: _gl } = await import('../../services/live_execution.js');
            const acct = await _gl(mode);
            const eq = Number(acct?.total_equity ?? acct?.usdt_available ?? acct?.equity ?? 0);
            if (eq > 0) {
              // Populate our module cache so future calls see it.
              _equityCache = { freshTs: Date.now(), staleTs: Date.now(), equity: eq, mode, error: '' };
              return { equity: eq, stale: false };
            }
          } catch (err) {
            console.warn('[popup] live equity fetch failed, falling back to cache:', err?.message || err);
          }
          const cached = getCachedEquity(mode);
          return { equity: cached, stale: cached > 0 };
        };

        try {
          if (cfg.size_mode === 'risk_pct' && Number(cfg.risk_pct) > 0) {
            const { equity, stale } = await _resolveEquity();
            if (!(equity > 0)) {
              setStatus('⚠ 风险%模式需要 equity,实时+缓存都无,暂停挂单', '#ff5252');  // SAFE
              _applyPlacingState(false);
              return;
            }
            if (stale) {
              setStatus(`⏳ 用缓存权益 $${equity.toFixed(2)} 算 (实时 fetch 失败)…`, '#fbbf24');  // SAFE
            }
            const _buffer = Math.abs(Number(cfg.buffer_pct) || 0);
            const _stop = Math.abs(Number(cfg.stop_pct) || 0);
            let _tfee = 0.06;
            try {
              const _ov = parseFloat(localStorage.getItem('v2.trade.takerFeePct') || '');
              if (Number.isFinite(_ov) && _ov > 0) _tfee = _ov;
            } catch {}
            const _totalCostPct = _buffer + _stop + (_tfee * 2);
            if (_totalCostPct <= 0) {
              setStatus('⚠ buffer+stop+fee 必须 > 0 才能算风险%仓位', '#ff5252');  // SAFE
              _applyPlacingState(false);
              return;
            }
            size_usdt = equity * (Number(cfg.risk_pct) / 100) / (_totalCostPct / 100);
          } else if (Number(cfg.equity_pct) > 0) {
            const { equity, stale } = await _resolveEquity();
            if (equity > 0) {
              size_usdt = equity * (Number(cfg.equity_pct) / 100) * lev;
              if (stale) {
                setStatus(`⏳ 用缓存权益 $${equity.toFixed(2)} 算 (实时 fetch 失败)…`, '#fbbf24');  // SAFE
              }
            } else {
              setStatus('⚠ 账户权益拉取失败 (实时+缓存都无), 暂停挂单防止下错金额', '#ff5252');  // SAFE
              _applyPlacingState(false);
              return;
            }
          } else {
            size_usdt = (Number(cfg.notional_usd) || 0) * lev;
          }
        } catch (e) {
          if ((cfg.size_mode === 'risk_pct' && Number(cfg.risk_pct) > 0) || Number(cfg.equity_pct) > 0) {
            setStatus(`⚠ 算仓位时报错: ${e.message || e} — 暂停挂单`, '#ff5252');  // SAFE: setStatus uses textContent
            _applyPlacingState(false);
            return;
          }
          size_usdt = (Number(cfg.notional_usd) || 0) * lev;
        }
        if (!(size_usdt > 0)) {
          setStatus('仓位必须 > 0 USDT', '#ff5252');
          _applyPlacingState(false);       // unlock so user can pick a different setup
          return;
        }

        const payload = {
          manual_line_id: line.manual_line_id,
          direction,                                // user override
          kind: cfg.order_kind === 'breakout' ? 'break' : 'bounce',
          tolerance_pct: Number(cfg.buffer_pct) || 0.1,
          stop_offset_pct: Number(cfg.stop_pct) || 0.1,
          size_usdt,
          leverage: cfg.leverage > 0 ? Math.round(cfg.leverage) : 1,
          mode: cfg.exchange_mode === 'paper' ? 'demo' : 'live',
          rr_target: Number(cfg.rr_target) || 2,
          // Dynamic qty resize on account equity change.
          equity_pct: cfg.size_mode === 'equity_pct' && cfg.equity_pct > 0 ? cfg.equity_pct : null,
          reverse_enabled: reverseEnabled,          // user override
          reverse_entry_offset_pct: 0,              // reverse has no buffer (same spec as full modal)
          reverse_stop_offset_pct: Number(cfg.reverse_stop_pct) || 0,
          reverse_rr_target: Number(cfg.reverse_rr_target) || null,
          reverse_leverage: cfg.leverage > 0 ? Math.round(cfg.leverage) : null,
        };
        try {
          const resp = await placeLineOrder(payload);
          if (resp?.ok) {
            const cid = resp.conditional?.conditional_id || resp.conditional_id || '';
            const oid = resp.exchange_order_id || resp.conditional?.exchange_order_id || '';
            const parts = ['✓ Bitget 已确认挂单'];
            if (cid) parts.push(`cond ${String(cid).slice(0, 14)}`);
            if (oid) parts.push(`oid ...${String(oid).slice(-8)}`);
            setStatus(parts.join(' · '), '#00e676');  // SAFE: setStatus uses textContent
            // Notify sidebar panels to refresh NOW (not wait 5s for next
            // poll). User 2026-04-22: "挂单后 panel 看不到,必须刷新"
            try {
              window.dispatchEvent(new CustomEvent('cond-placed', {
                detail: { symbol: line.symbol, tf: line.timeframe, cid, oid },
              }));
            } catch {}
            setTimeout(() => { cleanup(); resolve(resp); }, 2500);
            // DON'T unlock on success — popup is closing anyway, prevent
            // any last-ms double click.
          } else {
            // Translate common backend reasons into actionable Chinese
            // so user knows which setting to change (not just "失败").
            // 2026-04-21: ETH rejected with size_below_min_trade because
            // equity × % × leverage produced <$30 notional (ETH min).
            const raw = String(resp?.reason || '未知错误');
            let friendly = raw;
            if (raw.includes('size_below_min_trade') || raw.includes('below_min')) {
              friendly = `仓位太小 (notional $${(payload.size_usdt || 0).toFixed(2)}) — 低于 ${line.symbol} 最小挂单额。把 setup 的 % 或杠杆调高再试`;
            } else if (raw.includes('insufficient_balance') || raw.includes('insufficient')) {
              friendly = `余额不足 — 当前可用 < 需要的保证金 $${((payload.size_usdt || 0) / (payload.leverage || 1)).toFixed(2)}`;
            } else if (raw.includes('leverage')) {
              friendly = `杠杆设置被拒: ${raw}`;
            } else if (raw.includes('price') && raw.includes('distance')) {
              friendly = `线距离当前价太远,Bitget 拒绝`;
            }
            setStatus(`✗ ${friendly}`, '#ff5252');  // SAFE: setStatus uses textContent
            _applyPlacingState(false);     // let user retry or pick another setup
          }
        } catch (err) {
          // "signal is aborted without reason" = request exceeded timeout.
          // Bitget might have RECEIVED the order; we just didn't get ack.
          // Tell user to check the app, don't pretend it definitely failed.
          // DO NOT unlock the popup on abort — the in-flight request may
          // have actually succeeded. User must manually close + reopen
          // to try again. This prevents the 3-duplicates bug user hit
          // 2026-04-21.
          const msg = String(err?.message || err || '');
          const aborted = /abort|timeout|signal/i.test(msg);
          if (aborted) {
            setStatus('⚠ 网络超时 — 去 Bitget app 查是否已挂上。本 popup 锁定防止重复挂单，关掉再开', '#fbbf24');  // SAFE: setStatus uses textContent
            // popup stays LOCKED; user must explicitly close it
          } else {
            setStatus(`✗ 错误: ${msg}`, '#ff5252');  // SAFE: setStatus uses textContent
            _applyPlacingState(false);     // non-abort errors are safe to retry
          }
        }
      }
    });

    setTimeout(() => { document.addEventListener('mousedown', onDocClick); }, 0);
    document.addEventListener('keydown', onKey);
  });
}
