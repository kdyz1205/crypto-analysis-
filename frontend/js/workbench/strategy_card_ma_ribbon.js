// MA Ribbon EMA21 Auto — strategy card. Pure DOM, no framework.
//
// Mounts under a parent element. Polls /api/ma_ribbon_auto/status every 5 s.
// Provides: Enable (with double-confirm modal) / Disable / Emergency Stop
// (with type-to-confirm "STOP") buttons. Surfaces strategy state, ramp-up
// progress, open positions count, realized PnL, pending signals count.

export function mountMaRibbonCard(rootEl) {
  rootEl.innerHTML = `
    <div class="strategy-card" id="ma-ribbon-card">
      <div class="card-header">
        <h3>MA Ribbon EMA21 · Auto Live</h3>
        <span class="status-badge" id="mar-status">DISABLED</span>
      </div>
      <div class="card-body">
        <div class="ramp-row">
          <span>Ramp-up:</span>
          <span id="mar-ramp-day">—</span>
          <progress id="mar-ramp-bar" value="0" max="100"></progress>
        </div>
        <div class="metric-row">
          <div><label>Open positions</label><span id="mar-open">0</span></div>
          <div><label>Realized PnL</label><span id="mar-pnl">$0.00</span></div>
          <div><label>Pending signals</label><span id="mar-pending">0</span></div>
        </div>
        <details>
          <summary>Config (click to expand)</summary>
          <div id="mar-config-form">
            <div class="config-row">
              <label>max_concurrent_orders</label>
              <span id="mar-cfg-concurrent">—</span>
            </div>
            <div class="config-row">
              <label>dd_halt_pct</label>
              <span id="mar-cfg-dd">—</span>
            </div>
            <div class="config-row">
              <label>per_symbol_risk_cap_pct</label>
              <span id="mar-cfg-symcap">—</span>
            </div>
            <div class="config-row">
              <label>strategy_capital_usd</label>
              <span id="mar-cfg-capital">—</span>
            </div>
          </div>
        </details>
        <div class="actions">
          <button id="mar-enable-btn" class="primary">Enable</button>
          <button id="mar-disable-btn">Disable</button>
          <button id="mar-stop-btn" class="danger">EMERGENCY STOP</button>
        </div>
        <div class="errors" id="mar-errors"></div>
      </div>
    </div>
  `;
  attachHandlers(rootEl);
  startPolling(rootEl);
}

async function fetchStatus() {
  const r = await fetch("/api/ma_ribbon_auto/status");
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

function attachHandlers(rootEl) {
  rootEl.querySelector("#mar-enable-btn").addEventListener("click", onEnableClick);
  rootEl.querySelector("#mar-disable-btn").addEventListener("click", onDisableClick);
  rootEl.querySelector("#mar-stop-btn").addEventListener("click", onEmergencyClick);
}

async function onEnableClick() {
  const ack1 = window.confirm(
    "I have run the strategy on the backtest panel (port 8765) and reviewed the cohort + Phase 2 results."
  );
  if (!ack1) return;
  const ack2 = window.confirm(
    "I understand the first-day total-risk cap is 2 % and ramps over 14 days to 15 %."
  );
  if (!ack2) return;
  const cap = window.prompt("Strategy capital (USD)?", "1000");
  if (!cap) return;
  const r = await fetch("/api/ma_ribbon_auto/enable", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      confirm_acknowledged_p2_gate: true,
      confirm_first_day_cap_2pct: true,
      strategy_capital_usd: parseFloat(cap),
    }),
  });
  if (!r.ok) {
    alert("enable failed: " + await r.text());
    return;
  }
  refresh();
}

async function onDisableClick() {
  const r = await fetch("/api/ma_ribbon_auto/disable", {method: "POST"});
  if (!r.ok) {
    alert("disable failed: " + await r.text());
    return;
  }
  refresh();
}

async function onEmergencyClick() {
  const typed = window.prompt('Type "STOP" to flatten all MA-ribbon positions and lock for 24 h:');
  if (typed !== "STOP") return;
  const r = await fetch("/api/ma_ribbon_auto/emergency_stop", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({reason: "user click"}),
  });
  if (!r.ok) {
    alert("emergency stop failed: " + await r.text());
    return;
  }
  refresh();
}

let _pollTimer = null;
function startPolling(_rootEl) {
  refresh();
  _pollTimer = setInterval(refresh, 5000);
}

async function refresh() {
  try {
    const s = await fetchStatus();
    setText("mar-status",
      s.enabled ? (s.halted ? `HALTED (${s.halt_reason || "?"})` : "ENABLED") : "DISABLED");
    setText("mar-open", s.ledger.open_positions_count);
    setText("mar-pnl", "$" + Number(s.ledger.realized_pnl_usd_cumulative).toFixed(2));
    setText("mar-pending", s.pending_signals_count);
    const rampPct = (s.current_ramp_cap_pct * 100).toFixed(1);
    setText("mar-ramp-day", `cap ${rampPct} %`);
    const bar = document.getElementById("mar-ramp-bar");
    if (bar) bar.value = parseFloat(rampPct);
    setText("mar-cfg-concurrent", s.config.max_concurrent_orders);
    setText("mar-cfg-dd", (s.config.dd_halt_pct * 100).toFixed(1) + " %");
    setText("mar-cfg-symcap", (s.config.per_symbol_risk_cap_pct * 100).toFixed(1) + " %");
    setText("mar-cfg-capital", "$" + Number(s.config.strategy_capital_usd).toLocaleString());
    setText("mar-errors", "");
  } catch (e) {
    setText("mar-errors", "status fetch failed: " + e.message);
  }
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(value);
}
