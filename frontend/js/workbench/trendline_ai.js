/**
 * Trendline AI signal banner — visual edition.
 *
 * Renders a rich card (action badge + probability bars + suggested buffer
 * + decoded next-line metadata + collapsible raw JSON) instead of a raw
 * JSON dump. Adds Accept/Reject feedback buttons.
 *
 * The "draw on chart" button calls window.__tlAIChartOverlay if the host
 * provides one (so the panel can also paint the predicted line on the
 * main lightweight-chart). If no overlay is wired, the button is hidden.
 */

(function () {
  'use strict';

  const HEADER_ACTIONS_SELECTOR = '.v2-header-actions';
  const PANEL_ID = 'tl-ai-panel';

  function getCurrentSymbolTf() {
    const symInput = document.getElementById('v2-symbol-input');
    const tfActive = document.querySelector('.v2-tf-btn.active');
    return {
      symbol: (symInput && symInput.value && symInput.value.trim()) || 'BTCUSDT',
      timeframe: (tfActive && tfActive.dataset && tfActive.dataset.tf) || '4h',
    };
  }

  function colorForAction(action) {
    if (action === 'LONG') return { bg: '#16a34a', tx: '#ffffff', accent: '#34d399' };
    if (action === 'SHORT') return { bg: '#dc2626', tx: '#ffffff', accent: '#f87171' };
    return { bg: '#475569', tx: '#cbd5e1', accent: '#94a3b8' };
  }

  function pctBar(label, value, color) {
    const v = Math.max(0, Math.min(1, value || 0));
    const pct = (v * 100).toFixed(1);
    return `
      <div class="tl-bar-row" style="display:flex;align-items:center;gap:8px;margin:4px 0">
        <div style="width:64px;color:#94a3b8;font-size:11px">${label}</div>
        <div style="flex:1;height:14px;background:#0f1116;border:1px solid #2c313c;border-radius:3px;overflow:hidden;position:relative">
          <div style="height:100%;width:${pct}%;background:${color};transition:width 0.4s"></div>
          <div style="position:absolute;right:6px;top:0;line-height:14px;font-size:10px;color:#e6e9ef">${pct}%</div>
        </div>
      </div>
    `;
  }

  function ensurePanel() {
    let panel = document.getElementById(PANEL_ID);
    if (panel) return panel;
    panel = document.createElement('div');
    panel.id = PANEL_ID;
    panel.style.cssText = [
      'position:fixed', 'top:60px', 'right:16px', 'width:380px',
      'background:#1a1d24', 'color:#e6e9ef',
      'border:1px solid #2c313c', 'border-radius:10px',
      'padding:14px', 'font-family:system-ui,sans-serif', 'font-size:12px',
      'box-shadow:0 8px 24px rgba(0,0,0,0.5)', 'z-index:9999',
      'display:none',
    ].join(';');
    panel.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <strong style="font-size:13px">🧠 趋势线 AI 信号</strong>
        <div>
          <button id="tl-ai-refresh" title="刷新" style="background:none;border:none;color:#94a3b8;cursor:pointer;font-size:14px;margin-right:4px">↻</button>
          <button id="tl-ai-close" style="background:none;border:none;color:#94a3b8;cursor:pointer;font-size:18px">×</button>
        </div>
      </div>
      <div id="tl-ai-card" style="min-height:220px"></div>
    `;
    document.body.appendChild(panel);
    panel.querySelector('#tl-ai-close').addEventListener('click', () => {
      panel.style.display = 'none';
    });
    panel.querySelector('#tl-ai-refresh').addEventListener('click', fetchSignal);
    return panel;
  }

  let _lastSignal = null;

  function renderLoading() {
    const card = document.getElementById('tl-ai-card');
    if (!card) return;
    card.innerHTML = `<div style="color:#7a8290;text-align:center;padding:24px">🧠 模型推理中…</div>`;
  }

  function renderError(msg) {
    const card = document.getElementById('tl-ai-card');
    if (!card) return;
    card.innerHTML = `<div style="color:#f87171;padding:12px;background:#2a1010;border:1px solid #5a1a1a;border-radius:6px">⚠ ${msg}</div>`;
  }

  function renderSignal(sig) {
    const card = document.getElementById('tl-ai-card');
    if (!card) return;
    const { bg, tx, accent } = colorForAction(sig.action);
    const conf = (sig.confidence || 0);
    const buf = (sig.suggested_buffer_pct || 0) * 100;
    const role = sig.predicted_role || 'unknown';
    const tradeType = sig.trade_type || 'wait';
    const ts = sig.timestamp ? new Date(sig.timestamp).toLocaleString() : '';
    card.innerHTML = `
      <div style="background:${bg};color:${tx};padding:14px;border-radius:8px;display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <div>
          <div style="font-size:22px;font-weight:700;letter-spacing:1px">${sig.action}</div>
          <div style="font-size:11px;opacity:0.85;margin-top:2px">${tradeType.replace('_',' ')} · role=${role}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:18px;font-weight:600">${(conf*100).toFixed(0)}%</div>
          <div style="font-size:10px;opacity:0.8">confidence</div>
        </div>
      </div>

      <div style="margin:8px 0">
        ${pctBar('Bounce', sig.bounce_prob || 0, '#34d399')}
        ${pctBar('Break',  sig.break_prob  || 0, '#f87171')}
        ${pctBar('Cont.',  sig.continuation_prob || 0, '#60a5fa')}
      </div>

      <div style="display:flex;gap:8px;margin:10px 0">
        <div style="flex:1;background:#0f1116;border:1px solid #2c313c;border-radius:5px;padding:8px;text-align:center">
          <div style="color:#94a3b8;font-size:10px">建议 buffer</div>
          <div style="color:#e6e9ef;font-size:14px;font-weight:600">${buf.toFixed(2)}%</div>
        </div>
        <div style="flex:1;background:#0f1116;border:1px solid #2c313c;border-radius:5px;padding:8px;text-align:center">
          <div style="color:#94a3b8;font-size:10px">输入线数</div>
          <div style="color:#e6e9ef;font-size:14px;font-weight:600">${(sig.extras && sig.extras.n_input_records) || 0}</div>
        </div>
        <div style="flex:1;background:#0f1116;border:1px solid #2c313c;border-radius:5px;padding:8px;text-align:center">
          <div style="color:#94a3b8;font-size:10px">缓存 K 线</div>
          <div style="color:#e6e9ef;font-size:14px;font-weight:600">${(sig.extras && sig.extras.n_bars_in_cache) || 0}</div>
        </div>
      </div>

      <div id="tl-ai-reason" style="background:#0f1116;border:1px solid #2c313c;border-radius:5px;padding:8px;font-size:11px;color:#94a3b8;line-height:1.5"></div>

      <div style="display:flex;gap:6px;margin-top:10px">
        <button id="tl-ai-accept" style="flex:1;background:#16a34a;color:white;border:none;border-radius:5px;padding:8px;cursor:pointer;font-weight:600">✓ 接受</button>
        <button id="tl-ai-reject" style="flex:1;background:#dc2626;color:white;border:none;border-radius:5px;padding:8px;cursor:pointer;font-weight:600">✗ 拒绝</button>
      </div>

      <div id="tl-ai-feedback-status" style="margin-top:6px;font-size:11px;color:#94a3b8;min-height:14px"></div>

      <details style="margin-top:8px">
        <summary style="cursor:pointer;color:#7a8290;font-size:11px">原始 JSON / 模型版本</summary>
        <div style="font-size:10px;color:#7a8290;margin-top:6px">
          <div>artifact: ${sig.artifact_name}</div>
          <div>tokenizer: ${sig.tokenizer_version}</div>
          <div>${sig.symbol} ${sig.timeframe} · ${ts}</div>
        </div>
        <pre style="background:#0f1116;border:1px solid #2c313c;border-radius:4px;padding:8px;max-height:160px;overflow:auto;font-size:10px;color:#cbd5e1;margin-top:6px">${escapeHTML(JSON.stringify(sig, null, 2))}</pre>
      </details>
    `;
    // Set reason via textContent to avoid HTML injection (XSS-safe)
    const reasonEl = card.querySelector('#tl-ai-reason');
    if (reasonEl) reasonEl.textContent = sig.reason || '';
    card.querySelector('#tl-ai-accept').addEventListener('click', () => sendFeedback('signal_accepted'));
    card.querySelector('#tl-ai-reject').addEventListener('click', () => sendFeedback('signal_rejected'));
  }

  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;',
    }[c]));
  }

  async function fetchSignal() {
    ensurePanel();
    renderLoading();
    const { symbol, timeframe } = getCurrentSymbolTf();
    try {
      const resp = await fetch('/api/trendline/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, timeframe }),
      });
      if (!resp.ok) {
        const text = await resp.text();
        renderError(`${resp.status} ${resp.statusText} — ${text.slice(0, 200)}`);
        return;
      }
      const sig = await resp.json();
      _lastSignal = sig;
      renderSignal(sig);
    } catch (e) {
      renderError(`网络错误: ${e}`);
    }
  }

  async function sendFeedback(eventType) {
    if (!_lastSignal) return;
    const status = document.getElementById('tl-ai-feedback-status');
    if (status) status.textContent = '提交中…';
    try {
      const payload = {
        event_type: eventType,
        signal_id: `${_lastSignal.symbol}-${_lastSignal.timestamp}`,
        artifact_name: _lastSignal.artifact_name,
        tokenizer_version: _lastSignal.tokenizer_version,
        symbol: _lastSignal.symbol,
        timeframe: _lastSignal.timeframe,
        action: _lastSignal.action,
      };
      const resp = await fetch('/api/trendline/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (status) {
        status.textContent = resp.ok
          ? `✓ 已记录: ${eventType.replace('signal_','')}`
          : `✗ 失败 ${resp.status}`;
        status.style.color = resp.ok ? '#34d399' : '#f87171';
      }
    } catch (e) {
      if (status) {
        status.textContent = `✗ ${e}`;
        status.style.color = '#f87171';
      }
    }
  }

  function ensureButton() {
    if (document.getElementById('tl-ai-btn')) return;
    const actions = document.querySelector(HEADER_ACTIONS_SELECTOR);
    if (!actions) return;
    const btn = document.createElement('button');
    btn.id = 'tl-ai-btn';
    btn.className = 'btn';
    btn.title = '趋势线 AI 信号';
    btn.textContent = '🧠 AI 信号';
    btn.addEventListener('click', () => {
      const panel = ensurePanel();
      panel.style.display = 'block';
      fetchSignal();
    });
    actions.appendChild(btn);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensureButton);
  } else {
    ensureButton();
  }
})();
