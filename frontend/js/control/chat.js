// frontend/js/control/chat.js — v2 chat dock with context auto-injection

import { $, setHtml, on } from '../util/dom.js';
import { uiState, setChatDock } from '../state/ui.js';
import { marketState } from '../state/market.js';
import { agentState } from '../state/agent.js';
import * as chatSvc from '../services/chat.js';
import { subscribe } from '../util/events.js';

let messagesEl = null;
let inputEl = null;
let modelSelect = null;
let sending = false;

function buildDock() {
  const existing = $('#v2-chat-dock');
  if (existing) return existing;

  const dock = document.createElement('div');
  dock.id = 'v2-chat-dock';
  dock.className = 'chat-dock hidden';
  dock.innerHTML = `
    <div class="chat-dock-header">
      <span class="chat-title">AI Chat</span>
      <select class="chat-model" id="v2-chat-model">
        <option value="claude-sonnet-4-5-20250929">Sonnet</option>
        <option value="claude-opus-4-5-20250929">Opus</option>
        <option value="claude-haiku-4-5-20251001">Haiku</option>
      </select>
      <button class="chat-close" id="v2-chat-close">×</button>
    </div>
    <div class="chat-messages" id="v2-chat-messages">
      <div class="chat-welcome">
        Hi! I'm your trading assistant. I can:
        <ul>
          <li>Analyze current charts</li>
          <li>Run backtests</li>
          <li>Control the agent</li>
          <li>Remember things — just say "记住..."</li>
          <li>Schedule tasks — "每 15 分钟扫一次"</li>
        </ul>
      </div>
    </div>
    <div class="chat-input-row">
      <textarea id="v2-chat-input" placeholder="Ask about the chart, or tell me to do something..." rows="2"></textarea>
      <button class="btn btn-primary" id="v2-chat-send">Send</button>
    </div>
  `;
  document.body.appendChild(dock);
  messagesEl = $('#v2-chat-messages');
  inputEl = $('#v2-chat-input');
  modelSelect = $('#v2-chat-model');
  return dock;
}

function buildContextLine() {
  const sym = marketState.currentSymbol;
  const iv = marketState.currentInterval;
  const s = agentState.lastStatus || {};
  const posCount = Object.keys(s.positions || {}).length;
  return `[Context] symbol=${sym} tf=${iv} agent_mode=${s.mode || '?'} equity=$${(s.equity ?? 0).toFixed(2)} positions=${posCount} regime=${s.harness?.market_regime || '?'}`;
}

function appendMessage(role, text) {
  if (!messagesEl) return;
  const div = document.createElement('div');
  div.className = `chat-msg chat-msg-${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function send() {
  if (sending || !inputEl) return;
  const text = inputEl.value.trim();
  if (!text) return;
  sending = true;

  appendMessage('user', text);
  inputEl.value = '';

  // Prepend context automatically
  const ctx = buildContextLine();
  const finalMessage = `${ctx}\n\n${text}`;

  try {
    const model = modelSelect?.value || null;
    const resp = await chatSvc.sendChat(finalMessage, 'v2', model);
    const reply = resp?.reply || resp?.message || JSON.stringify(resp);
    appendMessage('assistant', reply);
  } catch (err) {
    // SAFE: appendMessage uses textContent, no HTML parsing
    appendMessage('assistant', `Error: ${err?.message || String(err)}`);
  } finally {
    sending = false;
  }
}

export function initChatDock() {
  buildDock();
  on('#v2-chat-close', 'click', () => setChatDock(false));
  on('#v2-chat-send', 'click', send);
  on('#v2-chat-input', 'keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  subscribe('ui.chat.toggled', (open) => {
    const dock = $('#v2-chat-dock');
    if (open) {
      dock?.classList.remove('hidden');
      inputEl?.focus();
    } else {
      dock?.classList.add('hidden');
    }
  });
}

export function openChatDock() { setChatDock(true); }
export function closeChatDock() { setChatDock(false); }
