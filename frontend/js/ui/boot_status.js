// frontend/js/ui/boot_status.js
// Top-right badge showing status of each boot component.
// Each component calls markBoot(name, state) where state is 'pending'|'ok'|'error'.

import { $ } from '../util/dom.js';

const COMPONENTS = [
  { key: 'symbols', label: 'Symbols' },
  { key: 'chart', label: 'Chart' },
  { key: 'patterns', label: 'Patterns' },
  { key: 'rail', label: 'Decision Rail' },
  { key: 'stream', label: 'SSE Stream' },
  { key: 'exec', label: 'Execution' },
];

const state = new Map(); // key → { state, detail }
let root = null;

function render() {
  if (!root) return;
  root.innerHTML = COMPONENTS.map(({ key, label }) => {
    const s = state.get(key);
    const cls = s?.state || 'pending';
    const icon = cls === 'ok' ? '●' : cls === 'error' ? '✕' : '○';
    return `<span class="boot-pill boot-${cls}" title="${label}: ${s?.detail || cls}">${icon} ${label}</span>`;
  }).join('');
}

export function initBootStatus() {
  if (root) return;
  root = document.createElement('div');
  root.id = 'v2-boot-status';
  root.className = 'boot-status';
  const mountPoint = $('#v2-boot-status-slot');
  if (mountPoint) {
    mountPoint.appendChild(root);
  } else {
    root.classList.add('boot-status-floating');
    document.body.appendChild(root);
  }
  for (const { key } of COMPONENTS) state.set(key, { state: 'pending', detail: '' });
  render();
}

export function markBoot(key, stateStr, detail = '') {
  state.set(key, { state: stateStr, detail });
  render();
}

export function getBootState() {
  const out = {};
  for (const [k, v] of state) out[k] = v.state;
  return out;
}
