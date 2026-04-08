// frontend/js/command_palette/palette.js — Cmd+K modal UI

import { fuzzySort } from './fuzzy.js';
import { buildCommands } from './commands.js';

let overlayEl = null;
let inputEl = null;
let listEl = null;
let isOpen = false;
let selectedIdx = 0;
let results = [];

function build() {
  overlayEl = document.createElement('div');
  overlayEl.id = 'cmdk-overlay';
  overlayEl.className = 'cmdk-overlay hidden';
  overlayEl.innerHTML = `
    <div class="cmdk-modal" role="dialog" aria-label="Command palette">
      <input class="cmdk-input" type="text" placeholder="Type a command, symbol, or timeframe...">
      <ul class="cmdk-list" role="listbox"></ul>
      <div class="cmdk-footer">
        <kbd>↑↓</kbd> navigate · <kbd>Enter</kbd> select · <kbd>Esc</kbd> close
      </div>
    </div>
  `;
  document.body.appendChild(overlayEl);
  inputEl = overlayEl.querySelector('.cmdk-input');
  listEl = overlayEl.querySelector('.cmdk-list');

  overlayEl.addEventListener('click', (e) => { if (e.target === overlayEl) closePalette(); });
  inputEl.addEventListener('input', render);
  inputEl.addEventListener('keydown', onKey);
}

function render() {
  const q = inputEl.value;
  const cmds = buildCommands();
  results = q
    ? fuzzySort(q, cmds, (c) => c.label).map((r) => ({ item: r.item }))
    : cmds.slice(0, 30).map((c) => ({ item: c }));
  selectedIdx = 0;

  listEl.innerHTML = results.map((r, i) => `
    <li class="cmdk-item ${i === selectedIdx ? 'selected' : ''}" data-idx="${i}" role="option">
      <span class="cmdk-label">${r.item.label}</span>
      <span class="cmdk-category">${r.item.category}</span>
    </li>
  `).join('') || '<li class="cmdk-empty">No matches</li>';

  Array.from(listEl.children).forEach((el, i) => {
    el.addEventListener('click', () => { selectedIdx = i; execute(); });
    el.addEventListener('mouseenter', () => { selectedIdx = i; updateSelection(); });
  });
}

function onKey(e) {
  if (e.key === 'Escape') { closePalette(); return; }
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    selectedIdx = Math.min(selectedIdx + 1, results.length - 1);
    updateSelection();
    return;
  }
  if (e.key === 'ArrowUp') {
    e.preventDefault();
    selectedIdx = Math.max(selectedIdx - 1, 0);
    updateSelection();
    return;
  }
  if (e.key === 'Enter') {
    e.preventDefault();
    execute();
  }
}

function updateSelection() {
  Array.from(listEl.children).forEach((el, i) => el.classList.toggle('selected', i === selectedIdx));
  listEl.children[selectedIdx]?.scrollIntoView({ block: 'nearest' });
}

function execute() {
  const entry = results[selectedIdx];
  if (!entry) return;
  try { entry.item.action(); }
  catch (err) { console.error('[cmdk] action failed:', err); }
  closePalette();
}

export function openPalette() {
  if (!overlayEl) build();
  isOpen = true;
  overlayEl.classList.remove('hidden');
  inputEl.value = '';
  inputEl.focus();
  render();
}

export function closePalette() {
  if (!overlayEl) return;
  isOpen = false;
  overlayEl.classList.add('hidden');
}

export function initCommandPalette() {
  document.addEventListener('keydown', (e) => {
    const ctrlK = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k';
    if (ctrlK) {
      e.preventDefault();
      isOpen ? closePalette() : openPalette();
    }
  });
}
