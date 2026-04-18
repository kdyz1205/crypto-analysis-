// frontend/js/workbench/timeframe.js — timeframe button group

import { $, $$, on } from '../util/dom.js';
import { marketState, setIntervalTF } from '../state/market.js';
import { subscribe } from '../util/events.js';

const initialized = new WeakSet();

function syncButtons(container, activeTf = marketState.currentInterval) {
  $$('.v2-tf-btn', container).forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tf === activeTf);
  });
}

export function initTimeframe(containerSelector = '#v2-tf-group') {
  const container = $(containerSelector);
  if (!container) return;

  syncButtons(container);
  if (initialized.has(container)) return;
  initialized.add(container);

  on(container, 'click', (event) => {
    const btn = event.target?.closest?.('.v2-tf-btn');
    if (!btn || !container.contains(btn)) return;
    const tf = btn.dataset.tf;
    if (!tf) return;
    syncButtons(container, tf);
    setIntervalTF(tf);
  });

  subscribe('market.interval.changed', (tf) => {
    syncButtons(container, tf);
  });
}
