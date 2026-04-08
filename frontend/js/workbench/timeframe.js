// frontend/js/workbench/timeframe.js — timeframe button group

import { $$, on } from '../util/dom.js';
import { marketState, setIntervalTF } from '../state/market.js';

export function initTimeframe(containerSelector = '#v2-tf-group') {
  const buttons = $$(containerSelector + ' .v2-tf-btn');
  buttons.forEach((btn) => {
    on(btn, 'click', () => {
      const tf = btn.dataset.tf;
      if (!tf) return;
      buttons.forEach((b) => b.classList.toggle('active', b === btn));
      setIntervalTF(tf);
    });
    if (btn.dataset.tf === marketState.currentInterval) {
      btn.classList.add('active');
    }
  });
}
