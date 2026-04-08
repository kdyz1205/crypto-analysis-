// frontend/js/workbench/ticker.js — minimal symbol picker

import { $, setHtml, on } from '../util/dom.js';
import { marketState, setSymbol, setAllSymbols } from '../state/market.js';
import * as marketSvc from '../services/market.js';

const PINNED = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'HYPEUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT'];

export async function initTicker(selectId = 'v2-symbol-select') {
  try {
    const symbols = await marketSvc.getSymbols();
    setAllSymbols(symbols);
    renderSelect(selectId, symbols);
  } catch (err) {
    console.error('[ticker] failed to load symbols:', err);
  }
}

function renderSelect(selectId, symbols) {
  const el = $('#' + selectId);
  if (!el) return;

  const pinned = PINNED.filter(s => symbols.includes(s));
  const rest = symbols.filter(s => !pinned.includes(s));

  let html = '<optgroup label="Popular">';
  for (const s of pinned) html += `<option value="${s}">${s}</option>`;
  html += '</optgroup><optgroup label="All">';
  for (const s of rest) html += `<option value="${s}">${s}</option>`;
  html += '</optgroup>';

  setHtml(el, html);
  el.value = marketState.currentSymbol;

  on(el, 'change', (e) => setSymbol(e.target.value));
}
