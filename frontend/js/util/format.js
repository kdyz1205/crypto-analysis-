// frontend/js/util/format.js — number / currency formatting

export function inferPrecision(price) {
  if (price == null || !isFinite(price)) return 2;
  if (price >= 1000) return 2;
  if (price >= 1) return 4;
  if (price >= 0.01) return 6;
  return 8;
}

export function formatPrice(price, precision) {
  if (price == null || !isFinite(price)) return '—';
  const p = precision ?? inferPrecision(price);
  return Number(price).toFixed(p);
}

export function formatPct(pct, digits = 2) {
  if (pct == null || !isFinite(pct)) return '—';
  const sign = pct > 0 ? '+' : '';
  return `${sign}${Number(pct).toFixed(digits)}%`;
}

export function formatUsd(value, digits = 2) {
  if (value == null || !isFinite(value)) return '—';
  const sign = value < 0 ? '-' : '';
  return `${sign}$${Math.abs(Number(value)).toFixed(digits)}`;
}

export function pnlColorClass(pnl) {
  if (pnl == null || pnl === 0) return '';
  return pnl > 0 ? 'pnl-pos' : 'pnl-neg';
}
