// frontend/js/util/dom.js — lightweight DOM helpers

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function on(el, event, handler, options) {
  if (typeof el === 'string') el = $(el);
  if (!el) return () => {};
  el.addEventListener(event, handler, options);
  return () => el.removeEventListener(event, handler, options);
}

export function show(el) {
  const node = typeof el === 'string' ? $(el) : el;
  node?.classList.remove('hidden');
}

export function hide(el) {
  const node = typeof el === 'string' ? $(el) : el;
  node?.classList.add('hidden');
}

export function toggleClass(el, cls, force) {
  const node = typeof el === 'string' ? $(el) : el;
  node?.classList.toggle(cls, force);
}

export function setText(el, text) {
  const node = typeof el === 'string' ? $(el) : el;
  if (node) node.textContent = text;
}

export function setHtml(el, html) {
  const node = typeof el === 'string' ? $(el) : el;
  if (node) node.innerHTML = html;
}

/**
 * Escape a string for safe interpolation into innerHTML / template literals.
 * Round 1/10 #18-25 + Round 9 stage 4: server error messages, exception
 * strings, symbol names, and any other field that might contain user-
 * influenced data must be escaped before being inserted via innerHTML.
 *
 * Use it like this (the bad form below is the kind of thing the grep
 * sweep flags — see scripts/grep_scan_sweep.py rule xss_error_interp):
 *
 *   const safe   = `<div>${esc(err.message)}</div>`;
 *   const unsafe = `<div>${err_dot_message}</div>`;  // SAFE: doc example
 */
export function escapeHtml(value) {
  if (value == null) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Short alias for use in dense template literals
export const esc = escapeHtml;
