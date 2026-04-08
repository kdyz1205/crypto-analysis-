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
