// frontend/js/util/storage.js — localStorage wrapper

const PREFIX = 'cryptoTA:';

export function loadJSON(key, fallback = null) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}

export function saveJSON(key, value) {
  try { localStorage.setItem(PREFIX + key, JSON.stringify(value)); return true; }
  catch { return false; }
}

export function remove(key) { localStorage.removeItem(PREFIX + key); }
