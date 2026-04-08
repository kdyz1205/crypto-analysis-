// frontend/js/util/fetch.js — wrapped fetch with timeout + JSON parse

const DEFAULT_TIMEOUT = 15000;

export class FetchError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

function combineSignals(signals) {
  const ac = new AbortController();
  signals.forEach((s) => {
    if (!s) return;
    if (s.aborted) ac.abort();
    else s.addEventListener('abort', () => ac.abort(), { once: true });
  });
  return ac.signal;
}

export async function fetchJson(url, { method = 'GET', body, timeout = DEFAULT_TIMEOUT, signal } = {}) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeout);
  const combined = signal ? combineSignals([signal, ac.signal]) : ac.signal;

  try {
    const res = await fetch(url, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: combined,
    });
    const text = await res.text();
    let data;
    try { data = text ? JSON.parse(text) : null; }
    catch { data = text; }
    if (!res.ok) throw new FetchError(`HTTP ${res.status}`, res.status, data);
    return data;
  } finally {
    clearTimeout(timer);
  }
}
