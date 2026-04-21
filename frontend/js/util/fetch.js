// frontend/js/util/fetch.js — fetch with cache + timeout + JSON parse

const DEFAULT_TIMEOUT = 45000;
const CACHE_TTL = 30_000; // 30s stale-while-revalidate
const _cache = new Map();  // url → { data, ts, promise }

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

/**
 * Fetch JSON with built-in caching.
 * - GET requests are cached for CACHE_TTL ms
 * - Returns cached data immediately if fresh
 * - Deduplicates in-flight requests to same URL
 * - POST/DELETE/etc always bypass cache
 */
export async function fetchJson(url, { method = 'GET', body, timeout = DEFAULT_TIMEOUT, signal, noCache = false } = {}) {
  const isGet = method === 'GET' && !body;

  // Return cached if fresh
  if (isGet && !noCache) {
    const cached = _cache.get(url);
    if (cached && (Date.now() - cached.ts < CACHE_TTL)) {
      return cached.data;
    }
    // Deduplicate in-flight
    if (cached?.promise) return cached.promise;
  }

  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeout);
  const combined = signal ? combineSignals([signal, ac.signal]) : ac.signal;

  const promise = (async () => {
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
      catch(e) { data = text; }
      if (!res.ok) {
        // Surface FastAPI's `{"detail": "..."}` message so the user sees
        // the ACTUAL rejection reason in the UI, not just "HTTP 400".
        // 2026-04-21: user's place-line-order got rejected (line too far
        // from mark, or missing drawing, etc.) but modal only showed
        // "创建失败: HTTP 400" with no clue why.
        let detail = '';
        if (data && typeof data === 'object') {
          detail = data.detail || data.message || data.reason || '';
          if (typeof detail !== 'string') detail = JSON.stringify(detail);
        } else if (typeof data === 'string') {
          detail = data;
        }
        const msg = detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status}`;
        throw new FetchError(msg, res.status, data);
      }

      // Cache GET results
      if (isGet) {
        _cache.set(url, { data, ts: Date.now(), promise: null });
      }
      return data;
    } finally {
      clearTimeout(timer);
      // Clear in-flight marker
      const c = _cache.get(url);
      if (c?.promise === promise) c.promise = null;
    }
  })();

  // Mark in-flight for dedup
  if (isGet && !noCache) {
    const existing = _cache.get(url);
    if (existing) { existing.promise = promise; }
    else { _cache.set(url, { data: null, ts: 0, promise }); }
  }

  return promise;
}

/** Get cached data without fetching. Returns null if not cached. */
export function getCached(url) {
  const c = _cache.get(url);
  return c?.data || null;
}

/** Invalidate cache for a URL (after mutations). */
export function invalidateCache(url) {
  _cache.delete(url);
}

/** Invalidate all caches matching a prefix. */
export function invalidateCachePrefix(prefix) {
  for (const key of _cache.keys()) {
    if (key.startsWith(prefix)) _cache.delete(key);
  }
}
