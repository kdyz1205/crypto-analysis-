// frontend/sw.js — Service Worker for Trading OS
//
// 2026-04-23: user wants TradingView-level smoothness. One of TV's tricks
// is a Service Worker that intercepts every fetch and short-circuits to
// local cache when possible. This SW does the same thing for:
//   - Static assets (JS / CSS / HTML)           → cache-first
//   - /api/symbols, /api/symbols/extended       → stale-while-revalidate
//   - /api/ohlcv  (historical OHLCV)            → stale-while-revalidate
// All other /api/* requests go to network directly (no caching), so
// order placement, accounts, conditionals are NEVER served from cache.
//
// KILLSWITCH: if anything breaks, open DevTools → Application →
// Service Workers → Unregister. Or set `localStorage.disableSW = '1'`
// BEFORE reload and the sw.js registration in main.js will skip.

// 2026-04-25 ROOT-CAUSE FIX: SW was using cache-first for JS modules.
// That meant every code change required user Ctrl+Shift+R + manual SW
// unregister to take effect — chart.js fixes shipped on the server were
// never reaching the browser. Now JS / CSS are stale-while-revalidate
// so user sees yesterday's code once (free), and the next reload picks
// up today's fix automatically. SW_VERSION bump auto-evicts the old
// cache below.
const SW_VERSION = 'v3-2026-04-25-jschart-fix';
const STATIC_CACHE = `tos-static-${SW_VERSION}`;
const OHLCV_CACHE = `tos-ohlcv-${SW_VERSION}`;

// URLs that MUST go to network every time. Any match = bypass SW.
// Execution / account / orders / drawings / conditionals / stream MUST
// NEVER be served from cache — they are live state, stale = wrong.
const NEVER_CACHE_PATTERNS = [
  /\/api\/account/,
  /\/api\/live-execution\//,
  /\/api\/conditionals/,
  /\/api\/drawings/,
  /\/api\/stream/,
  /\/api\/agent\//,
  /\/api\/chat\//,
  /\/api\/place/,
  /\/api\/cancel/,
  /\/api\/modify/,
  /\/api\/order/,
  /\/api\/market\/structure-summary/,   // SWR'd server-side already
  /\/api\/strategy\/snapshot/,          // SWR'd server-side already
];

// Pre-cache the app shell on install so /v2 opens offline.
const SHELL_URLS = [
  '/v2',
  '/v2.html',
  '/v2.css',
  '/v2-theme-enhanced.css',
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => cache.addAll(SHELL_URLS).catch(() => {}))
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((k) => !k.endsWith(SW_VERSION)).map((k) => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;         // POST/PATCH/DELETE never cached
  const url = new URL(req.url);

  // Same-origin only (don't touch external resources)
  if (url.origin !== self.location.origin) return;

  // Hard bypass for live/state endpoints.
  if (NEVER_CACHE_PATTERNS.some((re) => re.test(url.pathname))) return;

  // OHLCV → stale-while-revalidate (history is immutable; bar boundaries
  // refresh cache). Matches our IndexedDB logic on the frontend.
  if (url.pathname === '/api/ohlcv' || url.pathname.startsWith('/api/ohlcv/')) {
    event.respondWith(staleWhileRevalidate(req, OHLCV_CACHE));
    return;
  }

  // Symbols lists → stale-while-revalidate (changes at most daily).
  if (url.pathname === '/api/symbols' ||
      url.pathname === '/api/symbols/extended' ||
      url.pathname === '/api/symbols/categorized') {
    event.respondWith(staleWhileRevalidate(req, STATIC_CACHE));
    return;
  }

  // 2026-04-25 ROOT FIX: JS / CSS / HTML use stale-while-revalidate, NOT
  // cache-first. The original cache-first served the user yesterday's
  // chart.js forever (until they manually killed the SW), making every
  // code fix invisible. SWR shows the cached file once for instant load,
  // then the BG fetch updates the cache so the NEXT reload picks up the
  // newest code automatically. /v2 HTML included so cache-buster bumps
  // (?v=...) take effect on the very next reload.
  if (url.pathname === '/v2' ||
      url.pathname === '/v2.html' ||
      url.pathname.endsWith('.js') ||
      url.pathname.endsWith('.css') ||
      url.pathname.endsWith('.mjs')) {
    event.respondWith(staleWhileRevalidate(req, STATIC_CACHE));
    return;
  }

  // Everything else same-origin (favicon, fonts, images): static assets
  // that don't ship code → cache-first is fine.
  event.respondWith(cacheFirst(req, STATIC_CACHE));
});

async function cacheFirst(req, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(req);
  if (hit) return hit;
  try {
    const resp = await fetch(req);
    if (resp.ok) cache.put(req, resp.clone()).catch(() => {});
    return resp;
  } catch (err) {
    // Offline fallback for /v2 — return shell if cached.
    const shell = await cache.match('/v2') || await cache.match('/v2.html');
    if (shell) return shell;
    throw err;
  }
}

async function staleWhileRevalidate(req, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(req);
  // Fire the fetch in parallel — update cache when it finishes.
  const fetchPromise = fetch(req).then((resp) => {
    if (resp.ok) cache.put(req, resp.clone()).catch(() => {});
    return resp;
  }).catch((err) => {
    // Network failure: we still have cached response (if any) to fall back to.
    if (cached) return cached;
    throw err;
  });
  // Return cached immediately if we have it; otherwise wait for network.
  return cached || fetchPromise;
}

// 2026-04-25: respond to version pings from the page (used by the
// inline self-recovery script in v2.html). Old SWs don't have this
// handler → no reply → page assumes "stale" → auto-unregisters.
// New SW (this version) replies with SW_VERSION → page is satisfied.
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'ping-version') {
    if (event.ports && event.ports[0]) {
      event.ports[0].postMessage({ version: SW_VERSION });
    }
    return;
  }
  if (event.data === 'killswitch') {
    caches.keys().then((keys) => Promise.all(keys.map((k) => caches.delete(k))))
      .then(() => self.registration.unregister())
      .then(() => event.source?.postMessage({ type: 'sw-killed' }));
  }
});
