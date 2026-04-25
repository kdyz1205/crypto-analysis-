// frontend/js/services/ohlcv_cache.js
//
// IndexedDB-backed persistent cache for OHLCV bars.
//
// Why this exists (2026-04-23, user asked for "TradingView smooth"):
//   Historical bars are immutable (a 4h candle from 2024-07-01 16:00 UTC
//   is fixed forever). Re-downloading 2,936 bars on every F5 is what
//   makes the page feel slow. With IndexedDB we:
//     1. show the chart INSTANTLY from cache on page load,
//     2. fire a small delta fetch in the background for the newest bars,
//     3. append new bars + live-update the chart when delta arrives.
//
// Schema:
//   DB:    v2_ohlcv_cache (version 1)
//   store: bars  — keyPath '(symbol|interval)', value = {
//                    key, symbol, interval, bars: [{time, open, high, low, close, volume}],
//                    updatedAt: epoch_ms, firstTs, lastTs,
//                  }
// We keep one row per (symbol, interval) with ALL known bars — not a
// partitioned scheme — because 4h over 3y = 6,570 bars × ~80B/row = 0.5MB,
// which fits comfortably in memory and in one IDB write.

const DB_NAME = 'v2_ohlcv_cache';
const DB_VERSION = 1;
const STORE = 'bars';

let _dbPromise = null;

function openDB() {
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB unavailable'));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (ev) => {
      const db = ev.target.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'key' });
      }
    };
    req.onsuccess = (ev) => resolve(ev.target.result);
    req.onerror = (ev) => reject(ev.target.error);
  });
  return _dbPromise;
}

function _key(symbol, interval) {
  return `${String(symbol).toUpperCase()}|${String(interval)}`;
}

/**
 * Return cached bars for (symbol, interval) or null if nothing cached.
 * Does not check freshness — caller decides what to do with stale data.
 *
 * @returns {Promise<{bars: Array, updatedAt: number, firstTs: number, lastTs: number}|null>}
 */
export async function getCached(symbol, interval) {
  let db;
  try { db = await openDB(); } catch { return null; }
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, 'readonly');
      const req = tx.objectStore(STORE).get(_key(symbol, interval));
      req.onsuccess = () => {
        const row = req.result;
        if (!row || !Array.isArray(row.bars) || row.bars.length === 0) {
          resolve(null);
          return;
        }
        resolve({
          bars: row.bars,
          volume: Array.isArray(row.volume) ? row.volume : [],
          pricePrecision: row.pricePrecision ?? null,
          updatedAt: row.updatedAt || 0,
          firstTs: row.firstTs || row.bars[0]?.time || 0,
          lastTs: row.lastTs || row.bars[row.bars.length - 1]?.time || 0,
        });
      };
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

/**
 * Store bars + volume + metadata for (symbol, interval). Full replace —
 * caller should merge with existing if doing incremental updates.
 *
 * @param {string} symbol
 * @param {string} interval
 * @param {Array} bars — [{time, open, high, low, close}]
 * @param {Object} [extra] — optional {volume: [...], pricePrecision, ...}
 */
export async function storeBars(symbol, interval, bars, extra = {}) {
  if (!Array.isArray(bars) || bars.length === 0) return;
  let db;
  try { db = await openDB(); } catch { return; }
  const row = {
    key: _key(symbol, interval),
    symbol: String(symbol).toUpperCase(),
    interval: String(interval),
    bars,
    volume: Array.isArray(extra.volume) ? extra.volume : [],
    pricePrecision: extra.pricePrecision ?? null,
    updatedAt: Date.now(),
    firstTs: bars[0]?.time || 0,
    lastTs: bars[bars.length - 1]?.time || 0,
  };
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, 'readwrite');
      const req = tx.objectStore(STORE).put(row);
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}

/**
 * Integrity check: compare overlapping bars between cached and incoming.
 * Any OHLC mismatch on a CLOSED bar means the cache is corrupted — Bitget
 * never revises closed-bar OHLC. Returns count of mismatches found.
 *
 * Tolerance on the LAST cached bar — it's the live/forming bar and WILL
 * legitimately differ (open is fixed, close/high/low tick). We exclude it
 * from the check by only comparing bars whose timestamp < lastTs - 1
 * period.
 */
function _diffClosedBars(cached, incoming, periodSec) {
  const cMap = new Map(cached.map((b) => [b.time, b]));
  const mismatches = [];
  const cutoff = incoming.length > 0
    ? incoming[incoming.length - 1].time - periodSec  // exclude the last (live) bar
    : 0;
  for (const b of incoming) {
    if (b.time >= cutoff) continue;        // skip forming bar
    const c = cMap.get(b.time);
    if (!c) continue;                      // bar not in cache (fine)
    const eps = 1e-6;
    if (
      Math.abs(c.open  - b.open)  > eps ||
      Math.abs(c.high  - b.high)  > eps ||
      Math.abs(c.low   - b.low)   > eps ||
      Math.abs(c.close - b.close) > eps
    ) {
      mismatches.push({ time: b.time, cached: c, incoming: b });
    }
  }
  return mismatches;
}

const TF_SECONDS_LOCAL = {
  '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
  '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200,
  '1d': 86400, '3d': 259200, '1w': 604800,
};

/**
 * Merge new bars + volume into the existing cache. Deduplicates by `time`,
 * keeps the incoming version (which is newer / authoritative).
 *
 * 2026-04-23 integrity check: before merging, verify closed bars in the
 * overlap region match. If server says closed bar OHLC differs from cache,
 * the cache is corrupt — wipe it and restart from scratch with server
 * data. Also logs the discrepancy for debugging.
 *
 * Returns {bars: [...], volume: [...], integrityOk: bool, mismatchCount}.
 */
export async function mergeBars(symbol, interval, newBars, newVolume = null, extra = {}) {
  if (!Array.isArray(newBars) || newBars.length === 0) {
    const cached = await getCached(symbol, interval);
    return { bars: cached?.bars || [], volume: cached?.volume || [],
             integrityOk: true, mismatchCount: 0 };
  }
  const cached = await getCached(symbol, interval);
  const existing = cached?.bars || [];
  const existingVol = cached?.volume || [];

  // ── Integrity check ──────────────────────────────────────────
  const periodSec = TF_SECONDS_LOCAL[interval] || 3600;
  const mismatches = existing.length > 0
    ? _diffClosedBars(existing, newBars, periodSec)
    : [];
  if (mismatches.length > 0) {
    console.warn(
      `[ohlcv_cache] INTEGRITY VIOLATION ${symbol} ${interval}: `,
      `${mismatches.length} closed bars disagree with server. `,
      `Wiping cache and resyncing.`,
      mismatches.slice(0, 3),
    );
    // Full rebuild from server data — do NOT merge with existing.
    await storeBars(symbol, interval, newBars, {
      volume: Array.isArray(newVolume) ? newVolume : [],
      pricePrecision: extra.pricePrecision ?? null,
    });
    return { bars: newBars, volume: newVolume || [],
             integrityOk: false, mismatchCount: mismatches.length };
  }

  // Integrity OK → normal merge by timestamp, server wins on overlap.
  const barMap = new Map();
  for (const b of existing) {
    if (b && typeof b.time === 'number') barMap.set(b.time, b);
  }
  for (const b of newBars) {
    if (b && typeof b.time === 'number') barMap.set(b.time, b);
  }
  const mergedBars = [...barMap.values()].sort((a, b) => a.time - b.time);

  let mergedVol = existingVol;
  if (Array.isArray(newVolume) && newVolume.length > 0) {
    const vmap = new Map();
    for (const v of existingVol) {
      if (v && typeof v.time === 'number') vmap.set(v.time, v);
    }
    for (const v of newVolume) {
      if (v && typeof v.time === 'number') vmap.set(v.time, v);
    }
    mergedVol = [...vmap.values()].sort((a, b) => a.time - b.time);
  }

  await storeBars(symbol, interval, mergedBars, {
    volume: mergedVol,
    pricePrecision: extra.pricePrecision ?? cached?.pricePrecision ?? null,
  });
  return { bars: mergedBars, volume: mergedVol,
           integrityOk: true, mismatchCount: 0 };
}

/** Nuke the entire cache. Called from a dev console when debugging. */
export async function clearCache() {
  let db;
  try { db = await openDB(); } catch { return; }
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}

/** Best-effort bar-period→seconds lookup used to gauge "recent enough". */
const TF_SECONDS = {
  '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
  '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200,
  '1d': 86400, '3d': 259200, '1w': 604800,
};

/** Is the cache's last bar within one TF period of now? */
export function isFreshEnough(cachedMeta, interval) {
  if (!cachedMeta || !cachedMeta.lastTs) return false;
  const period = TF_SECONDS[interval] || 3600;
  const nowSec = Date.now() / 1000;
  return (nowSec - cachedMeta.lastTs) < period * 1.5;
}

// Expose a debug hook for the console.
if (typeof window !== 'undefined') {
  window.__ohlcvCache = { getCached, storeBars, mergeBars, clearCache, isFreshEnough };
}
