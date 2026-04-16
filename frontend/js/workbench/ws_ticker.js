// frontend/js/workbench/ws_ticker.js
//
// Direct Bitget public WebSocket ticker stream. Used to push tick-level
// price updates to the chart without going through our own server.
//
// Stream: wss://ws.bitget.com/v2/ws/public
// Subscribe: {"op":"subscribe","args":[{"instType":"USDT-FUTURES","channel":"ticker","instId":"BTCUSDT"}]}
// Push frame: {"action":"snapshot","arg":{...},"data":[{"lastPr":"...","markPrice":"...","ts":"..."}]}

const WS_URL = 'wss://ws.bitget.com/v2/ws/public';

let _ws = null;
let _currentSymbol = null;
let _onTick = null;
let _reconnectTimer = null;
let _pingTimer = null;

export function startTickerWS(initialSymbol, onTick) {
  _onTick = onTick;
  _currentSymbol = (initialSymbol || '').toUpperCase();
  connect();
}

export function setTickerSymbol(newSymbol) {
  const sym = (newSymbol || '').toUpperCase();
  if (!sym || sym === _currentSymbol) return;
  const old = _currentSymbol;
  _currentSymbol = sym;
  if (_ws?.readyState === WebSocket.OPEN) {
    try { _ws.send(unsubscribeMsg(old)); } catch {}
    try { _ws.send(subscribeMsg(sym)); } catch {}
  }
}

export function stopTickerWS() {
  if (_pingTimer) { clearInterval(_pingTimer); _pingTimer = null; }
  if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
  if (_ws) { try { _ws.close(); } catch {} }
  _ws = null;
}

// ─────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────
function connect() {
  try {
    _ws = new WebSocket(WS_URL);
  } catch (err) {
    console.warn('[ws_ticker] WebSocket ctor failed', err);
    scheduleReconnect();
    return;
  }
  _ws.addEventListener('open', onOpen);
  _ws.addEventListener('message', onMessage);
  _ws.addEventListener('close', onClose);
  _ws.addEventListener('error', onError);
}

function onOpen() {
  console.log('[ws_ticker] connected');
  if (_currentSymbol) {
    try { _ws.send(subscribeMsg(_currentSymbol)); } catch {}
  }
  // Bitget requires pings every ~30s to keep the connection alive
  if (_pingTimer) clearInterval(_pingTimer);
  _pingTimer = setInterval(() => {
    if (_ws?.readyState === WebSocket.OPEN) {
      try { _ws.send('ping'); } catch {}
    }
  }, 20000);
}

function onMessage(event) {
  // Bitget sends 'pong' as plain text in response to 'ping'
  if (typeof event.data === 'string' && event.data === 'pong') return;
  let parsed;
  try { parsed = JSON.parse(event.data); } catch { return; }
  if (parsed?.event === 'subscribe' || parsed?.event === 'unsubscribe') return;
  if (parsed?.code != null && parsed.code !== 0) {
    console.warn('[ws_ticker] bitget error', parsed);
    return;
  }

  const data = parsed?.data;
  if (!Array.isArray(data) || data.length === 0) return;
  const row = data[0];
  const instId = String(parsed?.arg?.instId || row.instId || '').toUpperCase();
  if (instId !== _currentSymbol) return;   // stale update from old subscription

  const tick = {
    symbol: instId,
    lastPrice: parseFloat(row.lastPr ?? row.last ?? '0'),
    markPrice: parseFloat(row.markPrice ?? row.lastPr ?? '0'),
    bidPrice: parseFloat(row.bidPr ?? '0'),
    askPrice: parseFloat(row.askPr ?? '0'),
    ts: Number(row.ts ?? parsed?.ts ?? Date.now()),
  };
  if (!isFinite(tick.lastPrice) || tick.lastPrice <= 0) return;
  try { _onTick && _onTick(tick); } catch (err) { console.warn('[ws_ticker] onTick threw', err); }
}

function onClose() {
  console.warn('[ws_ticker] closed, reconnecting');
  if (_pingTimer) { clearInterval(_pingTimer); _pingTimer = null; }
  scheduleReconnect();
}

function onError(err) {
  console.warn('[ws_ticker] error', err);
  // close handler will schedule reconnect
}

function scheduleReconnect() {
  if (_reconnectTimer) return;
  _reconnectTimer = setTimeout(() => {
    _reconnectTimer = null;
    connect();
  }, 2000);
}

function subscribeMsg(symbol) {
  return JSON.stringify({
    op: 'subscribe',
    args: [{ instType: 'USDT-FUTURES', channel: 'ticker', instId: symbol }],
  });
}
function unsubscribeMsg(symbol) {
  return JSON.stringify({
    op: 'unsubscribe',
    args: [{ instType: 'USDT-FUTURES', channel: 'ticker', instId: symbol }],
  });
}
