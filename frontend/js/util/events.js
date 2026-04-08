// frontend/js/util/events.js — UI-local pub/sub (not backend event bus)

const handlers = new Map();

export function subscribe(event, fn) {
  if (!handlers.has(event)) handlers.set(event, new Set());
  handlers.get(event).add(fn);
  return () => handlers.get(event)?.delete(fn);
}

export function publish(event, data) {
  const set = handlers.get(event);
  if (!set) return;
  for (const fn of set) {
    try { fn(data); }
    catch (err) { console.error(`[events] ${event} handler error:`, err); }
  }
}

export function unsubscribeAll(event) {
  handlers.delete(event);
}
