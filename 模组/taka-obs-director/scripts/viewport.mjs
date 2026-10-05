const PAN_HOOK = 'canvasPan';
// Audited OBS Utils socket-Bdx6f0PS.js sender; require its complete body.
const SENDER_SOURCE = `function socketCanvas(_canvas, position) {
  const mode = getSetting("cameraTrackingMode") ?? "smooth";
  if (mode === "dragRelease") dragReleaseEmit(position);
  else if (mode === "raw") throttledEmit(position);
  else smoothedEmit(position);
}`.replace(/\s+/g, '');

function isSender(fn) {
  return typeof fn === 'function' && fn.name === 'socketCanvas'
    && Function.prototype.toString.call(fn).replace(/\s+/g, '') === SENDER_SOURCE;
}

/** Pause only the active recorder's recurring pan broadcast; return its cleanup. */
export function suspendRecorderViewportBroadcast({ game, hooks, isActive } = {}) {
  const noop = () => {};
  const module = game?.modules?.get?.('obs-utils');
  if (module?.active !== true
    || typeof isActive !== 'function' || typeof hooks?.on !== 'function'
    || typeof hooks?.off !== 'function') return noop;

  try {
    if (isActive() !== true) return noop;
  } catch {
    return noop;
  }

  const entries = hooks.events?.[PAN_HOOK];
  if (!Array.isArray(entries)) return noop;
  const candidates = entries.filter((entry) => isSender(entry?.fn));
  if (candidates.length !== 1) return noop;
  const entry = candidates[0];
  if (entry.hook !== PAN_HOOK || entry.once !== false) return noop;

  const sender = entry.fn;
  hooks.off(PAN_HOOK, sender);
  let restored = false;
  return () => {
    if (restored) return;
    const current = hooks.events?.[PAN_HOOK];
    if (!Array.isArray(current)) return;
    // A later registration may have already restored or replaced this sender.
    if (!current.some((listener) => listener?.fn === sender || isSender(listener?.fn))) {
      hooks.on(PAN_HOOK, sender, { once: entry.once });
    }
    restored = true;
  };
}
