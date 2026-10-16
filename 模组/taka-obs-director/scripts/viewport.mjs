const PAN_HOOK = 'canvasPan';
// OBS Utils registers its camera broadcast as the named function socketCanvas. Its build is not minified, so the
// name is stable while the body changes between releases (5.3.1 completes partial pans before sending).
const SENDER_NAME = 'socketCanvas';

function isSender(fn) {
  return typeof fn === 'function' && fn.name === SENDER_NAME;
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
