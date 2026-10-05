const policies = new WeakMap();

/** Native preferences are applied on each recorder login, including a new browser. */
export function applyRecorderPreferences({ game, isActive = () => false } = {}) {
  const userId = game?.user?.id;
  const sameUser = () => !!userId && game.user?.id === userId && game.user.isGM === false;
  const active = () => sameUser() && isActive() === true;
  const preferences = [], changed = [];
  let disposed = false, closing = null;
  const previous = game && policies.get(game);
  const waitFor = previous?.dispose();
  const warn = key => console.warn(`Taka OBS Director could not update recorder preference: ${key}`);
  function readPreference(key, read) {
    try { return read(); } catch { warn(key); return undefined; }
  }
  function preference(key, read, write, value) {
    const previous = readPreference(key, read);
    if (previous !== undefined && previous !== value) preferences.push({ key, read, write, value, previous });
  }
  function registered(namespace, key, value, scope) {
    if (game.settings?.settings?.get?.(`${namespace}.${key}`)?.scope !== scope) return;
    preference(`${namespace}.${key}`, () => game.settings.get(namespace, key), next => game.settings.set(namespace, key, next), value);
  }
  function plan() {
    if (!active() || disposed) return;
    const av = game.webrtc?.settings;
    if (typeof av?.get === 'function' && typeof av?.set === 'function') {
      preference('core.rtcClientSettings.disableVideo', () => av.get('client', 'disableVideo'), value => av.set('client', 'disableVideo', value), true);
    }
    const fps = readPreference('core.maxFPS', () => game.settings?.get?.('core', 'maxFPS'));
    if (Number.isFinite(fps) && fps > 30) registered('core', 'maxFPS', 30, 'client');
    if (game.modules?.get?.('pf2e-hud')?.active === true) {
      for (const [key, value] of [
        ['persistent.display', 'disabled'], ['token.activation', 'disabled'], ['tracker.enabled', false],
        ['tooltip.distance', 'never'], ['tooltip.status', false],
      ]) registered('pf2e-hud', key, value, 'user');
    }
  }
  async function apply() {
    plan();
    // The native AV setter updates memory immediately and persists after debounce.
    for (const item of preferences) {
      if (disposed || !active()) break;
      try {
        await item.write(item.value);
        changed.push(item);
      } catch { warn(item.key); }
    }
  }
  const ready = (waitFor ? waitFor.then(apply) : apply()).catch(() => { warn('startup'); });
  function dispose() {
    if (closing) return closing;
    disposed = true;
    closing = ready.then(async () => {
      for (const item of changed.reverse()) {
        if (!sameUser()) break;
        try {
          // A manual edit made after startup belongs to the user.
          if (item.read() === item.value) await item.write(item.previous);
        } catch { warn(item.key); }
      }
      changed.length = 0;
      if (game && policies.get(game) === policy) policies.delete(game);
    });
    return closing;
  }
  const policy = { ready, dispose };
  if (game) policies.set(game, policy);
  return policy;
}
