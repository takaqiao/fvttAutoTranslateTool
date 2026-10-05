/** Pure layout and public identity decisions; never collects actor details. */
export function resolveMode({ manualMode = 'auto', publicImages = [], combatStarted = false } = {}) {
  if (['explore', 'combat', 'story'].includes(manualMode)) return manualMode;
  if ((publicImages?.size ?? publicImages?.length ?? 0) > 0) return 'story';
  return combatStarted === true ? 'combat' : 'explore';
}

/** Missing values stay missing. Capacity zero is not a displayable pool. */
export function normalizeCounter(value, max) {
  return Number.isFinite(value) && value >= 0 && Number.isFinite(max) && max > 0
    ? { value, max }
    : null;
}

/** seatOrder contains configured User IDs, not actor IDs or cached assignments. */
export function selectCastUsers({ users = [], recorderId, seatOrder = [], connectedUserIds = [] } = {}) {
  const current = new Map(Array.from(users).map((user) => [user.id, user]));
  const connected = new Set(connectedUserIds);
  // Foundry's native Array setting input can save all IDs in one comma-separated entry.
  const ids = Array.isArray(seatOrder) ? seatOrder.flatMap(value => typeof value === 'string' ? value.split(',') : []).map(id => id.trim()).filter(Boolean) : [];
  const ordered = [...new Set(ids)].map((id) => current.get(id)).filter((user) => user && user.id !== recorderId);
  const gm = ordered.find((user) => user.isGM === true);
  const players = ordered.filter((user) => user.isGM === false).slice(0, 5);
  const cast = players.filter((user, index) => index < 4 || user.active === true || connected.has(user.id));
  return gm ? [gm, ...cast] : cast;
}

/**
 * Only a bound, observed PC yields an actor reference. Everything else uses
 * exclusively the combatant's public name/img, without touching its actor.
 */
export function selectFocus({ combatant, allowedActorIds = [], viewer } = {}) {
  if (!combatant || combatant.hidden === true || combatant.visible !== true || combatant.token?.hidden === true) return null;
  const publicFocus = {
    actor: null,
    publicOnly: true,
    name: typeof combatant.name === 'string' ? combatant.name : '',
    portrait: typeof combatant.img === 'string' ? combatant.img : null,
  };
  if (!new Set(allowedActorIds).has(combatant.actorId) || !viewer || viewer.isGM === true) return publicFocus;
  try {
    const actor = combatant.actor;
    if (actor?.id !== combatant.actorId || actor.testUserPermission?.(viewer, 'OBSERVER') !== true || actor.type !== 'character') return publicFocus;
    return { ...publicFocus, actor, publicOnly: false };
  } catch {
    return publicFocus;
  }
}

export function sceneRect({ width, height, mode = 'explore' } = {}) {
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) return null;
  const left = width * 0.056;
  const top = 0;
  const right = width * (mode === 'combat' ? 0.25 : 0.056);
  return { left, top, width: width - left - right, height: height * 0.745 };
}
