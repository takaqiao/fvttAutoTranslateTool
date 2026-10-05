const centered = () => ({ x: 0, y: 0, scale: 1 });
const number = (value, fallback, min, max) => typeof value === 'number' && Number.isFinite(value)
  ? Math.min(max, Math.max(min, value)) : fallback;

function plain(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function ownValue(object, key) {
  const descriptor = Object.getOwnPropertyDescriptor(object, key);
  if (!descriptor) return undefined;
  if (!Object.hasOwn(descriptor, 'value')) throw new TypeError('Portrait calibration must contain data fields.');
  return descriptor.value;
}

function entry(mapping, key) {
  if (!plain(mapping) || typeof key !== 'string' || !key) return undefined;
  return ownValue(mapping, key);
}

function matches(candidate, source) {
  return plain(candidate) && ownValue(candidate, 'source') === source;
}

/** Positions are percentages of the portrait slot; scale changes the whole image. */
export function normalizePortraitLayout(candidate, { source } = {}) {
  try {
    if (typeof source !== 'string' || !source || !matches(candidate, source)) return centered();
    return {
      x: number(ownValue(candidate, 'x'), 0, -75, 75),
      y: number(ownValue(candidate, 'y'), 0, -75, 75),
      scale: number(ownValue(candidate, 'scale'), 1, 0.25, 3),
    };
  } catch { return centered(); }
}

export function resolvePortraitLayout({ worldId, actorId, source, overrides, defaults } = {}) {
  try {
    if (typeof source !== 'string' || !source || typeof actorId !== 'string' || !actorId) return centered();
    const override = entry(overrides, actorId);
    if (matches(override, source)) return normalizePortraitLayout(override, { source });
    const candidate = entry(entry(defaults, worldId), actorId);
    return normalizePortraitLayout(candidate, { source });
  } catch { return centered(); }
}
