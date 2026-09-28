const own = (value, key) => Object.hasOwn(value, key);
const isRecord = value => {
  if (value === null || typeof value !== 'object') return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
};
const BACKGROUND_FIELDS = ['src', 'tint', 'alphaThreshold'];
const TEXTURE_FIELDS = ['anchorX', 'anchorY', 'fit', 'scaleX', 'scaleY', 'rotation'];
const LEGACY_ROOT_FIELDS = ['img', 'background', 'backgroundColor', 'foreground', 'foregroundElevation'];
const LEGACY_DOTTED_FIELDS = [
  ...BACKGROUND_FIELDS.map(key => `background.${key}`),
  ...TEXTURE_FIELDS.map(key => `background.${key}`),
  'background.offsetX', 'background.offsetY', 'fog.overlay'
];

function mapChanged(data, migrate) {
  if (!Array.isArray(data)) return migrate(data);
  let result = data;
  for (let index = 0; index < data.length; index++) {
    const next = migrate(data[index]);
    if (next === data[index]) continue;
    if (result === data) result = data.slice();
    result[index] = next;
  }
  return result;
}

/** Numeric pre-v14 types are styles; a caller-specified style always wins. */
export function migrateLegacyChatData(data) {
  return mapChanged(data, item => {
    if (!isRecord(item) || !own(item, 'type') || typeof item.type !== 'number') return item;
    const result = {...item};
    if (!own(item, 'style')) result.style = item.type;
    delete result.type;
    return result;
  });
}

function hasLegacySceneFields(data) {
  return isRecord(data) && (LEGACY_ROOT_FIELDS.some(key => own(data, key))
    || LEGACY_DOTTED_FIELDS.some(key => own(data, key))
    || (isRecord(data.fog) && own(data.fog, 'overlay')));
}

function collectSceneChanges(data) {
  const result = {...data};
  const level = {};
  const shifts = {};
  const set = (part, key, value) => { (level[part] ??= {})[key] = value; };
  const background = isRecord(data.background) ? data.background : null;
  if (background) {
    for (const key of BACKGROUND_FIELDS) if (own(background, key)) set('background', key, background[key]);
    for (const key of TEXTURE_FIELDS) if (own(background, key)) set('textures', key, background[key]);
    if (own(background, 'offsetX')) shifts.shiftX = background.offsetX;
    if (own(background, 'offsetY')) shifts.shiftY = background.offsetY;
  }
  if (own(data, 'img')) set('background', 'src', data.img);
  if (own(data, 'backgroundColor')) set('background', 'color', data.backgroundColor);
  if (own(data, 'foreground')) set('foreground', 'src', isRecord(data.foreground) ? data.foreground.src : data.foreground);
  if (own(data, 'foregroundElevation')) set('elevation', 'top', data.foregroundElevation);
  if (isRecord(data.fog) && own(data.fog, 'overlay')) {
    set('fog', 'src', data.fog.overlay);
    result.fog = {...data.fog};
    delete result.fog.overlay;
  }
  for (const key of BACKGROUND_FIELDS) {
    if (own(data, `background.${key}`)) set('background', key, data[`background.${key}`]);
  }
  for (const key of TEXTURE_FIELDS) {
    if (own(data, `background.${key}`)) set('textures', key, data[`background.${key}`]);
  }
  if (own(data, 'background.offsetX')) shifts.shiftX = data['background.offsetX'];
  if (own(data, 'background.offsetY')) shifts.shiftY = data['background.offsetY'];
  if (own(data, 'fog.overlay')) set('fog', 'src', data['fog.overlay']);
  for (const key of LEGACY_ROOT_FIELDS) delete result[key];
  for (const key of LEGACY_DOTTED_FIELDS) delete result[key];
  for (const [key, value] of Object.entries(shifts)) if (!own(data, key)) result[key] = value;
  return {result, level};
}

/** Fill only absent fields, keeping existing Level identities and unrelated references. */
function fillMissing(explicit, fallback, root = explicit, prefix = '') {
  let result = explicit;
  for (const [key, value] of Object.entries(fallback)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (prefix && own(root, path)) continue;
    let next;
    if (!own(explicit, key)) {
      next = isRecord(value) ? fillMissing({}, value, root, path) : value;
      if (isRecord(next) && !Object.keys(next).length) continue;
    }
    else if (isRecord(explicit[key]) && isRecord(value)) next = fillMissing(explicit[key], value, root, path);
    else continue;
    if (own(explicit, key) && next === explicit[key]) continue;
    if (result === explicit) result = {...explicit};
    result[key] = next;
  }
  return result;
}

function invalidExplicitLevels(data) {
  return own(data, 'levels') && !Array.isArray(data.levels);
}

export function migrateLegacySceneCreateData(data) {
  return mapChanged(data, item => {
    if (!hasLegacySceneFields(item) || invalidExplicitLevels(item)) return item;
    const {result, level} = collectSceneChanges(item);
    if (!Object.keys(level).length) return result;
    const existing = item.levels ?? [];
    let index = existing.findIndex(entry => isRecord(entry) && entry._id === item.initialLevel);
    if (index < 0) index = existing.findIndex(isRecord);
    if (index >= 0) {
      const merged = fillMissing(existing[index], level);
      if (merged !== existing[index]) {
        result.levels = existing.slice();
        result.levels[index] = merged;
      }
    } else {
      const created = {_id: 'defaultLevel0000', name: 'Default', ...level};
      result.levels = [...existing, created];
      if (!own(item, 'initialLevel')) result.initialLevel = created._id;
    }
    return result;
  });
}

export function migrateLegacySceneUpdateData(scene, data) {
  if (!hasLegacySceneFields(data) || invalidExplicitLevels(data)) return data;
  const {result, level} = collectSceneChanges(data);
  if (!Object.keys(level).length) return result;
  const id = scene.firstLevel?.id ?? scene._source?.levels?.[0]?._id ?? 'defaultLevel0000';
  const existing = data.levels ?? [];
  const index = existing.findIndex(entry => isRecord(entry) && entry._id === id);
  if (index < 0) result.levels = [...existing, {_id: id, ...level}];
  else {
    const merged = fillMissing(existing[index], level);
    if (merged !== existing[index]) {
      result.levels = existing.slice();
      result.levels[index] = merged;
    }
  }
  return result;
}

function wrapperTargets(base, implementation, basePath, implementationPath, method, prototype = false) {
  const baseOwner = prototype ? base?.prototype : base;
  const implementationOwner = prototype ? implementation?.prototype : implementation;
  const suffix = `${prototype ? '.prototype' : ''}.${method}`;
  const targets = [];
  if (typeof baseOwner?.[method] === 'function') targets.push(basePath + suffix);
  const inherited = baseOwner && implementationOwner && baseOwner.isPrototypeOf(implementationOwner);
  if (implementationOwner && implementationOwner !== baseOwner && typeof implementationOwner[method] === 'function'
    && (!inherited || own(implementationOwner, method))) targets.push(implementationPath + suffix);
  return targets;
}

/** Call during setup, after system classes exist. The caller owns version gating and libWrapper registration. */
export function registerLegacyCompat({moduleId = 'av-v14-hotfix', g = globalThis, registerWrapper, report} = {}) {
  if (typeof registerWrapper !== 'function') throw new TypeError('registerLegacyCompat requires registerWrapper');
  const wrappers = [];
  const adventureSettings = [];
  const plans = [];
  // Native Document.create delegates to implementation.createDocuments before validation.
  // Keep the single-create receiver untouched for consumers such as CotCT's Harrow display.
  for (const target of wrapperTargets(g.ChatMessage, g.CONFIG?.ChatMessage?.documentClass,
    'ChatMessage', 'CONFIG.ChatMessage.documentClass', 'createDocuments')) plans.push([target, migrateLegacyChatData]);
  for (const method of ['create', 'createDocuments']) {
    for (const target of wrapperTargets(g.Scene, g.CONFIG?.Scene?.documentClass,
      'Scene', 'CONFIG.Scene.documentClass', method)) plans.push([target, migrateLegacySceneCreateData]);
  }
  for (const target of wrapperTargets(g.Scene, g.CONFIG?.Scene?.documentClass,
    'Scene', 'CONFIG.Scene.documentClass', 'update', true)) plans.push([target, null]);
  for (const [target, migrate] of plans) {
    registerWrapper(target, function(wrapped, data, ...args) {
      const next = migrate ? migrate(data) : migrateLegacySceneUpdateData(this, data);
      return wrapped.call(this, next, ...args);
    }, 'WRAPPER');
    wrappers.push(target);
  }

  // The current Season of Ghosts importer is fixed; only this audited older importer still reads every pack.
  const importer = g.game?.modules?.get('sf2e-murder-in-metal-city');
  if (importer?.active && importer.version === '13.2.0') {
    const settings = g.game.settings;
    for (const pack of g.game.packs ?? []) {
      const namespace = pack.metadata?.packageName;
      if (pack.metadata?.type !== 'Adventure' || !namespace || settings.settings.has(`${namespace}.autoOpenAdventures`)) continue;
      settings.register(namespace, 'autoOpenAdventures', {scope: 'world', config: false, type: Boolean, default: false});
      adventureSettings.push(`${namespace}.autoOpenAdventures`);
    }
  }
  const result = {wrappers, adventureSettings};
  report?.({moduleId, feature: 'legacy-compat', status: 'installed', ...result});
  return result;
}
