const supported = {
  version: 4, killAnim: false, animLevel: false, override: true,
  sourceToken: {enable: false}, targetToken: {enable: false}, levels3d: {type: ''},
  macro: {enable: false}, animType: 'static',
  options: {staticType: 'source', menuType: 'liquid', variant: '01', enableCustom: false,
    repeat: 3, delay: null, scale: null, opacity: null, unbindAlpha: false,
    unbindVisibility: false, persistent: false},
  animation: 'splash', color: 'blue', audio: {a01: {enable: false}},
  preview: false, explosions: {enable: false}
};

function matches(value, expected) {
  if (value === expected) return true;
  if (!value || !expected || typeof value !== 'object' || typeof expected !== 'object') return false;
  const keys = Object.keys(expected);
  return Object.keys(value).length === keys.length
    && keys.every(key => Object.hasOwn(value, key) && matches(value[key], expected[key]));
}

/** Build an AA8 entry seed. Validate with AA's DataModels.validateEntry before saving. */
export function migrateQuenchFlags(flags) {
  // This maintenance operation only covers the audited source splash. Unknown extras must not be discarded.
  if (!matches(flags, supported)) throw new TypeError('Unsupported legacy Quench animation configuration');
  return {
    version: 5, isEnabled: true, isCustomized: true, fromAmmo: false, menu: 'ontoken',
    primary: {
      video: {dbSection: 'static', menuType: 'liquid', animation: 'splash', variant: '01',
        color: 'blue', enableCustom: false, customPath: ''},
      // AA7 migrated null delay to 0; AA8's new-entry delay and repeatDelay default to 1.
      options: {delay: 0, playOn: 'source', repeat: 3, repeatDelay: 0, size: 1,
        opacity: 1, persistent: false, unbindAlpha: false, unbindVisibility: false},
      sound: {enable: false}
    },
    source: {enable: false}, secondary: {enable: false}, target: {enable: false},
    macro: {enable: false}, soundOnly: {sound: {enable: false}}, levels3d: {enable: false}
  };
}
