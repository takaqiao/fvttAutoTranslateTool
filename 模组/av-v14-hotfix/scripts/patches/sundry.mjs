import {adaptNativeHooks} from '../native-hook-adapter.mjs';

// These versions share the same hook callbacks; the visibility loop changed in 1.11.0.
const CALLBACKS = {
  refreshToken: `(token) => {
    const surfaceMode = getSetting("hide.effects.token.surface");
    const characterTypes = getSetting("hide.effects.token.enabled-for");
    if (isValidCharacter(token?.actor?.type, characterTypes)) {
      refreshEffectVisibility(token, { surfaceMode });
    }
  }`,
  highlightObjects: `(state) => {
    const surfaceMode = getSetting("hide.effects.token.surface");
    for (const token of canvas.tokens.placeables) {
      const characterTypes = getSetting("hide.effects.token.enabled-for");
      if (isValidCharacter(token?.actor?.type, characterTypes)) {
        refreshEffectVisibility(token, { surfaceMode });
      }
    }
  }`
};
const BACKGROUND_OWNERS = ['pf2e-dorako-ux', 'pathfinder-ui', 'pf2e-effects-halo'];
const installations = new WeakMap();

function enabledFor(type, setting) {
  switch (setting) {
    case 'all': return true;
    case 'pcs': return type === 'character';
    case 'npcs': return type === 'npc';
    default: return false;
  }
}

/** Call after Sundry's ready hook. Preserve the exact core's live callback records. */
export function installSundryPatch({g = globalThis, report} = {}) {
  const finish = result => {report?.({feature: 'sundry', ...result}); return result;};
  const module = g.game?.modules?.get('sundry');
  if (!module?.active) return finish({status: 'skipped', reason: 'module-inactive'});
  if (!['1.10.2', '1.10.3', '1.11.0'].includes(module.version)) return finish({status: 'skipped', reason: 'version-mismatch'});
  if ((g.game.release?.generation??Number.parseInt(g.game.version,10)) !== 14) return finish({status: 'skipped', reason: 'core-version-mismatch'});
  const Hooks = g.Hooks;
  if (!Hooks || typeof g.game?.settings?.get !== 'function') return finish({status: 'skipped', reason: 'hook-api-unavailable'});
  const prior = installations.get(Hooks);
  if (prior) return finish(prior);
  let originalRefresh;
  function refresh(token, ...args) {
    const effects = token?.effects;
    if (!token?.actor || !effects?.bg || !Array.isArray(effects.children)) {
      return Reflect.apply(originalRefresh,this,[token,...args]);
    }
    const value = Boolean(token.hover || g.canvas.tokens.highlightObjects);
    const nativeSprites = g.game.system?.id === 'pf2e' && Number.parseInt(g.game.system.version,10) === 8
      && g.PIXI?.Sprite?.prototype;
    for (const effect of effects.children) {
      if (effect === effects.bg || effect === effects.overlay) continue;
      // Sundry either keeps an always-shown icon's value or assigns `value`.
      // Both are no-ops when a native Sprite already has that exact boolean.
      // Inspect current children on every call: no effect/time/Actor cache.
      if (!nativeSprites || !effect || Object.getPrototypeOf(effect) !== nativeSprites) {
        return Reflect.apply(originalRefresh,this,[token,...args]);
      }
      const visible = Object.getOwnPropertyDescriptor(effect,'visible');
      if (!visible?.writable || typeof visible.value !== 'boolean' || visible.value !== value) {
        return Reflect.apply(originalRefresh,this,[token,...args]);
      }
    }
    if (!enabledFor(token.actor.type, g.game.settings.get('sundry', 'hide.effects.token.enabled-for'))) return;
    if (!BACKGROUND_OWNERS.some(id => g.game.modules.get(id)?.active)) {
      effects.bg.visible = value;
    }
    // No ordinary icon can change; appliedEffects wrappers are unnecessary.
  }
  function highlight() {
    for (const token of g.canvas.tokens.placeables) refresh.call(this,token);
  }

  const adapted = adaptNativeHooks({Hooks,callbacks:[
    {hook:'refreshToken',source:CALLBACKS.refreshToken,wrap:original=>{originalRefresh=original;return refresh;}},
    {hook:'highlightObjects',source:CALLBACKS.highlightObjects,wrap:()=>highlight}
  ]});
  if (adapted.status !== 'installed') return finish(adapted);

  const result = {
    ...adapted, version: module.version,
    restore() {
      adapted.restore();
      if (installations.get(Hooks) === result) installations.delete(Hooks);
    }
  };
  installations.set(Hooks, result);
  return finish(result);
}
