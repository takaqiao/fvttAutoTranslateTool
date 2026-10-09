import {adaptNativeHooks} from '../native-hook-adapter.mjs';
import {matchesDeclarations, namedImportMatches} from '../source-contract.mjs';

// The live callbacks are checked separately from their private dependencies.
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
// Only this guarded loop leaves shouldAlwaysShowEffect uncalled on the fast path.
const VISIBILITY_CONTRACT = {
  setEffectVisibility:"function setEffectVisibility(token, value, { surfaceMode }) {\n  const fxInfoList = token.actor.appliedEffects;\n  if (!shouldSkipEffectBackground()) {\n    token.effects.bg.visible = value;\n  }\n\n  // Skip base overlay\n  // Skip background frames\n  const fxs = token.effects.children.filter(\n    (fx) => fx !== token.effects.overlay && fx !== token.effects.bg,\n  );\n\n  let cnt = -1;\n  for (const fx of fxs) {\n    cnt++;\n    if (fx.visible !== value) {\n      if (shouldAlwaysShowEffect(fxInfoList[cnt], { surfaceMode })) continue; // Skip effects that should always be shown\n      fx.visible = value;\n    }\n  }\n}",
  shouldShowEffects:"function shouldShowEffects(token) {\n  if (token.hover) return true;\n\n  if (canvas.tokens.highlightObjects) return true;\n\n  return false;\n\n  //   if (isCombatRunning() && getShowDuringCombat()) return true;\n\n  //   return game.settings.get(MODULE_ID, \"hideEffects\") === \"never\";\n}",
  isValidCharacter:"function isValidCharacter(actorType, setting) {\n  switch (setting) {\n    case \"all\":\n      return true;\n    case \"pcs\":\n      return actorType === \"character\";\n    case \"npcs\":\n      return actorType === \"npc\";\n    case \"none\":\n      return false;\n  }\n}",
  refreshEffectVisibility:"function refreshEffectVisibility(token, { surfaceMode }) {\n  if (shouldShowEffects(token)) {\n    setEffectVisibility(token, true, { surfaceMode });\n  } else {\n    setEffectVisibility(token, false, { surfaceMode });\n  }\n}",
  shouldSkipEffectBackground:"function shouldSkipEffectBackground() {\n  return BG_FRAME_SKIP_MODULES.some((id) => game.modules.get(id)?.active);\n}",
};
const SETTINGS_CONTRACT = {getSetting:"function getSetting(settingID) {\n  return game.settings.get(MODULE_ID, settingID);\n}"};
const MODULE_CONTRACT = {MODULE_ID:"const MODULE_ID = \"sundry\";"};
const BACKGROUND_CONTRACT = {BG_FRAME_SKIP_MODULES:"const BG_FRAME_SKIP_MODULES = [\n  \"pf2e-dorako-ux\",\n  \"pathfinder-ui\",\n  \"pf2e-effects-halo\",\n];"};
const installations = new WeakMap();
const sourceContracts = new WeakMap();

/** Validate only the source dependencies used by the unchanged-icon shortcut. */
export async function prepareSundryPatch({g = globalThis} = {}) {
  const module = g.game?.modules?.get('sundry');
  if (!module?.active) return {status:'skipped', reason:'module-inactive'};
  const version = module.version;
  sourceContracts.delete(module);
  let result;
  try {
    const files = await Promise.all(['lib/tokenEffectHider.js', 'lib/const.js', 'lib/helpers.js', 'module.js'].map(async path => {
      const url = `modules/sundry/scripts/${path}`;
      const response = await g.fetch(g.foundry?.utils?.getRoute?.(url) ?? url);
      if (!response.ok) throw new Error('Sundry source unavailable');
      return response.text();
    }));
    const [visibility, constants, helpers, entry] = files;
    const valid = matchesDeclarations(visibility, VISIBILITY_CONTRACT)
      && matchesDeclarations(constants, BACKGROUND_CONTRACT)
      && matchesDeclarations(helpers, SETTINGS_CONTRACT)
      && matchesDeclarations(entry, MODULE_CONTRACT)
      && namedImportMatches(visibility, 'getSetting', './helpers.js')
      && namedImportMatches(visibility, 'BG_FRAME_SKIP_MODULES', './const.js')
      && namedImportMatches(helpers, 'MODULE_ID', '../module.js');
    result = valid ? {status:'validated'} : {status:'skipped', reason:'source-contract-mismatch'};
  } catch { result = {status:'skipped', reason:'source-unavailable'}; }
  if (g.game.modules.get('sundry') !== module || module.version !== version || !module.active) {
    return {status:'skipped', reason:'module-changed-during-validation'};
  }
  sourceContracts.set(module, {...result, version});
  return result;
}

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
  const contract = sourceContracts.get(module);
  if (contract?.version !== module.version || contract?.status !== 'validated') return finish({status:'skipped', reason:contract?.reason ?? 'source-not-validated'});
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
    const nativeSprites = g.game.system?.id === 'pf2e'
      && g.PIXI?.Sprite?.prototype;
    for (const effect of effects.children) {
      if (effect === effects.bg || effect === effects.overlay) continue;
      // The verified loop skips its private helper when visibility already matches.
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
