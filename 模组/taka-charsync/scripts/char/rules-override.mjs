import {canonicalJson} from '../lib/canon.mjs';
import {CAMPAIGN_RULE_KEYS, resolveActorCampaign, validateRuleSnapshot} from '../lib/campaign-rules.mjs';
import {OWN_FLAG} from '../lib/rules.mjs';
import {installActorCraftingAdapter, installRuleSheetAdapters, isRuleReadError} from './rules-sheets.mjs';

export const CHAR_WORLD = 'pf2creation';

let snapshot = null;
let snapshotError = 'missing-rule-snapshot';
let refreshError = null;
let latestGeneratedAt = 0;
const scopes = [];
const installFailures = new Set();
const actorFailures = new WeakMap();
const wrapped = new WeakMap();

const clone = value => foundry.utils.deepClone(value);
const inCharacterWorld = () => globalThis.game?.world?.id === CHAR_WORLD;
const actors = () => Array.from(globalThis.game?.actors?.values?.() ?? [])
  .filter(actor => ['character', 'familiar'].includes(actor.type));

function documents(collection, rows = []) {
  return {get: id => collection?.get?.(id) ?? rows.find(row => (row._id ?? row.id) === id)};
}

function campaignState(actor) {
  const assigned = resolveActorCampaign(actor, {
    snapshot,
    actors: documents(game.actors, game.data?.actors),
    folders: documents(game.folders, game.data?.folders),
  });
  const state = {...assigned, hash: assigned.world ? snapshot?.hashes[assigned.world] ?? null : null};
  const reason = refreshError ? `rule-refresh-failed:${refreshError}` : snapshotError;
  if (reason) return {...state, ok: false, reason};
  if (!state.ok) return state;
  if (!snapshot.rules[state.world] || !state.hash) return {...state, ok: false, reason: 'missing-profile'};
  return state;
}

function calculationState(actor) {
  if (!inCharacterWorld()) return {ok: false, world: null, hash: null, reason: 'not-character-world'};
  const state = campaignState(actor);
  if (!state.ok) return state;
  const missing = CAMPAIGN_RULE_KEYS.find(key => !game.settings?.settings?.has(key));
  if (missing) return {...state, ok: false, reason: `unregistered-rule:${missing}`};
  if (!game.pf2e?.settings?.variants || !game.pf2e?.settings?.campaign) {
    return {...state, ok: false, reason: 'missing-pf2e-rule-cache'};
  }
  return state;
}

export function actorRuleState(actor) {
  const state = calculationState(actor);
  if (!state.ok) return state;
  const failures = [...installFailures, ...actorFailures.get(actor) ?? []];
  return failures.length ? {...state, ok: false, reason: `adapter-failed:${failures.join(',')}`} : state;
}

/** 工作魔宠的候选主人；冻结副本等仍按提交时的关联解析，不应用选择器限制。 */
export function familiarMasterState(actor, masterId) {
  const flags = actor._source.flags?.[OWN_FLAG] ?? {};
  if (actor.type !== 'familiar' || flags.copy || flags.backupOf || flags.retired) return null;
  const unavailable = reason => ({ok: false, world: null, hash: null, reason});
  if (masterId === null || masterId === '') return unavailable('missing-master');
  const master = game.actors.get(masterId);
  const own = master?._source.flags?.[OWN_FLAG] ?? {};
  if (!master || master.type !== 'character' || own.copy || own.backupOf || own.retired) {
    return unavailable('invalid-master');
  }
  if (!master.isOwner) return unavailable('master-permission');
  const state = actorRuleState(master);
  if (!state.ok) return state;
  return actorRuleState({
    id: actor.id, type: actor.type,
    _source: {...actor._source, system: {...actor._source.system,
      master: {...actor._source.system?.master, id: masterId}}},
  });
}

export function recordRuleAdapterFailure(actor, reason) {
  if (actor) {
    const errors = actorFailures.get(actor) ?? new Set();
    errors.add(reason);
    actorFailures.set(actor, errors);
  } else installFailures.add(reason);
}

function valueFromTable(table, namespace, key) {
  const id = `${namespace}.${key}`;
  const config = game.settings.settings.get(id);
  if (!config) throw new Error(`团规设置未注册：${id}`);
  if (!Object.hasOwn(table, id)) throw new Error(`团规缺少设置：${id}`);
  const value = table[id];
  return clone(value?.$charsyncDefault === true ? config.default : value);
}

/** Read a rule for one Actor. Missing rules are errors, not character-world defaults. */
export function ruleValue(actor, namespace, key) {
  const state = actorRuleState(actor);
  if (!state.ok) throw new Error(`角色团规不可用：${state.reason}`);
  return valueFromTable(snapshot.rules[state.world], namespace, key);
}

function cachedSettings(base, table) {
  const get = key => valueFromTable(table, 'pf2e', key);
  const campaignType = get('campaignType');
  return {
    ...base,
    variants: {
      ...base.variants,
      abp: get('automaticBonusVariant'),
      fa: get('freeArchetypeVariant'),
      gab: get('gradualBoostsVariant'),
      stamina: get('staminaVariant'),
      pwol: {
        enabled: get('proficiencyVariant'),
        modifiers: ['Untrained', 'Trained', 'Expert', 'Master', 'Legendary'].map(rank => get(`proficiency${rank}Modifier`)),
      },
    },
    campaign: {
      ...base.campaign,
      feats: {enabled: get('campaignFeats'), sections: get('campaignFeatSections')},
      mythic: get('mythic'),
      type: campaignType === 'none' ? null : campaignType,
    },
  };
}

/** The callback must finish synchronously. A returned Promise never keeps the shared scope alive. */
export function withActorRules(actor, callback) {
  if (!inCharacterWorld()) return callback();
  const previous = game.pf2e?.settings;
  const worldSettings = scopes[0]?.worldSettings ?? previous;
  // 卡面适配失败仍保留已知团规用于计算；可编辑状态继续由 actorRuleState 锁定。
  const state = calculationState(actor);
  const table = state.ok ? snapshot.rules[state.world] : null;
  const scope = {actor, table, worldSettings};
  let cached = worldSettings;
  if (table) {
    try {
      cached = cachedSettings(worldSettings, table);
    } catch (error) {
      recordRuleAdapterFailure(actor, error.message);
      scope.table = null;
    }
  }
  scopes.push(scope);
  try {
    if (game.pf2e) game.pf2e.settings = cached;
    const result = callback();
    if (result && typeof result.then === 'function') throw new TypeError('Actor 团规回调必须同步完成（synchronous callback required）');
    return result;
  } finally {
    scopes.pop();
    if (game.pf2e) game.pf2e.settings = previous;
  }
}

function wrapMethod(target, name, wrapper, path = null) {
  if (!target || typeof target[name] !== 'function') return false;
  const names = wrapped.get(target) ?? new Set();
  if (names.has(name)) return true;
  try {
    if (path && globalThis.libWrapper?.register) {
      libWrapper.register('taka-charsync', path, function (next, ...args) {
        return wrapper.call(this, next, args);
      }, 'WRAPPER');
    } else {
      const original = target[name];
      target[name] = function (...args) { return wrapper.call(this, original, args); };
    }
  } catch (error) {
    console.error('taka-charsync | 团规适配入口安装失败', path ?? name, error);
    return false;
  }
  names.add(name);
  wrapped.set(target, names);
  return true;
}

function adapterContext() {
  return {
    state: actorRuleState,
    masterState: familiarMasterState,
    value: ruleValue,
    fail: recordRuleAdapterFailure,
    clear: (actor, reason) => actor ? actorFailures.get(actor)?.delete(reason) : installFailures.delete(reason),
  };
}

function prepareActor(actor, name, original, args) {
  const errors = actorFailures.get(actor);
  errors?.delete(name);
  if (name === 'prepareData') errors?.delete('prepareDataFromItems');
  try {
    return withActorRules(actor, () => {
      const result = original.apply(actor, args);
      if (name === 'prepareData' && actor.type === 'character') installActorCraftingAdapter(actor, adapterContext());
      return result;
    });
  } catch (error) {
    recordRuleAdapterFailure(actor, name);
    throw error;
  }
}

/** Install before document construction; no user or saved browser campaign participates. */
export function installActorRuleHooks() {
  if (!inCharacterWorld()) return;
  const classes = globalThis.CONFIG?.PF2E?.Actor?.documentClasses;
  for (const type of ['character', 'familiar']) {
    const prototype = classes?.[type]?.prototype;
    const path = `CONFIG.PF2E.Actor.documentClasses.${type}.prototype`;
    if (!wrapMethod(prototype, 'prepareData', function (original, args) {
      return prepareActor(this, 'prepareData', original, args);
    }, `${path}.prepareData`)) installFailures.add(`${type}.prepareData`);
    // Skill Issue wraps this entry point with libWrapper; direct calls need the same Actor scope.
    if (!wrapMethod(prototype, 'prepareDataFromItems', function (original, args) {
      return prepareActor(this, 'prepareDataFromItems', original, args);
    }, `${path}.prepareDataFromItems`)) installFailures.add(`${type}.prepareDataFromItems`);
  }
  const abp = game.pf2e?.variantRules?.AutomaticBonusProgression;
  if (!wrapMethod(abp, 'isEnabled', function (original, args) {
    const actor = args[0];
    return actor ? withActorRules(actor, () => original.apply(this, args)) : original.apply(this, args);
  })) installFailures.add('AutomaticBonusProgression.isEnabled');
  // registerSheet only queues entries before ready; core populates sheetClasses after setup.
  const sheetConfig = globalThis.foundry?.applications?.apps?.DocumentSheetConfig;
  if (!wrapMethod(sheetConfig, 'initializeSheets', async function (original, args) {
    const result = await original.apply(this, args);
    installRuleSheetAdapters(adapterContext());
    return result;
  }, 'foundry.applications.apps.DocumentSheetConfig.initializeSheets')) {
    installFailures.add('DocumentSheetConfig.initializeSheets');
  }
}

/** Install at module load. Without an Actor scope, ClientSettings keeps its exact original behavior. */
export function installOverride(initialSnapshot) {
  updateRuleSnapshot(initialSnapshot);
  const target = globalThis.foundry?.helpers?.ClientSettings?.prototype;
  if (!wrapMethod(target, 'get', function (original, [namespace, key, options]) {
    const table = inCharacterWorld() && !options?.document ? scopes.at(-1)?.table : null;
    const id = `${namespace}.${key}`;
    if (table && Object.hasOwn(table, id) && this.settings.has(id)) return valueFromTable(table, namespace, key);
    return original.call(this, namespace, key, options);
  })) installFailures.add('ClientSettings.get');
}

function fingerprint(actor) {
  const state = campaignState(actor);
  return canonicalJson({state, rules: state.world ? snapshot?.rules[state.world] : null});
}

function resetChanged(previous) {
  for (const actor of actors()) {
    if (previous.get(actor) === fingerprint(actor)) continue;
    refreshActorRules(actor);
  }
}

function renderOpenApps(actor) {
  for (const app of Object.values(actor.apps ?? {})) {
    if (!app.rendered) continue;
    try {
      const ApplicationV2 = globalThis.foundry?.applications?.api?.ApplicationV2;
      const options = ApplicationV2 && app instanceof ApplicationV2 ? {force: false} : false;
      Promise.resolve(app.render(options)).catch(error => {
        // Async adapters already attribute errors to their latest read, including obsolete renders.
        if (!isRuleReadError(error)) recordRuleAdapterFailure(actor, `render:${error.message}`);
        console.error('taka-charsync | 团规更新后重绘角色卡失败', actor.id, error);
      });
    } catch (error) {
      recordRuleAdapterFailure(actor, `render:${error.message}`);
      console.error('taka-charsync | 团规更新后重绘角色卡失败', actor.id, error);
    }
  }
}

/** Recompute an Actor, or all character/familiar Actors after provenance changes. */
export function refreshActorRules(target) {
  if (!inCharacterWorld()) return;
  for (const actor of target ? [target] : actors()) {
    actorFailures.delete(actor);
    try {
      actor.reset();
    } catch (error) {
      recordRuleAdapterFailure(actor, `reset:${error.message}`);
      console.error('taka-charsync | 按更新后的团规重算角色失败', actor.id, error);
    }
    renderOpenApps(actor);
  }
}

/** Adopt a complete snapshot. Invalid input replaces availability rather than silently retaining stale rules. */
export function updateRuleSnapshot(next) {
  const error = validateRuleSnapshot(next);
  if (!error && next.generatedAt < latestGeneratedAt) return false;
  const previous = new Map(inCharacterWorld() ? actors().map(actor => [actor, fingerprint(actor)]) : []);
  const accepted = error ? null : clone(next);
  const changed = snapshotError !== error || canonicalJson(snapshot && {
    rules: snapshot.rules, hashes: snapshot.hashes, cardWorld: snapshot.cardWorld, folderWorld: snapshot.folderWorld,
  }) !== canonicalJson(accepted && {
    rules: accepted.rules, hashes: accepted.hashes, cardWorld: accepted.cardWorld, folderWorld: accepted.folderWorld,
  });
  snapshot = accepted;
  snapshotError = error;
  if (accepted) latestGeneratedAt = accepted.generatedAt;
  if (changed && inCharacterWorld()) resetChanged(previous);
  return changed;
}

export function setRuleRefreshError(reason) {
  const next = reason == null ? null : String(reason);
  if (next === refreshError) return false;
  refreshError = next;
  if (inCharacterWorld()) {
    if (next === null) refreshActorRules();
    else for (const actor of actors()) renderOpenApps(actor);
  }
  return true;
}

/** Report Actor-specific unavailable rules or failed adapters; never a selected global campaign. */
export function selfCheck(actor) {
  const state = actorRuleState(actor);
  return state.ok ? [] : [state.reason];
}
