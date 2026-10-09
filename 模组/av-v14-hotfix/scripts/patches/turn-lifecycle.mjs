import {adaptNativeHooks} from '../native-hook-adapter.mjs';
import {matchesDeclarations, sourceTokens} from '../source-contract.mjs';

// Exact installed callback sources; changed upstream implementations are skipped.
const CALLBACKS = {
  "reaction": "async a=>{await Na(a,!0),await Ha(a),d(a?.actor)&&A().forEach(e=>{var t=F(e.actor,\"petrifying-glance\");t&&u(a.token,e.token<=30)&&N(h(t),e)});var e=H(a?.actor,\"scapegoat-parallel-self\");e&&await N(h(e),a),1===combat.round&&await ja(a),w.allReactionEffect&&setTimeout(async function(){var e=Ga(a.actor,Aa);!game.settings.get(\"pf2e\",\"automation.removeExpiredEffects\")&&e?await e.update({\"system.start.value\":game.time.worldTime}):((e=(await fromUuid(Aa)).toObject()).system.badge.value=Oa(a),game.settings.get(S,\"customReactionIcon\")&&(e.img=game.settings.get(S,\"customReactionIcon\")),await a.actor.createEmbeddedDocuments(\"Item\",[e]))},300)}",
  "sustain": "async (combatant, encounter, userId) => {\n\tconst { token, actor } = combatant;\n\tconst sustainedEffects = getTokenSustainEffects(token);\n\tif(!sustainedEffects || sustainedEffects.length === 0)\n\t\treturn;\n\t\n\tconst useChat = game.settings.get(moduleId, useChatSetting);\n\n\tif( useChat ) {\n\t\tlet templateData = {};\n\t\ttemplateData.actor = actor;\n\t\ttemplateData.effects = sustainedEffects;\n\t\t// Owners include players with specific setting and game masters\n\t\tconst owners = Object.keys(actor.ownership).filter((key) => actor.ownership[key] == CONST.DOCUMENT_OWNERSHIP_LEVELS.OWNER);\n\t\t\n\t\tconst content = await renderTemplate(`modules/${moduleId}/templates/sustain-reminder.hbs`, templateData);\n\t\t\n\t\tawait ChatMessage.create({\n\t\t\tcontent: content,\n\t\t\tspeaker: ChatMessage.getSpeaker({ token, actor, user: game.users.get(userId) }),\n\t\t\twhisper: owners,\n\t\t\tflags: {[`${moduleId}`]: true }\n\t\t});\n\t}\n}",
  "summons": "async function autoDeleteThrall(effect, info) {\n  if (!game.user.isGM) return;\n  if (effect.rollOptionSlug !== \"thrall-expiration-date\") return;\n\n  const tokDoc = info?.parent?.parent;\n  return await tokDoc.delete();\n}"
};

const installations = new WeakMap();
const sustainContracts = new WeakMap();
const SUSTAIN_CONTRACT = {
  moduleId:"const moduleId = 'pf2e-sustain-reminder';",
  susainedEffectPrefix:"const susainedEffectPrefix = 'Sustaining: ';",
  useChatSetting:"const useChatSetting = 'useChat';",
  getTokenSustainEffects:`function getTokenSustainEffects(token) {
    return token.actor.items.filter((item) => item.type === 'effect' && item.name.startsWith(susainedEffectPrefix));
  }`
};
const CONSUMERS = {
  reaction: {id:'pf2e-reaction', hook:'pf2e.startTurn'},
  sustain: {id:'pf2e-sustain-reminder', hook:'pf2e.startTurn'},
  summons: {id:'pf2e-summons-assistant', hook:'deleteItem'}
};
const skipped = reason => ({status:'skipped', reason});

export async function prepareTurnLifecyclePatch({g = globalThis} = {}) {
  const module = g.game?.modules?.get('pf2e-sustain-reminder');
  if (!module?.active) return skipped('module-inactive');
  const version = module.version;
  sustainContracts.delete(module);
  let result;
  try {
    const path = 'modules/pf2e-sustain-reminder/scripts/sustain-main.mjs';
    const response = await g.fetch(g.foundry?.utils?.getRoute?.(path) ?? path);
    if (!response.ok) throw new Error('Sustain source unavailable');
    result = matchesDeclarations(await response.text(), SUSTAIN_CONTRACT)
      ? {status:'validated'} : skipped('source-contract-mismatch');
  } catch { result = skipped('source-unavailable'); }
  if (g.game.modules.get('pf2e-sustain-reminder') !== module || module.version !== version || !module.active) {
    return skipped('module-changed-during-validation');
  }
  sustainContracts.set(module, {...result, version});
  return result;
}

function summonsSource(Hooks, hook) {
  // Upstream changed indentation without changing this complete callback.
  // Select one matching callback, then let the adapter verify its exact live text.
  let entries;
  try {
    const events = Hooks.events;
    entries = events && Object.getOwnPropertyDescriptor(events, hook)?.value;
  } catch { return null; }
  if (!Array.isArray(entries)) return null;
  const expected = JSON.stringify(sourceTokens(CALLBACKS.summons).map(token => token.value));
  const matches = [];
  for (const entry of entries) {
    const fn = entry && Object.getOwnPropertyDescriptor(entry, 'fn')?.value;
    if (typeof fn !== 'function') continue;
    const source = Reflect.apply(Function.prototype.toString, fn, []);
    try {
      if (JSON.stringify(sourceTokens(source).map(token => token.value)) === expected) matches.push(source);
    } catch { /* Unknown source keeps its original callback. */ }
  }
  return matches.length === 1 ? matches[0] : null;
}

function reactionWrapper(original, {Combatant}) {
  return function(combatant, ...args) {
    // Toolbelt's shared-turn Combatant is not an encounter document. Every
    // Reaction Checker action here assumes a persistent Combatant (including
    // reaction chat cards that later resolve its id), so none belongs to this
    // temporary object. Other Hook consumers still receive the original object.
    if (combatant instanceof Combatant && combatant.id == null && combatant._id == null) return;
    return Reflect.apply(original, this, [combatant, ...args]);
  };
}

async function remindActor(g, actor, userId) {
  // The verified selector reads only token.actor to select effects. The remaining
  // native logic already uses the Actor; Foundry getSpeaker supports no Token.
  const effects = actor.items.filter(item => item.type === 'effect' && item.name.startsWith('Sustaining: '));
  if (!effects || effects.length === 0) return;
  if (g.game.settings.get('pf2e-sustain-reminder', 'useChat')) {
    const owners = Object.keys(actor.ownership).filter(key => actor.ownership[key] == g.CONST.DOCUMENT_OWNERSHIP_LEVELS.OWNER);
    const content = await g.foundry.applications.handlebars.renderTemplate('modules/pf2e-sustain-reminder/templates/sustain-reminder.hbs', {actor, effects});
    await g.ChatMessage.create({content,
      speaker:g.ChatMessage.getSpeaker({token:null, actor, user:g.game.users.get(userId)}),
      whisper:owners, flags:{'pf2e-sustain-reminder':true}});
  }
}

function sustainWrapper(original, {g, Actor, Combatant}) {
  return function(combatant, encounter, userId, ...args) {
    if (combatant instanceof Combatant && combatant.token === null && combatant.actor instanceof Actor) {
      return remindActor(g, combatant.actor, userId);
    }
    return Reflect.apply(original, this, [combatant, encounter, userId, ...args]);
  };
}

function summonsWrapper(original, {g, Actor, Item, TokenDocument, Scene}) {
  return function(effect, info, ...args) {
    // Keep native early returns and errors for all other effects and clients.
    if (!g.game.user.isGM || effect?.rollOptionSlug !== 'thrall-expiration-date') {
      return Reflect.apply(original, this, [effect, info, ...args]);
    }
    const actor = info?.parent, token = actor?.parent;
    // Never infer a Token by actorId, active-token lookup, or a matching slug.
    // The deleted effect must belong to this exact synthetic Actor/Token pair.
    if (!(effect instanceof Item) || effect.type !== 'effect' || effect.parent !== actor
      || !(actor instanceof Actor) || actor.isToken !== true
      || !(token instanceof TokenDocument) || actor.token !== token || token.actorLink !== false
      || !(token.parent instanceof Scene)) return;
    return Reflect.apply(original, this, [effect, info, ...args]);
  };
}

/** Install after module ready callbacks. Each known consumer can adapt independently. */
export function installTurnLifecyclePatch({g = globalThis, report} = {}) {
  const finish = result => {report?.({feature:'turn-lifecycle', ...result}); return result;};
  const skipAll = reason => finish({status:'skipped', reason,
    parts:Object.fromEntries(Object.keys(CONSUMERS).map(key => [key, skipped(reason)]))});
  if (g.game.system?.id !== 'pf2e') return skipAll('unsupported-system');
  const Hooks = g.Hooks;
  if (!Hooks) return skipAll('hook-api-unavailable');
  const prior = installations.get(Hooks);
  if (prior) return finish(prior);
  const {Actor, Combatant, Item, TokenDocument, Scene} = g.foundry?.documents ?? {};
  const context = {g, Actor, Combatant, Item, TokenDocument, Scene};
  const wrappers = {reaction:reactionWrapper, sustain:sustainWrapper, summons:summonsWrapper};
  const available = {
    reaction:typeof Combatant === 'function',
    sustain:typeof Combatant === 'function' && typeof Actor === 'function'
      && typeof g.foundry?.applications?.handlebars?.renderTemplate === 'function' && typeof g.ChatMessage?.getSpeaker === 'function'
      && typeof g.ChatMessage?.create === 'function' && typeof g.game.settings?.get === 'function',
    summons:[Actor, Item, TokenDocument, Scene].every(type => typeof type === 'function')
  };
  const parts = {}, adaptations = [];
  for (const [key, {id, hook}] of Object.entries(CONSUMERS)) {
    const module = g.game.modules?.get(id);
    if (!module?.active) {parts[key] = skipped('module-inactive'); continue;}
    if (!available[key]) {parts[key] = skipped('document-api-unavailable'); continue;}
    if (key === 'sustain') {
      const contract = sustainContracts.get(module);
      if (contract?.version !== module.version || contract?.status !== 'validated') {
        parts[key] = skipped(contract?.reason ?? 'source-not-validated');
        continue;
      }
    }
    const source = key === 'summons' ? summonsSource(Hooks, hook) : CALLBACKS[key];
    if (!source) {parts[key] = skipped('callback-fingerprint-mismatch'); continue;}
    const adapted = adaptNativeHooks({Hooks, callbacks:[{hook, source, wrap:original => wrappers[key](original, context)}]});
    const {restore, ...diagnostics} = adapted;
    parts[key] = {...diagnostics, version:module.version};
    if (adapted.status === 'installed') adaptations.push(adapted);
  }
  if (!adaptations.length) return finish({status:'skipped', reason:'no-supported-consumers', parts});
  const result = {status:'installed', parts, restore() {
    for (const adapted of [...adaptations].reverse()) adapted.restore();
    if (installations.get(Hooks) === result) installations.delete(Hooks);
  }};
  installations.set(Hooks, result);
  return finish(result);
}
