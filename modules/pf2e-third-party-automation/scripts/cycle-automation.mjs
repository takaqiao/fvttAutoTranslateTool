import { MODULE_ID, findFeature } from "./rules.mjs";

const DAMAGE_BY_TRAIT = Object.freeze({
  air: "slashing", cold: "cold", earth: "bludgeoning", electricity: "electricity",
  fire: "fire", metal: "slashing", poison: "poison", sonic: "sonic",
  vitality: "vitality", void: "void", water: "bludgeoning", wood: "piercing",
});
const NONCE = /^[A-Za-z0-9_-]{16,64}$/;

function sourceTraits(item, options = []) {
  const traits = new Set(item?.traits ?? item?.system?.traits?.value ?? []);
  for (const option of options) {
    // self/target traits and damage types do not establish the triggering effect's traits.
    const match = /^(?:item:trait|origin:action:trait|origin:item:trait):([a-z-]+)$/.exec(option);
    if (match) traits.add(match[1]);
  }
  return traits;
}

/** Only a source-identified Cycle user and an actual, unevaded DamageRoll qualify. */
export function getCycleTrigger(actor, { damage, item, rollOptions = [], skipIWR = false, final = false, cycleTrait = null } = {}) {
  if (!findFeature(actor, "cycle") || !findFeature(actor, "attunement") || actor?.isDead) return null;
  if (skipIWR || final || !damage || typeof damage === "number" || !Array.isArray(damage.instances)
    || !Number.isFinite(damage.total) || damage.total <= 0) return null;
  const level = actor.level ?? actor.system?.details?.level?.value;
  if (!Number.isInteger(level) || level < 1 || level > 25) return null;
  const attunement = actor.flags?.pf2e?.cultivator?.energy;
  if (!Object.hasOwn(DAMAGE_BY_TRAIT, attunement)) return null;
  const traits = sourceTraits(item, rollOptions);
  const choices = [attunement, ...(attunement === "void" ? ["vitality"] : attunement === "vitality" ? ["void"] : [])]
    .filter(choice => traits.has(choice));
  if (cycleTrait !== null && !choices.includes(cycleTrait)) return null;
  if (choices.length > 1 && cycleTrait === null) return { trait: null, damageType: null, level,
    choices: choices.map(trait => ({ trait, damageType: DAMAGE_BY_TRAIT[trait] })) };
  const trait = cycleTrait ?? choices[0];
  return trait ? { trait, damageType: DAMAGE_BY_TRAIT[trait], level } : null;
}

/** Ephemeral clone input only. Never embed this effect in an actor. */
export function buildCycleResistance({ level, nonce }) {
  if (!Number.isInteger(level) || level < 1 || level > 25 || !NONCE.test(nonce)) {
    throw new TypeError("循环能量需要有效等级和单次伤害标识。");
  }
  return {
    _id: nonce.slice(0, 16), name: "循环能量（本次伤害）", type: "effect", img: "icons/svg/shield.svg",
    system: {
      slug: "third-party-cycle-incoming", level: { value: level }, description: { value: "" },
      traits: { value: [], rarity: "common" },
      duration: { value: -1, unit: "unlimited", expiry: null, sustained: false },
      start: { value: 0, initiative: null }, tokenIcon: { show: false },
      rules: [{ key: "Resistance", type: "custom", label: "循环能量", value: level,
        definition: [`${MODULE_ID}:cycle:${nonce}`] }],
    },
  };
}

function nativeResistance(actor, effect, options) {
  if (typeof actor.getContextualClone !== "function") throw new Error("当前系统缺少原生伤害上下文接口。");
  const clone = actor.getContextualClone([...options], [effect]);
  const definition = effect.system.rules[0].definition[0];
  const resistance = clone.attributes?.resistances?.find(entry =>
    entry.type === "custom" && entry.definition?.length === 1 && entry.definition[0] === definition);
  if (!resistance || typeof resistance.test !== "function" || typeof resistance.getDoubledValue !== "function") {
    throw new Error("无法生成循环能量的原生单次抗力；此次伤害尚未应用。");
  }
  return resistance;
}

function validClaim(claim, context) {
  if (!claim || !NONCE.test(claim.nonce) || typeof claim.reactionId !== "string" || !claim.reactionId) return false;
  const matchingTarget = ["actorUuid", "tokenUuid", "messageId", "rollIndex", "level"]
    .every(key => claim[key] === context[key]);
  const choices = context.choices ?? [context];
  return matchingTarget && choices.some(choice => choice.trait === claim.trait && choice.damageType === claim.damageType);
}

/**
 * PF2e 8.5 / Toolbelt 3.56 both call contextualActor.applyDamage and DamageRoll.alter.
 * The latter drops top-level options, so a WeakMap propagates source-card identity.
 * No reaction/pending state lives in this client: onClaim MUST run on the elected GM,
 * serialize by actor, validate the sender owns actor, re-read the pending actor flag,
 * verify target/card/roll/trait and atomically consume it before returning a nonce.
 * Claims are never released after a native handler throws: a document update may
 * already have happened. onComplete may record uncertainty, but must not retry/heal.
 * onComplete is also called with claim:null for ordinary tracked damage, allowing
 * shared application history. It must not throw after a successful native update.
 */
export function createCycleAutomation({ onClaim, onComplete = async () => {}, onError = () => {},
  createResistance = nativeResistance, iwrEnabled = () => globalThis.game?.pf2e?.settings?.iwr !== false } = {}) {
  if (typeof onClaim !== "function") throw new TypeError("循环能量需要主 GM 单次认领接口。");
  const rollContexts = new WeakMap();
  const report = error => { try { onError(error); } catch { /* Reporting must never replay damage. */ } };
  const complete = async result => { try { await onComplete(result); } catch (error) { report(error); } };
  return {
    recordDamageMessage(message) {
      if (!message?.isDamageRoll || !message.id) return;
      for (const [rollIndex, roll] of (message.rolls ?? []).entries()) {
        if (roll && typeof roll === "object" && Array.isArray(roll.instances)) {
          rollContexts.set(roll, { messageId: message.id, rollIndex });
        }
      }
    },
    getRollContext(roll) { return roll && typeof roll === "object" ? rollContexts.get(roll) ?? null : null; },
    alterDamageRoll(roll, wrapped, ...args) {
      const result = wrapped(...args);
      const context = rollContexts.get(roll);
      if (context && result && typeof result === "object") rollContexts.set(result, context);
      return result;
    },
    async applyDamage(actor, wrapped, params) {
      const origin = params?.damage && typeof params.damage === "object" ? rollContexts.get(params.damage) : null;
      if (!origin) return wrapped(params);
      // Other native post-damage feats need the same exact receipt identity. This
      // adds metadata only; ordinary targets never enter the Cycle RPC/effect path.
      if (!findFeature(actor, "cycle")) {
        const rollOptions = new Set(params.rollOptions ?? []);
        rollOptions.add(`${MODULE_ID}:source:${origin.messageId}:${origin.rollIndex}`);
        return wrapped({ ...params, rollOptions });
      }
      const context = { actorUuid: actor.uuid, tokenUuid: params.token?.uuid ?? null, ...origin };
      const trigger = iwrEnabled() ? getCycleTrigger(actor, params) : null;
      let accepted = null;
      if (trigger) {
        Object.assign(context, trigger);
        const candidate = await onClaim(context);
        if (candidate && !validClaim(candidate, context)) report(new Error("循环能量认领与本次伤害不匹配，未给予抗力。"));
        else accepted = candidate ?? null;
      }
      let resistance = null;
      let originalResistances = null;
      let enteredNative = false;
      try {
        const options = new Set(params.rollOptions ?? []);
        // PF2e retains this option in its damage-taken receipt. Native Undo can then
        // clear shared application history for this source card without guessing.
        options.add(`${MODULE_ID}:source:${origin.messageId}:${origin.rollIndex}`);
        let actual = { ...params, rollOptions: options };
        if (accepted) {
          const effect = buildCycleResistance(accepted);
          options.add(effect.system.rules[0].definition[0]);
          resistance = createResistance(actor, effect, options);
          originalResistances = actor.attributes?.resistances;
          if (!Array.isArray(originalResistances)) throw new Error("当前角色缺少原生抗力列表；此次伤害尚未应用。");
          originalResistances.push(resistance);
          actual = { ...params, rollOptions: options };
        }
        enteredNative = true;
        const result = await wrapped(actual);
        await complete({ context, claim: accepted, applied: true, uncertain: false });
        return result;
      } catch (error) {
        await complete({ context, claim: accepted, applied: false, uncertain: enteredNative, error: String(error?.message ?? error) });
        throw error;
      } finally {
        // actor.update may replace derived arrays; remove the exact object from both.
        for (const list of new Set([originalResistances, actor.attributes?.resistances])) {
          if (!resistance || !Array.isArray(list)) continue;
          const index = list.indexOf(resistance);
          if (index !== -1) list.splice(index, 1);
        }
      }
    },
  };
}

function messageTargetsActor(actor, message, token, rollIndex) {
  const contextTarget = message.flags?.pf2e?.context?.target;
  const helper = message.flags?.["pf2e-toolbelt"]?.targetHelper;
  const targets = helper?.splashIndex === rollIndex
    ? [...helper?.targets ?? [], ...helper?.splashTargets ?? []] : helper?.targets ?? [];
  return contextTarget?.actor === actor.uuid || !!(token?.uuid &&
    (contextTarget?.token === token.uuid || targets.includes(token.uuid)));
}

/** Read-only selection, shared with ordinary skill use and the damage-card shortcut. */
export function getCycleContext(actor, message, { token = null, rollIndex = 0, trait = null } = {}) {
  const base = { actorUuid: actor.uuid, tokenUuid: token?.uuid ?? null, messageId: message?.id, rollIndex };
  if (!message?.isDamageRoll || !Number.isInteger(rollIndex) || rollIndex < 0) return { ...base, status: "not-damage" };
  const trigger = getCycleTrigger(actor, { damage: message.rolls?.[rollIndex], item: message.item,
    rollOptions: message.flags?.pf2e?.context?.options ?? [], cycleTrait: trait });
  if (!trigger) return { ...base, status: "not-eligible" };
  if (!messageTargetsActor(actor, message, token, rollIndex)) return { ...base, ...trigger, status: "not-targeted" };
  const helperApplied = token?.id && message.flags?.["pf2e-toolbelt"]?.targetHelper?.applied?.[token.id]?.[rollIndex];
  const recorded = message.flags?.[MODULE_ID]?.cycleApplications?.findLast(entry =>
    entry.actorUuid === actor.uuid && entry.rollIndex === rollIndex);
  const applied = recorded ? recorded.applied !== false : helperApplied;
  return { ...base, ...trigger, status: applied ? "already-applied" : trigger.choices ? "choice-required" : "ready" };
}

/** Never fall back to an older attack after the latest eligible attack was settled. */
export function resolveCycleDamageMessage(actor, messages, { now = Date.now(), maxAgeMs = 120000, token = null } = {}) {
  const recent = [...messages].filter(message => Number.isFinite(message.timestamp) &&
    message.timestamp <= now && now - message.timestamp <= maxAgeMs).sort((a, b) => b.timestamp - a.timestamp);
  for (const message of recent) {
    for (let rollIndex = 0; rollIndex < (message.rolls?.length ?? 0); rollIndex++) {
      const context = getCycleContext(actor, message, { token, rollIndex });
      if (["ready", "choice-required", "already-applied"].includes(context.status)) return context;
    }
  }
  return null;
}

/** Add one ordinary reaction entry per eligible target; onUse routes to the normal GM action backend. */
export function addCycleReactionButtons(message, html, { targets = [], onUse, onError = () => {} } = {}) {
  const root = html?.[0] ?? html;
  if (!message?.isDamageRoll || !root?.querySelector || typeof onUse !== "function") return;
  const document = root.ownerDocument;
  for (const { actor, token } of targets) {
    if (!actor?.isOwner) continue;
    for (let rollIndex = 0; rollIndex < (message.rolls?.length ?? 0); rollIndex++) {
      const context = getCycleContext(actor, message, { token, rollIndex });
      if (!["ready", "choice-required", "already-applied"].includes(context.status)) continue;
      for (const choice of context.choices ?? [context]) {
        const key = `${actor.uuid}:${rollIndex}:${choice.trait}`;
        if ([...root.querySelectorAll("[data-cycle-reaction]")].some(button => button.dataset.cycleReaction === key)) continue;
        const rows = [...root.querySelectorAll("[data-target-uuid]")];
        const row = rows.find(entry => entry.dataset.targetUuid === token?.uuid && Number(entry.dataset.targetRollIndex ?? 0) === rollIndex)
          ?? root.querySelector(".message-content");
        if (!row) continue;
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.cycleReaction = key;
        const label = context.choices ? (choice.trait === "void" ? "（虚能）" : "（命能）") : "";
        button.textContent = `↻ 循环能量${label}${targets.length > 1 ? `：${actor.name}` : ""}`;
        button.title = "在应用本次伤害前使用反应；抗力与后续打击效果会自动处理。";
        button.addEventListener("click", async event => {
          event.preventDefault(); event.stopPropagation();
          if (button.disabled) return;
          button.disabled = true;
          try { await onUse(actor, { messageId: message.id, rollIndex, tokenUuid: token?.uuid ?? null, trait: choice.trait }); }
          catch (error) { onError(error); }
          finally { button.disabled = false; }
        });
        row.append(button);
      }
    }
  }
}
