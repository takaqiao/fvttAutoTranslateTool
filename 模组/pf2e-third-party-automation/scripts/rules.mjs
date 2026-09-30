export const MODULE_ID = "pf2e-third-party-automation";

export const SOURCES = Object.freeze({
  breath: "Compendium.pf2e-team-plus-feats.pf2e-player-options.Item.YDMPRW1g5igmuoy0",
  legend: "Compendium.pf2e-team-plus-feats.pf2e-player-options.Item.DDiRbAbDboplwTzn",
  circadian: "Compendium.pf2e-team-plus-feats.pf2e-player-options.Item.OILWJgTq4PlQgwLZ",
  attunement: "Compendium.pf2e-team-plus-tian-xia.player-options.Item.PMmHJWL8NFC5OPlo",
  cycle: "Compendium.pf2e-team-plus-tian-xia.misc.Item.RAp8SWvF6AZqR2zr",
  swordQiFeat: "Compendium.pf2e-team-plus-tian-xia.player-options.Item.ny1FcO87ZDyDQYAu",
  swordQiSpell: "Compendium.pf2e-team-plus-tian-xia.player-options.Item.cRnnxbFQLUzvgnPG",
});

const LEVEL_DCS = Object.freeze([
  14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30,
  31, 32, 34, 35, 36, 38, 39, 40, 42, 44, 46, 48, 50,
]);

export const ATTUNEMENT_DAMAGE = Object.freeze({
  air: "slashing",
  cold: "cold",
  earth: "bludgeoning",
  electricity: "electricity",
  fire: "fire",
  metal: "slashing",
  poison: "poison",
  sonic: "sonic",
  vitality: "vitality",
  void: "void",
  water: "bludgeoning",
  wood: "piercing",
});

const DAMAGE_TYPES = new Set([
  "acid", "bleed", "bludgeoning", "cold", "electricity", "fire", "force",
  "mental", "piercing", "poison", "slashing", "sonic", "spirit", "vitality", "void", "untyped",
]);

/** Canonicalize compendium Item UUIDs without changing embedded world UUIDs. */
export function normalizeUuid(uuid) {
  if (typeof uuid !== "string") return "";
  return uuid.trim().replace(/^(Compendium\.[^.]+\.[^.]+)\.Item\./, "$1.");
}

export function hasSource(item, uuid) {
  const expected = normalizeUuid(uuid);
  if (!item || !expected) return false;
  return [item.sourceId, item._stats?.compendiumSource, item.flags?.core?.sourceId]
    .some(source => normalizeUuid(source) === expected);
}

export function findFeature(actor, key) {
  if (!Object.hasOwn(SOURCES, key)) return undefined;
  const items = actor?.items;
  const values = Array.isArray(items) ? items
    : typeof items?.values === "function" ? [...items.values()]
      : Array.isArray(items?.contents) ? items.contents : [];
  return values.find(item => hasSource(item, SOURCES[key]));
}

export function focusRecovery({ value, max, uses }) {
  for (const [name, number] of Object.entries({ value, max, uses })) {
    if (!Number.isInteger(number) || number < 0) throw new RangeError(`无效的${name}资源值。`);
  }
  if (uses === 0) throw new RangeError("鼓舞之息没有剩余使用次数。");
  if (value >= max) throw new RangeError("没有可以恢复的聚能点。");
  return { value: value + 1, uses: uses - 1 };
}

export function levelDC(level) {
  if (!Number.isInteger(level) || level < 0 || level >= LEVEL_DCS.length) {
    throw new RangeError("等级必须是 0 至 25 的整数。");
  }
  return LEVEL_DCS[level];
}

export function restRecovery({ level, constitution, degree }) {
  levelDC(level);
  if (!Number.isFinite(constitution)) throw new RangeError("体质调整值必须是有效数字。");
  const fullRecovery = level * Math.max(1, constitution);
  return degree === 3 ? Math.floor(fullRecovery) : degree === 2 ? Math.floor(fullRecovery / 2) : 0;
}

function uniqueItemIds(ids) {
  if (!Array.isArray(ids) || ids.some(id => typeof id !== "string" || !/^[A-Za-z0-9_-]+$/.test(id))) {
    throw new TypeError("必须提供角色物品 ID 数组。");
  }
  return [...new Set(ids)];
}

function makeEffect(kind, name, rules, data = {}) {
  return {
    name,
    type: "effect",
    img: "icons/svg/aura.svg",
    flags: { [MODULE_ID]: { kind, ...data } },
    system: {
      slug: `third-party-${kind}`,
      description: { value: "" },
      duration: { value: -1, unit: "unlimited", expiry: null, sustained: false },
      tokenIcon: { show: true },
      rules,
    },
  };
}

/** The runtime replaces/removes this module-owned effect at daily preparation. */
export function buildLegendEffect(itemIds) {
  const selected = uniqueItemIds(itemIds);
  if (selected.length > 2) throw new RangeError("魔法物品传奇最多选择两件物品。");
  return makeEffect("legend", "效果：魔法物品传奇", selected.map(id => ({
    key: "AdjustModifier",
    selector: `${id}-inline-dc`,
    slug: "base",
    mode: "override",
    // PF2e getCheckDC stores the base modifier as DC - 10, then adds 10.
    value: "@actor.system.attributes.classOrSpellDC.value - 12",
  })), { itemIds: selected });
}

/** Only the ensuing Strike bonus persists; triggering-damage resistance is not an effect. */
export function buildCycleEffect(damageType, { worldTime, initiative }) {
  if (!DAMAGE_TYPES.has(damageType)) throw new RangeError("无效的循环能量伤害类型。");
  if (!Number.isFinite(worldTime) || (initiative !== null && !Number.isFinite(initiative))) {
    throw new RangeError("必须提供当前世界时间与先攻值（或 null）。");
  }
  const effect = makeEffect("cycle", "效果：循环能量", [{
    key: "FlatModifier",
    selector: "strike-damage",
    slug: "third-party-cycle-energy",
    type: "untyped",
    value: "@weapon.system.damage.dice",
    damageType,
  }], { damageType });
  effect.system.duration = { value: 1, unit: "rounds", expiry: "turn-end", sustained: false };
  effect.system.start = { value: worldTime, initiative };
  return effect;
}

/** Caller supplies only embedded Cultivator spell IDs that the player chose to convert. */
export function buildAttunementEffect(trait, spellIds) {
  if (!Object.hasOwn(ATTUNEMENT_DAMAGE, trait)) throw new RangeError("无效的能量调谐特征。");
  const selected = uniqueItemIds(spellIds);
  const damageType = ATTUNEMENT_DAMAGE[trait];
  const rules = [{
    key: "ActiveEffectLike",
    mode: "override",
    path: "flags.pf2e.cultivator.energy",
    value: trait,
  }];
  if (selected.length) {
    rules.push({key:'RollOption',domain:'all',option:'third-party-attunement-conversion',toggleable:true,value:true,label:'能量调谐：转换修炼者法术',placement:'spellcasting'});
    rules.push({
      key: "DamageAlteration",
      selectors: selected.map(id => `${id}-damage`),
      mode: "override",
      property: "damage-type",
      value: damageType,
      predicate: ['third-party-attunement-conversion',{ or: ["damage:type:force", "damage:type:spirit"] }],
    });
    // PF2e ItemAlteration explicitly supports spell traits and specific itemId targeting.
    rules.push(...selected.map(itemId => ({
      key: "ItemAlteration",
      itemId,
      itemType: "spell",
      mode: "add",
      property: "traits",
      value: trait,
      predicate: ['third-party-attunement-conversion'],
    })));
  }
  return makeEffect("attunement", "效果：能量调谐", rules, { trait, damageType, spellIds: selected });
}
