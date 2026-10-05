import { normalizeCounter } from './model.mjs';

const numeric = (value) => Number.isFinite(value) ? value : null;
const text = (value, fallback = '') => typeof value === 'string' ? value : fallback;
const slugify = (value) => value.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase();
const values = (collection) => collection ? Array.from(typeof collection.values === 'function' ? collection.values() : collection) : [];
const noPool = (rank, label, unlimited = false) => ({ rank, label, value: null, max: null, unlimited });

function isPrivate(item, policy) {
  try {
    return policy?.isPrivateItem?.(item) === true;
  } catch {
    return true;
  }
}

function castingKind(data) {
  if (data.category === 'items') return 'items';
  if (data.isRitual === true) return 'ritual';
  if (data.isFocusPool === true) return 'focus';
  if (data.isInnate === true) return 'innate';
  if (data.isFlexible === true) return 'flexible';
  if (data.isSpontaneous === true) return 'spontaneous';
  if (data.isPrepared === true) return 'prepared';
  return 'unknown';
}

/**
 * Project one PF2e getSheetData result into the fixed ActorView entry shape.
 * No spell/item Document or raw system field survives this projection.
 * Focus numbers live exclusively in ActorView.counters; labels retain costs.
 */
export function summarizeSpellcasting(sheetData, { localize = (label) => label, resourcePolicy } = {}) {
  const data = sheetData ?? {};
  const kind = castingKind(data);
  const groups = [];
  const seenItems = new Set();
  for (const group of data.groups ?? []) {
    const cantrip = group.id === 'cantrips' || group.number === 0 || group.id === 0;
    const rank = cantrip ? 0 : numeric(group.number ?? group.id);
    const label = text(localize(text(group.label)), rank === null ? '' : `${rank}环`);
    const active = (group.active ?? []).filter((slot) => slot?.spell);
    if (kind === 'items') {
      for (const { spell } of active) {
        const parent = spell.parentItem;
        if (!parent || parent.isIdentified !== true || parent.isStowed === true || isPrivate(parent, resourcePolicy) || seenItems.has(parent.id)) continue;
        seenItems.add(parent.id);
        const name = text(parent.name);
        const quantity = numeric(parent.quantity);
        if (parent.category === 'scroll' || parent.category === 'spell-gem') {
          if (quantity !== null && quantity >= 0) groups.push({ rank, label: `${name} · 库存`, value: quantity, max: null, unlimited: false });
        } else {
          if (quantity !== null && quantity > 1) groups.push({ rank, label: `${name} · 库存`, value: quantity, max: null, unlimited: false });
          const uses = normalizeCounter(parent.uses?.value, parent.uses?.max);
          groups.push(uses ? { rank, label: `${name} · 本件次数`, ...uses, unlimited: false } : noPool(rank, name));
        }
      }
    } else if (kind === 'ritual') {
      if (active.length) groups.push(noPool(rank, label, true));
    } else if (kind === 'focus') {
      for (const { spell } of active) {
        const cost = numeric(spell.system?.cast?.focusPoints);
        const costLabel = cost !== null && cost >= 0 ? `${text(spell.name)} · 专注 ${cost}` : text(spell.name);
        groups.push(noPool(rank, costLabel, cantrip && cost === 0));
      }
    } else if (kind === 'innate') {
      for (const slot of active) {
        const spell = slot.spell;
        const cost = numeric(spell.system?.cast?.focusPoints);
        if (cost !== null && cost > 0) {
          groups.push(noPool(rank, `${text(spell.name)} · 专注 ${cost}`));
        } else if (cantrip && cost === 0) {
          groups.push(noPool(rank, text(spell.name), true));
        } else {
          const uses = normalizeCounter(slot.uses?.value ?? spell.system?.location?.uses?.value, slot.uses?.max ?? spell.system?.location?.uses?.max);
          groups.push(uses ? { rank, label: text(spell.name), ...uses, unlimited: false } : noPool(rank, text(spell.name)));
        }
      }
    } else if (cantrip && ['prepared', 'spontaneous', 'flexible'].includes(kind)) {
      let hasFreeCantrip = false;
      for (const { spell } of active) {
        const cost = numeric(spell.system?.cast?.focusPoints);
        if (cost !== null && cost > 0) groups.push(noPool(rank, `${text(spell.name)} · 专注 ${cost}`));
        else if (cost === 0) hasFreeCantrip = true;
      }
      if (hasFreeCantrip) groups.push(noPool(rank, label, true));
    } else if (kind === 'prepared') {
      const pool = normalizeCounter(active.filter((slot) => slot.expended !== true).length, group.uses?.max);
      if (pool) groups.push({ rank, label, ...pool, unlimited: false });
    } else if (kind === 'spontaneous' || kind === 'flexible') {
      const pool = normalizeCounter(group.uses?.value, group.uses?.max);
      if (pool) groups.push({ rank, label, ...pool, unlimited: false });
    } else if (active.length) {
      groups.push(noPool(rank, label));
    }
  }
  return { id: text(data.id), label: text(data.name), dc: kind === 'ritual' ? null : numeric(data.statistic?.dc?.value), kind, groups };
}

/** Mandatory gate: callers derive both allowlist and callback from current
 * configured User.character assignments, never from actor.type/playerOwner.
 * The callback must synchronously return exactly true, checking actor identity
 * and runtime generation. Missing/false/throwing/async callbacks fail closed.
 */
function mayCollect(actor, options, actorId) {
  try {
    return !!actor && !!options.viewer && options.viewer.isGM !== true
      && typeof actorId === 'string' && actor.id === actorId
      && new Set(options.allowedActorIds ?? []).has(actorId)
      && typeof options.isStillAssigned === 'function'
      && options.isStillAssigned(actor) === true
      && actor.testUserPermission?.(options.viewer, 'OBSERVER') === true
      && actor.type === 'character';
  } catch {
    return false;
  }
}

function collectStats(actor, attributes, localize) {
  const stats = [{ label: 'AC', value: numeric(actor.armorClass?.value ?? attributes.ac?.value) }];
  for (const [slug, label] of [['fortitude', '强韧'], ['reflex', '反射'], ['will', '意志'], ['perception', '察觉']]) {
    const statistic = actor.getStatistic?.(slug);
    const dc = numeric(statistic?.dc?.value);
    stats.push({ label, value: numeric(statistic?.mod), ...(dc !== null ? { dc } : {}) });
  }
  const speeds = Object.values(actor.movement?.speeds ?? {});
  speeds.sort((a, b) => (a?.type === 'land' ? -1 : 0) - (b?.type === 'land' ? -1 : 0));
  for (const speed of speeds) {
    const value = numeric(speed?.value);
    if (speed?.type === 'travel' || value === null || value < 0 || (speed?.type !== 'land' && value === 0)) continue;
    const fallback = speed.type === 'land' ? '地面' : text(speed.type);
    stats.push({ label: text(localize(text(speed.label, fallback))), value, ...(text(speed.type) ? { movementType: speed.type } : {}) });
  }
  return stats;
}

function collectCounters(actor, attributes, localize, policy) {
  const resources = actor.system?.resources ?? {};
  const counters = [];
  const seen = new Set();
  const add = (id, label, pool) => {
    const counter = normalizeCounter(pool?.value, pool?.max);
    if (counter && !seen.has(id)) {
      seen.add(id);
      counters.push({ id, label: text(localize(label)), ...counter });
    }
  };
  const mythic = normalizeCounter(resources.mythicPoints?.value, resources.mythicPoints?.max);
  if (mythic) add('mythic-points', '神话点', mythic);
  else add('hero-points', '英雄点', resources.heroPoints);
  add('focus', '专注', resources.focus);
  add('investiture', '投资已占用', resources.investiture);
  add('resolve', '决心', resources.resolve);
  add('infused-reagents', '注入试剂', resources.crafting?.infusedReagents);
  add('stamina', '耐力', attributes.hp?.sp);
  const synthetic = actor.synthetics?.resources ?? {};
  const candidates = new Set([...Object.keys(synthetic), ...(policy?.customResourceSlugs ?? [])]);
  for (const candidate of candidates) {
    const slug = slugify(candidate);
    if (seen.has(slug) || ['hero-points', 'mythic-points', 'focus', 'investiture', 'resolve'].includes(slug)) continue;
    const key = candidate.replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
    const rule = synthetic[key];
    // Check the backing item before invoking getResource: even its label can
    // disclose an unidentified effect. Unknown custom provenance is omitted.
    if (!rule || rule.ignored === true || rule.item?.isIdentified !== true || isPrivate(rule.item, policy)) continue;
    const resource = actor.getResource?.(slug);
    if (resource) add(slug, text(resource.label, text(rule.label)), resource);
  }
  return counters;
}

function collectStatuses(actor, policy) {
  const statuses = [];
  const seen = new Set();
  for (const item of [...values(actor.conditions?.active), ...values(actor.itemTypes?.effect)]) {
    if (item.isIdentified !== true || item.system?.expired === true || isPrivate(item, policy) || seen.has(item.id)) continue;
    seen.add(item.id);
    statuses.push({ id: text(item.id), label: text(item.name) });
  }
  return statuses;
}

/**
 * Read-only PF2e 8.5.1 collector. Returns a fresh complete ActorView or null;
 * null means the runtime must clear a previous snapshot, never retain it.
 * No globals, mutation, preparation, cast, consume or resource updates.
 */
export async function collectActorView(actor, options = {}) {
  let actorId;
  try { actorId = actor?.id; } catch { return null; }
  if (!mayCollect(actor, options, actorId)) return null;
  const localize = typeof options.localize === 'function' ? options.localize : (label) => label;
  const policy = options.resourcePolicy;
  const attributes = actor.attributes ?? actor.system?.attributes ?? {};
  const health = normalizeCounter(attributes.hp?.value, attributes.hp?.max);
  const shield = attributes.shield;
  const classStatistic = actor.getStatistic?.('class');
  const classValue = numeric(classStatistic?.dc?.value);
  const view = {
    actorId,
    name: text(actor.name),
    portrait: text(options.portraitOverride) || text(actor.img, null),
    hp: health ? { ...health, temp: numeric(attributes.hp?.temp) } : null,
    stats: collectStats(actor, attributes, localize),
    shield: shield && (shield.itemId || shield.raised === true) ? {
      hp: shield.itemId ? normalizeCounter(shield.hp?.value, shield.hp?.max) : null,
      hardness: shield.itemId ? numeric(shield.hardness) : null,
      raised: shield.raised === true,
      broken: shield.broken === true,
    } : null,
    counters: collectCounters(actor, attributes, localize, policy),
    classDc: classValue === null ? null : { label: text(localize(text(classStatistic.label, '职业 DC'))), value: classValue },
    spellcasting: [],
    conditions: collectStatuses(actor, policy),
    publicOnly: false,
  };
  const collections = values(actor.spellcasting?.collections);
  const settled = await Promise.allSettled(collections.map(async (spells) => {
    return await spells.entry.getSheetData({ spells, prepList: false });
  }));
  // Do not even project the completed sheet Documents after binding/permission
  // revocation: a stale request cannot return previously collected numbers.
  if (!mayCollect(actor, options, actorId)) return null;
  view.spellcasting = settled.filter((result) => result.status === 'fulfilled').map((result) => summarizeSpellcasting(result.value, { localize, resourcePolicy: policy }));
  return mayCollect(actor, options, actorId) ? view : null;
}
