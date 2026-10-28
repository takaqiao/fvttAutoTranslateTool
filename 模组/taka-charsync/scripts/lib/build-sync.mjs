import {canonicalJson, stripVolatile} from './canon.mjs';
import {copyAppearanceGroups, mergeAppearanceGroups} from './appearance.mjs';
import {merge3} from './merge.mjs';
import {combineMediaMetadata, mediaComparisonDocs} from './media.mjs';
import {isAppearancePath, LOCAL_ACTOR_KEYS, LOCAL_ITEM_KEYS, OWN_FLAG} from './rules.mjs';

const BUILD_TYPES = new Set([
  'feat', 'class', 'ancestry', 'heritage', 'background', 'spell', 'spellcastingEntry', 'lore', 'action', 'deity',
]);

const CHARACTER_PATHS = [
  'system.build.attributes', 'system.abilities', 'system.details.level', 'system.details.keyability',
  'system.details.languages', 'system.details.biography', 'system.skills.*.rank',
  'system.proficiencies', 'system.crafting.formulas', 'system.resources.*.max',
  'flags.pf2e-skill-issue',
].map(path => path.split('.'));
const FAMILIAR_PATHS = ['system.master.id', 'system.master.ability', 'system.details.creature.value'].map(path => path.split('.'));

const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const clone = value => structuredClone(value);
const matches = (path, patterns) => {
  const segments = path.split('.');
  return patterns.some(pattern => pattern.length <= segments.length
    && pattern.every((part, index) => part === '*' || part === segments[index]));
};

function buildActorPath(path, type) {
  if (isAppearancePath(path)) return true;
  if (type === 'familiar') return matches(path, FAMILIAR_PATHS);
  return type === 'character' && matches(path, CHARACTER_PATHS);
}

function projectActor(before, candidate, path, type) {
  if (buildActorPath(path, type)) return clone(candidate);
  if (!isObject(before) && !isObject(candidate)) return clone(before);
  const out = {};
  for (const key of new Set([...Object.keys(before ?? {}), ...Object.keys(candidate ?? {})])) {
    const value = projectActor(before?.[key], candidate?.[key], path ? `${path}.${key}` : key, type);
    if (value !== undefined) out[key] = value;
  }
  return Object.keys(out).length || isObject(before) ? out : clone(before);
}

function preserveValue(current, candidate) {
  if (!isObject(current)) return candidate;
  if (!isObject(candidate)) return clone(current);
  if (Object.hasOwn(current, 'value')) candidate.value = clone(current.value);
  return candidate;
}

function preservePrepared(current, candidate, max) {
  if (!current) return candidate;
  const out = candidate ?? (Array.isArray(current) ? [] : {});
  for (const [index, entry] of Object.entries(current)) {
    if (Number.isInteger(max) && Number(index) >= max) continue;
    if (entry?.expended === undefined) continue;
    out[index] ??= {id: null};
    out[index].expended = entry.expended;
  }
  return out;
}

function projectItem(current, candidate) {
  const out = clone(candidate);
  out.effects = clone(current?.effects ?? []);
  if (!current) return out;
  const source = current.system ?? {};
  out.system ??= {};
  if (source.frequency) out.system.frequency = preserveValue(source.frequency, out.system.frequency);
  if (source.location?.uses) {
    out.system.location ??= {};
    out.system.location.uses = preserveValue(source.location.uses, out.system.location.uses);
  }
  if (Object.hasOwn(source, 'containerId')) out.system.containerId = clone(source.containerId);
  if (source.slots) {
    out.system.slots ??= {};
    for (const [key, slot] of Object.entries(source.slots)) {
      // PF2e 预存所有环级的空槽；从零容量解锁时，来源值是新法术位的初始状态。
      if (slot.max === 0 && out.system.slots[key]?.max > 0) continue;
      const next = out.system.slots[key] = preserveValue(slot, out.system.slots[key]);
      if (slot.prepared) next.prepared = preservePrepared(slot.prepared, next.prepared, next.max);
    }
  }
  return out;
}

// 同 id 任何一版属于库存或状态条目，就完整保留主服那份；改 type 不会改变这条边界。
function protectedItemIds(...docs) {
  return new Set(docs.flatMap(doc => (doc.items ?? []).filter(item => !BUILD_TYPES.has(item.type)).map(item => item._id)));
}

/** 将 candidate 的构筑与外观投影到 before，其余内容保留 before；返回独立副本，供草稿摘要和普通套用共用。 */
export function projectBuildChanges(before, candidate, {base = before} = {}) {
  const doc = projectWithProtectedItems(before, candidate, protectedItemIds(base, before, candidate));
  return combineMediaMetadata(doc, [before, candidate, base]);
}

function projectWithProtectedItems(before, candidate, protectedIds) {
  const projected = projectActor(before, candidate, '', before.type ?? candidate.type);
  const incoming = new Map((candidate.items ?? []).filter(item => !protectedIds.has(item._id)).map(item => [item._id, item]));
  projected.items = (before.items ?? []).flatMap(item => {
    if (protectedIds.has(item._id)) return [clone(item)];
    const next = incoming.get(item._id);
    incoming.delete(item._id);
    return next ? [projectItem(item, next)] : [];
  });
  projected.items.push(...[...incoming.values()].map(item => projectItem(null, item)));
  projected.effects = clone(before.effects ?? []);
  return projected;
}

function copyLocalKeys(out, source, keys) {
  for (const key of keys) {
    if (Object.hasOwn(source, key)) out[key] = clone(source[key]);
    else delete out[key];
  }
}

/** 两个方向共用：主服提供库存和状态，构筑仍三方合并；target 只决定本地键和 changed 的比较对象。 */
export function mergeBuildChanges({base, main, char, target = 'main'}) {
  const protectedIds = protectedItemIds(base, main, char);
  const comparisonDocs = mediaComparisonDocs([base, main, char]);
  // 主服已删掉的构筑也用基线消耗值比较，避免建卡服只改剩余次数就把它复活。
  const withDeletedItems = (current, previous) => {
    const ids = new Set((current.items ?? []).map(item => item._id));
    return {...current, items: [...(current.items ?? []), ...(previous.items ?? []).filter(item => !ids.has(item._id))]};
  };
  const stateSource = withDeletedItems(main, base);
  const comparisonState = withDeletedItems(comparisonDocs[1], comparisonDocs[0]);
  const buildOnly = (source, state) => {
    const doc = projectWithProtectedItems(state, source, protectedIds);
    // 图像组另行整体选版，不让 merge3 将新图与另一版的旧配方逐字段拼接。
    copyAppearanceGroups(doc, state);
    doc.items = doc.items.filter(item => !protectedIds.has(item._id));
    doc.effects = [];
    return doc;
  };
  // 三份里的消耗值先统一成主服当前值，既不生成状态冲突，也不会把消耗误认作整条构筑修改。
  const merged = merge3({
    base: buildOnly(base, stateSource), main: buildOnly(main, stateSource), char: buildOnly(char, stateSource), target,
    comparisonDocs: comparisonDocs.map(source => buildOnly(source, comparisonState)),
    actorOwner: path => buildActorPath(path, main.type) ? 'char' : 'main',
  });
  const doc = projectWithProtectedItems(main, merged.doc, protectedIds);
  const appearance = mergeAppearanceGroups({base, main, char, target, comparisonDocs});
  copyAppearanceGroups(doc, appearance.doc);
  const targetDoc = target === 'char' ? char : main;
  copyLocalKeys(doc, targetDoc, LOCAL_ACTOR_KEYS);
  doc.flags ??= {};
  if (targetDoc.flags?.[OWN_FLAG] !== undefined) doc.flags[OWN_FLAG] = clone(targetDoc.flags[OWN_FLAG]);
  else delete doc.flags[OWN_FLAG];
  for (const key of ['items', 'effects']) {
    const current = new Map((targetDoc[key] ?? []).map(item => [item._id, item]));
    for (const item of doc[key]) {
      if (current.has(item._id)) copyLocalKeys(item, current.get(item._id), LOCAL_ITEM_KEYS);
    }
  }
  const conflicts = merged.conflicts.filter(conflict => conflict.scope === 'actor'
    ? buildActorPath(conflict.path, main.type) : conflict.scope === 'item' && !protectedIds.has(conflict.id));
  conflicts.push(...appearance.conflicts);
  combineMediaMetadata(doc, [base, main, char]);
  const changed = canonicalJson(stripVolatile(doc)) !== canonicalJson(stripVolatile(targetDoc));
  return {doc, conflicts, changed};
}
