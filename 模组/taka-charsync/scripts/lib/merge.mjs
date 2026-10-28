import { canonicalJson, stripVolatile } from './canon.mjs';
import {combineMediaMetadata, managedMediaHash, MEDIA_FLAG, mediaComparisonDocs} from './media.mjs';
import {
  LOCAL_ACTOR_KEYS, LOCAL_ITEM_KEYS, OWN_FLAG, ownerOfActorPath, ownerOfItem, ownerOfItemPath,
} from './rules.mjs';

/**
 * @typedef {'main'|'char'} Side
 * @typedef {{scope:'actor'|'item'|'effect', id?:string, name?:string, path:string, winner:Side}} Conflict
 *   `path` 是条目内或角色上的路径；为空串时指整条：一方删了（条目本身或它的授予根）、另一方改了。
 * @typedef {{doc:object, conflicts:Conflict[], changed:boolean}} MergeResult
 */

const PREPARED_PATH = /^system\.slots\.[^.]+\.prepared$/;

const EMBEDDED_RULES = {
  item: {ownerOfEntry: ownerOfItem, ownerOfPath: ownerOfItemPath},
  effect: {ownerOfEntry: () => 'main', ownerOfPath: () => 'main'},
};

const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const otherSide = side => (side === 'main' ? 'char' : 'main');
const byId = docs => new Map(docs.map(doc => [doc._id, doc]));

function omit(doc, keys) {
  const out = {...doc};
  for (const key of keys) delete out[key];
  return out;
}

function pick(doc, keys) {
  const out = {};
  for (const key of keys) if (doc[key] !== undefined) out[key] = doc[key];
  return out;
}

// 比较内容时与内容哈希一致，忽略各层的 _stats、ownership、sort。包一层再交给 stripVolatile，
// 免得它把被比较的这棵子树自己的 folder 键当成角色的文件夹去掉。
const comparable = value => canonicalJson(stripVolatile({value}).value);
const same = (a, b) => comparable(a) === comparable(b);

// 内容相同的旧 URL 可升级为受管路径；真实改图和不同配方数组仍采用已选中的版本。
function retainManagedMedia(selected, other, comparedSelected, comparedOther) {
  if (typeof selected === 'string' && typeof other === 'string') {
    return !managedMediaHash(selected) && managedMediaHash(other) && same(comparedSelected, comparedOther) ? other : selected;
  }
  if (Array.isArray(selected) && Array.isArray(other) && same(comparedSelected, comparedOther)) {
    return selected.map((value, i) => retainManagedMedia(value, other[i], comparedSelected?.[i], comparedOther?.[i]));
  }
  if (isObject(selected) && isObject(other)) {
    return Object.fromEntries(Object.entries(selected).map(([key, value]) => [key,
      retainManagedMedia(value, other[key], comparedSelected?.[key], comparedOther?.[key])]));
  }
  return selected;
}

/**
 * 三方合并一个值：对象逐键递归，数组和标量是叶子。只有一方改了取改的一方；两方改得不同按
 * `ctx.ownerOf(path)` 取，并记进 `ctx.conflicts`。两方一样时取写入方那份，少改它一点。
 */
function mergeValue(base, main, char, path, ctx, compared = [base, main, char]) {
  const [b, m, c] = compared.map(comparable);
  if (m === c) return ctx.target === 'main'
    ? retainManagedMedia(main, char, compared[1], compared[2])
    : retainManagedMedia(char, main, compared[2], compared[1]);
  if (b === m) return retainManagedMedia(char, main, compared[2], compared[1]);
  if (b === c) return retainManagedMedia(main, char, compared[1], compared[2]);
  if (isObject(main) && isObject(char)) {
    const baseObject = isObject(base) ? base : {};
    const out = {};
    for (const key of new Set([...Object.keys(main), ...Object.keys(char)])) {
      const value = mergeValue(baseObject[key], main[key], char[key], path ? `${path}.${key}` : key, ctx,
        compared.map(value => value?.[key]));
      if (value !== undefined) out[key] = value;
    }
    return out;
  }
  if (ctx.preparedSlots && PREPARED_PATH.test(path) && Array.isArray(main) && Array.isArray(char)) {
    return mergePrepared(base, main, char, path, ctx);
  }
  const winner = ctx.ownerOf(path);
  ctx.conflicts.push({path, winner});
  return winner === 'main' ? main : char;
}

// 备法表按下标合并：备哪个法术（id）归建卡服，用没用掉（expended）归主服。
// 该下标三份备的是同一个法术才三方合并 expended，否则跟着结果 id 来源的那一方。
function mergePrepared(base, main, char, path, ctx) {
  const out = [];
  for (let i = 0; i < Math.max(main.length, char.length); i++) {
    const b = Array.isArray(base) ? base[i] : undefined;
    const m = main[i];
    const c = char[i];
    const id = mergeValue(b?.id, m?.id, c?.id, `${path}.${i}.id`, ctx);
    // 一方删掉了这一格、合并结果也是删：整格去掉，不拼出没有 id 的空壳。
    // 不只看 id === undefined：旧数据里两边都在、本来就没有 id 键的格子要留下。
    if (id === undefined && (m == null || c == null)) continue;
    const source = m != null && m.id === id ? m : c;
    const sameSpell = b != null && m != null && c != null && b.id === m.id && m.id === c.id;
    const expended = sameSpell
      ? mergeValue(b.expended, m.expended, c.expended, `${path}.${i}.expended`, ctx)
      : source.expended;
    const slot = {...source, id};
    if (expended === undefined) delete slot.expended;
    else slot.expended = expended;
    out.push(slot);
  }
  return out;
}

/**
 * 按 `_id` 配对合并 items 或 effects，把整条冲突和保留条目内部的冲突追加进 `conflicts`。
 * 两边都有的逐路径合并；只剩一边有的按条目归属定去留；授予树里的条目跟着根走。
 */
function mergeEmbedded(scope, baseList, mainList, charList, target, conflicts, compared) {
  const {ownerOfEntry, ownerOfPath} = EMBEDDED_RULES[scope];
  const bases = byId(baseList);
  const versions = {main: byId(mainList), char: byId(charList)};
  const [comparedBase, comparedMain, comparedChar] = compared.map(byId);

  const entries = new Map();
  for (const id of new Set([...versions[target].keys(), ...versions[otherSide(target)].keys()])) {
    const entry = {base: bases.get(id), main: versions.main.get(id), char: versions.char.get(id), conflicts: [],
      compared: {base: comparedBase.get(id), main: comparedMain.get(id), char: comparedChar.get(id)}};
    if (entry.main && entry.char) {
      const targetVersion = entry[target];
      const ctx = {
        target,
        ownerOf: path => ownerOfPath(targetVersion.type, path),
        conflicts: entry.conflicts,
        preparedSlots: targetVersion.type === 'spellcastingEntry',
      };
      const merged = mergeValue(
        omit(entry.base ?? {}, LOCAL_ITEM_KEYS), omit(entry.main, LOCAL_ITEM_KEYS), omit(entry.char, LOCAL_ITEM_KEYS), '', ctx,
        ['base', 'main', 'char'].map(side => omit(entry.compared[side] ?? {}, LOCAL_ITEM_KEYS)));
      entry.doc = {...merged, ...pick(targetVersion, LOCAL_ITEM_KEYS)};
    } else {
      entry.doc = entry.main ?? entry.char;
    }
    entries.set(id, entry);
  }

  // 去留记成 {keep, by}：by 是「一方删了」时状态被采纳的一方，谈不上删除时为 null。
  function standaloneFate(entry) {
    if (entry.main && entry.char) return {keep: true, by: null};
    const holder = entry.main ? 'main' : 'char';
    const deleter = otherSide(holder);
    if (!entry.base) return {keep: true, by: null};
    if (same(entry.compared.base, entry.compared[holder])) return {keep: false, by: deleter};
    return ownerOfEntry(entry.doc.type) === deleter ? {keep: false, by: deleter} : {keep: true, by: holder};
  }

  const known = new Set([...bases.keys(), ...entries.keys()]);
  const fates = new Map();
  const resolving = new Set();

  function fateOf(id) {
    if (fates.has(id)) return fates.get(id);
    const entry = entries.get(id);
    if (!entry) return {keep: false, by: null}; // 只在 base 里：两边都删了
    if (resolving.has(id)) return standaloneFate(entry); // 授予关系成环，当根处理
    resolving.add(id);
    const fate = inheritedFate(id, entry) ?? standaloneFate(entry);
    resolving.delete(id);
    fates.set(id, fate);
    return fate;
  }

  // 授予方没留下，被授予物跟着去；授予方留下且 itemGrants 里还列着它，跟着留。
  // grantedBy 指向哪一份里都没有的条目时它就是根，自己定去留。
  function inheritedFate(id, entry) {
    const parentId = entry.doc.flags?.pf2e?.grantedBy?.id;
    if (!parentId || parentId === id || !known.has(parentId)) return null;
    const parent = fateOf(parentId);
    if (!parent.keep) return parent;
    const grants = Object.values(entries.get(parentId).doc.flags?.pf2e?.itemGrants ?? {});
    return grants.some(grant => grant?.id === id) ? parent : null;
  }

  // 一方删了、另一方改了算整条冲突：留下时看留的一方改没改，删掉时看被删的那份改没改。
  function overridesEdit(entry, fate) {
    if (!fate.by) return false;
    const loser = entry.compared[otherSide(fate.by)];
    const winner = entry.compared[fate.by];
    if (fate.keep) return Boolean(entry.compared.base && winner) && !loser && !same(entry.compared.base, winner);
    return Boolean(loser) && !same(entry.compared.base, loser);
  }

  const out = [];
  for (const [id, entry] of entries) {
    const fate = fateOf(id);
    const name = entry.doc.name;
    if (fate.keep) {
      out.push(entry.doc);
      for (const conflict of entry.conflicts) conflicts.push({scope, id, name, ...conflict});
    }
    if (overridesEdit(entry, fate)) conflicts.push({scope, id, name, path: '', winner: fate.by});
  }
  return out;
}

/**
 * 三方合并两边的角色卡。四份都是经数据模型清洗过的 actor 源（带 `items`、`effects` 数组），
 * `target` 是要写入的一方：`LOCAL_ACTOR_KEYS`、`flags[OWN_FLAG]` 和已有条目的 `LOCAL_ITEM_KEYS` 取它那份。
 * `actorOwner` 可为特定同步范围指定 Actor 字段冲突归属；省略时使用通用规则。
 * `comparisonDocs` 供构筑投影复用原始三份文档的媒体比较表，顺序为 base、main、char；不会写回。
 * @param {{base:object, main:object, char:object, target:'main'|'char', actorOwner?:(path:string) => Side, comparisonDocs?:object[]}} input
 * @returns {MergeResult} `doc` 不与入参共享引用；`changed` 表示 `doc` 与 `target` 那份内容不同（忽略易变字段）。
 */
export function merge3({base, main, char, target, actorOwner = ownerOfActorPath,
  comparisonDocs = mediaComparisonDocs([base, main, char])}) {
  const targetDoc = target === 'main' ? main : char;
  const content = doc => {
    const out = omit(doc, [...LOCAL_ACTOR_KEYS, 'items', 'effects']);
    if (out.flags) out.flags = omit(out.flags, [OWN_FLAG, MEDIA_FLAG]);
    return out;
  };

  const actorConflicts = [];
  const merged = mergeValue(content(base), content(main), content(char), '', {
    target, ownerOf: actorOwner, conflicts: actorConflicts,
  }, comparisonDocs.map(content));
  const conflicts = actorConflicts.map(conflict => ({scope: 'actor', ...conflict}));
  const doc = {
    ...merged,
    ...pick(targetDoc, LOCAL_ACTOR_KEYS),
    items: mergeEmbedded('item', base.items, main.items, char.items, target, conflicts, comparisonDocs.map(doc => doc.items)),
    effects: mergeEmbedded('effect', base.effects, main.effects, char.effects, target, conflicts, comparisonDocs.map(doc => doc.effects)),
  };
  if (targetDoc.flags?.[OWN_FLAG] !== undefined) doc.flags = {...doc.flags, [OWN_FLAG]: targetDoc.flags[OWN_FLAG]};

  const detached = structuredClone(doc);
  combineMediaMetadata(detached, [base, main, char]);
  const changed = canonicalJson(stripVolatile(detached)) !== canonicalJson(stripVolatile(targetDoc));
  return {doc: detached, conflicts, changed};
}

/** 两版之间的变动：等级、增删改的条目（按 `_id` 配对，忽略易变字段）、技能熟练等级。 */
export function summarize(before, after, {comparisonDocs = mediaComparisonDocs([before, after])} = {}) {
  // 魔宠等非角色卡的源数据里不一定有等级和技能；没记录的技能就是未受训。
  const levelOf = doc => doc.system?.details?.level?.value;
  const rankOf = (doc, skill) => doc.system?.skills?.[skill]?.rank ?? 0;
  const label = entry => ({type: entry.type, name: entry.name});
  const [comparedBefore, comparedAfter] = comparisonDocs;
  const beforeItems = byId(comparedBefore.items);
  const afterItems = byId(comparedAfter.items);
  const skills = new Set([...Object.keys(before.system?.skills ?? {}), ...Object.keys(after.system?.skills ?? {})]);
  return {
    level: levelOf(before) === levelOf(after) ? null : [levelOf(before), levelOf(after)],
    added: after.items.filter(entry => !beforeItems.has(entry._id)).map(label),
    removed: before.items.filter(entry => !afterItems.has(entry._id)).map(label),
    changed: after.items
      .filter(entry => beforeItems.has(entry._id) && !same(beforeItems.get(entry._id), afterItems.get(entry._id)))
      .map(label),
    skillRanks: [...skills]
      .filter(skill => rankOf(before, skill) !== rankOf(after, skill))
      .map(skill => ({skill, from: rankOf(before, skill), to: rankOf(after, skill)})),
  };
}

// 标签取自 pf2_cn 的 TYPES.Item.*、PF2E.Skill*、PF2E.ProficiencyLevel*。
const ITEM_TYPE_LABELS = {
  action: '能力', affliction: '苦难', ammo: '弹药', ancestry: '族裔', armor: '护甲', background: '背景',
  backpack: '容器', book: '书', campaignFeature: '战役特性', class: '职业', condition: '状态', consumable: '消耗品',
  deity: '神祇', effect: '效果', equipment: '装备', feat: '专长/特性', heritage: '传承', kit: '工具包',
  lore: '学识技能', melee: 'NPC攻击', shield: '盾牌', spell: '法术', spellcastingEntry: '法术列表', treasure: '宝物',
  weapon: '武器',
};
const SKILL_LABELS = {
  acrobatics: '特技', arcana: '奥法', athletics: '运动', crafting: '手艺', deception: '欺骗', diplomacy: '交涉',
  intimidation: '威吓', medicine: '医疗', nature: '自然', occultism: '神秘', performance: '表演', religion: '宗教',
  society: '社群', stealth: '隐秘', survival: '生存', thievery: '贼活',
};
const RANK_LABELS = ['未受训', '受训', '专家', '大师', '传奇'];
const SIDE_LABELS = {main: '主服', char: '建卡服'};

const HTML_ESCAPES = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
const escapeHtml = text => String(text).replace(/[&<>"']/g, ch => HTML_ESCAPES[ch]);

const describeEntries = entries =>
  entries.map(({type, name}) => `${escapeHtml(ITEM_TYPE_LABELS[type] ?? type)}「${escapeHtml(name)}」`).join('、');

function describeConflict({scope, name, path, winner}) {
  const side = SIDE_LABELS[winner];
  if (scope === 'actor') return `冲突：<code>${escapeHtml(path)}</code> 两边改得不同，取${side}的值`;
  if (path === '') return `冲突：「${escapeHtml(name)}」一边删了、一边改了，以${side}为准`;
  return `冲突：「${escapeHtml(name)}」的 <code>${escapeHtml(path)}</code> 两边改得不同，取${side}的值`;
}

/** 把 `summarize` 的结果和合并冲突写成简体中文 HTML 列表；都没有时返回空串。 */
export function formatSummary(summary, conflicts) {
  const lines = [];
  if (summary.level) lines.push(`等级 ${summary.level[0]} → ${summary.level[1]}`);
  if (summary.added.length) lines.push(`新增：${describeEntries(summary.added)}`);
  if (summary.removed.length) lines.push(`删除：${describeEntries(summary.removed)}`);
  if (summary.changed.length) lines.push(`修改：${describeEntries(summary.changed)}`);
  for (const {skill, from, to} of summary.skillRanks) {
    lines.push(`${escapeHtml(SKILL_LABELS[skill] ?? skill)}：${RANK_LABELS[from] ?? from} → ${RANK_LABELS[to] ?? to}`);
  }
  for (const conflict of conflicts) lines.push(describeConflict(conflict));
  return lines.length ? `<ul>${lines.map(line => `<li>${line}</li>`).join('')}</ul>` : '';
}
