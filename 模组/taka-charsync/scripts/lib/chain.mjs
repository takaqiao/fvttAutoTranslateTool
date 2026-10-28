import { OWN_FLAG } from './rules.mjs';

/**
 * 卡的一个版本：主服版本按内容哈希；冻结副本按 id 和提交时刻；`char` 是手动同步时 GM 直接拉的建卡服版本。
 * @typedef {{kind:'main', hash:string} | {kind:'copy', copyId:string, rev:number} | {kind:'char', cardId:string, hash:string}} Ref
 */
/**
 * 冻结副本的记号（`flags['taka-charsync'].copy`）。`rev` 是提交时的 `Date.now()`；`name` 是工作卡原名；
 * 同伴的副本 `companionOf` 是主人副本的 copyId，主人自己的为 null。
 * @typedef {{
 *   copyId:string, rev:number, cardId:string, level:number,
 *   kind:'level'|'new'|'force', pred:Ref|null, targetWorld:string, replaces:string|null,
 *   withdrawn:boolean, name:string, submittedBy:string, companionOf:string|null,
 *   companions:{copyId:string, rev:number, sourceId:string}[]
 * }} CopyMeta
 */

const ID_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

const copyRef = copy => ({kind: 'copy', copyId: copy.copyId, rev: copy.rev});

/** 引用的字符串形式，`undone`、`published` 这些列表里存的就是它；null 原样返回。 */
export function refKey(ref) {
  if (!ref) return null;
  switch (ref.kind) {
    case 'main': return `main:${ref.hash}`;
    case 'copy': return `copy:${ref.copyId}:${ref.rev}`;
    case 'char': return `char:${ref.cardId}:${ref.hash}`;
  }
  throw new Error(`未知的版本引用类型：${ref.kind}`);
}

/** 两个都为 null 也算相同。 */
export const sameRef = (a, b) => refKey(a) === refKey(b);

/** 发布目录里存这一版原始文档的相对路径。 */
export function historyFile(ref) {
  switch (ref.kind) {
    case 'main': return `history/main-${ref.hash}.json`;
    case 'copy': return `history/copy-${ref.copyId}-${ref.rev}.json`;
    case 'char': return `history/char-${ref.hash}.json`;
  }
  throw new Error(`未知的版本引用类型：${ref.kind}`);
}

// Foundry 风格的 16 位 id。由来源和等级决定，两个标签页同时提交同一级会落到同一个 id 上，只出一份副本。
// 浏览器端与 contentHash 一样要求安全上下文。
async function idFromSeed(seed) {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(seed));
  return Array.from(new Uint8Array(digest, 0, 16), byte => ID_ALPHABET[byte % 62]).join('');
}

/** 某张卡某一级的冻结副本 id。 */
export const copyIdFor = (sourceActorId, level) => idFromSeed(`taka-charsync|${sourceActorId}|${level}`);

/** 某张卡的整卡副本 id，每张卡只有一份。 */
export const forceCopyIdFor = cardId => idFromSeed(`taka-charsync|${cardId}|force`);

/** 经验拉满即放行升一级。`max` 为 0 的卡不算满。 */
export const isXpFull = xp => xp.max > 0 && xp.value >= xp.max;

/** 升 `levels` 级后剩下的经验。 */
export const xpAfter = (xp, levels) => Math.max(0, xp.value - xp.max * levels);

/**
 * 建卡服工作卡怎样跟上主服最新版本。
 *
 * `levelOf(ref)` 给出那一版的等级；`copyCard(ref)` 给出副本或 `char` 版本属于哪张工作卡，主服版本为 null；
 * `appliedAt(ref)` 给出那一版代表的套用状态：主服版本是它当时的 `applied`，副本和 `char` 版本是它自己。
 *
 * 主服套用了本卡的副本或建卡服版本、而基线还没吸收它时，基线快进到那一版，套用带来的等级变化不触发重置。
 * 除此之外等级对不上，说明 GM 在团里直接升过级：PF2e 升级时会用新 id 生成职业特性，草稿里也有一份，
 * 合并就会重复，所以只能重置。
 *
 * @returns {{action:'none'|'merge'|'reset'|'unmapped'|'wait', mergeBase:Ref|null, newBase:Ref|null}}
 *   `none` 时 `newBase` 是现有基线，供补写卡上缺失的 base 记号。
 */
export function planCatchUp({cardId, mapped, base, bootstrap, mainLatest, levelOf, copyCard, appliedAt}) {
  if (!mainLatest) return {action: 'wait', mergeBase: null, newBase: null};
  let effective = base ?? bootstrap ?? null;
  const {applied} = mainLatest;
  if (applied && (applied.kind === 'copy' || applied.kind === 'char') && copyCard(applied) === cardId
      && !sameRef(applied, effective ? appliedAt(effective) : null)) {
    effective = applied;
  }
  if (!effective) return {action: mapped ? 'wait' : 'unmapped', mergeBase: null, newBase: null};
  if (effective.kind === 'main' && effective.hash === mainLatest.hash) {
    return {action: 'none', mergeBase: null, newBase: effective};
  }
  const latest = {kind: 'main', hash: mainLatest.hash};
  if (levelOf(effective) !== mainLatest.level) return {action: 'reset', mergeBase: null, newBase: latest};
  return {action: 'merge', mergeBase: effective, newBase: latest};
}

/**
 * 副本的前驱是否正是主服现在的状态。前驱是副本时，主服最后套用的必须就是它；前驱是主服版本时，
 * 那一版之后主服不能再套用过别的。只有新卡可以没有前驱。
 */
export function predOk(copy, mainApplied, appliedAt) {
  const {pred} = copy;
  if (!pred) return copy.kind === 'new';
  if (pred.kind === 'main') return sameRef(appliedAt(pred), mainApplied);
  return sameRef(pred, mainApplied);
}

/**
 * 主服这张卡下一步套用哪份升级副本、能不能自动套。只看 `kind:'level'` 的副本，新卡和整卡副本
 * 分别交给 `planNew`、`planForce`。
 *
 * `stale`：GM 在团里直接升过级，副本谱系失效，再套会重复职业特性，面板只给「忽略此副本」（记进 `undone`）。
 *
 * @returns {{action:'auto'|'gm'|'wait'|'none', copy:CopyMeta|null,
 *   reason:'ready'|'session'|'same-level'|'pred'|'stale'|'gap'|'xp'|'none'}}
 */
export function planApply({main, copies, appliedAt, levelOf, autoAllowed: allowAuto}) {
  const undone = new Set(main.undone);
  const copy = copies
    .filter(candidate => candidate.kind === 'level'
      && !candidate.withdrawn
      && !undone.has(refKey(copyRef(candidate)))
      && candidate.level >= main.level
      && !sameRef(copyRef(candidate), main.applied))
    .sort((a, b) => a.level - b.level)[0];
  if (!copy) return {action: 'none', copy: null, reason: 'none'};
  const plan = (action, reason) => ({action, copy, reason});
  if (!predOk(copy, main.applied, appliedAt)) return plan('gm', 'pred');
  if (levelOf(copy.pred) !== main.level) return plan('gm', 'stale');
  if (copy.level === main.level) return plan('gm', 'same-level');
  if (copy.level > main.level + 1) return plan('gm', 'gap');
  if (!isXpFull(main.xp)) return plan('wait', 'xp');
  if (!allowAuto) return plan('gm', 'session');
  return plan('auto', 'ready');
}

/**
 * 把冻结副本的文档还原成它代表的那张卡，返回深拷贝：id 换成写入那一侧的卡（主服套用时是主服卡 id，
 * 建卡服快进时是工作卡 id），名字换回原名，去掉副本记号；魔宠的 `system.master.id` 换成那一侧的主人。
 * 副本文档不论当合并的 base 还是 char，进合并前都先过这里。
 *
 * 提交时只给 actor 名加后缀、不动 token 名，所以 token 名只在和冻结后的 actor 名相同时才换回，
 * 玩家另起的 token 昵称保留。
 */
export function normalizeCopy(copyDoc, {actorId, masterId}) {
  const doc = structuredClone(copyDoc);
  const flags = doc.flags[OWN_FLAG];
  doc._id = actorId;
  if (doc.prototypeToken?.name === doc.name) doc.prototypeToken.name = flags.copy.name;
  doc.name = flags.copy.name;
  delete flags.copy;
  if (doc.type === 'familiar' && masterId) doc.system.master.id = masterId;
  return doc;
}

/** 要提给本世界 GM 的新卡副本：目标是本世界、未撤回、没被忽略、建卡服那张还没对应到主服卡。 */
export function planNew({world, copies, mappedCardIds, undone}) {
  return copies.filter(copy => copy.kind === 'new'
    && copy.targetWorld === world
    && !copy.withdrawn
    && !undone.has(refKey(copyRef(copy)))
    && !mappedCardIds.has(copy.cardId));
}

/** 等 GM 套用的整卡副本：未撤回、没被忽略、也不是主服已经套用的那份；没有为 null。 */
export function planForce({copies, undone, mainApplied}) {
  return copies.find(copy => copy.kind === 'force'
    && !copy.withdrawn
    && !undone.has(refKey(copyRef(copy)))
    && !sameRef(copyRef(copy), mainApplied)) ?? null;
}

/** 没有玩家在线、也没有已开始的战斗时才自动套用。 */
export const autoAllowed = ({playersOnline, combatStarted}) => playersOnline === 0 && !combatStarted;

/** 团世界和建卡服的规则版本一致才同步；任一方还没有记录（缺失或为 null）也不同步。 */
export function syncAllowed(meta, world) {
  const worldSchema = meta.worlds?.[world]?.schema;
  return worldSchema != null && worldSchema === meta.char?.schema;
}
