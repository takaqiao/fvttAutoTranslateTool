// 建卡服：卡主的工作卡跟上主服最新版本（三方合并），主服直接改过等级时先另存草稿备份再重置成主服版本。
import { normalizeCopy, planCatchUp, refKey, sameRef, syncAllowed } from '../lib/chain.mjs';
import { summarize } from '../lib/merge.mjs';
import { mergeBuildChanges } from '../lib/build-sync.mjs';
import { canonicalJson, stripVolatile } from '../lib/canon.mjs';
import { OWN_FLAG } from '../lib/rules.mjs';
import { formatTime } from '../lib/time.mjs';
import { cleanDoc, importFull } from '../foundry/import.mjs';
import { snapshotActorMedia } from '../foundry/media.mjs';
import { getHistoryDoc, getMainLatest, getMeta } from '../foundry/published.mjs';
import { withdrawUnapplied } from './submit.mjs';

/** 文档的等级；魔宠这类没有自己等级的为 null。和引擎取法一致。 */
export const levelOf = doc => doc.system?.details?.level?.value ?? null;

/**
 * 当前用户是不是映射表里这张卡的卡主（引擎按 ownership 认出的非 GM 所有者，同伴各有各的）。
 * GM 一律不算：Foundry 建卡时给创建者补显式 OWNER，GM 建的卡上 GM 也是所有者；让 GM 的客户端跟上，
 * 会改动玩家的卡，重置提示和草稿备份也落到 GM 那边。
 */
export const ownsCard = (actor, meta) => !game.user.isGM && meta?.cards?.[actor.id]?.charOwner === game.user.id;

/** 魔宠在建卡服的主人卡 id；幻灵是 character 类型，没有 `system.master`，为 null。 */
export const masterIdOf = (actor, card) => (actor.type === 'familiar' ? card.companionOf : null);

/** 主服文档换成建卡服这一侧：魔宠的 `system.master.id` 指向建卡服的主人卡。返回新对象。 */
export function onCharSide(doc, masterId) {
  if (doc.type !== 'familiar' || !masterId) return doc;
  const out = foundry.utils.deepClone(doc);
  out.system.master = {...out.system.master, id: masterId};
  return out;
}

/**
 * 某个版本引用对应的文档，已换成本卡在建卡服这一侧的样子，附上 `planCatchUp` 要查的三样：
 * 等级、它代表的套用状态（主服版本是它当时的 `applied`，副本和 `char` 版本是它自己）、它属于哪张工作卡。
 * 历史文件还没发布或已被清掉时为 null。
 */
async function loadVersion(ref, {actorId, masterId, mainLatest}) {
  if (ref.kind === 'main' && ref.hash === mainLatest.hash) {
    return {doc: onCharSide(mainLatest.doc, masterId), level: mainLatest.level, applied: mainLatest.applied, cardId: null};
  }
  const raw = await getHistoryDoc(ref);
  if (!raw) return null;
  switch (ref.kind) {
    case 'main':
      return {doc: onCharSide(raw, masterId), level: levelOf(raw), applied: raw.flags?.[OWN_FLAG]?.applied?.ref ?? null, cardId: null};
    case 'copy':
      return {doc: normalizeCopy(raw, {actorId, masterId}), level: levelOf(raw), applied: ref, cardId: raw.flags[OWN_FLAG].copy.cardId};
    case 'char':
      return {doc: raw, level: levelOf(raw), applied: ref, cardId: ref.cardId};
  }
  throw new Error(`未知的版本引用类型：${ref.kind}`);
}

function ownFlagsAfter(own, base, conflicts, mainHash) {
  const next = {...foundry.utils.deepClone(own), base};
  delete next.conflicts;
  if (conflicts.length) next.conflicts = {at: Date.now(), mainHash, list: conflicts};
  return next;
}

// 只换本模组的记号。整个 flags 一起写、不递归合并：递归合并会把旧 base 里新 base 没有的键留下来
function writeOwnFlags(actor, own) {
  const flags = {...foundry.utils.deepClone(actor._source.flags), [OWN_FLAG]: own};
  return actor.update({flags}, {recursive: false, takaCharsync: true});
}

/** 建卡服上带 `role` 记号的角色文件夹 id。找不到时提示一句，返回 null（放在根目录）。 */
export function roleFolder(role) {
  const folder = game.folders.find(entry => entry.type === 'Actor' && entry.getFlag(OWN_FLAG, 'role') === role);
  if (!folder) console.warn(`taka-charsync | 找不到 role 为 ${role} 的角色文件夹，放在根目录`);
  return folder?.id ?? null;
}

async function createDraftBackup(actor) {
  const data = actor.toObject();
  delete data._id;
  data.name = `${actor.name}（草稿备份 ${formatTime(Date.now())}）`;
  data.folder = roleFolder('backup');
  data.flags = {...data.flags, [OWN_FLAG]: {backupOf: actor.id}};
  const backup = await Actor.create(data, {takaCharsync: true});
  if (!backup) throw new Error(`taka-charsync：没能建出 ${actor.name} 的草稿备份，卡没有动`);
  return backup;
}

/**
 * 先把卡另存成草稿备份，再整卡导入主服最新版本。备份建不出来就不动卡。
 * `withdrawCopies`：主服直接改过等级时为 true，最后撤回本卡主服没套用的副本；放弃草稿不动已提交的副本。
 */
async function resetToMain(actor, {own, mainLatest, masterId, withdrawCopies}) {
  const backup = await createDraftBackup(actor);
  const flags = ownFlagsAfter(own, {kind: 'main', hash: mainLatest.hash}, [], mainLatest.hash);
  const imported = await importFull(actor, onCharSide(mainLatest.doc, masterId), {flags});
  if (withdrawCopies) await withdrawUnapplied(actor, mainLatest.applied);
  return {action: 'reset', backupId: backup.id, backupName: backup.name, ...imported, readAt: mainLatest.readAt};
}

async function mergeFromMain(actor, {own, plan, baseDoc, mainLatest, masterId, sameVersion = false}) {
  const base = cleanDoc(baseDoc);
  const main = cleanDoc(onCharSide(mainLatest.doc, masterId));
  const captured = actor.toObject();
  const char = cleanDoc(captured);
  const {docs: [frozenChar]} = await snapshotActorMedia([char]);
  const {doc, conflicts} = mergeBuildChanges({base, main, char: frozenChar, target: 'char'});
  const changed = canonicalJson(stripVolatile(doc)) !== canonicalJson(stripVolatile(char));
  if (canonicalJson(actor.toObject()) !== canonicalJson(captured)) {
    throw new Error('工作卡在读取媒体期间已有修改，请重新跟上主服');
  }
  // 同版本只是校正游戏状态，不清掉上次跟随后尚未处理的构筑冲突。
  const flags = sameVersion ? {...own, base: plan.newBase} : ownFlagsAfter(own, plan.newBase, conflicts, mainLatest.hash);
  let imported = {skipped: [], removed: []};
  if (changed) imported = await importFull(actor, doc, {flags});
  else if (canonicalJson(flags) !== canonicalJson(own)) await writeOwnFlags(actor, flags);
  if (sameVersion && !changed) return {action: 'none'};
  return {action: 'merge', changed, summary: summarize(char, doc), conflicts, ...imported, readAt: mainLatest.readAt};
}

/**
 * 让工作卡跟上主服最新版本，按 `planCatchUp` 的结论执行：
 * - `merge`：基线、主服最新版、卡三份清洗后三方合并，内容有变才整卡导入；`base` 记成主服最新版，
 *   合并冲突记进 `flags.taka-charsync.conflicts`（`{at, mainHash, list}`），没有冲突就清掉。
 * - `reset`：主服直接改过等级。先另存草稿备份，再导入主服最新版，撤回本卡主服没套用的副本。
 * - `none`：同版本也校正库存与游戏状态，保留构筑草稿；内容无变化不导入，缺 `base` 时补写。
 * - `wait`：主服版本或基线文档还没发布；`paused`：两边规则版本不一致；`unmapped`：没有对应的主服卡。都不动卡。
 *
 * @returns {Promise<{action: 'none'|'merge'|'reset'|'wait'|'paused'|'unmapped', changed?: boolean, summary?: object,
 *   conflicts?: object[], skipped?: object[], removed?: object[], backupId?: string, backupName?: string, readAt?: number}>}
 *   `readAt` 是主服最新版的读取时刻；`skipped`、`removed` 同 `importFull`。
 */
export async function catchUpCard(actor, {meta}) {
  const card = meta.cards?.[actor.id];
  if (!card) return {action: 'unmapped'};
  if (!syncAllowed(meta, card.world)) return {action: 'paused'};
  const mainLatest = await getMainLatest(card.world, card.mainId);
  if (!mainLatest) return {action: 'wait'};

  const own = actor._source.flags?.[OWN_FLAG] ?? {};
  const base = own.base ?? null;
  const bootstrap = meta.bootstrap?.[actor.id] ?? null;
  const masterId = masterIdOf(actor, card);
  const versions = new Map();
  const load = async ref => {
    const version = await loadVersion(ref, {actorId: actor.id, masterId, mainLatest});
    if (version) versions.set(refKey(ref), version);
    return version;
  };

  // 只取 planCatchUp 会查到的版本：现有基线；主服套用了、基线还没吸收的本卡副本或建卡服版本。
  // 别的卡的副本要读了文档才知道归属，所以副本一律读
  const current = base ?? bootstrap;
  if (current && !(await load(current))) return {action: 'wait'};
  const {applied} = mainLatest;
  const absorbed = current && sameRef(applied, versions.get(refKey(current)).applied);
  const appliedHere = applied?.kind === 'copy' || (applied?.kind === 'char' && applied.cardId === actor.id);
  if (appliedHere && !absorbed && !(await load(applied))) return {action: 'wait'};

  const plan = planCatchUp({
    cardId: actor.id, mapped: true, base, bootstrap, mainLatest,
    levelOf: ref => versions.get(refKey(ref))?.level,
    copyCard: ref => versions.get(refKey(ref))?.cardId ?? null,
    appliedAt: ref => versions.get(refKey(ref))?.applied ?? null,
  });
  switch (plan.action) {
    case 'none':
      return mergeFromMain(actor, {own, plan, baseDoc: onCharSide(mainLatest.doc, masterId), mainLatest, masterId, sameVersion: true});
    case 'merge':
      return mergeFromMain(actor, {own, plan, baseDoc: versions.get(refKey(plan.mergeBase)).doc, mainLatest, masterId});
    case 'reset':
      return resetToMain(actor, {own, mainLatest, masterId, withdrawCopies: true});
    default:
      return {action: plan.action};
  }
}

/**
 * 放弃草稿：先另存草稿备份，再恢复成主服最新版本；已提交的副本不动，要收回在副本上点撤回。
 * 返回值同 `catchUpCard` 的 `reset`。当前用户不是卡主（含 GM）时不动卡，返回 null；暂停同步或还没有主服版本时抛错。
 */
export async function discardDraft(actor, {meta}) {
  if (!ownsCard(actor, meta)) {
    console.warn(`taka-charsync | ${actor.name} 的卡主不是当前用户，没有放弃草稿`);
    return null;
  }
  const card = meta.cards[actor.id];
  if (!syncAllowed(meta, card.world)) throw new Error('两边规则版本不一致，同步暂停中');
  const mainLatest = await getMainLatest(card.world, card.mainId);
  if (!mainLatest) throw new Error('还没读到这张卡的主服版本');
  return resetToMain(actor, {
    own: actor._source.flags?.[OWN_FLAG] ?? {}, mainLatest, masterId: masterIdOf(actor, card), withdrawCopies: false,
  });
}

function notifyDropped(name, {skipped = [], removed = []}) {
  const names = entries => entries.map(entry => `「${foundry.utils.escapeHTML(entry.name || entry.id)}」`).join('、');
  const parts = [];
  if (skipped.length) parts.push(`${skipped.length} 个条目过不了校验没有套用：${names(skipped)}`);
  if (removed.length) parts.push(`卡上 ${removed.length} 个过不了校验的旧条目已移除：${names(removed)}`);
  if (parts.length) ui.notifications.warn(`${name}：${parts.join('；')}`, {permanent: true});
}

function notifyResult(actor, result) {
  const name = foundry.utils.escapeHTML(actor.name);
  const time = formatTime(result.readAt);
  const backupName = foundry.utils.escapeHTML(result.backupName ?? '');
  const conflicts = result.conflicts?.length ?? 0;
  if (result.action === 'merge' && (result.changed || conflicts)) {
    const conflictNote = conflicts ? `，有 ${conflicts} 处两边都改了，打开卡面看详情` : '';
    ui.notifications.info(`${name}已跟上主服 ${time} 的版本${conflictNote}`);
  } else if (result.action === 'reset') {
    ui.notifications.warn(`${name}的等级在主服被直接改过，已恢复成主服 ${time} 的版本；原来的草稿另存为「${backupName}」`, {permanent: true});
  } else if (result.action === 'discard') {
    ui.notifications.info(`已放弃${name}的草稿，恢复成主服 ${time} 的版本；草稿另存为「${backupName}」`);
  }
  notifyDropped(name, result);
}

/** 某张卡的某个操作失败：记进控制台并弹错误提示。 */
export function notifyFailure(actor, what, error) {
  console.error(`taka-charsync | ${actor.name} ${what}失败`, error);
  const message = error instanceof Error ? error.message : String(error);
  ui.notifications.error(`${foundry.utils.escapeHTML(actor.name)}${what}失败：${foundry.utils.escapeHTML(message)}`);
}

// 登录和打开卡面可能同时跟上同一张卡，后来的等前一次跑完，不重复导入
const running = new Map();

/** 跟上主服并按结果提示卡主；出错时提示，不抛出。同一张卡同时只跑一次。 */
export function catchUpWithNotice(actor, meta) {
  let job = running.get(actor.id);
  if (!job) {
    job = catchUpCard(actor, {meta})
      .then(result => notifyResult(actor, result), error => notifyFailure(actor, '跟上主服', error))
      .finally(() => running.delete(actor.id));
    running.set(actor.id, job);
  }
  return job;
}

/** 放弃草稿并提示；出错时提示，不抛出。 */
export async function discardWithNotice(actor) {
  try {
    const result = await discardDraft(actor, {meta: await getMeta()});
    if (result) notifyResult(actor, {...result, action: 'discard'});
  } catch (error) {
    notifyFailure(actor, '放弃草稿', error);
  }
}

/** 登录时：当前用户是卡主的卡（含同伴）逐张跟上主服。 */
export async function catchUpOnLogin({meta = null, canPrepare = () => true} = {}) {
  meta ??= await getMeta();
  if (!meta) return;
  for (const actor of game.actors) {
    if (canPrepare(actor) && ownsCard(actor, meta)) await catchUpWithNotice(actor, meta);
  }
}
