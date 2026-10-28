// 建卡服：提交冻结副本（升级、新卡、整卡同步）、撤回、副本锁和 GM 清理。
import { copyIdFor, forceCopyIdFor, refKey, sameRef } from '../lib/chain.mjs';
import { OWN_FLAG } from '../lib/rules.mjs';
import { collectModuleUsage } from '../lib/module-usage.mjs';
import { importFull } from '../foundry/import.mjs';
import { snapshotActorMedia } from '../foundry/media.mjs';
import { getHistoryDoc, getMainLatest, getMeta } from '../foundry/published.mjs';
import { levelOf, ownsCard, roleFolder } from './catch-up.mjs';
import { prepareSubmissionRules } from './rules-session.mjs';

const LOCK_NOTICE = '已提交的副本不能修改，要改请改原卡后重新提交';
const LOCK_NOTICE_INTERVAL_MS = 10_000;
const CLEANUP = '清理已处理的副本';

// 钩子名 → 选项在参数里的位置：preCreate/preUpdate 是 (文档, 数据, 选项, 用户)，preDelete 是 (文档, 选项, 用户)
const LOCKED_HOOKS = {
  preUpdateActor: 2,
  preCreateItem: 2,
  preUpdateItem: 2,
  preDeleteItem: 1,
  preCreateActiveEffect: 2,
  preUpdateActiveEffect: 2,
  preDeleteActiveEffect: 1,
};

const NAME_SUFFIX = {
  level: level => `〔已提交 ${level} 级〕`,
  new: level => `〔已提交 ${level} 级〕`,
  force: () => '〔整卡同步〕',
};

const copyRef = copy => ({kind: 'copy', copyId: copy.copyId, rev: copy.rev});

/** actor 上的副本记号 `CopyMeta`；不是冻结副本时为 null。 */
export const copyOf = actor => actor._source.flags?.[OWN_FLAG]?.copy ?? null;

/** 某张卡在建卡服上当前用户看得到的全部冻结副本（含已撤回的）。 */
export const copyActorsOf = cardId => game.actors.filter(actor => copyOf(actor)?.cardId === cardId);

async function loadMeta() {
  const meta = await getMeta();
  if (!meta) throw new Error('读不到同步数据（meta.json），引擎可能还没发布过');
  return meta;
}

/**
 * 还没对应主服卡的新卡，当前用户是不是卡主：和引擎认卡主一样，`ownership` 里明确写成所有者的非 GM 用户。
 * 只认角色卡，副本和草稿备份不算。
 */
export function ownsNewCard(actor, meta) {
  if (game.user.isGM || meta?.cards?.[actor.id] || actor.type !== 'character') return false;
  const own = actor._source.flags?.[OWN_FLAG] ?? {};
  return !own.copy && !own.backupOf && actor._source.ownership[game.user.id] === CONST.DOCUMENT_OWNERSHIP_LEVELS.OWNER;
}

/** 当前用户能不能撤回这份副本：只有提交者本人（GM 不行），同伴的副本随主人撤回。 */
export const canWithdraw = copy => !game.user.isGM && !copy.companionOf && copy.submittedBy === game.user.id;

/** 作为新卡能提交到的团：当前用户在那个团有对应账号的世界 id。 */
export const newCardWorlds = meta => Object.keys(meta?.users?.[game.user.id] ?? {});

/** 新卡在某个团能替换的卡：映射里那个团、卡主是当前用户的主卡（同伴不算）。 */
export const replaceableCards = (meta, world) => Object.entries(meta.cards ?? {})
  .filter(([, card]) => card.world === world && card.charOwner === game.user.id && !card.companionOf)
  .map(([cardId]) => cardId);

// 卡上的基线，还没有时用起步基线
const baseOf = (actor, meta) => actor._source.flags?.[OWN_FLAG]?.base ?? meta.bootstrap?.[actor.id] ?? null;

// 随主人一起冻结的同伴：映射里记在主人名下的（幻灵、魔宠），加上建卡服上 system.master.id 指向主人的魔宠
export function companionsOf(actor, meta) {
  const ids = new Set(Object.keys(meta.cards ?? {}).filter(id => meta.cards[id].companionOf === actor.id));
  for (const other of game.actors.values()) {
    const own = other._source.flags?.[OWN_FLAG] ?? {};
    if (other.type === 'familiar' && other._source.system.master?.id === actor.id && !own.copy && !own.backupOf) ids.add(other.id);
  }
  return [...ids].flatMap(id => {
    const companion = game.actors.get(id);
    if (!companion) console.warn(`taka-charsync | 找不到 ${actor.name} 的同伴 ${id}，没有一起提交`);
    return companion ? [companion] : [];
  });
}

// 映射里当前用户是卡主的主卡才能提交升级和整卡同步。不是卡主（含 GM）时为 null；同伴随主人提交
function cardToSubmit(actor, meta, what) {
  if (!ownsCard(actor, meta)) {
    console.warn(`taka-charsync | ${actor.name} 的卡主不是当前用户，没有${what}`);
    return null;
  }
  const card = meta.cards[actor.id];
  if (card.companionOf) throw new Error(`${actor.name}是同伴，随主人一起提交`);
  return card;
}

async function levelOfRef(ref, card) {
  if (ref.kind === 'main') {
    const latest = await getMainLatest(card.world, card.mainId);
    if (latest?.hash === ref.hash) return latest.level;
  }
  const doc = await getHistoryDoc(ref);
  if (!doc) throw new Error(`基线版本 ${refKey(ref)} 还没发布，稍后再提交`);
  return levelOf(doc);
}

/**
 * 升级副本的前驱：本卡上一级的副本没撤回、主服也还没吸收它（它的等级高于基线）时接在它后面，否则接在卡的基线后面。
 * 和主服同级的提交、上一级已经套用并跟上过，都以基线为前驱。
 */
async function levelPred(actor, {level, base, card}) {
  const previous = game.actors.get(await copyIdFor(actor.id, level - 1));
  const copy = previous && copyOf(previous);
  if (!copy || copy.withdrawn || copy.kind !== 'level' || copy.cardId !== actor.id) return base;
  return level - 1 > await levelOfRef(base, card) ? copyRef(copy) : base;
}

// 冻结一张卡：换上副本 id、带后缀的名字、「已提交」文件夹和副本记号。token 名不动，套用时 normalizeCopy 换回
function frozenData(source, copy, folder) {
  const data = source.toObject();
  copy.moduleUsage = collectModuleUsage(data, game.modules?.values() ?? []);
  data._id = copy.copyId;
  data.name = `${source.name}${NAME_SUFFIX[copy.kind](copy.level)}`;
  data.folder = folder;
  data.flags = {...data.flags, [OWN_FLAG]: {copy}};
  return data;
}

// 副本照常交了，但有条目没写进去：提示提交者修好再交
function warnLeftOut(name, entries) {
  console.warn(`taka-charsync | ${name} 有 ${entries.length} 个条目过不了校验，没有进副本`, entries);
  const names = entries.map(entry => `「${foundry.utils.escapeHTML(entry.name || entry.id)}」`).join('、');
  ui.notifications.warn(
    `${foundry.utils.escapeHTML(name)}：${entries.length} 个条目过不了校验，没有进副本：${names}。修好这些条目后重新提交`,
    {permanent: true},
  );
}

/**
 * 写一份副本，已有同 id 的就整卡覆盖。Foundry 服务端不查顶层 id 重复，撞 id 的 create 会整行覆盖、
 * 旧副本独有的条目留成孤儿行，所以先查有没有，不能靠 create 失败来判断。
 */
async function saveCopy(data) {
  const existing = game.actors.get(data._id);
  if (!existing) {
    const created = await Actor.create(data, {keepId: true, takaCharsync: true});
    if (!created) throw new Error(`没能建出副本「${data.name}」`);
    return;
  }
  const own = data.flags[OWN_FLAG];
  const {skipped, removed} = await importFull(existing, data, {flags: own});
  // 旧副本上坏掉的条目整条删掉，卡上同 id 的条目这次也没写进去
  const sent = new Set([...data.items, ...data.effects].map(entry => entry._id));
  const leftOut = [...skipped, ...removed.filter(entry => sent.has(entry.id))];
  if (leftOut.length) warnLeftOut(own.copy.name, leftOut);
  // importFull 保留卡上的文件夹，撤回后重新提交的要挪回「已提交」
  if ((existing.folder?.id ?? null) !== data.folder) await existing.update({folder: data.folder}, {takaCharsync: true});
}

/**
 * 冻结主人和同伴，各写一份副本，返回主人副本的记号。同伴先写，主人的副本出现时同伴已经齐了。
 * `companions` 每项 `{actor, copyId, pred}`；魔宠副本的 `system.master.id` 指向主人副本。
 */
async function freeze(actor, {kind, level, copyId, pred, targetWorld, replaces = null, companions = []}) {
  const {rulesHash} = await prepareSubmissionRules(actor, targetWorld, companions.map(companion => companion.actor));
  const rev = Date.now();
  const folder = roleFolder('submitted');
  const stamp = {rev, level, kind, targetWorld, rulesHash, withdrawn: false, submittedBy: game.user.id};
  const companionRefs = [];
  const payloads = [];
  for (const companion of companions) {
    const copy = {
      ...stamp, copyId: companion.copyId, cardId: companion.actor.id, pred: companion.pred, replaces: null,
      name: companion.actor.name, companionOf: copyId, companions: [],
    };
    const data = frozenData(companion.actor, copy, folder);
    if (data.type === 'familiar') data.system.master = {...data.system.master, id: copyId};
    payloads.push(data);
    companionRefs.push({copyId: companion.copyId, rev, sourceId: companion.actor.id});
  }
  const copy = {...stamp, copyId, cardId: actor.id, pred, replaces, name: actor.name, companionOf: null, companions: companionRefs};
  payloads.push(frozenData(actor, copy, folder));
  const frozen = await snapshotActorMedia(payloads);
  for (const data of frozen.docs) await saveCopy(data);
  return copy;
}

// 撤回一份副本和随它冻结的同伴副本：打上 withdrawn，挪进「已撤回」文件夹
async function markWithdrawn(copyActor, folder) {
  const targets = [copyActor, ...game.actors.filter(actor => copyOf(actor)?.companionOf === copyActor.id)];
  for (const target of targets) {
    if (copyOf(target).withdrawn) continue;
    await target.update({folder, [`flags.${OWN_FLAG}.copy.withdrawn`]: true}, {takaCharsync: true});
  }
}

/**
 * 提交卡当前等级 N 的升级副本（id 由卡和 N 决定，同级再交就覆盖那一份），同伴各冻结一份；然后撤回本卡高于 N 级的升级副本。
 * 前驱见 `levelPred`；卡上还没有基线、也没有起步基线时抛错。当前用户不是卡主（含 GM）时不动，返回 null。
 * @returns {Promise<CopyMeta|null>}
 */
export async function submitLevel(actor) {
  const meta = await loadMeta();
  const card = cardToSubmit(actor, meta, '提交');
  if (!card) return null;
  const base = baseOf(actor, meta);
  if (!base) throw new Error('这张卡还没跟上过主服，跟上之后再提交');
  const level = levelOf(actor._source);
  const companions = await Promise.all(companionsOf(actor, meta).map(async companion => ({
    actor: companion, copyId: await copyIdFor(companion.id, level), pred: baseOf(companion, meta),
  })));
  const copy = await freeze(actor, {
    kind: 'level', level, copyId: await copyIdFor(actor.id, level), pred: await levelPred(actor, {level, base, card}),
    targetWorld: card.world, companions,
  });
  const folder = roleFolder('withdrawn');
  for (const other of copyActorsOf(actor.id)) {
    const higher = copyOf(other);
    if (higher.kind === 'level' && higher.level > level && !higher.withdrawn) await markWithdrawn(other, folder);
  }
  return copy;
}

/**
 * 把还没对应主服卡的卡作为新卡提交到 `targetWorld`（当前用户在那个团有账号的世界），`replaces` 是要替换的
 * 当前用户在那个团的卡，不替换为 null。副本没有前驱，id 同升级副本。当前用户不是卡主（含 GM）时不动，返回 null。
 * @returns {Promise<CopyMeta|null>}
 */
export async function submitNew(actor, {targetWorld, replaces = null}) {
  const meta = await loadMeta();
  if (!ownsNewCard(actor, meta)) {
    console.warn(`taka-charsync | ${actor.name} 不是当前用户还没对应主服的卡，没有作为新卡提交`);
    return null;
  }
  if (!newCardWorlds(meta).includes(targetWorld)) throw new Error(`你在 ${targetWorld} 没有对应的账号`);
  if (replaces && !replaceableCards(meta, targetWorld).includes(replaces)) {
    throw new Error(`${replaces} 不是你在 ${targetWorld} 的卡，不能替换`);
  }
  const level = levelOf(actor._source);
  const companions = await Promise.all(companionsOf(actor, meta).map(async companion => ({
    actor: companion, copyId: await copyIdFor(companion.id, level), pred: null,
  })));
  return freeze(actor, {kind: 'new', level, copyId: await copyIdFor(actor.id, level), pred: null, targetWorld, replaces, companions});
}

/**
 * 整卡同步到主服：每张卡一份整卡副本（再交就覆盖），同伴各一份，不看前驱、不撤回别的副本，要主服 GM 点套用。
 * 当前用户不是卡主（含 GM）时不动，返回 null。
 * @returns {Promise<CopyMeta|null>}
 */
export async function submitForce(actor) {
  const meta = await loadMeta();
  const card = cardToSubmit(actor, meta, '整卡同步');
  if (!card) return null;
  const companions = await Promise.all(companionsOf(actor, meta).map(async companion => ({
    actor: companion, copyId: await forceCopyIdFor(companion.id), pred: null,
  })));
  return freeze(actor, {
    kind: 'force', level: levelOf(actor._source), copyId: await forceCopyIdFor(actor.id), pred: null,
    targetWorld: card.world, companions,
  });
}

/**
 * 撤回一份副本，连同它的同伴副本和本卡更高级的同类副本。同伴的副本不能单独撤回；
 * 当前用户不是提交者（含 GM）时不动。
 */
export async function withdraw(copyActor) {
  const copy = copyOf(copyActor);
  if (!copy) throw new Error(`${copyActor.name}不是冻结副本`);
  if (copy.companionOf) throw new Error(`${copyActor.name}是同伴的副本，随主人的副本撤回`);
  if (!canWithdraw(copy)) {
    console.warn(`taka-charsync | ${copyActor.name} 不是当前用户提交的，没有撤回`);
    return;
  }
  const folder = roleFolder('withdrawn');
  await markWithdrawn(copyActor, folder);
  for (const other of copyActorsOf(copy.cardId)) {
    const higher = copyOf(other);
    if (higher.kind === copy.kind && higher.level > copy.level && !higher.withdrawn) await markWithdrawn(other, folder);
  }
}

/**
 * 主服直接改过等级、卡被重置成主服版本后，本卡还没被主服套用的副本谱系已经失效，全部撤回；
 * 主服最后套用的那份（`applied`）不动。
 */
export async function withdrawUnapplied(actor, applied) {
  const folder = roleFolder('withdrawn');
  for (const copyActor of copyActorsOf(actor.id)) {
    const copy = copyOf(copyActor);
    if (!copy.withdrawn && !sameRef(copyRef(copy), applied)) await markWithdrawn(copyActor, folder);
  }
}

// 内嵌文档往上找所属的 actor
function actorOf(document) {
  for (let current = document; current; current = current.parent) {
    if (current.documentName === 'Actor') return current;
  }
  return null;
}

let lastLockNotice = 0;

function noticeLocked() {
  const now = Date.now();
  if (now - lastLockNotice < LOCK_NOTICE_INTERVAL_MS) return;
  lastLockNotice = now;
  ui.notifications.warn(LOCK_NOTICE);
}

/**
 * 冻结副本只能由 GM 或本模组自己（选项带 `takaCharsync`）改：别的对副本本身、它的条目和效果的修改在发起的客户端拦下。
 * 提示每 10 秒最多一次，卡面一次操作连带的几次写入不刷屏。
 */
export function installCopyLock() {
  for (const [hook, optionsAt] of Object.entries(LOCKED_HOOKS)) {
    Hooks.on(hook, (...args) => {
      if (game.user.isGM || args[optionsAt]?.takaCharsync) return;
      const actor = actorOf(args[0]);
      if (!actor || !copyOf(actor)) return;
      noticeLocked();
      return false;
    });
  }
}

// 主服已经处理过、留着没用的副本：撤回了、主服忽略了、主服已经套用，或者是主服已经比它高级的升级副本。
// 整卡副本不在等级链上，等级低于主服也还要等 GM 决定
function processed(copy, main) {
  if (copy.withdrawn) return true;
  if (!main) return false;
  const key = refKey(copyRef(copy));
  const undone = main.doc.flags?.[OWN_FLAG]?.undone ?? [];
  if (undone.includes(key) || refKey(main.applied) === key) return true;
  return copy.kind === 'level' && main.level != null && copy.level < main.level;
}

/**
 * GM 清理建卡服上已处理的副本：已撤回的、在该卡主服 `undone` 里的、正是主服 `applied` 的、等级低于主服最新等级的升级副本，
 * 再加上这些主人副本的同伴副本（魔宠在主服没有等级，按它自己的卡判断永远删不掉）。
 * 只删引擎已经发布过的（`refKey` 在 `meta.published` 里），还没读到的提交不动；等 GM 决定的同级副本和整卡副本不删。
 * 不是 GM 时抛错。
 * @returns {Promise<{deleted: string[]}>}
 */
export async function cleanupCopies(meta) {
  if (!game.user.isGM) throw new Error('只有 GM 能清理副本');
  const published = new Set(meta.published ?? []);
  const copies = game.actors.contents.flatMap(actor => {
    const copy = copyOf(actor);
    return copy && published.has(refKey(copyRef(copy))) ? [{id: actor.id, copy}] : [];
  });
  const mains = new Map();
  const mainOf = cardId => {
    const card = meta.cards?.[cardId];
    if (!card) return null;
    if (!mains.has(cardId)) mains.set(cardId, getMainLatest(card.world, card.mainId));
    return mains.get(cardId);
  };
  const deleted = new Set();
  const owners = [];
  for (const {id, copy} of copies) {
    if (!processed(copy, await mainOf(copy.cardId))) continue;
    deleted.add(id);
    if (!copy.companionOf) owners.push(copy);
  }
  const ownerIds = new Set(owners.map(copy => copy.copyId));
  const companionIds = new Set(owners.flatMap(copy => copy.companions.map(companion => companion.copyId)));
  for (const {id, copy} of copies) {
    if (companionIds.has(copy.copyId) || ownerIds.has(copy.companionOf)) deleted.add(id);
  }
  const ids = [...deleted];
  if (ids.length) await Actor.deleteDocuments(ids);
  return {deleted: ids};
}

async function confirmCleanup(button) {
  const confirmed = await foundry.applications.api.DialogV2.confirm({
    window: {title: CLEANUP},
    content: '<p>删掉已撤回、已被主服套用或忽略、以及等级低于主服的升级副本，连同它们的同伴副本；还在等 GM 决定的不删。删掉的不能恢复。</p>',
    yes: {label: '删除'},
    no: {label: '取消'},
  });
  if (!confirmed) return;
  button.disabled = true;
  try {
    const {deleted} = await cleanupCopies(await loadMeta());
    ui.notifications.info(deleted.length ? `已删除 ${deleted.length} 份副本` : '没有要清理的副本');
  } catch (error) {
    console.error('taka-charsync | 清理副本失败', error);
    const message = error instanceof Error ? error.message : String(error);
    ui.notifications.error(`清理副本失败：${foundry.utils.escapeHTML(message)}`);
  } finally {
    button.disabled = false;
  }
}

/** `renderActorDirectory` 钩子：GM 在角色目录顶部的按钮下面多一行「清理已处理的副本」。 */
export function addCleanupButton(app, html) {
  if (!game.user.isGM) return;
  const actions = html.querySelector('.directory-header .header-actions');
  if (!actions || html.querySelector('.taka-charsync-cleanup')) return;
  const row = document.createElement('div');
  row.className = 'header-actions action-buttons flexrow taka-charsync-cleanup';
  const button = document.createElement('button');
  button.type = 'button';
  button.innerHTML = `<i class="fa-solid fa-broom" inert></i><span>${CLEANUP}</span>`;
  button.addEventListener('click', () => confirmCleanup(button));
  row.append(button);
  actions.after(row);
}
