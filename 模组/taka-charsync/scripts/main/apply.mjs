// 主服：GM 客户端套用建卡服提交的副本（自动和手动）、导入新卡、整卡覆盖、撤回。主服卡只由在线的 activeGM 客户端写，
// 写卡的操作都经 `exclusive` 排成一队。
import {
  autoAllowed, historyFile, normalizeCopy, planApply, planForce, planNew, refKey, sameRef, syncAllowed, xpAfter,
} from '../lib/chain.mjs';
import { formatSummary, summarize } from '../lib/merge.mjs';
import { mergeBuildChanges } from '../lib/build-sync.mjs';
import { canonicalJson } from '../lib/canon.mjs';
import { describeChanges } from '../lib/change-summary.mjs';
import { OWN_FLAG } from '../lib/rules.mjs';
import { copyRuleError } from '../lib/rule-validation.mjs';
import { cleanDoc, importFull } from '../foundry/import.mjs';
import { snapshotActorMedia } from '../foundry/media.mjs';
import { fetchPublished, getCopies, getHistoryDoc, getMeta, getPublishDir } from '../foundry/published.mjs';

const ARCHIVE_FOLDER = 'charsync 归档';
const MAX_LEVEL = 30;
const SPEAKER = '角色卡同步';
const NO_CHANGES = {level: null, added: [], removed: [], changed: [], skillRanks: []};

const copyRef = copy => ({kind: 'copy', copyId: copy.copyId, rev: copy.rev});
const ownFlags = actor => actor._source.flags?.[OWN_FLAG] ?? {};
const levelOf = doc => doc.system?.details?.level?.value ?? null;
const escape = text => foundry.utils.escapeHTML(String(text));
const errorMessage = error => (error instanceof Error ? error.message : String(error));

/** 套用要用的历史版本还没发布或已被清掉。检查时记成 pending 的 `pred`，之后的检查照常再试。 */
class MissingVersionError extends Error {}

let queue = Promise.resolve();

/** 写主服卡的操作排成一队依次执行：定时检查和面板上的手动操作不会同时改一张卡。`task` 里不能再调 `exclusive`。 */
export function exclusive(task) {
  const run = queue.then(task);
  queue = run.catch(() => {});
  return run;
}

// 本会话里自动套用抛过错的卡 → 错误信息：不再自动套，手动套用成功后清掉
const failedCards = new Map();
// 读到过的历史版本的等级和它代表的套用状态，键是 refKey；历史文档的内容不会变
const versionInfo = new Map();
// working 条目的卡类型，键是 `cardId:hash`
const workingTypes = new Map();
let lastResult = null;

/** 本客户端最近一次检查的结果（`runApplyCheck` 的返回值加 `trigger`、`at`）；还没检查过、或不是 activeGM 时为 null。 */
export const lastCheck = () => lastResult;

async function loadMeta() {
  const meta = await getMeta();
  if (!meta) throw new Error('taka-charsync：读不到同步数据（meta.json），引擎可能还没发布过');
  return meta;
}

function requireCopyRules(copy, meta) {
  if (copy.targetWorld !== game.world.id) throw new Error('副本所属团与当前主服世界不同，不能套用');
  const error = copyRuleError(copy, meta.ruleSnapshot);
  if (error) throw new Error(error);
  if (!meta.published?.includes(refKey(copyRef(copy)))) {
    throw new MissingVersionError('这份副本尚未通过当前团规则检查或尚未发布');
  }
}

async function currentCopy(copy, meta, actor = null) {
  requireCopyRules(copy, meta);
  const latest = (await getCopies(copy.cardId)).find(entry => entry.copyId === copy.copyId);
  if (!latest || latest.rev !== copy.rev || latest.kind !== copy.kind || latest.cardId !== copy.cardId) {
    throw new Error('这份副本已经换版或不再发布，请刷新同步面板后重试');
  }
  if (latest.withdrawn) throw new Error('这份副本已经撤回，请刷新同步面板');
  requireCopyRules(latest, meta);
  if (latest.companionOf) throw new Error('同伴副本随主人一起套用，请刷新同步面板');
  const card = meta.cards?.[latest.cardId];
  if (latest.kind === 'new') {
    if (card || mainActorOf(meta, latest.cardId)) throw new Error('这张新卡已经有主服对应卡，请刷新同步面板');
    const owner = meta.working?.[latest.cardId]?.owner;
    if ((owner && owner !== latest.submittedBy) || !meta.users?.[latest.submittedBy]?.[game.world.id]) {
      throw new Error('新卡的卡主归属已变化，请刷新同步面板');
    }
  } else {
    if (!actor || card?.world !== game.world.id || card.mainId !== actor.id || card.companionOf
        || card.charOwner !== latest.submittedBy || ownFlags(actor).retired) {
      throw new Error('副本与主服卡的对应或卡主归属已变化，请刷新同步面板');
    }
    const own = ownFlags(actor), ref = copyRef(latest);
    if (sameRef(own.applied?.ref ?? null, ref) || own.undone?.includes(refKey(ref))) {
      throw new Error('这份副本已经处理过，请刷新同步面板');
    }
    if (latest.kind === 'level' && latest.level < levelOf(actor._source)) {
      throw new Error('主服等级已提高，请刷新同步面板');
    }
  }
  return latest;
}

/** 本团映射里的主卡和它们的主服卡；同伴随主人走，不单列。主服卡不在或已退役的不算。 */
export function campaignCards(meta) {
  const world = game.world.id;
  return Object.entries(meta.cards ?? {}).flatMap(([cardId, card]) => {
    if (card.world !== world || card.companionOf) return [];
    const actor = game.actors.get(card.mainId);
    return actor && !ownFlags(actor).retired ? [{cardId, card, actor}] : [];
  });
}

// 本世界卡上记着的建卡服卡 id：新卡导入后到引擎下一次读主服之前，映射里还没有它
const importedCardIds = () => new Set(game.actors.contents.map(actor => ownFlags(actor).charId).filter(Boolean));

/**
 * 建卡服上本团玩家还没对应主服卡的卡（`meta.working` 的条目加上 `cardId`）：所有者在本世界有对应账号，
 * 不在映射里，本世界也没有记着它的卡。
 */
export function newCardCandidates(meta) {
  const world = game.world.id;
  const imported = importedCardIds();
  return Object.entries(meta.working ?? {})
    .filter(([cardId, entry]) => !meta.cards?.[cardId] && !imported.has(cardId) && meta.users?.[entry.owner]?.[world])
    .map(([cardId, entry]) => ({cardId, ...entry}));
}

/** 面板上「建卡服上本团玩家的其他卡」：新卡候选里除去魔宠，魔宠随主人一起覆盖。 */
export async function otherCards(meta) {
  const cards = [];
  for (const candidate of newCardCandidates(meta)) {
    const key = `${candidate.cardId}:${candidate.hash}`;
    if (!workingTypes.has(key)) {
      workingTypes.set(key, (await fetchPublished(`working/${candidate.cardId}.json`))?.doc.type ?? null);
    }
    if (workingTypes.get(key) !== 'familiar') cards.push(candidate);
  }
  return cards;
}

// 建卡服卡在本世界对应的主服卡：映射里的，或者刚导入、卡上记着它的；都没有时为 null
function mainActorOf(meta, cardId) {
  const card = meta.cards?.[cardId];
  if (card?.world === game.world.id) {
    const actor = game.actors.get(card.mainId);
    if (!actor) throw new Error(`taka-charsync：找不到映射里的主服卡 ${card.mainId}`);
    return actor;
  }
  return game.actors.find(actor => ownFlags(actor).charId === cardId && !ownFlags(actor).retired) ?? null;
}

function cardIdOf(meta, actor) {
  const world = game.world.id;
  const mapped = Object.entries(meta.cards ?? {}).find(([, card]) => card.world === world && card.mainId === actor.id);
  const cardId = mapped?.[0] ?? ownFlags(actor).charId;
  if (!cardId) throw new Error(`taka-charsync：${actor.name} 没有对应的建卡服卡`);
  return cardId;
}

/** 新卡副本要替换的主服卡 id：玩家选了替换、那张卡还在本团时为它，否则为 null。 */
export function replaceTarget(meta, copy) {
  const card = copy.replaces ? meta.cards?.[copy.replaces] : null;
  const actor = card?.world === game.world.id ? game.actors.get(card.mainId) : null;
  return actor && !ownFlags(actor).retired ? actor.id : null;
}

async function requireHistory(ref) {
  const doc = await getHistoryDoc(ref);
  if (!doc) throw new MissingVersionError(`版本 ${refKey(ref)} 还没发布或已被清理`);
  return doc;
}

// 某个版本的等级和它代表的套用状态（主服版本是它当时的 applied，副本和建卡服版本是它自己）。读不到时为 null
async function versionOf(ref, copies) {
  const key = refKey(ref);
  if (versionInfo.has(key)) return versionInfo.get(key);
  let info = null;
  const listed = ref.kind === 'copy' && copies.find(copy => copy.copyId === ref.copyId && copy.rev === ref.rev);
  if (listed) {
    info = {level: listed.level, applied: ref};
  } else {
    const doc = await getHistoryDoc(ref);
    if (!doc) return null;
    info = {level: levelOf(doc), applied: ref.kind === 'main' ? doc.flags?.[OWN_FLAG]?.applied?.ref ?? null : ref};
  }
  versionInfo.set(key, info);
  return info;
}

// planApply 可能选中的升级副本的前驱
async function predVersions(copies, mainLevel) {
  const versions = new Map();
  for (const copy of copies) {
    if (copy.kind !== 'level' || copy.withdrawn || !copy.pred || copy.level < mainLevel) continue;
    const info = await versionOf(copy.pred, copies);
    if (info) versions.set(refKey(copy.pred), info);
  }
  return versions;
}

async function checkCard({cardId, actor, meta, allowAuto, result}) {
  const own = ownFlags(actor);
  const main = {
    level: levelOf(actor._source),
    xp: actor._source.system.details?.xp ?? {value: 0, max: 0},
    applied: own.applied?.ref ?? null,
    undone: own.undone ?? [],
  };
  const copies = await getCopies(cardId);
  const force = planForce({copies, undone: new Set(main.undone), mainApplied: main.applied});
  if (force) result.pending.push({actorId: actor.id, reason: 'force', copy: force});

  const versions = await predVersions(copies, main.level);
  const plan = planApply({
    main, copies, autoAllowed: allowAuto,
    levelOf: ref => versions.get(refKey(ref))?.level,
    appliedAt: ref => versions.get(refKey(ref))?.applied ?? null,
  });
  if (plan.action === 'none') return;
  if (plan.action === 'wait') {
    result.waiting.push({actorId: actor.id, copy: plan.copy});
    return;
  }
  const pend = (reason, message) => {
    result.pending.push({actorId: actor.id, reason, copy: plan.copy, ...(message ? {message} : {})});
  };
  // 前驱读不到时 planApply 会把它当成谱系失效
  if (plan.copy.pred && !versions.has(refKey(plan.copy.pred))) return pend('pred', '前驱版本还没发布');
  if (plan.action === 'gm') return pend(plan.reason);
  if (failedCards.has(actor.id)) return pend('error', failedCards.get(actor.id));
  try {
    if ((await applyCopy(actor, plan.copy, {meta})) === 'conflict') return pend('conflict');
    result.applied.push(actor.id);
  } catch (error) {
    const message = errorMessage(error);
    // 版本还没发布是暂时的，每次检查都会再试，不当错误记
    if (error instanceof MissingVersionError) {
      console.warn(`taka-charsync | ${actor.name} 暂时套用不了：${message}`);
      return pend('pred', message);
    }
    console.error(`taka-charsync | ${actor.name} 自动套用失败：${message}`, error);
    failedCards.set(actor.id, message);
    return pend('error', message);
  }
}

async function addNewCards(meta, result) {
  const world = game.world.id;
  const mappedCardIds = new Set([...Object.keys(meta.cards ?? {}), ...importedCardIds()]);
  for (const {cardId} of newCardCandidates(meta)) {
    try {
      const copies = await getCopies(cardId);
      for (const copy of planNew({world, copies, mappedCardIds, undone: new Set()})) {
        if (copy.companionOf) continue;
        result.pending.push({actorId: null, reason: 'new', copy});
      }
    } catch (error) {
      console.error(`taka-charsync | 读不到建卡服卡 ${cardId} 的副本`, error);
    }
  }
}

async function checkCampaign(trigger) {
  const result = {applied: [], pending: [], waiting: []};
  if (!game.users.activeGM?.isSelf) {
    // 不再是 activeGM 时，上次检查的结果不作数，角标归零
    lastResult = null;
    return result;
  }
  const meta = await getMeta();
  if (meta) {
    const cards = campaignCards(meta);
    if (!syncAllowed(meta, game.world.id)) {
      for (const {actor} of cards) result.pending.push({actorId: actor.id, reason: 'paused', copy: null});
    } else {
      const allowAuto = autoAllowed({
        playersOnline: game.users.filter(user => user.active && !user.isGM).length,
        combatStarted: game.combats.some(combat => combat.started),
      });
      for (const {cardId, actor} of cards) {
        try {
          await checkCard({cardId, actor, meta, allowAuto, result});
        } catch (error) {
          console.error(`taka-charsync | 检查 ${actor.name} 失败`, error);
          result.pending.push({actorId: actor.id, reason: 'error', copy: null, message: errorMessage(error)});
        }
      }
      await addNewCards(meta, result);
    }
  }
  lastResult = {...result, trigger, at: Date.now()};
  return result;
}

/**
 * 检查本团每张映射卡（同伴随主人套用）：放行了、能自动套的套上，要 GM 点的记进 `pending`，经验还没满的记进 `waiting`。
 * 只在当前用户是 activeGM 时做，否则返回空结果。
 *
 * `pending` 的 `reason`：`paused`（两边规则版本不一致，每张卡一条）、`force`（整卡副本，永不自动套）、
 * `new`（新卡副本，`actorId` 为 null）、`planApply` 给出的 `pred`/`stale`/`same-level`/`gap`/`session`、
 * `conflict`（合并有冲突）、`error`（套用抛错，带 `message`；本会话里不再自动重试这张卡）。
 * 历史版本读不到时也记成 `pred`，带 `message`。
 *
 * @returns {Promise<{applied: string[], pending: {actorId: string|null, reason: string, copy: object|null, message?: string}[],
 *   waiting: {actorId: string, copy: object}[]}>}
 */
export function runApplyCheck({trigger}) {
  return exclusive(() => checkCampaign(trigger));
}

const HP_PATH = 'system.attributes.hp.value';

// 三方合并一张主服卡：基线是副本的前驱，建卡服一侧是副本，都先换成这张主服卡的样子。
// 建卡服升级时 PF2e 已经按新上限加过一次当前血量，合并会把它当成建卡服的改动带过来，主服有伤势时还会记成冲突、
// 挡住自动套用。当前血量是主服的状态：一律取主服的值、不算冲突，导入后再按上限的增量补（addHpGain）
async function mergeVersion(actor, {pred, ref, masterId, clean = cleanDoc}) {
  if (!pred) throw new MissingVersionError(`${actor.name} 的副本没有前驱版本`);
  for (const key of ['items', 'effects']) {
    const invalid = (actor._source[key] ?? []).filter(entry => actor[key].invalidDocumentIds.has(entry._id) && !actor[key].has(entry._id));
    if (invalid.length) throw new Error(`「${actor.name}」有非法条目（${invalid.map(entry => entry.name ?? entry._id).join('、')}），请先修复；普通同步不会清理这些条目`);
  }
  const sides = {actorId: actor.id, masterId};
  const base = clean(onMainSide(await requireHistory(pred), pred, sides));
  const char = clean(onMainSide(await requireHistory(ref), ref, sides));
  const captured = actor.toObject();
  const {docs: [frozenMain]} = await snapshotActorMedia([clean(captured)]);
  const main = clean(frozenMain);
  const {doc, conflicts} = mergeBuildChanges({base, main, char});
  const mainHp = main.system.attributes?.hp;
  if (mainHp && doc.system.attributes?.hp) doc.system.attributes.hp.value = mainHp.value;
  return {main, doc, captured, conflicts: conflicts.filter(conflict => conflict.scope !== 'actor' || conflict.path !== HP_PATH)};
}

// 文档换到主服这张卡上：换 id，魔宠的主人换成主服主人
function asMainCard(doc, {actorId, masterId}) {
  const out = foundry.utils.deepClone(doc);
  out._id = actorId;
  if (out.type === 'familiar' && masterId) out.system.master = {...out.system.master, id: masterId};
  return out;
}

// 历史文档换成主服这张卡的样子，副本文档先过 normalizeCopy
const onMainSide = (doc, ref, sides) => (ref.kind === 'copy' ? normalizeCopy(doc, sides) : asMainCard(doc, sides));

const isTaken = id => game.actors.has(id) || game.actors.invalidDocumentIds.has(id);

// 新卡用的 id：原 id 在本世界没被占用就沿用，否则换新的
const freeId = id => (isTaken(id) ? foundry.utils.randomID() : id);

// 主服还没有的同伴必须沿用建卡服的 id（Ruling 40）：引擎靠「主服有同 id 的卡」把魔宠认成同伴，
// 换了 id 会被当成独立主卡。被占用时不写，交给 GM
function companionIdFor(sourceId, name) {
  if (isTaken(sourceId)) throw new Error(`taka-charsync：同伴「${name}」的 id 已被占用（${sourceId}），没有动卡`);
  return sourceId;
}

// 主人和同伴必须是同一次冻结的版本；少一张就不套，避免只写入半组卡。
async function companionCopies(meta, copy) {
  const found = [];
  for (const {copyId, sourceId, rev} of copy.companions ?? []) {
    const companionCopy = (await getCopies(sourceId)).find(entry => entry.copyId === copyId);
    if (!companionCopy || companionCopy.withdrawn || companionCopy.rev !== rev
        || companionCopy.companionOf !== copy.copyId || companionCopy.cardId !== sourceId
        || companionCopy.kind !== copy.kind || companionCopy.submittedBy !== copy.submittedBy) {
      throw new Error(`同伴「${companionCopy?.name ?? sourceId}」的副本缺失、已撤回或版本不符，请刷新后重新提交整组卡`);
    }
    requireCopyRules(companionCopy, meta);
    const card = meta.cards?.[sourceId];
    if (card && (card.world !== game.world.id || card.companionOf !== copy.cardId || card.charOwner !== copy.submittedBy)) {
      throw new Error(`同伴「${companionCopy.name}」的主服对应关系已变化，请刷新同步面板`);
    }
    found.push({sourceId, copy: companionCopy});
  }
  return {found, skipped: []};
}

function upload(dir, name, value) {
  const file = new File([JSON.stringify(value)], name, {type: 'application/json'});
  return foundry.applications.apps.FilePicker.implementation.upload('data', dir, file, {}, {notify: false});
}

async function ensureDirectory(target) {
  try {
    await foundry.applications.apps.FilePicker.implementation.createDirectory('data', target, {});
  } catch (error) {
    // 目录已经在也报错；真建不出来时，之后的上传会失败并报出来
    console.debug(`taka-charsync | 建目录 ${target}：${errorMessage(error)}`);
  }
}

/** 把卡写前的快照上传成撤回点 `undo/<世界>/<id>-<毫秒>.json`，返回它在发布目录下的路径。传不上去时抛错。 */
async function saveUndoPoint(actor, snapshot) {
  const publishDir = getPublishDir();
  const rel = `undo/${game.world.id}`;
  const name = `${actor.id}-${Date.now()}.json`;
  let uploaded = await upload(`${publishDir}/${rel}`, name, snapshot);
  // 目录不在时上传返回空（还会无视 notify:false 弹一条错误）：补建目录再传一次
  if (!uploaded?.path) {
    await ensureDirectory(`${publishDir}/undo`);
    await ensureDirectory(`${publishDir}/${rel}`);
    uploaded = await upload(`${publishDir}/${rel}`, name, snapshot);
  }
  if (!uploaded?.path) throw new Error(`taka-charsync：${actor.name} 的撤回点没有传上去，没有动卡`);
  return `${rel}/${name}`;
}

/** 建卡服的某个 working 版本上传成主服 Data 的 `history/char-<hash>.json`（已有就不传），返回它的引用。 */
async function publishCharVersion(cardId, working) {
  const ref = {kind: 'char', cardId, hash: working.hash};
  const rel = historyFile(ref);
  // Foundry 不让非媒体文件同名覆盖，先看有没有
  if (await fetchPublished(rel)) return ref;
  const uploaded = await upload(`${getPublishDir()}/history`, rel.slice('history/'.length), working.doc);
  if (!uploaded?.path) throw new Error(`taka-charsync：${working.name} 的建卡服版本没有传上去，没有动卡`);
  return ref;
}

async function loadWorking(cardId) {
  const working = await fetchPublished(`working/${cardId}.json`);
  if (!working) throw new Error(`taka-charsync：建卡服还没发布卡 ${cardId} 现在的版本`);
  return working;
}

// 当前血量和上限，要在写第一张卡之前取：主人一导入完，PF2e 就按主人的新等级重算魔宠的上限，晚取就拿不到旧值
function hpOf(actor) {
  const hp = actor.system.attributes?.hp;
  return hp ? {value: actor._source.system.attributes.hp.value, max: hp.max} : null;
}

// 按最大血量的增量补当前血量（PF2e 正常升级会这样做，整卡导入不会），从写卡前的血量算起，夹在 0 到新上限之间
async function addHpGain(actor, before) {
  const max = actor.system.attributes?.hp?.max;
  if (!before || max == null || max === before.max) return;
  const value = Math.clamp(before.value + max - before.max, 0, max);
  if (value === actor._source.system.attributes.hp.value) return;
  const updated = await actor.update({'system.attributes.hp.value': value});
  if (!updated) throw new Error(`taka-charsync：${actor.name} 的当前血量没有补上`);
}

// 整卡导入一张已有的卡，`flags` 是写入后完整的同步记号；导入成功就记进 `written`（后面失败时要恢复的卡）。
// 返回给悄悄话用的变动
async function writeCard(target, doc, flags, {hpBefore, written}) {
  const before = cleanDoc(target.toObject());
  const dropped = await importFull(target, doc, {flags});
  written.push(target);
  if (hpBefore) await addHpGain(target, hpBefore);
  return {summary: summarize(before, target.toObject()), ...dropped};
}

/** 按文档在本世界新建一张卡：先建不带条目的空卡，再整卡导入（逐条校验条目）；导入失败时删掉空卡再抛错。 */
async function createActor(doc, {id, ownership, folder, flags}) {
  const shell = foundry.utils.deepClone(doc);
  Object.assign(shell, {_id: id, ownership, folder, items: [], effects: []});
  delete shell._stats;
  delete shell.sort;
  shell.flags = {...shell.flags, [OWN_FLAG]: flags};
  const actor = await Actor.implementation.create(shell, {keepId: true});
  if (!actor) throw new Error(`taka-charsync：没能在主服建出「${doc.name}」`);
  try {
    return {actor, ...(await importFull(actor, doc, {flags}))};
  } catch (error) {
    try {
      await actor.delete();
    } catch (deleteError) {
      throw new Error(`${errorMessage(error)}；没导入完的「${doc.name}」（${actor.id}）没能删掉：${errorMessage(deleteError)}`, {cause: error});
    }
    throw error;
  }
}

// 建卡服新建、主服还没有的同伴：在主服新建，所有者和文件夹照主服主人
function createCompanion({sourceId, id, ref, doc}, master, at) {
  return createActor(doc, {
    id, ownership: foundry.utils.deepClone(master.ownership), folder: master.folder?.id ?? null,
    flags: {charId: sourceId, applied: {ref, at, undoPath: null, level: levelOf(doc)}},
  });
}

const entryList = entries => entries.map(entry => `「${escape(entry.name || entry.id)}」`).join('、');
const nameList = names => (names.length ? names.map(name => `「${escape(name)}」`).join('、') : '无');

function cardReport({summary = NO_CHANGES, conflicts = [], skipped = [], removed = []}) {
  let html = formatSummary(summary, conflicts);
  if (skipped.length) html += `<p>过不了校验、没有套用的条目：${entryList(skipped)}</p>`;
  if (removed.length) html += `<p>卡上原有的过不了校验的条目已移除：${entryList(removed)}</p>`;
  return html;
}

async function whisperGMs(actorId, content) {
  const whisper = ChatMessage.implementation.getWhisperRecipients('GM').map(user => user.id);
  const message = await ChatMessage.implementation.create({
    content, whisper, speaker: {alias: SPEAKER}, flags: {[OWN_FLAG]: {actorId}},
  });
  if (!message) throw new Error('taka-charsync：给 GM 的悄悄话没有发出去');
}

// 卡已经写完（或已经失败）之后的悄悄话：发不出去不回滚，只提示
async function whisperOrWarn(actorId, content) {
  try {
    await whisperGMs(actorId, content);
  } catch (error) {
    console.error('taka-charsync | 给 GM 的悄悄话没有发出去', error);
    ui.notifications.warn('角色卡同步：给 GM 的悄悄话没有发出去，详见控制台');
  }
}

/**
 * 主人整卡导入之后某一步失败时的补偿（Ruling 43）：写过的卡用写前的快照整卡恢复（同步记号一并恢复），
 * 这次新建的同伴删掉。返回要抛出的错误；恢复也没做完时给 GM 发悄悄话，写明写过哪些卡、哪些没恢复。
 */
async function rollBack(error, {master, written, created, snapshots}) {
  if (!written.length && !created.length) return error;
  const failed = [];
  for (const target of [...written].reverse()) {
    const snapshot = snapshots.get(target.id);
    try {
      await importFull(target, snapshot, {flags: snapshot.flags?.[OWN_FLAG] ?? {}});
    } catch (restoreError) {
      console.error(`taka-charsync | ${target.name} 没能恢复到套用前`, restoreError);
      failed.push(`「${escape(target.name)}」没能恢复：${escape(errorMessage(restoreError))}`);
    }
  }
  const leftover = created.filter(actor => game.actors.has(actor.id));
  if (leftover.length) {
    try {
      await Actor.implementation.deleteDocuments(leftover.map(actor => actor.id));
    } catch (deleteError) {
      console.error('taka-charsync | 这次新建的卡没能删掉', deleteError);
      failed.push(`这次新建的卡${nameList(leftover.map(actor => actor.name))}没能删掉：${escape(errorMessage(deleteError))}`);
    }
  }
  const message = errorMessage(error);
  if (!failed.length) return new Error(`${message}；写过的卡已恢复到套用前`, {cause: error});
  const touched = nameList([...written, ...created].map(actor => actor.name));
  await whisperOrWarn(master.id, `<p><strong>${escape(master.name)}</strong>：套用中途失败（${escape(message)}），`
    + `写过的卡：${touched}，恢复没有做完：</p><ul>${failed.map(line => `<li>${line}</li>`).join('')}</ul>`);
  return new Error(`${message}；恢复没有做完，详见给 GM 的悄悄话`, {cause: error});
}

/**
 * 写卡（套用的第 2–6 步）：主人和已有同伴先各存一份撤回点，再整卡导入主人、逐个导入已有同伴、新建主服还没有的同伴，
 * 最后给 GM 发悄悄话。`hpGain` 时每张卡按最大血量的增量补当前血量。主人的 `applied` 里记下一起写的同伴，撤回时用；
 * `consumed` 里的版本同一次记进主人的 `undone`。主人导入之后哪一步失败，都把写过的卡恢复、删掉新建的同伴再抛错；
 * 只有悄悄话发不出去不算失败。
 */
async function writeCards(actor, {doc, ref, level, conflicts, updated, created, skipped, hpGain, consumed = [], title, captured}) {
  const at = Date.now();
  const existing = [actor, ...updated.map(entry => entry.actor)];
  // 快照既存成撤回点文件，也留在内存里给失败时恢复用
  const snapshots = new Map(existing.map(target => [target.id, target.toObject()]));
  const hpBefore = new Map(existing.map(target => [target.id, hpGain ? hpOf(target) : null]));
  const undoPaths = new Map();
  for (const target of existing) undoPaths.set(target.id, await saveUndoPoint(target, snapshots.get(target.id)));

  for (const entry of [{actor, captured}, ...updated]) {
    if (entry.captured && canonicalJson(entry.actor.toObject()) !== canonicalJson(entry.captured)) {
      throw new Error(`「${entry.actor.name}」在准备套用期间已有修改，请刷新后重试`);
    }
  }

  const applied = {ref, at, undoPath: undoPaths.get(actor.id), level};
  if (updated.length) applied.companions = updated.map(entry => entry.actor.id);
  if (created.length) applied.createdCompanions = created.map(entry => entry.id);
  const flags = {...ownFlags(actor), applied};
  if (consumed.length) flags.undone = [...new Set([...(flags.undone ?? []), ...consumed])];

  const written = [];
  const createdActors = [];
  const parts = [`<p><strong>${escape(actor.name)}</strong>：${title}</p>`];
  try {
    const report = await writeCard(actor, doc, flags, {hpBefore: hpBefore.get(actor.id), written});
    parts.push(cardReport({...report, conflicts}));
    for (const entry of updated) {
      const companionFlags = {
        ...ownFlags(entry.actor),
        applied: {ref: entry.ref, at, undoPath: undoPaths.get(entry.actor.id), level: levelOf(entry.doc)},
      };
      const companionReport = await writeCard(entry.actor, entry.doc, companionFlags, {hpBefore: hpBefore.get(entry.actor.id), written});
      parts.push(`<p>同伴「${escape(entry.actor.name)}」一起写入</p>`, cardReport({...companionReport, conflicts: entry.conflicts}));
    }
    for (const entry of created) {
      const companionReport = await createCompanion(entry, actor, at);
      createdActors.push(companionReport.actor);
      parts.push(`<p>同伴「${escape(companionReport.actor.name)}」是建卡服新建的，已在主服新建</p>`, cardReport(companionReport));
    }
  } catch (error) {
    throw await rollBack(error, {master: actor, written, created: createdActors, snapshots});
  }
  if (skipped.length) {
    parts.push(`<p>跳过的同伴：${skipped.map(entry => `「${escape(entry.name || entry.id)}」（${entry.why}）`).join('、')}</p>`);
  }
  await whisperOrWarn(actor.id, parts.join(''));
}

/**
 * 把一份升级副本套到主服卡上，同伴随主人一起（规格第 8 节）。三方合并、撤回点都在第一次写卡之前；任何一步失败都抛错，
 * 主人的整卡导入失败时卡没有动、也不发悄悄话。非手动时主人或任一同伴合并有冲突就不写，返回 'conflict'；
 * 手动时冲突按归属取值，列进悄悄话。经验扣掉升的级数，等级夹在 0–30，当前血量补上最大血量的增量。
 * @returns {Promise<'applied'|'conflict'>}
 */
export async function applyCopy(actor, copy, {manual = false, meta} = {}) {
  meta = await loadMeta();
  copy = await currentCopy(copy, meta, actor);
  const plan = await prepareLevel(actor, copy, meta);
  if (!manual && [plan, ...plan.updated].some(entry => entry.conflicts.length)) return 'conflict';
  await writeCards(actor, plan);
  return 'applied';
}

async function prepareLevel(actor, copy, meta, {preview = false} = {}) {
  const master = await mergeVersion(actor, {pred: copy.pred, ref: copyRef(copy), masterId: null});
  const {doc} = master;
  const fromLevel = levelOf(actor._source);
  const level = Math.clamp(levelOf(doc), 0, MAX_LEVEL);
  doc.system.details.level.value = level;
  const xp = actor._source.system.details.xp;
  doc.system.details.xp = {...doc.system.details.xp, value: xpAfter(xp, Math.max(0, copy.level - fromLevel))};
  const previewMaster = preview ? preparePreviewActor(doc) : null;
  const {found, skipped} = await companionCopies(meta, copy);
  const updated = [];
  const created = [];
  for (const {sourceId, copy: companionCopy} of found) {
    const ref = copyRef(companionCopy);
    const target = mainActorOf(meta, sourceId);
    if (target) {
      const clean = preview ? doc => preparePreviewActor(doc, previewMaster).toObject() : cleanDoc;
      const merged = await mergeVersion(target, {pred: companionCopy.pred, ref, masterId: actor.id, clean});
      updated.push({actor: target, ref, doc: merged.doc, conflicts: merged.conflicts, captured: merged.captured});
    } else {
      const id = companionIdFor(sourceId, companionCopy.name);
      created.push({sourceId, id, ref, doc: normalizeCopy(await requireHistory(ref), {actorId: id, masterId: actor.id})});
    }
  }
  return {
    doc, ref: copyRef(copy), level, conflicts: master.conflicts, captured: master.captured,
    updated, created, skipped, hpGain: true, previewMaster,
    title: `套用了建卡服提交的 ${copy.level} 级副本`,
  };
}

/**
 * 整卡覆盖（手动同步）：主服卡整张换成 `doc`，只保留 id、所有者、文件夹、同步记号和经验，不补血量。
 * `companions` 每项 `{actor, sourceId, doc, ref}`，`actor` 为 null 的是主服还没有的同伴，新建；魔宠的主人换成主服主人。
 * `skippedCompanions` 只列进悄悄话；`consumed` 是写入时一起记进 `undone` 的版本。撤回点、失败时的恢复同 `applyCopy`。
 * @returns {Promise<'applied'>}
 */
export async function applyWhole(actor, version, options = {}) {
  await writeCards(actor, prepareWhole(actor, version, options));
  return 'applied';
}

function prepareWhole(actor, {doc, ref}, {companions = [], skippedCompanions = [], consumed = []} = {}) {
  const masterDoc = asMainCard(doc, {actorId: actor.id, masterId: null});
  const xp = actor._source.system.details?.xp;
  if (xp && masterDoc.system.details) masterDoc.system.details.xp = foundry.utils.deepClone(xp);
  const updated = [];
  const created = [];
  for (const companion of companions) {
    if (companion.actor) {
      const companionDoc = asMainCard(companion.doc, {actorId: companion.actor.id, masterId: actor.id});
      updated.push({actor: companion.actor, ref: companion.ref, doc: companionDoc, conflicts: []});
    } else {
      const id = companionIdFor(companion.sourceId, companion.doc.name);
      created.push({sourceId: companion.sourceId, id, ref: companion.ref, doc: asMainCard(companion.doc, {actorId: id, masterId: actor.id})});
    }
  }
  return {
    doc: masterDoc, ref, level: levelOf(masterDoc), conflicts: [], updated, created, skipped: skippedCompanions, hpGain: false, consumed,
    title: ref.kind === 'copy' ? '整卡覆盖成玩家「整卡同步」提交的版本' : '整卡覆盖成建卡服现在的版本',
  };
}

/**
 * 套用玩家「整卡同步」提交的整卡副本：副本文档还原成主服卡，同伴取副本记着的同伴副本，交给 `applyWhole`。
 * 套上后这份副本记进 `undone`（用掉了，Ruling 39）：之后主服再套别的，`planForce` 也不会再提它。
 */
export async function applyForce(actor, copy, {meta} = {}) {
  meta = await loadMeta();
  copy = await currentCopy(copy, meta, actor);
  await writeCards(actor, await prepareForce(actor, copy, meta));
  return 'applied';
}

async function prepareForce(actor, copy, meta) {
  const ref = copyRef(copy);
  const doc = normalizeCopy(await requireHistory(ref), {actorId: actor.id, masterId: null});
  const {found, skipped} = await companionCopies(meta, copy);
  const companions = [];
  for (const {sourceId, copy: companionCopy} of found) {
    const ref = copyRef(companionCopy);
    const target = mainActorOf(meta, sourceId);
    const companionDoc = normalizeCopy(await requireHistory(ref), {actorId: target?.id ?? sourceId, masterId: actor.id});
    companions.push({actor: target, sourceId, doc: companionDoc, ref});
  }
  return prepareWhole(actor, {doc, ref}, {companions, skippedCompanions: skipped, consumed: [refKey(ref)]});
}

// 建卡服上主人是这张卡、还没进映射的魔宠（玩家新建的）。要读了 working 才知道主人是谁
export async function unmappedFamiliars(meta, cardId) {
  const owner = meta.working?.[cardId]?.owner;
  const found = [];
  for (const [id, entry] of Object.entries(meta.working ?? {})) {
    if (meta.cards?.[id] || entry.owner !== owner) continue;
    const working = await fetchPublished(`working/${id}.json`);
    if (working?.doc.type === 'familiar' && working.doc.system?.master?.id === cardId) found.push({id, working});
  }
  return found;
}

/**
 * GM 在面板上用建卡服现在的版本（`working/<cardId>.json`）整卡覆盖：主人和同伴（映射里的，加上建卡服新建、
 * 主人是这张卡的魔宠）的文档先上传成主服 Data 的 `history/char-<hash>.json`，再交给 `applyWhole`。
 */
export async function pullWorking(actor) {
  const meta = await loadMeta();
  const cardId = cardIdOf(meta, actor);
  const working = await loadWorking(cardId);
  const companions = [];
  const skipped = [];
  for (const [companionId, card] of Object.entries(meta.cards ?? {})) {
    if (card.companionOf !== cardId) continue;
    const target = mainActorOf(meta, companionId);
    const companionWorking = await fetchPublished(`working/${companionId}.json`);
    if (!companionWorking) {
      skipped.push({id: companionId, name: target?.name, why: '建卡服还没发布它现在的版本'});
      continue;
    }
    const ref = await publishCharVersion(companionId, companionWorking);
    companions.push({actor: target, sourceId: companionId, doc: companionWorking.doc, ref});
  }
  for (const {id, working: familiar} of await unmappedFamiliars(meta, cardId)) {
    const ref = await publishCharVersion(id, familiar);
    companions.push({actor: mainActorOf(meta, id), sourceId: id, doc: familiar.doc, ref});
  }
  const ref = await publishCharVersion(cardId, working);
  return applyWhole(actor, {doc: working.doc, ref}, {companions, skippedCompanions: skipped});
}

async function archiveFolder() {
  const existing = game.folders.find(folder => folder.type === 'Actor' && folder.getFlag(OWN_FLAG, 'role') === 'archive');
  if (existing) return existing;
  const folder = await Folder.implementation.create({name: ARCHIVE_FOLDER, type: 'Actor', flags: {[OWN_FLAG]: {role: 'archive'}}});
  if (!folder) throw new Error(`taka-charsync：没能建出「${ARCHIVE_FOLDER}」文件夹`);
  return folder;
}

// 被新卡替换的旧卡挪进归档文件夹、打上 retired，引擎不再把它放进映射
async function retire(actor) {
  const folder = await archiveFolder();
  const updated = await actor.update({folder: folder.id, [`flags.${OWN_FLAG}.retired`]: true});
  if (!updated) throw new Error(`taka-charsync：${actor.name} 没能挪进「${ARCHIVE_FOLDER}」`);
}

/** 被新卡替换时一起归档的旧主人和同伴，供确认和执行共用。 */
export function replacementActors(meta, replaceId) {
  const master = game.actors.get(replaceId);
  if (!master) throw new Error(`taka-charsync：找不到要替换的主服卡 ${replaceId}`);
  const cardId = cardIdOf(meta, master);
  const ids = new Set([master.id]);
  for (const [sourceId, card] of Object.entries(meta.cards ?? {})) {
    if (card.world === game.world.id && card.companionOf === cardId) ids.add(mainActorOf(meta, sourceId).id);
  }
  for (const actor of game.actors.contents) {
    if (actor.type === 'familiar' && actor._source.system.master?.id === master.id && !ownFlags(actor).retired) ids.add(actor.id);
  }
  for (const id of [...(ownFlags(master).applied?.companions ?? []), ...(ownFlags(master).applied?.createdCompanions ?? [])]) {
    if (game.actors.has(id) && !ownFlags(game.actors.get(id)).retired) ids.add(id);
  }
  return [...ids].map(id => game.actors.get(id));
}

// 新卡先准备整组文档和目标 id，再创建；归档旧卡失败时恢复已改的旧卡，并删除整组新卡。
async function importNewCard({cardId, ref, source, owner, replaceId, meta, companions = []}) {
  if (meta.cards?.[cardId] || mainActorOf(meta, cardId)) throw new Error('这张卡已经有主服对应卡，请刷新同步面板');
  const replaced = replaceId ? game.actors.get(replaceId) : null;
  if (replaceId && !replaced) throw new Error(`taka-charsync：找不到要替换的主服卡 ${replaceId}`);
  if (replaced && !campaignCards(meta).some(entry => entry.actor.id === replaceId && entry.card.charOwner === owner)) {
    throw new Error('替换对象的卡主归属已变化，请刷新同步面板');
  }
  const archived = replaced ? replacementActors(meta, replaceId) : [];
  const archiveSnapshots = new Map(archived.map(actor => [actor.id, {folder: actor._source.folder ?? null, flags: foundry.utils.deepClone(actor._source.flags ?? {})}]));
  const id = freeId(cardId);
  const doc = onMainSide(source, ref, {actorId: id, masterId: null});
  const prepared = companions.map(companion => {
    const companionId = companionIdFor(companion.sourceId, companion.source.name);
    if (companionId === id) throw new Error('主人和同伴的目标 id 冲突，没有动卡');
    return {...companion, id: companionId, doc: onMainSide(companion.source, companion.ref, {actorId: companionId, masterId: id})};
  });
  const worldUser = meta.users?.[owner]?.[game.world.id];
  if (!worldUser) throw new Error('卡主在本团没有对应的账号，不能导入');
  const {NONE, OWNER} = CONST.DOCUMENT_OWNERSHIP_LEVELS;
  const ownership = {default: replaced?.ownership.default ?? NONE, [worldUser]: OWNER};
  const at = Date.now();
  const applied = {ref, at, undoPath: null, level: levelOf(doc)};
  if (prepared.length) applied.createdCompanions = prepared.map(entry => entry.id);
  const created = [], retired = [], reports = [];
  let master;
  try {
    const report = await createActor(doc, {id, ownership, folder: replaced?.folder?.id ?? null, flags: {charId: cardId, applied}});
    master = report.actor;
    created.push(master);
    reports.push(cardReport(report));
    for (const companion of prepared) {
      const companionReport = await createCompanion(companion, master, at);
      created.push(companionReport.actor);
      reports.push(`<p>同伴「${escape(companionReport.actor.name)}」已在主服新建</p>`, cardReport(companionReport));
    }
    for (const actor of archived) {
      retired.push(actor);
      await retire(actor);
    }
  } catch (error) {
    const failures = [];
    for (const actor of retired.reverse()) {
      try {
        const restored = await actor.update(archiveSnapshots.get(actor.id), {recursive: false, takaCharsync: true});
        if (!restored) throw new Error('归档前的文件夹和同步记号没有恢复');
      } catch (restoreError) {
        failures.push(`「${actor.name}」的归档没有恢复：${errorMessage(restoreError)}`);
      }
    }
    const rolledBack = await rollBack(error, {master: master ?? {id, name: doc.name}, written: [], created, snapshots: new Map()});
    if (!failures.length) throw rolledBack;
    await whisperOrWarn(id, `<p>新卡导入失败，旧卡恢复没有做完：${escape(failures.join('；'))}</p>`);
    throw new Error(`${errorMessage(rolledBack)}；旧卡恢复没有做完：${failures.join('；')}`, {cause: error});
  }
  const replacedNote = archived.length ? `，旧卡及同伴${nameList(archived.map(actor => actor.name))}已移进「${ARCHIVE_FOLDER}」` : '';
  await whisperOrWarn(master.id, `<p><strong>${escape(master.name)}</strong>：建卡服的新卡已导入${replacedNote}</p>${reports.join('')}`);
  return master.id;
}

/**
 * 导入建卡服提交的新卡副本，`replaceId` 是要替换的主服卡（不替换为 null）。所有者是提交者在本世界的账号。
 * @returns {Promise<string>} 新卡在主服的 id
 */
export async function applyNew(copy, {replaceId = null, meta} = {}) {
  meta = await loadMeta();
  copy = await currentCopy(copy, meta);
  return importNewCard({...await newCopySources(copy, meta), replaceId});
}

async function newCopySources(copy, meta) {
  const ref = copyRef(copy);
  const source = await requireHistory(ref);
  const {found} = await companionCopies(meta, copy);
  const companions = [];
  for (const {sourceId, copy: companion} of found) {
    const companionRef = copyRef(companion);
    companions.push({sourceId, ref: companionRef, source: await requireHistory(companionRef)});
  }
  return {cardId: copy.cardId, ref, source, owner: copy.submittedBy, meta, companions};
}

/**
 * 不经提交，直接导入建卡服上的卡现在的版本（`working/<cardId>.json`，先上传成 `history/char-<hash>.json`）。
 * 所有者是建卡服卡主在本世界的账号。
 * @returns {Promise<string>} 新卡在主服的 id
 */
export async function importNewFromWorking(cardId, {replaceId = null} = {}) {
  const meta = await loadMeta();
  const working = await loadWorking(cardId);
  const familiars = await unmappedFamiliars(meta, cardId);
  const companions = [];
  for (const {id, working: familiar} of familiars) {
    companions.push({sourceId: id, ref: await publishCharVersion(id, familiar), source: familiar.doc});
  }
  const ref = await publishCharVersion(cardId, working);
  return importNewCard({cardId, ref, source: working.doc, owner: meta.working?.[cardId]?.owner ?? working.owner, replaceId, meta, companions});
}

/**
 * 恢复到最后一次套用前：主人和那次一起写的已有同伴从各自的快照整卡导入，同步记号换成快照里的，
 * 那次套用的版本记进 `undone`（之后不再自动套）；那次新建的同伴删掉。快照全部取到才开始写；
 * 写到一半失败时给 GM 发悄悄话，写明恢复到了哪一步，再抛错。
 */
export async function undoLast(actor) {
  const {applied} = ownFlags(actor);
  if (!applied?.undoPath) throw new Error(`taka-charsync：${actor.name} 没有套用前的快照`);
  const targets = [{actor, applied}];
  for (const id of applied.companions ?? []) {
    const companion = game.actors.get(id);
    const companionApplied = companion ? ownFlags(companion).applied : null;
    if (companionApplied?.at === applied.at && companionApplied.undoPath) targets.push({actor: companion, applied: companionApplied});
  }
  const restores = [];
  for (const target of targets) {
    const snapshot = await fetchPublished(target.applied.undoPath);
    if (!snapshot) throw new Error(`taka-charsync：找不到 ${target.actor.name} 套用前的快照 ${target.applied.undoPath}`);
    restores.push({...target, snapshot});
  }
  const created = (applied.createdCompanions ?? []).map(id => game.actors.get(id)).filter(Boolean);
  let done = 0;
  try {
    for (const {actor: target, applied: targetApplied, snapshot} of restores) {
      const restored = snapshot.flags?.[OWN_FLAG] ?? {};
      const undone = [...new Set([...(ownFlags(target).undone ?? []), ...(restored.undone ?? []), refKey(targetApplied.ref)])];
      await importFull(target, snapshot, {flags: {...restored, undone}});
      done++;
    }
    if (created.length) await Actor.implementation.deleteDocuments(created.map(companion => companion.id));
  } catch (error) {
    const names = restores.map(entry => entry.actor.name);
    const createdNote = created.length ? `；那次新建的同伴${nameList(created.map(companion => companion.name))}没删掉` : '';
    await whisperOrWarn(actor.id, `<p><strong>${escape(actor.name)}</strong>：恢复到套用前中途失败（${escape(errorMessage(error))}）。`
      + `已恢复：${nameList(names.slice(0, done))}；没恢复：${nameList(names.slice(done))}${createdNote}</p>`);
    throw error;
  }
}

/** 忽略一份副本（谱系失效的那种）：记进这张卡的 `undone`，之后不再提。 */
export async function ignoreCopy(actor, copy) {
  const undone = ownFlags(actor).undone ?? [];
  const key = refKey(copyRef(copy));
  if (undone.includes(key)) return;
  const updated = await actor.update({[`flags.${OWN_FLAG}.undone`]: [...undone, key]});
  if (!updated) throw new Error(`taka-charsync：${actor.name} 的忽略记号没有写上`);
}

/**
 * 手动处理 `runApplyCheck` 给出的一项：升级副本手动套用（有冲突也套），整卡副本整卡覆盖，新卡按玩家选的替换对象导入。
 * 手动套用成功后，这张卡恢复自动套用。这一项已经套用过时什么都不做，返回 'already'。
 */
export async function applyPending(item, {meta} = {}) {
  if (!item.copy) throw new Error('taka-charsync：这一项没有可以套用的副本');
  meta ??= await loadMeta();
  // 面板上的列表可能是几秒前的，定时检查也许已经套上了；再套一遍会把撤回点换成套用后的样子
  if (item.copy.kind === 'new') {
    if (mainActorOf(meta, item.copy.cardId)) return 'already';
    return applyNew(item.copy, {replaceId: replaceTarget(meta, item.copy), meta});
  }
  const actor = game.actors.get(item.actorId);
  if (!actor) throw new Error(`taka-charsync：找不到主服卡 ${item.actorId}`);
  const ref = copyRef(item.copy);
  const own = ownFlags(actor);
  if (sameRef(own.applied?.ref ?? null, ref)) return 'already';
  if (item.copy.kind === 'force') return own.undone?.includes(refKey(ref)) ? 'already' : applyForce(actor, item.copy, {meta});
  const outcome = await applyCopy(actor, item.copy, {manual: true, meta});
  failedCards.delete(actor.id);
  return outcome;
}

function preparePreviewActor(doc, master = null) {
  const source = foundry.utils.deepClone(doc);
  if (doc.type !== 'familiar') return new Actor.implementation(source);
  // PF2e 的魔宠构造时就会读取主人。局部子类让整次准备使用预览主人，也避免把临时魔宠缓存到真实主人上。
  class PreviewFamiliar extends CONFIG.PF2E.Actor.documentClasses.familiar {
    get master() { return master; }
  }
  return new PreviewFamiliar(source);
}

function previewCard({actor = null, doc, conflicts = [], hpGain = false, master = null, prepared = null}, label) {
  const before = actor ? preparePreviewActor(actor.toObject(), master).toObject() : {};
  prepared ??= preparePreviewActor(doc, master);
  const after = prepared.toObject();
  if (hpGain && actor) {
    const hp = hpOf(actor), max = prepared.system.attributes?.hp?.max;
    if (hp && max != null && max !== hp.max && after.system.attributes?.hp) {
      after.system.attributes.hp.value = Math.clamp(hp.value + max - hp.max, 0, max);
    }
  }
  const summary = describeChanges(before, after);
  return `<section><h3>${escape(label)}「${escape(doc.name)}」</h3>`
    + (summary.html || '<p>没有内容变动。</p>') + formatSummary(NO_CHANGES, conflicts) + '</section>';
}

/** 按实际套用计划预览主人和同伴；整卡覆盖保留主服经验，普通提交使用三方合并结果。 */
export async function previewPending(item) {
  const meta = await loadMeta();
  const actor = item.actorId ? game.actors.get(item.actorId) : null;
  const copy = await currentCopy(item.copy, meta, actor);
  if (copy.kind === 'new') {
    const {ref, source, companions} = await newCopySources(copy, meta);
    const doc = onMainSide(source, ref, {actorId: copy.cardId, masterId: null});
    const master = preparePreviewActor(doc);
    const parts = [previewCard({doc, prepared: master}, '新增主人')];
    for (const companion of companions) {
      parts.push(previewCard({doc: onMainSide(companion.source, companion.ref, {actorId: companion.sourceId, masterId: copy.cardId}), master}, '新增同伴'));
    }
    return parts.join('');
  }
  const plan = copy.kind === 'force' ? await prepareForce(actor, copy, meta) : await prepareLevel(actor, copy, meta, {preview: true});
  const master = plan.previewMaster ?? preparePreviewActor(plan.doc);
  return [previewCard({actor, ...plan, prepared: master}, '更新主人'),
    ...plan.updated.map(entry => previewCard({...entry, hpGain: plan.hpGain, master}, '更新同伴')),
    ...plan.created.map(entry => previewCard({...entry, master}, '新增同伴'))].join('');
}
