// 同步引擎：每轮读两边的库副本，把主服版本、冻结副本、玩家卡和映射表发布成两台 Data 下的静态 JSON，
// 重写建卡服的规则表，镜像卡上引用的图片。引擎从不写 LevelDB。
// 用法：node engine/main.mjs --config <path> [--once] [--dry]
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { canonicalJson, contentHash } from '../scripts/lib/canon.mjs';
import { historyFile, refKey } from '../scripts/lib/chain.mjs';
import { OWN_FLAG } from '../scripts/lib/rules.mjs';
import { copyRuleError } from '../scripts/lib/rule-validation.mjs';
import { resolveActorCampaign } from '../scripts/lib/campaign-rules.mjs';
import { bootstrapBases, bootstrapTargets } from './bootstrap.mjs';
import { isFoundryId, loadConfig, loadSeed } from './config.mjs';
import { collectImagePaths, ensureManagedMediaRoot, mirrorImages, snapshotMainMedia, sourceMediaHashes } from './images.mjs';
import { buildMapping, charOwnerOf } from './mapping.mjs';
import { gcHistory, mirrorCharHistory, publishAll, writeJsonAtomic, writeTextAtomic } from './publish.mjs';
import { buildRuleSnapshot, renderRulesModule, RULE_KEYS } from './rules-table.mjs';
import { cleanStaleSnapshots, mainActiveWorld, readWorld, Throttle, worldFingerprint } from './sources.mjs';

const DAY_MS = 24 * 60 * 60 * 1000;
const SCHEMA_KEY = 'pf2e.worldSchemaVersion';
const REF_KINDS = new Set(['main', 'copy', 'char']);

const ownFlags = doc => doc?.flags?.[OWN_FLAG] ?? {};
const copyRef = copy => ({kind: 'copy', copyId: copy.copyId, rev: copy.rev});
const levelOf = doc => doc.system?.details?.level?.value ?? null;
const errorMessage = error => (error instanceof Error ? error.message : String(error));
// 副本记号由玩家的客户端写入，id 和 rev 要拼进发布路径
const validCopy = copy => isFoundryId(copy.copyId) && isFoundryId(copy.cardId) && Number.isSafeInteger(copy.rev);

// 各世界几千个 actor 的整行常驻内存太占地方，只留映射和历史清理要看的字段
const slimMainRow = row => ({_id: row._id, flags: {[OWN_FLAG]: ownFlags(row)}});
const slimCharRow = row => ({
  _id: row._id, type: row.type, folder: row.folder, ownership: row.ownership,
  system: {master: {id: row.system?.master?.id}}, flags: {[OWN_FLAG]: ownFlags(row)},
});
const slimRows = (rows, slim) => new Map([...rows].map(([id, row]) => [id, slim(row)]));

function readJsonIfExists(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

function listDir(dir) {
  try {
    return fs.readdirSync(dir);
  } catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
}

/** 手动同步用的玩家卡的卡主：角色或魔宠、不是冻结副本也不是草稿备份、有非 GM 卡主。不是玩家卡时为 null。 */
function playerOwnerOf(row, users) {
  if (row.type !== 'character' && row.type !== 'familiar') return null;
  const {copy, backupOf} = ownFlags(row);
  return copy || backupOf ? null : charOwnerOf(row, users);
}

/**
 * 还没算起步基线、这一轮要读建卡服文档的卡：种子卡、种子里列出的同伴，以及主人是种子卡的魔宠。
 * 映射认魔宠还要主服有同 id 的卡，这里还不知道，先都读上；算不算基线以映射为准。
 */
function pendingCharTargets(ctx, rows) {
  const seedIds = new Set(ctx.seed.cards.map(card => card.charId));
  const ids = new Set(Object.keys(bootstrapTargets(ctx.seed).worlds));
  for (const row of rows.values()) {
    if (row.type === 'familiar' && seedIds.has(row.system?.master?.id)) ids.add(row._id);
  }
  return [...ids].filter(id => !(id in ctx.bootstrap));
}

/**
 * 一个引擎进程的状态：配置、种子表、读取节流，以及各库最后一次读到并发布成功的结果，没读的库沿用它们。
 * `dry` 时一切写入落到 `dryOutDir` 下的 `main/`、`char/` 两个假 Data 目录和 `state/`，库和图片仍从配置里的 Data 读。
 */
export function createContext(config, seed, {dry = false} = {}) {
  if (dry && !config.dryOutDir) throw new Error('--dry 要在配置里写 dryOutDir');
  const out = dry
    ? {main: path.join(config.dryOutDir, 'main'), char: path.join(config.dryOutDir, 'char'), state: path.join(config.dryOutDir, 'state')}
    : {main: config.main.dataDir, char: config.char.dataDir, state: config.stateDir};
  const lastMeta = readJsonIfExists(path.join(out.main, config.publishDir, 'meta.json')) ?? {};
  const lastRules = readJsonIfExists(path.join(out.state, 'rules.json'));
  const enforcement = readJsonIfExists(path.join(out.state, 'rules-enforcement.json'));
  const enforcementDates = [enforcement?.enforcedSince, lastRules?.version === 2 && lastRules.enforcedSince,
    lastMeta.ruleSnapshot?.version === 2 && lastMeta.ruleSnapshot.enforcedSince].filter(value => Number.isSafeInteger(value) && value > 0);
  return {
    config,
    seed,
    out,
    throttle: new Throttle(config.intervalSeconds * 1000),
    mediaPending: new Set(),
    worlds: {},
    char: null,
    // 进程刚起来时没有上一轮，没读到的世界沿用磁盘上最后发布的 meta.json
    lastMeta,
    lastRules,
    enforcedSince: enforcementDates.length ? Math.min(...enforcementDates) : null,
    bootstrap: readJsonIfExists(path.join(out.state, 'bootstrap.json')) ?? {},
    // 还没算起步基线的卡，两边各自最近读到的文档；两边都有了才算
    bootstrapDocs: {main: new Map(), char: new Map()},
    lastGcAt: readJsonIfExists(path.join(out.state, 'gc.json'))?.at ?? 0,
  };
}

const readOptions = config => ({tmpRoot: config.tmpRoot, foundryAppDir: config.foundryAppDir});

/**
 * 建卡服的库该读时读一次（建卡服开着也读）：映射要的 actor 行和用户、规则表要的设置、冻结副本、玩家卡，
 * 以及还没算起步基线的卡。条目不全的副本和玩家卡沿用上次发布的版本、名字记进报告，这一轮不算读完，下一轮再读。
 * 不该读或读失败时返回 null。
 */
async function readChar(ctx, run) {
  const {config, out} = ctx;
  const {dataDir, world} = config.char;
  try {
    const fp = worldFingerprint(dataDir, world, {includeFolders: true});
    if (!ctx.mediaPending.has('char') && !ctx.throttle.due('char', fp, run.now)) return null;
    let pendingTargets = [];
    const publishedOnBoth = rel => [out.main, out.char].every(data => fs.existsSync(path.join(data, config.publishDir, rel)));
    const {actors, users, settings, folders, docs} = await readWorld(dataDir, world, {...readOptions(config), includeFolders: true}, (rows, rowUsers) => {
      pendingTargets = pendingCharTargets(ctx, rows);
      const ids = [...pendingTargets];
      for (const row of rows.values()) {
        const {copy} = ownFlags(row);
        // 副本重交会换 rev，已经发布过的那一版内容不会再变，不用再读
        const savedCopy = copy && validCopy(copy) && [out.main, out.char].some(data => fs.existsSync(path.join(data, config.publishDir, historyFile(copyRef(copy)))));
        if (copy ? validCopy(copy) && !savedCopy : playerOwnerOf(row, rowUsers)) ids.push(row._id);
      }
      return ids;
    });

    let complete = true;
    const skip = row => {
      complete = false;
      run.report.copiesSkippedIncomplete.push(row.name);
    };

    const previousLists = new Map();
    const previousCopy = (cardId, copyId) => {
      if (!previousLists.has(cardId)) {
        previousLists.set(cardId, readJsonIfExists(path.join(out.main, config.publishDir, 'copies', `${cardId}.json`)) ?? []);
      }
      return previousLists.get(cardId).find(copy => copy.copyId === copyId);
    };
    const copyLists = {};
    for (const row of actors.values()) {
      const {copy} = ownFlags(row);
      if (!copy) continue;
      if (!validCopy(copy)) {
        run.fail(world, new Error(`副本 ${row._id} 的记号缺少合法的 copyId、cardId 或 rev，没有发布`));
        continue;
      }
      let listed = copy;
      const read = docs.get(row._id);
      if (read?.complete) {
        run.history.push({ref: copyRef(copy), doc: read.doc});
        run.toMain.push(read.doc);
      } else if (read) {
        skip(row);
        listed = copy.withdrawn ? copy : previousCopy(copy.cardId, copy.copyId);
      }
      if (listed) (copyLists[copy.cardId] ??= []).push(listed);
    }

    const previousWorking = ctx.char?.working ?? ctx.lastMeta.working ?? {};
    const working = {};
    const workingMeta = {};
    for (const row of actors.values()) {
      const owner = playerOwnerOf(row, users);
      if (!owner) continue;
      const read = docs.get(row._id);
      const previous = previousWorking[row._id];
      if (!read.complete) {
        skip(row);
        if (previous) workingMeta[row._id] = previous;
        continue;
      }
      const hash = await contentHash(read.doc);
      run.toMain.push(read.doc);
      // 建卡服开着时每轮都读，内容和卡主都没变、两台都有文件的不重写
      if (previous?.hash === hash && previous.owner === owner && publishedOnBoth(`working/${row._id}.json`)) {
        workingMeta[row._id] = previous;
        continue;
      }
      const meta = {name: read.doc.name, owner, level: levelOf(read.doc), readAt: run.now, hash};
      workingMeta[row._id] = meta;
      working[row._id] = {hash, readAt: run.now, level: meta.level, owner, name: meta.name, doc: read.doc};
    }

    for (const id of pendingTargets) {
      const read = docs.get(id);
      if (!read) ctx.bootstrapDocs.char.set(id, null);
      else if (read.complete) ctx.bootstrapDocs.char.set(id, read.doc);
      else complete = false;
    }

    if (complete) run.marks.push(['char', fp]);
    run.report.charRead = true;
    return {
      state: {actors: slimRows(actors, slimCharRow), users, settings, folders, readAt: run.now, copyLists, working: workingMeta},
      working,
    };
  } catch (error) {
    run.fail(world, error);
    for (const id of config.worlds) run.unavailableRules.add(id);
    return null;
  }
}

/**
 * 一个团世界的库该读时读一次：actor 行、用户、设置，以及本世界映射卡的完整文档（带内容哈希）。
 * 不该读或读失败时返回 null。
 */
async function readMainWorld(ctx, run, world, charState) {
  const {config, seed} = ctx;
  try {
    const dbFingerprint = worldFingerprint(config.main.dataDir, world);
    const previous = ctx.worlds[world];
    const mediaHashes = sourceMediaHashes(Object.keys(previous?.legacyHashes ?? {}), config.main.dataDir, {publishDir: config.publishDir});
    const fp = `${dbFingerprint}|${canonicalJson(mediaHashes)}`;
    if (previous && !previous.mediaFailed && !ctx.mediaPending.has(`main:${world}`)
        && !ctx.throttle.due(`main:${world}`, fp, run.now)) return null;
    let cards = {};
    const read = await readWorld(config.main.dataDir, world, readOptions(config), (actors, users) => {
      // 认魔宠要建卡服的行、认新卡和退役要本世界的行，与这一轮最后发布的映射同一套算法
      cards = buildMapping({seed, charActors: charState.actors, charUsers: charState.users, mainActors: {[world]: actors}, worldUsers: {[world]: users}}).cards;
      return Object.values(cards).filter(card => card.world === world).map(card => card.mainId);
    });
    for (const row of read.actors.values()) {
      const {charId} = ownFlags(row);
      if (charId != null && !isFoundryId(charId)) {
        run.fail(world, new Error(`主服 actor ${row._id} 的 charId 记号不是合法的卡 id，没有收进映射：${JSON.stringify(charId)}`));
      }
    }
    const docs = new Map();
    const legacyHashes = {};
    let mediaFailed = false;
    for (const [id, {doc, complete}] of read.docs) {
      if (!complete) { docs.set(id, {doc, complete, hash: null}); continue; }
      try {
        const frozen = await snapshotMainMedia(doc, config.main.dataDir, ctx.out.main, {publishDir: config.publishDir});
        Object.assign(legacyHashes, frozen.legacyHashes);
        docs.set(id, {doc: frozen.doc, complete, hash: await contentHash(frozen.doc)});
      } catch (error) {
        mediaFailed = true;
        docs.set(id, {doc, complete, hash: null, mediaError: true});
        run.fail('images', new Error(`主服卡 ${id} 的媒体未就绪，没有发布新版本：${errorMessage(error)}`, {cause: error}));
      }
    }

    for (const [id, targetWorld] of Object.entries(bootstrapTargets(seed, cards).worlds)) {
      if (targetWorld !== world || id in ctx.bootstrap) continue;
      const entry = cards[id]?.world === world ? docs.get(cards[id].mainId) : undefined;
      if (!entry) ctx.bootstrapDocs.main.set(id, null);
      else if (entry.complete && !entry.mediaError) ctx.bootstrapDocs.main.set(id, {doc: entry.doc, hash: entry.hash});
    }

    run.marks.push([`main:${world}`, `${dbFingerprint}|${canonicalJson(legacyHashes)}`]);
    run.report.worldsRead.push(world);
    return {state: {actors: slimRows(read.actors, slimMainRow), users: read.users, settings: read.settings,
      readAt: run.now, legacyHashes, mediaFailed}, docs};
  } catch (error) {
    run.fail(world, error);
    run.unavailableRules.add(world);
    return null;
  }
}

/**
 * 本轮的映射。这个进程还没读到过的团世界（正在开，或一直读不出来）没有主服的行，认不出新卡和魔宠，
 * 它的映射条目和账号对照沿用上次发布的 meta.json。
 */
function currentMapping(ctx, charState, worldStates) {
  const mainActors = {};
  const worldUsers = {};
  for (const [world, state] of Object.entries(worldStates)) {
    mainActors[world] = state.actors;
    worldUsers[world] = state.users;
  }
  const mapping = buildMapping({seed: ctx.seed, charActors: charState.actors, charUsers: charState.users, mainActors, worldUsers});
  const unread = new Set(ctx.config.worlds.filter(world => !worldStates[world]));
  for (const [cardId, card] of Object.entries(ctx.lastMeta.cards ?? {})) {
    const current = mapping.cards[cardId];
    if (unread.has(card.world) && (!current || unread.has(current.world))) mapping.cards[cardId] = card;
  }
  for (const [charUser, pairs] of Object.entries(ctx.lastMeta.users ?? {})) {
    for (const [world, worldUser] of Object.entries(pairs)) {
      if (unread.has(world)) (mapping.users[charUser] ??= {})[world] ??= worldUser;
    }
  }
  return mapping;
}

function worldsMeta(ctx, worldStates) {
  const worlds = {};
  for (const world of ctx.config.worlds) {
    const state = worldStates[world];
    if (state) worlds[world] = {title: ctx.config.titles[world] ?? world, schema: state.settings.get(SCHEMA_KEY) ?? null, readAt: state.readAt};
    else if (ctx.lastMeta.worlds?.[world]) worlds[world] = ctx.lastMeta.worlds[world];
  }
  return worlds;
}

function mainEntry({doc, hash}, readAt) {
  const xp = doc.system?.details?.xp;
  return {hash, readAt, level: levelOf(doc), xp: xp ? {value: xp.value, max: xp.max} : null, applied: ownFlags(doc).applied?.ref ?? null, doc};
}

/** 每张映射卡、每张有副本的卡、发布目录里已有 `copies/` 文件的卡都写一份，没有副本的写 `[]`，清理掉的副本才不会一直列着。 */
function copiesToPublish(ctx, copyLists, mapping) {
  const existing = listDir(path.join(ctx.out.main, ctx.config.publishDir, 'copies'))
    .filter(name => name.endsWith('.json'))
    .map(name => name.slice(0, -'.json'.length));
  const copies = {};
  for (const cardId of new Set([...Object.keys(mapping.cards), ...Object.keys(copyLists), ...existing])) copies[cardId] = copyLists[cardId] ?? [];
  return copies;
}

/**
 * 拼进发布路径（`main/<w>/`、`copies/`、`working/`）的卡 id 最后再查一遍：映射里有从上次 meta.json 沿用来的条目，
 * 不能只靠上游的校验。不是 Foundry id 的丢掉并记错。
 */
function keepFoundryIds(entries, idOf, folder, fail) {
  return entries.filter(entry => {
    if (isFoundryId(idOf(entry))) return true;
    fail('publish', new Error(`${folder} 下的卡 id 不合法，没有发布：${JSON.stringify(idOf(entry))}`));
    return false;
  });
}

/**
 * 两边文档都到齐了的卡算起步基线；`char-unchanged` 的建卡服文档放进这一轮要发布的历史。
 * `missing` 的不收：卡之后可能才出现在主服或建卡服，每轮重算到算出来为止。
 */
async function newBootstrapBases(ctx, run, mapping) {
  const {worlds: targetWorlds, forced} = bootstrapTargets(ctx.seed, mapping.cards);
  const {main, char} = ctx.bootstrapDocs;
  const ready = Object.keys(targetWorlds).filter(id => !(id in ctx.bootstrap) && main.has(id) && (forced[id] === 'main' || char.has(id)));
  if (!ready.length) return {};
  const known = docs => Object.fromEntries(ready.filter(id => docs.get(id)).map(id => [id, docs.get(id)]));
  const bases = await bootstrapBases({
    cards: ready, mainDocs: known(main), charDocs: known(char), since: Date.parse(ctx.seed.bootstrapSince), forced,
  });
  const entries = {};
  for (const [id, base] of Object.entries(bases)) {
    if (base.case === 'missing') continue;
    if (base.historyDoc) {
      run.history.push({ref: base.ref, doc: base.historyDoc});
      run.toMain.push(base.historyDoc);
    }
    entries[id] = {case: base.case, ref: base.ref};
  }
  return entries;
}

/** 一个方向的闭包镜像；按文档返回是否就绪，失败不阻断无关卡。 */
function mirrorDocImages(docs, fromData, targets, publishDir, fail) {
  const pathsByDoc = new Map(docs.map(doc => [doc, collectImagePaths(doc)]));
  const ready = new Map([...pathsByDoc.keys()].map(doc => [doc, true]));
  try {
    const paths = [...new Set([...pathsByDoc.values()].flat())];
    if (!paths.length) return ready;
    for (const toData of new Set(targets)) {
      const {copied, present, missing, refused, failed} = mirrorImages(paths, fromData, toData, {publishDir});
      const available = new Set([...copied, ...present]);
      for (const [doc, refs] of pathsByDoc) {
        if (refs.some(rel => !available.has(rel))) ready.set(doc, false);
      }
      for (const rel of missing) fail('images', new Error(`缺少图片来源 ${JSON.stringify(rel)}（${fromData}），没有发布新版本`));
      for (const rel of refused) {
        fail('images', new Error(`图片路径不合要求（不是图片或视频、越出 Data 根、落进共享目录或发布目录），没有拷贝：${JSON.stringify(rel)}`));
      }
      for (const {path: rel, message} of failed) fail('images', new Error(`拷贝图片 ${JSON.stringify(rel)} 到 ${toData} 失败：${message}`));
    }
  } catch (error) {
    fail('images', error);
    for (const doc of ready.keys()) ready.set(doc, false);
  }
  return ready;
}

/** 已有 history 才是冻结内容；缺一侧时也不能用同 rev 的可变数据库行补写。 */
function copyMedia(ctx, copies, run) {
  const docs = new Map(run.history.filter(({ref}) => ref.kind === 'copy').map(({ref, doc}) => [refKey(ref), doc]));
  const previous = new Map();
  const load = copy => {
    const ref = copyRef(copy);
    const key = refKey(ref);
    if (docs.has(key)) return;
    try {
      const files = [ctx.out.main, ctx.out.char].map(data => path.join(data, ctx.config.publishDir, historyFile(ref)));
      const doc = files.map(readJsonIfExists).find(Boolean);
      if (!doc) return;
      docs.set(key, doc);
      if (files.some(file => !fs.existsSync(file))) run.history.push({ref, doc});
    } catch (error) {
      run.fail('images', new Error(`读取副本 ${copy.copyId} 的冻结历史失败：${errorMessage(error)}`));
    }
  };
  for (const [cardId, list] of Object.entries(copies)) {
    const oldList = readJsonIfExists(path.join(ctx.out.main, ctx.config.publishDir, 'copies', `${cardId}.json`)) ?? [];
    for (const copy of oldList) {
      previous.set(copy.copyId, copy);
      load(copy);
    }
    for (const copy of list) load(copy);
  }
  run.toMain.push(...docs.values());
  return {docs, previous};
}

/** 主人声明的同伴必须全部是同一批提交；同伴不能单独变成可套用版本。 */
function readyCopyGroups(copies, ready) {
  const all = Object.values(copies).flat();
  const byId = new Map(all.map(copy => [copy.copyId, copy]));
  const accepted = new Set();
  for (const copy of all) {
    if (copy.companionOf || copy.withdrawn) continue;
    const group = [copy];
    let complete = true;
    for (const ref of copy.companions ?? []) {
      const child = byId.get(ref.copyId);
      if (!child || child.rev !== ref.rev || child.cardId !== ref.sourceId || child.companionOf !== copy.copyId
          || child.kind !== copy.kind || child.submittedBy !== copy.submittedBy || child.targetWorld !== copy.targetWorld
          || child.withdrawn) {
        complete = false;
        break;
      }
      group.push(child);
    }
    if (complete && group.every(member => ready.has(refKey(copyRef(member))))) {
      for (const member of group) accepted.add(refKey(copyRef(member)));
    }
  }
  return accepted;
}

/** 只恢复完整且哈希一致的 v2 团规；旧差值表无法补成完整表。 */
function cachedWorldSettings(snapshot, world) {
  const rules = snapshot?.rules?.[world];
  if (snapshot?.version !== 2 || !rules || !RULE_KEYS.every(key => Object.hasOwn(rules, key))) return null;
  const settings = new Map(Object.entries(rules));
  const restored = buildRuleSnapshot({charSettings: new Map(), worldSettings: {[world]: settings}});
  return restored.hashes[world] === snapshot.hashes?.[world] && canonicalJson(restored.rules[world]) === canonicalJson(rules) ? settings : null;
}

function currentRuleSnapshot(ctx, charState, worldStates, mapping, run) {
  const worldSettings = {};
  for (const world of ctx.config.worlds) {
    if (run.unavailableRules.has(world)) continue;
    const settings = worldStates[world]?.settings ?? cachedWorldSettings(ctx.lastRules, world);
    if (settings) worldSettings[world] = settings;
  }
  const cardWorld = Object.fromEntries(Object.entries(mapping.cards).filter(([cardId]) => isFoundryId(cardId)).map(([cardId, card]) => [cardId, card.world]));
  const next = buildRuleSnapshot({
    charSettings: charState.settings, worldSettings, cardWorld, folderWorld: ctx.config.campaignFolders ?? {},
    generatedAt: run.now, enforcedSince: ctx.enforcedSince ?? run.now,
  });
  if (ctx.lastRules && canonicalJson({...next, generatedAt: ctx.lastRules.generatedAt}) === canonicalJson(ctx.lastRules)) {
    return ctx.lastRules;
  }
  return next;
}

/** 新提交必须能独立追溯到工作卡；冻结副本的 targetWorld 不能反过来证明工作卡归属。 */
function copySourceRuleError(copy, snapshot, charState) {
  if (!copy.rulesHash && copy.rev < snapshot.enforcedSince) return null;
  const actor = charState.actors.get(copy.cardId);
  const flags = ownFlags(actor);
  if (!actor || !['character', 'familiar'].includes(actor.type) || flags.copy || flags.backupOf) {
    return '找不到对应的角色工作卡，请从工作卡重新提交';
  }
  const resolved = resolveActorCampaign(actor, {snapshot, actors: charState.actors, folders: charState.folders});
  if (!resolved.ok) return `工作卡归属不可用（${resolved.reason}），请检查团文件夹或角色关联后重新提交`;
  if (resolved.world !== copy.targetWorld) return '工作卡所属团与副本提交目标不同，请核对角色归属后重新提交';
  return null;
}

const rulesFile = ctx => path.join(ctx.out.char, 'modules', 'taka-charsync', 'generated', 'rules.mjs');

/** 文件不能跨目录原子替换。失败时逐处撤销可用规则；撤销也失败的路径必须保留在错误报告里。 */
function invalidateRules(ctx, meta, fail) {
  const snapshot = {...meta.ruleSnapshot, rules: {}, hashes: {}};
  const blockedMeta = {
    char: {}, worlds: {}, cards: {}, users: {}, bootstrap: {}, working: {},
    ...ctx.lastMeta, generatedAt: meta.generatedAt, ruleSnapshot: snapshot, published: [],
  };
  const files = [
    ...[ctx.out.main, ctx.out.char].map(data => [path.join(data, ctx.config.publishDir, 'meta.json'), JSON.stringify(blockedMeta)]),
    [rulesFile(ctx), renderRulesModule(snapshot)],
    [path.join(ctx.out.state, 'rules.json'), JSON.stringify(snapshot, null, 2)],
  ];
  let complete = true;
  for (const [file, text] of files) {
    try {
      writeTextAtomic(file, text);
    } catch (error) {
      complete = false;
      fail('rules', new Error(`规则不可用状态写入失败 ${file}：${errorMessage(error)}`));
    }
  }
  ctx.lastRules = snapshot;
  ctx.lastMeta = blockedMeta;
  return complete;
}

/**
 * 清理两台 Data 的 `history/`。留下仍被引用的：工作卡的 base、副本的 pred、列着的副本本身、主服卡的 `applied`、
 * 起步基线，以及发布目录里每份 `main/<w>/*.json` 的版本和 `applied`（含这个进程没读到过的世界）。
 */
function collectGarbage(ctx, {charState, worldStates, copies, meta}, now, fail) {
  const {out, config} = ctx;
  const keep = new Set();
  const add = ref => {
    if (REF_KINDS.has(ref?.kind)) keep.add(historyFile(ref));
  };
  for (const row of charState.actors.values()) {
    const {base, copy} = ownFlags(row);
    add(base);
    add(copy?.pred);
  }
  for (const list of Object.values(copies)) {
    for (const copy of list) {
      add(copyRef(copy));
      add(copy.pred);
    }
  }
  for (const state of Object.values(worldStates)) {
    for (const row of state.actors.values()) add(ownFlags(row).applied?.ref);
  }
  for (const ref of Object.values(meta.bootstrap)) add(ref);
  const mainDir = path.join(out.main, config.publishDir, 'main');
  for (const world of listDir(mainDir)) {
    for (const name of listDir(path.join(mainDir, world)).filter(file => file.endsWith('.json'))) {
      const entry = JSON.parse(fs.readFileSync(path.join(mainDir, world, name), 'utf8'));
      add({kind: 'main', hash: entry.hash});
      add(entry.applied);
    }
  }
  for (const dataDir of [out.main, out.char]) gcHistory({dataDir, publishDir: config.publishDir, keep, maxAgeDays: config.historyDays, now});
  try {
    cleanStaleSnapshots(config.tmpRoot, now);
  } catch (error) {
    fail('gc', new Error(`清理残留快照失败：${errorMessage(error)}`));
  }
  writeJsonAtomic(path.join(out.state, 'gc.json'), {at: now});
  ctx.lastGcAt = now;
}

/**
 * @typedef {{
 *   mainActive: string|null, worldsRead: string[], charRead: boolean, copiesPublished: number,
 *   copiesSkippedIncomplete: string[], errors: {scope: string, message: string}[]
 * }} TickReport
 * `copiesPublished` 是这一轮新写出历史文件的冻结副本数；`copiesSkippedIncomplete` 是条目不全、这一轮沿用上次版本的
 * 副本和玩家卡的名字；`errors` 的 `scope` 是世界 id，或 `main-status`、`publish`、`images`、`rules`、`undo`、
 * `char-history`、`bootstrap`、`gc`。
 */

/**
 * 跑一轮：主服状态 → 建卡服 → 各团世界 → 映射、团规、起步基线 → 图片 → 发布 → 撤回目录、`history/char-*` 镜像 → 每天一次清理。
 * 正在开的团世界不读；单个库出错只记进报告，不影响别的库。读到的结果和节流记录在发布成功后才记下，发布失败下一轮重读。
 * 建卡服在这个进程里一次都没读到时不读主服、不发布新版本；已有规则快照会标为不可用。
 * @returns {Promise<TickReport>}
 */
export async function tick(ctx, now = Date.now()) {
  const {config, out} = ctx;
  const report = {mainActive: null, worldsRead: [], charRead: false, copiesPublished: 0, copiesSkippedIncomplete: [], errors: []};
  const fail = (scope, error) => report.errors.push({scope, message: errorMessage(error)});
  const run = {now, report, fail, marks: [], history: [], toChar: [], toMain: [], unavailableRules: new Set()};

  for (const dataDir of new Set([out.main, out.char])) {
    try {
      ensureManagedMediaRoot(dataDir);
    } catch (error) {
      fail('images', error);
    }
  }

  let mainReadable = true;
  try {
    report.mainActive = await mainActiveWorld(config.main.statusUrl);
  } catch (error) {
    mainReadable = false;
    for (const world of config.worlds) run.unavailableRules.add(world);
    fail('main-status', error);
  }

  // 建卡服先读：各团世界选读哪些卡要用它的 actor 行认魔宠
  const freshChar = await readChar(ctx, run);
  const charState = freshChar?.state ?? ctx.char;
  if (!charState) {
    const previous = ctx.lastMeta.ruleSnapshot ?? ctx.lastRules;
    if (previous?.version === 2) invalidateRules(ctx, {
      ...ctx.lastMeta, generatedAt: now, ruleSnapshot: {...previous, generatedAt: now},
    }, fail);
    return report;
  }

  const freshWorlds = {};
  if (mainReadable) {
    for (const world of config.worlds) {
      if (world === report.mainActive) continue;
      const fresh = await readMainWorld(ctx, run, world, charState);
      if (fresh) freshWorlds[world] = fresh;
    }
  }
  const worldStates = {...ctx.worlds};
  for (const [world, fresh] of Object.entries(freshWorlds)) worldStates[world] = fresh.state;
  const mapping = currentMapping(ctx, charState, worldStates);
  const ruleSnapshot = currentRuleSnapshot(ctx, charState, worldStates, mapping, run);

  let mains = [];
  for (const [world, {docs}] of Object.entries(freshWorlds)) {
    const mainIds = new Set(Object.values(mapping.cards).filter(card => card.world === world).map(card => card.mainId));
    for (const mainId of mainIds) {
      const entry = docs.get(mainId);
      if (!entry) {
        fail(world, new Error(`映射里的主服卡 ${mainId} 不在库里`));
      } else if (!entry.complete) {
        fail(world, new Error(`主服卡 ${mainId} 条目不全，沿用上次发布的版本`));
      } else if (entry.mediaError) {
        continue;
      } else {
        run.history.push({ref: {kind: 'main', hash: entry.hash}, doc: entry.doc});
        run.toChar.push(entry.doc);
        mains.push({world, mainId, entry: mainEntry(entry, now)});
      }
    }
  }

  const newBases = await newBootstrapBases(ctx, run, mapping);
  let copies = Object.fromEntries(keepFoundryIds(Object.entries(copiesToPublish(ctx, charState.copyLists, mapping)), ([cardId]) => cardId, 'copies/', fail));
  const working = Object.fromEntries(keepFoundryIds(Object.entries(freshChar?.working ?? {}), ([cardId]) => cardId, 'working/', fail));
  const {docs: copyDocs, previous: previousCopies} = copyMedia(ctx, copies, run);

  // 主服先补到建卡服，再搬提交的新素材；旧路径仍只补缺，不会用草稿覆盖主服同名图。
  // dry 的两侧都是新 Data，原始来源侧的发布目录也需要自己的媒体文件。
  const mainReady = mirrorDocImages(run.toChar, out.main, [out.char, out.main], config.publishDir, fail);
  const charReady = mirrorDocImages(run.toMain, config.char.dataDir, [out.main, out.char], config.publishDir, fail);
  const mediaReady = new Map([...mainReady, ...charReady]);
  const pending = new Set(ctx.mediaPending);
  if (freshChar) pending.delete('char');
  for (const world of Object.keys(freshWorlds)) pending.delete(`main:${world}`);
  mains = mains.filter(({world, entry}) => {
    if (mediaReady.get(entry.doc)) return true;
    pending.add(`main:${world}`);
    return false;
  });
  const previousWorking = ctx.char?.working ?? ctx.lastMeta.working ?? {};
  for (const [id, entry] of Object.entries(working)) {
    if (mediaReady.get(entry.doc)) continue;
    delete working[id];
    if (previousWorking[id]) charState.working[id] = previousWorking[id];
    else delete charState.working[id];
    pending.add('char');
  }
  const readyCopies = new Set([...copyDocs].filter(([, doc]) => mediaReady.get(doc)).map(([key]) => key));
  const readyGroups = readyCopyGroups(copies, readyCopies);
  const withdrawnGroups = new Set(Object.values(copies).flat().filter(copy => copy.withdrawn).map(copy => copy.companionOf ?? copy.copyId));
  copies = Object.fromEntries(Object.entries(copies).map(([cardId, list]) => [cardId, list.flatMap(copy => {
    // 撤回逐张写入；整组保留已读到的状态，不因媒体尚未就绪回滚成旧的未撤回版本。
    if (withdrawnGroups.has(copy.companionOf ?? copy.copyId)) return [copy];
    if (readyGroups.has(refKey(copyRef(copy)))) return [copy];
    pending.add('char');
    fail('images', new Error(`副本 ${copy.copyId}（${copy.rev}）的图片或主从组尚未齐全，沿用上次发布的版本`));
    const previous = previousCopies.get(copy.copyId);
    return previous ? [previous] : [];
  })]));
  const selectedGroups = readyCopyGroups(copies, readyCopies);
  run.history = run.history.filter(({ref, doc}) => ref.kind === 'copy'
    ? selectedGroups.has(refKey(ref)) : mediaReady.get(doc));
  const readyHistory = new Set(run.history.map(({ref}) => refKey(ref)));
  for (const [id, base] of Object.entries(newBases)) {
    if (!base.ref || readyHistory.has(refKey(base.ref))) continue;
    const files = [out.main, out.char].map(data => path.join(data, config.publishDir, historyFile(base.ref)));
    if (!files.every(file => fs.existsSync(file))) delete newBases[id];
  }
  const bootstrap = {...ctx.bootstrap, ...newBases};
  const writtenCopies = new Set(run.history.filter(({ref}) => ref.kind === 'copy').map(({ref}) => refKey(ref)));
  const ruleReady = new Set();
  for (const list of Object.values(copies)) {
    for (const copy of list) {
      const ruleError = copyRuleError(copy, ruleSnapshot) ?? copySourceRuleError(copy, ruleSnapshot, charState);
      if (ruleError) {
        fail('rules', new Error(`副本 ${copy.copyId}（目标团 ${copy.targetWorld}）不能套用：${ruleError}`));
        continue;
      }
      const ref = copyRef(copy);
      if (selectedGroups.has(refKey(ref))) ruleReady.add(refKey(ref));
    }
  }
  const published = [...readyCopyGroups(copies, ruleReady)];
  const meta = {
    generatedAt: now,
    char: {schema: charState.settings.get(SCHEMA_KEY) ?? null},
    worlds: worldsMeta(ctx, worldStates),
    cards: mapping.cards,
    users: mapping.users,
    bootstrap: Object.fromEntries(Object.entries(bootstrap).filter(([, base]) => base.ref).map(([id, base]) => [id, base.ref])),
    published: published.sort(),
    working: charState.working,
    ruleSnapshot,
  };

  try {
    // 先保存首次启用时间，后续发布失败或进程重启也不能把旧提交变成豁免提交。
    writeJsonAtomic(path.join(out.state, 'rules-enforcement.json'), {enforcedSince: ruleSnapshot.enforcedSince});
    ctx.enforcedSince = ruleSnapshot.enforcedSince;
    if (canonicalJson(ruleSnapshot) !== canonicalJson(ctx.lastRules) && !invalidateRules(ctx, meta, fail)) {
      return report;
    }
    writeTextAtomic(rulesFile(ctx), renderRulesModule(ruleSnapshot));
    publishAll({
      dataDirs: [out.main, out.char], publishDir: config.publishDir, meta,
      mains: keepFoundryIds(mains, ({mainId}) => mainId, 'main/', fail), copies, working, history: run.history,
    });
    writeTextAtomic(path.join(out.state, 'rules.json'), JSON.stringify(ruleSnapshot, null, 2));
  } catch (error) {
    fail('publish', error);
    invalidateRules(ctx, meta, fail);
    return report;
  }
  for (const [key, fp] of run.marks) ctx.throttle.mark(key, fp, now);
  for (const world of run.unavailableRules) delete ctx.worlds[world];
  for (const [world, fresh] of Object.entries(freshWorlds)) ctx.worlds[world] = fresh.state;
  if (freshChar) ctx.char = freshChar.state;
  ctx.mediaPending = pending;
  ctx.lastMeta = meta;
  ctx.lastRules = ruleSnapshot;
  report.copiesPublished = writtenCopies.size;

  if (Object.keys(newBases).length) {
    try {
      writeTextAtomic(path.join(out.state, 'bootstrap.json'), JSON.stringify(bootstrap, null, 2));
      ctx.bootstrap = bootstrap;
      for (const id of Object.keys(newBases)) {
        ctx.bootstrapDocs.main.delete(id);
        ctx.bootstrapDocs.char.delete(id);
      }
    } catch (error) {
      fail('bootstrap', error);
    }
  }

  // GM 客户端上传撤回快照要求目录已经在
  try {
    for (const world of config.worlds) fs.mkdirSync(path.join(out.main, config.publishDir, 'undo', world), {recursive: true});
  } catch (error) {
    fail('undo', error);
  }
  try {
    mirrorCharHistory({fromData: out.main, toData: out.char, publishDir: config.publishDir});
  } catch (error) {
    fail('char-history', error);
  }
  if (now - ctx.lastGcAt >= DAY_MS) {
    try {
      collectGarbage(ctx, {charState, worldStates, copies, meta}, now, fail);
    } catch (error) {
      fail('gc', error);
    }
  }
  return report;
}

/** 每轮一行的日志。`read` 是这一轮读了的库数，建卡服算一个。 */
export function formatTickLine(report) {
  const read = report.worldsRead.length + (report.charRead ? 1 : 0);
  return `[charsync] tick main=${report.mainActive ?? 'none'} read=${read} copies=${report.copiesPublished} skipped=${report.copiesSkippedIncomplete.length} errors=${report.errors.length}`;
}

function logTick(report) {
  console.log(formatTickLine(report));
  for (const {scope, message} of report.errors) console.error(`[charsync] error ${scope}: ${message}`);
}

/** 每 `intervalSeconds` 跑一轮，单轮抛错只记日志。收到 SIGTERM / SIGINT 时等这一轮跑完（快照随之关掉）再退出。 */
async function runLoop(ctx) {
  let stopping = false;
  let wake = () => {};
  const stop = () => {
    stopping = true;
    wake();
  };
  process.once('SIGTERM', stop);
  process.once('SIGINT', stop);
  while (!stopping) {
    try {
      logTick(await tick(ctx));
    } catch (error) {
      console.error(`[charsync] tick failed: ${error.stack ?? error}`);
    }
    if (stopping) break;
    await new Promise(resolve => {
      const timer = setTimeout(resolve, ctx.config.intervalSeconds * 1000);
      wake = () => {
        clearTimeout(timer);
        resolve();
      };
    });
  }
  console.log('[charsync] stopped');
}

async function cli(argv) {
  const {values} = parseArgs({
    args: argv,
    options: {config: {type: 'string'}, once: {type: 'boolean', default: false}, dry: {type: 'boolean', default: false}},
  });
  if (!values.config) throw new Error('用法：node engine/main.mjs --config <path> [--once] [--dry]');
  const config = loadConfig(values.config);
  const ctx = createContext(config, loadSeed(config.seedPath, config.worlds), {dry: values.dry});
  // 上次崩溃留下的快照目录
  cleanStaleSnapshots(config.tmpRoot);
  if (!values.once) return runLoop(ctx);
  const report = await tick(ctx);
  logTick(report);
  // 调用方取最后一行解析
  console.log(JSON.stringify(report));
}

// `pm2 start main.mjs` 时 argv[1] 是 pm2 自己的容器脚本，本文件经 import() 加载，路径在 pm_exec_path 里；
// 参数仍在 argv[2] 以后
function isEntryPoint() {
  return [process.argv[1], process.env.pm_exec_path].some(file => {
    if (!file) return false;
    try {
      return pathToFileURL(fs.realpathSync(file)).href === import.meta.url;
    } catch {
      return false;
    }
  });
}

if (isEntryPoint()) {
  cli(process.argv.slice(2)).catch(error => {
    console.error(`[charsync] ${error.stack ?? error}`);
    process.exitCode = 1;
  });
}
