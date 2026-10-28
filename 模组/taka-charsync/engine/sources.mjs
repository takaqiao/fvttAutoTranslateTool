// 引擎的数据源：主服状态接口、库目录指纹与读取节流、按世界拷快照读库。
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { openSnapshot, readActorDoc, readFolders, readSettings, readUsers, scanActors } from './leveldb.mjs';

const STATUS_TIMEOUT_MS = 5000;
const STALE_SNAPSHOT_MS = 60 * 60 * 1000;
// 只认 openSnapshot 起的名字；测试目录、dry 输出这些 charsync-* 不动
const SNAPSHOT_DIR = /^charsync-\d+-\d+$/;
const WORLD_DBS = ['actors', 'settings', 'users'];

/** 主服正在开的世界 id，没开世界时为 null。连不上、超时、返回非 2xx 时抛错。 */
export async function mainActiveWorld(statusUrl) {
  let response;
  try {
    response = await fetch(statusUrl, {signal: AbortSignal.timeout(STATUS_TIMEOUT_MS)});
  } catch (error) {
    throw new Error(`状态接口 ${statusUrl} 连不上：${error.cause?.message ?? error.message}`, {cause: error});
  }
  if (!response.ok) throw new Error(`状态接口 ${statusUrl} 返回 ${response.status}`);
  const status = await response.json();
  if (!status.active) return null;
  if (typeof status.world !== 'string') throw new Error(`状态接口 ${statusUrl} 报告世界已开，但没有给出世界 id`);
  return status.world;
}

/**
 * 库目录里除 `LOCK` 外的普通文件按 `name:size:mtimeMs` 排序后的 SHA-1。目录不存在时抛错；
 * 列出后、取属性前被删掉的文件不算（库开着时 LevelDB 压缩会删旧文件）。
 */
export function fingerprint(dir) {
  const parts = fs.readdirSync(dir, {withFileTypes: true})
    .filter(entry => entry.isFile() && entry.name !== 'LOCK')
    .flatMap(entry => {
      const stat = fs.statSync(path.join(dir, entry.name), {throwIfNoEntry: false});
      return stat ? [`${entry.name}:${stat.size}:${stat.mtimeMs}`] : [];
    })
    .sort();
  return crypto.createHash('sha1').update(parts.join('\n')).digest('hex');
}

/** 建卡服还要检查 folders：只移动父文件夹也会改变角色所属团。 */
export function worldFingerprint(dataDir, world, {includeFolders = false} = {}) {
  const names = includeFolders ? [...WORLD_DBS, 'folders'] : WORLD_DBS;
  return names.map(name => fingerprint(path.join(dataDir, 'worlds', world, 'data', name))).join('|');
}

/** 同一个库的读取节流：指纹变了、且离上次读不少于 `minIntervalMs` 才再读；从没读过的总是要读。 */
export class Throttle {
  #minIntervalMs;
  #last = new Map();

  constructor(minIntervalMs) {
    this.#minIntervalMs = minIntervalMs;
  }

  due(key, fp, now) {
    const last = this.#last.get(key);
    return !last || (last.fp !== fp && now - last.at >= this.#minIntervalMs);
  }

  mark(key, fp, now) {
    this.#last.set(key, {fp, at: now});
  }
}

/**
 * 删掉 `tmpRoot` 下引擎进程崩溃留下的快照目录：名字形如 `charsync-<pid>-<n>`、mtime 早于一小时前。
 * 返回删掉的目录名。不按 pid 判断，同时跑着的 `--once` 进程的快照目录还新，不会被删。
 */
export function cleanStaleSnapshots(tmpRoot, now = Date.now()) {
  const removed = [];
  for (const entry of fs.readdirSync(tmpRoot, {withFileTypes: true})) {
    if (!entry.isDirectory() || !SNAPSHOT_DIR.test(entry.name)) continue;
    const dir = path.join(tmpRoot, entry.name);
    if (fs.statSync(dir).mtimeMs >= now - STALE_SNAPSHOT_MS) continue;
    fs.rmSync(dir, {recursive: true, force: true});
    removed.push(entry.name);
  }
  return removed;
}

async function withSnapshot(dbDir, options, read) {
  const snap = await openSnapshot(dbDir, options);
  try {
    return await read(snap);
  } finally {
    await snap.close();
  }
}

/**
 * 拷快照读一个世界：全部 actor 行、世界设置、用户，以及 `pick(actors, users)` 选中的 actor 的完整文档
 * （`readActorDoc` 的结果，按 actor id；库里没有的 id 略过）。返回前快照全部关掉、删掉。
 * `options` 是 `openSnapshot` 的 `{tmpRoot, foundryAppDir}`，建卡服另传 `includeFolders: true` 读取文件夹层级。
 */
export async function readWorld(dataDir, world, options, pick) {
  const dbDir = name => path.join(dataDir, 'worlds', world, 'data', name);
  const settings = await withSnapshot(dbDir('settings'), options, readSettings);
  const users = await withSnapshot(dbDir('users'), options, readUsers);
  const folders = options.includeFolders ? await withSnapshot(dbDir('folders'), options, readFolders) : new Map();
  return withSnapshot(dbDir('actors'), options, async snap => {
    const actors = await scanActors(snap);
    const docs = new Map();
    for (const id of new Set(pick(actors, users))) {
      const row = actors.get(id);
      if (row) docs.set(id, await readActorDoc(snap, row));
    }
    return {actors, settings, users, folders, docs};
  });
}
