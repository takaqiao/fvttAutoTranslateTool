// 引擎只读 Foundry 的 LevelDB：把库目录拷到临时目录再开副本，从不打开生产库。
// Foundry 运行时持有库里的 LOCK，所以拷贝时跳过它，副本里由 LevelDB 自己新建。
import { copyFile, mkdir, readdir, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';

/** 加载 Foundry 自带的 `classic-level`，引擎不另装依赖。`foundryAppDir` 是含 `node_modules/classic-level` 的 Foundry 程序目录。 */
export function loadClassicLevel(foundryAppDir) {
  const require = createRequire(path.join(foundryAppDir, 'package.json'));
  return require('classic-level').ClassicLevel;
}

let snapshotCount = 0;

/**
 * 把 `dbDir` 里除 `LOCK` 外的普通文件拷到 `${tmpRoot}/charsync-${pid}-${n}` 并以 utf8 键值打开。
 * 源库开着时逐个文件拷，拷到的副本可能缺刚写入的内嵌行，由 `readActorDoc` 的 `complete` 判断。
 * 用完必须调用 `close()`：关库并删掉副本目录。
 */
export async function openSnapshot(dbDir, {tmpRoot, foundryAppDir}) {
  const ClassicLevel = loadClassicLevel(foundryAppDir);
  const dir = path.join(tmpRoot, `charsync-${process.pid}-${++snapshotCount}`);
  // 进程异常退出会留下目录，pid 复用后可能撞名；旧文件混进副本会读到错的数据
  await rm(dir, {recursive: true, force: true});
  await mkdir(dir);
  try {
    for (const entry of await readdir(dbDir, {withFileTypes: true})) {
      if (entry.isFile() && entry.name !== 'LOCK') await copyFile(path.join(dbDir, entry.name), path.join(dir, entry.name));
    }
    const db = new ClassicLevel(dir, {keyEncoding: 'utf8', valueEncoding: 'utf8', createIfMissing: false});
    await db.open();
    return {
      db,
      dir,
      async close() {
        try {
          await db.close();
        } finally {
          await rm(dir, {recursive: true, force: true});
        }
      },
    };
  } catch (error) {
    await rm(dir, {recursive: true, force: true});
    throw error;
  }
}

/** 读出所有以 `prefix` 开头的行，返回「键去掉前缀 → 解析后的值」。 */
async function readRows(db, prefix) {
  // 前缀都以 `!` 或 `.` 结尾，末字符加一作上界，区间正好是带这个前缀的全部键
  const upper = prefix.slice(0, -1) + String.fromCharCode(prefix.charCodeAt(prefix.length - 1) + 1);
  const rows = new Map();
  for await (const [key, value] of db.iterator({gte: prefix, lt: upper})) {
    rows.set(key.slice(prefix.length), JSON.parse(value));
  }
  return rows;
}

/** actor 行，按 actor id；`items`、`effects` 还是 id 列表。 */
export function scanActors(snap) {
  return readRows(snap.db, '!actors!');
}

/** 文件夹的父级关系，供建卡服解析角色所属团。 */
export async function readFolders(snap) {
  return new Map([...(await readRows(snap.db, '!folders!'))].map(([id, row]) => [
    id, {_id: id, folder: row.folder ?? null},
  ]));
}

/**
 * 按 actor id 区间读它的条目、actor effect 和条目 effect，组装成 Foundry 导出的形状：
 * `items`、`effects` 和每个条目的 `effects` 都换成文档数组，顺序照行里的 id 列表。
 * 行里列出的 id 有任何一个找不到对应的行，`complete` 为 false，文档里略过缺的那几条。
 * 返回新文档，不改 `actorRow`。
 */
export async function readActorDoc(snap, actorRow) {
  const actorId = actorRow._id;
  const [items, effects, itemEffects] = await Promise.all([
    readRows(snap.db, `!actors.items!${actorId}.`),
    readRows(snap.db, `!actors.effects!${actorId}.`),
    readRows(snap.db, `!actors.items.effects!${actorId}.`),
  ]);

  let complete = true;
  const resolve = (rows, ids, keyOf = id => id) => ids.flatMap(id => {
    const doc = rows.get(keyOf(id));
    if (doc) return [doc];
    complete = false;
    return [];
  });

  const doc = structuredClone(actorRow);
  doc.items = resolve(items, actorRow.items ?? []);
  for (const item of doc.items) {
    if (Array.isArray(item.effects)) item.effects = resolve(itemEffects, item.effects, effectId => `${item._id}.${effectId}`);
  }
  doc.effects = resolve(effects, actorRow.effects ?? []);
  return {doc, complete};
}

function parseSettingValue(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

/**
 * 世界设置：键为设置的 `key`，值为 `JSON.parse(value)`，解析不了的保留原字符串。
 * 按用户存的设置（`user` 有值）和世界设置同在一个库里、可以同名，这里不收。
 */
export async function readSettings(snap) {
  const settings = new Map();
  for (const row of (await readRows(snap.db, '!settings!')).values()) {
    if (!row.user) settings.set(row.key, parseSettingValue(row.value));
  }
  return settings;
}

/** 用户表，按用户 id；`character` 没指定时为 null。 */
export async function readUsers(snap) {
  const users = new Map();
  for (const [userId, row] of await readRows(snap.db, '!users!')) {
    users.set(userId, {name: row.name, role: row.role, character: row.character ?? null});
  }
  return users;
}
