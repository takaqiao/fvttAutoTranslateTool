import os from 'node:os';
import path from 'node:path';
import { loadClassicLevel } from '../../engine/leveldb.mjs';

export const foundryAppDir = process.env.CHARSYNC_FOUNDRY_APP ?? 'C:/Program Files/Foundry Virtual Tabletop/resources/app';
export const tmpRoot = os.tmpdir();

/** 在 `dir` 建一个测试库：键原样写入，值 `JSON.stringify` 后写入（与 Foundry 的存法一致）。写完关库，目录里会留下 `LOCK`。 */
export async function makeDb(dir, rows) {
  const ClassicLevel = loadClassicLevel(foundryAppDir);
  const db = new ClassicLevel(dir, {keyEncoding: 'utf8', valueEncoding: 'utf8'});
  await db.open();
  await db.batch(rows.map(([key, value]) => ({type: 'put', key, value: JSON.stringify(value)})));
  await db.close();
}

/** 一张卡在 actors 库里的行：actor 行（`items`、`effects` 换成 id 列表）、条目行、条目 effect 行、actor effect 行。 */
export function actorRows(doc) {
  const {items = [], effects = [], ...actor} = doc;
  const rows = [[`!actors!${doc._id}`, {...actor, items: items.map(item => item._id), effects: effects.map(effect => effect._id)}]];
  for (const {effects: itemEffects = [], ...item} of items) {
    rows.push([`!actors.items!${doc._id}.${item._id}`, {...item, effects: itemEffects.map(effect => effect._id)}]);
    for (const effect of itemEffects) rows.push([`!actors.items.effects!${doc._id}.${item._id}.${effect._id}`, effect]);
  }
  for (const effect of effects) rows.push([`!actors.effects!${doc._id}.${effect._id}`, effect]);
  return rows;
}

/**
 * 在 `<dataDir>/worlds/<world>/data` 下建 actors、settings、users 三个库。`actors` 是导出形状的文档，
 * `settings` 是「键 → 值」的世界设置，`users` 是用户行。
 */
export async function makeWorld(dataDir, world, {actors = [], settings = {}, users = [], folders = []}) {
  const dbDir = name => path.join(dataDir, 'worlds', world, 'data', name);
  await makeDb(dbDir('actors'), actors.flatMap(actorRows));
  await makeDb(dbDir('settings'), Object.entries(settings).map(([key, value], index) => {
    const id = `setting${String(index).padStart(9, '0')}`;
    return [`!settings!${id}`, {_id: id, key, value: JSON.stringify(value), user: null}];
  }));
  await makeDb(dbDir('users'), users.map(user => [`!users!${user._id}`, user]));
  await makeDb(dbDir('folders'), folders.map(folder => [`!folders!${folder._id}`, folder]));
}
