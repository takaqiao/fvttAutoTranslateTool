#!/usr/bin/env node
/**
 * 灵敏度回测：往 LevelDB pack 的**临时副本**里注入一个「mapping 从没声明过的
 * 人类可读字段」，看判据能不能报出来；同时注入一个枚举 id 作阴性对照，
 * 看判据会不会误报。
 *
 * 绝不碰真 packs：全程只在 scratchpad 里的副本上做。
 *
 * 用法: node inject_and_verify.mjs --work <scratchpad dir>
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const WORK = arg('--work');
if (!WORK) { console.error('need --work <dir>'); process.exit(1); }

const SRC_PACK = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs/ancestry';
const PKG = path.join(WORK, 'inject_pkg');
const DST_PACK = path.join(PKG, 'packs', 'ancestry');

// 阳性：一段真的像正文的英文；阴性：一个 camelCase 枚举 id
const POSITIVE = "<p>The lantern-keeper's ledger names every soul who crossed the Reaver Ocean.</p>";
const NEGATIVE = 'coldIronPlated';

fs.rmSync(PKG, { recursive: true, force: true });
fs.mkdirSync(path.join(PKG, 'packs'), { recursive: true });
fs.cpSync(SRC_PACK, DST_PACK, { recursive: true });
fs.writeFileSync(path.join(PKG, 'system.json'), JSON.stringify({
  id: 'crucible', version: 'inject-test',
  packs: [{ name: 'ancestry', type: 'Item', path: 'packs/ancestry' }],
}, null, 1), 'utf8');

const db = new ClassicLevel(DST_PACK, { createIfMissing: false });
let injected = 0;
for await (const [k, v] of db.iterator()) {
  const key = k.toString();
  if (!key.startsWith('!items!')) continue;
  const doc = JSON.parse(v.toString());
  doc.system ??= {};
  doc.system.loreExcerpt = POSITIVE;          // mapping 里不存在的可读字段
  doc.system.materialTag = NEGATIVE;          // mapping 里不存在的枚举 id
  await db.put(key, JSON.stringify(doc));
  injected += 1;
  if (injected >= 3) break;
}
await db.close();
console.log(`injected into ${injected} docs -> ${PKG}`);
console.log(`positive field: system.loreExcerpt = ${JSON.stringify(POSITIVE)}`);
console.log(`negative field: system.materialTag = ${JSON.stringify(NEGATIVE)}`);
