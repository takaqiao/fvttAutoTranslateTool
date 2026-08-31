#!/usr/bin/env node
/**
 * dump_embedded_items.mjs —— 只读转储：actor 内嵌 Item 的
 *   { pack, actor, itemName, itemType, hasCompendiumSource, sourcePack }
 *
 * 用途：判定「按 name 跨包回落」的暴露面。
 * babele 的 document 转换器对 actor.items 的回落顺序是
 *   exact-source（_stats.compendiumSource 指向的包，且该包已翻译）
 *   → owner-package → generic（同 documentType 的**任意**已翻译包，按 name 命中）
 * 只有 exact-source 失败的内嵌物品才会落到按名字匹配那一档。
 *
 * 只读打开 LevelDB，不写任何东西。若 Foundry 正在运行会拿不到锁，直接报错退出。
 *
 * 用法：node dump_embedded_items.mjs > out.json
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');

const PACKAGES = [
  { id: 'crucible', dir: 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible', manifest: 'system.json' },
  { id: 'ember', dir: 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember', manifest: 'module.json' },
];

const out = [];

for (const pkg of PACKAGES) {
  const man = JSON.parse(fs.readFileSync(path.join(pkg.dir, pkg.manifest), 'utf8'));
  for (const p of man.packs ?? []) {
    const dbPath = path.join(pkg.dir, p.path.replace(/^\//, ''));
    if (!fs.existsSync(dbPath)) continue;
    const db = new ClassicLevel(dbPath, { valueEncoding: 'json', createIfMissing: false });
    try {
      await db.open();
    } catch (e) {
      console.error(`skip ${p.name}: ${e.message}`);
      continue;
    }
    for await (const [key, value] of db.iterator()) {
      const collect = (docs, ownerName, ownerKind) => {
        for (const it of docs ?? []) {
          out.push({
            pack: `${pkg.id}.${p.name}`,
            packType: p.type,
            owner: ownerName,
            ownerKind,
            name: it.name,
            type: it.type,
            src: it?._stats?.compendiumSource ?? it?.flags?.core?.sourceId ?? null,
          });
        }
      };
      if (p.type === 'Actor' && key.startsWith('!actors!')) collect(value.items, value.name, 'actor');
      if (p.type === 'Adventure' && key.startsWith('!adventures!')) {
        for (const a of value.actors ?? []) collect(a.items, a.name, 'adventure-actor');
      }
    }
    await db.close();
  }
}

process.stdout.write(JSON.stringify(out, null, 1));
console.error(`embedded items: ${out.length}`);
