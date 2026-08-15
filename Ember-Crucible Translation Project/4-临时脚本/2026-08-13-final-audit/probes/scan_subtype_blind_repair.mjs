#!/usr/bin/env node
/**
 * probe: scan_subtype_blind_repair
 *
 * 判据（把种子实例抽象出来的那一类）：
 *   一个「修复/防御」变换 T，只对**某个子类型**的输入是正确的，
 *   却被挂在一个**会同时收到全部子类型**的调用点上，
 *   而闸只测「形状/类型」不测「子类型」——于是正常数据被静默改写或丢弃。
 *
 * 本脚本只做**取证**：从 Foundry 的 LevelDB 包里读出真实数据，回答
 *   1. crucible 13 个 Item 子类型里，哪些 system.description 是**字符串**（HTMLField）、
 *      哪些是 {public, private}（SchemaField）——以及各有多少条真实数据。
 *   2. 演员内嵌 item 里这些子类型各占多少（重导入 / 世界迁移的打击面）。
 *   3. causticPhial 这个 action 的 effects 数组在真实数据里是不是空的
 *      （register.js 的 sanitizeActionEffects 会往空数组里塞一个 {}）。
 *
 * 只读。不写任何库文件。
 *
 * 已知假阳性模式：
 *   - 包里可能存在**旧版**残留文档，其 description 形状与当前 schema 不符；
 *     本脚本按 type 分组统计，形状少数派会单独列出来，不会被当成多数派结论。
 *   - LevelDB 里 `!items!` 与 `!actors.items!` 两种键都要数，只数前者会低估。
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const require_ = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require_('classic-level');

const DATA = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data';
const PKGS = [
  ['system', path.join(DATA, 'systems/crucible')],
  ['module', path.join(DATA, 'modules/ember')],
];

const STRING_DESC_TYPES = new Set(['ancestry', 'archetype', 'background', 'spell', 'talent', 'taxonomy']);

const byType = {};          // type -> {string:n, object:n, missing:n, blank:n}
const embeddedByType = {};  // type -> n   (items living inside an actor / adventure actor)
const causticEffects = [];  // {where, len}
const actionEmptyEffects = { total: 0, empty: 0 };

function bump(m, k, f) { (m[k] ??= { string: 0, object: 0, other: 0 })[f] += 1; }

function noteItem(item, embedded, where) {
  if (!item || typeof item !== 'object') return;
  const t = item.type ?? '(none)';
  const d = item.system?.description;
  bump(byType, t, typeof d === 'string' ? 'string' : (d && typeof d === 'object' ? 'object' : 'other'));
  if (embedded) embeddedByType[t] = (embeddedByType[t] ?? 0) + 1;
  for (const a of item.system?.actions ?? []) {
    if (!a || typeof a !== 'object') continue;
    actionEmptyEffects.total += 1;
    if (!Array.isArray(a.effects) || a.effects.length === 0) actionEmptyEffects.empty += 1;
    if (a.id === 'causticPhial') {
      causticEffects.push({ where: `${where}/${item.name}`, len: Array.isArray(a.effects) ? a.effects.length : null });
    }
  }
}

function walkActor(actor, where) {
  for (const it of actor?.items ?? []) noteItem(it, true, where);
}

for (const [kind, dir] of PKGS) {
  const manifest = path.join(dir, kind === 'system' ? 'system.json' : 'module.json');
  if (!fs.existsSync(manifest)) { console.error('missing manifest', manifest); continue; }
  const m = JSON.parse(fs.readFileSync(manifest, 'utf8'));
  for (const pack of m.packs ?? []) {
    const pdir = path.join(dir, 'packs', path.basename(pack.path ?? pack.name));
    if (!fs.existsSync(pdir)) continue;
    let db;
    try { db = new ClassicLevel(pdir, { createIfMissing: false }); await db.open(); }
    catch (e) { console.error('open fail', pdir, e.message); continue; }
    for await (const [k, v] of db.iterator()) {
      const key = k.toString();
      let doc; try { doc = JSON.parse(v.toString()); } catch { continue; }
      if (key.startsWith('!items!')) noteItem(doc, false, pack.name);
      else if (key.includes('.items!') || key.startsWith('!actors.items!')) noteItem(doc, true, pack.name);
      else if (key.startsWith('!actors!')) walkActor(doc, pack.name);
      else if (key.startsWith('!adventures!')) {
        for (const a of doc.actors ?? []) walkActor(a, `${pack.name}/adventure`);
        for (const it of doc.items ?? []) noteItem(it, false, `${pack.name}/adventure`);
      }
    }
    await db.close();
  }
}

const out = {
  legend: 'system.description shape per Item subtype, from real pack data',
  stringDescTypes: [...STRING_DESC_TYPES],
  byType,
  embeddedByType,
  actionEmptyEffects,
  causticEffects,
};
console.log(JSON.stringify(out, null, 2));
