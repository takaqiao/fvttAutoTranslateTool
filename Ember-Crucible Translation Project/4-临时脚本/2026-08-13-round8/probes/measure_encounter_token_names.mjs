#!/usr/bin/env node
/**
 * 量化 `JournalEntryPage.ember.questEvent/standaloneEvent`
 * `system.encounter.tokens[].actors[].tokenData.name` 这条盲区。
 *
 * ember.mjs `_spawnEncounterTokens()` -> `_createActorToken(actor, tokenData)`
 * -> `actor.getTokenDocument(tokenData)`，并且
 *   `if ( !token.actorLink && (token.name !== actor.prototypeToken.name) )
 *      token.delta.updateSource({name: token.name});`
 * 所以这个字符串会**原样落到场上 Token 的名牌与战斗轮上**。它写在页面的
 * system 数据里，不在任何 mapping 的定义域内。
 *
 * 输出：唯一名清单 + 该名字在中文包里是否已有对应 actor 译名
 * （有译名 = 玩家会看到「actor 是中文、token 名牌是英文」的分裂）。
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PACKAGE_DIR = arg('--package', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember');
const CN_DIR = arg('--cn', 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/cn');
const OUT = arg('--out');

async function readPack(dir) {
  const db = new ClassicLevel(dir, { createIfMissing: false });
  const b = {};
  for await (const [k, v] of db.iterator()) {
    const m = k.toString().match(/^!([^!]+)!(.+)$/);
    if (!m) continue;
    let doc; try { doc = JSON.parse(v.toString()); } catch { continue; }
    (b[m[1]] ||= []).push({ idPart: m[2], doc });
  }
  await db.close();
  return b;
}

/** 中文包里 actor 的 name/tokenName 译文（key = 英文名）。 */
function cnActorNames() {
  const map = new Map();
  for (const fn of fs.readdirSync(CN_DIR)) {
    if (!fn.endsWith('.json')) continue;
    const d = JSON.parse(fs.readFileSync(path.join(CN_DIR, fn), 'utf8'));
    for (const entry of Object.values(d.entries ?? {})) {
      const actors = entry?.actors ?? (entry?.name && entry?.tokenName ? { [entry.name]: entry } : {});
      for (const [k, v] of Object.entries(actors)) {
        if (typeof v?.name === 'string') map.set(k, { name: v.name, tokenName: v.tokenName ?? null, pack: fn });
      }
    }
  }
  return map;
}

const hits = [];
const manifest = JSON.parse(fs.readFileSync(path.join(PACKAGE_DIR, 'module.json'), 'utf8'));
for (const pack of manifest.packs ?? []) {
  if (pack.type !== 'Adventure') continue;
  const dir = path.join(PACKAGE_DIR, 'packs', path.basename(pack.path ?? pack.name));
  if (!fs.existsSync(dir)) continue;
  const b = await readPack(dir);
  for (const { doc: adv } of (b.adventures || [])) {
    for (const j of (adv.journal || [])) {
      for (const p of (j.pages || [])) {
        const groups = p?.system?.encounter?.tokens;
        if (!Array.isArray(groups)) continue;
        for (const g of groups) {
          for (const a of (g.actors || [])) {
            const nm = a?.tokenData?.name;
            if (typeof nm !== 'string' || !nm.trim()) continue;
            hits.push({ pack: pack.name, journal: j.name, page: p.name, pageType: p.type, actorUuid: a.actor, tokenName: nm });
          }
        }
      }
    }
  }
}

const cn = cnActorNames();
const byName = new Map();
for (const h of hits) {
  const e = byName.get(h.tokenName) ?? { n: 0, packs: new Set(), pages: new Set() };
  e.n += 1; e.packs.add(h.pack); e.pages.add(`${h.journal} / ${h.page}`);
  byName.set(h.tokenName, e);
}
const rows = [...byName.entries()].sort((a, b) => b[1].n - a[1].n).map(([nm, e]) => ({
  tokenName: nm, n: e.n, packs: [...e.packs],
  cnActorTranslation: cn.get(nm)?.name ?? null,
  samplePages: [...e.pages].slice(0, 3),
}));
const translatedCounterpart = rows.filter((r) => r.cnActorTranslation).length;
const res = { totalOccurrences: hits.length, uniqueNames: rows.length,
  uniqueWithTranslatedActorCounterpart: translatedCounterpart, rows };
if (OUT) fs.writeFileSync(OUT, `${JSON.stringify(res, null, 1)}\n`, 'utf8');
console.log(`occurrences=${hits.length} unique=${rows.length} with-translated-actor=${translatedCounterpart}`);
for (const r of rows.slice(0, 25)) {
  console.log(`  ${String(r.n).padStart(3)}x  "${r.tokenName}"  cnActor=${r.cnActorTranslation ?? '—'}  [${r.packs}]  eg ${r.samplePages[0]}`);
}
