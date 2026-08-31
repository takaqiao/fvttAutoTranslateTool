import { ClassicLevel } from "classic-level";
import { readdirSync, existsSync } from "node:fs";
const ROOTS = [
  ["crucible", "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs"],
  ["ember",    "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs"]
];
const bad = d => !!d && ((d.units === "turns") || (d.turns != null && d.turns !== 0));
const byId = new Map(); let looseEffects = 0; const loose = [];
for ( const [src, root] of ROOTS ) {
  if ( !existsSync(root) ) continue;
  for ( const p of readdirSync(root) ) {
    let db;
    try { db = new ClassicLevel(`${root}/${p}`, {keyEncoding:"utf8", valueEncoding:"json"}); await db.open(); } catch { continue; }
    for await (const [, doc] of db.iterator()) {
      const scanItem = it => {
        const acts = it?.system?.actions;
        for ( const a of (Array.isArray(acts) ? acts : Object.values(acts ?? {})) ) {
          for ( const e of (a?.effects ?? []) ) if ( bad(e?.duration) ) {
            if ( !byId.has(a.id) ) byId.set(a.id, new Set());
            byId.get(a.id).add(src);
          }
        }
        for ( const e of (it?.effects ?? []) ) if ( bad(e?.duration) ) { looseEffects++; loose.push(`${src}/${p}|${it.name}|${e.name}`); }
      };
      scanItem(doc);
      for ( const it of (doc?.items ?? []) ) scanItem(it);
      for ( const act of (doc?.actors ?? []) ) { scanItem(act); for ( const it of (act?.items ?? []) ) scanItem(it); }
    }
    await db.close();
  }
}
console.log(`受影响的不同动作 id：${byId.size}`);
console.log(`  crucible 侧：${[...byId].filter(([,s])=>s.has("crucible")).length}`);
console.log(`  ember 侧：${[...byId].filter(([,s])=>s.has("ember")).length}`);
console.log(`物品自带效果（N12 那半边）：${looseEffects}`);
console.log('\nember 侧仍中招的 id：');
console.log('  ' + [...byId].filter(([,s])=>s.has("ember")).map(([k])=>k).sort().join(', ') || '  （无）');
