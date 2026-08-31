import { ClassicLevel } from "classic-level";
import { readFileSync, writeFileSync } from "node:fs";
const SRC = ["C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/crucible-compiled.mjs",
             "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs"]
            .map(f => readFileSync(f, "utf8")).join("\n");
const P="C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs";
const E="C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs";
const strip=h=>String(h??"").replace(/<[^>]*>/g," ").replace(/\s+/g," ").trim();
const rows=[];
for ( const [root,pk] of [[P,"talent"],[E,"crucible-character"]] ) {   // 天赋树本体
  let db; try{db=new ClassicLevel(`${root}/${pk}`,{keyEncoding:"utf8",valueEncoding:"json"});await db.open();}catch{continue;}
  for await ( const [key,doc] of db.iterator() ) {
    if ( doc?.type !== "talent" ) continue;
    const sys = doc.system ?? {};
    const list = Array.isArray(sys.actions) ? sys.actions : Object.values(sys.actions ?? {});
    if ( list.length ) continue;                                   // 有动作 → 不算
    if ( (doc.effects ?? []).length ) continue;                    // 有效果 → 不算
    if ( sys.training?.type ) continue;                            // 给训练 → 数据驱动，合法
    if ( sys.rune || sys.gesture || sys.inflection ) continue;     // 给施法要素 → 合法
    if ( sys.node ) { /* 天赋树节点仍可能是纯被动，继续往下判 */ }
    const id = doc._id;
    // id 在源码里出现过 = 有硬编码逻辑认它
    const idHit = id && SRC.includes(id);
    if ( idHit ) continue;
    rows.push({ pk, name: doc.name, id, node: sys.node ?? null,
                desc: strip(sys.description).slice(0, 150) });
  }
  await db.close();
}
writeFileSync("dead.json", JSON.stringify(rows, null, 1), "utf8");
console.log(`天赋树里「0 动作 / 0 效果 / 不给训练或施法要素 / id 在源码零命中」的：${rows.length} 条`);
for ( const r of rows.slice(0, 40) ) console.log(`  [${r.pk}] ${r.name}  —— ${r.desc.slice(0,100)}`);
if ( rows.length > 40 ) console.log(`  …还有 ${rows.length-40} 条，全量在 dead.json`);
