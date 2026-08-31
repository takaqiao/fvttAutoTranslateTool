import { ClassicLevel } from "classic-level";
import { readdirSync, existsSync, writeFileSync } from "node:fs";

const ROOTS = [
  ["crucible", "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs"],
  ["ember",    "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs"]
];
// 「没自动化」的标记词。只认明确说自动化缺失的，不认 Under Development 这种泛泛的开发中横幅
const MARK = /(not yet automated|not automated|isn['’]t automated|no automation|not implemented)/i;
const strip = h => String(h ?? "").replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();

const rows = [];
let talents = 0, actions = 0;

for ( const [src, root] of ROOTS ) {
  if ( !existsSync(root) ) continue;
  for ( const pk of readdirSync(root) ) {
    let db;
    try { db = new ClassicLevel(`${root}/${pk}`, {keyEncoding:"utf8", valueEncoding:"json"}); await db.open(); }
    catch { continue; }
    for await ( const [, doc] of db.iterator() ) {
      const visit = (it, owner) => {
        if ( it?.type !== "talent" ) return;
        talents++;
        const acts = it.system?.actions;
        const actList = Array.isArray(acts) ? acts : Object.values(acts ?? {});
        actions += actList.length;
        // 天赋自身描述
        const tDesc = strip(it.system?.description);
        const hits = [];
        if ( MARK.test(tDesc) ) hits.push({ where: "天赋描述", text: tDesc.match(new RegExp(`.{0,110}(${MARK.source}).{0,110}`,"i"))?.[0] ?? "" });
        // 动作描述
        for ( const a of actList ) {
          const aDesc = strip(a?.description);
          if ( MARK.test(aDesc) )
            hits.push({ where: `动作 ${a.id}`, text: aDesc.match(new RegExp(`.{0,110}(${MARK.source}).{0,110}`,"i"))?.[0] ?? "" });
        }
        if ( !hits.length ) return;
        rows.push({
          src, pack: pk, owner, name: it.name, id: it._id,
          actionIds: actList.map(a => a.id),
          actionCount: actList.length,
          hasEffects: actList.some(a => (a.effects ?? []).length),
          tags: [...new Set(actList.flatMap(a => a.tags ?? []))],
          hits
        });
      };
      visit(doc, "(包级条目)");
      for ( const it of (doc?.items ?? []) ) visit(it, doc.name);
      for ( const ac of (doc?.actors ?? []) ) for ( const it of (ac?.items ?? []) ) visit(it, `${doc.name} > ${ac.name}`);
    }
    await db.close();
  }
}
// 按「天赋名」去重（同一天赋在多个 actor 上会重复）
const uniq = new Map();
for ( const r of rows ) {
  const k = `${r.src}|${r.name}`;
  if ( !uniq.has(k) ) uniq.set(k, { ...r, copies: 1 });
  else uniq.get(k).copies++;
}
writeFileSync("scan_auto.json", JSON.stringify([...uniq.values()], null, 1), "utf8");
console.log(`扫过 ${talents} 个 talent 条目（含重复副本）/ ${actions} 个动作`);
console.log(`明确写着「未自动化」的天赋：${uniq.size} 个（去重后），命中实例 ${rows.length} 个`);
console.log(`  crucible 侧 ${[...uniq.values()].filter(r=>r.src==="crucible").length} 个`);
console.log(`  ember 侧 ${[...uniq.values()].filter(r=>r.src==="ember").length} 个`);
