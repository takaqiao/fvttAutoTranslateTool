import { ClassicLevel } from "classic-level";
import { readdirSync, existsSync } from "node:fs";
const ROOTS = [["crucible","C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs"],
               ["ember","C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs"]];
const want = new Set(["noxiousSpray","selfDestruct","devourThoughts"]);
for ( const [src, root] of ROOTS ) {
  if ( !existsSync(root) ) continue;
  for ( const p of readdirSync(root) ) {
    let db; try { db=new ClassicLevel(`${root}/${p}`,{keyEncoding:"utf8",valueEncoding:"json"}); await db.open(); } catch { continue; }
    for await (const [,doc] of db.iterator()) {
      const scan = (it, where) => {
        const acts = it?.system?.actions;
        for ( const a of (Array.isArray(acts)?acts:Object.values(acts??{})) )
          if ( want.has(a?.id) ) console.log(`${src}/${p} | ${where} | ${a.id} | tags=${JSON.stringify(a.tags)}`);
      };
      scan(doc, doc.name);
      for ( const it of (doc?.items ?? []) ) scan(it, `${doc.name}>${it.name}`);
      for ( const ac of (doc?.actors ?? []) ) { scan(ac, ac.name); for ( const it of (ac?.items ?? []) ) scan(it, `${ac.name}>${it.name}`); }
    }
    await db.close();
  }
}
