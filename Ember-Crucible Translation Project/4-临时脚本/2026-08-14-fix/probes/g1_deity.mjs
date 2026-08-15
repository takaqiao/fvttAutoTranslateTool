import {ClassicLevel} from "file:///C:/Program Files/Foundry Virtual Tabletop/resources/app/node_modules/classic-level/index.js";
const base = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs";
const out = {clerics:{}, warlocks:{}, sorcerers:{}, leaves:0, byPack:{}};
for (const pack of ["adventure","crucible-adventure"]) {
  const db = new ClassicLevel(`${base}/${pack}`, {valueEncoding:"json"});
  await db.open();
  let n=0;
  for await (const [key, val] of db.iterator()) {
    const walk = (doc) => {
      for (const p of doc.pages ?? []) {
        const s = p.system ?? {};
        for (const f of ["clerics","warlocks","sorcerers"]) {
          if (Array.isArray(s[f])) for (const v of s[f]) { out[f][v]=(out[f][v]??0)+1; n++; }
        }
      }
    };
    if (val?.pages) walk(val);
    for (const j of val?.journal ?? []) walk(j);
  }
  out.byPack[pack]=n;
  await db.close();
}
console.log(JSON.stringify(out,null,1));
