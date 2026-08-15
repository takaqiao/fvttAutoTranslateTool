import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
for (const pack of ['crucible-adventure','adventure']) {
  const db = new ClassicLevel(path.join(PKG,pack), { valueEncoding:'json' });
  await db.open();
  const dev=new Map(); const pagesWithNames=new Set(); const types=new Map(); const journals=new Set();
  let namedGroups=0;
  for await (const [k,doc] of db.iterator()) {
    if(!k.toString().startsWith('!adventures!')) continue;
    for(const j of doc.journal??[]) for(const p of j.pages??[]) {
      const toks=p?.system?.encounter?.tokens; if(!Array.isArray(toks)) continue;
      let n=0;
      for(const t of toks){ let g=0; for(const a of t.actors??[]) if(a?.tokenData?.name?.trim()) {n++;g++;} if(g) namedGroups++; }
      if(n){ pagesWithNames.add(`${j.name}/${p.name}`); journals.add(j.name);
        types.set(p.type,(types.get(p.type)||0)+1);
        const d=JSON.stringify(p.system?.development??null); dev.set(d,(dev.get(d)||0)+1); }
    }
  }
  await db.close();
  console.log(`\n=== ${pack}`);
  console.log(' pages carrying >=1 named encounter token:',pagesWithNames.size,' journals:',journals.size,' token-groups with names:',namedGroups);
  console.log(' page types:',JSON.stringify([...types]));
  console.log(' development status:',JSON.stringify([...dev]).slice(0,600));
}
