import { createRequire } from 'module';
const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');
const base='C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs/';
for (const pack of ['adventure','crucible-adventure']) {
  const db = new ClassicLevel(base+pack, {createIfMissing:false});
  const stats={pageTypes:{}, encPages:0, tokens:0, actors:0, names:0, uniq:new Set(), deltaEffects:0, shapes:new Set()};
  let sample=null;
  for await (const [k,v] of db.iterator()) {
    const key=k.toString();
    if(!key.startsWith('!adventures!')) continue;
    const doc=JSON.parse(v.toString());
    for(const j of doc.journal??[]) for(const p of j.pages??[]) {
      const enc=p.system?.encounter;
      if(!enc) continue;
      stats.pageTypes[p.type]=(stats.pageTypes[p.type]||0)+1;
      if(!Array.isArray(enc.tokens)||!enc.tokens.length) continue;
      stats.encPages++;
      for(const t of enc.tokens){
        stats.tokens++;
        stats.shapes.add(Object.keys(t).sort().join(','));
        for(const a of t.actors??[]){
          stats.actors++;
          const td=a.tokenData;
          if(td && typeof td.name==='string' && td.name.trim()){ stats.names++; stats.uniq.add(td.name); if(!sample) sample={page:p.name,type:p.type,token:JSON.parse(JSON.stringify(t))}; }
          for(const e of td?.delta?.effects??[]) if(typeof e?.name==='string') stats.deltaEffects++;
        }
      }
    }
  }
  await db.close();
  console.log(pack, JSON.stringify({pageTypes:stats.pageTypes, encPages:stats.encPages, tokens:stats.tokens, actorEntries:stats.actors, names:stats.names, uniq:stats.uniq.size, deltaEffects:stats.deltaEffects, tokenShapes:[...stats.shapes]},null,1));
  if(sample) console.log('SAMPLE', JSON.stringify(sample,null,1).slice(0,2500));
}
