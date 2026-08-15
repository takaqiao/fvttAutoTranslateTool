import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const db=new ClassicLevel(path.join('C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs','crucible-adventure'),{valueEncoding:'json'});
await db.open();
const all=new Map(); let pages=0; const eff=[];
for await(const [k,doc] of db.iterator()){
  if(!k.toString().startsWith('!adventures!'))continue;
  for(const j of doc.journal??[])for(const p of j.pages??[]){pages++;const s=p.system?.development?.status;all.set(s,(all.get(s)||0)+1);
    for(const t of p?.system?.encounter?.tokens??[])for(const a of t.actors??[])for(const e of a?.tokenData?.delta?.effects??[])eff.push([e.name,e.statuses,e.img?1:0]);}
}
await db.close();
console.log('total pages',pages,'development.status distribution:',JSON.stringify([...all].sort((a,b)=>b[1]-a[1])));
console.log('delta effects leaves:',eff.length, JSON.stringify(eff));
