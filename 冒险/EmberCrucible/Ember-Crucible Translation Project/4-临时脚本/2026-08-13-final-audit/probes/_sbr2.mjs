import fs from 'fs'; import path from 'path'; import { createRequire } from 'module';
const require_ = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require_('classic-level');
const DATA='C:/Users/Taka/AppData/Local/FoundryVTT/Data';
const PKGS=[['crucible(system)',path.join(DATA,'systems/crucible'),'system.json'],['ember(module)',path.join(DATA,'modules/ember'),'module.json']];
const rows={}; // pack -> type -> {string,object}
const advActorItems={}; // pack -> type -> n  (items embedded in Adventure actors: the re-import blast radius)
function bump(o,a,b,f){ ((o[a]??={})[b]??={string:0,object:0,other:0})[f]+=1; }
function note(pack,item,sink){ if(!item||typeof item!=='object')return; const t=item.type??'(none)'; const d=item.system?.description;
  bump(sink,pack,t, typeof d==='string'?'string':(d&&typeof d==='object'?'object':'other')); }
for(const [label,dir,mf] of PKGS){
  const m=JSON.parse(fs.readFileSync(path.join(dir,mf),'utf8'));
  for(const pack of m.packs??[]){
    const pdir=path.join(dir,'packs',path.basename(pack.path??pack.name));
    if(!fs.existsSync(pdir))continue;
    const db=new ClassicLevel(pdir,{createIfMissing:false}); await db.open();
    const key0=`${label}:${pack.name}`;
    for await(const [k,v] of db.iterator()){
      const key=k.toString(); let doc; try{doc=JSON.parse(v.toString())}catch{continue}
      if(key.startsWith('!items!')) note(key0,doc,rows);
      else if(key.startsWith('!actors!')) for(const it of doc.items??[]) note(key0+'|actorEmbedded',it,rows);
      else if(key.startsWith('!adventures!')){
        for(const a of doc.actors??[]) for(const it of a.items??[]) { note(key0+'|advActorEmbedded',it,rows); bump(advActorItems,key0,it.type??'(none)', typeof it.system?.description==='string'?'string':'object'); }
        for(const it of doc.items??[]) note(key0+'|advItems',it,rows);
      }
    }
    await db.close();
  }
}
console.log(JSON.stringify({rows,advActorItems},null,1));
