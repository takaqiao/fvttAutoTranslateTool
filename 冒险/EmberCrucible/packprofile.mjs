import { ClassicLevel } from "classic-level";
import { readdirSync, existsSync } from "node:fs";
const ROOTS=[["crucible","C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs"],
             ["ember","C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs"]];
for(const [src,root] of ROOTS){ if(!existsSync(root)) continue;
  for(const pk of readdirSync(root)){ let db;
    try{db=new ClassicLevel(`${root}/${pk}`,{keyEncoding:"utf8",valueEncoding:"json"});await db.open();}catch{continue;}
    let docs=0, byType={}, topTalents=0, embeddedTalents=0, actorTypes={};
    for await(const [,doc] of db.iterator()){
      docs++; byType[doc.type??"?"]=(byType[doc.type??"?"]??0)+1;
      if(doc.type==="talent") topTalents++;
      const eat = ac => { actorTypes[ac.type??"?"]=(actorTypes[ac.type??"?"]??0)+1;
                          for(const it of (ac.items??[])) if(it.type==="talent") embeddedTalents++; };
      if(doc.type==="hero"||doc.type==="adversary"||doc.type==="group") eat(doc);
      for(const ac of (doc?.actors??[])) eat(ac);
    }
    await db.close();
    const t=Object.entries(byType).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([k,v])=>`${k}:${v}`).join(" ");
    const a=Object.entries(actorTypes).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${k}:${v}`).join(" ");
    console.log(`${src}/${pk.padEnd(22)} 文档${String(docs).padStart(5)} | 顶层天赋${String(topTalents).padStart(4)} 内嵌天赋${String(embeddedTalents).padStart(5)} | 类型 ${t}${a?` | actor ${a}`:""}`);
  } }
