import { ClassicLevel } from "classic-level";
import { readdirSync, existsSync, writeFileSync } from "node:fs";
const WANT = new Set(["Assassin","Beastmaster","Fulcrum","Gesture: Step","Inflection: Extend",
  "Inflection: Negate","Intercept","Shapeshifter","Strategic Repositioning","Unshakeable Stance",
  "Abyssal Remains","Wirrun Lineage"]);
const ROOTS=[["crucible","C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs"],
             ["ember","C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs"]];
const strip=h=>String(h??"").replace(/<[^>]*>/g," ").replace(/\s+/g," ").trim();
const out={};
for(const [src,root] of ROOTS){ if(!existsSync(root)) continue;
  for(const pk of readdirSync(root)){ let db;
    try{db=new ClassicLevel(`${root}/${pk}`,{keyEncoding:"utf8",valueEncoding:"json"});await db.open();}catch{continue;}
    for await(const [,doc] of db.iterator()){
      const visit=it=>{ if(it?.type!=="talent"||!WANT.has(it.name)) return;
        if(out[it.name]) return;
        const acts=it.system?.actions; const L=Array.isArray(acts)?acts:Object.values(acts??{});
        out[it.name]={src,pack:pk,node:it.system?.node??null,description:strip(it.system?.description),
          actions:L.map(a=>({id:a.id,name:a.name,cost:a.cost,target:a.target,range:a.range,
            tags:a.tags,effects:(a.effects??[]).map(e=>({name:e.name,duration:e.duration,statuses:e.statuses,system:e.system})),
            description:strip(a.description)}))};
      };
      visit(doc); for(const it of (doc?.items??[])) visit(it);
      for(const ac of (doc?.actors??[])) for(const it of (ac?.items??[])) visit(it);
    }
    await db.close(); } }
writeFileSync("auto_full.json", JSON.stringify(out,null,1),"utf8");
console.log("取到", Object.keys(out).length, "个：", Object.keys(out).join(", "));
