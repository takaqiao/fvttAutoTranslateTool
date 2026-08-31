import { ClassicLevel } from "classic-level";
const WANT=["Assassin","Beastmaster","Fulcrum","Gesture: Step","Inflection: Extend","Inflection: Negate",
  "Intercept","Shapeshifter","Strategic Repositioning","Unshakeable Stance","Abyssal Remains","Wirrun Lineage"];
const found={}; for(const w of WANT) found[w]=[];
const P="C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible/packs";
const E="C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs";
// 玩家可及的「天赋树 / 建卡」来源
const PLAYER=[[P,"talent"],[P,"spell"],[P,"archetype"],[P,"ancestry"],[P,"background"],[P,"taxonomy"],
              [P,"summons"],[P,"pregens"],[E,"crucible-character"]];
for(const [root,pk] of PLAYER){ let db;
  try{db=new ClassicLevel(`${root}/${pk}`,{keyEncoding:"utf8",valueEncoding:"json"});await db.open();}catch{continue;}
  for await(const [,doc] of db.iterator()){
    if(doc.type==="talent"&&found[doc.name]) found[doc.name].push(`${pk}(顶层)`);
    for(const it of (doc?.items??[])) if(it.type==="talent"&&found[it.name]) found[it.name].push(`${pk}>${doc.name}`);
    for(const ac of (doc?.actors??[])) for(const it of (ac?.items??[]))
      if(it.type==="talent"&&found[it.name]) found[it.name].push(`${pk}>${ac.type}:${ac.name}`);
  }
  await db.close(); }
// playtest 单独看：里面 hero 与 adversary 混装
{ let db; try{db=new ClassicLevel(`${P}/playtest`,{keyEncoding:"utf8",valueEncoding:"json"});await db.open();
  for await(const [,doc] of db.iterator())
    for(const ac of (doc?.actors??[])) for(const it of (ac?.items??[]))
      if(it.type==="talent"&&found[it.name]) found[it.name].push(`playtest>${ac.type}:${ac.name}`);
  await db.close(); }catch{} }
for(const w of WANT){
  const hits=[...new Set(found[w])];
  const hero=hits.some(h=>h.includes("hero:")||h.includes("(顶层)")&&!h.startsWith("adversary"));
  console.log(`${hero?"✅ 玩家":"❔     "} ${w.padEnd(24)} ${hits.length?hits.slice(0,4).join(" ; "):"（玩家侧未找到）"}`);
}
