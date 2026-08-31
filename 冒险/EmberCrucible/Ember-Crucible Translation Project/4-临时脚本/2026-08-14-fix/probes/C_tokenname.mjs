import fs from 'fs';
import path from 'path';
const ROOT='C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';
const repos=[['1-Ember汉化插件'],['2-Crucible汉化插件']];
function walkActors(node, cb, trail=[]){
  if(!node||typeof node!=='object')return;
  for(const [k,v] of Object.entries(node)){
    if(k==='actors'&&v&&typeof v==='object'&&!Array.isArray(v)){
      for(const [an,av] of Object.entries(v)) cb([...trail,'actors',an], av);
    }
    if(v&&typeof v==='object') walkActors(v, cb, [...trail,k]);
  }
}
for(const [repo] of repos){
  for(const f of fs.readdirSync(path.join(ROOT,repo,'compendium','en'))){
    if(f==='_source.json')continue;
    const en=JSON.parse(fs.readFileSync(path.join(ROOT,repo,'compendium','en',f),'utf8'));
    const cnp=path.join(ROOT,repo,'compendium','cn',f);
    const cn=fs.existsSync(cnp)?JSON.parse(fs.readFileSync(cnp,'utf8')):null;
    // top-level entries as actors too
    let enTN=0, enName=0, enDiff=0;
    const enMap=new Map();
    const collect=(obj,map)=>{
      if(!obj)return;
      const rec=(node,trail)=>{
        if(!node||typeof node!=='object')return;
        if(typeof node.tokenName==='string'){ map.set(trail.join('/'),{name:node.name,tokenName:node.tokenName}); }
        for(const [k,v] of Object.entries(node)) if(v&&typeof v==='object') rec(v,[...trail,k]);
      };
      rec(obj.entries||{}, []);
    };
    collect(en,enMap);
    const cnMap=new Map(); if(cn)collect(cn,cnMap);
    if(enMap.size===0&&cnMap.size===0)continue;
    let bothDiffEn=0;
    for(const [k,v] of enMap) if(v.name!==v.tokenName) bothDiffEn++;
    let cnMissing=0;
    for(const k of enMap.keys()) if(!cnMap.has(k)) cnMissing++;
    let cnDiffFromName=0;
    for(const [k,v] of cnMap) if(v.name!==v.tokenName) cnDiffFromName++;
    console.log(`${repo}/${f}: en.tokenName=${enMap.size} (en name!=tokenName ${bothDiffEn}) cn.tokenName=${cnMap.size} (cn name!=tokenName ${cnDiffFromName}) enKeysMissingInCn=${cnMissing}`);
  }
}
