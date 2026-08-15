import fs from 'fs'; import path from 'path';
const ROOT='C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';
const repos=['1-Ember汉化插件','2-Crucible汉化插件'];
for(const repo of repos) for(const dir of ['en','cn']){
  const d=path.join(ROOT,repo,'compendium',dir);
  for(const f of fs.readdirSync(d)){ if(f==='_source.json')continue;
    const j=JSON.parse(fs.readFileSync(path.join(d,f),'utf8'));
    let adj=0, act=0;
    for(const e of Object.values(j.entries??{})){ if(e&&typeof e==='object'){ if('adjective' in e) adj++; if('actions' in e) act++; } }
    if(adj||act) console.log(`${repo}/${dir}/${f}: rootAdjective=${adj} rootActions=${act}`);
  }
}
