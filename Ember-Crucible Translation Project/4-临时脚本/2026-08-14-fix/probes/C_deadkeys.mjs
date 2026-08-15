import fs from 'fs'; import path from 'path';
const ROOT='C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';
const repos=['1-Ember汉化插件','2-Crucible汉化插件'];
const hits={actorActions:[], itemAdjective:[], actorAdjective:[], effAdjective:0, itemActions:0, actorLevelAny:{}};
function rec(node, trail, file){
  if(!node||typeof node!=='object')return;
  for(const [k,v] of Object.entries(node)){
    const t=[...trail,k];
    if(k==='actions'||k==='adjective'){
      // determine container context: nearest preceding container key
      const idx=[...trail].reverse().findIndex(x=>['items','actors','effects','entries'].includes(x));
      const container = idx>=0 ? [...trail].reverse()[idx] : '(root)';
      const rk=`${container}:${k}`;
      hits.actorLevelAny[rk]=(hits.actorLevelAny[rk]||0)+1;
      if(k==='actions'&&container==='actors') hits.actorActions.push(file+' /'+t.join('/'));
      if(k==='adjective'&&container==='items') hits.itemAdjective.push(file+' /'+t.join('/'));
      if(k==='adjective'&&container==='actors') hits.actorAdjective.push(file+' /'+t.join('/'));
    }
    if(v&&typeof v==='object') rec(v,t,file);
  }
}
for(const repo of repos) for(const dir of ['en','cn']){
  const d=path.join(ROOT,repo,'compendium',dir);
  for(const f of fs.readdirSync(d)){ if(f==='_source.json')continue;
    rec(JSON.parse(fs.readFileSync(path.join(d,f),'utf8')).entries??{}, [], `${repo}/${dir}/${f}`); }
}
console.log('container:key counts', hits.actorLevelAny);
console.log('actor-level actions:', hits.actorActions.length, hits.actorActions.slice(0,5));
console.log('item-level adjective:', hits.itemAdjective.length, hits.itemAdjective.slice(0,5));
console.log('actor-level adjective:', hits.actorAdjective.length);
