import fs from 'fs'; import path from 'path';
const ROOT='C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';
for(const repo of ['1-Ember汉化插件','2-Crucible汉化插件']){
  const d=path.join(ROOT,repo,'compendium');
  for(const f of fs.readdirSync(path.join(d,'en'))){
    if(f==='_source.json')continue;
    const en=JSON.parse(fs.readFileSync(path.join(d,'en',f),'utf8'));
    const cn=JSON.parse(fs.readFileSync(path.join(d,'cn',f),'utf8'));
    const rec=(n,c,trail)=>{
      if(!n||typeof n!=='object')return;
      if(n.tokens&&typeof n.tokens==='object'&&!Array.isArray(n.tokens)){
        for(const [k,v] of Object.entries(n.tokens)) console.log(`${repo}/${f} scene[${trail.join('/')}] token EN="${k}" CN="${c?.tokens?.[k]??'(none)'}"`);
      }
      for(const [k,v] of Object.entries(n)) if(v&&typeof v==='object') rec(v, c?.[k], [...trail,k]);
    };
    rec(en.entries??{}, cn.entries??{}, []);
  }
}
