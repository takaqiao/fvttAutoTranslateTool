import fs from 'fs';
const S='C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/892fdb28-d096-4415-94d1-99d12c38ef86/scratchpad/';
const data = JSON.parse(fs.readFileSync(S+'raw_adv.json','utf8'));
const cnDir='C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/cn/';

for (const pack of ['adventure','crucible-adventure']) {
  const adv = data[pack][0][1];
  const byId = new Map((adv.actors||[]).map(a=>[a._id,a]));
  const cn = JSON.parse(fs.readFileSync(cnDir+`ember.${pack}.json`,'utf8'));
  const advEntry = Object.values(cn.entries)[0];
  const cnActors = advEntry?.actors || {};
  console.log('=====',pack,'cn actors keys:',Object.keys(cnActors).length);
  const used = new Map(); // actorId -> {actorName, cnName, protoName, displayName, tokenNames:Set}
  for (const j of adv.journal||[]) for (const p of j.pages||[]) {
    const toks=p.system?.encounter?.tokens; if(!Array.isArray(toks)) continue;
    for (const t of toks) for (const a of (t.actors||[])) {
      const tn=a.tokenData?.name; if(!tn) continue;
      const id=(a.actor||'').split('.').pop(); const act=byId.get(id); if(!act) continue;
      if(!used.has(id)) used.set(id,{actorName:act.name,protoName:act.prototypeToken?.name,
        displayName:act.prototypeToken?.displayName, actorLink:act.prototypeToken?.actorLink,
        cn: cnActors[act.name]?.name ?? cnActors[act._id]?.name ?? null, tokenNames:new Set()});
      used.get(id).tokenNames.add(tn);
    }
  }
  const arr=[...used.values()];
  console.log(' distinct actors referenced by named tokens:',arr.length);
  console.log(' actors with a CN name:',arr.filter(a=>a.cn).length,' without:',arr.filter(a=>!a.cn).length);
  const dn={}; for(const a of arr) dn[a.displayName]=(dn[a.displayName]||0)+1;
  console.log(' prototypeToken.displayName distribution:',JSON.stringify(dn));
  console.log(' sample:',JSON.stringify(arr.slice(0,8).map(a=>({actor:a.actorName,cn:a.cn,proto:a.protoName,dn:a.displayName,toks:[...a.tokenNames].slice(0,4)})),null,0));
  fs.writeFileSync(S+`ctx_${pack}.json`, JSON.stringify(arr.map(a=>({...a,tokenNames:[...a.tokenNames]})),null,1));
}
