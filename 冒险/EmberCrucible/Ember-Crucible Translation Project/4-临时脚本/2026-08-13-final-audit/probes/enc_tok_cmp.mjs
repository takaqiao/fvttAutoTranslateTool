import fs from 'fs';
const S='C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/892fdb28-d096-4415-94d1-99d12c38ef86/scratchpad/';
const data = JSON.parse(fs.readFileSync(S+'raw_adv.json','utf8'));
const enDir='C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/en/';
const cnDir='C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/cn/';

// gather all strings present in en baseline (as values, any depth) per pack + globally
function collectStrings(o, set) {
  if (typeof o === 'string') { set.add(o); return; }
  if (Array.isArray(o)) { o.forEach(x=>collectStrings(x,set)); return; }
  if (o && typeof o === 'object') { for (const [k,v] of Object.entries(o)) { set.add(k); collectStrings(v,set); } }
}
const enAll = new Set();
for (const f of fs.readdirSync(enDir)) if (f.endsWith('.json') && f!=='_source.json')
  collectStrings(JSON.parse(fs.readFileSync(enDir+f,'utf8')), enAll);

for (const pack of ['adventure','crucible-adventure']) {
  const adv = data[pack][0][1];
  const byId = new Map((adv.actors||[]).map(a=>[a._id,a]));
  const rows=[];
  for (const j of adv.journal||[]) for (const p of j.pages||[]) {
    const toks=p.system?.encounter?.tokens; if(!Array.isArray(toks)) continue;
    for (const t of toks) for (const a of (t.actors||[])) {
      const tn=a.tokenData?.name; if(!tn) continue;
      const id=(a.actor||'').split('.').pop();
      const act=byId.get(id);
      rows.push({j:j.name,p:p.name,tn,id,
        actorName: act? act.name : null,
        protoName: act? act.prototypeToken?.name : null,
        actorLink: act? act.prototypeToken?.actorLink : null});
    }
  }
  const distinct=new Map();
  for(const r of rows){ if(!distinct.has(r.tn)) distinct.set(r.tn,[]); distinct.get(r.tn).push(r); }
  let sameAsProto=0, diffProto=0, missingActor=0, linked=0;
  const diffs=[];
  for (const r of rows){
    if(!r.actorName){missingActor++;continue;}
    if(r.actorLink) linked++;
    if(r.tn===r.protoName) sameAsProto++; else {diffProto++; }
  }
  const distSame=[...distinct.entries()].filter(([tn,rs])=>rs.every(r=>r.tn===r.protoName));
  const distDiff=[...distinct.entries()].filter(([tn,rs])=>rs.some(r=>r.tn!==r.protoName));
  console.log('=====',pack);
  console.log(' rows',rows.length,'distinct',distinct.size,'missingActorDoc',missingActor);
  console.log(' rows tn===prototypeToken.name:',sameAsProto,' differ:',diffProto,' actorLink=true rows:',linked);
  console.log(' distinct names identical to proto:',distSame.length,' differing:',distDiff.length);
  console.log(' sample differing:',JSON.stringify(distDiff.slice(0,12).map(([tn,rs])=>[tn,rs[0].actorName,rs[0].protoName])));
  console.log(' sample identical:',JSON.stringify(distSame.slice(0,12).map(([tn,rs])=>[tn,rs[0].protoName])));
  // presence in en baseline
  const inEn=[...distinct.keys()].filter(n=>enAll.has(n));
  const notInEn=[...distinct.keys()].filter(n=>!enAll.has(n));
  console.log(' distinct tokenNames present as some string/key in compendium/en:',inEn.length,' absent:',notInEn.length);
  console.log(' absent sample:',JSON.stringify(notInEn.slice(0,20)));
  fs.writeFileSync(S+`cmp_${pack}.json`, JSON.stringify({rows,distDiff:distDiff.map(([tn,rs])=>({tn,actorName:rs[0].actorName,protoName:rs[0].protoName,n:rs.length})),distSame:distSame.map(([tn,rs])=>({tn,n:rs.length})),inEn,notInEn},null,1));
}
