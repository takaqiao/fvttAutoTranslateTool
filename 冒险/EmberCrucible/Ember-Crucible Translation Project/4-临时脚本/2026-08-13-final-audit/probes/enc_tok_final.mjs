import fs from 'fs';
const S='C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/892fdb28-d096-4415-94d1-99d12c38ef86/scratchpad/';
const data = JSON.parse(fs.readFileSync(S+'raw_adv.json','utf8'));
const sets={};
for (const pack of ['adventure','crucible-adventure']) {
  const adv=data[pack][0][1];
  const tdKeys={}, rows=[], gated={y:0,n:0};
  const pages=new Set(), journals=new Set();
  for (const j of adv.journal||[]) for (const p of j.pages||[]) {
    const enc=p.system?.encounter; if(!enc?.tokens) continue;
    for (const g of enc.tokens) {
      const isGated = !!(g.outcomes?.length);
      for (const a of (g.actors||[])) {
        for (const k of Object.keys(a.tokenData||{})) tdKeys[k]=(tdKeys[k]||0)+1;
        const tn=a.tokenData?.name; if(!tn) continue;
        gated[isGated?'y':'n']++;
        rows.push(tn); pages.add(j.name+'::'+p.name); journals.add(j.name);
      }
    }
  }
  sets[pack]=new Set(rows);
  console.log('===',pack);
  console.log(' tokenData key histogram:',JSON.stringify(Object.entries(tdKeys).sort((a,b)=>b[1]-a[1])));
  console.log(' named rows',rows.length,'distinct',sets[pack].size,'pages',pages.size,'journals',journals.size);
  console.log(' rows in outcome-gated groups:',gated.y,' ungated:',gated.n);
}
const A=sets['adventure'],B=sets['crucible-adventure'];
console.log('set identical:', A.size===B.size && [...A].every(x=>B.has(x)));
console.log('union size:', new Set([...A,...B]).size);
