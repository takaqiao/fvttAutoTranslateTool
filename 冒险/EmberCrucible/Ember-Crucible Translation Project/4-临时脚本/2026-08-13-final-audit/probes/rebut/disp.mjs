import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
const db = new ClassicLevel(path.join(PKG,'crucible-adventure'), { valueEncoding:'json' });
await db.open();
const proto=new Map(); const dispTok=new Map(); const dispProto=new Map();
let combatGroups=0, totalGroups=0, partyGroups=0;
for await (const [k,doc] of db.iterator()) {
  if(!k.toString().startsWith('!adventures!')) continue;
  for(const a of doc.actors??[]) proto.set(a._id,a.prototypeToken??{});
  for(const j of doc.journal??[]) for(const p of j.pages??[]) for(const t of p?.system?.encounter?.tokens??[]) {
    totalGroups++; if(t.combat) combatGroups++; if(t.party) partyGroups++;
    for(const a of t.actors??[]) {
      const td=a.tokenData; if(!td?.name) continue;
      dispTok.set(td.displayName, (dispTok.get(td.displayName)||0)+1);
      const id=String(a.actor??'').replace(/^Actor\./,'').split('.').pop();
      const pt=proto.get(id)??{};
      dispProto.set(pt.displayName,(dispProto.get(pt.displayName)||0)+1);
    }
  }
}
await db.close();
console.log('token groups',totalGroups,'combat:true',combatGroups,'party:true',partyGroups);
console.log('tokenData.displayName distribution:',JSON.stringify([...dispTok]));
console.log('prototypeToken.displayName distribution (for named overrides):',JSON.stringify([...dispProto]));
console.log('CONST TOKEN_DISPLAY_MODES: NONE=0 CONTROL=10 OWNER_HOVER=20 HOVER=30 OWNER=40 ALWAYS=50');
