import test from 'node:test';
import assert from 'node:assert/strict';
import {METAPOWER_SOURCES,POWER_PROFILES,buildChannelSnapshot,siphonMultiplier} from '../scripts/metapower/rules.mjs';

function channel(feats){
 const actor={uuid:'Actor.a',level:1,items:feats,flags:{pf2e:{eldamon:{element:{trait:'electricity'}}}}};
 const item={actor,uuid:'Actor.a.Item.power',sourceId:Object.values(POWER_PROFILES).find(p=>p.id==='electric-surge').sourceUuid,system:{traits:{value:['electricity']}}};
 return buildChannelSnapshot({kind:'siphoning',actor,item});
}
const feat=extra=>({sourceId:METAPOWER_SOURCES.disruptiveSiphon,...extra});
test('active exact Disruptive source grants full associated-trait damage',()=>{
 assert.equal(siphonMultiplier(channel([feat()]),['electricity']),1);
});
for(const extra of [{suppressed:true},{isSuppressed:true},{system:{suppressed:true}}])test(`suppressed Disruptive source preserves ordinary half damage: ${JSON.stringify(extra)}`,()=>{
 const snapshot=channel([feat(extra)]);assert.equal(snapshot.disruptive,false);assert.equal(siphonMultiplier(snapshot,['electricity']),0.5);
});
test('an active exact duplicate remains effective after a suppressed duplicate',()=>{
 const snapshot=channel([feat({suppressed:true}),feat({suppressed:false})]);assert.equal(snapshot.disruptive,true);assert.equal(siphonMultiplier(snapshot,['electricity']),1);
});
test('same slug and conflicting authoritative source cannot supply Disruptive',()=>{
 const snapshot=channel([{system:{slug:'disruptive-siphon'}},{sourceId:'custom',_stats:{compendiumSource:METAPOWER_SOURCES.disruptiveSiphon}}]);assert.equal(snapshot.disruptive,false);assert.equal(siphonMultiplier(snapshot,['electricity']),0.5);
});
