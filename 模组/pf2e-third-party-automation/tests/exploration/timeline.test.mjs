import {test} from 'node:test';import assert from 'node:assert/strict';
import {reconstructEarliest} from '../../scripts/exploration/timeline.mjs';
const treatment=(id,actor,patients,order,immunity=3600)=>({id,actorUUID:actor,patientUUIDs:patients,order,durationSeconds:600,treatmentImmunitySeconds:immunity,kind:'treatment'});
const solve=activities=>reconstructEarliest({startedAt:0,activities,assumptions:['different-actors-may-overlap']});
test('same patient three treatments take130 minutes, Continual Recovery takes30',()=>{
 for(const [immune,duration]of [[3600,7800],[600,1800]])assert.equal(solve([0,1,2].map(i=>treatment(`T${i}`,'H',['P'],i,immune))).durationSeconds,duration);
 assert.equal(solve([0,1,2].map(i=>treatment(`T${i}`,'H',[`P${i}`],i))).durationSeconds,1800);
});
test('two healers overlap; receiving treatment does not occupy patient',()=>{
 assert.equal(solve([treatment('A','H1',['P1'],0),treatment('B','H2',['P2'],0)]).durationSeconds,600);
 assert.equal(solve([treatment('A','H1',['P1'],0),treatment('B','H2',['P2'],0),treatment('C','H1',['P3'],1)]).durationSeconds,1200);
 assert.equal(solve([treatment('A','H',['P'],0),{id:'R',actorUUID:'P',patientUUIDs:[],order:0,kind:'refocus',durationSeconds:600}]).durationSeconds,600);
});
test('only proven same-use Ward groups merge; missing group remains separate',()=>{
 const a=treatment('A','H',['P1'],0),b=treatment('B','H',['P2'],1);
 assert.equal(solve([a,b]).durationSeconds,1200);
 const proven=[a,b].map(x=>({...x,groupId:'G',groupProof:'native-use:N',order:0}));
 assert.equal(solve(proven).durationSeconds,600);
 assert.equal(solve([a,{...b,groupId:'G'}]).certainty,'incomplete');
});
test('cycles and observed cooldown contradictions remain explicit',()=>{
 const cycle=solve([{...treatment('A','H1',['P1'],0),dependsOn:['B']},{...treatment('B','H2',['P2'],0),dependsOn:['A']}]);assert.equal(cycle.certainty,'incomplete');assert.ok(cycle.missing.some(x=>x.reason==='dependency-cycle'));
 const bad=solve([{...treatment('A','H',['P'],0),observedStart:0,observedEnd:600},{...treatment('B','H',['P'],1),observedStart:600,observedEnd:1200}]);assert.equal(bad.certainty,'contradictory');assert.ok(bad.missing.some(x=>x.reason==='observed-before-ready'));
});
