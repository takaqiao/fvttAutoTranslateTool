import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createRefocusAdapter,createRefocusProvider} from '../../scripts/exploration/refocus.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {chooseNext} from '../../scripts/exploration/policy.mjs';

function sourceFixture(){
 const game={},actor={actorUUID:'H',pool:{poolUUID:'H',ready:true},hp:{value:10,max:10},focus:{value:0,max:1},hasActiveToken:true,items:[],slugs:[],assuranceSkills:[],medicine:{rank:0},refocusUnsupported:[]};
 const capabilities={snapshot:async()=>[actor],discover:async()=>actor};
 const adapter=createRefocusAdapter({game,fromUuid:async()=>null,ownerOperations:{}});
 const provider=createRefocusProvider({game,capabilities,refocusEvents:adapter});
 return {game,actor,capabilities,provider};
}

test('missing native refocus source pauses before world time or an executor grant',async()=>{
 const f=sourceFixture();let data={sessions:{},activities:{},clocks:{}},advances=0;
 const ledger=createLedger({read:async()=>data,write:async value=>{data=value},isAuthority:()=>true});
 const coordinator=createCoordinator({ledger,capabilities:f.capabilities,providers:[f.provider],clock:{advanceTo:async()=>{advances++;return {status:'confirmed'}}},policy:chooseNext,isAuthority:()=>true,now:()=>0,ownerOperations:{createActivityContext:async()=>({})}});
 const session=await coordinator.start({id:'S',actorUUIDs:['H'],goalsByPool:[],requireFullFocus:true,autoRun:false});
 await coordinator.step(session.id);const snapshot=await coordinator.snapshot(session.id);
 assert.equal(snapshot.session.stopReason,'native-refocus-unavailable');assert.equal(advances,0);
 assert.equal(snapshot.activities.length,1);assert.equal(snapshot.activities[0].state,'blocked');assert.equal(snapshot.activities[0].executor,undefined);assert.deepEqual(snapshot.clocks,[]);
});

test('a newly available native source permits the activity without invoking it at preflight',async()=>{
 const f=sourceFixture(),activity={actorUUID:'H',options:{}};
 assert.equal((await f.provider.begin(activity,{})).status,'blocked');let calls=0;
 f.game.PF2eWorkbench={refocus:async()=>{calls++}};
 assert.equal((await f.provider.begin(activity,{})).status,'started');assert.equal(calls,0);
});

test('missing native refocus source cannot claim a Three Pecks activity',async()=>{
 const f=sourceFixture();f.actor.threePecks=true;let claims=0;
 const provider=createRefocusProvider({game:f.game,capabilities:f.capabilities,refocusEvents:createRefocusAdapter({game:f.game,fromUuid:async()=>null,ownerOperations:{}}),salubriousKiss:{claimActivity:async()=>{claims++;return {status:'started'}}}});
 assert.equal((await provider.begin({actorUUID:'H',options:{threePecks:true}},{})).reason,'native-refocus-unavailable');assert.equal(claims,0);
});
