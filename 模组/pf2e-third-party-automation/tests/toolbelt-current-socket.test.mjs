import test from 'node:test';
import assert from 'node:assert/strict';
import {socketFixture} from './toolbelt-current-fixture.mjs';

const binding={permitNonce:'permit',applicationNonce:'application',ownerUserId:'O',patientUUID:'Actor.P',poolUUID:'Actor.M'};
function subscribe(f,{authorized=true}={}){
 const terminals=[];
 f.owner.api.subscribe(event=>event.phase==='prepare'?{binding,validate:()=>true,beforeWrite:()=>true}:event.phase==='write'?void terminals.push(event.terminalPromise):undefined);
 f.gm.api.subscribe(event=>event.phase==='authorize'?{validate:()=>authorized,beforeWrite:()=>true}:event.phase==='write'?void terminals.push(event.terminalPromise):undefined);
 return terminals;
}
for(const remote of [false,true])test(`current ${remote?'authenticated GM socket':'OWNER'} retains the original master Promise`,async()=>{
 const f=socketFixture({remote}),terminals=subscribe(f);await f.owner.tool.pre(f.owner.patient,{'system.attributes.hp.value':20});await f.turn();
 assert.equal(f.writes.length,1);assert.equal(f.writes[0].id,remote?'G':'O');assert.equal(f.writes[0].fields.__explorationManualPool,undefined);
 let settled=false;terminals[0].then(()=>settled=true);await f.turn();assert.equal(settled,false);f.finish();
 assert.equal((await terminals[0]).writerUserId,remote?'G':'O');
});
test('current original unpatched socket resolves the patient before the master',async()=>{
 const f=socketFixture({patched:false});await f.owner.tool.pre(f.owner.patient,{'system.attributes.hp.value':20});await f.turn();
 assert.equal(f.writes.length,1);f.finish();
});
test('current forward awaits prepare authorization before the patient continuation',async()=>{
 const f=socketFixture({remote:false});let allow,returned=false;
 const approval=new Promise(resolve=>allow=resolve);
 f.owner.api.subscribe(event=>event.phase==='prepare'?approval:undefined);
 const pending=f.owner.tool.pre(f.owner.patient,{'system.attributes.hp.value':20}).then(()=>returned=true);
 await f.turn();assert.equal(returned,false);assert.equal(f.writes.length,0);
 allow({binding,validate:()=>true,beforeWrite:()=>true});await pending;assert.equal(f.writes.length,1);f.finish();
});
test('current socket awaits document resolution before independent sender authorization',async()=>{
 const f=socketFixture(),terminals=subscribe(f);let resolve;
 f.gm.context.fromUuid=()=>new Promise(finish=>resolve=finish);
 await f.owner.tool.pre(f.owner.patient,{'system.attributes.hp.value':20});await f.turn();
 assert.equal(f.writes.length,0);assert.equal(terminals.length,0);
 resolve(f.gm.master);await f.turn();assert.equal(f.writes.length,1);f.finish();await terminals[0];
});
for(const reason of ['tampered owner','unauthorized','wrong type','inactive GM'])test(`current socket refuses ${reason} without a master write`,async()=>{
 const f=socketFixture(),terminals=subscribe(f,{authorized:reason!=='unauthorized'});
 if(reason==='tampered owner')f.tamper(packet=>({...packet,__explorationManualPool:{...packet.__explorationManualPool,ownerUserId:'G'}}));
 if(reason==='wrong type')f.tamper(packet=>({...packet,__type__:'other'}));
 if(reason==='inactive GM')f.gm.game.user.isActiveGM=false;
 await f.owner.tool.pre(f.owner.patient,{'system.attributes.hp.value':20});await f.turn();
 assert.equal(f.writes.length,0);assert.equal(terminals.length,0);f.finish();
});
test('current master rejection cannot produce terminal success or reuse the application nonce',async()=>{
 const f=socketFixture(),terminals=subscribe(f);await f.owner.tool.pre(f.owner.patient,{'system.attributes.hp.value':20});await f.turn();
 assert.equal(f.writes.length,1);f.reject(Error('master failed'));await assert.rejects(terminals[0],/master failed/);
 f.replay();await f.turn();assert.equal(f.writes.length,1);assert.equal(terminals.length,1);
});
test('current emitter binds GM-local sender and registers and unregisters the same callback once',async()=>{
 const f=socketFixture({patched:false}),emitter=f.gm.context.emitter;let sender;
 f.gm.context.directCallback=(_packet,id)=>{sender=id;return 'local result'};
 assert.equal(await emitter.call({senderId:'spoofed'}),'local result');assert.equal(sender,'G');
 const initial=f.gm.handlers.size;emitter.activate();emitter.activate();assert.equal(f.gm.handlers.size,initial+1);
 emitter.disable();emitter.disable();assert.equal(f.gm.handlers.size,initial);f.finish();
});
