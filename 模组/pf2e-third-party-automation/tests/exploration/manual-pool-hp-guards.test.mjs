import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './manual-pool-provider-fixture.mjs';

function hp(f,value){for(const c of [f.owner,f.gm]){c.master.system.attributes.hp.value=value;c.patient.system={attributes:{hp:{value,max:30,temp:0}}}}}
function operation(f,actor,{beforeUpdate=()=>{},beforePreUpdate=()=>{}}={}){
 return async()=>{await Promise.resolve();beforeUpdate();
  await f.owner.update.call(actor,async(changes,options)=>{await Promise.resolve();beforePreUpdate();await f.owner.patient._preUpdate(changes,options);return f.owner.patient},{'system.attributes.hp.value':20},{});
  f.messages.set(f.receipt.id,f.receipt);return {receipt:f.receipt};
 };
}
for(const stage of ['old clone','owner begin wait','native await','patient preUpdate await','local forwarding await','GM authorization await'])test('stale HP blocks the original master write: '+stage,async()=>{
 const remote=stage==='GM authorization await',f=await fixture(remote);hp(f,1);
 const clone={...f.owner.patient,system:structuredClone(f.owner.patient.system)};
 try{
  const grant=await f.owner.broker.claim(f.request);
  if(stage==='old clone')hp(f,10);
  if(stage==='owner begin wait'){
   const original=f.ledger.beginManualPoolApplication.bind(f.ledger);
   f.ledger.beginManualPoolApplication=async(...args)=>{const result=await original(...args);hp(f,10);return result};
  }
  if(stage==='local forwarding await'||stage==='GM authorization await'){
   const c=remote?f.gm:f.owner,original=c.api.subscribe.bind(c.api);
   // The adapter still receives the one real subscriber answer. This models
   // another HP source completing while that answer's Promise is pending.
   c.api.subscribe=fn=>original(event=>{
    const answer=fn(event);
    return answer&&event.phase===(remote?'authorize':'prepare')?Promise.resolve(answer).then(value=>{hp(f,10);return value}):answer;
   });
   if(remote){f.gm.broker.stop();f.gm.broker.start()}
  }
  const pending=f.owner.broker.withApplication(grant,f.owner.patient,operation(f,clone,{
   beforeUpdate:()=>{if(stage==='native await')hp(f,10)},
   beforePreUpdate:()=>{if(stage==='patient preUpdate await')hp(f,10)}
  }),{updateActor:clone});
  pending.catch(()=>{});await f.turn();f.finish();
  await assert.rejects(pending,stage==='GM authorization await'?/owner-response-timeout-unknown-no-retry/:/manual-pool/);assert.equal(f.writes.length,0);
  if(remote){
   const saved=Object.values((await f.ledger.getActivity('A')).proof.poolApplications)[0];
   assert.equal(saved.state,'applying');assert.equal(saved.terminal,undefined);
   const lookup=await f.owner.broker.lookup(f.request);assert.equal(lookup.status,'reserved');assert.equal(lookup.permitNonce,undefined);
   await assert.rejects(f.owner.broker.withApplication(grant,f.owner.patient,f.operation),/private-grant/);assert.equal(f.writes.length,0);
  }
 }finally{f.close()}
});

for(const change of ['missing','value','extra'])test('GM binds forwarding baseline to its own authenticated begin: '+change,async()=>{
 const f=await fixture(true);hp(f,1);try{
  const grant=await f.owner.broker.claim(f.request),emit=f.owner.game.socket.emit;
  f.owner.game.socket.emit=(channel,packet,...args)=>{
   if(packet.status==='forward'){
    packet=structuredClone(packet);
    if(change==='missing')delete packet.proof.hpBaseline;
    if(change==='value')packet.proof.hpBaseline.value=10;
    if(change==='extra')packet.proof.hpBaseline.extra=1;
   }
   return emit(channel,packet,...args);
  };
  const pending=f.owner.broker.withApplication(grant,f.owner.patient,f.operation);pending.catch(()=>{});f.finish();
  await assert.rejects(pending,/manual-pool/);assert.equal(f.writes.length,0);
  assert.equal((await f.owner.broker.lookup(f.request)).status,'reserved');
 }finally{f.close()}
});

for(const remote of [false,true])test('normal '+(remote?'GM':'OWNER')+' write may change HP before terminal validation',async()=>{
 const f=await fixture(remote);hp(f,1);try{
  const grant=await f.owner.broker.claim(f.request),pending=f.owner.broker.withApplication(grant,f.owner.patient,f.operation);pending.catch(()=>{});
  await f.writeStarted;assert.equal(f.writes.length,1);hp(f,20);f.finish();
  const result=await pending;assert.equal(result.poolReceipt.master.terminal,'fulfilled');
 }finally{f.close()}
});
test('full HP original noChange receipt remains valid without any master write',async()=>{
 const f=await fixture(false);hp(f,30);try{
  const grant=await f.owner.broker.claim(f.request);f.receipt.flags.pf2e.appliedDamage=null;f.messages.set(f.receipt.id,f.receipt);
  const result=await f.owner.broker.withApplication(grant,f.owner.patient,async()=>({receipt:f.receipt}));
  assert.equal(result.poolReceipt.noChange,true);assert.equal(f.writes.length,0);
 }finally{f.close()}
});

test('a legacy provider cannot bypass the synchronous baseline consumption contract',async()=>{
 const f=await fixture(false);hp(f,1);try{
  const grant=await f.owner.broker.claim(f.request),old={...f.owner.api.descriptor};delete old.hpBaselineGuardVersion;f.owner.api.descriptor=old;
  const pending=f.owner.broker.withApplication(grant,f.owner.patient,f.operation);pending.catch(()=>{});f.finish();
  await assert.rejects(pending,/manual-pool-provider-unavailable/);assert.equal(f.writes.length,0);assert.equal(f.applications(),0);
 }finally{f.close()}
});
for(const key of ['max','temp','sp missing','sp value','invalid number','computed hitPoints'])test('contextual HP input equality includes '+key,async()=>{
 const f=await fixture(false);hp(f,1);try{
  for(const c of [f.owner,f.gm])for(const actor of [c.patient,c.master])actor.system.attributes.hp.sp={value:2,max:4};
  const clone={...f.owner.patient,system:structuredClone(f.owner.patient.system)};
  if(key==='max')clone.system.attributes.hp.max=31;
  if(key==='temp')clone.system.attributes.hp.temp=1;
  if(key==='sp missing')delete clone.system.attributes.hp.sp;
  if(key==='sp value')clone.system.attributes.hp.sp.value=1;
  if(key==='invalid number')clone.system.attributes.hp.value=NaN;
  if(key==='computed hitPoints')Object.defineProperty(clone,'hitPoints',{get:()=>({value:10,max:30,temp:0})});
  const grant=await f.owner.broker.claim(f.request),pending=f.owner.broker.withApplication(grant,f.owner.patient,operation(f,clone),{updateActor:clone});pending.catch(()=>{});f.finish();
  await assert.rejects(pending,/manual-pool-hp/);assert.equal(f.writes.length,0);
 }finally{f.close()}
});
