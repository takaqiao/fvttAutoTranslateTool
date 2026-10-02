import {test} from 'node:test';
import assert from 'node:assert/strict';
import {seamFixture} from '../../tools/toolbelt-manual-pool/seam.test.mjs';

async function pendingNoop({remote=false,before=()=>{},updates,returnPromise=true}={}){
 const f=seamFixture({remote}),client=remote?f.gm:f.owner,master=client.master;
 master.system.attributes.hp={value:30,max:30,temp:0,sp:{value:4,max:4}};
 master._source={system:{attributes:{hp:{value:30,temp:0,sp:{value:4},details:''}}}};
 const fields=updates??{'system.attributes.hp.value':30,'system.attributes.hp.temp':0,'system.attributes.hp.sp.value':4};
 const binding={permitNonce:'permit',applicationNonce:'application',ownerUserId:'O',patientUUID:'Actor.P',poolUUID:'Actor.M'};
 let finish,reject,terminal,calls=0,valid=true;
 const promise=new Promise((resolve,fail)=>{finish=resolve;reject=fail});promise.catch(()=>{});
 master.update=actual=>{calls++;assert.deepEqual(structuredClone(actual),fields);return returnPromise?promise:undefined};
 const receive=event=>{
  if(event.phase==='prepare')return {binding,validate:()=>valid,beforeWrite:()=>true};
  if(event.phase==='authorize'){assert.equal(event.senderId,'O');return {validate:()=>valid,beforeWrite:()=>true}}
  if(event.phase==='write'){terminal=event.terminalPromise;terminal.catch(()=>{})}
 };
 f.owner.api.subscribe(receive);f.gm.api.subscribe(receive);before(master);
 await f.owner.tool.pre(f.owner.patient,structuredClone(fields));await f.turn();
 assert.equal(calls,1);assert.ok(terminal);
 return {f,client,master,terminal,finish,reject,calls:()=>calls,revoke:()=>{valid=false},fields};
}

for(const remote of [false,true]){
 const route=remote?'authenticated GM':'local OWNER';
 test(`${route} exact unchanged original master Promise is a fulfilled terminal`,async()=>{
  const f=await pendingNoop({remote});let settled=false;f.terminal.then(()=>settled=true,()=>settled=true);
  await f.f.turn();assert.equal(settled,false);f.finish(undefined);
  const proof=await f.terminal;assert.equal(proof.terminal,'fulfilled');assert.equal(proof.updateOutcome,'unchanged');
  assert.equal(proof.writerUserId,remote?'G':'O');assert.deepEqual(proof.fields,f.fields);assert.equal(f.calls(),1);
  const duplicate=f.f.owner.tool.pre(f.f.owner.patient,structuredClone(f.fields));
  if(remote)await duplicate;else await assert.rejects(duplicate,/manual-pool-native-authorization-changed/);
  await f.f.turn();assert.equal(f.calls(),1);
 });
 test(`${route} original master return still permits actual HP changes`,async()=>{
  const f=await pendingNoop({remote});f.master.system.attributes.hp.value=31;f.master._source.system.attributes.hp.value=31;f.finish(f.master);
  const proof=await f.terminal;assert.equal(proof.terminal,'fulfilled');assert.equal(Object.hasOwn(proof,'updateOutcome'),false);
 });
 test(`${route} absent stamina remains absent without defaulting to zero`,async()=>{
  const f=await pendingNoop({remote,updates:{'system.attributes.hp.value':30},before:m=>{delete m.system.attributes.hp.sp;delete m._source.system.attributes.hp.sp}});
  f.finish(undefined);assert.equal((await f.terminal).updateOutcome,'unchanged');
 });
 test(`${route} newly added stamina is not unchanged HP`,async()=>{
  const f=await pendingNoop({remote,updates:{'system.attributes.hp.value':30},before:m=>{delete m.system.attributes.hp.sp}});
  f.master.system.attributes.hp.sp={value:0,max:0};f.finish(undefined);await assert.rejects(f.terminal,/manual-pool-native-terminal-unavailable/);
 });
 test(`${route} a synchronous undefined return is not a Promise terminal`,async()=>{
  const f=await pendingNoop({remote,returnPromise:false});await assert.rejects(f.terminal,/manual-pool-native-promise-required/);assert.equal(f.calls(),1);
 });
 for(const [name,before,after] of [
  ['raw requested value differs',m=>{m._source.system.attributes.hp.value=29}],
  ['prepared requested value differs',m=>{m.system.attributes.hp.value=29}],
  ['raw HP absent',m=>{delete m._source.system.attributes.hp}],
  ['raw requested temp absent',m=>{delete m._source.system.attributes.hp.temp}],
  ['prepared requested temp absent',m=>{delete m.system.attributes.hp.temp}],
  ['raw stamina leaf differs',m=>{m._source.system.attributes.hp.sp.value=3}],
  ['prepared stamina leaf differs',m=>{m.system.attributes.hp.sp.value=3}],
  ['raw non-requested field changes',()=>{},f=>{f.master._source.system.attributes.hp.details='changed'}],
  ['prepared maximum changes',()=>{},f=>{f.master.system.attributes.hp.max=31}],
  ['prepared stamina maximum changes',()=>{},f=>{f.master.system.attributes.hp.sp.max=5}],
  ['prepared stamina disappears',()=>{},f=>{delete f.master.system.attributes.hp.sp}],
  ['raw stamina disappears',()=>{},f=>{delete f.master._source.system.attributes.hp.sp}],
  ['prepared missing property becomes undefined',m=>{delete m.system.attributes.hp.sp.max},f=>{f.master.system.attributes.hp.sp.max=undefined}],
  ['raw missing property becomes undefined',()=>{},f=>{f.master._source.system.attributes.hp.extra=undefined}],
  ['raw value changes',()=>{},f=>{f.master._source.system.attributes.hp.value=29}],
  ['prepared temp changes',()=>{},f=>{f.master.system.attributes.hp.temp=1}],
  ['authorization revoked',()=>{},f=>f.revoke()],
  ['ownership revoked',()=>{},f=>{f.master.isOwner=false}],
  ['writer replaced',()=>{},f=>{f.client.game.user={...f.client.game.user}}],
 ])test(`${route} undefined result rejects when ${name}`,async()=>{
  const f=await pendingNoop({remote,before});after?.(f);f.finish(undefined);
  await assert.rejects(f.terminal,/manual-pool-native-terminal-unavailable/);assert.equal(f.calls(),1);
 });
 for(const [name,result] of [['null',null],['false',false],['foreign document',{uuid:'Actor.Other'}]])test(`${route} ${name} is not unchanged success`,async()=>{
  const f=await pendingNoop({remote});f.finish(result);await assert.rejects(f.terminal,/manual-pool-native-terminal-unavailable/);assert.equal(f.calls(),1);
 });
 test(`${route} original rejection remains rejected`,async()=>{
  const f=await pendingNoop({remote});f.reject(Error('original-update-rejected'));await assert.rejects(f.terminal,/original-update-rejected/);assert.equal(f.calls(),1);
 });
}
