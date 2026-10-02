import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {fixture} from './manual-pool-provider-fixture.mjs';

// Only the database boundary is replaced. Core's two original methods convert
// its empty updated-document list into the undefined Document.update result.
const coreFile=process.env.FOUNDRY_DOCUMENT_SOURCE??'C:/Program Files/Foundry Virtual Tabletop/resources/app/common/abstract/document.mjs';
const core=fs.readFileSync(coreFile,'utf8');
assert.equal(crypto.createHash('sha256').update(fs.readFileSync(coreFile)).digest('hex'),'303e6bc84fbaa7d564dab144b1f59fd0a1f6584a14e872b293e74358ed9b49dc');
function method(start){const at=core.indexOf(start);assert.ok(at>=0);const end=core.indexOf('\n  }',at);assert.ok(end>at);return core.slice(at,end+4)}
const CoreActor=Function(`return class {${method('static async updateDocuments(updates=[], operation={})')}\n${method('async update(data={}, operation={})')}}`)();

async function run(t,{remote=false,mutate,fields={'system.attributes.hp.value':20}}={}){
 const f=await fixture(remote);t.after(()=>f.close());const writer=remote?f.gm:f.owner;
 for(const client of [f.owner,f.gm]){
  client.master.system.attributes.hp={value:20,max:20,temp:0,sp:{value:4,max:4}};
  client.master._source={system:{attributes:{hp:{value:20,temp:0,sp:{value:4}}}}};
  client.patient.system={attributes:{hp:structuredClone(client.master.system.attributes.hp)}};
 }
 let calls=0,databaseCalls=0,returnedUndefined=false,observed,submitted,ledgerError;
 const master=writer.master,before=structuredClone(master._source);
 class Master extends CoreActor {}
 Master.implementation=Master;
 Master.database={async update(implementation,operation){
  databaseCalls++;assert.equal(implementation,Master);assert.equal(operation.updates.length,1);
  assert.deepEqual(structuredClone(operation.updates[0]),{...fields,_id:'M'});
  return [];
 }};
 master.constructor=Master;
 master.update=function(...args){calls++;return CoreActor.prototype.update.apply(this,args).then(result=>{returnedUndefined=result===undefined;return result})};
 writer.api.subscribe(event=>{if(event.phase==='write')event.terminalPromise.then(proof=>{observed=proof},()=>{})});
 const record=f.ledger.recordManualPoolTerminal.bind(f.ledger);
 f.ledger.recordManualPoolTerminal=async(input,options)=>{submitted=structuredClone(input);try{return await record(mutate?mutate(structuredClone(input)):input,options)}catch(error){ledgerError=error;throw error}};
 const grant=await f.owner.broker.claim(f.request);
 const operation=async()=>{
  await f.owner.update.call(f.owner.patient,async(changes,options)=>{await f.owner.patient._preUpdate(changes,options);return f.owner.patient},structuredClone(fields),{});
  f.messages.set(f.receipt.id,f.receipt);return {receipt:f.receipt};
 };
 let result,error;try{result=await f.owner.broker.withApplication(grant,f.owner.patient,operation)}catch(e){error=e}
 assert.equal(calls,1);assert.equal(databaseCalls,1);assert.equal(returnedUndefined,true);assert.deepEqual(master._source,before);
 assert.equal(observed.terminal,'fulfilled');assert.equal(observed.updateOutcome,'unchanged');assert.equal(observed.writerUserId,remote?'G':'O');
 assert.equal(submitted.noChange,false);assert.ok(f.receipt.flags.pf2e.appliedDamage);assert.equal(submitted.receiptId,f.receipt.id);
 const claim=Object.values((await f.ledger.getActivity('A')).proof.poolApplications)[0];
 return {f,grant,operation,result,error,ledgerError,claim,submitted,observed,calls:()=>calls};
}

for(const remote of [false,true])test(`original ${remote?'GM forward':'local OWNER'} no-op terminal reaches the atomic ledger with its non-null receipt`,async t=>{
 const f=await run(t,{remote});assert.ifError(f.ledgerError);assert.ifError(f.error);
 assert.equal(f.claim.state,'settled');assert.equal(f.claim.terminal.noChange,false);
 assert.equal(f.claim.terminal.master.updateOutcome,'unchanged');assert.deepEqual(f.claim.terminal.master,f.observed);
 assert.equal(f.result.poolReceipt.noChange,false);assert.equal(f.result.poolReceipt.master.updateOutcome,'unchanged');
 const lookup=await f.f.owner.broker.lookup(f.f.request);assert.equal(lookup.status,'settled');assert.equal(lookup.permitNonce,undefined);
 await assert.rejects(f.f.owner.broker.withApplication(f.grant,f.f.owner.patient,f.operation),/private-grant/);assert.equal(f.calls(),1);
});

test('the unchanged terminal validates all three native HP paths including stamina',async t=>{
 const f=await run(t,{fields:{'system.attributes.hp.value':20,'system.attributes.hp.temp':0,'system.attributes.hp.sp.value':4}});
 assert.ifError(f.ledgerError);assert.ifError(f.error);assert.equal(f.claim.state,'settled');assert.equal(f.claim.terminal.master.before.sp.value,4);
});

for(const [name,change] of [
 ['unknown field',m=>{m.mark=true}],['wrong outcome',m=>{m.updateOutcome='written'}],['null outcome',m=>{m.updateOutcome=null}],
 ['missing original fields',m=>{delete m.fields}],['missing original before',m=>{delete m.before}],
 ['foreign binding',m=>{m.binding.applicationNonce='foreign'}],['unknown binding field',m=>{m.binding.extra=true}],
 ['unequal prior value',m=>{m.before.value=19}],['missing prior value',m=>{delete m.before.value}],
 ['missing prior temp',m=>{m.fields={'system.attributes.hp.temp':0};delete m.before.temp}],
 ['unequal prior stamina',m=>{m.fields={'system.attributes.hp.sp.value':4};m.before.sp.value=3}],
 ['missing prior stamina leaf',m=>{m.fields={'system.attributes.hp.sp.value':4};delete m.before.sp.value}],
])test(`ledger rejects a captured terminal altered to ${name}`,async t=>{
 const f=await run(t,{mutate:input=>{change(input.master);return input}});
 assert.match(f.ledgerError?.message??'',/invalid-manual-pool-terminal/);assert.match(f.error?.message??'',/manual-pool-application-unavailable/);assert.equal(f.claim.state,'applying');assert.equal(f.claim.terminal,undefined);
 assert.equal((await f.f.owner.broker.lookup(f.f.request)).status,'reserved');assert.equal(f.calls(),1);
});
