import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHpPools} from '../../scripts/exploration/hp-pool.mjs';
import {seamFixture} from '../../tools/toolbelt-manual-pool/seam.test.mjs';

const permit={permitNonce:'permit',applicationNonce:'application',ownerUserId:'O',selectedPatientUUID:'Actor.P',poolUUID:'Actor.M'};
function fixture(remote=false){
 const f=seamFixture({remote}),c=f.owner;let middleware,valid=true,master=c.master;
 c.game.actors.set(c.patient.id,c.patient);
 c.patient.modules={'pf2e-toolbelt':{shareData:{data:{health:true}}}};c.master.system={attributes:{hp:{value:1,max:30,temp:0}}};
 c.game.settings={get:()=>true};c.game.modules.get('pf2e-toolbelt').active=true;c.game.toolbelt={api:{shareData:{getMasterInMemory:()=>master,getSlavesInMemory:()=>[c.patient]}}};
 c.game.messages=new Map();c.patient._preUpdate=(changes,options)=>c.tool.pre(c.patient,changes,options);
 const pools=createHpPools({game:c.game,actorUpdateEvents:{addActorUpdateMiddleware:fn=>{middleware=fn}}});
 const receipt={id:'receipt',speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:['pf2e-third-party-automation:source:R:0']},appliedDamage:{uuid:'Actor.P',isHealing:true,isReverted:false}}}};c.game.messages.set(receipt.id,receipt);
 const options={permit,provider:c.api,validate:()=>valid,request:{resultId:'R',rollIndex:0}};
 const operation=async()=>{await middleware.call(c.patient,async(changes,opts)=>{await Promise.resolve();await c.patient._preUpdate(changes,opts);return c.patient},{'system.attributes.hp.value':20},{});return {receipt}};
 return {...f,pools,options,operation,receipt,invalid:()=>{valid=false},relink:()=>{master={...c.master,uuid:'Actor.Other'}}};
}

test('manual shared application awaits the original master half after the patient returns',async()=>{
 const f=fixture();let done=false;assert.equal(typeof f.pools.withManualApplication,'function');
 const pending=f.pools.withManualApplication(f.options,f.owner.patient,f.operation).then(value=>{done=true;return value});await f.turn();assert.equal(done,false);assert.equal(f.writes.length,1);
 f.finish();const result=await pending;assert.equal(result.poolReceipt.actorUUID,'Actor.M');assert.equal(result.poolReceipt.receiptId,'receipt');assert.equal(f.writes.length,1);
});

test('manual noChange requires the exact original saved receipt and no forward',async()=>{
 const f=fixture();f.receipt.flags.pf2e.appliedDamage=null;assert.equal(typeof f.pools.withManualApplication,'function');
 const result=await f.pools.withManualApplication(f.options,f.owner.patient,async()=>({receipt:f.receipt}));
 assert.equal(result.poolReceipt.noChange,true);assert.equal(f.writes.length,0);
});

for(const mutation of ['deleted receipt','reverted receipt','edited receipt','local master permission','relinked pool','stopped source'])test(`waiting for the original master cannot outlive ${mutation}`,async()=>{
 const f=fixture();const pending=f.pools.withManualApplication(f.options,f.owner.patient,f.operation);pending.catch(()=>{});await f.turn();assert.equal(f.writes.length,1);
 if(mutation==='deleted receipt')f.owner.game.messages.delete(f.receipt.id);
 if(mutation==='reverted receipt')f.receipt.flags.pf2e.appliedDamage.isReverted=true;
 if(mutation==='edited receipt')f.receipt.flags.pf2e.appliedDamage.total=99;
 if(mutation==='local master permission')f.owner.master.isOwner=false;
 if(mutation==='relinked pool')f.relink();
 if(mutation==='stopped source')f.invalid();
 f.finish();await assert.rejects(pending,/manual-pool/);assert.equal(f.writes.length,1);
});

test('the original rejected master Promise never becomes a receipt',async()=>{
 const f=fixture();const pending=f.pools.withManualApplication(f.options,f.owner.patient,f.operation);pending.catch(()=>{});await f.turn();f.reject(Error('native-master-rejected'));
 await assert.rejects(pending,/native-master-rejected/);assert.equal(f.writes.length,1);
});

test('a directly treated shared master retains its own original update Promise',async()=>{
 const f=fixture(),master=f.owner.master;master._preUpdate=async()=>{};
 const options={...f.options,permit:{...permit,selectedPatientUUID:master.uuid}};
 f.receipt.speaker.actor='M';f.receipt.flags.pf2e.appliedDamage.uuid=master.uuid;
 let middleware;f.pools.dispose();const pools=createHpPools({game:f.owner.game,actorUpdateEvents:{addActorUpdateMiddleware:fn=>{middleware=fn}}});
 const pending=pools.withManualApplication(options,master,async()=>{await middleware.call(master,fields=>master.update(fields),{'system.attributes.hp.value':20},{});return {receipt:f.receipt}});pending.catch(()=>{});
 await f.turn();assert.equal(f.writes.length,1);f.finish();const result=await pending;assert.equal(result.poolReceipt.master.writerUserId,'O');assert.equal(result.poolReceipt.actorUUID,master.uuid);
});
