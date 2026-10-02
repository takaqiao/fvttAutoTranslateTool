import test from 'node:test';
import assert from 'node:assert/strict';
import {createMetapowerLedger,MODULE_ID as ID} from '../scripts/metapower/lifecycle.mjs';
import {POWER_PROFILES,METAPOWER_SOURCES} from '../scripts/metapower/rules.mjs';
const chargedSource='Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR';

function fixture({badge=2,synthetic=false}={}){
 const gm={id:'gm'},owner={id:'owner'},nextGM={id:'next-gm'},users=new Map([[gm.id,gm],[owner.id,owner],[nextGM.id,nextGM]]);users.activeGM=gm;
 const actor={id:'a',uuid:synthetic?'Scene.s.Token.t.Actor.a':'Actor.a',isToken:synthetic,type:'character',level:1,flags:{},items:new Map(),getRollOptions:()=>['active-power-one:electric-surge']};
 let owned=true;actor.testUserPermission=user=>owned&&(user===owner||user===gm);
 const token={id:'t',uuid:'Scene.s.Token.t',parent:{id:'s'},actor},scene={id:'s',tokens:new Map([['t',token]])};if(synthetic)actor.token=token;
 const game={user:gm,users,actors:new Map(synthetic?[]:[[actor.id,actor]]),scenes:new Map([['s',scene]]),messages:new Map(),combat:null};
 const power={id:'p',uuid:actor.uuid+'.Item.p',actor,type:'feat',sourceId:Object.values(POWER_PROFILES).find(p=>p.id==='electric-surge').sourceUuid,system:{traits:{value:['electricity']},frequency:{value:1}}};
 const charge={id:'q',uuid:actor.uuid+'.Item.q',actor,type:'effect',sourceId:chargedSource,system:{badge:{value:badge}},flags:{}};
 actor.items.set(power.id,power);actor.items.set(charge.id,charge);
 const documents=new Map([[actor.uuid,actor],[power.uuid,power],[charge.uuid,charge]]),hooks=new Map(),writes=[];
 const boundary=async key=>{const fn=hooks.get(key);if(fn){hooks.delete(key);await fn();}};
 actor.update=async changes=>{const state=structuredClone(changes[`flags.${ID}.metapower`]);actor.flags[ID]={metapower:state};writes.push({kind:'actor',paymentStarted:state.receipts.use?.paymentStarted===true});if(state.receipts.use?.paymentStarted)await boundary('intent');if(state.receipts.use?.status==='committed')await boundary('commit');return actor;};
 charge.update=async changes=>{writes.push({kind:'charge',activeGM:game.users.activeGM.id});charge.system.badge.value=changes['system.badge.value'];charge.flags[ID]={payment:changes[`flags.${ID}.payment`]};await boundary('payment');return charge;};
 const ledger=createMetapowerLedger({game,fromUuid:async uuid=>{const original=documents.get(uuid);await boundary('resolve:'+uuid);return original;},validateSelection:async()=>boundary('selection')});
 const message={id:'m',uuid:'ChatMessage.m',speaker:{actor:actor.id},author:owner,flags:{pf2e:{origin:{uuid:power.uuid}},[ID]:{metapowerUse:{nonce:'use',actorUuid:actor.uuid,itemUuid:power.uuid}}}};documents.set(message.uuid,message);game.messages.set(message.id,message);
 const input={actorUuid:actor.uuid,itemUuid:power.uuid,nonce:'use',startNative:true,selection:{discharge:true,baseDistance:40}};
 const finishInput={actorUuid:actor.uuid,nonce:'use',messageUuid:message.uuid,status:'committed'};
 return {gm,owner,nextGM,game,actor,power,charge,message,documents,hooks,writes,ledger,input,finishInput,setOwned:value=>owned=value,payments:()=>writes.filter(w=>w.kind==='charge').length};
}
async function admitted(options){const f=fixture(options);await f.ledger.begin(f.input,f.owner);return f;}
test('unchanged world authority pays once and completed retry does not charge again',async()=>{
 const f=await admitted();const result=await f.ledger.finish(f.finishInput,f.owner);assert.equal(result.status,'committed');assert.equal(f.charge.system.badge.value,1);assert.equal(f.payments(),1);await f.ledger.finish(f.finishInput,f.owner);assert.equal(f.payments(),1);
});
test('current native synthetic actor pays through its exact token actor binding',async()=>{
 const f=await admitted({synthetic:true});await f.ledger.finish(f.finishInput,f.owner);assert.equal(f.payments(),1);assert.equal(f.charge.system.badge.value,1);
});
const changes=[
 ['active GM',f=>f.game.users.activeGM=f.nextGM],
 ['current GM object',f=>f.game.users.set(f.gm.id,{...f.gm})],
 ['OWNER permission',f=>f.setOwned(false)],
 ['original OWNER object',f=>f.game.users.set(f.owner.id,{...f.owner})],
 ['current Actor',f=>f.game.actors.set(f.actor.id,{...f.actor})],
 ['current charge Item',f=>f.actor.items.set(f.charge.id,{...f.charge})],
 ['charge source',f=>f.charge.sourceId='custom'],
 ['charge badge',f=>f.charge.system.badge.value=8],
 ['charge payment proof',f=>f.charge.flags[ID]={payment:{nonce:'foreign',after:1}}],
 ['current power Item',f=>f.actor.items.set(f.power.id,{...f.power})],
 ['power source',f=>f.power.sourceId='custom'],
 ['current original card',f=>f.game.messages.set(f.message.id,{...f.message})],
 ['original card source',f=>f.message.flags.pf2e.origin.uuid='custom'],
 ['current pending lease',f=>f.actor.flags[ID].metapower.pending='foreign'],
 ['current receipt',f=>f.actor.flags[ID].metapower.receipts.use.selection.discharge=false]
];
for(const [label,change]of changes)test(`payment intent await loses ${label}: no counter write and no automatic release`,async()=>{
 const f=await admitted();f.hooks.set('intent',()=>change(f));await assert.rejects(f.ledger.finish(f.finishInput,f.owner));assert.equal(f.payments(),0);assert.equal(f.actor.flags[ID].metapower.receipts.use.paymentStarted,true);assert.equal(f.actor.flags[ID].metapower.receipts.use.status,'started');if(label!=='current pending lease')assert.equal(f.actor.flags[ID].metapower.pending,'use');
});
for(const phase of ['actor','card','charge'])test(`${phase} lookup await changes GM before payment: no intent or charge write`,async()=>{
 const f=await admitted(),uuid=phase==='actor'?f.actor.uuid:phase==='card'?f.message.uuid:f.charge.uuid;f.hooks.set('resolve:'+uuid,()=>f.game.users.activeGM=f.nextGM);
 await assert.rejects(f.ledger.finish(f.finishInput,f.owner));assert.equal(f.payments(),0);assert.equal(f.actor.flags[ID].metapower.receipts.use.paymentStarted,undefined);assert.equal(f.actor.flags[ID].metapower.pending,'use');
});
test('selection await replaces the original power before begin: no durable native admission',async()=>{
 const f=fixture();f.hooks.set('selection',()=>f.actor.items.set(f.power.id,{...f.power}));await assert.rejects(f.ledger.begin(f.input,f.owner));assert.equal(f.actor.flags[ID],undefined);assert.equal(f.payments(),0);
});
test('item lookup await loses OWNER before begin: no durable native admission',async()=>{
 const f=fixture();f.hooks.set('resolve:'+f.power.uuid,()=>f.setOwned(false));await assert.rejects(f.ledger.begin(f.input,f.owner));assert.equal(f.actor.flags[ID],undefined);
});
test('synthetic token replacement during payment intent cannot pay',async()=>{
 const f=await admitted({synthetic:true});f.hooks.set('intent',()=>f.game.scenes.get('s').tokens.set('t',{...f.actor.token}));await assert.rejects(f.ledger.finish(f.finishInput,f.owner));assert.equal(f.payments(),0);assert.equal(f.actor.flags[ID].metapower.pending,'use');
});
test('normal zero-counter native deletion completes without requiring the removed effect',async()=>{
 const f=await admitted({badge:1});f.hooks.set('payment',()=>{f.actor.items.delete(f.charge.id);f.documents.delete(f.charge.uuid);});const result=await f.ledger.finish(f.finishInput,f.owner);assert.equal(result.paymentPaid,true);assert.equal(result.status,'committed');assert.equal(f.payments(),1);await f.ledger.finish(f.finishInput,f.owner);assert.equal(f.payments(),1);
});
test('lost zero-counter deletion response retains payment intent and cannot pay twice',async()=>{
 const f=await admitted({badge:1});f.hooks.set('payment',()=>{f.actor.items.delete(f.charge.id);f.documents.delete(f.charge.uuid);throw Error('lost native deletion response');});await assert.rejects(f.ledger.finish(f.finishInput,f.owner),/lost/);assert.equal(f.actor.flags[ID].metapower.receipts.use.paymentStarted,true);await assert.rejects(f.ledger.finish(f.finishInput,f.owner),/不确定|核对/);assert.equal(f.payments(),1);assert.equal(f.actor.flags[ID].metapower.pending,'use');
});
test('GM handover after intent preserves an unknown payment and blocks repeated payment',async()=>{
 const f=await admitted();f.hooks.set('intent',()=>f.game.users.activeGM=f.nextGM);await assert.rejects(f.ledger.finish(f.finishInput,f.owner));f.game.users.activeGM=f.gm;await assert.rejects(f.ledger.finish(f.finishInput,f.owner),/不确定|核对/);assert.equal(f.payments(),0);assert.equal(f.charge.system.badge.value,2);assert.equal(f.actor.flags[ID].metapower.pending,'use');
});
test('original atomic same-nonce payment can finish without charging twice',async()=>{
 const f=await admitted();f.charge.system.badge.value=1;f.charge.flags[ID]={payment:{nonce:'use',after:1}};const result=await f.ledger.finish(f.finishInput,f.owner);assert.equal(result.paymentPaid,true);assert.equal(result.status,'committed');assert.equal(f.payments(),0);
});
test('same-nonce proof with the wrong paid amount cannot mark payment complete',async()=>{
 const f=await admitted();f.charge.system.badge.value=1;f.charge.flags[ID]={payment:{nonce:'use',after:99}};await assert.rejects(f.ledger.finish(f.finishInput,f.owner));assert.equal(f.actor.flags[ID].metapower.receipts.use.status,'started');assert.equal(f.payments(),0);
});
test('readback of an existing same-nonce payment loses its charge binding during commit: no false response or second payment',async()=>{
 const f=await admitted();f.charge.system.badge.value=1;f.charge.flags[ID]={payment:{nonce:'use',after:1}};f.hooks.set('commit',()=>f.actor.items.set(f.charge.id,{...f.charge}));await assert.rejects(f.ledger.finish(f.finishInput,f.owner));assert.equal(f.payments(),0);
});
for(const [label,change]of [
 ['payment proof',f=>delete f.charge.flags[ID]],
 ['paid counter',f=>f.charge.system.badge.value=8],
 ['nonzero charge deletion',f=>f.actor.items.delete(f.charge.id)]
])test(`charge update returns after losing ${label}: keep intent without false commit`,async()=>{
 const f=await admitted();f.hooks.set('payment',()=>change(f));await assert.rejects(f.ledger.finish(f.finishInput,f.owner));assert.equal(f.payments(),1);assert.equal(f.actor.flags[ID].metapower.receipts.use.paymentStarted,true);assert.equal(f.actor.flags[ID].metapower.receipts.use.status,'started');assert.equal(f.actor.flags[ID].metapower.pending,'use');
});
test('native zero-counter deletion may remove GrantItem children without rejecting the original payment',async()=>{
 const f=await admitted({badge:1}),child={id:'child',uuid:f.actor.uuid+'.Item.child',actor:f.actor,flags:{pf2e:{grantedBy:{id:f.charge.id}}}};f.actor.items.set(child.id,child);f.documents.set(child.uuid,child);
 f.hooks.set('payment',()=>{for(const item of [f.charge,child]){f.actor.items.delete(item.id);f.documents.delete(item.uuid);}});const result=await f.ledger.finish(f.finishInput,f.owner);assert.equal(result.status,'committed');assert.equal(result.paymentPaid,true);assert.equal(f.payments(),1);assert.equal(f.actor.items.has(child.id),false);
});
