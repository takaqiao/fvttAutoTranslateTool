import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createExplorationOwnerOperations} from '../../scripts/exploration/owner-operations.mjs';
import {createNativeTreatment} from '../../scripts/exploration/native-treatment.mjs';
import {createSalubriousDamageGuard} from '../../scripts/salubrious-kiss-damage-guard.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';

async function fixture(owner){
 const f=await authorityFixture(),ledger=f.client('driver'),gm={id:'G',active:true,isGM:true},player={id:'O',active:true,isGM:false},users=new Map([['G',gm],['O',player]]);users.activeGM=gm;
 const documents=new Map(),messages=new Map(),tabs=[],listeners=new Map();let hookId=0;
 for(const id of ['H','P']){
  const actor={id,uuid:`Actor.${id}`,flags:{},getSelfRollOptions:()=>[],getRollOptions:()=>[],testUserPermission:user=>user?.isGM||user===player,getActiveTokens:()=>[{document:{uuid:`Scene.S.Token.${id}`}}],getContextualClone(){return this},update:async data=>{for(const [key,value]of Object.entries(data)){let at=actor;const keys=key.split('.');for(const part of keys.slice(0,-1))at=at[part]??={};at[keys.at(-1)]=structuredClone(value)}return actor}};documents.set(actor.uuid,actor);
 }
 const session=await ledger.createSession({id:'S',actorUUIDs:[...documents.keys()],status:'running',startedAt:0,cursorAt:0,budgetEndsAt:1200,nativeOwnerByActor:{'Actor.H':owner}}),scope={leaseNonce:session.driver.leaseNonce};
 await ledger.insertActivity({id:'A',sessionId:'S',providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],state:'planned',startedAt:0,endsAt:600,options:{},source:{ownerId:owner}},scope);
 await ledger.transitionActivity('A',{...scope,expected:['planned'],patch:{state:'started'}});const activity=await ledger.transitionActivity('A',{...scope,expected:['started'],patch:{state:'completing'}});
 const Hooks={on:(event,fn)=>{const id=++hookId;listeners.set(id,{event,fn});return id},off:(_event,id)=>listeners.delete(id)},emit=(event,...args)=>{for(const row of listeners.values())if(row.event===event)row.fn(...args)};
 function client(userId,clientNonce){
  const callbacks=new Set(),game={user:users.get(userId),users,actors:new Map([...documents.values()].map(actor=>[actor.id,actor])),messages,time:{worldTime:600},socket:{on:(_event,fn)=>callbacks.add(fn),off:(_event,fn)=>callbacks.delete(fn),emit:(_event,packet,routing,ack)=>{for(const tab of tabs.filter(t=>routing.recipients.includes(t.game.user.id)))for(const fn of tab.callbacks)queueMicrotask(()=>fn(structuredClone(packet),userId));ack?.()}}};
  const ops=createExplorationOwnerOperations({game,fromUuid:async uuid=>documents.get(uuid),ledger:userId==='G'?ledger:{atomic:true},runtimeIdentity:()=>({userId,clientNonce}),getDriverScope:()=>userId==='G'?scope:null,timeoutMs:200});const tab={game,ops,callbacks};tabs.push(tab);ops.register();return tab;
 }
 const driver=client('G','driver'),receiver=owner==='G'?driver:client('O','owner');
 return {ledger,driver,receiver,activity,documents,messages,Hooks,emit,dispose:()=>tabs.forEach(tab=>tab.ops.dispose())};
}

for(const owner of ['G','O'])test(`the real treatment HP guard consumes an atomic ${owner==='G'?'GM':'player OWNER'} context once`,async()=>{
 const f=await fixture(owner);try{
  const {game,ops}=f.receiver,patient=f.documents.get('Actor.P'),guard=createSalubriousDamageGuard({game,isExplorationContext:ops.isActivityContext});let calls=0,usedParams;
  patient.applyDamage=params=>guard.applyDamage(patient,params,async(current,assertNative)=>{assertNative(patient,current);usedParams=current;calls++;const receipt={id:'HP',author:game.user,speaker:{actor:patient.id},flags:{pf2e:{context:{type:'damage-taken',options:[...current.rollOptions]},appliedDamage:{uuid:patient.uuid,isReverted:false}}}};f.messages.set(receipt.id,receipt);f.emit('createChatMessage',receipt);return patient});
  const card={id:'R',author:game.user,rolls:[{_evaluated:true,total:8,toJSON:()=>({evaluated:true,total:8,formula:'{2d8[healing]}'})}],flags:{pf2e:{context:{options:['exploration-activity:A']}}}};f.messages.set(card.id,card);
  const native=createNativeTreatment({game,Hooks:f.Hooks,fromUuid:async uuid=>f.documents.get(uuid),ownerOperations:ops,damageGuard:guard,hpPools:{withNativeApplication:async(_activity,_patient,operation)=>({result:await operation()})}});
  ops.registerOperation('treat-wounds',async(activity,ctx)=>{await native.applySavedResult(activity,{message:card,patient,stage:'healing',outcome:'success'},ctx);return {status:'blocked',reason:'guard-test-completed'}});
  assert.equal((await f.driver.ops.runActivityWithOwner(f.activity,'treat-wounds')).status,'blocked');assert.equal(calls,1);
  await assert.rejects(guard.applyDamage(patient,usedParams,()=>assert.fail('a spent guard must not reach native HP')),/本次内部结算/);assert.equal(calls,1);
 }finally{f.dispose()}
});

test('player HP permission is revalidated at the final native boundary after a provider awaits',async()=>{
 const f=await fixture('O');try{
  const {game,ops}=f.receiver,patient=f.documents.get('Actor.P'),guard=createSalubriousDamageGuard({game,isExplorationContext:ops.isActivityContext});let calls=0,reason;
  patient.applyDamage=params=>guard.applyDamage(patient,params,async(current,assertNative)=>{await Promise.resolve();patient.testUserPermission=user=>user?.isGM;assertNative(patient,current);calls++});
  const card={id:'R',author:game.user,rolls:[{_evaluated:true,total:8,toJSON:()=>({evaluated:true,total:8,formula:'{2d8[healing]}'})}],flags:{pf2e:{context:{options:['exploration-activity:A']}}}};f.messages.set(card.id,card);
  const native=createNativeTreatment({game,Hooks:f.Hooks,fromUuid:async uuid=>f.documents.get(uuid),ownerOperations:ops,damageGuard:guard,hpPools:{withNativeApplication:async(_activity,_patient,operation)=>({result:await operation()})}});
  ops.registerOperation('treat-wounds',async(activity,ctx)=>{try{await native.applySavedResult(activity,{message:card,patient,stage:'healing',outcome:'success'},ctx)}catch(error){reason=error.message;throw error}});
  await assert.rejects(f.driver.ops.runActivityWithOwner(f.activity,'treat-wounds'));assert.equal(calls,0);assert.match(reason,/owner|上下文|拥有/);
 }finally{f.dispose()}
});
