import test from 'node:test';
import assert from 'node:assert/strict';
import {MODULE_ID as ID} from '../scripts/rules.mjs';
import {createRoaringSource,reduceRoaringSource} from '../scripts/roaring-lifecycle.mjs';
import {createRoaringEffects} from '../scripts/roaring-effects.mjs';

function fixture({outcome='criticalFailure',immunity={},veto=false,lostReply=false,deleteVeto=false,deleteLostReply=false,synthetic=false}={}){
 const gm={id:'gm',isGM:true,active:true},game={user:gm,users:{activeGM:gm},time:{worldTime:10},actors:new Map(),scenes:new Map()},actor={id:'target',uuid:synthetic?'Scene.scene.Token.target.Actor.target':'Actor.target',type:'npc',flags:{},items:new Map()};
 if(synthetic){const scene={id:'scene',tokens:new Map()},token={id:'target',uuid:'Scene.scene.Token.target',parent:scene,actor};scene.tokens.set(token.id,token);game.scenes.set(scene.id,scene);actor.token=token;}else game.actors.set(actor.id,actor);
 const docs=new Map([[actor.uuid,actor]]),calls={create:0,delete:[],update:0};let sequence=0;
 const merge=(target,changes)=>{for(const[k,v]of Object.entries(changes)){const keys=k.split('.');let ptr=target;for(const name of keys.slice(0,-1))ptr=ptr[name]??={};const name=keys.at(-1);if(v&&typeof v==='object'&&!Array.isArray(v))merge(ptr[name]??={},v);else ptr[name]=structuredClone(v);}};
 actor.update=async changes=>{calls.update++;merge(actor,changes);return actor};
 const make=(data,id)=>{const doc={...structuredClone(data),id,uuid:`${actor.uuid}.Item.${id}`,actor,sourceId:data._stats?.compendiumSource};doc.update=async changes=>{merge(doc,changes);return doc};actor.items.set(id,doc);docs.set(doc.uuid,doc);return doc};
 actor.createEmbeddedDocuments=async(_type,data)=>{
  calls.create++;if(veto)return [];
  const result=[];
  for(const source of data){const parent=make(source,`parent${++sequence}`);parent._source=structuredClone(source);parent.flags.pf2e??={};parent.flags.pf2e.itemGrants={};result.push(parent);
   // PF2e 8.5.1 ItemAlteration's EmbeddedDataField adds this default to the
   // prepared rule source; the persisted Item source remains unchanged.
   for(const rule of parent.system.rules)for(const alteration of rule.alterations??[])if(alteration.fromEquipment===undefined)alteration.fromEquipment=true;
   for(const rule of parent.system.rules){const name=rule.uuid.endsWith('xYTAsEpcJE1Ccni3')?'slowed':'fascinated',id=`child${++sequence}`;
    const child=make({type:'condition',system:{slug:name},flags:{pf2e:{grantedBy:{id:parent.id,onDelete:'cascade'}}},_stats:{compendiumSource:rule.uuid}},id);
    parent.flags.pf2e.itemGrants[rule.flag]={id:child.id,onDelete:'detach'};result.push(child);
   }
  }
  if(lostReply)throw Error('created; reply lost');return result;
 };
 actor.deleteEmbeddedDocuments=async(_type,ids)=>{
  calls.delete.push([...ids]);if(deleteVeto)return [];
  const removed=[];
  for(const id of ids){const doc=actor.items.get(id);if(!doc)continue;
   for(const grant of Object.values(doc.flags?.pf2e?.itemGrants??{})){const child=actor.items.get(grant.id);if(child?.flags?.pf2e?.grantedBy?.id===id&&child.flags.pf2e.grantedBy.onDelete==='cascade'){actor.items.delete(child.id);docs.delete(child.uuid);removed.push(child);}}
   // PF2e processGrantDeletions keeps the parent's stale itemGrants receipt.
   actor.items.delete(id);docs.delete(doc.uuid);removed.push(doc);
  }
  if(deleteLostReply)throw Error('deleted; reply lost');return removed;
 };
 const turn={combatId:'combat',combatantId:'caster',actorUuid:'Actor.caster',tokenUuid:'Scene.scene.Token.caster',started:true,round:4,turn:0,lastTurnEnd:3,order:[{id:'caster',initiative:20,overridePriority:null}]};
 let state=createRoaringSource({sourceNonce:'source-one',castNonce:'cast-one',sourceId:'Compendium.pf2e.spells-srd.Item.czO0wbT1i320gcu9',itemUuid:'Actor.caster.Item.spell',entryUuid:'Actor.caster.Item.entry',rank:3,casterActorUuid:'Actor.caster',casterTokenUuid:'Scene.scene.Token.caster',targetActorUuid:actor.uuid,targetTokenUuid:'Scene.scene.Token.target',originalMessageUuid:'ChatMessage.original',completedWorldTime:10,turn,finiteEnvelope:{start:{value:10,initiative:20},duration:{value:1,unit:'rounds',expiry:'turn-end',sustained:false}}});
 state=reduceRoaringSource(state,{type:'save-confirmed',sourceNonce:state.sourceNonce,revision:1,receiptId:'save-one',outcome,observation:{worldTime:10,turn}}).source;
 const context={userId:'owner',gmId:'gm',dc:21,paymentId:'cast-one',immunity:{checked:true,spell:false,slowed:false,fascinated:false,systemVersion:'8.5.1',...immunity}};
 const effects=createRoaringEffects({game,fromUuid:async uuid=>docs.get(uuid),randomId:()=>`op-${++sequence}`});
 return {game,gm,actor,docs,calls,make,state,turn,context,effects};
}
 for(const outcome of ['failure','criticalFailure'])test(`native alteration preparation preserves exact declared rules for ${outcome}`,async()=>{
  const f=fixture({outcome});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});
  const r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  const parent=f.actor.items.get(r.effects.parentId),rule=r.effects.rules.find(rule=>rule.alterations);
  assert.equal(r.effects.status,'created');assert.equal(f.calls.create,1);
  assert.deepEqual(rule.alterations,[{mode:'override',property:'badge-value',value:1,fromEquipment:true}]);
  assert.deepEqual(parent.system.rules,parent._source.system.rules);
  assert.deepEqual(parent.system.rules,r.effects.rules);
  assert.ok(r.effects.children.slowed);assert.equal(!!r.effects.children.fascinated,outcome==='criticalFailure');
 });
 for(const [name,alter]of [
  ['fromEquipment false',a=>a.fromEquipment=false],
  ['changed badge value',a=>a.value=2],
  ['changed mode',a=>a.mode='add'],
  ['changed property',a=>a.property='description'],
  ['extra alteration property',a=>a.unrecognized=true],
 ])test(`exact alteration comparison still rejects ${name}`,async()=>{
  const f=fixture({outcome:'failure',lostReply:true});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});
  await assert.rejects(f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce}));
  const parent=[...f.actor.items.values()].find(item=>item.type==='effect');alter(parent.system.rules[0].alterations[0]);
  await assert.rejects(f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce}),/来源效果授予规则已改变/);
  assert.equal(f.calls.create,1);assert.equal(f.calls.delete.length,0);
  assert.equal(f.effects.get(f.actor,f.state.sourceNonce).effects.status,'uncertain');
 });
 test('native parent owns only its two grants and repeating materialize never creates twice',async()=>{
  const f=fixture();const other=f.make({type:'condition',system:{slug:'slowed',value:{value:2}},flags:{}},'other');
  await f.effects.claim({actor:f.actor,state:f.state,context:f.context});
  const first=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  assert.equal(first.effects.status,'created');assert.equal(f.calls.create,1);
  assert.deepEqual(Object.keys(first.effects.children).sort(),['fascinated','slowed']);
  for(const role of ['slowed','fascinated'])assert.ok(first.effects.children[role]?.id);
  await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});assert.equal(f.calls.create,1);
  assert.equal(f.actor.items.get(other.id),other);
  const parent=f.actor.items.get(first.effects.parentId);assert.equal(parent.system.context.origin.actor,'Actor.caster');
  for(const rule of parent.system.rules){assert.equal(rule.allowDuplicate,true);assert.equal(rule.reevaluateOnUpdate,false);assert.deepEqual(rule.onDeleteActions,{granter:'cascade',grantee:'detach'});}
 });
 for(const synthetic of [false,true])test(`reaction parent inspection is synchronous and read-only for non-GM ${synthetic?'synthetic':'world'} actor`,async()=>{
  const f=fixture({synthetic});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});const r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  f.game.user={id:'player',isGM:false};f.docs.get=()=>{throw Error('asynchronous resolver must not be used')};
  const snapshot=()=>JSON.stringify({flags:f.actor.flags,items:[...f.actor.items.values()].map(i=>({id:i.id,system:i.system,flags:i.flags,raw:i._source})),calls:f.calls});
  const before=snapshot();
  for(let i=0;i<3;i++){const result=f.effects.inspectReactionParent({actor:f.actor,nonce:r.state.sourceNonce});assert.equal(result?.then,undefined);assert.deepEqual(result,{status:'present',reason:'parent-confirmed'});}
  assert.equal(snapshot(),before);
 });
 test('reaction parent inspection does not require removed children or inspect unrelated Slowed 2',async()=>{
  const f=fixture();await f.effects.claim({actor:f.actor,state:f.state,context:f.context});const r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  for(const child of Object.values(r.effects.children))f.actor.items.delete(child.id);
  f.make({type:'condition',system:{slug:'slowed',value:{value:2}},flags:{}},'independent');
  const counts=structuredClone(f.calls);assert.equal(f.effects.inspectReactionParent({actor:f.actor,nonce:f.state.sourceNonce}).status,'present');assert.deepEqual(f.calls,counts);
 });
 test('reaction parent inspection recognizes only an already confirmed parent removal',async()=>{
  const f=fixture();await f.effects.claim({actor:f.actor,state:f.state,context:f.context});const r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});f.actor.items.delete(r.effects.parentId);
  const counts=structuredClone(f.calls);assert.deepEqual(f.effects.inspectReactionParent({actor:f.actor,nonce:f.state.sourceNonce}),{status:'removed',reason:'parent-removed'});assert.deepEqual(f.calls,counts);
 });
 for(const status of ['not-started','creating','uncertain','deleting','ended','immune'])test(`reaction parent inspection does not infer a confirmed parent from phase ${status}`,async()=>{
  const f=fixture();await f.effects.claim({actor:f.actor,state:f.state,context:f.context});const r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  f.actor.flags[ID].roaringApplause.sources[r.state.sourceNonce].effects.status=status;
  assert.equal(f.effects.inspectReactionParent({actor:f.actor,nonce:f.state.sourceNonce}).status,'unproven');f.actor.items.delete(r.effects.parentId);
  assert.equal(f.effects.inspectReactionParent({actor:f.actor,nonce:f.state.sourceNonce}).status,'unproven');
 });
 for(const [name,mutate]of [
  ['missing operation',(f,r,p)=>delete f.actor.flags[ID].roaringApplause.sources[r.state.sourceNonce].effects.operationId],
  ['missing parent id',(f,r,p)=>delete f.actor.flags[ID].roaringApplause.sources[r.state.sourceNonce].effects.parentId],
  ['missing stored rules',(f,r,p)=>delete f.actor.flags[ID].roaringApplause.sources[r.state.sourceNonce].effects.rules],
  ['foreign target',(f,r,p)=>f.actor.flags[ID].roaringApplause.sources[r.state.sourceNonce].state.source.targetActorUuid='Actor.foreign'],
  ['parent reparented',(f,r,p)=>p.actor={uuid:'Actor.foreign'}],
  ['copied operation',(f,r,p)=>p.flags[ID].roaringEffect.operationId='other-operation'],
  ['foreign origin',(f,r,p)=>p.system.context.origin.actor='Actor.foreign'],
  ['changed prepared rule',(f,r,p)=>p.system.rules[0].alterations[0].value=2],
  ['prepared ignored even with valid finite cleanup proof',(f,r,p)=>{f.actor.flags[ID].roaringApplause.sources[r.state.sourceNonce].state.timing.mode='manual-finite';p.system.duration=structuredClone(r.state.timing.finiteEnvelope.duration);p.system.start=structuredClone(r.state.timing.finiteEnvelope.start);p.system.rules[0].ignored=true;p.isExpired=true;}],
 ])test(`reaction parent inspection rejects ${name} without side effects`,async()=>{
  const f=fixture();await f.effects.claim({actor:f.actor,state:f.state,context:f.context});const r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce}),parent=f.actor.items.get(r.effects.parentId);mutate(f,r,parent);
  const counts=structuredClone(f.calls);assert.equal(f.effects.inspectReactionParent({actor:f.actor,nonce:f.state.sourceNonce}).status,'unproven');assert.deepEqual(f.calls,counts);
 });
 test('reaction parent inspection rejects a replaced world actor or detached synthetic token',async()=>{
  for(const synthetic of [false,true]){const f=fixture({synthetic});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
   if(synthetic)f.actor.token.parent.tokens.delete(f.actor.token.id);else f.game.actors.set(f.actor.id,{id:f.actor.id,uuid:f.actor.uuid});
   assert.deepEqual(f.effects.inspectReactionParent({actor:f.actor,nonce:f.state.sourceNonce}),{status:'unproven',reason:'actor-not-live'});
  }
 });
 test('reaction parent inspection treats missing or unsafe sources as unproven',()=>{
  const f=fixture();for(const nonce of ['absent','constructor','unsafe.path',null])assert.equal(f.effects.inspectReactionParent({actor:f.actor,nonce}).status,'unproven');
  assert.deepEqual(f.calls,{create:0,delete:[],update:0});
 });
 for(const outcome of ['criticalSuccess','success','failure','criticalFailure'])test(`only exact own conditions are projected for ${outcome}`,async()=>{
  const f=fixture({outcome});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});const r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  assert.equal(f.calls.create,outcome==='criticalSuccess'?0:1);
  assert.equal(!!r.effects.children.slowed,['failure','criticalFailure'].includes(outcome));
  assert.equal(!!r.effects.children.fascinated,outcome==='criticalFailure');
 });
 for(const role of ['spell','slowed','fascinated'])test(`native immunity assessment preserves unaffected components: ${role}`,async()=>{
  const f=fixture({immunity:{[role]:true}});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});const r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  if(role==='spell'){assert.equal(f.calls.create,0);assert.equal(r.effects.status,'immune');}
  else{assert.equal(r.effects.children[role],null);assert.ok(r.effects.children[role==='slowed'?'fascinated':'slowed']);}
 });
 test('unknown immunity and foreign target/source claims cannot write',async()=>{
  for(const change of [f=>f.context.immunity.checked=false,f=>delete f.context.immunity.spell,f=>f.state.source.targetActorUuid='Actor.other',f=>f.game.user={id:'player',isGM:false}]){const f=fixture();change(f);await assert.rejects(f.effects.claim({actor:f.actor,state:f.state,context:f.context}));assert.equal(f.calls.update,0);}
 });
 test('create veto stays uncertain and never retries the native create',async()=>{
  const f=fixture({veto:true});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});await assert.rejects(f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce}));
  assert.equal(f.effects.get(f.actor,f.state.sourceNonce).effects.status,'uncertain');
  await assert.rejects(f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce}));assert.equal(f.calls.create,1);
 });
 test('lost create reply can observe exact committed operation but cannot create a duplicate',async()=>{
  const f=fixture({lostReply:true});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});
  try{await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});}catch{}
  const r=f.effects.get(f.actor,f.state.sourceNonce);assert.ok(['created','uncertain'].includes(r.effects.status));
  try{await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});}catch{}
  assert.equal(f.calls.create,1);assert.equal([...f.actor.items.values()].filter(i=>i.type==='effect').length,1);
 });
 test('lost create reply cannot leave an unbounded source effect after its proven expiry',async()=>{
  const f=fixture({lostReply:true});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});
  await assert.rejects(f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce}));
  let r=f.effects.get(f.actor,f.state.sourceNonce);
  const next=reduceRoaringSource(r.state,{type:'clock',sourceNonce:r.state.sourceNonce,observation:{worldTime:610}}).source;
  r=await f.effects.saveState({actor:f.actor,nonce:r.state.sourceNonce,state:next,expectedRevision:r.revision});
  await f.effects.end({actor:f.actor,nonce:r.state.sourceNonce});
  assert.equal(f.calls.create,1);assert.equal(f.actor.items.size,0);assert.equal(f.effects.get(f.actor,r.state.sourceNonce).effects.status,'ended');
 });
 test('ending source deletes only exact parent and native grants, preserving other conditions',async()=>{
  const f=fixture();const other=f.make({type:'condition',system:{slug:'fascinated'},flags:{}},'other');await f.effects.claim({actor:f.actor,state:f.state,context:f.context});let r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  const next=reduceRoaringSource(r.state,{type:'clock',sourceNonce:r.state.sourceNonce,observation:{worldTime:610}}).source;
  r=await f.effects.saveState({actor:f.actor,nonce:r.state.sourceNonce,state:next,expectedRevision:r.revision});
  await f.effects.end({actor:f.actor,nonce:r.state.sourceNonce});assert.equal(f.actor.items.size,1);assert.equal(f.actor.items.get(other.id),other);
  assert.equal(f.effects.get(f.actor,r.state.sourceNonce).effects.status,'ended');
 });
 test('reparented or foreign grant aborts cascade deletion instead of touching another source',async()=>{
  const f=fixture();await f.effects.claim({actor:f.actor,state:f.state,context:f.context});let r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  f.actor.items.get(r.effects.children.slowed.id).flags.pf2e.grantedBy.id='foreign';
  const next=reduceRoaringSource(r.state,{type:'clock',sourceNonce:r.state.sourceNonce,observation:{worldTime:610}}).source;
  r=await f.effects.saveState({actor:f.actor,nonce:r.state.sourceNonce,state:next,expectedRevision:r.revision});
  await assert.rejects(f.effects.end({actor:f.actor,nonce:r.state.sourceNonce}));assert.equal(f.calls.delete.length,0);
 });
 test('manual fascination end preserves the parent and cannot be undone by renewal',async()=>{
  const f=fixture();await f.effects.claim({actor:f.actor,state:f.state,context:f.context});let r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  const child=f.actor.items.get(r.effects.children.fascinated.id);assert.equal(f.effects.identifyOwnedItem(child).condition,'fascinated');
  const next=reduceRoaringSource(r.state,{type:'end-fascination',sourceNonce:r.state.sourceNonce,receiptId:'end-fascination',gmId:'gm',observation:{worldTime:10}}).source;
  r=await f.effects.saveState({actor:f.actor,nonce:r.state.sourceNonce,state:next,expectedRevision:r.revision});
  await f.effects.endFascination({actor:f.actor,nonce:r.state.sourceNonce});await f.effects.renew({actor:f.actor,nonce:r.state.sourceNonce});
  assert.equal(f.actor.items.has(child.id),false);assert.equal(f.actor.items.has(r.effects.parentId),true);assert.equal(f.calls.create,1);
  assert.equal(f.effects.identifyOwnedItem(child,{deleted:true}).condition,'fascinated');
 });
 test('document delete veto is not reported as ended',async()=>{
  const f=fixture({deleteVeto:true});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});let r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  const next=reduceRoaringSource(r.state,{type:'clock',sourceNonce:r.state.sourceNonce,observation:{worldTime:610}}).source;
  r=await f.effects.saveState({actor:f.actor,nonce:r.state.sourceNonce,state:next,expectedRevision:r.revision});
  await assert.rejects(f.effects.end({actor:f.actor,nonce:r.state.sourceNonce}));assert.notEqual(f.effects.get(f.actor,r.state.sourceNonce).effects.status,'ended');
  await assert.rejects(f.effects.end({actor:f.actor,nonce:r.state.sourceNonce}));assert.equal(f.calls.delete.length,1,'uncertain native delete is not replayed');
 });
 test('a lost deletion reply is reconciled by absence without repeating native deletion',async()=>{
  const f=fixture({deleteLostReply:true});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});let r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  const next=reduceRoaringSource(r.state,{type:'clock',sourceNonce:r.state.sourceNonce,observation:{worldTime:610}}).source;
  r=await f.effects.saveState({actor:f.actor,nonce:r.state.sourceNonce,state:next,expectedRevision:r.revision});
  await assert.rejects(f.effects.end({actor:f.actor,nonce:r.state.sourceNonce}));
  assert.equal(f.effects.get(f.actor,r.state.sourceNonce).effects.status,'deleting');
  await f.effects.end({actor:f.actor,nonce:r.state.sourceNonce});
  assert.equal(f.calls.delete.length,1);assert.equal(f.effects.get(f.actor,r.state.sourceNonce).effects.status,'ended');
 });
 test('finite fallback preserves frozen start and duration instead of extending from now',async()=>{
  const f=fixture();await f.effects.claim({actor:f.actor,state:f.state,context:f.context});let r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  const next=reduceRoaringSource(r.state,{type:'reconcile',sourceNonce:r.state.sourceNonce,observation:{worldTime:14,turn:null}}).source;
  r=await f.effects.saveState({actor:f.actor,nonce:r.state.sourceNonce,state:next,expectedRevision:r.revision});await f.effects.restoreFinite({actor:f.actor,nonce:r.state.sourceNonce});
  const parent=f.actor.items.get(r.effects.parentId);assert.equal(parent.system.start.value,10);assert.equal(parent.system.duration.value,1);assert.equal(parent.system.duration.unit,'rounds');assert.equal(f.calls.create,1);
 });
 test('source CAS and GM handoff reject stale writes',async()=>{
  const f=fixture();const r=await f.effects.claim({actor:f.actor,state:f.state,context:f.context});
  await assert.rejects(f.effects.saveState({actor:f.actor,nonce:f.state.sourceNonce,state:f.state,expectedRevision:r.revision+1}));
  f.game.users.activeGM={id:'new-gm'};await assert.rejects(f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce}));assert.equal(f.calls.create,0);
 });

 async function expired(f){
  const r=f.effects.get(f.actor,f.state.sourceNonce);
  const state=reduceRoaringSource(r.state,{type:'clock',sourceNonce:r.state.sourceNonce,observation:{worldTime:610}}).source;
  return f.effects.saveState({actor:f.actor,nonce:r.state.sourceNonce,state,expectedRevision:r.revision});
 }
 for(const recoverActive of [false,true])test(`lost create reply with a manually deleted child permits safe ${recoverActive?'active recovery':'expiry cleanup'}`,async()=>{
  const f=fixture({lostReply:true});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});
  await assert.rejects(f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce}));
  const child=[...f.actor.items.values()].find(i=>i.system.slug==='fascinated');
  await f.actor.deleteEmbeddedDocuments('Item',[child.id]);
  if(recoverActive){
   const r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
   assert.ok(r.state.tombstones.fascinated,'observed absence prevents the missing grant from returning');
   await f.effects.renew({actor:f.actor,nonce:f.state.sourceNonce});
   assert.equal(f.actor.items.has(child.id),false);
  }
  await expired(f);await f.effects.end({actor:f.actor,nonce:f.state.sourceNonce});
  assert.equal(f.actor.items.size,0);assert.equal(f.calls.create,1);
 });
 test('lost create recovery still rejects a surviving grant reparented to a different effect',async()=>{
  const f=fixture({lostReply:true});await f.effects.claim({actor:f.actor,state:f.state,context:f.context});
  await assert.rejects(f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce}));
  [...f.actor.items.values()].find(i=>i.system.slug==='slowed').flags.pf2e.grantedBy.id='foreign';
  await expired(f);await assert.rejects(f.effects.end({actor:f.actor,nonce:f.state.sourceNonce}));
  assert.equal(f.calls.delete.length,0);assert.equal(f.calls.create,1);
 });
 for(const editedSource of [false,true])test(`finite expiry ${editedSource?'rejects an edited source rule':'cleans native prepared ignored rules'}`,async()=>{
  const f=fixture();await f.effects.claim({actor:f.actor,state:f.state,context:f.context});let r=await f.effects.materialize({actor:f.actor,nonce:f.state.sourceNonce});
  const state=reduceRoaringSource(r.state,{type:'reconcile',sourceNonce:r.state.sourceNonce,observation:{worldTime:14,turn:null}}).source;
  r=await f.effects.saveState({actor:f.actor,nonce:r.state.sourceNonce,state,expectedRevision:r.revision});await f.effects.restoreFinite({actor:f.actor,nonce:r.state.sourceNonce});
  const parent=f.actor.items.get(r.effects.parentId);parent._source={system:structuredClone(parent.system)};
  parent.isExpired=true;for(const rule of parent.system.rules)rule.ignored=true;
  if(editedSource)parent._source.system.rules[0].uuid='Compendium.foreign.Item.condition';
  await expired(f);
  if(editedSource){await assert.rejects(f.effects.end({actor:f.actor,nonce:r.state.sourceNonce}));assert.equal(f.calls.delete.length,0);}
  else {await f.effects.end({actor:f.actor,nonce:r.state.sourceNonce});assert.equal(f.actor.items.size,0);}
 });
