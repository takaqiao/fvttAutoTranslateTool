import test from 'node:test';
import assert from 'node:assert/strict';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

function fixture({detached=false,veto=false,noHook=false,lostReply=false,deferred=false,differential=false,changeEvent,wrongReturn=false,noop=false}={}){
 const wrappers=new Map(),rpc=new Map(),hooks=new Map(),writes=[],docs=new Map(),gm={id:'gm',active:true},player={id:'player',active:true};
 const game={user:gm,users:new Map([['gm',gm],['player',player]]),actors:new Map(),messages:new Map(),time:{worldTime:1},scenes:{current:{id:'scene'}}};game.users.activeGM=gm;game.settings={get:()=>game.mode??'public'};gm.targets=new Set();
 const actor={id:'pc',uuid:'Actor.pc',type:'character',canAct:true,flags:{},items:new Map(),testUserPermission:u=>u===gm||u===player,getActiveTokens:()=>[]};game.actors.set(actor.id,actor);
 const apply=(doc,changes)=>{for(const [path,value]of Object.entries(changes)){let obj=doc;const keys=path.split('.');for(const k of keys.slice(0,-1))obj=obj[k]??={};obj[keys.at(-1)]=structuredClone(value);}};
 actor.update=async changes=>{writes.push({doc:'actor',changes:structuredClone(changes)});apply(actor,changes);return actor};
 const entry={id:'entry',uuid:'Actor.pc.Item.entry',type:'spellcastingEntry',actor,isSpontaneous:true,system:{prepared:{value:'spontaneous'},slots:{slot1:{value:3},slot2:{value:3},slot3:{value:2}}}};
 class SpellFixture{constructor(data){Object.assign(this,data)}loadVariant({castRank}={}){if(castRank===this.rank)return null;const original=this.original??this;return new SpellFixture({...original,rank:castRank??original.rank,original,appliedOverlays:new Map()})}}
 globalThis.CONFIG={PF2E:{Item:{documentClasses:{spell:SpellFixture}}}};
 const item=new SpellFixture({id:'spell',uuid:'Actor.pc.Item.spell',type:'spell',actor,rank:1,sourceId:'Compendium.test.spell',spellcasting:entry,system:{location:{value:'entry',signature:true},cast:{focusPoints:0},time:{value:'1 to 3'}}});
 const other=new SpellFixture({...item,id:'other',uuid:'Actor.pc.Item.other',sourceId:'Compendium.test.other'});
 for(const d of [actor,entry,item,other]){docs.set(d.uuid,d);if(d!==actor)actor.items.set(d.id,d);}
 const Hooks={on:(name,fn)=>{const list=hooks.get(name)??[];list.push(fn);hooks.set(name,list);return fn},off:(name,fn)=>hooks.set(name,(hooks.get(name)??[]).filter(f=>f!==fn))};
 const difference=(before,after)=>{if(JSON.stringify(before)===JSON.stringify(after))return undefined;if(after&&typeof after==='object'&&!Array.isArray(after))return Object.fromEntries(Object.entries(after).map(([k,v])=>[k,difference(before?.[k],v)]).filter(([,v])=>v!==undefined));return structuredClone(after)};
 const update=async(changes,options)=>{
  writes.push({doc:'entry',changes:structuredClone(changes),options:structuredClone(options)});
  if(veto)return undefined;
  const event=differential?{}:structuredClone(changes),eventOptions=structuredClone(options);
  if(differential)for(const [path,value]of Object.entries(changes)){const before=path.split('.').reduce((v,k)=>v?.[k],entry),delta=difference(before,value);if(delta!==undefined)apply(event,{[path]:delta})}
  if(!noop)apply(entry,changes);changeEvent?.({entry,changes:event,options:eventOptions});
  if(!noHook)for(const fn of hooks.get('updateItem')??[])fn(entry,event,eventOptions,game.user.id);
  if(lostReply)throw Error('reply lost');return wrongReturn?{...entry}:entry;
 };
 entry.update=(changes,options={})=>{const fn=wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.update');return fn?fn.call(entry,update,changes,options):update(changes,options)};
 let consumes=0,messages=0,castCalls=0;
 const nativeConsume=async(spell,rank)=>{consumes++;if(deferred)await Promise.resolve();const path=`slot${rank}`;if(entry.system.slots[path].value<1)return false;await entry.update({[`system.slots.${path}.value`]:entry.system.slots[path].value-1});return true};
 entry.consume=(spell,rank,slot,capability)=>wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.consume').call(entry,nativeConsume,spell,rank,slot,capability);
 const casts=createNativeCastEvents({game,fromUuid:async uuid=>docs.get(uuid),messageTimeoutMs:30});
 const emit=async(spell,rank)=>{
  const options={actualCast:true,data:{castRank:rank}},flags=casts.captureUsage(spell,{options});
  const message={id:`m${++messages}`,uuid:`ChatMessage.m${messages}`,author:game.user,rolls:[],blind:false,whisper:[],flags:{[ID]:flags,pf2e:{origin:{uuid:spell.uuid,actor:actor.uuid},context:{type:'spell-cast'}}}};
  game.messages.set(message.id,message);docs.set(message.uuid,message);
  casts.captureMessageOutcome(spell,options,{message,castNonce:flags?.nativeCast?.id});return message;
 };
 const nativeCast=async(spell,options)=>{castCalls++;if(options.consume===false||await entry.consume(spell,options.rank??spell.rank,options.slotId)){if(options.message!==false)await emit(spell,options.rank??spell.rank)}};
 const socket={register:(name,fn)=>rpc.set(name,fn)};
 casts.register({Hooks,libWrapper:{register:(_id,path,fn)=>wrappers.set(path,fn),unregister:()=>{}},socket});
 const cast=(spell=item,options={rank:1,slotId:NaN})=>wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast').call(entry,detached?(s,o)=>{nativeCast(s,o);}:nativeCast,spell,options);
 const enroll=(data={sourceTokenUuid:null,messageMode:'public',bridgeNonce:'bridge-one',fingerprint:'a'.repeat(64)})=>casts.addCastMiddleware(({item:spell},next)=>spell===item?next.withOutcome({kind:'force-barrage',data}):next());
 const adapter={validate:()=>true,consumePolicy:async(context,next)=>{context.expectSlotCommit({before:entry.system.slots[`slot${context.payload.rank}`].value,changes:()=>({})});return next()}};
 return {game,gm,player,actor,entry,item,other,casts,wrappers,rpc,hooks,writes,cast,enroll,adapter,emit,docs,counters:()=>({consumes,messages,castCalls})};
}

test('slot debit, core proof, and provider extension share one native entry update',async()=>{
 const f=fixture();f.casts.addInvocationAdapter('force-barrage',{...f.adapter,consumePolicy:async(c,next)=>{
  c.expectSlotCommit({before:3,changes:proof=>({[`flags.${ID}.providerSlotProof`]:proof})});return next();
 }});f.enroll();const out=await f.cast();
 const updates=f.writes.filter(w=>w.doc==='entry');assert.equal(updates.length,1);const change=updates[0].changes;
 assert.equal(change['system.slots.slot1.value'],2);assert.deepEqual(change[`flags.${ID}.nativeSlotCommit`],out.receipt.slotCommit);assert.deepEqual(change[`flags.${ID}.providerSlotProof`],out.receipt.slotCommit);
 assert.equal(out.receipt.slotCommit.userId,'gm');assert.equal(out.receipt.slotCommit.gmId,'gm');assert.equal(out.receipt.slotCommit.entryUuid,f.entry.uuid);
 await f.entry.update({[`flags.${ID}.nativeSlotCommit`]:{later:true}});assert.deepEqual(f.actor.flags[ID].nativeCasts[0].slotCommit,out.receipt.slotCommit);
});
for(const mode of ['veto','noHook','lostReply','deferred'])test(`${mode} is never a paid outcome or a second debit`,async()=>{
 const f=fixture({[mode]:true});f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();await assert.rejects(f.cast());assert.equal(f.counters().consumes,1);assert.equal(f.counters().messages,0);
 assert.equal(f.actor.flags[ID].nativeCasts[0].state,'uncertain');assert.equal(f.entry.system.slots.slot1.value,mode==='veto'?3:2);
});
test('provider extension cannot overwrite a slot or core marker',async()=>{
 for(const path of ['system.slots.slot1.value',`flags.${ID}.nativeSlotCommit`]){const f=fixture();f.casts.addInvocationAdapter('force-barrage',{...f.adapter,consumePolicy:async(c,next)=>{c.expectSlotCommit({before:3,changes:()=>({[path]:0})});return next()}});f.enroll();await assert.rejects(f.cast());assert.equal(f.entry.system.slots.slot1.value,3);assert.equal(f.counters().messages,0);}
});
test('missing slot commitment, wrong cost, and unsupported rank do not grant a paid result',async()=>{
 for(const mode of ['missing','cost','rank']){const f=fixture();f.casts.addInvocationAdapter('force-barrage',{...f.adapter,consumePolicy:async(c,next)=>{if(mode!=='missing')c.expectSlotCommit({before:3,cost:2,changes:()=>({})});return next()}});f.enroll();await assert.rejects(f.cast(f.item,{rank:mode==='rank'?4:1}));assert.equal(f.counters().messages,0);assert.equal(f.counters().consumes,0);}
});

test('a vetoed durable native claim never starts slot consumption',async()=>{
 const f=fixture(),update=f.actor.update;f.actor.update=changes=>changes[`flags.${ID}.nativeCasts`]?.some(r=>r.state==='claiming')?Promise.resolve(f.actor):update(changes);
 f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();await assert.rejects(f.cast());assert.equal(f.counters().consumes,0);assert.equal(f.entry.system.slots.slot1.value,3);
});
test('an ambiguous claim update return never authorizes the next resource write',async()=>{
 const f=fixture(),update=f.actor.update;f.actor.update=async changes=>{const result=await update(changes);return changes[`flags.${ID}.nativeCasts`]?.some(r=>r.state==='claiming')?undefined:result};
 f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();await assert.rejects(f.cast());assert.equal(f.counters().consumes,0);assert.equal(f.entry.system.slots.slot1.value,3);
});
test('successive different-rank and same-rank payments accept native differential events with fresh nonces',async()=>{
 const observed=[],f=fixture({differential:true,changeEvent:event=>observed.push(structuredClone(event.changes))});f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();
 const outcomes=[];for(const rank of [3,2,2])outcomes.push(await f.cast(f.item,{rank,slotId:NaN}));
 assert.equal(new Set(outcomes.map(o=>o.castNonce)).size,3);assert.equal(f.counters().consumes,3);assert.equal(f.counters().messages,3);
 assert.deepEqual(outcomes.map(o=>[o.receipt.slotCommit.rank,o.receipt.slotCommit.before,o.receipt.slotCommit.after]),[[3,2,1],[2,3,2],[2,2,1]]);
 assert.deepEqual(Object.keys(observed[1].flags[ID].nativeSlotCommit).sort(),['after','before','castNonce','rank']);
 assert.deepEqual(Object.keys(observed[2].flags[ID].nativeSlotCommit).sort(),['after','before','castNonce']);
 for(const o of outcomes){assert.equal(o.status,'completed');assert.equal(o.receipt.state,'used');assert.deepEqual(f.actor.flags[ID].nativeCasts.find(r=>r.id===o.castNonce).slotCommit,o.receipt.slotCommit)}
});
for(const mode of ['missing-nonce','stale-nonce','wrong-options','options-only','unrelated-update','unknown-field','deletion-field','wrong-delta-value','wrong-live-marker','wrong-slot','wrong-return','noop'])test(`slot witness rejects ${mode} without a completed card or retry`,async()=>{
 const f=fixture({differential:true,wrongReturn:mode==='wrong-return',noop:mode==='noop',changeEvent:({entry,changes,options})=>{
  const delta=changes.flags[ID].nativeSlotCommit;
  if(mode==='missing-nonce')delete delta.castNonce;
  if(mode==='stale-nonce')delta.castNonce='old-cast';
  if(mode==='wrong-options')options[ID].nativeSlotCommit.castNonce='old-cast';
  if(mode==='options-only')delete changes.flags[ID].nativeSlotCommit;
  if(mode==='unrelated-update'){delete changes.flags;delete changes.system;changes.name='Unrelated update'}
  if(mode==='unknown-field')delta.unverified=true;
  if(mode==='deletion-field')delta['-=userId']=null;
  if(mode==='wrong-delta-value')delta.cost=2;
  if(mode==='wrong-live-marker')entry.flags[ID].nativeSlotCommit.castNonce='wrong-persisted';
  if(mode==='wrong-slot')changes.system.slots.slot1.value=1;
 }});f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();await assert.rejects(f.cast());assert.equal(f.counters().consumes,1);assert.equal(f.counters().messages,0);assert.equal(f.actor.flags[ID].nativeCasts[0].state,'uncertain');
});
