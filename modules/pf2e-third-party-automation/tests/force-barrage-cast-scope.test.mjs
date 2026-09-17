import test from 'node:test';
import assert from 'node:assert/strict';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

function fixture({detached=false,veto=false,noHook=false,lostReply=false,deferred=false}={}){
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
 const update=async(changes,options)=>{
  writes.push({doc:'entry',changes:structuredClone(changes),options:structuredClone(options)});
  if(veto)return undefined;
  apply(entry,changes);if(!noHook)for(const fn of hooks.get('updateItem')??[])fn(entry,changes,options,game.user.id);
  if(lostReply)throw Error('reply lost');return entry;
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
 return {game,gm,player,actor,entry,item,other,casts,wrappers,rpc,hooks,writes,cast,enroll,adapter,emit,docs,socket,counters:()=>({consumes,messages,castCalls})};
}

test('outcome enrollment leaves other spells and independent consumes unmanaged',async()=>{
 const f=fixture();assert.equal(typeof f.casts.addInvocationAdapter,'function');f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();
 f.game.users.activeGM=null;
 const result=await f.cast(f.other,{rank:1});assert.equal(result,undefined);assert.equal(f.actor.flags[ID],undefined);
 await f.entry.consume(f.other,1);assert.equal(f.actor.flags[ID],undefined);assert.equal(f.counters().consumes,2);
});
for(const detached of [false,true])test(`exact outcome binds the final card after native ${detached?'detached':'void'} return`,async()=>{
 const f=fixture({detached});f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();
 const out=await f.cast();assert.equal(out.status,'completed');assert.equal(out.nativeResult,undefined);assert.equal(out.input.slotId,null);
 assert.equal(out.receipt.state,'used');assert.equal(out.receipt.messageId,out.message.id);assert.equal(out.receipt.id,out.castNonce);assert.equal(out.receipt.invocation.data.bridgeNonce,'bridge-one');
 assert.equal(out.receipt.slotCommit.castNonce,out.castNonce);assert.equal(out.receipt.slotCommit.before,3);assert.equal(out.receipt.slotCommit.after,2);
 assert.equal(f.actor.flags[ID].nativeCasts[0].messageId,out.message.id);assert.deepEqual(f.counters(),{consumes:1,messages:1,castCalls:1});
});
test('unknown and duplicate adapter kinds are rejected without native calls',async()=>{
 const f=fixture();f.casts.addInvocationAdapter('force-barrage',f.adapter);assert.throws(()=>f.casts.addInvocationAdapter('force-barrage',f.adapter));
 f.casts.addCastMiddleware((_ctx,next)=>next.withOutcome({kind:'unknown',data:{}}));await assert.rejects(f.cast());assert.equal(f.counters().consumes,0);
});
test('an enrolled continuation is single-use and cannot be restored from its old payload',async()=>{
 const f=fixture();f.casts.addInvocationAdapter('force-barrage',f.adapter);let saved;
 f.casts.addCastMiddleware(async(_ctx,next)=>{saved=next;const out=await next.withOutcome({kind:'force-barrage',data:{sourceTokenUuid:null,messageMode:'public',bridgeNonce:'one'}});await assert.rejects(next.withOutcome({kind:'force-barrage',data:{sourceTokenUuid:null,messageMode:'public',bridgeNonce:'two'}}));return out});
 const out=await f.cast();await assert.rejects(saved.withOutcome({kind:'force-barrage',data:{sourceTokenUuid:null,messageMode:'public',bridgeNonce:'one'}}));
 const reply=await f.rpc.get('native-cast-pay').call({socketdata:{userId:'gm'}},{...out.input,id:out.castNonce,invocation:out.receipt.invocation,nativeCastScope:out.receipt.nativeCastScope});assert.equal(reply.ok,false);assert.equal(f.counters().consumes,1);
});
test('explicit preview/silent modes cannot enroll and ordinary paths are not changed',async()=>{
 for(const options of [{rank:1,consume:false},{rank:1,message:false}]){const f=fixture();f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();await assert.rejects(f.cast(f.item,options));assert.equal(f.counters().consumes,0);}
});
test('validator rejection and GM handoff after validation stop before native debit',async()=>{
 for(const handoff of [false,true]){const f=fixture();f.casts.addInvocationAdapter('force-barrage',{...f.adapter,validate:async()=>{if(handoff)f.game.users.activeGM={id:'new'};return handoff}});f.enroll();await assert.rejects(f.cast());assert.equal(f.counters().consumes,0);assert.equal(f.counters().messages,0);}
});
test('only exact spontaneous enrolled button normalizes NaN slot, prepared input does not',async()=>{
 const f=fixture();f.entry.isSpontaneous=false;f.entry.system.prepared.value='prepared';f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();await assert.rejects(f.cast());assert.equal(f.counters().consumes,0);
});
test('paid disruption returns its exact terminal receipt without an actualCast message',async()=>{
 const f=fixture();f.casts.addInvocationAdapter('force-barrage',f.adapter);f.casts.addPaidCastPolicy(()=>({disrupted:true,reason:'test disruption'}));f.enroll();
 const out=await f.cast();assert.equal(out.status,'disrupted');assert.equal(out.receipt.state,'disrupted');assert.equal(out.message,null);assert.equal(f.counters().consumes,1);assert.equal(f.counters().messages,0);
});

test('a detached cast cannot lend its enrollment to a direct same-item consume',async()=>{
 const f=fixture();f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();let begin,finish;
 const entered=new Promise(r=>begin=r),release=new Promise(r=>finish=r);
 const wrapper=f.wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast');
 const task=wrapper.call(f.entry,async(spell,options)=>{begin();await release;if(await f.entry.consume(spell,options.rank,options.slotId))await f.emit(spell,options.rank)},f.item,{rank:1});
 await entered;
 await f.entry.consume(f.item,1);assert.equal(f.actor.flags[ID],undefined,'Independent consume must not claim/pay the enrolled cast');
 finish();const out=await task;assert.equal(out.status,'completed');assert.equal(out.receipt.slotCommit.before,2);assert.equal(f.counters().consumes,2);
});

test('same-rank factory uses the installed PF8.5.1 loadVariant body, including late preparation',{skip:!process.env.PF2E_NATIVE_BUNDLE},async()=>{
 const text=await readFile(process.env.PF2E_NATIVE_BUNDLE,'utf8'),start=text.indexOf('\tloadVariant(e = {}) {'),end=text.indexOf('\n\tgetHeightenLayers(',start);
 assert.ok(start>=0&&end>start,'Review installed native factory boundary after a PF update');
 const scope=vm.createContext({structuredClone});
 vm.runInContext(`function mergeObject(a,b){for(const [k,v]of Object.entries(b))a[k]=v&&typeof v==='object'&&!Array.isArray(v)?mergeObject(a[k]??{},v):v;return a}globalThis.foundry={utils:{mergeObject}};Math.clamp=(v,a,b)=>Math.min(Math.max(v,a),b);function performLatePreparation(s){s.latePrepared=true}class SpellPF2e{constructor(data,{parent=null,parentItem=null}={}){this.data=data;this.id=data._id;this.system=data.system;this.rank=this.system.location.heightenedLevel??this.system.level.value;this.parent=parent;this.parentItem=parentItem;this.traits=new Set();this.overlays=new Map()}toObject(){return structuredClone(this.data)}getHeightenLayers(){return []}${text.slice(start,end)}}globalThis.SpellPF2e=SpellPF2e;`,scope);
 const base=new scope.SpellPF2e({_id:'spell',system:{level:{value:1},location:{value:'entry'},traits:{value:[]}}});
 assert.equal(base.loadVariant({castRank:1}),null);const same=base.loadVariant({});assert.notEqual(same,base);assert.equal(same.original,base);assert.equal(same.rank,1);assert.equal(same.latePrepared,true);
 const high=base.loadVariant({castRank:3});assert.equal(high.rank,3);assert.equal(high.original,base);assert.equal(high.latePrepared,true);
});

test('original-item late cards cannot become the enrolled final card',async()=>{
 const f=fixture();f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();
 const wrapper=f.wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast');
 const out=await wrapper.call(f.entry,async(spell,options)=>{
  assert.notEqual(spell,f.item);if(await f.entry.consume(spell,1,options.slotId)){
   const late=await f.emit(f.item,1);assert.equal(late.flags[ID],null);await f.emit(spell,1);
  }
 },f.item,{rank:1});assert.equal(out.message.id,'m2');assert.equal(out.receipt.messageId,'m2');
});

test('proof endpoint refuses foreign caller, altered source token and replay after completion',async()=>{
 const f=fixture();let observed;
 f.casts.addInvocationAdapter('force-barrage',{...f.adapter,validate:async c=>{
  const proof=f.rpc.get('native-cast-invocation-proof'),payload={...c.payload,userId:'gm'};
  assert.equal(await proof.call({socketdata:{userId:'player'}},payload),false);
  assert.equal(await proof.call({socketdata:{userId:'gm'}},{...payload,nativeCastScope:{...payload.nativeCastScope,tokenUuid:'Scene.other.Token.other'}}),false);
  assert.equal(await proof.call({socketdata:{userId:'gm'}},payload),true);observed=payload;return true;
 }});f.enroll();await f.cast();assert.equal(await f.rpc.get('native-cast-invocation-proof').call({socketdata:{userId:'gm'}},observed),false);
});

test('a real player-side enrollment is proved to GM and binds the same durable result',async()=>{
 const f=fixture(),clientWrappers=new Map(),clientRPC=new Map(),clientGame={...f.game,user:f.player};f.player.targets=new Set();
 const client=createNativeCastEvents({game:clientGame,fromUuid:async uuid=>f.docs.get(uuid),messageTimeoutMs:40});
 const invoke=(map,sender,name,payload)=>map.get(name).call({socketdata:{userId:sender}},payload);
 const gmSocket={register:(name,fn)=>f.rpc.set(name,fn),executeAsUser:(name,user,payload)=>{assert.equal(user,'player');return invoke(clientRPC,'gm',name,payload)}};
 // Re-register socket transport through the object already captured by GM core.
 // Fixture exposes its original socket so this does not install another wrapper.
 Object.assign(f.socket,gmSocket);
 client.register({libWrapper:{register:(_id,p,fn)=>clientWrappers.set(p,fn)},socket:{register:(n,fn)=>clientRPC.set(n,fn),executeAsUser:(name,user,payload)=>{assert.equal(user,'gm');return invoke(f.rpc,'player',name,payload)}}});
 const originalConsume=f.entry.consume;
 f.entry.consume=(spell,rank,slot,cap)=>cap!==undefined?originalConsume(spell,rank,slot,cap):clientWrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.consume').call(f.entry,()=>{throw Error('Unexpected uncontrolled player native consume')},spell,rank,slot);
 f.casts.addInvocationAdapter('force-barrage',f.adapter);client.addInvocationAdapter('force-barrage',f.adapter);
 client.addCastMiddleware((_ctx,next)=>next.withOutcome({kind:'force-barrage',data:{sourceTokenUuid:null,messageMode:'public',bridgeNonce:'player-bridge'}}));
 const out=await clientWrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast').call(f.entry,async(spell,options)=>{
  if(await f.entry.consume(spell,1,options.slotId)){
   const messageOptions={actualCast:true,data:{castRank:1}},own=client.captureUsage(spell,{options:messageOptions});
   const message={id:'playerMessage',uuid:'ChatMessage.playerMessage',author:f.player,rolls:[],blind:false,whisper:[],flags:{[ID]:own,pf2e:{origin:{uuid:spell.uuid,actor:f.actor.uuid}}}};
   f.game.messages.set(message.id,message);f.docs.set(message.uuid,message);client.captureMessageOutcome(spell,messageOptions,{message,castNonce:own.nativeCast.id});
  }
 },f.item,{rank:1});
 assert.equal(out.status,'completed');assert.equal(out.receipt.state,'used');assert.equal(out.receipt.userId,'player');assert.equal(out.receipt.slotCommit.userId,'player');assert.equal(out.receipt.slotCommit.gmId,'gm');assert.equal(f.entry.system.slots.slot1.value,2);
});

test('a cancelled middleware cannot retain an unused outcome capability for later',async()=>{
 const f=fixture();f.casts.addInvocationAdapter('force-barrage',f.adapter);let saved;
 f.casts.addCastMiddleware((_ctx,next)=>{saved=next;return 'cancelled'});assert.equal(await f.cast(),'cancelled');
 await assert.rejects(saved.withOutcome({kind:'force-barrage',data:{sourceTokenUuid:null,messageMode:'public',bridgeNonce:'late'}}));assert.equal(f.counters().consumes,0);
});

test('GM handoff while the native draft renders vetoes its publication and preserves uncertain debit',async()=>{
 const f=fixture();f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();let published=0;
 const wrapper=f.wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast');
 await assert.rejects(wrapper.call(f.entry,async(spell,options)=>{
  if(await f.entry.consume(spell,1,options.slotId)){
   const opts={actualCast:true,data:{castRank:1}},own=f.casts.captureUsage(spell,{options:opts});
   f.game.users.activeGM={id:'new'};
   const message={flags:{[ID]:own,pf2e:{origin:{uuid:spell.uuid,actor:f.actor.uuid}}}};
   const allowed=(f.hooks.get('preCreateChatMessage')??[]).every(fn=>fn(message,{}, {},'gm')!==false);
   if(allowed)published++;
   f.casts.captureMessageOutcome(spell,opts,{error:Error('publication veto'),castNonce:own.nativeCast.id});
  }
 },f.item,{rank:1}));assert.equal(published,0);assert.equal(f.entry.system.slots.slot1.value,2);
});

test('only the enrolled public invocation freezes explicit message mode across later defaults',async()=>{
 const f=fixture();f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();
 const wrapper=f.wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast');
 const opts={rank:1};const out=await wrapper.call(f.entry,async(spell,options)=>{f.game.mode='gm';assert.equal(options.messageMode,'public');if(await f.entry.consume(spell,1,options.slotId))await f.emit(spell,1)},f.item,opts);
 assert.equal(out.input.messageMode,'public');assert.equal(out.receipt.invocation.data.messageMode,'public');assert.equal(opts.messageMode,undefined);
});
test('an invocation cannot reinterpret an initially private native mode as public',async()=>{
 const f=fixture();f.game.mode='gm';f.casts.addInvocationAdapter('force-barrage',f.adapter);f.enroll();await assert.rejects(f.cast());assert.equal(f.counters().consumes,0);
});
