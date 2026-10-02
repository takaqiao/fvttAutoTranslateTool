import test from 'node:test';
import assert from 'node:assert/strict';
import {registerAvRefocusEvents} from '../scripts/av-refocus-events.mjs';
import {WAVE_SOURCE} from '../scripts/av-wave-repair.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';
import {createRefocusAdapter} from '../scripts/exploration/refocus.mjs';

function fixture(t,{at,change,repair=false,notify=false,synthetic=false,note=false}={}){
 const calls=[],errors=[],hooks=new Map(),gm={id:'gm',isGM:true,active:true},nextGM={id:'next',isGM:true,active:true},user={id:'owner',active:true};
 const users=new Map([gm,nextGM,user].map(u=>[u.id,u]));users.activeGM=gm;
 let permitted=true,energy='fire';
 const actor={id:'pc',uuid:synthetic?'Scene.scene.Token.token.Actor.pc':'Actor.pc',isToken:synthetic,type:'character',items:new Map(),flags:{},system:{resources:{focus:{value:1,max:1}}},testUserPermission:u=>permitted&&u===user,getRollOptions:()=>['conservation-of-energy:'+energy]};
 const token={id:'token',uuid:'Scene.scene.Token.token',actor},scene={id:'scene',tokens:new Map([[token.id,token]])};
 if(synthetic)actor.token=token;
 const game={user:gm,users,actors:new Map([[actor.id,actor]]),scenes:new Map([[scene.id,scene]]),time:{worldTime:100}};
 const mutate=point=>{if(point!==at)return;if(change==='gm')users.activeGM=nextGM;else if(change==='owner')permitted=false;else if(change==='user')users.set(user.id,{...user});else if(change==='actor')game.actors.set(actor.id,{...actor});else if(change==='token')scene.tokens.delete(token.id);else if(change==='token-actor')token.actor={...actor,uuid:'Scene.scene.Token.other.Actor.pc'};else if(change==='item')actor.items.set('wave',{...wave});else if(change==='parent')wave.actor={...actor};else if(change==='source')wave.sourceId='Compendium.pf2e.classfeatures.Item.other';};
 const apply=(doc,changes)=>{for(const [path,value]of Object.entries(changes)){const parts=path.split('.');let target=doc;for(const part of parts.slice(0,-1))target=target[part]??={};target[parts.at(-1)]=structuredClone(value)}};
 actor.update=async changes=>{calls.push({point:'event',changes:structuredClone(changes)});apply(actor,changes);mutate('event');return actor};
 actor.toggleRollOption=async(_domain,_option,_item,_enabled,selection)=>{calls.push({point:'toggle',selection});energy=selection;mutate('toggle');return null};
 const originalSlugs=['arctic-rift','breathe-fire','blazing-bolt','falling-star','fireball','frostbite','frozen-fog','ice-storm','ignition','volcanic-eruption'];
 const wave={id:'wave',uuid:actor.uuid+'.Item.wave',actor,sourceId:WAVE_SOURCE,flags:{},system:{rules:repair?[{key:'DamageAlteration',mode:'override',property:'damage-type',slug:'base',selectors:['spell-damage'],value:'{item|flags.system.rulesSelections.conservationOfEnergy}',predicate:[{or:originalSlugs.map(s=>'item:slug:'+s)}]}]:[]},async update(changes){
  const point=Object.hasOwn(changes,'system.rules')?'repair':changes[`flags.${ID}.avRefocusReceipts`]?.at(-1)?.state??'other';calls.push({point,changes:structuredClone(changes)});apply(wave,changes);mutate(point);return wave;
 }};actor.items.set(wave.id,wave);
 const proof={nonce:'native-original',actorUuid:actor.uuid,itemUuid:wave.uuid,userId:user.id,before:0,after:1,tokenUuid:token.uuid,startedAt:100};actor.flags[ID]={avRefocusIntent:structuredClone(proof)};
 const Hooks={on:(name,fn)=>{hooks.set(name,fn);return fn},off:name=>hooks.delete(name)};
 const off=registerAvRefocusEvents({game,Hooks,onError:e=>errors.push(String(e)),actorMatchers:notify?[a=>a===actor]:[],onRefocus:notify?async()=>{calls.push({point:'subscriber'});mutate('subscriber')}:undefined,refocusPrivacy:note?{waitRefocusNote:async()=>{calls.push({point:'note'});mutate('note');return 'native-note'}}:undefined});t.after(off);
 if(note){proof.privacy={mode:'public',audience:[]};actor.flags[ID].avRefocusIntent=structuredClone(proof)}
 const execute=()=>hooks.get('updateActor')(actor,{[`flags.${ID}.avRefocusIntent`]:structuredClone(proof),'system.resources.focus.value':1},{[ID]:{refocusReceipt:structuredClone(proof)}},user.id);
 return {actor,wave,calls,errors,execute,energy:()=>energy};
}

for(const synthetic of [false,true])test(`exact ${synthetic?'synthetic':'world'} Refocus wave completes once and does not repeat its saved claim`,async t=>{
 const f=fixture(t,{synthetic});await f.execute();assert.deepEqual(f.errors,[]);assert.equal(f.energy(),'none');assert.equal(f.wave.flags[ID].avRefocusReceipts[0].state,'done');
 await f.execute();assert.equal(f.calls.filter(c=>c.point==='toggle').length,1);assert.deepEqual(f.errors,[]);
});

for(const change of ['gm','owner','user','actor','token','token-actor','item','parent','source'])test(`Refocus stops after its saved claim when original ${change} changes`,async t=>{
 const f=fixture(t,{at:'claimed',change});await f.execute();assert.equal(f.calls.filter(c=>c.point==='toggle').length,0);assert.equal(f.energy(),'fire');assert.equal(f.wave.flags[ID].avRefocusReceipts[0].state,'claimed');assert.equal(f.errors.length,1);
 await f.execute();assert.equal(f.calls.filter(c=>c.point==='toggle').length,0);
});

test('a GM change while repairing the native wave keeps the saved claim without toggling',async t=>{
 const f=fixture(t,{at:'repair',change:'gm',repair:true});await f.execute();assert.equal(f.calls.filter(c=>c.point==='repair').length,1);assert.equal(f.calls.filter(c=>c.point==='toggle').length,0);assert.equal(f.wave.flags[ID].avRefocusReceipts[0].state,'claimed');assert.equal(f.errors.length,1);
});

test('a GM change during the original toggle cannot mark its receipt done or replay it',async t=>{
 const f=fixture(t,{at:'toggle',change:'gm'});await f.execute();assert.equal(f.energy(),'none');assert.equal(f.wave.flags[ID].avRefocusReceipts[0].state,'claimed');assert.equal(f.errors.length,1);await f.execute();assert.equal(f.calls.filter(c=>c.point==='toggle').length,1);
});

test('losing authority while saving a subscriber event prevents all following wave writes',async t=>{
 const f=fixture(t,{at:'event',change:'gm',notify:true});await f.execute();assert.equal(f.actor.flags[ID].refocusEvents[0].state,'claimed');assert.equal(f.wave.flags[ID],undefined);assert.equal(f.calls.filter(c=>c.point==='subscriber').length,0);assert.equal(f.errors.length,1);
});

test('losing actor ownership while awaiting a native note prevents the subscriber',async t=>{
 const f=fixture(t,{at:'note',change:'owner',notify:true,note:true});await f.execute();assert.equal(f.calls.filter(c=>c.point==='note').length,1);assert.equal(f.calls.filter(c=>c.point==='subscriber').length,0);assert.equal(f.actor.flags[ID].refocusEvents[0].state,'claimed');assert.equal(f.errors.length,1);
});

test('losing GM authority during a subscriber preserves claimed evidence without a done write',async t=>{
 const f=fixture(t,{at:'subscriber',change:'gm',notify:true});await f.execute();assert.equal(f.calls.filter(c=>c.point==='subscriber').length,1);assert.equal(f.actor.flags[ID].refocusEvents[0].state,'claimed');assert.equal(f.errors.length,1);
});

test('an ordinary exploration Refocus keeps its admitted subscriber after the native scope completes',async t=>{
 const gm={id:'gm',isGM:true,active:true},users=new Map([[gm.id,gm]]);users.activeGM=gm;
 const actor={id:'pc',uuid:'Actor.pc',type:'character',items:new Map(),flags:{},system:{resources:{focus:{value:0,max:1}}},testUserPermission:()=>true,hasCondition:()=>false};
 const token={id:'token',uuid:'Scene.scene.Token.token',actor},scene={id:'scene',tokens:new Map([[token.id,token]])};
 const game={user:gm,users,actors:new Map([[actor.id,actor]]),scenes:new Map([[scene.id,scene]]),time:{worldTime:600},PF2eWorkbench:{}};
 const hooks=new Map(),errors=[],observations=[],Hooks={on:(name,fn)=>{hooks.set(name,fn);return fn},off:name=>hooks.delete(name)};
 actor.update=async changes=>{for(const [path,value]of Object.entries(changes)){const parts=path.split('.');let target=actor;for(const part of parts.slice(0,-1))target=target[part]??={};target[parts.at(-1)]=structuredClone(value)}return actor};
 const activity={id:'ordinary-native',actorUUID:actor.uuid,startedAt:0,endsAt:600,options:{}},ctx={validate(){}};
 const adapter=createRefocusAdapter({game,canvas:{tokens:{controlled:[{actor,document:token}]}},fromUuid:async()=>actor,ownerOperations:{isActivityContext:c=>c===ctx}});
 const off=registerAvRefocusEvents({game,Hooks,actorMatchers:[a=>!!adapter.getCurrent(a)],onRefocus:async event=>{observations.push(event.proof.nonce);adapter.capture(event)},onError:e=>errors.push(String(e))});t.after(off);
 let observer;
 game.PF2eWorkbench.refocus=async()=>{
  const proof={nonce:activity.id,actorUuid:actor.uuid,itemUuid:null,userId:gm.id,before:0,after:1,tokenUuid:token.uuid,startedAt:0},changes={'system.resources.focus.value':1,[`flags.${ID}.avRefocusIntent`]:proof};
  await actor.update(changes);
  // Foundry emits updateActor without awaiting asynchronous subscribers.
  observer=hooks.get('updateActor')(actor,changes,{[ID]:{refocusReceipt:proof}},gm.id);return actor;
 };
 const receipt=await adapter.complete(activity,ctx);await observer;
 assert.equal(receipt.id,activity.id);assert.equal(adapter.getCurrent(actor),null);assert.deepEqual(errors,[]);assert.deepEqual(observations,[activity.id]);assert.equal(actor.flags[ID].refocusEvents[0].state,'done');
});

test('unrelated actor updates never enter a private Refocus admission matcher',t=>{
 const gm={id:'gm'},hooks=new Map(),game={user:gm,users:{activeGM:gm,get:()=>gm}};
 const off=registerAvRefocusEvents({game,Hooks:{on:(name,fn)=>{hooks.set(name,fn);return fn},off(){}},actorMatchers:[()=>{throw Error('private admission must not be read')} ]});t.after(off);
 assert.doesNotThrow(()=>hooks.get('updateActor')({items:[],flags:{}},{'system.resources.focus.value':1},{},gm.id));
});
