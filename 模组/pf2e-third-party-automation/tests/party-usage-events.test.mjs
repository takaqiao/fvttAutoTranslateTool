import test from 'node:test';
import assert from 'node:assert/strict';
import {createPartyAutomation,PARTY_SOURCES} from '../scripts/party-automation.mjs';
import {registerUsageEvents,USE_ACTION_OPTION} from '../scripts/usage-events.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

const targetError='使用此能力时请选中一个其他生物作为目标。';
const names={clue:'线索指引',anoint:'符血点化',guardian:'忠诚卫士'};
const patch=(doc,changes)=>{for(const[key,value]of Object.entries(changes)){let at=doc;const parts=key.split('.');for(const part of parts.slice(0,-1))at=at[part]??={};const last=parts.at(-1);if(last.startsWith('-='))delete at[last.slice(2)];else at[last]=structuredClone(value);}};
function hooks(){const callbacks=new Map();let id=0;return {on(name,fn){const entry={id:++id,fn},list=callbacks.get(name)??[];list.push(entry);callbacks.set(name,list);return entry.id;},off(name,id){callbacks.set(name,(callbacks.get(name)??[]).filter(entry=>entry.id!==id));},async emit(name,...args){for(const entry of [...callbacks.get(name)??[]])await entry.fn(...args);}};}

function fixture(t){
 const gm={id:'gm',active:true,targets:new Set()},owner={id:'owner',active:true,targets:new Set()},users=Object.assign(new Map([[gm.id,gm],[owner.id,owner]]),{activeGM:gm});
 const Hooks=hooks(),docs=new Map(),messages=new Map(),dispatches=[],errors=[],writes={usage:0,frequency:0,nativeFrequency:0,effects:0,actor:0,cast:0};let messageId=0,effectId=0;
 const actor=(id)=>({id,uuid:`Actor.${id}`,type:'character',items:new Map(),flags:{},attributes:{shield:{raised:true,broken:false,destroyed:false,itemId:'shield'}},testUserPermission:user=>user===owner||user===gm,isAllyOf:()=>true,getRollOptions:()=>[],async update(changes){writes.actor++;patch(this,changes);return this;},async createEmbeddedDocuments(_name,items){writes.effects+=items.length;return items.map(data=>{const item={...structuredClone(data),id:`effect-${++effectId}`};this.items.set(item.id,item);return item;});},async deleteEmbeddedDocuments(_name,ids){for(const id of ids)this.items.delete(id);}});
 const source=actor('source'),recipient=actor('recipient'),scene={id:'scene',tokens:new Map()};
 const token=(id,actor)=>{const doc={id,uuid:`Scene.scene.Token.${id}`,documentName:'Token',actor,parent:scene,object:{}};doc.object.document=doc;scene.tokens.set(id,doc);docs.set(doc.uuid,doc);return doc;};
 const origin=token('origin',source),target=token('target',recipient);source.getActiveTokens=()=>[origin];recipient.getActiveTokens=()=>[target];owner.targets.add(target.object);
 const game={user:gm,users,actors:new Map([[source.id,source],[recipient.id,recipient]]),scenes:new Map([[scene.id,scene]]),messages,time:{worldTime:100},combats:new Map()};
 for(const kind of Object.keys(names)){
  const item={id:kind,uuid:`${source.uuid}.Item.${kind}`,name:names[kind],type:kind==='clue'?'action':'feat',actor:source,sourceId:PARTY_SOURCES[kind],system:{frequency:kind==='clue'?{max:1,per:'PT10M',value:1}:null},getOriginData:()=>({rollOptions:[]}),async update(changes){if(Object.hasOwn(changes,'system.frequency.value'))writes.frequency++;patch(this,changes);return this;}};
  source.items.set(item.id,item);docs.set(item.uuid,item);
 }
 source.items.set('shield',{id:'shield',type:'shield',baseType:'shield'});
 for(const key of ['clueEffect','anointEffect','imperialEffect'])docs.set(PARTY_SOURCES[key],{toObject:()=>({type:'effect',system:{rules:[]}})});
 const fromUuid=async uuid=>docs.get(uuid),party=createPartyAutomation({game,fromUuid,choose:async({choices})=>choices[0].value,castEvents:{addMatcher(){},async ensurePaid(){writes.cast++;}}});
 t.after(party.register({Hooks,onError:error=>errors.push(error)}));
 const unregister=registerUsageEvents({game,Hooks,fromUuid,libWrapper:null,canvas:null,resolveAction:party.resolveAction,requiresActualUse:(item,action)=>party.requiresActualUse?.(item,action)??false,tracksFrequency:item=>item.sourceId===PARTY_SOURCES.clue,executeUsage:context=>{dispatches.push(context);return party.executeUsage(context);},onError:error=>errors.push(error)});t.after(unregister);
 function document(data){return {...data,updateSource(changes){patch(this,changes);},async update(changes){if(Object.hasOwn(changes,`flags.${ID}.usage`))writes.usage++;patch(this,changes);return this;},toObject(){return structuredClone({id:this.id,author:this.author.id??this.author,speaker:this.speaker,flags:this.flags,rolls:this.rolls});}};}
 function card(item,{actualUse=false,targets=[],nativeUse=false,context=null,roll=false,input=true}={}){
  const message=document({id:`card-${++messageId}`,author:owner,speaker:{actor:source.id,scene:scene.id,token:origin.id},flags:{pf2e:{origin:{uuid:item.uuid,actor:source.uuid,type:item.type,rollOptions:nativeUse?[USE_ACTION_OPTION]:[]},...context?{context}:{}},...input?{[ID]:{usageInput:{actualUse,targetUuids:targets}}}:{}},rolls:roll?[{total:24}]:[],isCheckRoll:roll,isRoll:roll});
  Object.defineProperties(message,{actor:{get:()=>source},item:{get:()=>item}});return message;
 }
 const receive=async(message)=>{messages.set(message.id,message);await Hooks.emit('createChatMessage',message,{},owner.id);return message;};
 const raiseShield=()=>Hooks.emit('createChatMessage',{actor:source,item:{slug:'raise-a-shield'},flags:{}},{},owner.id);
 function ownerEntrances(){
  const previousConfig=globalThis.CONFIG,previousClass=globalThis.getDocumentClass;t.after(()=>{globalThis.CONFIG=previousConfig;globalThis.getDocumentClass=previousClass;});
  class Sheet{activateClickListener(){}}globalThis.CONFIG={Actor:{sheetClasses:{character:{'pf2e.Native':{cls:Sheet}}}}};
  const ownerHooks=hooks(),paths=new Map(),ownerGame={...game,user:owner},libWrapper={register(_module,path,handler){assert.equal(paths.has(path),false);paths.set(path,handler);},unregister(_module,path){paths.delete(path);}};
  const stop=registerUsageEvents({game:ownerGame,Hooks:ownerHooks,fromUuid,libWrapper,canvas:null,resolveAction:party.resolveAction,requiresActualUse:(item,action)=>party.requiresActualUse?.(item,action)??false,tracksFrequency:item=>item.sourceId===PARTY_SOURCES.clue,executeUsage:()=>assert.fail('owner must not dispatch GM Party usage')});t.after(stop);
  globalThis.getDocumentClass=()=>({async create(data){const item=docs.get(data.flags.pf2e.origin.uuid),message=card(item);Object.assign(message,structuredClone(data),{author:owner});await ownerHooks.emit('preCreateChatMessage',message);return receive(message);}});
  const pay=async(item)=>{const changes={'system.frequency.value':0},options={};await ownerHooks.emit('preUpdateItem',item,changes,options,owner.id);item.system.frequency.value=0;writes.nativeFrequency++;await Hooks.emit('updateItem',item,changes,options,owner.id);await ownerHooks.emit('updateItem',item,changes,options,owner.id);};
  const build=(item,options={})=>paths.get(`CONFIG.PF2E.Item.documentClasses.${item.type}.prototype.toMessage`).call(item,async()=>card(item,{nativeUse:options.actualUse===true,input:false}),null,options);
  const use=async(entry,kind,{paid=true}={})=>{
   const item=source.items.get(kind),native=async()=>{if(paid&&item.system.frequency)await pay(item);return build(item,entry==='HUD'?{actualUse:true}:{});};
   if(entry==='HUD')return native();
   if(entry==='hotbar')return paths.get('game.pf2e.rollItemMacro')(native,item.uuid,null);
   const app=new Sheet();app.actor=source;const handlers=paths.get('CONFIG.Actor.sheetClasses.character["pf2e.Native"].cls.prototype.activateClickListener').call(app,()=>({'use-action':native}));
   return handlers['use-action']({}, {closest:()=>({dataset:{itemId:item.id}})});
  };
  return {use,pay,display:kind=>build(source.items.get(kind))};
 }
 return {game,party,source,recipient,origin,target,card,receive,raiseShield,ownerEntrances,dispatches,errors,writes};
}

for(const kind of Object.keys(names))for(const targeted of [false,true])test(`${names[kind]} display card with ${targeted?'one other':'no'} target does not claim or settle a usage`,async t=>{
 const f=fixture(t),item=f.source.items.get(kind);await f.raiseShield();const message=f.card(item,{targets:targeted?[f.target.uuid]:[]});await f.receive(message);
 assert.equal(f.dispatches.length,0);assert.equal(message.flags[ID].usage,undefined);assert.equal(f.errors.length,0);assert.deepEqual(f.writes,{usage:0,frequency:0,nativeFrequency:0,effects:0,actor:0,cast:0});if(kind==='clue')assert.equal(item.system.frequency.value,1);
});

for(const kind of ['clue','anoint'])test(`showing ${names[kind]} after Raise Shield does not replace the action before real Devoted Guardian Use`,async t=>{
 const f=fixture(t);await f.raiseShield();await f.receive(f.card(f.source.items.get(kind)));const message=f.card(f.source.items.get('guardian'),{actualUse:true,targets:[f.target.uuid]});await f.receive(message);
 assert.equal(f.dispatches.length,1);assert.equal(message.flags[ID].usage.status,'done');assert.equal(f.errors.length,0);assert.equal(f.writes.effects,1);
});

for(const kind of Object.keys(names))test(`actual ${names[kind]} Use with another target settles once even if its create event is duplicated`,async t=>{
 const f=fixture(t);await f.raiseShield();const message=f.card(f.source.items.get(kind),{actualUse:true,targets:[f.target.uuid]});await Promise.all([f.receive(message),f.receive(message)]);
 assert.equal(f.dispatches.length,1);assert.equal(message.flags[ID].usage.status,'done');assert.equal(f.errors.length,0);assert.equal(f.writes.effects,1);assert.equal(f.writes.frequency,kind==='clue'?1:0);
});

for(const targets of ['missing','self','multiple'])test(`actual Party Use still rejects ${targets} targets and identifies its source ability`,async t=>{
 const f=fixture(t),item=f.source.items.get('anoint'),targetUuids=targets==='missing'?[]:targets==='self'?[f.origin.uuid]:[f.origin.uuid,f.target.uuid],message=f.card(item,{actualUse:true,targets:targetUuids});await f.receive(message);
 assert.equal(f.dispatches.length,1);assert.equal(message.flags[ID].usage.status,'error');assert.equal(f.errors.length,1);assert.ok(f.errors[0].message.startsWith(targetError));assert.match(f.errors[0].message,/能力：符血点化/);assert.equal(f.writes.effects,0);assert.equal(f.writes.frequency,0);
});

for(const entry of ['native-sheet','HUD','hotbar'])test(`player ${entry} Party Use retains the original actual-use boundary`,async t=>{
 const f=fixture(t),entrances=f.ownerEntrances(),message=await entrances.use(entry,'anoint');assert.equal(f.dispatches.length,1);assert.equal(message.flags[ID].usage.status,'done');assert.equal(f.writes.effects,1);assert.equal(f.errors.length,0);assert.deepEqual(message.flags[ID].usageInput.targetUuids,[f.target.uuid]);
});

test('a Clue In display card leaves its native paid receipt available for one actual HUD Use without a second frequency deduction',async t=>{
 const f=fixture(t),entrances=f.ownerEntrances(),item=f.source.items.get('clue');await entrances.pay(item);const display=await entrances.display('clue');assert.equal(f.dispatches.length,0);assert.equal(display.flags[ID].usage,undefined);assert.equal(item.system.frequency.value,0);assert.equal(f.writes.frequency,0);assert.equal(f.errors.length,0);
 const use=await entrances.use('HUD','clue',{paid:false});await f.receive(use);assert.equal(f.dispatches.length,1);assert.equal(use.flags[ID].usage.status,'done');assert.ok(use.flags[ID].usage.frequencyReceipt);assert.equal(use.flags[ID].usage.frequencyReceipt.before,1);assert.equal(use.flags[ID].usage.frequencyReceipt.after,0);assert.equal(f.writes.nativeFrequency,1);assert.equal(f.writes.frequency,0);assert.equal(f.writes.effects,1);assert.equal(item.system.frequency.value,0);assert.equal(f.errors.length,0);
});

test('a paid actual Clue In Use without another target returns only its undelivered original frequency payment',async t=>{
 const f=fixture(t),entrances=f.ownerEntrances(),item=f.source.items.get('clue');f.game.users.get('owner').targets.clear();const message=await entrances.use('HUD','clue');assert.equal(f.dispatches.length,1);assert.equal(message.flags[ID].usage.status,'error');assert.equal(f.writes.nativeFrequency,1);assert.equal(f.writes.frequency,1);assert.equal(item.system.frequency.value,1);assert.equal(f.writes.effects,0);assert.equal(f.errors.length,1);assert.ok(f.errors[0].message.startsWith(targetError));assert.match(f.errors[0].message,/能力：线索指引/);
});

for(const name of ['Arbalest','Needle Darts'])for(const type of ['attack-roll','damage-roll'])test(`native ${name} ${type} card cannot dispatch Party usage`,async t=>{
 const f=fixture(t),item={id:name,uuid:`${f.source.uuid}.Item.${name}`,actor:f.source,type:name==='Arbalest'?'weapon':'spell',sourceId:`Compendium.pf2e.${name==='Arbalest'?'equipment':'spells'}-srd.Item.original`,system:{traits:{otherTags:[]}}};f.source.items.set(item.id,item);const message=f.card(item,{roll:true,context:{type,target:{token:f.target.uuid}}});await f.receive(message);
 assert.equal(f.dispatches.length,0);assert.equal(message.flags[ID].usage,undefined);assert.equal(f.errors.length,0);assert.deepEqual(f.writes,{usage:0,frequency:0,nativeFrequency:0,effects:0,actor:0,cast:0});
});

test('Imperial blood magic keeps its existing native-cast payment path without requiring these Party Use routes',async t=>{
 const f=fixture(t);f.source.items.set('imperial',{sourceId:PARTY_SOURCES.imperial});f.source.getRollOptions=()=>['blood-magic:imperial'];const spell={id:'spell',uuid:`${f.source.uuid}.Item.spell`,type:'spell',actor:f.source,sourceId:'Compendium.pf2e.spells-srd.Item.original',system:{traits:{otherTags:['blood-magic-spell']}},getOriginData:()=>({rollOptions:[]})};f.source.items.set(spell.id,spell);const message=f.card(spell);message.item.actor.items.set(spell.id,spell);
 // The native cast path is independently authenticated by ensurePaid.
 f.game.messages.set(message.id,message);
 const action=f.party.resolveAction(spell);assert.equal(action,'party:imperial');assert.equal(f.party.requiresActualUse?.(spell,action)??false,false);await f.party.executeUsage({actor:f.source,item:spell,message,user:f.game.users.get('owner'),action});assert.equal(f.writes.cast,1);assert.equal(f.writes.effects,1);
});

const negativeWorldTime=-3703997212;
for(const cooldown of ['absent',null,negativeWorldTime-1])test(`paid native Clue In Use at a negative world time accepts ${cooldown===null?'null':cooldown==='absent'?'absent':'expired negative'} cooldown`,async t=>{
 const f=fixture(t),item=f.source.items.get('clue');f.game.time.worldTime=negativeWorldTime;if(cooldown!=='absent')f.source.flags[ID]={party:{clueUntil:cooldown}};
 const message=await f.ownerEntrances().use('native-sheet','clue');assert.equal(message.flags[ID].usage.status,'done');assert.equal(f.dispatches.length,1);assert.equal(f.errors.length,0);assert.equal(f.writes.nativeFrequency,1);assert.equal(f.writes.frequency,0);assert.equal(f.writes.effects,1);assert.equal(item.system.frequency.value,0);assert.equal(f.source.flags[ID].party.clueUntil,negativeWorldTime+600);assert.equal(message.flags[ID].usage.frequencyReceipt.before,1);assert.equal(message.flags[ID].usage.frequencyReceipt.after,0);
});

for(const cooldown of [0,negativeWorldTime+1])test(`paid native Clue In Use still rejects a future ${cooldown===0?'zero':'negative'} absolute expiry`,async t=>{
 const f=fixture(t),item=f.source.items.get('clue');f.game.time.worldTime=negativeWorldTime;f.source.flags[ID]={party:{clueUntil:cooldown}};
 const message=await f.ownerEntrances().use('native-sheet','clue');assert.equal(message.flags[ID].usage.status,'error');assert.equal(f.dispatches.length,1);assert.match(f.errors[0].message,/尚未结束10分钟冷却/);assert.equal(f.writes.nativeFrequency,1);assert.equal(f.writes.frequency,1);assert.equal(f.writes.effects,0);assert.equal(item.system.frequency.value,1);assert.equal(f.source.flags[ID].party.clueUntil,cooldown);
});

test('zero Clue In expiry remains active before zero and restores frequency and deletes the flag exactly at zero',async t=>{
 const f=fixture(t),item=f.source.items.get('clue');item.system.frequency.value=0;f.source.flags[ID]={party:{clueUntil:0}};f.game.time.worldTime=-1;await f.party.maintain(f.source);
 assert.equal(item.system.frequency.value,0);assert.equal(f.source.flags[ID].party.clueUntil,0);assert.equal(f.writes.frequency,0);assert.equal(f.writes.actor,0);
 f.game.time.worldTime=0;await f.party.maintain(f.source);assert.equal(item.system.frequency.value,1);assert.equal(Object.hasOwn(f.source.flags[ID].party,'clueUntil'),false);assert.equal(Object.hasOwn(f.source.flags[ID].party,'-=clueUntil'),false);assert.equal(f.writes.frequency,1);assert.equal(f.writes.actor,1);
});

test('expired negative Clue In expiry restores frequency and removes its actual Foundry flag',async t=>{
 const f=fixture(t),item=f.source.items.get('clue');item.system.frequency.value=0;f.source.flags[ID]={party:{clueUntil:negativeWorldTime-1}};f.game.time.worldTime=negativeWorldTime;await f.party.maintain(f.source);
 assert.equal(item.system.frequency.value,1);assert.equal(Object.hasOwn(f.source.flags[ID].party,'clueUntil'),false);assert.equal(Object.hasOwn(f.source.flags[ID].party,'-=clueUntil'),false);assert.equal(f.writes.frequency,1);assert.equal(f.writes.actor,1);
});
