import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';
import {createBob2RuleLinks,BOB2_RULE_SOURCES} from '../scripts/bob2-rule-links.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

const bundle=process.env.PF2E_NATIVE_861_BUNDLE,source=bundle?readFileSync(bundle,'utf8'):null;
const castPath='CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast';
const consumePath='CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.consume';
function nativeCast(){
 const entry=source.indexOf('SpellcastingEntryPF2e = class'),start=source.indexOf('\tasync cast(e, t = {}) {',entry),end=source.indexOf('\n\tasync consume(',start);
 assert.ok(entry>0&&start>entry&&end>start,'the PF2e 8.6.1 native cast method is required');
 return Function(`return ({${source.slice(start,end)}}).cast`)();
}
function nativeConsume(game){
 const entry=source.indexOf('SpellcastingEntryPF2e = class'),start=source.indexOf('\tasync consume(e, t, n) {',entry),end=source.indexOf('\n\tasync addSpell(',start);
 assert.ok(entry>0&&start>entry&&end>start);
 return vm.runInNewContext(`Number.prototype.between=function(min,max){return this>=min&&this<=max};({${source.slice(start,end)}}).consume`,{game,_loc:key=>key,ordinalString:String});
}
function castFromSheet(f,dataset={}){
 const prefix='t[`cast-spell`] = ',start=source.indexOf(prefix),end=source.indexOf(', t[`browse-spells`] =',start);assert.ok(start>0&&end>start);
 const collection=Object.assign(new Map([[f.spell.id,f.spell]]),{entry:{cast(item,options){f.sheetOptions=options;return f.cast(item,options)}}});
 const n={spellcasting:{collections:new Map([[f.entry.id,collection]])}};
 const handler=vm.runInNewContext(`Number.prototype.between=function(min,max){return this>=min&&this<=max};(${source.slice(start+prefix.length,end)})`,{n,htmlClosest:target=>target});
 return handler({target:{dataset:{itemId:f.spell.id,entryId:f.entry.id,castRank:'3',...dataset}}});
}
const set=(document,changes)=>{for(const[path,value]of Object.entries(changes)){let object=document;const keys=path.split('.');for(const key of keys.slice(0,-1))object=object[key]??={};object[keys.at(-1)]=structuredClone(value)}};
function fixture({detached=false}={}){
 const gm={id:'gm',active:true,targets:new Set()},wrappers=new Map(),docs=new Map(),calls={consume:0,cards:0};
 const game={world:{id:'bob-2'},system:{id:'pf2e'},user:gm,users:Object.assign(new Map([['gm',gm]]),{activeGM:gm}),actors:new Map(),messages:new Map(),time:{worldTime:1}};
 const actor={id:'bard',uuid:'Actor.bard',type:'character',items:new Map(),flags:{},system:{resources:{focus:{value:2,max:3}}},isOfType:(...types)=>types.includes('character'),testUserPermission:user=>user===gm,async update(changes){set(this,changes);return this}};
 const entry={id:'entry',uuid:'Actor.bard.Item.entry',type:'spellcastingEntry',actor,spells:{},system:{prepared:{value:'focus'},slots:{slot3:{value:2}}},get isPrepared(){return this.system.prepared.value==='prepared'},get isFocusPool(){return this.system.prepared.value==='focus'},get isSpontaneous(){return this.system.prepared.value==='spontaneous'},async update(changes){set(this,changes);return this}};
 const spell={id:'overture',uuid:'Actor.bard.Item.overture',type:'spell',sourceId:BOB2_RULE_SOURCES.overture,actor,spellcasting:entry,rank:3,atWill:true,isCantrip:true,system:{location:{value:entry.id},cast:{focusPoints:0},time:{value:'1'}},computeCastRank:rank=>rank};
 const other={...spell,id:'other',uuid:'Actor.bard.Item.other',sourceId:'Compendium.test.spells.Item.other'};
 game.actors.set(actor.id,actor);for(const doc of [actor,entry,spell,other]){docs.set(doc.uuid,doc);if(doc!==actor)actor.items.set(doc.id,doc)}
 const casts=createNativeCastEvents({game,fromUuid:async uuid=>docs.get(uuid),messageTimeoutMs:50});
 createBob2RuleLinks({game,castEvents:casts});
 casts.register({libWrapper:{register:(_id,path,fn)=>wrappers.set(path,fn)}});
 const consume=nativeConsume(game);
 entry.consume=(item,rank,slotId,capability)=>wrappers.get(consumePath).call(entry,async(...args)=>{calls.consume++;return consume.call(entry,...args)},item,rank,slotId,capability);
 for(const item of [spell,other])item.toMessage=async(_event,options)=>{
  if(detached)await new Promise(resolve=>setImmediate(resolve));
  const captured=casts.captureUsage(item,{options}),id=`card${++calls.cards}`;
  const message={id,uuid:'ChatMessage.'+id,author:gm,rolls:[],speaker:{actor:actor.id},flags:{[ID]:captured??{},pf2e:{origin:{actor:actor.uuid,uuid:item.uuid,castRank:options.data?.castRank}}},async update(changes){set(this,changes);return this}};
  game.messages.set(id,message);docs.set(message.uuid,message);
  casts.captureMessageOutcome(item,options,{message,castNonce:captured?.nativeCast?.id});return message;
 };
 const native=nativeCast();let pending;
 const cast=(item=spell,options={rank:3})=>wrappers.get(castPath).call(entry,detached?(...args)=>{pending=native.call(entry,...args)}:native.bind(entry),item,options);
 return {game,actor,entry,spell,other,casts,calls,cast,pending:()=>pending,receipts:()=>actor.flags[ID]?.nativeCasts??[]};
}

for(const detached of [false,true])test(`matched at-will Cast binds a zero-cost receipt despite native consume bypass (${detached?'detached':'awaited'})`,{skip:!source},async()=>{
 const f=fixture({detached});await f.cast();await f.pending();
 const message=f.game.messages.get('card1'),proof=message.flags[ID].nativeCast;
 assert.ok(proof,'the actual Cast needs its native receipt before provider dispatch');
 assert.equal(f.receipts().length,1);assert.equal(proof.id,f.receipts()[0].id);assert.equal(proof.itemUuid,f.spell.uuid);
 assert.equal(message.flags[ID].nativeCastInput.rank,3);assert.equal(message.flags[ID].nativeCastInput.focusPoints,0);
 const context={actor:f.actor,item:f.spell,message,user:f.game.user};
 const first=await f.casts.ensurePaid(context),repeat=await f.casts.ensurePaid(context);
 assert.equal(first.messageId,message.id);assert.equal(repeat.id,first.id);assert.equal(f.receipts().length,1);
 assert.equal(f.calls.consume,0);assert.equal(f.actor.system.resources.focus.value,2);assert.equal(f.entry.system.slots.slot3.value,2);
 const displayed=await f.spell.toMessage(null,{actualCast:true,data:{castRank:3}});
 assert.equal(displayed.flags[ID].nativeCast,undefined,'actualCast options outside the live invocation cannot borrow its proof');
 assert.equal(f.receipts().length,1);
});

test('preview, silent Cast and another spell on a managed actor do not create zero-cost receipts',{skip:!source},async()=>{
 for(const options of [{rank:3,consume:false},{rank:3,message:false}]){
  const f=fixture();await f.cast(f.spell,options);assert.equal(f.receipts().length,0);assert.equal(f.calls.consume,0);
  for(const message of f.game.messages.values())assert.equal(message.flags[ID].nativeCast,undefined);
 }
 const f=fixture();await f.cast(f.other);assert.equal(f.receipts().length,0);assert.equal(f.game.messages.get('card1').flags[ID].nativeCast,undefined);
 const displayed=await f.spell.toMessage(null,{data:{castRank:3}});assert.equal(displayed.flags[ID].nativeCast,undefined);assert.equal(f.receipts().length,0);
});

test('real focus-cantrip sheet handler canonicalizes its absent slot dataset through Cast and capture',{skip:!source},async()=>{
 const f=fixture();await castFromSheet(f);assert.ok(Number.isNaN(f.sheetOptions.slotId),'native sheet passes Number(undefined)');
 const message=f.game.messages.get('card1');assert.equal(message.flags[ID].nativeCastInput.slotId,null);assert.equal(f.receipts()[0].slotId,null);
 await f.casts.ensurePaid({actor:f.actor,item:f.spell,message,user:f.game.user});assert.equal(f.calls.consume,0);assert.equal(f.actor.system.resources.focus.value,2);
});
test('real nonprepared sheet inputs share the canonical slot through ordinary native consume and receipt',{skip:!source},async()=>{
 for(const category of ['focus','spontaneous']){
  const f=fixture();f.entry.system.prepared.value=category;f.spell.atWill=false;f.spell.isCantrip=false;f.spell.isFocusSpell=category==='focus';f.spell.system.cast.focusPoints=category==='focus'?1:0;
  await castFromSheet(f);assert.ok(Number.isNaN(f.sheetOptions.slotId));
  const message=f.game.messages.get('card1'),receipt=f.receipts()[0];assert.equal(message.flags[ID].nativeCastInput.slotId,null);assert.equal(receipt.slotId,null);
  await f.casts.ensurePaid({actor:f.actor,item:f.spell,message,user:f.game.user});assert.equal(f.calls.consume,1);assert.equal(f.receipts().length,1);
  assert.equal(f.actor.system.resources.focus.value,category==='focus'?1:2);assert.equal(f.entry.system.slots.slot3.value,category==='spontaneous'?1:2);
 }
});
test('prepared and unknown entries retain invalid-slot rejection, and other malformed inputs stay rejected',{skip:!source},async()=>{
 for(const slotId of [NaN,-1,Infinity,0.5,'0']){
  const f=fixture();f.entry.system.prepared.value='prepared';f.entry.system.slots.slot3.prepared=[{id:f.spell.id,expended:false}];
  await assert.rejects(f.cast(f.spell,{rank:3,slotId}),/法术位编号无效/);assert.equal(f.receipts().length,0);assert.equal(f.calls.cards,0);
 }
 for(const slotId of [-1,Infinity,0.5,'0']){const f=fixture();await assert.rejects(f.cast(f.spell,{rank:3,slotId}),/法术位编号无效/);assert.equal(f.receipts().length,0);}
 const unknown=fixture();Object.defineProperty(unknown.entry,'isPrepared',{value:undefined});await assert.rejects(unknown.cast(unknown.spell,{rank:3,slotId:NaN}),/法术位编号无效/);
});
