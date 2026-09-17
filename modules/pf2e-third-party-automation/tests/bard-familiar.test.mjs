import test from 'node:test';
import assert from 'node:assert/strict';
import {MODULE_ID as ID} from '../scripts/rules.mjs';
import {createBardFamiliarProvider,BARD_FAMILIAR_SOURCES as S} from '../scripts/bard-familiar.mjs';
import {registerUsageEvents} from '../scripts/usage-events.mjs';

function assign(doc,changes){for(const[path,value]of Object.entries(changes)){const parts=path.split('.');let obj=doc;for(const part of parts.slice(0,-1))obj=obj[part]??={};obj[parts.at(-1)]=structuredClone(value);}}
class Modifier {constructor(data){Object.assign(this,data);}clone(){return new Modifier({...this});}}
class CheckModifier {constructor(slug,check,extra=[],options=new Set()){this.slug=slug;this.modifiers=[...check.modifiers.map(m=>m.clone()),...extra];this.options=options;this.totalModifier=this.modifiers.filter(m=>m.type!=='circumstance').reduce((n,m)=>n+m.modifier,0)+Math.max(0,...this.modifiers.filter(m=>m.type==='circumstance').map(m=>m.modifier));}}
function fixture({focus=1,max=2,rank=2,confirm=async()=> 'yes'}={}){
 const user={id:'player'},gm={id:'gm',isGM:true},users=new Map([[user.id,user],[gm.id,gm]]);users.activeGM=gm;
 const mk=(id,type)=>({id,uuid:`Actor.${id}`,type,flags:{},system:{},items:new Map(),canAct:true,isDead:false,updates:[],testUserPermission:u=>u?.id===user.id||u?.isGM,async update(changes){this.updates.push(changes);assign(this,changes);return this;}});
 const master=mk('master','character'),familiar=mk('familiar','familiar');master.system.resources={focus:{value:focus,max}};master.skills={performance:{rank}};master.getStatistic=slug=>master.skills[slug];familiar.system.master={id:master.id};Object.defineProperty(familiar,'master',{get:()=>game.actors.get(familiar.system.master.id)});
 const makeItem=(id,source)=>({id,uuid:`${familiar.uuid}.Item.${id}`,type:'action',sourceId:source,actor:familiar,flags:{},system:{category:'familiar'},updates:[],async update(changes){this.updates.push(changes);assign(this,changes);return this;}});
 const focusItem=makeItem('focus',S.focus),accompanist=makeItem('accompanist',S.accompanist);focusItem.system.frequency={value:1,max:1,per:'day'};familiar.items.set('focus',focusItem);familiar.items.set('accompanist',accompanist);
 const docs=new Map([master,familiar,focusItem,accompanist].map(d=>[d.uuid,d]));const game={user:gm,users,actors:new Map([[master.id,master],[familiar.id,familiar]]),messages:new Map(),pf2e:{Modifier,CheckModifier}};
 const callbacks=new Map(),Hooks={on(name,fn){const all=callbacks.get(name)??[];all.push(fn);callbacks.set(name,all);return fn;},off(name,fn){callbacks.set(name,(callbacks.get(name)??[]).filter(f=>f!==fn));}};
 const emit=(name,...args)=>{for(const fn of callbacks.get(name)??[])fn(...args);};
 const errors=[],prompts=[];const options={game,fromUuid:async uuid=>docs.get(uuid),confirm:async context=>{prompts.push(context);return confirm(context);},onError:error=>errors.push(error),randomId:()=>`payment-${++serial}`};let serial=0;
 const provider=createBardFamiliarProvider(options);provider.register({Hooks});
 function pay(id='receipt-1'){
  const changes={'system.frequency.value':0},updateOptions={};emit('preUpdateItem',focusItem,changes,updateOptions,user.id);
  const receipt=updateOptions[ID]?.frequencyReceipt??{id,itemUuid:focusItem.uuid,userId:user.id,before:1,after:0,createdAt:100};updateOptions[ID]={...updateOptions[ID],frequencyReceipt:receipt};assign(focusItem,changes);emit('updateItem',focusItem,changes,updateOptions,user.id);
  const message={id:`message-${receipt.id}`,uuid:`ChatMessage.message-${receipt.id}`,author:user,speaker:{actor:familiar.id},rolls:[],flags:{pf2e:{origin:{uuid:focusItem.uuid,actor:familiar.uuid,type:'action',rollOptions:['origin:action:slug:use-action']}},[ID]:{usageInput:{actualUse:true,frequencyReceiptId:receipt.id}}},async update(changes){assign(this,changes);return this;}};game.messages.set(message.id,message);docs.set(message.uuid,message);
  return {actor:familiar,item:focusItem,message,user,action:'bard-familiar:focus',frequencyReceipt:receipt};
 }
 const check=new CheckModifier('performance',{modifiers:[new Modifier({slug:'base',modifier:10,type:'untyped'})]});const context={actor:master,type:'skill-check',domains:['all','skill-check','performance'],options:new Set(['skill:performance'])};
 return {game,user,gm,master,familiar,focusItem,accompanist,provider,options,Hooks,callbacks,emit,errors,prompts,docs,pay,check,context};
}

test('routes only exact owned familiar Focus source and reuses existing frequency policy',()=>{
 const f=fixture();assert.equal(f.provider.resolveAction(f.focusItem),'bard-familiar:focus');assert.equal(f.provider.requiresActualUse(f.focusItem),true);assert.equal(f.provider.tracksFrequency(f.focusItem),true);
 assert.equal(f.provider.resolveAction(f.accompanist),undefined);assert.equal(f.provider.resolveAction({...f.focusItem,sourceId:'fake',name:'Familiar Focus'}),undefined);
 f.focusItem.name='renamed';assert.equal(f.provider.resolveAction(f.focusItem),'bard-familiar:focus');
});
test('restores one to the native master, atomically with receipt, without a second payment',async()=>{
 const f=fixture(),ctx=f.pay();assert.match(await f.provider.executeUsage(ctx),/1/);assert.equal(f.master.system.resources.focus.value,2);assert.equal(f.familiar.updates.length,0);assert.equal(f.focusItem.system.frequency.value,0);assert.equal(f.focusItem.updates.length,0);
 assert.equal(f.master.updates.length,1);assert.equal(f.master.updates[0]['system.resources.focus.value'],2);assert(Object.keys(f.master.updates[0]).some(p=>p.includes('bardFamiliar.focus.')));
});
test('two providers serialize the same paid receipt and persist duplicate protection across recreation',async()=>{
 const f=fixture(),ctx=f.pay(),second=createBardFamiliarProvider(f.options);await Promise.all([f.provider.executeUsage(ctx),second.executeUsage(ctx)]);assert.equal(f.master.updates.length,1);
 f.focusItem.system.frequency.value=1;f.master.system.resources.focus.value=0;await createBardFamiliarProvider(f.options).executeUsage(ctx);assert.equal(f.master.system.resources.focus.value,0);assert.equal(f.master.updates.length,1);
});
test('uncertain successful actor update cannot add another Focus point on retry',async()=>{
 const f=fixture({focus:0}),ctx=f.pay(),update=f.master.update.bind(f.master);f.master.update=async changes=>{await update(changes);throw Error('reply lost');};await assert.rejects(f.provider.executeUsage(ctx),/reply lost/);await createBardFamiliarProvider(f.options).executeUsage(ctx);assert.equal(f.master.system.resources.focus.value,1);assert.equal(f.master.updates.length,1);
});
test('full pool is blocked before Use; sheet capture stops Toolbelt before payment',()=>{
 const f=fixture({focus:2});assert.throws(()=>f.provider.beforeUse(f.focusItem,f.user),/已满/);let listener,stopped=0;const button={closest:()=>({dataset:{itemId:'focus'}})},element={addEventListener:(_name,fn)=>{listener=fn;},removeEventListener(){}};
 f.emit('renderFamiliarSheetPF2e',{actor:f.familiar},element);listener({target:{closest:()=>button},preventDefault:()=>stopped++,stopImmediatePropagation:()=>stopped++});assert.equal(stopped,2);assert.equal(f.focusItem.system.frequency.value,1);assert.equal(f.errors.length,1);
});
test('full-pool race refunds only its still-current payment and persists once',async()=>{
 const f=fixture(),ctx=f.pay();f.master.system.resources.focus.value=2;assert.match(await f.provider.executeUsage(ctx),/退回/);assert.equal(f.focusItem.system.frequency.value,1);assert.equal(f.focusItem.updates.length,1);assert.equal(f.master.updates.length,0);
 assert(Object.keys(f.focusItem.updates[0]).some(p=>p.includes('bardFamiliar.refunds.')));await createBardFamiliarProvider(f.options).executeUsage(ctx);assert.equal(f.focusItem.updates.length,1);
});
test('refund cannot overwrite a later payment even if its frequency happens to be zero again',async()=>{
 const f=fixture(),old=f.pay('old');f.focusItem.system.frequency.value=1;f.pay('new');f.master.system.resources.focus.value=2;await assert.rejects(f.provider.executeUsage(old),/付款|次数/);assert.equal(f.focusItem.system.frequency.value,0);assert.equal(f.focusItem.updates.length,0);
});
test('full-pool refund with lost reply stays idempotent',async()=>{
 const f=fixture({focus:2}),ctx=f.pay(),update=f.focusItem.update.bind(f.focusItem);f.focusItem.update=async changes=>{await update(changes);throw Error('reply lost');};await assert.rejects(f.provider.executeUsage(ctx),/reply lost/);await createBardFamiliarProvider(f.options).executeUsage(ctx);assert.equal(f.focusItem.updates.length,1);
});
for(const[reason,mutate]of[
 ['inactive GM',f=>f.game.user=f.user],['wrong source',f=>f.focusItem.sourceId='wrong'],['source deleted',f=>f.familiar.items.delete('focus')],['master reassigned',f=>f.familiar.system.master.id='absent'],['master not owned',f=>f.master.testUserPermission=()=>false],['familiar not owned',f=>f.familiar.testUserPermission=()=>false],['cannot act',f=>f.familiar.canAct=false],['dead',f=>f.familiar.isDead=true],['no native focus pool',f=>f.master.system.resources.focus.max=0],['invalid pool',f=>f.master.system.resources.focus.value=-1],['display card',(_f,c)=>{c.message.flags[ID].usageInput.actualUse=false;c.message.flags.pf2e.origin.rollOptions=[];}],['copied card',(_f,c)=>c.message={...c.message,id:'copy',uuid:'ChatMessage.copy'}],['wrong author',(_f,c)=>c.message.author={id:'other'}],['wrong origin',(_f,c)=>c.message.flags.pf2e.origin.uuid='wrong'],['wrong receipt',(_f,c)=>c.frequencyReceipt={...c.frequencyReceipt,id:'other'}],['forged receipt',(_f,c)=>c.frequencyReceipt={...c.frequencyReceipt,before:2}],['no observed payment',f=>delete f.focusItem.flags[ID].bardFamiliar.payment],
])test(`Focus rejects ${reason} without changing resources`,async()=>{const f=fixture(),ctx=f.pay();mutate(f,ctx);await assert.rejects(f.provider.executeUsage(ctx));assert.equal(f.master.updates.length,0);assert.equal(f.focusItem.updates.length,0);});
test('preflight allows normal Focus and ignores unrelated items',()=>{const f=fixture();assert.equal(f.provider.beforeUse(f.focusItem,f.user),true);assert.equal(f.provider.beforeUse(f.accompanist,f.user),true);f.focusItem.system.frequency.value=0;assert.throws(()=>f.provider.beforeUse(f.focusItem,f.user),/次数/);});
test('real UsageEvents tracker and hook order bind the native payment and dispatch only once',async()=>{
 const f=fixture();let dispatches=0;const unregister=registerUsageEvents({...f.options,Hooks:f.Hooks,libWrapper:null,canvas:null,resolveAction:f.provider.resolveAction,requiresActualUse:f.provider.requiresActualUse,tracksFrequency:f.provider.tracksFrequency,executeUsage:context=>{dispatches++;return f.provider.executeUsage(context);}});
 const ctx=f.pay();assert.notEqual(ctx.frequencyReceipt.id,'receipt-1');const receive=f.callbacks.get('createChatMessage')[0];await Promise.all([receive(ctx.message,{},f.user.id),receive(ctx.message,{},f.user.id)]);
 assert.equal(ctx.message.flags[ID].usage.status,'done');assert.equal(dispatches,1);assert.equal(f.master.system.resources.focus.value,2);assert.equal(f.focusItem.system.frequency.value,0);assert.equal(f.errors.length,0);unregister();
});
test('real tracker observes daily recharge before the second payment and denies the old refund',async()=>{
 const f=fixture();const unregister=registerUsageEvents({...f.options,Hooks:f.Hooks,libWrapper:null,canvas:null,resolveAction:f.provider.resolveAction,tracksFrequency:f.provider.tracksFrequency,executeUsage:f.provider.executeUsage});const old=f.pay();
 f.focusItem.system.frequency.value=1;f.emit('updateItem',f.focusItem,{'system.frequency.value':1},{},f.user.id);const fresh=f.pay();f.master.system.resources.focus.value=2;
 await assert.rejects(f.provider.executeUsage(old),/付款/);assert.equal(f.focusItem.system.frequency.value,0);assert.match(await f.provider.executeUsage(fresh),/退回/);assert.equal(f.focusItem.system.frequency.value,1);unregister();
});
test('missing native canAct is never assumed to permit Focus or Accompanist',async()=>{
 const f=fixture(),ctx=f.pay();delete f.familiar.canAct;assert.throws(()=>f.provider.beforeUse(f.focusItem,f.user),/行动/);await assert.rejects(f.provider.executeUsage(ctx),/行动/);await f.provider.interceptCheck(c=>assert.equal(c,f.check),f.check,f.context);assert.equal(f.prompts.length,0);
});
test('source master reassignment to another owned character cannot redirect an already paid Focus',async()=>{
 const f=fixture(),ctx=f.pay(),other={...f.master,id:'other',uuid:'Actor.other'};f.game.actors.set(other.id,other);f.familiar.system.master.id=other.id;await assert.rejects(f.provider.executeUsage(ctx),/付款/);assert.equal(f.master.updates.length,0);
});

test('any native Performance check gets conditional +1 circumstance without mutating original check',async()=>{
 const f=fixture();f.game.user=f.user;let received;const result=await f.provider.interceptCheck((...args)=>{received=args;return 'native';},f.check,f.context,'event','callback');assert.equal(result,'native');assert.equal(received[0].totalModifier,11);assert.equal(f.check.modifiers.length,1);assert.equal(received[1],f.context);assert.deepEqual(received.slice(2),['event','callback']);assert.equal(f.prompts.length,1);assert.match(JSON.stringify(f.prompts),/身边/);assert.match(JSON.stringify(f.prompts),/行动/);assert.doesNotMatch(JSON.stringify(f.prompts),/听见|auditory|Perform动作/);assert.equal(received[0].modifiers.at(-1).type,'circumstance');
});
test('native master Performance proficiency raises Accompanist to +2',async()=>{const f=fixture({rank:3});await f.provider.interceptCheck(c=>assert.equal(c.totalModifier,12),f.check,f.context);});
test('native stacking retains a stronger circumstance bonus and other status modifiers',async()=>{
 const f=fixture();f.check.modifiers.push(new Modifier({slug:'other',modifier:3,type:'circumstance'}),new Modifier({slug:'status',modifier:1,type:'status'}));await f.provider.interceptCheck(c=>assert.equal(c.totalModifier,14),f.check,f.context);
});
for(const[reason,mutate]of[
 ['another skill',f=>{f.check.slug='diplomacy';f.context.domains=['diplomacy'];}],['attack roll',f=>f.context.type='attack-roll'],['wrong ability source',f=>f.accompanist.sourceId='fake'],['familiar cannot act',f=>f.familiar.canAct=false],['dead familiar',f=>f.familiar.isDead=true],['wrong native master',f=>f.familiar.system.master.id='other'],['unowned master',f=>f.master.testUserPermission=()=>false],['native reroll',f=>f.context.isReroll=true],
])test(`Accompanist bypasses ${reason}`,async()=>{const f=fixture();mutate(f);await f.provider.interceptCheck(c=>assert.equal(c,f.check),f.check,f.context);assert.equal(f.prompts.length,0);});
test('declined or closed condition confirmation keeps the native check intact',async()=>{for(const value of ['no',null]){const f=fixture({confirm:async()=>value});await f.provider.interceptCheck(c=>assert.equal(c,f.check),f.check,f.context);assert.equal(f.prompts.length,1);}});
test('ability removal while confirming cannot leave a bonus on this roll',async()=>{let f;f=fixture({confirm:async()=>{f.familiar.items.delete('accompanist');return 'yes';}});await f.provider.interceptCheck(c=>assert.equal(c,f.check),f.check,f.context);});
test('native contextual clone resolves the current master and one repeated middleware adds only once',async()=>{
 const f=fixture();f.context.actor={...f.master};await f.provider.interceptCheck((check,context)=>f.provider.interceptCheck(c=>assert.equal(c.totalModifier,11),check,context),f.check,f.context);assert.equal(f.prompts.length,1);
});
