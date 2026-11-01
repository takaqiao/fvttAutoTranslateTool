import test from 'node:test';
import assert from 'node:assert/strict';
import {createNativeOwnerOperations} from '../scripts/native-owner-operations.mjs';

const module=await import('../scripts/counter-performance.mjs').catch(error=>{
 if(error.code==='ERR_MODULE_NOT_FOUND')return {};
 throw error;
});
const ID='pf2e-third-party-automation',TOOL='pf2e-toolbelt';
const SOURCE='Compendium.pf2e.spells-srd.Item.WILXkjU5Yq3yw10r';
const clone=value=>structuredClone(value);
function document(data){
 return {...data,async update(changes){for(const[path,value]of Object.entries(changes)){const keys=path.split('.'),key=keys.pop();let target=this;for(const part of keys)target=target[part]??={};target[key]=clone(value);}return this;}};
}
function row(total,die=10,success='failure'){
 return {die,statistic:'will',value:total,success,private:false,rerolled:false,dosAdjustments:{},modifiers:[],notes:[],unadjustedOutcome:success,roll:JSON.stringify({class:'CheckRoll',total,evaluated:true,options:{type:'saving-throw',degreeOfSuccess:['criticalFailure','failure','success','criticalSuccess'].indexOf(success)},terms:[{class:'Die',faces:20,results:[{result:die,active:true}],total:die}]})};
}
function fixture({answers=['singing','source:null','yes','yes','confirm'],total=25,die=12,applied={},cancelRoll=false}={}){
 assert.equal(typeof module.createCounterPerformance,'function','Counter Performance provider must expose its native Cast entry');
 const gm={id:'gm',isGM:true,active:true},user={id:'owner',isGM:false,active:true};
 const actor=document({id:'bard',uuid:'Actor.bard',name:'Bard',system:{resources:{focus:{value:2}}},items:new Map(),testUserPermission:u=>['gm','owner'].includes(u?.id)});
 const item={id:'counter',uuid:'Actor.bard.Item.counter',type:'spell',sourceId:SOURCE,actor,system:{time:{value:'reaction'},traits:{value:['bard','composition','focus','fortune','mental','manipulate']}}},entry={id:'entry',actor};actor.items.set(item.id,item);
 const tokens=new Map(['a','b'].map(id=>{const targetActor={id,uuid:`Actor.${id}`,name:id};return[`Scene.scene.Token.${id}`,{id,uuid:`Scene.scene.Token.${id}`,documentName:'Token',actor:targetActor,name:id}]}));
 const helper={type:'spell',item:'Actor.enemy.Item.spell',targets:[...tokens.keys()],private:false,applied,saveVariants:{null:{dc:22,basic:true,statistic:'will',saves:{a:row(17),b:row(29,18,'success')}}}};
 const source=document({id:'source',uuid:'ChatMessage.source',blind:false,whisper:[],content:'Enemy spell',flags:{[TOOL]:{targetHelper:helper}}});
 const message=document({id:'cast',uuid:'ChatMessage.cast',author:user,speaker:{actor:actor.id},blind:false,whisper:[],rolls:[],flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid,type:'spell'}},[ID]:{nativeCast:{id:'payment',actorUuid:actor.uuid,itemUuid:item.uuid,userId:user.id}}}});
 const messages=new Map([[source.id,source],[message.id,message]]),game={world:{id:'bob-2'},user,users:new Map([[gm.id,gm],[user.id,user]]),actors:new Map([[actor.id,actor]]),messages,system:{id:'pf2e',version:'8.6.1'},modules:new Map([[TOOL,{active:true,version:'3.59.2'}]])};game.users.activeGM=gm;
 const rolls=[],payments=[],casts=[],choices=[];
 const nativeCasts={async ensurePaid(context){payments.push(context);return {id:'payment'};}};
 const runNative=async(context,request)=>{
  rolls.push(request);if(cancelRoll)return {status:'cancelled'};
  const roll={total,_evaluated:true,dice:[{faces:20,total:die}],toJSON:()=>({class:'CheckRoll',total,options:{type:'skill-check'},terms:[{class:'Die',faces:20,total:die,results:[{result:die,active:true}]}]})};
  const check=document({id:'performance',uuid:'ChatMessage.performance',author:user,speaker:{actor:actor.id},blind:false,whisper:[],flags:{pf2e:{context:{type:'skill-check',options:['check:statistic:performance']}}},rolls:[roll]});messages.set(check.id,check);return {status:'rolled',check};
 };
 let executeNative=runNative;
 const provider=module.createCounterPerformance({game,nativeCasts,runNative:(...args)=>executeNative(...args),fromUuid:async uuid=>tokens.get(uuid)??messages.get(uuid.split('.').at(-1)),chooseCast:async options=>{choices.push(options);return answers.shift();}});
 const ctx={actor,item,entry,user,message};
 const cast=async()=>provider.interceptCast({item,entry,options:{}},async()=>{casts.push('cast');actor.system.resources.focus.value--;message.flags[ID].counterPerformanceInput=provider.captureCast(item);return message;});
 const execute=async()=>{game.user=gm;return provider.executeUsage(ctx);};
 return {provider,actor,item,entry,user,gm,game,ctx,message,source,tokens,rolls,payments,casts,choices,cast,execute,setRunNative:fn=>executeNative=fn,runNative};
}

test('native Cast binds selected original saves, rolls Performance once and updates only the lower total',async()=>{
 const f=fixture(),before=clone(f.source.flags[TOOL].targetHelper.saveVariants.null.saves);
 await f.cast();await f.execute();await f.execute();
 assert.equal(f.actor.system.resources.focus.value,1);
 assert.equal(f.rolls.length,1);assert.equal(f.payments.length,1);
 assert.equal(f.rolls[0].statistic,'performance');assert.equal(f.rolls[0].type,'check');
 assert.ok(f.rolls[0].traits.includes('auditory'));assert.ok(f.rolls[0].options.includes('action:perform:singing'));
 const saves=f.source.flags[TOOL].targetHelper.saveVariants.null.saves;
 assert.equal(saves.a.value,25);assert.equal(saves.a.success,'success');
 assert.equal(saves.a.roll,before.a.roll);assert.equal(saves.a.die,before.a.die);
 assert.deepEqual(saves.b,before.b);
 const record=f.source.flags[ID].counterPerformance.cast;
 assert.equal(record.performanceMessageUuid,'ChatMessage.performance');
 assert.equal(record.results[0].targetUuid,'Scene.scene.Token.a');
 assert.equal(record.results[0].original.roll,before.a.roll);
 assert.equal(record.results[0].effectiveTotal,25);
 assert.equal(record.results[1].effectiveTotal,29);
 assert.equal(f.message.flags[ID].counterPerformance.status,'done');
});

test('closing pre-cast choices or declining the trigger never calls native Cast',async()=>{
 for(const answers of [[null],['singing','source:null','yes','yes','cancel']]){
  const f=fixture({answers});await f.cast();assert.equal(f.casts.length,0);assert.equal(f.actor.system.resources.focus.value,2);assert.equal(f.provider.captureCast(f.item),undefined);
 }
});

test('original natural 20 success is not worsened by a higher Performance total',async()=>{
 const f=fixture({total:26,die:11});f.source.flags[TOOL].targetHelper.saveVariants.null.dc=20;
 f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a=row(20,20,'criticalSuccess');
 await f.cast();await f.execute();const result=f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a;
 assert.equal(result.success,'criticalSuccess');assert.equal(result.value,20);
});

test('Performance natural 20 and natural 1 use their own dice with original save adjustments',async()=>{
 for(const [total,die,want]of [[22,20,'criticalSuccess'],[32,1,'success']]){
  const f=fixture({total,die});await f.cast();await f.execute();
  assert.equal(f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a.success,want);
  assert.equal(f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a.die,10);
 }
 const f=fixture({total:18,die:20});await f.cast();await f.execute();
 assert.equal(f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a.success,'success');
});

test('visual performance uses current native variant and traits, and only explicitly selected targets change',async()=>{
 const f=fixture({answers:['dance','source:null','yes','no','confirm']});await f.cast();await f.execute();
 assert.ok(f.rolls[0].traits.includes('move'));assert.ok(f.rolls[0].traits.includes('visual'));assert.ok(!f.rolls[0].traits.includes('auditory'));
 assert.equal(f.source.flags[ID].counterPerformance.cast.results.length,1);
});

test('changing an original result during Performance stops writeback and never retries the roll',async()=>{
 const f=fixture();f.setRunNative(async(...args)=>{const result=await f.runNative(...args);f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a=row(19);return result;});
 await f.cast();await assert.rejects(f.execute(),/改变/);await f.execute();
 assert.equal(f.rolls.length,1);assert.equal(f.message.flags[ID].counterPerformance.status,'uncertain');
 assert.equal(f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a.value,19);
});

test('a second queued cast cannot roll again for original saves already resolved by another cast',async()=>{
 const f=fixture({total:5});await f.cast();const input=clone(f.message.flags[ID].counterPerformanceInput);
 await f.execute();
 const message=document({...f.message,id:'second',uuid:'ChatMessage.second',flags:{pf2e:clone(f.message.flags.pf2e),[ID]:{nativeCast:{id:'payment-2'},counterPerformanceInput:input}}});
 f.game.messages.set(message.id,message);
 await assert.rejects(f.provider.executeUsage({...f.ctx,message}),/使用过/);
 assert.equal(f.rolls.length,1);assert.equal(f.payments.length,1);
});

test('display cards and consume:false bypass settlement and never acquire a payment',async()=>{
 const f=fixture();delete f.message.flags[ID].nativeCast;
 await f.provider.interceptCast({item:f.item,entry:f.entry,options:{consume:false}},async()=>{f.casts.push('display');});
 await f.execute();assert.equal(f.rolls.length,0);assert.equal(f.payments.length,0);assert.equal(f.choices.length,0);
});

test('a same-name spell from a different source is not intercepted',async()=>{
 const f=fixture();f.item.sourceId='Compendium.other.Item.counter';let called=0;
 await f.provider.interceptCast({item:f.item,entry:f.entry,options:{}},async()=>called++);
 assert.equal(called,1);assert.equal(f.choices.length,0);assert.equal(f.provider.resolveAction(f.item),null);
});

test('other worlds retain native Cast without provider match, capture or chat hooks',async()=>{
 const f=fixture();f.game.world.id='av';let casts=0,hooks=0;
 await f.provider.interceptCast({item:f.item,options:{}},async()=>casts++);
 f.provider.register({Hooks:{on:()=>hooks++}});
 assert.equal(casts,1);assert.equal(f.choices.length,0);assert.equal(f.provider.matches(f.item),false);
 assert.equal(f.provider.resolveAction(f.item),null);assert.equal(f.provider.captureCast(f.item),undefined);assert.equal(hooks,0);
});

test('an applied target retains applied flags and gets an explicit Undo and BUFF correction notice',async()=>{
 const f=fixture({applied:{a:{0:true}}});await f.cast();await f.execute();
 assert.deepEqual(f.source.flags[TOOL].targetHelper.applied,{a:{0:true}});
 const result=f.source.flags[ID].counterPerformance.cast.results[0];assert.equal(result.damageAlreadyApplied,true);
 const html=f.provider.resultHTML(f.message);assert.match(html,/Undo/);assert.match(html,/BUFF/);assert.match(html,/source/);assert.match(html,/performance/);
});

test('original save changes while the player selects or rolls prevent overwrite and automatic retry',async()=>{
 const f=fixture();await f.cast();f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a=row(20);
 await assert.rejects(f.execute(),/改变/);assert.equal(f.rolls.length,0);
 assert.equal(f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a.value,20);
});

test('cancelling Performance after accepted Cast records the spent spell without rolling again',async()=>{
 const f=fixture({cancelRoll:true});await f.cast();const result=await f.execute();await f.execute();
 assert.equal(f.actor.system.resources.focus.value,1);assert.equal(f.rolls.length,1);assert.equal(result.status,'cancelled');
 assert.equal(f.message.flags[ID].counterPerformance.status,'cancelled');
 assert.equal(f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a.value,17);
});

test('private saves, unsupported Toolbelt versions and missing original dice cannot be selected',async()=>{
 for(const change of [f=>f.source.blind=true,f=>f.game.modules.get(TOOL).version='4.0.0',f=>f.source.flags[TOOL].targetHelper.saveVariants.null.saves={}]){
  const f=fixture();change(f);await assert.rejects(f.cast(),/原豁免|Toolbelt/);assert.equal(f.casts.length,0);
 }
});

test('render hook gives both original result and Performance links without a second roll button',async()=>{
 const f=fixture();await f.cast();await f.execute();const events=new Map();
 const stop=f.provider.register({Hooks:{on:(name,fn)=>(events.set(name,fn),name),off:name=>events.delete(name)}});
 let html='';events.get('renderChatMessageHTML')(f.message,{querySelector:()=>null,insertAdjacentHTML:(_position,value)=>html=value});
 assert.match(html,/data-counter-performance-message="source"/);assert.match(html,/data-counter-performance-message="performance"/);assert.doesNotMatch(html,/button/);stop();assert.equal(events.size,0);
});

test('result links open the exact saved native chat message',async()=>{
 const f=fixture();await f.cast();await f.execute();const hooks=new Map(),listeners=new Map(),opened=[];
 const node={dataset:{counterPerformanceMessage:'source'},addEventListener:(event,fn)=>listeners.set(event,fn)};
 const old=globalThis.CONFIG;
 try{
  class Popout {constructor(options){opened.push(options.message);}render(options){assert.deepEqual(options,{force:true});}}
  globalThis.CONFIG={ChatMessage:{popoutClass:Popout}};
  f.provider.register({Hooks:{on:(name,fn)=>hooks.set(name,fn)}});
  hooks.get('renderChatMessageHTML')(f.message,{querySelector:()=>null,insertAdjacentHTML(){},querySelectorAll:()=>[node]});
  assert.equal(typeof listeners.get('click'),'function');listeners.get('click')({preventDefault(){}});
  assert.deepEqual(opened,[f.source]);
 }finally{globalThis.CONFIG=old;}
});

test('the real owner operation runs one native Performance on the casting player and binds its durable result',async()=>{
 const f=fixture(),hooks=new Map(),ownerHandlers=new Map(),gmHandlers=new Map();let seq=0,nativeRolls=0;
 const Hooks={on(event,fn){const id=++seq;hooks.set(id,{event,fn});return id;},off(_event,id){hooks.delete(id);},call(event,...args){for(const entry of [...hooks.values()])if(entry.event===event)entry.fn(...args);}};
 const ownerGame={...f.game,user:f.user},fromUuid=async uuid=>uuid===f.actor.uuid?f.actor:uuid===f.item.uuid?f.item:f.tokens.get(uuid);
 class Message {
  constructor(data){Object.assign(this,data);if(typeof this.author==='string')this.author=f.game.users.get(this.author);}toObject(){return {...this};}
  static async create(data){const check=new Message({...data,id:'performance',uuid:'ChatMessage.performance'});f.game.messages.set(check.id,check);Hooks.call('createChatMessage',check);return check;}
 }
 f.actor.getStatistic=statistic=>{
  assert.equal(statistic,'performance');
  return {check:{roll:async parameters=>{
   nativeRolls++;assert.equal(ownerGame.user,f.user);assert.equal(parameters.createMessage,false);
   let accept;const accepted=new Promise(resolve=>accept=resolve),app={context:{options:parameters.extraRollOptions},resolve:accept};Hooks.call('renderCheckModifiersDialog',app);await app.resolve(true);assert.equal(await accepted,true);
   const roll={total:25,_evaluated:true,dice:[{faces:20,total:12}],toJSON:()=>({class:'CheckRoll',total:25,options:{type:'skill-check'},terms:[{class:'Die',faces:20,total:12}]})};
   const raw=new Message({author:f.user.id,speaker:{actor:f.actor.id},blind:false,whisper:[],rolls:[roll],flags:{pf2e:{origin:{uuid:f.item.uuid,actor:f.actor.uuid},context:{type:'skill-check',options:[...parameters.extraRollOptions,'check:statistic:performance']}}}});
   await parameters.callback(roll,undefined,raw);return roll;
  }}};
 };
 const owner=createNativeOwnerOperations({game:ownerGame,fromUuid,scope:'counter-performance'}),gm=createNativeOwnerOperations({game:f.game,fromUuid,scope:'counter-performance'});
 const ownerSocket={register:(name,fn)=>ownerHandlers.set(name,fn),executeAsUser:(name,_user,payload)=>gmHandlers.get(name).call({socketdata:{userId:f.user.id}},payload)};
 const gmSocket={register:(name,fn)=>gmHandlers.set(name,fn),executeAsUser:(name,_user,payload)=>ownerHandlers.get(name).call({socketdata:{userId:f.gm.id}},payload)};
 owner.register({Hooks,socket:ownerSocket});gm.register({Hooks,socket:gmSocket});f.setRunNative(gm.run);
 const old=globalThis.CONFIG;
 try{
  globalThis.CONFIG={ChatMessage:{documentClass:Message}};
  await f.cast();await f.execute();await f.execute();assert.equal(nativeRolls,1);
  assert.equal(f.source.flags[TOOL].targetHelper.saveVariants.null.saves.a.value,25);
  const operation=Object.values(f.message.flags[ID].nativeOwnerOperations)[0];assert.equal(operation.status,'done');assert.equal(operation.scope,'counter-performance');assert.equal(operation.userId,'owner');
  assert.equal(f.game.messages.get('performance').flags[ID].nativeOwnerOperation.activityMessageId,'cast');
 }finally{globalThis.CONFIG=old;}
});
