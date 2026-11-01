import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';

const providerModule=await import('../scripts/deaths-call.mjs').catch(error=>{
 if(error.code==='ERR_MODULE_NOT_FOUND')return {};
 throw error;
});
const ID='pf2e-third-party-automation',SOURCE='Compendium.pf2e.spells-srd.Item.HG4afO9EOGEU9bZN';
const clone=value=>structuredClone(value),deleted=Symbol('delete');
function update(target,changes){
 for(const[path,value]of Object.entries(changes)){
  const parts=path.split('.'),key=parts.pop();let parent=target;
  for(const part of parts)parent=parent[part]??={};
  if(value===deleted)delete parent[key];
  else if(value&&typeof value==='object'&&!Array.isArray(value))update(parent[key]??={},value);
  else parent[key]=clone(value);
 }
 return target;
}
const property=(object,path)=>path.split('.').reduce((value,key)=>value?.[key],object);
function nativeTempHP(){
 const path=process.env.BOB2_CURRENT_PF2E_SOURCE;assert.ok(path,'run through the fixture-aware runner');
 const source=readFileSync(path,'utf8'),start=source.indexOf('TempHPRuleElement = class extends q {'),end=source.indexOf('\n}, al =',start);
 assert.ok(start>0&&end>start,'native TempHP class boundary');
 const foundry={utils:{getProperty:property,mergeObject:(target,changes,options={})=>update(options.inplace===false?clone(target):target,changes)}};
 class Rule {resolveValue(value){return value;}test(){return true;}}
 const TempHP=Function('q','foundry','M','getTempHPSourceId','_del',`return (${source.slice(start+'TempHPRuleElement = '.length,end+2)})`)(Rule,foundry,value=>value!==null&&typeof value==='object',hp=>hp.tempSource??hp.tempsource,deleted);
 TempHP.prototype.broadcast=()=>{};
 const preStart=source.indexOf('\t\tlet c = e.system.attributes?.hp;'),preEnd=source.indexOf('\n\t\tlet l = e.system.resources?.focus;',preStart);
 assert.ok(preStart>0&&preEnd>preStart,'native temp source cleanup boundary');
 const math=Object.create(Math);math.clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
 const preUpdate=Function('getTempHPSourceId','_del','Math',`return function(e){${source.slice(preStart,preEnd)}}`)(hp=>hp.tempSource??hp.tempsource,deleted,math);
 const undoStart=source.indexOf('\tasync undoDamage(e) {'),undoEnd=source.indexOf('\n\tisLootableBy(',undoStart);
 const undo=Function('foundry',`return ({${source.slice(undoStart,undoEnd)}}).undoDamage`)(foundry);
 const createStart=source.indexOf('\t_onCreate(e, t, n) {\n\t\tif (super._onCreate(e, t, n), !(this.actor && game.user.id === n)) return;'),createEnd=source.indexOf('\n\t_onUpdate(',createStart);
 assert.ok(createStart>0&&createEnd>createStart,'native Item creation boundary');
 const itemOnCreate=game=>{
  const methods=Function('game',`return ({${source.slice(createStart,createEnd)}})`)(game);
  Object.setPrototypeOf(methods,{_onCreate(){}});return methods._onCreate;
 };
 const trackerStart=source.indexOf('EffectTracker = class {'),trackerEnd=source.indexOf('\n}, ModuleArt = class {',trackerStart);
 assert.ok(trackerStart>0&&trackerEnd>trackerStart,'native effect tracker boundary');
 const effectTracker=game=>new (Function('game','Collection',`return (${source.slice(trackerStart+'EffectTracker = '.length,trackerEnd+2)})`)(game,Map))();
 return {TempHP,preUpdate,undo,itemOnCreate,effectTracker};
}
function fixture({answers=['Scene.scene.Token.trigger','confirm'],mode='living',level=6,modifier=4,temp=0,tempSource=null,deferTempHP=false,removeEffects=false}={}){
 assert.equal(typeof providerModule.createDeathsCall,'function','Death\'s Call must expose its native Cast provider');
 const native=nativeTempHP(),hooks=new Map(),errors=[];
 const Hooks={on(name,fn){const id=Symbol(name);hooks.set(id,{name,fn});return id;},off(_name,id){hooks.delete(id);},async emit(name,...args){await Promise.all([...hooks.values()].filter(h=>h.name===name).map(h=>h.fn(...args)));}};
 const gm={id:'gm',isGM:true,active:true},user={id:'owner',isGM:false,active:true};
 const actor={id:'cleric',uuid:'Actor.cleric',type:'character',items:new Map(),flags:{},system:{resources:{focus:{value:2}},attributes:{hp:{value:20,max:30,temp,...(tempSource?{tempSource}:{})}}},testUserPermission:u=>['gm','owner'].includes(u?.id),getRollOptions:()=>[],itemTypes:{weapon:[]}};
 Object.defineProperty(actor,'_source',{get:()=>({system:clone(actor.system)})});
 Object.defineProperty(actor.itemTypes,'effect',{get:()=>[...actor.items.values()].filter(item=>item.type==='effect')});
 const pendingUpdates=[];
 actor.reset=()=>{};
 actor.update=async changes=>{
  const expanded=update({},changes);
  const commit=async()=>{if(expanded.system)native.preUpdate.call(actor,expanded);update(actor,expanded);await Hooks.emit('updateActor',actor,expanded,{},'gm');return actor;};
  if(deferTempHP&&expanded.system?.attributes?.hp?.tempSource)return new Promise(resolve=>pendingUpdates.push(async()=>resolve(await commit())));
  return commit();
 };
 const entry={id:'entry',uuid:'Actor.cleric.Item.entry',type:'spellcastingEntry',actor,attribute:'wis',statistic:{attributeModifier:{value:modifier}}};
 const item={id:'spell',uuid:'Actor.cleric.Item.spell',type:'spell',name:"斩杀号令 Death's Call",img:'icons/spell.webp',sourceId:SOURCE,actor,rank:3,spellcasting:entry,system:{location:{value:entry.id},traits:{value:['focus','cleric']},time:{value:'reaction'}}};
 actor.items.set(item.id,item);actor.items.set(entry.id,entry);
 const trigger={id:'enemy',uuid:'Actor.enemy',type:'npc',name:'Hidden creature',level,modeOfBeing:mode,system:{traits:{value:mode==='undead'?['undead']:[]}},testUserPermission:()=>false};
 const token={id:'trigger',uuid:'Scene.scene.Token.trigger',documentName:'Token',name:'Hidden token',actor:trigger,hidden:false,playersCanSeeName:false,object:{isVisible:true}};
 const message={id:'cast',uuid:'ChatMessage.cast',author:user,flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid,type:'spell'}},[ID]:{nativeCast:{id:'payment',actorUuid:actor.uuid,itemUuid:item.uuid,userId:user.id}}},async update(changes){update(this,changes);return this;}};
 const game={world:{id:'bob-2'},system:{id:'pf2e'},user,users:new Map([[user.id,user],[gm.id,gm]]),actors:new Map([[actor.id,actor],[trigger.id,trigger]]),scenes:{current:{tokens:new Map([[token.id,token]])}},messages:new Map([[message.id,message]]),time:{worldTime:100},pf2e:{settings:{tokens:{nameVisibility:true},automation:{removeEffects}},effectPanel:{refresh(){}}}};game.users.activeGM=gm;user.targets=new Set();
 const documents=new Map([[actor.uuid,actor],[item.uuid,item],[entry.uuid,entry],[token.uuid,token],[trigger.uuid,trigger]]),created=[],removed=[],payments=[],choices=[],casts=[];
 const itemOnCreate=native.itemOnCreate(game);
 let seq=0;
 actor.createEmbeddedDocuments=async(_type,data)=>data.map(raw=>{
  const effect={...clone(raw),id:`effect${++seq}`,actor};effect.uuid=`${actor.uuid}.Item.${effect.id}`;effect.isExpired=false;
  const rule=Object.assign(new native.TempHP(),{actor,item:effect,value:raw.system.rules[0].value,events:{onCreate:true,onTurnStart:false}});
  effect.rules=[rule];effect.delete=async()=>actor.deleteEmbeddedDocuments('Item',[effect.id]);actor.items.set(effect.id,effect);created.push(effect);
  itemOnCreate.call(effect,raw,{},game.user.id);return effect;
 });
 actor.deleteEmbeddedDocuments=async(_type,ids)=>{
  for(const id of ids){const effect=actor.items.get(id);if(!effect)continue;const changes={};effect.rules?.[0]?.onDelete(changes);update(actor,changes);actor.items.delete(id);removed.push(id);await Hooks.emit('deleteItem',effect,{},'gm');}
  return ids;
 };
 const ctx={actor,item,entry,user,message};
 const nativeCasts={async ensurePaid(context){payments.push(context);return {id:'payment',state:'used',messageId:context.message.id};}};
 const provider=providerModule.createDeathsCall({game,fromUuid:async uuid=>documents.get(uuid),nativeCasts,chooseCast:async options=>{choices.push(options);return answers.shift();},onError:error=>errors.push(error)});
 const cast=()=>provider.interceptCast({item,entry,options:{}},async()=>{casts.push('cast');actor.system.resources.focus.value--;message.flags[ID].deathsCallInput=provider.captureCast(item);return message;});
 const execute=async()=>{game.user=gm;return provider.executeUsage(ctx);};
 const flushTempHP=async()=>{for(const commit of pendingUpdates.splice(0))await commit();};
 return {provider,actor,item,entry,trigger,token,message,game,user,gm,ctx,nativeCasts,native,Hooks,hooks,errors,documents,created,removed,payments,choices,casts,cast,execute,flushTempHP};
}

test('native Cast grants level plus current casting attribute as one one-minute TempHP effect and one paid use',async()=>{
 const f=fixture();await f.cast();await f.execute();await f.execute();
 assert.equal(f.actor.system.resources.focus.value,1);assert.equal(f.payments.length,1);assert.equal(f.created.length,1);
 assert.equal(f.actor.system.attributes.hp.temp,10);assert.equal(f.actor.system.attributes.hp.value,20);
 assert.equal(f.actor.system.attributes.hp.tempSource,f.created[0].id);
 assert.equal(f.created[0].system.duration.value,1);assert.equal(f.created[0].system.duration.unit,'minutes');
 assert.deepEqual(f.created[0].system.rules,[{key:'TempHP',value:10}]);assert.equal(f.provider.captureCast(f.item),undefined);
});
test('undead destruction doubles the complete level and casting-attribute sum',async()=>{
 const f=fixture({mode:'undead',level:7,modifier:3});await f.cast();await f.execute();assert.equal(f.actor.system.attributes.hp.temp,20);
});
test('negative creature levels and casting modifiers are read as numbers without healing ordinary HP',async()=>{
 const f=fixture({level:-1,modifier:0});await f.cast();await f.execute();assert.equal(f.actor.system.attributes.hp.temp,0);assert.equal(f.actor.system.attributes.hp.value,20);
});
test('cancelling either pre-cast choice leaves focus and native Cast untouched',async()=>{
 for(const answers of [[null],['Scene.scene.Token.trigger','cancel']]){const f=fixture({answers});await f.cast();assert.equal(f.casts.length,0);assert.equal(f.actor.system.resources.focus.value,2);assert.equal(f.provider.captureCast(f.item),undefined);}
});
test('display-only use and casts without consumption never acquire temporary HP or a payment',async()=>{
 for(const options of [{consume:false},{message:false}]){const f=fixture();delete f.message.flags[ID].nativeCast;await f.provider.interceptCast({item:f.item,options},async()=>f.casts.push('display'));await f.execute();assert.equal(f.choices.length,0);assert.equal(f.payments.length,0);assert.equal(f.created.length,0);}
});
test('same-name foreign spells and other worlds retain native behavior without hooks',async()=>{
 for(const change of [f=>f.item.sourceId='Compendium.other.Item.spell',f=>f.game.world.id='av']){const f=fixture();change(f);await f.cast();assert.equal(f.choices.length,0);assert.equal(f.provider.resolveAction(f.item),null);}
 const f=fixture();f.game.world.id='av';f.provider.register({Hooks:f.Hooks});assert.equal(f.hooks.size,0);
});
test('native creature category excludes ordinary constructs but accepts native living constructs',async()=>{
 const f=fixture({mode:'construct'});await assert.rejects(f.cast(),/生物|目标/);assert.equal(f.casts.length,0);
 const living=fixture();living.trigger.system.traits.value=['construct'];await living.cast();await living.execute();assert.equal(living.actor.system.attributes.hp.temp,10);
});
test('invisible tokens are not offered and hidden names or levels do not leak through the chooser',async()=>{
 const f=fixture();await f.cast();assert.doesNotMatch(JSON.stringify(f.choices),/Hidden|等级|level|\b6\b/);
 const hidden=fixture();hidden.token.object.isVisible=false;await assert.rejects(hidden.cast(),/生物|目标/);assert.equal(hidden.casts.length,0);
});
test('changed spell source, trigger identity and casting entry stop settlement before payment',async()=>{
 for(const change of [f=>f.item.sourceId='Compendium.other.Item.spell',f=>f.documents.delete(f.token.uuid),f=>f.item.spellcasting={...f.entry,id:'other',uuid:'Actor.cleric.Item.other'}]){
  const f=fixture();await f.cast();change(f);try{await f.execute();}catch(error){assert.match(error.message,/来源|生物|条目|改变/);}assert.equal(f.created.length,0);assert.equal(f.payments.length,0);
 }
});
test('settlement reads current native level and attribute rather than trusting captured numeric inputs',async()=>{
 const f=fixture();await f.cast();f.trigger.level=8;f.entry.statistic.attributeModifier.value=5;Object.assign(f.message.flags[ID].deathsCallInput,{level:200,modifier:200,amount:999});await f.execute();assert.equal(f.actor.system.attributes.hp.temp,13);
});
test('a failed native payment cannot create an effect',async()=>{
 const f=fixture();await f.cast();f.nativeCasts.ensurePaid=async()=>{throw Error('payment rejected');};await assert.rejects(f.execute(),/payment rejected/);assert.equal(f.created.length,0);
});
test('failed effect creation is terminal for the actual cast card and cannot silently retry',async()=>{
 const f=fixture();await f.cast();let attempts=0;f.actor.createEmbeddedDocuments=async()=>{attempts++;throw Error('creation failed');};await assert.rejects(f.execute(),/creation failed/);await f.execute();assert.equal(attempts,1);assert.equal(f.payments.length,1);
});
for(const temp of [10,15])test(`native TempHP preserves pre-existing ${temp} HP and its owner on creation and expiry`,async()=>{
 const f=fixture({temp,tempSource:'other'});await f.cast();await f.execute();assert.equal(f.actor.system.attributes.hp.temp,temp);assert.equal(f.actor.system.attributes.hp.tempSource,'other');
 const effect=f.created[0];effect.isExpired=true;f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateWorldTime',160);assert.equal(f.actor.system.attributes.hp.temp,temp);assert.equal(f.actor.system.attributes.hp.tempSource,'other');
});
test('source replacement and deletion of the old Death\'s Call effect preserve the new source',async()=>{
 const f=fixture();await f.cast();await f.execute();f.provider.register({Hooks:f.Hooks});
 await f.actor.update({'system.attributes.hp.temp':25,'system.attributes.hp.tempSource':'other'});
 assert.equal(f.actor.system.attributes.hp.temp,25);assert.equal(f.actor.system.attributes.hp.tempSource,'other');assert.equal(f.actor.items.has(f.created[0].id),false);await f.execute();assert.equal(f.created.length,1);
});
test('native depletion removes only the source marker, then the provider ends its own effect',async()=>{
 const f=fixture();await f.cast();await f.execute();await f.actor.update({'system.attributes.hp.temp':0});assert.equal(f.actor.system.attributes.hp.tempSource,undefined);assert.equal(f.actor.items.has(f.created[0].id),true);
 f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateActor',f.actor,{system:{attributes:{hp:{temp:0}}}});assert.equal(f.actor.items.has(f.created[0].id),false);await f.execute();assert.equal(f.created.length,1);
});
test('expiry with native automatic removal disabled deletes its own effect and clears only its still-owned temporary HP',async()=>{
 const f=fixture();await f.cast();await f.execute();f.provider.register({Hooks:f.Hooks});f.created[0].isExpired=true;await f.Hooks.emit('updateWorldTime',160);assert.equal(f.actor.items.has(f.created[0].id),false);assert.equal(f.actor.system.attributes.hp.temp,0);await f.execute();assert.equal(f.created.length,1);
});

test('native automatic expiry and the provider share one deletion during the same world-time update',async()=>{
 const f=fixture({removeEffects:true});await f.cast();await f.execute();
 const tracker=f.native.effectTracker(f.game);tracker.effects=[f.created[0]];f.actor.primaryUpdater=f.gm;f.created[0].isExpired=true;
 const remove=f.actor.deleteEmbeddedDocuments.bind(f.actor);
 f.actor.deleteEmbeddedDocuments=async(type,ids)=>{
  await new Promise(setImmediate);
  for(const id of ids)if(!f.actor.items.has(id))throw Error(`Item "${id}" does not exist!`);
  return remove(type,ids);
 };
 f.Hooks.on('updateWorldTime',()=>tracker.refresh({resetItemData:true}));f.provider.register({Hooks:f.Hooks});
 await f.Hooks.emit('updateWorldTime',160);
 assert.deepEqual(f.errors,[]);assert.deepEqual(f.removed,[f.created[0].id]);
 assert.equal(f.actor.items.has(f.created[0].id),false);assert.equal(f.actor.system.attributes.hp.temp,0);
});

for(const expiry of ['isExpired','remainingDuration'])for(const [name,changes,temp,source]of [
 ['depletion',{'system.attributes.hp.temp':0},0,undefined],
 ['source replacement',{'system.attributes.hp.temp':25,'system.attributes.hp.tempSource':'other'},25,'other'],
])test(`native automatic removal owns ${expiry} effects during ${name}`,async()=>{
 const f=fixture({removeEffects:true});await f.cast();await f.execute();f.provider.register({Hooks:f.Hooks});
 const effect=f.created[0];if(expiry==='isExpired')effect.isExpired=true;else effect.remainingDuration={expired:true};
 await f.actor.update(changes);await f.Hooks.emit('updateCombat',{});
 assert.equal(f.actor.items.has(effect.id),true);assert.deepEqual(f.removed,[]);assert.deepEqual(f.errors,[]);
 assert.equal(f.actor.system.attributes.hp.temp,temp);assert.equal(f.actor.system.attributes.hp.tempSource,source);
});

for(const [name,changes,temp,source]of [
 ['depletion',{'system.attributes.hp.temp':0},0,undefined],
 ['source replacement',{'system.attributes.hp.temp':25,'system.attributes.hp.tempSource':'other'},25,'other'],
])test(`the provider still removes unexpired effects on ${name} with native automatic removal enabled`,async()=>{
 const f=fixture({removeEffects:true});await f.cast();await f.execute();f.provider.register({Hooks:f.Hooks});
 await f.actor.update(changes);
 assert.equal(f.actor.items.has(f.created[0].id),false);assert.deepEqual(f.removed,[f.created[0].id]);assert.deepEqual(f.errors,[]);
 assert.equal(f.actor.system.attributes.hp.temp,temp);assert.equal(f.actor.system.attributes.hp.tempSource,source);
});

test('a failed provider-owned deletion still reports its original error',async()=>{
 const f=fixture({removeEffects:true});await f.cast();await f.execute();f.provider.register({Hooks:f.Hooks});
 const error=Error('delete rejected');f.actor.deleteEmbeddedDocuments=async()=>{throw error;};
 await f.actor.update({'system.attributes.hp.temp':0});
 assert.deepEqual(f.errors,[error]);assert.equal(f.actor.items.has(f.created[0].id),true);
});
test('Undo after depletion does not recreate the ended effect or delete a different HP source',async()=>{
 const f=fixture();await f.cast();await f.execute();f.provider.register({Hooks:f.Hooks});await f.actor.update({'system.attributes.hp.temp':0});
 await f.actor.update({'system.attributes.hp.temp':25,'system.attributes.hp.tempSource':'other'});
 await f.native.undo.call(f.actor,{updates:[{path:'system.attributes.hp.temp',value:10}],persistent:[]});
 assert.equal(f.actor.system.attributes.hp.temp,35);assert.equal(f.created.length,1);assert.equal(f.actor.items.has(f.created[0].id),false);
 await f.Hooks.emit('updateWorldTime',200);assert.equal(f.actor.system.attributes.hp.temp,35);assert.equal(f.errors.length,0);
});
test('player clients never clean up effects and unregister releases all listeners',async()=>{
 const f=fixture();await f.cast();await f.execute();const stop=f.provider.register({Hooks:f.Hooks});f.game.user=f.user;f.created[0].isExpired=true;await f.Hooks.emit('updateWorldTime',160);assert.equal(f.actor.items.has(f.created[0].id),true);stop();assert.equal(f.hooks.size,0);
});
test('the real native cast receipt binds one original card and rejects a second card using the same payment',async()=>{
 const f=fixture();f.item.system.cast={focusPoints:1};
 const request={actorUuid:f.actor.uuid,itemUuid:f.item.uuid,sourceId:SOURCE,entryUuid:f.entry.uuid,rank:3,slotId:null,focusPoints:1,overlayIds:[]};
 f.actor.flags[ID]={nativeCasts:[{...request,id:'payment',userId:f.user.id,state:'paid'}]};f.message.flags[ID].nativeCastInput=clone(request);
 const native=createNativeCastEvents({game:f.game,fromUuid:async uuid=>f.documents.get(uuid)});native.addMatcher(f.provider.matches);f.nativeCasts.ensurePaid=native.ensurePaid;
 await f.cast();await f.execute();
 assert.equal(f.actor.flags[ID].nativeCasts[0].state,'used');assert.equal(f.actor.flags[ID].nativeCasts[0].messageId,'cast');
 const second={...f.message,id:'second',uuid:'ChatMessage.second',flags:clone(f.message.flags)};delete second.flags[ID].deathsCall;f.game.messages.set(second.id,second);
 await assert.rejects(f.provider.executeUsage({...f.ctx,message:second}),/另一张消息/);assert.equal(f.created.length,1);assert.equal(f.actor.system.resources.focus.value,1);
});
test('a forged native payment flag without a durable receipt grants no effect',async()=>{
 const f=fixture();const native=createNativeCastEvents({game:f.game,fromUuid:async uuid=>f.documents.get(uuid)});native.addMatcher(f.provider.matches);f.nativeCasts.ensurePaid=native.ensurePaid;
 await f.cast();await assert.rejects(f.execute(),/支付参数不匹配/);assert.equal(f.created.length,0);
});
test('throwing native Cast releases its selection without retaining reusable capture data',async()=>{
 const f=fixture();await assert.rejects(f.provider.interceptCast({item:f.item,options:{}},async()=>{assert.ok(f.provider.captureCast(f.item));throw Error('cast failed');}),/cast failed/);
 assert.equal(f.provider.captureCast(f.item),undefined);assert.equal(f.created.length,0);
});
test('native Undo restores only numeric HP and does not revive an expired or depleted spell source',async()=>{
 const f=fixture();await f.cast();await f.execute();f.provider.register({Hooks:f.Hooks});await f.actor.update({'system.attributes.hp.temp':0});
 await f.native.undo.call(f.actor,{updates:[{path:'system.attributes.hp.temp',value:10}],persistent:[]});
 assert.equal(f.actor.system.attributes.hp.temp,10);assert.equal(f.actor.system.attributes.hp.tempSource,undefined);assert.equal(f.actor.items.has(f.created[0].id),false);
 await f.execute();assert.equal(f.created.length,1);await f.Hooks.emit('updateWorldTime',200);assert.equal(f.actor.system.attributes.hp.temp,10);
});
test('unrelated HP owners survive manual deletion of this spell effect',async()=>{
 const f=fixture();await f.cast();await f.execute();f.actor.system.attributes.hp.temp=25;f.actor.system.attributes.hp.tempSource='different';
 await f.created[0].delete();assert.equal(f.actor.system.attributes.hp.temp,25);assert.equal(f.actor.system.attributes.hp.tempSource,'different');await f.execute();assert.equal(f.created.length,1);
});
test('lifecycle listeners ignore unrelated item creation and non-HP actor changes',async()=>{
 const f=fixture();await f.cast();await f.execute();f.provider.register({Hooks:f.Hooks});
 const unrelated={uuid:'Actor.unrelated',get items(){throw Error('unrelated inventory scanned');}};
 await f.Hooks.emit('createItem',{type:'weapon',actor:unrelated,flags:{}});
 f.created[0].isExpired=true;await f.Hooks.emit('updateActor',f.actor,{name:'changed'});assert.equal(f.actor.items.has(f.created[0].id),true);
});

function gmObserver(f,user){
 const game={...f.game,user};game.users.set(user.id,user);
 const provider=providerModule.createDeathsCall({game,onError:error=>f.errors.push(error)});
 provider.register({Hooks:f.Hooks});return provider;
}
for(const [name,changes,temp,source]of [
 ['depletion',{'system.attributes.hp.temp':0},0,undefined],
 ['source replacement',{'system.attributes.hp.temp':25,'system.attributes.hp.tempSource':'other'},25,'other'],
])test(`a separate standby GM indexes existing effects and cleans ${name} after promotion`,async()=>{
 const f=fixture();await f.cast();await f.execute();
 const backup={id:'backup',isGM:true,active:true};gmObserver(f,backup);
 await f.actor.update(changes);assert.equal(f.actor.items.has(f.created[0].id),true);assert.equal(f.removed.length,0);
 f.game.users.activeGM=backup;await f.Hooks.emit('updateActor',f.actor,update({},changes));
 assert.equal(f.actor.items.has(f.created[0].id),false);assert.equal(f.removed.length,1);
 assert.equal(f.actor.system.attributes.hp.temp,temp);assert.equal(f.actor.system.attributes.hp.tempSource,source);assert.equal(f.errors.length,0);
});
test('a standby GM tracks newly created effects without writes and handles their expiry after promotion',async()=>{
 const f=fixture(),backup={id:'backup',isGM:true,active:true};gmObserver(f,backup);
 await f.cast();await f.execute();await f.Hooks.emit('createItem',f.created[0]);f.created[0].isExpired=true;
 await f.Hooks.emit('updateWorldTime',160);assert.equal(f.actor.items.has(f.created[0].id),true);assert.equal(f.removed.length,0);
 f.game.users.activeGM=backup;await f.Hooks.emit('updateWorldTime',161);
 assert.equal(f.actor.items.has(f.created[0].id),false);assert.equal(f.actor.system.attributes.hp.temp,0);assert.equal(f.removed.length,1);assert.equal(f.errors.length,0);
});
test('the original GM keeps new-effect indexes while inactive and resumes cleanup on return',async()=>{
 const f=fixture(),backup={id:'backup',isGM:true,active:true};gmObserver(f,f.gm);gmObserver(f,backup);
 f.game.users.activeGM=backup;
 const [effect]=await f.actor.createEmbeddedDocuments('Item',[{type:'effect',flags:{[ID]:{deathsCall:{sourceId:SOURCE,actorUuid:f.actor.uuid,messageId:'later',paymentId:'later-payment'}}},system:{context:{origin:{actor:f.actor.uuid,item:f.item.uuid}},rules:[{key:'TempHP',value:10}]}}]);
 await f.Hooks.emit('createItem',effect);f.game.users.activeGM=f.gm;effect.isExpired=true;
 await f.Hooks.emit('updateCombat',{});
 assert.equal(f.actor.items.has(effect.id),false);assert.equal(f.actor.system.attributes.hp.temp,0);assert.deepEqual(f.removed,[effect.id]);assert.equal(f.errors.length,0);
});
test('a player registering lifecycle hooks never scans world actors or scenes',()=>{
 const f=fixture(),game={...f.game,user:f.user};
 Object.defineProperty(game,'actors',{get(){throw Error('player scanned actors');}});Object.defineProperty(game,'scenes',{get(){throw Error('player scanned scenes');}});
 const provider=providerModule.createDeathsCall({game});const stop=provider.register({Hooks:f.Hooks});stop();assert.equal(f.hooks.size,0);
});
test('delayed native TempHP updates never turn a created effect into a false no-grant report',async()=>{
 const f=fixture({deferTempHP:true});f.provider.register({Hooks:f.Hooks});await f.cast();const result=await f.execute();
 assert.equal(f.actor.system.attributes.hp.temp,0);assert.equal(f.actor.system.attributes.hp.tempSource,undefined);assert.equal(f.created.length,1);
 assert.equal(Object.hasOwn(f.message.flags[ID].deathsCall,'granted'),false);
 assert.doesNotMatch(result.result,/已有临时生命值不低于|已通过原生效果获得/);assert.match(result.result,/效果/);
 await f.flushTempHP();assert.equal(f.actor.system.attributes.hp.temp,10);assert.equal(f.actor.system.attributes.hp.tempSource,f.created[0].id);assert.equal(f.actor.items.has(f.created[0].id),true);
 await f.execute();assert.equal(f.created.length,1);assert.equal(f.payments.length,1);assert.equal(f.errors.length,0);
});

for(const boundary of ['creating acknowledgement','effect acknowledgement'])test(`GM handoff at ${boundary} stops further settlement writes`,async()=>{
 const f=fixture();f.item.system.cast={focusPoints:1};
 const request={actorUuid:f.actor.uuid,itemUuid:f.item.uuid,sourceId:SOURCE,entryUuid:f.entry.uuid,rank:3,slotId:null,focusPoints:1,overlayIds:[]};
 f.actor.flags[ID]={nativeCasts:[{...request,id:'payment',userId:f.user.id,state:'paid'}]};f.message.flags[ID].nativeCastInput=clone(request);
 const native=createNativeCastEvents({game:f.game,fromUuid:async uuid=>f.documents.get(uuid)});native.addMatcher(f.provider.matches);f.nativeCasts.ensurePaid=native.ensurePaid;
 await f.cast();const backup={id:'backup',isGM:true,active:true};f.game.users.set(backup.id,backup);
 const statuses=[],save=f.message.update.bind(f.message),create=f.actor.createEmbeddedDocuments.bind(f.actor);
 f.message.update=async changes=>{
  const result=await save(changes),status=changes[`flags.${ID}.deathsCall`]?.status;statuses.push(status);
  if(status==='creating'&&boundary==='creating acknowledgement')f.game.users.activeGM=backup;
  return result;
 };
 f.actor.createEmbeddedDocuments=async(...args)=>{const result=await create(...args);if(boundary==='effect acknowledgement')f.game.users.activeGM=backup;return result;};
 await assert.rejects(f.execute(),/主GM已改变/);
 const expectedEffects=boundary==='effect acknowledgement'?1:0;
 assert.equal(f.created.length,expectedEffects);assert.equal(f.actor.system.attributes.hp.temp,expectedEffects*10);
 assert.deepEqual(statuses,['creating']);assert.equal(f.message.flags[ID].deathsCall.status,'creating');
 assert.equal(f.actor.flags[ID].nativeCasts[0].state,'used');assert.equal(f.actor.flags[ID].nativeCasts[0].messageId,'cast');
 assert.equal(f.actor.system.resources.focus.value,1);
 f.game.users.activeGM=f.gm;await f.execute();assert.equal(f.created.length,expectedEffects);assert.deepEqual(statuses,['creating']);
});
