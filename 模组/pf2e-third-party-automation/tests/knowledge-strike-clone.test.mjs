import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createKnowledgeAutomation} from '../scripts/knowledge-automation.mjs';
import {createWeaponSurgeAutomation} from '../scripts/weapon-surge.mjs';
import {createCompanionAutomation} from '../scripts/companion-automation.mjs';
import * as ownerApi from '../scripts/native-owner-operations.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

const bundlePath=process.env.PF2E_NATIVE_BUNDLE??process.env.FVTT_PF2E_BUNDLE;
const nativeApp=process.env.FVTT_NATIVE_APP;
const enabled=!!bundlePath&&fs.existsSync(bundlePath)&&!!nativeApp;
const bundle=enabled?fs.readFileSync(bundlePath,'utf8'):'';
const core=enabled?fs.readFileSync(path.join(nativeApp,'common/abstract/document.mjs'),'utf8'):'';
const coreData=enabled?fs.readFileSync(path.join(nativeApp,'common/abstract/data.mjs'),'utf8'):'';
const coreClient=enabled?fs.readFileSync(path.join(nativeApp,'client/documents/abstract/client-document.mjs'),'utf8'):'';
// Reduced from the failed owner's offline snapshot: native unarmed/infusion,
// the original Marshal states, and unrelated source items that must survive.
const snapshot=JSON.parse(fs.readFileSync(new URL('./fixtures/knowledge-strike-clone.json',import.meta.url),'utf8').replace(/^\uFEFF/,''));
function method(source,needle,start=0){
 const offset=source.indexOf(needle,start);assert.ok(offset>=0,needle);
 const brace=source.indexOf('{',offset+needle.length-1);let depth=1,end=brace+1;
 for(;depth&&end<source.length;end++){if(source[end]==='{')depth++;else if(source[end]==='}')depth--;}
 return source.slice(offset,end);
}
const merge=(base,changes)=>{for(const[key,value]of Object.entries(changes)){if(value&&typeof value==='object'&&!Array.isArray(value)&&base[key]&&typeof base[key]==='object')merge(base[key],value);else base[key]=structuredClone(value);}return base;};
const patch=function(changes){for(const[key,value]of Object.entries(changes)){let at=this;const parts=key.split('.');for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=value;}return this;};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
async function until(read){for(let n=0;n<50;n++){const value=read();if(value)return value;await tick();}throw Error('native boundary was not reached');}

function fixture({delaySync=false,throwPreparing=false,dropPreparedRule=false,dropPreparedEphemeral=false,nativeRebind=false,rebindDispatch=null}={}){
 const gm={id:'gm',isGM:true,active:true},player={id:'player',isGM:false,active:true,settings:{showCheckDialogs:false}},users=Object.assign(new Map([[gm.id,gm],[player.id,player]]),{activeGM:gm});
 const hooks=new Map(),gmHandlers=new Map(),ownerHandlers=new Map(),messages=new Map(),actors=new Map(),scenes=new Map(),docs=new Map(),windows=[],failures=[],claimRequests=[],completeRequests=[],rebindEntries=[];let hookId=0,dice=0,cloneWrites=0,liveSourceWrites=0,pendingSync,showWindow;
 const windowShown=new Promise(resolve=>{showWindow=resolve;});
 const Hooks={on(event,fn){const id=++hookId;hooks.set(id,{event,fn});return id;},off(_event,id){hooks.delete(id);},onError(_source,error){throw error;},call(event,...args){for(const h of [...hooks.values()])if(h.event===event)h.fn(...args);}};
 const common={_documentsReady:true,users,messages,actors,scenes,time:{worldTime:100},combat:{id:snapshot.combatId,started:true},modules:new Map(),settings:{get:()=> 'public'},pf2e:{settings:{tokens:{nameVisibility:true}},ConditionManager:{conditions:new Map()}}};
 const gmGame={...common,user:gm},ownerGame={...common,user:player};
 class Item{
  constructor(data,actor){this._source=structuredClone(data);Object.assign(this,structuredClone(data));this.id=data._id;this.actor=actor;this.uuid=`${actor?.uuid??'Compendium.pf2e.conditionitems'}.Item.${this.id}`;this.flags??={};this.system??={};}
  toObject(){return structuredClone(this._source);}
  isOfType(...types){return types.includes(this.type);}
  getRollOptions(prefix='item'){return [`${prefix}:id:${this.id}`,`${prefix}:type:${this.type}`,...this.system.traits?.value?.map(value=>`${prefix}:trait:${value}`)??[]];}
 }
 const condition=new Item({_id:'AJh5ex99aV6VTggg',name:'Off-Guard',type:'condition',system:{slug:'off-guard',rules:[]}});common.pf2e.ConditionManager.conditions.set('Compendium.pf2e.conditionitems.Item.AJh5ex99aV6VTggg',condition);
 class Predicate extends Array{test(options=[]){return this.every(option=>options.includes?.(option)||options.has?.(option));}}
 class Rule{
  static LOCALIZATION_PREFIXES=[];
  constructor(data,item){Object.assign(this,{affects:'target',adjustName:true,alterations:[],predicate:[],ignored:false},structuredClone(data));this.item=item;this.actor=item.actor;this.definition=new Predicate(...data.definition??[]);}
  test(options=[]){return new Predicate(...this.predicate).test(options);}
  resolveValue(value){return value;}
  resolveInjectedProperties(value){return value;}
  getReducedLabel(){return this.item.name;}
  failValidation(error){throw Error(error);}
 }
 const slice=(start,end)=>{const a=bundle.indexOf(start),b=bundle.indexOf(end,a);assert.ok(a>=0&&b>a,start);return bundle.slice(a,b);};
 const TokenMark=new Function('Y','UUIDUtils',`return ${slice('TokenMarkRuleElement = class','}, kl =').slice('TokenMarkRuleElement = '.length)} };`)(Rule,{isTokenUUID:uuid=>/^Scene\.[^.]+\.Token\.[^.]+$/.test(uuid)});
 const Ephemeral=new Function('Y','Z','UUIDUtils','game','fromUuid',`return ${slice('EphemeralEffectRuleElement = class','}, ll =').slice('EphemeralEffectRuleElement = '.length)} };`)(Rule,Item,{isItemUUID:()=>true},ownerGame,async uuid=>docs.get(uuid));
 const AdjustStrike=new Function('Y','CONFIG','objectHasKey','addOrUpgradeTrait',`return ${slice('AdjustStrikeRuleElement = class','}, Oc =').slice('AdjustStrikeRuleElement = '.length)} };`)(Rule,{PF2E:{actionTraits:{arcane:'Arcane'},weaponTraits:{magical:'Magical'}}},(object,key)=>Object.hasOwn(object,key),(traits,value)=>{if(!traits.value.includes(value))traits.value.push(value);});
 class StatisticModifier{constructor(slug=snapshot.preparedStrike.slug){this.slug=slug;this.domains=['attack-roll','strike-attack-roll'];this.totalModifier=11;}}
 const Context=new Function('extractEphemeralEffects','StatisticModifier','PCAttackTraitHelpers','getPropertyRuneStrikeAdjustments','getRangeIncrement','M','me','o','g','game','getPropertyRuneDegreeAdjustments','isOffGuardFromFlanking','calculateRangePenalty','he',`${slice('var RollContext = class','}, Ki = class')} };return CheckContext;`)(
  new Function('game','g',`${slice('async function extractEphemeralEffects','function extractRollTwice')};return extractEphemeralEffects;`)(ownerGame,value=>!!value),StatisticModifier,{adjustWeapon(){}},()=>[],()=>null,array=>[...new Set(array)],value=>value!=null,value=>value!=null,value=>!!value,ownerGame,()=>[],()=>false,()=>null,()=>true);
 const rawDocumentClone=method(core,'clone(data={}, context={}) {');
 class Data{_initialize(){}}
 class Document extends Data{
  constructor(data,context={}){super();this._source=structuredClone(data);this.parent=context.parent;this.pack=context.pack;this._dependentTokens=new Map();this.schema={_updateCommit(model,_key,copy){model._source=copy;}};this._initialize();}
  toObject(){return structuredClone(this._source);}
  prepareData(){this.prepare();}
 }
 Document.prototype.clone=new Function('mergeObject',`return class {${rawDocumentClone} #discardInvalidEmbedded(){throw Error('unexpected discard');}};`)(merge).prototype.clone;
 Document.prototype._updateCommit=new Function(`return ({${method(coreData,'_updateCommit(copy, diff, options, _state) {')}})._updateCommit;`)();
 Document.prototype._initialize=new Function('Parent','game',`return class extends Parent {${method(coreClient,'_initialize(options={}) {')}};`)(Data,ownerGame).prototype._initialize;
 Document.prototype._safePrepareData=new Function('Hooks',`return ({${method(coreClient,'_safePrepareData() {')}})._safePrepareData;`)(Hooks);
 let knowledge,prepareStrike;
 class Actor extends Document{
  prepare(){
   this.id=this._source._id;this.uuid=`Actor.${this.id}`;this.type=this._source.type;this.flags=structuredClone(this._source.flags??{});this.alliance=this._source.system?.details?.alliance??'party';this.items=new Map(this._source.items.map(data=>{const item=new Item(data,this);return [item.id,item];}));
   this.synthetics={tokenMarks:new Map(),ephemeralEffects:{},strikeAdjustments:[],rollNotes:{},rollTwice:{},rollSubstitutions:{}};this.rules=[];
   for(const item of this.items.values())for(const data of item.system.rules??[]){const Class={TokenMark,EphemeralEffect:Ephemeral,AdjustStrike}[data.key];if(Class){const rule=new Class(data,item);this.rules.push(rule);rule.beforePrepareData?.();}}
   for(const rule of this.rules)rule.afterPrepareData?.();
   const weapon=new Item(snapshot.preparedStrike.item,this);Object.assign(weapon,{isMelee:true,isRanged:false,isThrown:false,dealsDamage:true});
   for(const adjustment of this.synthetics.strikeAdjustments)adjustment.adjustWeapon?.(weapon);
   const strike=Object.assign(new StatisticModifier(),{type:'strike',item:weapon,ready:true,altUsages:[],traits:[]});
   const rollStart=bundle.indexOf('roll: async (n = {}) => {',bundle.indexOf('prepareStrike(e, { handsReallyFree:'));
   const body=method(bundle,'roll: async (n = {}) => {',rollStart);
   const makeVariant=map=>{const native=new Function('e','O','f','d','x','CheckContext','calculateMAPs','createMAPenalty','k','o','extractNotes','extractRollTwice','extractRollSubstitutions','getPropertyRuneDegreeAdjustments','extractDegreeOfSuccessAdjustments','_loc','_','u','Sa','traitSlugToObject','CONFIG','game','ui','t',`return ({${body}}).roll;`).call(this,weapon,strike,strike.domains,['unarmed','melee'],[],Context,()=>({}),()=>null,[()=>({}),()=>({}),()=>({})],value=>value!=null,()=>[],()=>false,()=>[],()=>[],()=>[],value=>value,'basic-unarmed','melee',{roll:async(_check,context,event,callback)=>{
    const app={context,event,resolve:undefined,close(){this.resolve(false);}};const accepted=new Promise(resolve=>{app.resolve=resolve;});windows.push(app);Hooks.call('renderCheckModifiersDialog',app);showWindow(app);if(!await accepted)return null;
    dice++;const roll={_evaluated:true,total:24,options:{degreeOfSuccess:2}};await callback?.(roll,'success',new Message({speaker:{actor:this.id,scene:context.origin.token.parent.id,token:context.origin.token.id},flags:{pf2e:{origin:{actor:this.uuid,uuid:weapon.uuid},context:{type:'attack-roll',outcome:'success',options:[...context.options],target:{actor:context.target.actor.uuid,token:context.target.token.uuid}}}},rolls:[roll]}));return roll;
   }},value=>value,{PF2E:{actionTraits:{arcane:'Arcane'}}},ownerGame,{notifications:{warn(){return null;}}},map);return {roll:native.bind(this)};};
   strike.variants=[0,1,2].map(makeVariant);this.system={actions:[strike]};if(prepareStrike)prepareStrike.call(this,()=>strike);else if(knowledge)knowledge.wrapStrike(strike,this);
  }
  getActiveTokens(){return [...this._dependentTokens.values()];}
  getRollOptions(){return Object.keys(this.flags.pf2e?.rollOptions?.all??{});}
  getSelfRollOptions(){return [];}
  getReach(){return 5;}
  isImmuneTo(){return false;}
  testUserPermission(user){return user===gm||user===player&&this.id==='mC5ltcJkLfQR0HtT';}
  getStatistic(){return {dc:{value:this.items.has(condition.id)?18:20}};}
  isAllyOf(actor){return this.alliance===actor.alliance;}
  async update(changes){patch.call(this,changes);patch.call(this._source,changes);return this;}
  updateSource(changes){if(actors.get(this.id)===this)liveSourceWrites++;else cloneWrites++;if(throwPreparing&&changes.items.some(item=>item.flags?.[ID]?.knowledge?.kind==='strategist-claim'))throw Error('clone rule preparation failed');this._updateCommit(merge(structuredClone(this._source),changes),changes,{},{});if(dropPreparedRule)this.synthetics.tokenMarks.clear();if(dropPreparedEphemeral)for(const entry of Object.values(this.synthetics.ephemeralEffects))entry.target.pop();}
  async createEmbeddedDocuments(_name,items){
   const data=items.map(item=>({...structuredClone(item),_id:'kCrXSWQeWbfl3QjR'}));const synchronize=()=>{this._source.items.push(...data);this.prepare();};
   if(delaySync){pendingSync=synchronize;return data.map(item=>new Item(item,this));}synchronize();return data.map(item=>this.items.get(item._id));
  }
  async deleteEmbeddedDocuments(_name,ids){this._source.items=this._source.items.filter(item=>!ids.includes(item._id));this.prepare();}
 }
 // Preserve the actual super.clone implementation and PF2e's dependent Token copy.
 const actorClone=method(bundle,'clone(e, t) {',bundle.indexOf('Ns = class ActorPF2e'));
 Actor.prototype.clone=new Function('Parent',`return class extends Parent {${actorClone}};`)(Document).prototype.clone;
 Actor.prototype.getContextualClone=new Function('foundry',`return ({${method(bundle,'getContextualClone(e, t = []) {')}}).getContextualClone;`)({utils:{deepClone:structuredClone}});
 const actor=new Actor(snapshot.actor),marshal=new Actor(snapshot.marshal);
 const targetActor=new Actor({_id:snapshot.request.targetActorUuid.split('.')[1],type:'npc',items:[],flags:{},system:{details:{alliance:'opposition'}}});
 const state=marshal.flags[ID].knowledge.strategistStates[0],scene={id:state.targetUuid.split('.')[1],tokens:new Map()};scenes.set(scene.id,scene);
 const token=(uuid,actor)=>{const id=uuid.split('.').at(-1),doc={id,uuid,documentName:'Token',parent:scene,actor,object:{controlled:uuid===snapshot.sourceTokenUuid,distanceTo:()=>5,isFlanking:()=>false},auras:new Map()};doc.object.document=doc;scene.tokens.set(id,doc);actor._dependentTokens.set(id,doc);docs.set(doc.uuid,doc);return doc;};
 const hero=token(snapshot.sourceTokenUuid,actor),enemy=token(state.targetUuid,targetActor),marshalToken=token(state.sourceTokenUuid,marshal);marshalToken.auras.set('marshals-aura',{containsToken:source=>source===hero});
 for(const doc of [actor,marshal,targetActor]){actors.set(doc.id,doc);docs.set(doc.uuid,doc);}player.targets=new Set([enemy.object]);gm.targets=new Set();
 class Message{constructor(data){Object.assign(this,data);if(typeof this.author==='string')this.author=users.get(this.author);this.flags??={};}get actor(){return actors.get(this.speaker?.actor);}get isCheckRoll(){return this.flags.pf2e?.context?.type==='attack-roll';}toObject(){return {...this,author:this.author?.id??this.author};}async update(changes){patch.call(this,changes);Hooks.call('updateChatMessage',this);}static async create(data){const message=new Message({...data,id:'check'});messages.set(message.id,message);Hooks.call('createChatMessage',message);return message;}}
 const activity=new Message({id:'4QbJ4ie1XYg7jw28',author:player,flags:{pf2e:{origin:{actor:actor.uuid}}}});messages.set(activity.id,activity);
 const fromUuid=async uuid=>docs.get(uuid);
 const gmKnowledge=createKnowledgeAutomation({game:gmGame,fromUuid});knowledge=createKnowledgeAutomation({game:ownerGame,fromUuid});
 if(nativeRebind){
  const surge=createWeaponSurgeAutomation({game:ownerGame}),companion=createCompanionAutomation({game:ownerGame,fromUuid,wrapStrike:(strike,actor)=>{
   knowledge.wrapStrike(surge.wrapStrike(strike,actor),actor);
   for(const[index,variant]of strike.variants.entries()){const native=variant.roll;variant.roll=params=>{if(!Object.getOwnPropertySymbols(params).some(key=>key.description==='knowledgeNativeAttack'))return native(params);const entry={actor,strike,index,params,native};rebindEntries.push(entry);return rebindDispatch?rebindDispatch(entry):native(params);};}
   return strike;
  }});
  companion.register({Hooks,libWrapper:{register(_id,wrapperPath,handler){if(wrapperPath.endsWith('.prepareStrike'))prepareStrike=handler;},unregister(){}}});assert.equal(typeof prepareStrike,'function');
 }
 const root=ownerApi.createNativeOwnerOperations({game:gmGame,fromUuid,scope:'clone-test'}),owner=ownerApi.createNativeOwnerOperations({game:ownerGame,fromUuid,scope:'clone-test'});
 const ownerSocket={register(name,handler){ownerHandlers.set(name,handler);},executeAsUser(name,userId,payload){assert.equal(userId,gm.id);if(name==='knowledge-claim')claimRequests.push(payload);if(name==='knowledge-complete')completeRequests.push(payload);return gmHandlers.get(name).call({socketdata:{userId:player.id}},payload);}};
 const gmSocket={register(name,handler){gmHandlers.set(name,handler);},executeAsUser(name,userId,payload){assert.equal(userId,player.id);return ownerHandlers.get(name).call({socketdata:{userId:gm.id}},payload).catch(error=>{failures.push(error);throw error;});}};
 owner.register({Hooks,socket:ownerSocket});root.register({Hooks,socket:gmSocket});knowledge.register({Hooks,socket:ownerSocket});gmKnowledge.register({Hooks,socket:gmSocket});
 const infusion=snapshot.request.transientItems[0];
 let roller;const actorNativeClone=actor.clone.bind(actor);actor.clone=(...args)=>{roller=actorNativeClone(...args);return roller;};
 const run=async(request={})=>{try{return await root.run({actor,message:activity,user:player},{type:'attack',weaponId:'xxPF2ExUNARMEDxx',map:0,targetUuid:enemy.uuid,options:['action:spellstrike'],transientItems:[infusion],...request});}catch(error){await tick();throw failures[0]??error;}};
 return {run,actor,marshal,hero,enemy,ownerGame,player,gm,infusion,Hooks,windows,windowShown,claimRequests,completeRequests,rebindEntries,knowledge,gmKnowledge,Message,docs,fromUuid,get roller(){return roller;},sync(){pendingSync?.();},get counts(){return {dice,cloneWrites,liveSourceWrites};},get state(){return marshal.flags[ID].knowledge.strategistStates[0];}};
}

function setup(t,options){const previous=globalThis.CONFIG;t.after(()=>{globalThis.CONFIG=previous;});const f=fixture(options);globalThis.CONFIG={ChatMessage:{documentClass:f.Message}};return f;}
const opening=f=>{const pending=f.run();return {pending,app:Promise.race([pending.then(()=>{throw Error('returned before manual native window');}),f.windowShown])};};

test('native Spellstrike clone receives the synchronized Marshal claim before its manual Strike window',{skip:!enabled},async t=>{
 const f=setup(t),{pending,app:shown}=opening(f),app=await shown;
 assert.equal(f.counts.dice,0);assert.equal(app.event.shiftKey,true);assert.equal(app.context.dc.value,18);assert.ok(app.context.traits.includes('arcane'));assert.ok(app.context.item.system.traits.value.includes('magical'));
 assert.equal(app.context.origin.token,f.hero);assert.equal(app.context.target.token,f.enemy);assert.notEqual(app.context.origin.actor,f.actor);
 assert.ok(f.roller._source.items.some(item=>item._id===f.infusion._id));assert.ok(!f.actor._source.items.some(item=>item._id===f.infusion._id));
 await app.resolve(true);assert.equal((await pending).status,'rolled');assert.equal(f.state.status,'consumed');assert.equal(f.counts.dice,1);assert.equal(f.counts.liveSourceWrites,0);
 assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);assert.deepEqual(f.actor._source.items,snapshot.actor.items);
 assert.equal(ownerApi.getNativeOwnerTransientActor(f.ownerGame,f.roller),null);
});

test('the native clone waits for live claim propagation while retaining its infusion snapshot',{skip:!enabled},async t=>{
 const f=setup(t,{delaySync:true}),{pending,app:shown}=opening(f);await until(()=>f.state.status==='claimed');
 assert.equal(f.windows.length,0);assert.equal(f.counts.dice,0);assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);
 f.sync();const app=await shown;assert.equal(app.context.dc.value,18);await app.resolve(false);assert.equal((await pending).status,'cancelled');
 assert.equal(f.state.status,'pending');assert.equal(f.counts.dice,0);assert.equal(f.counts.liveSourceWrites,0);assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);assert.deepEqual(f.actor._source.items,snapshot.actor.items);
});

test('a Marshal opportunity is consumed once and an unrelated older claim is retained',{skip:!enabled},async t=>{
 const f=setup(t),old={...f.state,id:'older',targetUuid:f.enemy.uuid.replace(f.enemy.id,'other'),status:'claimed',claim:'another'};f.marshal.flags[ID].knowledge.strategistStates.push(old);
 const first=opening(f),app=await first.app;await app.resolve(true);await first.pending;assert.equal(f.state.status,'consumed');assert.deepEqual(f.marshal.flags[ID].knowledge.strategistStates[1],old);
 const second=f.run(),next=await until(()=>f.windows[1]);assert.equal(next.context.dc.value,20);assert.equal(f.counts.dice,1);assert.ok(![...next.context.options].some(value=>value.startsWith(`${ID}:knowledge:claim:`)));
 await next.resolve(false);assert.equal((await second).status,'cancelled');assert.equal(f.counts.dice,1);assert.deepEqual(f.marshal.flags[ID].knowledge.strategistStates[1],old);
});

for(const option of ['throwPreparing','dropPreparedRule'])test(`a clone ${option==='throwPreparing'?'preparation error':'missing prepared TokenMark'} restores the pending Marshal opportunity without a die`,{skip:!enabled},async t=>{
 const f=setup(t,{[option]:true});await assert.rejects(f.run(),/preparation failed|未准备完成/);
 assert.equal(f.windows.length,0);assert.equal(f.counts.dice,0);assert.equal(f.state.status,'pending');assert.equal(f.counts.liveSourceWrites,0);assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);assert.deepEqual(f.actor._source.items,snapshot.actor.items);assert.equal(ownerApi.getNativeOwnerTransientActor(f.ownerGame,f.roller),null);
});

test('owner disconnection cancels the original native window and returns its unrolled Marshal claim',{skip:!enabled},async t=>{
 const f=setup(t),{pending,app:shown}=opening(f);await shown;f.player.active=false;f.Hooks.call('userConnected',f.player,false);await assert.rejects(pending,/离线|身份|连接/);await until(()=>f.state.status==='pending');
 assert.equal(f.counts.dice,0);assert.equal(f.counts.liveSourceWrites,0);assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);assert.equal(ownerApi.getNativeOwnerTransientActor(f.ownerGame,f.roller),null);
});

test('an unbound clone with the live actor UUID cannot request a Marshal claim',{skip:!enabled},async t=>{
 const f=setup(t),clone=f.actor.clone({items:[...snapshot.actor.items,f.infusion]},{keepId:true});assert.equal(ownerApi.getNativeOwnerTransientActor(f.ownerGame,clone),null);
 await assert.rejects(clone.system.actions[0].variants[0].roll({target:f.enemy.object}),/文档已经替换|身份已经变化/);assert.equal(f.state.status,'pending');assert.equal(f.windows.length,0);assert.equal(f.counts.dice,0);
});

for(const change of ['relink','delete','replace'])test(`${change} of the authenticated source Token before native acceptance returns its unrolled Marshal claim`,{skip:!enabled},async t=>{
 const f=setup(t),{pending,app:shown}=opening(f),app=await shown;if(change==='relink')f.hero.actor=f.enemy.actor;else if(change==='delete')f.hero.parent.tokens.delete(f.hero.id);else f.hero.parent.tokens.set(f.hero.id,{...f.hero});f.Hooks.call('updateToken',f.hero);await app.resolve(true);
 await assert.rejects(pending,/来源|身份/);await until(()=>f.state.status==='pending');assert.equal(f.counts.dice,0);assert.equal(f.counts.liveSourceWrites,0);assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);
});

test('an unrelated ephemeral effect cannot hide a missing prepared Marshal effect',{skip:!enabled},async t=>{
 const f=setup(t,{dropPreparedEphemeral:true});f.actor._source.items.push({_id:'unrelated',name:'Other effect',type:'effect',system:{rules:[{key:'EphemeralEffect',selectors:['strike-attack-roll','spell-attack-roll'],uuid:'Compendium.pf2e.conditionitems.Item.AJh5ex99aV6VTggg',predicate:['never']} ]}});f.actor.prepare();
 const pending=f.run(),outcome=await Promise.race([pending.then(value=>value,error=>error),f.windowShown.then(async app=>{await app.resolve(false);return 'native window opened';})]);await pending.catch(()=>{});
 assert.match(outcome.message??outcome,/未准备完成/);assert.equal(f.windows.length,0);assert.equal(f.state.status,'pending');assert.equal(f.counts.dice,0);assert.equal(f.counts.liveSourceWrites,0);assert.equal(f.roller._source.items.some(item=>item.flags?.[ID]?.knowledge?.kind==='strategist-claim'),false);assert.ok(f.roller._source.items.some(item=>item._id==='unrelated'));assert.ok(f.roller._source.items.some(item=>item._id===f.infusion._id));
});

test('native preparation rebinds Surge through Knowledge without claiming the same Marshal opportunity twice',{skip:!enabled},async t=>{
 const f=setup(t,{nativeRebind:true}),pending=f.run();let app;
 try{await until(()=>f.windows[0]||f.claimRequests.length>1);assert.equal(f.claimRequests.length,1);app=f.windows[0];assert.ok(app);assert.equal(f.counts.dice,0);assert.equal(app.context.dc.value,18);assert.equal(app.context.mapIncreases,0);assert.ok(app.context.traits.includes('arcane'));assert.ok(app.context.item.system.traits.value.includes('magical'));await app.resolve(true);assert.equal((await pending).status,'rolled');assert.equal(f.counts.dice,1);assert.equal(f.windows.length,1);assert.equal(f.completeRequests.length,1);assert.equal(f.state.status,'consumed');assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);assert.equal(f.counts.liveSourceWrites,0);}
 finally{if(!app){f.player.active=false;f.Hooks.call('userConnected',f.player,false);await pending.catch(()=>{});}}
});

test('cancelling the rebound native window returns the one Marshal claim and preserves infusion',{skip:!enabled},async t=>{
 const f=setup(t,{nativeRebind:true}),{pending,app:shown}=opening(f),app=await shown;await app.resolve(false);assert.equal((await pending).status,'cancelled');
 assert.equal(f.claimRequests.length,1);assert.equal(f.completeRequests.length,1);assert.equal(f.windows.length,1);assert.equal(f.counts.dice,0);assert.equal(f.state.status,'pending');assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);assert.deepEqual(f.actor._source.items,snapshot.actor.items);
});

for(const map of [1,2])test(`native rebind retains the original MAP ${map} closure and its one Marshal claim`,{skip:!enabled},async t=>{
 const f=setup(t,{nativeRebind:true}),pending=f.run({map}),app=await f.windowShown;assert.equal(app.context.mapIncreases,map);assert.equal(app.context.dc.value,18);assert.ok(app.context.traits.includes('arcane'));assert.ok(app.context.item.system.traits.value.includes('magical'));assert.equal(f.counts.dice,0);
 await app.resolve(true);assert.equal((await pending).status,'rolled');assert.equal(f.claimRequests.length,1);assert.equal(f.completeRequests.length,1);assert.equal(f.windows.length,1);assert.equal(f.counts.dice,1);assert.equal(f.state.status,'consumed');
});

test('a live Strike rebinds after its native embedded claim preparation without clone identity grants',{skip:!enabled},async t=>{
 const f=setup(t,{nativeRebind:true});f.actor.prepare();const pending=f.run({transientItems:[]}),app=await f.windowShown;assert.equal(app.context.dc.value,18);assert.equal(f.counts.dice,0);assert.equal(f.claimRequests.length,1);assert.equal(f.rebindEntries[0].actor,f.actor);assert.equal(ownerApi.getNativeOwnerTransientActor(f.ownerGame,f.actor),null);
 await app.resolve(true);assert.equal((await pending).status,'rolled');assert.equal(f.completeRequests.length,1);assert.equal(f.windows.length,1);assert.equal(f.counts.dice,1);assert.equal(f.state.status,'consumed');assert.equal(f.counts.liveSourceWrites,0);assert.deepEqual(f.actor._source.items,snapshot.actor.items);
});

test('owner disconnection closes the rebound manual window and revokes its private continuation',{skip:!enabled},async t=>{
 const f=setup(t,{nativeRebind:true}),{pending,app:shown}=opening(f);await shown;const entry=f.rebindEntries[0];f.player.active=false;f.Hooks.call('userConnected',f.player,false);await assert.rejects(pending,/离线|身份|连接/);await until(()=>f.state.status==='pending');
 await assert.rejects(entry.native({...entry.params}),/承接身份已失效/);assert.equal(f.claimRequests.length,1);assert.equal(f.completeRequests.length,1);assert.equal(f.windows.length,1);assert.equal(f.counts.dice,0);assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);assert.equal(ownerApi.getNativeOwnerTransientActor(f.ownerGame,f.roller),null);
});

for(const change of ['target','MAP','weapon','actor','source'])test(`a rebound ${change} mismatch cannot carry the Marshal claim into another native attack`,{skip:!enabled},async t=>{
 let f,other;f=setup(t,{nativeRebind:true,rebindDispatch:({actor,strike,index,params,native})=>{
  if(change==='target')return native({...params,target:f.hero.object});
  if(change==='MAP'&&index===0)return strike.variants[1].roll(params);
  if(change==='weapon')strike.item._source.name='different weapon';
  if(change==='actor'&&actor!==other)return other.system.actions[0].variants[0].roll(params);
  if(change==='source')f.hero.actor=f.enemy.actor;
  return native(params);
 }});if(change==='actor')other=f.actor.clone({items:[...snapshot.actor.items,f.infusion]},{keepId:true});
 await assert.rejects(f.run(),/承接身份|来源|身份|档位/);assert.equal(f.claimRequests.length,1);assert.equal(f.completeRequests.length,1);assert.equal(f.windows.length,0);assert.equal(f.counts.dice,0);assert.equal(f.state.status,'pending');assert.equal(f.counts.liveSourceWrites,0);assert.deepEqual(f.roller._source.items,[...snapshot.actor.items,f.infusion]);assert.deepEqual(f.actor._source.items,snapshot.actor.items);
});

test('a native continuation is accepted once and its private grant expires with the attack promise',{skip:!enabled},async t=>{
 const f=setup(t,{nativeRebind:true}),{pending,app:shown}=opening(f),app=await shown,entry=f.rebindEntries[0];assert.ok(entry);
 await assert.rejects(entry.native({...entry.params}),/承接身份已失效/);assert.equal(f.windows.length,1);assert.equal(f.claimRequests.length,1);assert.equal(f.counts.dice,0);
 await app.resolve(true);assert.equal((await pending).status,'rolled');assert.equal(f.state.status,'consumed');
 await assert.rejects(entry.native({...entry.params}),/承接身份已失效/);
 const key=Object.getOwnPropertySymbols(entry.params).find(key=>key.description==='knowledgeNativeAttack');await assert.rejects(entry.native({...entry.params,[key]:Object.freeze({})}),/承接身份已失效/);
 assert.equal(f.windows.length,1);assert.equal(f.claimRequests.length,1);assert.equal(f.completeRequests.length,1);assert.equal(f.counts.dice,1);
});

test('public claim options cannot replace a local native continuation grant',{skip:!enabled},async t=>{
 const f=setup(t,{nativeRebind:true}),pending=f.run({options:['action:spellstrike',`${ID}:knowledge:claim:untrusted`]});const app=await f.windowShown;
 assert.equal(f.claimRequests.length,1);assert.equal(app.context.dc.value,18);assert.equal(f.counts.dice,0);await app.resolve(false);assert.equal((await pending).status,'cancelled');assert.equal(f.state.status,'pending');assert.equal(f.completeRequests.length,1);
});
