import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import {createBravery} from '../scripts/bravery.mjs';
const BRAVERY='Compendium.pf2e.classfeatures.Item.GJKJafDGuX4BeAeN';
const pf=fs.readFileSync(process.env.PF2E_NATIVE_861_BUNDLE??'C:/Users/Taka/Desktop/fvtt/work/bob2-tpa-gap-research-20261101/snapshot/system/pf2e.mjs','utf8');
const core=fs.readFileSync(path.join(process.env.FOUNDRY_14369_APP_ROOT??'C:/Users/Taka/Desktop/fvtt/work/bob2-tpa-implementation-20261101/runtime/core','client/data/client-backend.mjs'),'utf8');
function section(source,start,end){const a=source.indexOf(start),b=source.indexOf(end,a+start.length);assert(a>=0&&b>a,start);return source.slice(a,b);}
const increase=section(pf,'\n\tasync increaseCondition(','\n\tasync toggleCondition(');
const decrease=section(pf,'\n\tasync decreaseCondition(','\n\tasync increaseCondition(');
const update=section(pf,'\n\tstatic async updateConditionValue(','\n}, zl =').replace('static ','');
const preCreate=section(core,'  static async #preCreateDocumentArray(','\n  /* -------------------------------------------- */');
const FRIGHTENED='Compendium.pf2e.conditionitems.Item.TBSHQspnbcqxsmjL';
const engine=fs.readFileSync('C:/Users/Taka/Desktop/fvtt/work/bob2-tpa-gap-research-20261101/snapshot/modules/trigger-engine/scripts/main.js','utf8');
const createTimedCondition=Function('game',section(engine,'function GT(t)','a(GT,')+section(engine,'function fo(','a(fo,')+';return GT;')({pf2e:{ConditionManager:{conditions:new Map([['frightened',{uuid:FRIGHTENED,name:'Frightened',system:{value:{isValued:true}}}]])}},i18n:{localize:value=>value}});
const assign=(object,key,value)=>{const parts=key.split('.');let at=object;for(const part of parts.slice(0,-1))at=at[part]??={};const last=parts.at(-1);if(last.startsWith('-='))delete at[last.slice(2)];else at[last]=value;};
const set=(object,changes)=>{for(const [key,value]of Object.entries(changes))key.includes('.')?assign(object,key,value):typeof value==='object'&&value!==null&&!Array.isArray(value)?set(object[key]??={},value):object[key]=value;};

function fixture({world='bob-2',source=BRAVERY,suppressed=false}={}){
 const listeners=new Map(),wrappers=new Map(),items=new Map(),errors=[];let nextId=0;
 const Hooks={on(name,fn){listeners.set(name,fn);return name;},off(name){listeners.delete(name);},call(name,...args){return listeners.get(name)?.(...args)!==false;},onError(_name,error){throw error;}};
 const game={world:{id:world},system:{id:'pf2e'},user:{id:'owner'},pf2e:{}};
 const actor={type:'character',id:'pc',uuid:'Actor.pc',items,prototypeToken:{},isOfType:type=>['character','creature'].includes(type),get itemTypes(){return {condition:[...items.values()].filter(i=>i.type==='condition')};},getCondition:slug=>[...items.values()].find(i=>i.slug===slug)??null,deleteEmbeddedDocuments:async(_type,ids)=>{for(const id of ids)items.delete(id);}};
 class Item{
  static documentName='Item';static cleanData=data=>data;
  constructor(data,{parent=actor}={}){this._source=structuredClone(data);this.actor=parent;this.id=data._id??`item${++nextId}`;this.type=data.type;this.flags=this._source.flags??={};this.system=this._source.system;}
  get slug(){return this.system.slug;}get value(){return this.system.value?.value;}get active(){return true;}get isLocked(){return false;}
  isOfType(type){return this.type===type;}toObject(){return structuredClone(this._source);}
  updateSource(changes){set(this._source,changes);}
  async _preCreate(){}async delete(){items.delete(this.id);}
  async update(changes){if(Hooks.call('preUpdateItem',this,changes,{},game.user.id))this.updateSource(changes);return this;}
  static async _preCreateOperation(documents,operation,user){return wrappers.get('CONFIG.Item.documentClass._preCreateOperation')?.(async()=>{},documents,operation,user);}
 }
 const Backend=Function('foundry','Hooks','CONST',`return class {${preCreate}static run(...args){return this.#preCreateDocumentArray(...args)}}`)({utils:{deepClone:structuredClone}},Hooks,{vtt:'Foundry'});
 actor.createEmbeddedDocuments=async(_type,data)=>{const operation={parent:actor,data};await Backend.run(Item,operation,game.user);for(const item of operation.data)items.set(item.id,item);return operation.data;};
 game.pf2e.ConditionManager={getCondition:slug=>({toObject:()=>({type:'condition',system:{slug,value:{value:1}},flags:{pf2e:{}}})}),...Function('tupleHasValue',`return {${update}}`)((list,item)=>list.includes(item))};
 Object.assign(actor,Function('game','Math',`return {${increase},${decrease}}`)(game,{...Math,clamp:(n,min,max)=>Math.min(Math.max(n,min),max),max:Math.max}));
 items.set('bravery',{type:'feat',sourceId:source,suppressed});
 const provider=createBravery({game,onError:error=>errors.push(error.message)});const cleanup=provider.register({Hooks,libWrapper:{register(_id,key,fn){wrappers.set(key,fn);},unregister(_id,key){wrappers.delete(key);}}});
 const add=(value,slug='frightened')=>{const item=new Item({_id:`condition${++nextId}`,type:'condition',system:{slug,value:{value}},flags:{pf2e:{}}});items.set(item.id,item);return item;};
 return {actor,game,items,Item,Hooks,add,cleanup,wrappers,errors};
}

for(const [incoming,want]of [[1,0],[3,2]])test(`Bravery reduces native new frightened ${incoming} to ${want}`,async()=>{
 const f=fixture();await f.actor.increaseCondition('frightened',{value:incoming});assert.equal(f.actor.getCondition('frightened')?.value??0,want);
});
test('native increase subtracts one from the final value once',async()=>{
 const f=fixture(),condition=f.add(1);await f.actor.increaseCondition('frightened',{value:4,max:5});assert.equal(condition.value,4);
 await f.game.pf2e.ConditionManager.updateConditionValue(condition.id,f.actor,5);assert.equal(condition.value,4);
});
test('same value, natural decrease, unrelated updates and other conditions keep native behavior',async()=>{
 const f=fixture(),condition=f.add(4);await condition.update({'system.value.value':4});assert.equal(condition.value,4);
 await f.actor.decreaseCondition('frightened');assert.equal(condition.value,3);
 await condition.update({name:'unchanged condition'});assert.equal(condition.value,3);
 const sickened=f.add(2,'sickened');await f.actor.increaseCondition('sickened',{value:2});assert.equal(sickened.value,4);
 await f.game.pf2e.ConditionManager.updateConditionValue(condition.id,f.actor,0);assert.equal(f.actor.getCondition('frightened'),null);
});
test('a native one-step increase never reduces existing frightened',async()=>{
 const f=fixture(),condition=f.add(5);await f.actor.increaseCondition('frightened');assert.equal(condition.value,5);
});
for(const options of [{world:'-'},{source:'custom-bravery'},{suppressed:true}])test(`Bravery eligibility leaves native conditions unchanged: ${JSON.stringify(options)}`,async()=>{
 const f=fixture(options);await f.actor.increaseCondition('frightened',{value:3});assert.equal(f.actor.getCondition('frightened').value,3);
});
test('creating a separate lower frightened condition does not reduce the existing higher condition',async()=>{
 const f=fixture(),higher=f.add(5);await f.actor.createEmbeddedDocuments('Item',[{type:'condition',system:{slug:'frightened',value:{value:3}}}]);
 assert.equal(higher.value,5);assert.deepEqual(f.actor.itemTypes.condition.map(i=>i.value),[5,2]);
});
test('GrantItem frightened 1 is removed together with its pending parent link',async()=>{
 const f=fixture();const created=await f.actor.createEmbeddedDocuments('Item',[
  {_id:'parent',type:'effect',system:{slug:'fear-effect'},flags:{pf2e:{itemGrants:{frightened:{id:'fear',onDelete:'detach'},other:{id:'other'}}}}},
  {_id:'fear',type:'condition',system:{slug:'frightened',value:{value:1}},flags:{pf2e:{grantedBy:{id:'parent',onDelete:'cascade'}}}}
 ]);
 assert.equal(created.length,1);assert.deepEqual(created[0].flags.pf2e.itemGrants,{other:{id:'other'}});assert.equal(f.items.has('fear'),false);
});
test('a cancelled GrantItem clears only its existing parent link',async()=>{
 const f=fixture(),parent=new f.Item({_id:'parent',type:'effect',system:{slug:'fear-effect'},flags:{pf2e:{itemGrants:{frightened:{id:'fear'},other:{id:'other'}}}}});f.items.set(parent.id,parent);
 await f.actor.createEmbeddedDocuments('Item',[{_id:'fear',type:'condition',system:{slug:'frightened',value:{value:1}},flags:{pf2e:{grantedBy:{id:'parent'}}}}]);
 assert.deepEqual(parent.flags.pf2e.itemGrants,{other:{id:'other'}});assert.equal(f.items.has('fear'),false);
});
test('GrantItem frightened 3 keeps both links and applies Bravery once',async()=>{
 const f=fixture();const created=await f.actor.createEmbeddedDocuments('Item',[
  {_id:'parent',type:'effect',system:{slug:'fear-effect'},flags:{pf2e:{itemGrants:{frightened:{id:'fear'}}}}},
  {_id:'fear',type:'condition',system:{slug:'frightened',value:{value:3}},flags:{pf2e:{grantedBy:{id:'parent'}}}}
 ]);
 assert.equal(created[1].value,2);assert.equal(created[0].flags.pf2e.itemGrants.frightened.id,'fear');assert.equal(created[1].flags.pf2e.grantedBy.id,'parent');
});
function prepareInMemory(effect){
 const actor={conditions:new Map(),rules:[],isImmuneTo:()=>false};
 const handlerSource=pf.slice(pf.indexOf('"badge-value": new ItemAlterationHandler'));
 const handle=Function('itemIsOfType','Vc','Math','return ({'+section(handlerSource,'handle: function(e) {','\n\t\t}\n\t}),')+'\n}}).handle;')((item,type)=>item.type===type,{getNewValue:(mode,_before,value)=>{assert.equal(mode,'override');return value;}},{clamp:(value,min,max)=>Math.min(Math.max(value,min),max)});
 const NativeAlteration=Function('Kc','return class {'+section(pf,'\n\tapplyTo(e) {\n\t\tlet t = Kc[this.property]','\n}, Jc =')+'};')({'badge-value':{fields:{value:{getInitialValue:()=>undefined}},handle:handle.bind({isValid:()=>true})}});
 const game={pf2e:{ConditionManager:{conditions:new Map([[FRIGHTENED,{toObject:()=>({type:'condition',system:{slug:'frightened',value:{value:1}}})}]])}}};
 class Condition{constructor(data){Object.assign(this,data);this.id=data._id;}prepareSiblingData(){}prepareActorData(){}prepareRuleElements(){return [];}}
 const NativeGrant=Function('UUIDUtils','game','ConditionPF2e','foundry','return class {apply(){this.#r();}'+section(pf,'\n\t#r() {\n\t\tif (!this.inMemoryOnly','\n\t#i(e) {')+'};')({isItemUUID:uuid=>uuid===FRIGHTENED},game,Condition,{utils:{randomID:()=> 'virtual',mergeObject:(data,changes)=>{set(data,changes);return data;}}});
 for(const rule of effect.system.rules??[]){
  if(rule.key!=='GrantItem'||rule.uuid!==FRIGHTENED)continue;
  const alterations=(rule.alterations??[]).map(source=>Object.assign(new NativeAlteration(),source,{parent:{ignored:false,resolveValue:value=>value}}));
  Object.assign(new NativeGrant(),rule,{actor,item:effect,alterations,test:()=>true,resolveInjectedProperties:value=>value,failValidation:error=>{throw Error(error);}}).apply();
 }
 return [...actor.conditions.values()].map(condition=>condition.system.value.value);
}
for(const [incoming,want]of [[2,1],[3,2]])test(`the current Trove timed frightened ${incoming} source prepares as ${want} repeatedly`,async()=>{
 const f=fixture(),duration={value:1,unit:'rounds',expiry:'turn-end'},source=createTimedCondition({slug:'frightened',counter:incoming,duration});
 source.system.rules.push({key:'RollOption',domain:'all',option:'other-effect-rule'});
 const [effect]=await f.actor.createEmbeddedDocuments('Item',[source]);
 assert.deepEqual(prepareInMemory(effect),[want]);assert.deepEqual(prepareInMemory(effect),[want]);assert.deepEqual(prepareInMemory(effect),[want]);
 assert.deepEqual(effect.system.duration,duration);assert.deepEqual(effect.system.rules.at(-1),{key:'RollOption',domain:'all',option:'other-effect-rule'});
 assert.equal(source.system.rules[0].alterations[0].value,incoming);
});
test('an in-memory frightened 1 grant is removed without changing its parent effect',async()=>{
 const f=fixture(),source=createTimedCondition({slug:'frightened',counter:1,duration:{value:1,unit:'rounds'}});source.system.rules[0].inMemoryOnly=true;
 const [effect]=await f.actor.createEmbeddedDocuments('Item',[source]);assert.deepEqual(prepareInMemory(effect),[]);assert.deepEqual(effect.system.rules,[]);assert.deepEqual(effect.system.duration,source.system.duration);
});
test('persistent Patreon-style grants remain unchanged at the source effect boundary',async()=>{
 const f=fixture(),source=createTimedCondition({slug:'frightened',counter:3});delete source.system.rules[0].inMemoryOnly;
 const [effect]=await f.actor.createEmbeddedDocuments('Item',[source]);assert.deepEqual(effect.system.rules,source.system.rules);
});
test('a formula-based in-memory value is preserved and reported for manual adjustment',async()=>{
 const f=fixture(),source=createTimedCondition({slug:'frightened',counter:3});source.system.rules[0].alterations[0].value='@item.badge.value';
 const [effect]=await f.actor.createEmbeddedDocuments('Item',[source]);assert.deepEqual(effect.system.rules,source.system.rules);assert.equal(f.errors.length,1);assert.match(f.errors[0],/手动/);
});
