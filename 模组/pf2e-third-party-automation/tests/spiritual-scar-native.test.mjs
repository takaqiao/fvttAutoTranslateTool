import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/spiritual-scar-native.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const compile=args=>{assert.equal(typeof api.compileSpiritualScarResistance,'function','Scar native compiler is required');return api.compileSpiritualScarResistance(args)};
const source='Compendium.pf2e.actionspf2e.Item.f8vqWQktvmtSRpUb';
const rule=()=>({key:'Resistance',definition:['origin:trait:fiend','damage:type:spirit'],predicate:['spiritual-scar'],label:'PF2E.IWR.Custom.SpiritFromFiends',type:'custom',value:'@actor.abilities.cha.mod + 2*@actor.level'});
function fixture({nativeValue=27,prepareError=false,option=false}={}){
 const existing={value:99,type:'custom'},cloneExisting={value:88},actor={uuid:'Actor.bard',system:{attributes:{resistances:[existing]}},rollOptions:{all:{'spiritual-scar':false}},items:new Map()};Object.defineProperty(actor,'attributes',{get:()=>actor.system.attributes});
 const ability={id:'scar',uuid:'Actor.bard.Item.scar',type:'action',sourceId:source,actor,_source:{system:{rules:[{key:'RollOption',option:'spiritual-scar',toggleable:true},rule()]}}};actor.items.set(ability.id,ability);
 const clone={system:{attributes:{resistances:[cloneExisting]}},rollOptions:{all:{'spiritual-scar':option}},items:new Map()};Object.defineProperty(clone,'attributes',{get:()=>clone.system.attributes});
 const cloneItem={id:ability.id,actor:clone,_source:structuredClone(ability._source)};clone.items.set(ability.id,cloneItem);const calls=[];
 actor.getContextualClone=options=>{calls.push({kind:'clone',options});return clone};
 class ResistanceRuleElement{
  constructor(data,{parent,sourceIndex}){this.data=data;this.parent=parent;this.actor=parent.actor;this.sourceIndex=sourceIndex;this.ignored=false;this.invalid=false;calls.push({kind:'rule',data:structuredClone(data),parent,sourceIndex});}
  test(){return this.actor.rollOptions.all['spiritual-scar']===true;}
  afterPrepareData(){assert.deepEqual(this.actor.attributes.resistances,[],'same-definition resistances must be isolated');if(prepareError)throw Error('native preparation failed');this.actor.attributes.resistances.push({type:'custom',value:nativeValue,definition:[...this.data.definition],applicationLabel:'spirit damage from fiends',exceptions:[],doubleVs:[],test:opts=>this.data.definition.every(s=>opts.has(s)),getDoubledValue:()=>nativeValue});}
 }
 const input={actor,ability,nonce:'scar-native-once',options:['original-option'],ResistanceRuleElement};return {actor,ability,clone,cloneItem,calls,existing,cloneExisting,input};
}
test('Scar delegates the unchanged original rule and isolates native merging, then restores clone and live data',()=>{
 const f=fixture(),original=f.clone.attributes.resistances,raw=structuredClone(f.ability._source),r=compile(f.input);
 assert.equal(r.value,27,'Use the native prepared result, not a locally rewritten level/Charisma formula');assert.notEqual(r,f.existing);assert.equal(f.calls.find(c=>c.kind==='rule').parent,f.cloneItem);assert.equal(f.calls.find(c=>c.kind==='rule').sourceIndex,1);assert.deepEqual(f.calls.find(c=>c.kind==='rule').data,rule());assert.deepEqual(f.ability._source,raw);assert.equal(f.clone.attributes.resistances,original);assert.deepEqual(original,[f.cloneExisting]);assert.equal(f.clone.rollOptions.all['spiritual-scar'],false);assert.deepEqual(f.actor.attributes.resistances,[f.existing]);assert.equal(f.actor.rollOptions.all['spiritual-scar'],false);
});
test('native damage predicate requires its private call marker as well as fiend spirit',()=>{
 const f=fixture(),r=compile(f.input),marker=api.spiritualScarMarker(f.input.nonce);
 assert.equal(r.test(new Set([marker,'origin:trait:fiend','damage:type:spirit'])),true);
 for(const opts of [['origin:trait:fiend','damage:type:spirit'],[marker,'origin:trait:fiend','damage:type:fire'],[marker,'damage:type:spirit'],[api.spiritualScarMarker('another'),'origin:trait:fiend','damage:type:spirit']])assert.equal(r.test(new Set(opts)),false);
 assert.equal(r.applicationLabel,'spirit damage from fiends');assert.equal(r.getDoubledValue(new Set([marker])),27);
});
test('native preparation failure restores the original clone list and exact option descriptor',()=>{
 const f=fixture({prepareError:true}),list=f.clone.attributes.resistances;delete f.clone.rollOptions.all['spiritual-scar'];
 assert.throws(()=>compile(f.input),/native preparation failed/);assert.equal(f.clone.attributes.resistances,list);assert.equal(Object.hasOwn(f.clone.rollOptions.all,'spiritual-scar'),false);assert.deepEqual(f.actor.attributes.resistances,[f.existing]);
});
test('already true live and cloned manual toggles are preserved by compilation',()=>{
 const f=fixture({option:true});f.actor.rollOptions.all['spiritual-scar']=true;const before=Object.getOwnPropertyDescriptor(f.clone.rollOptions.all,'spiritual-scar');compile(f.input);assert.deepEqual(Object.getOwnPropertyDescriptor(f.clone.rollOptions.all,'spiritual-scar'),before);assert.equal(f.actor.rollOptions.all['spiritual-scar'],true);
});
test('changed source, duplicate rule, or changed rule definition rejects before cloning',()=>{
 for(const alter of [f=>f.ability.sourceId='Other',f=>f.ability._source.system.rules.push(rule()),f=>f.ability._source.system.rules[1].definition=['damage:type:spirit'],f=>f.ability._source.system.rules[1].value=999,f=>f.ability._source.system.rules[1].predicate=[],f=>f.ability._source.system.rules[1].exceptions=['fire']]){const f=fixture();alter(f);assert.throws(()=>compile(f.input));assert.equal(f.calls.length,0);}
});
test('independent clone ownership is required before any preparation mutation',()=>{
 for(const mode of ['actor','item','resistances','options']){const f=fixture();if(mode==='actor')f.actor.getContextualClone=()=>f.actor;if(mode==='item')f.clone.items.set('scar',f.ability);if(mode==='resistances')f.clone.system.attributes=f.actor.system.attributes;if(mode==='options')f.clone.rollOptions=f.actor.rollOptions;assert.throws(()=>compile(f.input));assert.deepEqual(f.actor.attributes.resistances,[f.existing]);assert.equal(f.actor.rollOptions.all['spiritual-scar'],false);}
});
test('invalid native value or unavailable native constructor cannot create a usable resistance',()=>{
 for(const value of [0,-1,NaN,Infinity])assert.throws(()=>compile(fixture({nativeValue:value}).input));const f=fixture();assert.throws(()=>compile({...f.input,ResistanceRuleElement:null}));assert.throws(()=>compile({...f.input,nonce:''}));
});
test('scoped native entry returns original result and removes only the private instance',async()=>{
 const f=fixture(),r=compile(f.input),list=f.actor.attributes.resistances,answer={actual:'native'};const result=await api.withSpiritualScarResistance(f.actor,r,async()=>{assert.deepEqual(list,[f.existing,r]);return answer});assert.equal(result,answer);assert.equal(f.actor.attributes.resistances,list);assert.deepEqual(list,[f.existing]);
});
test('replacement arrays and native errors both release exactly the owned instance',async()=>{
 const f=fixture(),r=compile(f.input),original=f.actor.attributes.resistances,other={value:7};await assert.rejects(api.withSpiritualScarResistance(f.actor,r,async()=>{f.actor.system.attributes.resistances=[other,r,f.existing,r];throw Error('native application failed')}),/native application failed/);assert.deepEqual(original,[f.existing]);assert.deepEqual(f.actor.attributes.resistances,[other,f.existing]);
});
test('a copied instance, different actor, and replay must never enter native',async()=>{
 const f=fixture(),r=compile(f.input);let entered=0;await assert.rejects(api.withSpiritualScarResistance(f.actor,{...r},()=>entered++));await assert.rejects(api.withSpiritualScarResistance({...f.actor},r,()=>entered++));assert.equal(entered,0);await api.withSpiritualScarResistance(f.actor,r,()=>entered++);await assert.rejects(api.withSpiritualScarResistance(f.actor,r,()=>entered++));assert.equal(entered,1);
});
test('a pending native call cannot reenter its compiled resistance',async()=>{
 const f=fixture(),r=compile(f.input);let release;const pending=api.withSpiritualScarResistance(f.actor,r,()=>new Promise(resolve=>{release=resolve}));await assert.rejects(api.withSpiritualScarResistance(f.actor,r,()=>assert.fail('reentered')));release(42);assert.equal(await pending,42);assert.deepEqual(f.actor.attributes.resistances,[f.existing]);
});
test('failed native entry consumes the instance, while unrelated resistance remains',async()=>{
 const f=fixture(),r=compile(f.input);await assert.rejects(api.withSpiritualScarResistance(f.actor,r,()=>{throw Error('native failed')}));await assert.rejects(api.withSpiritualScarResistance(f.actor,r,()=>assert.fail('replay')));assert.deepEqual(f.actor.attributes.resistances,[f.existing]);
});
