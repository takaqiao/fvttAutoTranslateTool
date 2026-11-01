import {test} from 'node:test';
import assert from 'node:assert/strict';
import {enhanceComputedActors} from '../../scripts/exploration/computed-rules.mjs';
import {preparedTreatmentSelections} from '../../scripts/exploration/prepared-treatment.mjs';
import {LAY_ON_HANDS} from '../../scripts/exploration/refocus.mjs';
import {SALUBRIOUS_SOURCE} from '../../scripts/salubrious-kiss-rules.mjs';

class Predicate extends Array {test(options){return this.every(option=>typeof option==='string'?options.has(option):true)}toObject(){return [...this]}}
class Modifier {constructor(modifier){this.modifier=modifier;this.type='untyped';this.slug='base';this.predicate=new Predicate();this.adjustments=[];this.enabled=true;this.ignored=false}clone(){return new Modifier(this.modifier)}}
class CheckModifier {constructor(_name,{modifiers}){this.modifiers=modifiers;this.totalModifier=modifiers.reduce((sum,m)=>sum+m.modifier,0)}}
function fixture(){
 const hp={value:20,max:40,temp:0,negativeHealing:false},statistics={};
 for(const [skill,rank] of [['medicine',2],['nature',0],['occultism',1]])statistics[skill]={rank,check:{domains:['all','check','skill-check',skill,`${skill}-check`],modifiers:[new Modifier(10)],createRollOptions:({extraRollOptions,item})=>new Set([...extraRollOptions,...item?.getRollOptions?.('item')??[]])}};
 const actor={uuid:'Actor.H',type:'character',canAct:true,modeOfBeing:'living',isDead:false,hardness:0,hitPoints:hp,attributes:{hp,immunities:[],resistances:[],weaknesses:[]},system:{attributes:{hp},resources:{focus:{value:1,max:1}}},skills:{medicine:statistics.medicine,occultism:statistics.occultism},items:[],rules:[],synthetics:{},getStatistic:skill=>statistics[skill],getRollOptions:()=>[],hasCondition:()=>false};
 const snapshot={actorUUID:actor.uuid,name:'H',systemVersion:'8.6.1',hp,focus:{value:1,max:1},pool:{poolUUID:actor.uuid,ready:true},slugs:[],medicine:{rank:2},occultism:{rank:1},modeOfBeing:'living'};
 const game={system:{id:'pf2e',version:'8.6.1'},pf2e:{Predicate,CheckModifier}};
 return {actor,snapshot,statistics,game,enhance:async()=> (await enhanceComputedActors({actors:[snapshot],game,fromUuid:async()=>actor}))[0]};
}
function damageRoll(expression){return {formula:'本地化的显示公式不可用于解析',instances:[{type:'vitality',kinds:new Set(['healing']),persistent:false,materials:new Set(),head:{expression}}],evaluate(){throw Error('qualification cannot roll')}}}
function layOnHands(f,{formula='12',focusPoints=1,kind='healing'}={}){
 const variant={uuid:'Actor.H.Item.L',rank:2,system:{cast:{focusPoints}},damageKinds:new Set([kind]),getDamage:async options=>{assert.deepEqual(options,{skipDialog:true});return {template:{damage:{roll:damageRoll(formula)}}}}};
 const item={uuid:variant.uuid,sourceId:LAY_ON_HANDS,system:{overlays:{healing:{}}},rank:2,loadVariant:()=>variant};f.actor.items.push(item);return {item,variant};
}

test('qualifies separate executing models from prepared checks and patient receiving state',async()=>{
 const f=fixture(),result=await f.enhance();assert.equal(result.computed.version,1);assert.equal(result.type,'character');assert.equal(result.canAct,true);assert.equal(result.computed.treatments.length,2);assert.equal(result.computed.treatments[1].rank,'expert');assert.equal(result.computed.patient.healingReady,true);assert.equal(result.computed.patient.damageReady,true);assert.equal(result.computed.refocus.recovery,1);
});

test('unrelated speed adjustments do not prevent a prepared Medicine model',async()=>{
 const f=fixture();let invoked=0;const item={uuid:'Actor.H.Item.Speed'};f.actor.rules.push({key:'AdjustModifier',selector:'speed',item});f.actor.synthetics.modifierAdjustments={speed:[{getNewValue:()=>{invoked++;throw Error('speed closure invoked')}}]};
 const selection=preparedTreatmentSelections({actor:f.actor,game:f.game,skill:'medicine'}).find(s=>!s.riskySurgery&&!s.assurance);assert.equal(selection.ready,true,selection.reason);assert.equal(invoked,0);assert.equal((await f.enhance()).computed.treatments.length,2);
});

test('a relevant unknown adjustment still stops before lazy check construction',async()=>{
 const f=fixture();let reads=0;f.actor.synthetics.modifierAdjustments={medicine:[{}]};Object.defineProperty(f.statistics.medicine,'check',{get(){reads++;throw Error('unsafe getter')}});const result=await f.enhance();assert.equal(result.computed.treatments.length,0);assert.equal(reads,0);assert.ok(result.computed.treatmentReasons.includes('native-modifier-adjustment-unverified'));
});

test('healing-received modifiers are blocked, while unrelated damage IWR does not block numeric healing',async()=>{
 const f=fixture();let calls=0;f.actor.synthetics.modifiers={'healing-received':[()=>{calls++;return {modifier:3}}]};let result=await f.enhance();assert.equal(result.computed.patient.healingReady,false);assert.equal(result.computed.patient.healingReason,'healing-received-modifier-unverified');assert.equal(calls,0);
 f.actor.synthetics={};f.actor.attributes.resistances.push({type:'all-damage',value:5});result=await f.enhance();assert.equal(result.computed.patient.damageReady,false);assert.equal(result.computed.patient.healingReady,true);assert.equal(result.computed.patient.healingReason,null);
});

test('Focus requires a real healing overlay, one Focus point and a supported prepared formula',async()=>{
 const f=fixture(),{variant}=layOnHands(f);let result=await f.enhance();assert.equal(result.computed.focusHealing.ready,true);assert.deepEqual(result.computed.focusHealing.healing,{constant:12,dice:[]});assert.deepEqual(result.computed.focusHealing.effects,[{sourceUUID:'Compendium.pf2e.spell-effects.Item.lyLMiauxIVUM3oF1',durationSeconds:6,otherOnly:true}]);
 variant.system.cast.focusPoints=2;assert.equal((await f.enhance()).computed.focusHealing.ready,false);variant.system.cast.focusPoints=1;variant.damageKinds=new Set(['damage']);assert.equal((await f.enhance()).computed.focusHealing.ready,false);
});

test('additive healing dice are preserved for later sampling, unsupported roll expressions stop explicitly',async()=>{
 const f=fixture(),{variant}=layOnHands(f,{formula:'(2d6 + 12)'});assert.deepEqual((await f.enhance()).computed.focusHealing.healing,{constant:12,dice:[{number:2,faces:6}]});
 variant.getDamage=async()=>({template:{damage:{roll:damageRoll('2d6kh1 + 12')}}});const result=await f.enhance();assert.equal(result.computed.focusHealing.ready,false);assert.equal(result.computed.focusHealing.reason,'focus-healing-formula-unverified');
 variant.getDamage=async()=>({template:{damage:{roll:damageRoll('1+(2)d6')}}});assert.equal((await f.enhance()).computed.focusHealing.ready,false);
});

test('canAct and exceptional Refocus recovery are enforced independently of HP',async()=>{
 const f=fixture();layOnHands(f);f.actor.canAct=false;let result=await f.enhance();assert.equal(result.computed.focusHealing.ready,false);assert.equal(result.computed.refocus.ready,false);
 f.actor.canAct=true;f.actor.items.push({type:'feat',slug:'devoted-focus'});result=await f.enhance();assert.equal(result.computed.refocus.ready,false);assert.equal(result.computed.refocus.reason,'refocus-recovery-unadapted');
});

test('Three Pecks prepares an item and vitality context, uses Occultism tiers, and rejects unsupported variants',async()=>{
 const f=fixture(),feat={uuid:'Actor.H.Item.Kiss',type:'feat',sourceId:SALUBRIOUS_SOURCE,slug:'salubrious-kiss',getRollOptions:()=>['item:slug:salubrious-kiss']};f.actor.items.push(feat);
 const seen=[];f.statistics.occultism.check.createRollOptions=({extraRollOptions,item})=>{seen.push({extraRollOptions,item});return new Set(extraRollOptions)};
 const result=await f.enhance();assert.equal(result.computed.threePecks.ready,true);assert.equal(result.computed.threePecks.treatments.length,1);assert.equal(result.computed.threePecks.treatments[0].skill,'occultism');assert.ok(seen.every(c=>c.item===feat&&c.extraRollOptions.includes('vitality')&&c.extraRollOptions.includes('item:trait:vitality')));
 f.actor.items.push({type:'feat',slug:'continual-recovery'});assert.equal((await f.enhance()).computed.threePecks.ready,false);
});

test('Magic Hands remains an explicit unsupported healing model until its formula is adapted',async()=>{
 const f=fixture();f.actor.items.push({type:'feat',slug:'magic-hands'});const result=await f.enhance();assert.equal(result.computed.treatments.length,0);assert.ok(result.computed.treatmentReasons.includes('native-healing-model-unverified'));
});

test('prepared statistic domains exclude unrelated attack and damage adjustments without invoking closures',async()=>{
 const f=fixture();f.statistics.medicine.domains=['medicine','wis-based','skill-check','wis-skill-check','all'];
 f.actor.synthetics.modifierAdjustments={'melee-strike-damage':[{getNewValue(){throw Error('damage closure')}}],attack:[{}],survival:[{}]};
 f.actor.rules.push({key:'AdjustModifier',selector:['melee-strike-damage'],item:{}},{key:'AdjustModifier',selector:['attack','survival'],item:{}});
 assert.equal((await f.enhance()).computed.treatments.length,2);
});

test('native static boolean RollOptions update a copied option set without running live beforeRoll hooks',()=>{
 const f=fixture(),options=new Set(['recognize-spell']);f.statistics.medicine.check.createRollOptions=()=>options;
 f.actor.rules.push(...[{option:'student-of-the-canon',value:true},{option:'recognize-spell',value:false}].map(data=>({key:'RollOption',domain:'all',phase:'applyAEs',suboptions:[],predicate:new Predicate(),...data,beforeRoll(){throw Error('live RollOption invoked')}})));
 const result=preparedTreatmentSelections({actor:f.actor,game:f.game,skill:'medicine'}).find(s=>!s.riskySurgery&&!s.assurance);assert.equal(result.ready,true,result.reason);assert.ok(result.options.includes('student-of-the-canon'));assert.ok(!result.options.includes('recognize-spell'));assert.deepEqual([...options],['recognize-spell']);
});

test('dynamic or suboption RollOptions remain unsupported rather than assuming their prepared value',()=>{
 for(const data of [{option:'{item|slug}',value:true},{option:'example',value:'@actor.level'},{option:'example',value:true,suboptions:[{value:'one'}]},{option:'example',value:true,selection:'hidden'},{option:'example',value:true,toggleable:'totm'},{option:'example',value:true,_source:{option:'{item|slug}'}}]){
  const f=fixture();f.actor.rules.push({key:'RollOption',domain:'all',predicate:new Predicate(),suboptions:[],...data,beforeRoll(){throw Error('live RollOption invoked')}});
  const result=preparedTreatmentSelections({actor:f.actor,game:f.game,skill:'medicine'}).find(s=>!s.riskySurgery&&!s.assurance);assert.equal(result.ready,false);
 }
});

test('check-specific domains are checked before a lazy StatisticCheck can run custom adjustments',()=>{
 const f=fixture();f.statistics.medicine.domains=['all','skill-check','medicine'];f.statistics.medicine.data={check:{domains:['custom-treatment-check']}};let reads=0;
 f.actor.synthetics.modifierAdjustments={'custom-treatment-check':[{}]};Object.defineProperty(f.statistics.medicine,'check',{get(){reads++;throw Error('unsafe getter')}});
 const result=preparedTreatmentSelections({actor:f.actor,game:f.game,skill:'medicine'}).find(s=>!s.riskySurgery&&!s.assurance);assert.equal(result.ready,false);assert.equal(reads,0);
});

function robustHealth(f){
 const rule={key:'FlatModifier',predicate:[{or:['action:battle-medicine','action:treat-wounds']}],selector:'healing-received',type:'circumstance',value:'@actor.level'};
 const item={uuid:'Actor.H.Item.Robust',type:'feat',slug:'robust-health',sourceId:'Compendium.pf2e.feats-srd.Item.yTLGclKtWVFZLKIz',_source:{system:{rules:[rule]}}};f.actor.level=5;f.actor.items.push(item);f.actor.rules.push({...rule,item});f.actor.synthetics.modifiers={'healing-received':[()=>{throw Error('qualification must not execute the factory')}]};return item;
}

test('canonical Robust Health qualifies its level circumstance bonus without invoking a modifier factory',async()=>{
 const f=fixture();robustHealth(f);const result=await f.enhance();assert.equal(result.computed.patient.healingReady,true);assert.equal(result.computed.patient.robustHealth,5);
});

test('altered or mixed healing-received modifiers cannot inherit the Robust Health model',async()=>{
 for(const mutate of [item=>item.sourceId='Compendium.other.Item.fake',item=>item._source.system.rules[0].value=20,item=>item._source.system.rules[0].predicate=[],(_item,f)=>f.actor.synthetics.modifiers['healing-received'].push(()=>null),(_item,f)=>f.actor.rules[0].ignored=true,(_item,f)=>f.actor.rules[0].type='status',(_item,f)=>f.actor.rules[0].critical=true,(_item,f)=>f.actor.rules[0].removeAfterRoll=true]){
  const f=fixture(),item=robustHealth(f);mutate(item,f);assert.equal((await f.enhance()).computed.patient.healingReady,false);
 }
});
