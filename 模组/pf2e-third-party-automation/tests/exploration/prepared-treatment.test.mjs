import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {preparedTreatmentSelections,simpleTreatmentDamageModel} from '../../scripts/exploration/prepared-treatment.mjs';
import {createCapabilities} from '../../scripts/exploration/capabilities.mjs';

class Predicate extends Array {
 test(options){return this.every(p=>typeof p==='string'?options.has(p):!options.has(p.not));}
 toObject(){return [...this];}
}
class Modifier {
 constructor(data){Object.assign(this,{enabled:true,ignored:false,adjustments:[],predicate:new Predicate()},data);}
 clone(){return new Modifier({...this,adjustments:[...this.adjustments]});}
}
class CheckModifier {
 constructor(_name,{modifiers},_extra,options){
  this.modifiers=modifiers.map(m=>m.clone());const types=new Map();
  for(const m of this.modifiers){assert.equal(m.adjustments.length,0);if(m.predicate.length)m.ignored=!m.predicate.test(options);if(m.ignored)continue;types.set(m.type,Math.max(types.get(m.type)??-Infinity,m.modifier));}
  this.totalModifier=[...types.values()].reduce((sum,n)=>sum+n,0);
 }
}
function fixture(){
 const risky={uuid:'Actor.H.Item.R',type:'feat',slug:'risky-surgery',sourceId:'Compendium.pf2e.feats-srd.Item.bkZgWFSFV4cAf5Ot',system:{rules:[
  {key:'RollOption',domain:'medicine',option:'risky-surgery',toggleable:true},
  {key:'FlatModifier',selector:'medicine',type:'circumstance',value:2,predicate:['action:treat-wounds','risky-surgery']},
  {key:'Note',selector:'medicine',predicate:['action:treat-wounds','risky-surgery'],text:'PF2E.SpecificRule.Feat.RiskySurgery.Note',title:'{item|name}'},
  {key:'AdjustDegreeOfSuccess',selector:'medicine',predicate:['risky-surgery','action:treat-wounds'],adjustment:{success:'one-degree-better'}}
 ]}};
 const assurance={uuid:'Actor.H.Item.A',type:'feat',slug:'assurance',sourceId:'Compendium.pf2e.feats-srd.Item.W6Gl9ePmItfDHji0',flags:{pf2e:{rulesSelections:{assurance:'medicine'}}},system:{rules:[
  {key:'ChoiceSet',choices:{config:'skills',predicate:[{gte:['skill:{choice|value}:rank',1]}]},flag:'assurance',prompt:'PF2E.SpecificRule.Prompt.Skill',selection:'medicine'},
  {key:'SubstituteRoll',label:'PF2E.SpecificRule.SubstituteRoll.Assurance',selector:'{item|flags.system.rulesSelections.assurance}',slug:'assurance',value:10},
  {key:'AdjustModifier',selector:'{item|flags.system.rulesSelections.assurance}',predicate:['substitute:assurance',{not:'bonus:type:proficiency'}],suppress:true}
 ]}};
 const adjustment={slug:null,suppress:true,test:()=>{throw Error('original adjustment closure called')},getNewValue:()=>{throw Error('original adjustment closure called')},getDamageType:()=>{throw Error('original adjustment closure called')}};
 const modifiers=[new Modifier({slug:'proficiency',type:'proficiency',modifier:4}),new Modifier({slug:'wis',type:'ability',modifier:3}),new Modifier({slug:'higher',type:'circumstance',modifier:3}),new Modifier({slug:'risky-surgery',type:'circumstance',modifier:2,predicate:new Predicate('action:treat-wounds','risky-surgery'),rule:{item:risky}})];
 for(const m of modifiers)m.adjustments=[adjustment];
 const check={domains:['all','skill-check','medicine','check','medicine-check'],modifiers,createRollOptions:({origin,extraRollOptions})=>new Set([...extraRollOptions,...origin.getSelfRollOptions('origin')])};
 const stat={rank:2,check,withRollOptions:()=>{throw Error('new Statistic construction forbidden')}};
 const actor={uuid:'Actor.H',items:[risky,assurance],rules:[{key:'RollOption',domain:'medicine',option:'risky-surgery',value:false,item:risky,beforeRoll:()=>{throw Error('beforeRoll called')}},{key:'AdjustModifier',item:assurance}],getStatistic:()=>stat,getRollOptions:domains=>domains.map(d=>`domain:${d}`),getSelfRollOptions:prefix=>[`${prefix}:level:8`],synthetics:{modifierAdjustments:{medicine:[adjustment]},rollSubstitutions:{medicine:[{slug:'assurance',value:10,required:false,selected:false,effectType:'fortune',predicate:new Predicate()}]},degreeOfSuccessAdjustments:{medicine:[{predicate:new Predicate('risky-surgery','action:treat-wounds'),adjustments:{success:{label:'Risky',amount:1}}}]}}};
 const game={system:{version:'8.5.1'},pf2e:{Predicate,CheckModifier}};
 return {actor,game,stat,check,modifiers,risky,assurance,adjustment};
}
const selected=(f,riskySurgery=false,assurance=false)=>preparedTreatmentSelections({game:f.game,actor:f.actor,skill:'medicine',slugs:['risky-surgery']}).find(s=>s.riskySurgery===riskySurgery&&s.assurance===assurance);

function nativeItemFixture(){
 const f=fixture(),capture=JSON.parse(readFileSync(new URL('./fixtures/prepared-treatment-native.json',import.meta.url),'utf8'));
 for(const item of [f.risky,f.assurance]){
  const saved=capture.items.find(row=>row.sourceId===item.sourceId);
  item._source={system:{rules:structuredClone(saved.rawRules)}};
  item.toObject=source=>{assert.equal(source,true);return structuredClone(item._source)};
  item.system.rules=structuredClone(saved.preparedRules);item.flags=structuredClone(saved.preparedFlags);item.rules=structuredClone(saved.rules);
 }
 return f;
}
test('native prepared schema defaults do not invalidate canonical raw Risky and Assurance rules',()=>{
 const f=nativeItemFixture(),before=[f.risky,f.assurance].map(item=>structuredClone({raw:item._source,system:item.system,flags:item.flags,rules:item.rules}));
 const s=selected(f,true,true);assert.equal(s.ready,true,s.reason);
 assert.deepEqual([f.risky,f.assurance].map(item=>({raw:item._source,system:item.system,flags:item.flags,rules:item.rules})),before);
});
test('raw source qualification supports _source without calling derived rule preparation',()=>{
 const f=nativeItemFixture();delete f.risky.toObject;delete f.assurance.toObject;
 assert.equal(selected(f,true,true).ready,true);
});
test('canonical-looking prepared rules cannot hide modified raw Risky or Assurance',()=>{
 for(const mutate of [f=>f.risky._source.system.rules[1].value=20,f=>f.assurance._source.system.rules[1].value=20,f=>f.assurance._source.system.rules[0].selection='nature',f=>f.risky._source.system.rules.push({key:'Unknown'})]){
  const f=nativeItemFixture();mutate(f);assert.equal(selected(f,true,true).ready,false);
 }
});
test('ignored canonical rules cannot be revived by raw source qualification',()=>{
 for(const mutate of [f=>f.risky.system.rules[1].ignored=true,f=>f.risky.rules[3].ignored=true,f=>f.assurance.rules[1].ignored=true,f=>f.assurance._source.system.rules[2].ignored=true]){
  const f=nativeItemFixture();mutate(f);assert.equal(selected(f,true,true).ready,false);
 }
});
test('raw canonical rules do not bypass source UUID or missing original document data',()=>{
 for(const mutate of [f=>f.risky.sourceId='Compendium.other.Item.fake',f=>f.assurance.sourceId='Compendium.other.Item.fake',f=>f.risky.toObject=()=>({system:{}})]){
  const f=nativeItemFixture();mutate(f);assert.equal(selected(f,true,true).ready,false);
 }
});

test('prepared stack wins over Risky +2 without removing its independent success upgrade',()=>{
 const f=fixture(),before=f.modifiers.map(m=>({...m}));const s=selected(f,true);
 assert.equal(s.ready,true);assert.equal(s.modifier,10);assert.equal(s.outcomesByRank[0].cases[4].outcome,3);
 assert.deepEqual(f.modifiers.map(m=>({...m})),before);assert.equal(f.actor.rules[0].value,false);
 assert.ok(s.options.includes('domain:medicine'));assert.ok(s.options.includes('origin:level:8'));assert.ok(!s.options.some(o=>o.startsWith('target:')));
});
test('Assurance uses actual proficiency with Risky upgrade and no non-proficiency modifiers',()=>{
 const f=fixture();f.modifiers[0].modifier=10;
 const s=selected(f,true,true);assert.equal(s.ready,true);assert.equal(s.modifier,10);
 assert.deepEqual(s.outcomesByRank[1].cases,[{weight:1,outcome:3}]);
 assert.equal(f.adjustment.applications,undefined);
});
test('unsupported version and an altered canonical source do not become ready',()=>{
 const f=fixture();f.game.system.version='8.6.0';assert.equal(selected(f).ready,false);
 f.game.system.version='8.5.1';f.risky.system.rules[1].value=20;assert.equal(selected(f,true).ready,false);
});
test('unknown adjustment functions, beforeRoll and fortune paths are never executed',()=>{
 for(const alter of [f=>f.actor.synthetics.modifierAdjustments.all=[{suppress:false,test:()=>{throw Error('unknown called')}}],f=>f.actor.rules.push({key:'Unknown',beforeRoll:()=>{throw Error('unknown called')}}),f=>f.actor.synthetics.rollTwice={medicine:[{keep:'higher',predicate:new Predicate()}]}]){const f=fixture();alter(f);assert.equal(selected(f).ready,false);}
});
test('unknown adjustment is rejected before lazy check or mod getters can run',async()=>{
 const f=fixture();let calls=0;
 f.actor.synthetics.modifierAdjustments.all=[{suppress:false,getNewValue:()=>calls++}];
 Object.defineProperty(f.stat,'check',{get(){calls++;return f.check}});
 Object.defineProperty(f.stat,'mod',{get(){calls++;return 10}});
 assert.equal(selected(f).ready,false);assert.equal(calls,0);
 f.game.time={worldTime:0};f.actor.system={attributes:{hp:{value:30,max:100,temp:0}}};
 const c=createCapabilities({game:f.game,fromUuid:async()=>f.actor,hpPools:{discover:actor=>({poolUUID:actor.uuid,ready:true})}});
 assert.equal((await c.discover(f.actor.uuid)).medicine.mod,null);assert.equal(calls,0);
});
test('actual healing-received modifier factories block the simple receiving model without invocation',async()=>{
 const f=fixture();let calls=0;f.actor.synthetics.modifiers={'healing-received':[()=>{calls++;return {modifier:3}}]};
 f.game.time={worldTime:0};f.actor.system={attributes:{hp:{value:30,max:100,temp:0}}};
 const c=createCapabilities({game:f.game,fromUuid:async()=>f.actor,hpPools:{discover:actor=>({poolUUID:actor.uuid,ready:true})}});
 assert.equal((await c.discover(f.actor.uuid)).healingExpectationReady,false);assert.equal(calls,0);
});
test('missing or conflicting actual Assurance substitution and another skill are unavailable',()=>{
 for(const alter of [f=>f.actor.synthetics.rollSubstitutions.medicine=[],f=>f.actor.synthetics.rollSubstitutions.medicine.push({slug:'other',value:15,required:true,selected:true}),f=>f.assurance.flags.pf2e.rulesSelections.assurance='nature']){const f=fixture();alter(f);assert.equal(selected(f,false,true).ready,false);}
});
test('unverified Risky modifier cannot be invented from a feat label',()=>{
 const f=fixture();f.check.modifiers=f.modifiers.filter(m=>m.slug!=='risky-surgery');assert.equal(selected(f,true).ready,false);
});
test('canonical item alone cannot prove an active suppression or a changed Risky toggle',()=>{
 const missing=fixture();let calls=0;missing.actor.rules=missing.actor.rules.filter(rule=>rule.key!=='AdjustModifier');Object.defineProperty(missing.stat,'check',{get(){calls++;return missing.check}});assert.equal(selected(missing,false,true).ready,false);assert.equal(calls,0);
 const changed=fixture();changed.actor.rules[0].value='@future';assert.equal(selected(changed,true).ready,false);
});
test('simple damage model requires explicit empty IWR, zero hardness/temp/stamina and an independent pool',()=>{
 const actor={uuid:'Actor.P',hardness:0,attributes:{immunities:[],resistances:[],weaknesses:[],hp:{temp:0,sp:{max:0}}},synthetics:{}};
 assert.equal(simpleTreatmentDamageModel({actor,poolUUID:actor.uuid}),true);
 for(const alter of [a=>delete a.attributes.immunities,a=>a.attributes.resistances.push({type:'slashing'}),a=>a.hardness=1,a=>a.attributes.hp.temp=1,a=>a.attributes.hp.sp.max=10]){const copy=structuredClone(actor);alter(copy);assert.equal(simpleTreatmentDamageModel({actor:copy,poolUUID:copy.uuid}),false);}
 assert.equal(simpleTreatmentDamageModel({actor,poolUUID:'Actor.Master'}),false);
});
test('capabilities publishes exact selections and patient damage readiness without a native action',async()=>{
 const f=fixture();f.game.time={worldTime:0};f.actor.system={attributes:{hp:{value:30,max:100,temp:0}}};f.actor.hardness=0;f.actor.attributes={...f.actor.system.attributes,immunities:[],resistances:[],weaknesses:[]};
 const capabilities=createCapabilities({game:f.game,fromUuid:async()=>f.actor,hpPools:{discover:actor=>({poolUUID:actor.uuid,ready:true})}});
 const result=await capabilities.discover(f.actor.uuid);
 assert.equal(result.damageExpectationReady,true);
 assert.equal(result.treatmentEstimate.medicine.selections.find(s=>s.assurance&&s.riskySurgery).ready,true);
 assert.equal(result.treatmentEstimate.medicine.source.actorUUID,f.actor.uuid);
});
