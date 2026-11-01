import test from 'node:test';
import assert from 'node:assert/strict';
import {createFortressRuleCompatibility} from '../scripts/fortress-rule-compat.mjs';
import {Predicate,degree} from './knowledge-pf2e-861-fixture.mjs';
let api={};try{api=await import('../scripts/bob2-rule-links.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}

const ID='pf2e-third-party-automation';
const BLOOD='Compendium.pf2e.feat-effects.Item.52owimBcCQNBuNfM',OVERTURE='Compendium.pf2e.spell-effects.Item.ITErgFRfydm1xmnW';
const mindRules=()=>[
 {key:'ChoiceSet',rollOption:'blooded-by-the-dead',selection:'bolster-the-mind'},
 {key:'Note',predicate:['blooded-by-the-dead:bolster-the-fury','item:trait:incapacitation'],selector:['saving-throw'],text:'PF2E.SpecificRule.BastionOfBlasphemies.Background.BloodedByTheDead.BolsterTheMind.Note'},
 {key:'FlatModifier',predicate:['blooded-by-the-dead:bolster-the-mind','item:slug:dominate','origin:trait:vampire'],selector:['will'],value:2},
];
function fixture(source=BLOOD,rules=mindRules()){
 const item={id:'effect',type:'effect',sourceId:source,flags:{},system:{rules,expired:false},isExpired:false};
 const actor={id:'hero',uuid:'Actor.hero',level:5,flags:{},items:new Map([[item.id,item]]),updates:[],async updateEmbeddedDocuments(_type,patches){for(const patch of patches){this.updates.push(patch);item.system.rules=patch['system.rules'];item.flags[ID]={fortressRuleRepair:patch[`flags.${ID}.fortressRuleRepair`]}}}};item.actor=actor;
 const gm={id:'gm'},game={world:{id:'bob-2'},system:{id:'pf2e'},user:gm,users:{activeGM:gm},actors:new Map([[actor.id,actor]])};return {actor,item,game};
}
test('Bolster the Mind repairs only captured source rules and retains their original backup',async()=>{
 const f=fixture(),before=structuredClone(f.item.system.rules),provider=createFortressRuleCompatibility({game:f.game});
 await provider.maintain(f.actor);await provider.maintain(f.actor);
 assert.equal(f.actor.updates.length,1);
 const note=f.item.system.rules[1],bonus=f.item.system.rules[2];
 assert.ok(new Predicate(note.predicate).test(new Set(['blooded-by-the-dead:bolster-the-mind','item:trait:incapacitation','item:trait:mental'])));
 assert.equal(bonus.type,'circumstance');assert.equal(bonus.value,2);
 assert.deepEqual(f.item.flags[ID].fortressRuleRepair.before['system.rules'],before);
});
test('Bolster repairs preserve custom predicates, other sources and other worlds',async()=>{
 for(const alter of [f=>f.item.sourceId='custom',f=>f.game.world.id='other',f=>{f.item.system.rules[1].predicate.push('custom');f.item.system.rules[2].type='status'}]){
  const f=fixture();alter(f);const before=structuredClone(f.item.system.rules);await createFortressRuleCompatibility({game:f.game}).maintain(f.actor);assert.deepEqual(f.item.system.rules,before);
 }
});
test('level 5 mental incapacitation uses effective level 7 for rank 3, retaining the real actor and native degree',()=>{
 const f=fixture(),context={actor:f.actor,type:'saving-throw',item:{type:'spell',rank:3},dc:{value:20},options:new Set(['item:trait:mental','item:trait:incapacitation']),dosAdjustments:[]};
 const next=api.bolsterMindCheckContext?.(context)??context;
 assert.equal(degree({total:18,die:10,dc:20,dosAdjustments:next.dosAdjustments}),2);
 assert.equal(f.actor.level,5);assert.equal(next.actor,f.actor);assert.deepEqual(context.dosAdjustments,[]);
});
test('Bolster excludes nonmental, expired, wrong choice and already eligible checks',()=>{
 for(const alter of [c=>c.options.delete('item:trait:mental'),c=>c.options.delete('item:trait:incapacitation'),c=>c.item.rank=4,c=>c.actor.items.get('effect').isExpired=true,c=>c.actor.items.get('effect').system.rules[0].selection='bolster-the-fury',c=>c.actor.level=7]){
  const f=fixture(),context={actor:f.actor,type:'saving-throw',item:{type:'spell',rank:3},dc:{value:20},options:new Set(['item:trait:mental','item:trait:incapacitation']),dosAdjustments:[]};alter(context);
  assert.equal(api.bolsterMindCheckContext?.(context)??context,context);
 }
});
test('Bolster also lowers an attacker check against the mentally bolstered target without mutating either actor',()=>{
 const f=fixture(),attacker={level:6},context={actor:attacker,target:{actor:f.actor},type:'spell-attack-roll',item:{type:'spell',rank:3},dc:{value:20},options:new Set(['mental','incapacitation']),dosAdjustments:[]};
 const next=api.bolsterMindCheckContext?.(context)??context;
 assert.equal(degree({total:21,die:10,dc:20,dosAdjustments:next.dosAdjustments}),1);assert.equal(f.actor.level,5);assert.equal(attacker.level,6);
});
test('Uplifting both rank branches affect Aid only and preserve native critical failure and natural dice',async()=>{
 const rules=[{key:'AdjustDegreeOfSuccess',selector:'performance',predicate:[{lte:['skill:performance:rank',3]}],adjustment:{success:'one-degree-better'}},{key:'AdjustDegreeOfSuccess',selector:'performance',predicate:[{gte:['skill:performance:rank',4]}],adjustment:{all:'to-critical-success'}}],f=fixture(OVERTURE,rules);
 await createFortressRuleCompatibility({game:f.game}).maintain(f.actor);
 const dos=rank=>f.item.system.rules.map(r=>({predicate:new Predicate(r.predicate),adjustments:Object.fromEntries(Object.entries(r.adjustment).map(([k,v])=>[k,{amount:v==='to-critical-success'?'criticalSuccess':1,label:'Uplifting'}]))}));
 const check=(rank,total,aid=true,die=10)=>degree({total,die,dc:20,rollOptions:[`skill:performance:rank:${rank}`,...aid?['action:aid']:[]],dosAdjustments:dos(rank)});
 assert.equal(check(2,18),2);assert.equal(check(2,21),2);assert.equal(check(2,8),0);assert.equal(check(4,8),3);
 assert.equal(check(2,18,false),1);assert.equal(check(4,8,false),0);assert.equal(check(2,21,true,1),2);
});
