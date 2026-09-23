import test from 'node:test';
import assert from 'node:assert/strict';
import {medicAction,visitationBranches,validateTreatment,treatmentValue} from '../scripts/medic-rules.mjs';
const feat=id=>({_stats:{compendiumSource:`Compendium.pf2e.feats-srd.Item.${id}`},type:'feat',name:'Renamed',system:{traits:{value:[]}}});
const actor=()=>({items:[feat('rfnEcjxIFqwlJwJT'),feat('wYerMk6F1RZb0Fwt'),{type:'equipment',_stats:{compendiumSource:'Compendium.pf2e.equipment-srd.Item.s1vB3HdXjMigYAnY'},system:{quantity:1,equipped:{carryType:'worn'}}}],handsFree:1});
const condition=()=>({id:'c',type:'condition',slug:'clumsy',system:{value:{value:2}}});
test('routes exact feat sources despite rename and missing traits, not a same-name foreign item',()=>{
 assert.equal(medicAction(feat('rfnEcjxIFqwlJwJT')),'medic:treat-condition');
 assert.equal(medicAction(feat('1fBHZpM3Z3MQtzvi')),'medic:doctors-visitation');
 assert.equal(medicAction({...feat('other'),name:'Treat Condition',system:{slug:'treat-condition'}}),null);
});
test('ordinary Treat Wounds and standalone Battle Medicine remain native instead of entering the Visitation settlement',()=>{assert.equal(medicAction({type:'action',system:{slug:'treat-wounds'}}),null);assert.equal(medicAction(feat('wYerMk6F1RZb0Fwt')),null);});
test('four legal Visitation branches carry activity total costs; Treat Condition requires its exact feat',()=>{
 assert.deepEqual(visitationBranches(actor()).map(b=>[b.value,b.cost]),[['battle-medicine',1],['treat-poison',1],['administer-first-aid',2],['treat-condition',2]]);
 assert.equal(visitationBranches({items:[]}).some(b=>b.value==='treat-condition'),false);
});
for(const [degree,want] of [[0,3],[1,2],[2,1],[3,0]])test(`native degree ${degree} applies exact condition table`,()=>assert.equal(treatmentValue(2,degree),want));
test('reduction floors at zero and rejects an invented degree',()=>{assert.equal(treatmentValue(1,3),0);assert.throws(()=>treatmentValue(2,4));});
test('adjacent target and worn tools with a free hand permit a source-based check',()=>assert.equal(validateTreatment({actor:actor(),condition:condition(),distance:5,facts:{dc:22,restricted:false,continuous:false}}).dc,22));
for(const [name,change] of [
 ['wrong condition',x=>x.condition.slug='frightened'],['no tools',x=>x.actor.items=[]],['occupied hands',x=>x.actor.handsFree=0],['too far',x=>x.distance=10],['unknown distance',x=>x.distance=null],['missing DC',x=>delete x.facts.dc],['unknown circumstances',x=>delete x.facts.continuous],['artifact without Legendary Medic',x=>x.facts.restricted=true],['continuous source',x=>x.facts.continuous=true],['granted condition',x=>x.condition.flags={pf2e:{grantedBy:{id:'parent'}}}]
])test(`reject ${name} before rolling`,()=>{const x={actor:actor(),condition:condition(),distance:5,facts:{dc:22,restricted:false,continuous:false}};change(x);assert.throws(()=>validateTreatment(x));});
test('Legendary Medic source permits exceptional source with DC +10',()=>{const a=actor();a.items.push(feat('Kk4AMZtpQnLEgN0b'));assert.equal(validateTreatment({actor:a,condition:condition(),distance:5,facts:{dc:30,restricted:true,continuous:false}}).dc,40);});
test('an expanded toolkit has the same worn/held access requirement',()=>{const a=actor();a.items.at(-1)._stats.compendiumSource='Compendium.pf2e.equipment-srd.Item.SGkOHFyBbzWdBk8D';assert.equal(validateTreatment({actor:a,condition:condition(),distance:5,facts:{dc:22,restricted:false,continuous:false}}).dc,22);});
