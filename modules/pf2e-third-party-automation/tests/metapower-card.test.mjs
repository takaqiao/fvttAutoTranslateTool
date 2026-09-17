import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/metapower/card.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const snapshot=(changes={})=>({kind:'siphoning',powerId:'electric-surge',level:5,discharge:true,siphon:{applies:true},area:{type:'line',baseDistance:60,distance:30},...changes});
test('selected discharge formula and area are immutable across renders and ordinary scaling is retained',()=>{
 assert.equal(typeof api.cardLinkPlan,'function');const s=snapshot(),links=[{kind:'damage',baseFormula:'6d4[electricity]'},{kind:'area',type:'line',distance:20},{kind:'effect',uuid:'Compendium.battlezoo-eldamon-pf2e.conditions.Bi2aHykg6CZrQCnR'},{kind:'area',type:'line',distance:40},{kind:'damage',baseFormula:'6d4[electricity]'}];
 const a=api.cardLinkPlan(s,links);assert.equal(a[0].disabled,true);assert.equal(a[4].formula,'(1+5)d8[electricity]');assert.equal(a[2].disabled,true);assert.equal(a[3].distance,30);
 assert.deepEqual(api.cardLinkPlan(s,links),a);
});
test('fixed-failure damage stays fixed and chain uses confirmed triggering damage',()=>{
 const staticPlan=api.cardLinkPlan(snapshot({powerId:'static-shock',discharge:false}),[{kind:'damage'},{kind:'damage'},{kind:'damage'}]);assert.equal(staticPlan[2].formula,'(2+5)[electricity]');assert.equal(staticPlan[1].disabled,true);
 const chain=api.cardLinkPlan(snapshot({powerId:'reactive-chain',triggerDamage:19}),[{kind:'damage'}]);assert.equal(chain[0].formula,'9[electricity]');
});
test('Widen never suppresses native effect links or expands a nonmatching shape',()=>{
 const links=[{kind:'area',type:'line',distance:30},{kind:'area',type:'emanation',distance:10},{kind:'effect',uuid:'x'}];
 const plan=api.cardLinkPlan(snapshot({kind:'widen',siphon:{applies:false},discharge:false,area:{type:'line',distance:40}}),links);
 assert.equal(plan[0].distance,40);assert.equal(plan[1].distance,10);assert.equal(plan[2].disabled,undefined);
});
test('High Voltage has no immediate damage link on its activation card',()=>{
 const plan=api.cardLinkPlan(snapshot({powerId:'high-voltage'}),[{kind:'damage',baseFormula:'4d6[electricity]'}]);assert.equal(plan[0].disabled,true);
});
test('Electric Shot keeps fixed failure and offers the selected branch base for Shocked half-failure',()=>{
 const plan=api.cardLinkPlan(snapshot({powerId:'electric-shot'}),[{kind:'damage'},{kind:'damage'},{kind:'damage'}]);
 assert.equal(plan[2].formula,'5[electricity]');assert.equal(plan[2].shockedFailureFormula,'(2+5)d8[electricity]');
});
