import test from 'node:test';
import assert from 'node:assert/strict';
import {buildDefensiveAdvancePatreonRepairs,DEFENSIVE_ADVANCE_SOURCE,RAISED_SHIELD_SOURCE} from '../scripts/defensive-advance-compat.mjs';
import {USE_ACTION_OPTION} from '../scripts/usage-events.mjs';

const id='YmDi9KpCSdq3ATRP';
const original=()=>({[id]:{uuid:id,source:[DEFENSIVE_ADVANCE_SOURCE],isActive:true,baseRules:[{type:'base',triggerType:'postInfo',target:'SelfEffect',value:RAISED_SHIELD_SOURCE,predicate:['origin:item:defensive-advance']}]},unrelated:{baseRules:[{predicate:['custom'],value:'other'}]}});

test('Defensive Advance keeps the exact upstream shield executor but requires actual Use and an unraised shield',()=>{
 const before=original(),result=buildDefensiveAdvancePatreonRepairs(before),rule=result.rules[id].baseRules[0];
 assert.deepEqual(before,original());assert.equal(rule.target,'SelfEffect');assert.equal(rule.value,RAISED_SHIELD_SOURCE);
 assert.deepEqual(rule.predicate,['origin:item:defensive-advance',USE_ACTION_OPTION,{not:'self:shield:raised'}]);
 assert.deepEqual(result.rules.unrelated,before.unrelated);assert.equal(result.changes.length,1);
 assert.deepEqual(result.changes[0].before,before[id].baseRules[0].predicate);assert.deepEqual(result.changes[0].after,rule.predicate);
});

test('repeated configuration maintenance is a no-op and preserves custom predicate constraints',()=>{
 const before=original();before[id].baseRules[0].predicate.push('custom:held');
 const first=buildDefensiveAdvancePatreonRepairs(before),second=buildDefensiveAdvancePatreonRepairs(first.rules);
 assert.ok(second.rules[id].baseRules[0].predicate.includes('custom:held'));assert.deepEqual(second.rules,first.rules);assert.deepEqual(second.changes,[]);
});

test('different rule identity or trigger/effect/target is never rewritten',()=>{
 for(const alter of [r=>{r.other=r[id];delete r[id]},r=>r[id].source.push('custom'),r=>r[id].uuid='changed',r=>r[id].baseRules[0].triggerType='skill-check',r=>r[id].baseRules[0].target='TargetEffect',r=>r[id].baseRules[0].value='other',r=>r[id].baseRules[0].predicate=['custom-only']]){
  const before=original();alter(before);const result=buildDefensiveAdvancePatreonRepairs(before);assert.deepEqual(result.rules,before);assert.deepEqual(result.changes,[]);
 }
});

test('disabled rules stay disabled and duplicate gates are not appended',()=>{
 const before=original();before[id].isActive=false;before[id].baseRules[0].predicate.push(USE_ACTION_OPTION);
 const result=buildDefensiveAdvancePatreonRepairs(before);assert.equal(result.rules[id].isActive,false);assert.equal(result.rules[id].baseRules[0].predicate.filter(p=>p===USE_ACTION_OPTION).length,1);
});
