import {test} from 'node:test';
import assert from 'node:assert/strict';
import {classifyElectricityDamage,electricityRemovalPlan,sourceTurnExpiry,expiryReached,chainEligibility,receiptElectricityAmount} from '../scripts/eldamon-electricity.mjs';

test('only positive nonpersistent electricity instances establish the pure electricity path',()=>{
 assert.equal(classifyElectricityDamage({instances:[{type:'electricity',persistent:false,total:13}]}),'pure');
 assert.equal(classifyElectricityDamage({instances:[{type:'electricity',persistent:true,total:13}]}),'none');
 assert.equal(classifyElectricityDamage({instances:[{type:'untyped',total:13}]}),'none');
 assert.equal(classifyElectricityDamage({instances:[{type:'electricity',total:0}]}),'none');
 assert.equal(classifyElectricityDamage({instances:[{type:'electricity',total:3},{type:'slashing',total:10}]}),'mixed');
});
test('electricity removal reduces charge once; own-power shell charge survives while independent Shocked clears',()=>{
 assert.deepEqual(electricityRemovalPlan({charged:3,ownPowerCharge:false,shell:false,independentShocked:true}),{charged:2,removeShocked:false});
 assert.deepEqual(electricityRemovalPlan({charged:2,ownPowerCharge:true,shell:true,independentShocked:true}),{charged:2,removeShocked:true});
 assert.deepEqual(electricityRemovalPlan({charged:1,ownPowerCharge:false,shell:true}),{charged:0,removeShocked:false});
 assert.deepEqual(electricityRemovalPlan({charged:0}),{charged:0,removeShocked:true});
});
test('next-turn expiry belongs to source combatant, never target initiative',()=>{
 const combat={id:'c',started:true,round:3,turn:2,turns:[{id:'target',actor:{uuid:'Actor.target'}},{id:'source',actor:{uuid:'Actor.source'}},{id:'other'}]};
 const expiry=sourceTurnExpiry(combat,'Actor.source');
 assert.deepEqual(expiry,{combatId:'c',combatantId:'source',round:4,phase:'end'});
 assert.equal(expiryReached(expiry,{...combat,round:4},combat.turns[0],'end'),false);
 assert.equal(expiryReached(expiry,{...combat,round:4},combat.turns[1],'start'),false);
 assert.equal(expiryReached(expiry,{...combat,round:4},combat.turns[1],'end'),true);
});
test('chain validates both ranges, same effect victims, enemy, and normal discharge adjacency to caster',()=>{
 const base={triggerDamage:13,sourceDistance:30,targetDistance:30,adjacentCaster:false,enemy:true,shocked:true,hitBySameEffect:false,reactionAvailable:true};
 assert.equal(chainEligibility(base),true);
 for(const patch of [{sourceDistance:35},{targetDistance:35},{hitBySameEffect:true},{enemy:false},{shocked:false},{reactionAvailable:false}])assert.equal(chainEligibility({...base,...patch}),false);
 assert.equal(chainEligibility({...base,shocked:false,discharge:true,adjacentCaster:true}),true);
 assert.equal(chainEligibility({...base,shocked:false,discharge:true,adjacentCaster:true,siphoning:true}),false);
});
test('receipt needs exact application nonce and native applied electricity; rolled and mixed totals never qualify',()=>{
 const record={nonce:'application1',actorUuid:'Actor.target',tokenUuid:'Scene.s.Token.t',userId:'gm',sourceItemUuid:'Actor.source.Item.power',kind:'pure'};
 const message={id:'receipt',author:{id:'gm'},speaker:{actor:'target',scene:'s',token:'t'},flags:{pf2e:{context:{type:'damage-taken',options:['pf2e-third-party-automation:electricity-apply:application1']},origin:{uuid:record.sourceItemUuid},appliedDamage:{uuid:record.actorUuid,isHealing:false,updates:[{path:'system.attributes.hp.value',value:4}]}},'pf2e-third-party-automation':{electricityApplied:{nonce:'application1',amount:13}}}};
 assert.equal(receiptElectricityAmount(message,record),13); // overkill is still damage; HP loss alone is insufficient
 assert.equal(receiptElectricityAmount(message,{...record,nonce:'other'}),null);
 assert.equal(receiptElectricityAmount(message,{...record,kind:'mixed'}),null);
 assert.equal(receiptElectricityAmount({...message,flags:{...message.flags,'pf2e-third-party-automation':{}}},record),null);
 message.flags['pf2e-third-party-automation'].electricityApplied.amount=0;
 assert.equal(receiptElectricityAmount(message,record),0);
});
