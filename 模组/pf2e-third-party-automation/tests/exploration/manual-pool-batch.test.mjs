import test from 'node:test';
import assert from 'node:assert/strict';
import {deduplicatePoolEffects} from '../../scripts/exploration/hp-pool.mjs';

test('one effect chooses its largest pool contribution while an independent effect remains applicable',()=>{
 const effects=[{poolUUID:'M',effectId:'U1:R0:healing',amount:5},{poolUUID:'M',effectId:'U1:R0:healing',amount:10},{poolUUID:'M',effectId:'U2:R0:healing',amount:7}];
 assert.deepEqual(deduplicatePoolEffects(effects).map(effect=>effect.amount),[10,7]);
 assert.deepEqual(deduplicatePoolEffects([effects[1],effects[0],effects[2]]).map(effect=>effect.amount),[10,7]);
});
test('equal contributions retain the original target order',()=>{
 const first={poolUUID:'M',effectId:'R0:healing',amount:10,patientUUID:'A'},second={poolUUID:'M',effectId:'R0:healing',amount:10,patientUUID:'B'};
 assert.equal(deduplicatePoolEffects([first,second])[0],first);assert.equal(deduplicatePoolEffects([second,first])[0],second);
});
test('independent result and pool identities are not merged by a common use',()=>{
 const effects=[{poolUUID:'M',effectId:'U:R1:0:healing',amount:5},{poolUUID:'M',effectId:'U:R2:0:healing',amount:10},{poolUUID:'N',effectId:'U:R1:0:healing',amount:7}];
 assert.deepEqual(deduplicatePoolEffects(effects).map(effect=>effect.amount),[5,10,7]);
});
