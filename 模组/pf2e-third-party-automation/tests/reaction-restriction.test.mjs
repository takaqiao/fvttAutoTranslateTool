import test from 'node:test';
import assert from 'node:assert/strict';
import {reactionRestrictionStatus,reactionPermitted,requireReactionPermitted} from '../scripts/reaction-restriction.mjs';
const actor=Object.freeze({uuid:'Actor.test'});
test('no query preserves old behavior and a clear query reads the exact actor',()=>{
 assert.equal(reactionPermitted(actor),true);assert.doesNotThrow(()=>requireReactionPermitted(actor));
 assert.equal(reactionRestrictionStatus(actor,a=>{assert.equal(a,actor);return {status:'clear',sources:[]}}),'clear');
});
test('a restricted source is distinct from an unproven query and neither mutates the actor',()=>{
 const query=()=>({status:'restricted',sources:[{nonce:'one-source'}]});assert.equal(reactionPermitted(actor,query),false);
 assert.throws(()=>requireReactionPermitted(actor,query),/禁止.*反应/);assert.deepEqual(actor,{uuid:'Actor.test'});
});
for(const [name,query]of [['manual',()=>({status:'manual'})],['unknown',()=>({status:'unknown'})],['missing',()=>undefined],['throw',()=>{throw Error('query unavailable')}],['bad callback',true]])test(`${name} is manual review, not a proven rules prohibition`,()=>{
 assert.equal(reactionRestrictionStatus(actor,query),'manual');assert.equal(reactionPermitted(actor,query),false);
 assert.throws(()=>requireReactionPermitted(actor,query),error=>/待 GM 核对/.test(error.message)&&!/禁止/.test(error.message));
});
