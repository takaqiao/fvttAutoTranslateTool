import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createTimeEffects} from '../../scripts/exploration/time-effects.mjs';

test('a prepared adapter is retired when another selected provider blocks',async()=>{
 const retired=[];
 const effects=createTimeEffects({capabilities:{activePassiveRules:async()=>[{providerId:'one',passing:true},{providerId:'two',passing:true}]},completionAdapters:[
  {matches:rule=>rule.providerId==='one',beforeAdvance:async()=>({status:'ready'}),cancel:(checkpoint,reason)=>retired.push({id:checkpoint.id,reason})},
  {matches:rule=>rule.providerId==='two',beforeAdvance:async()=>({status:'blocked',reason:'unavailable'})}
 ]});
 await effects.beforeAdvance({id:'C'});
 assert.deepEqual(retired,[{id:'C',reason:'unavailable'}]);
});

test('runtime invalidation retires prepared effects and prevents settlement',async()=>{
 const reasons=[];
 const effects=createTimeEffects({capabilities:{activePassiveRules:async()=>[{passing:true}]},completionAdapters:[{matches:()=>true,beforeAdvance:async()=>({status:'ready'}),invalidate:reason=>reasons.push(reason),settle:async()=>({status:'ready',proof:[{}]})}]});
 await effects.beforeAdvance({id:'C'});
 assert.equal(typeof effects.invalidate,'function');
 effects.invalidate('client-disconnected');
 assert.deepEqual(reasons,['client-disconnected']);
 assert.equal((await effects.settle({id:'C'})).status,'uncertain');
});
