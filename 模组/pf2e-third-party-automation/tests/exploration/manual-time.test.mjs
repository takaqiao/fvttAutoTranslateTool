import {test} from 'node:test';
import assert from 'node:assert/strict';
import {normalizeManualActivity} from '../../scripts/exploration/manual-time.mjs';

const declaration=patch=>({actorUUID:'Actor.A',label:' 搜索 ',durationSeconds:37,...patch});

test('a declaration is trimmed, keeps its timing and names each dependency once',()=>{
 const a=normalizeManualActivity(declaration({durationSource:{type:'item-text',detail:' 物品条目所述37秒 '},order:2,dependsOn:['first','first'],notBefore:-980,observedStart:-980,observedEnd:-943}));
 assert.deepEqual(a,{actorUUID:'Actor.A',label:'搜索',durationSeconds:37,durationSource:{type:'item-text',detail:'物品条目所述37秒'},dependsOn:['first'],order:2,notBefore:-980,observedStart:-980,observedEnd:-943});
});

for(const patch of [{durationSeconds:-1},{durationSeconds:NaN},{durationSeconds:Infinity},{order:-1},{order:0.5},{order:Number.MAX_SAFE_INTEGER+1},{notBefore:Infinity},{observedStart:-20},{observedEnd:-10},{observedStart:-10,observedEnd:-20},{observedStart:NaN,observedEnd:0},{durationSource:{type:'native'}},{durationSource:{type:'item-text',detail:23}},{dependsOn:['']},{dependsOn:'id'}])test('invalid declaration rejected before persistence '+JSON.stringify(patch),()=>{
 assert.throws(()=>normalizeManualActivity(declaration(patch)),/^Error: invalid-manual-/);
});

test('sparse dependency arrays are invalid rather than creating an undefined reference',()=>{
 assert.throws(()=>normalizeManualActivity(declaration({dependsOn:new Array(1)})),/invalid-manual-dependency/);
});
