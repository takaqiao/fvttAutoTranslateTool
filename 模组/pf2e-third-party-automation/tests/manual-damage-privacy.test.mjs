import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/damage-message-privacy.mjs');}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
test('combined damage keeps a blind part restricted when the other part was public',()=>{
 assert.equal(typeof api.mergeDamageMessagePrivacy,'function');
 assert.deepEqual(api.mergeDamageMessagePrivacy([{blind:false,whisper:[]},{blind:true,whisper:['gm']}]),{blind:true,whisper:['gm']});
});
test('combining private parts uses their common recipients and does not expose one to a different player',()=>{
 assert.equal(typeof api.mergeDamageMessagePrivacy,'function');
 assert.deepEqual(api.mergeDamageMessagePrivacy([{blind:false,whisper:['owner','gm']},{blind:false,whisper:['gm','other']}]),{blind:false,whisper:['gm']});
 assert.throws(()=>api.mergeDamageMessagePrivacy([{blind:false,whisper:['owner']},{blind:false,whisper:['gm']}]),/可见|受众/);
 assert.throws(()=>api.mergeDamageMessagePrivacy([{blind:true,whisper:[]}]),/可见|受众/);
});
test('public damage remains public and private source data is not mutated by merging',()=>{
 assert.equal(typeof api.mergeDamageMessagePrivacy,'function');const parts=[{blind:false,whisper:['owner','gm']},{blind:true,whisper:['gm']}];
 const copy=structuredClone(parts);api.mergeDamageMessagePrivacy(parts);assert.deepEqual(parts,copy);
 assert.deepEqual(api.mergeDamageMessagePrivacy([{blind:false,whisper:[]},{blind:false,whisper:[]}]),{blind:false,whisper:[]});
});
