import test from 'node:test';
import assert from 'node:assert/strict';
let runtime={};try{runtime=await import('../scripts/runtime.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
test('actions require actor ownership',()=>{
 assert.equal(typeof runtime.requireOwner,'function');
 assert.throws(()=>runtime.requireOwner({type:'character',testUserPermission:()=>false},{id:'p'}),/权限/);
 assert.doesNotThrow(()=>runtime.requireOwner({type:'character',testUserPermission:()=>true},{id:'p'}));
});
test('same actor operations serialize including after a failure',async()=>{
 assert.equal(typeof runtime.SerialActions,'function');
 const q=new runtime.SerialActions(),events=[];
 const a=q.run('a',async()=>{events.push(1);await new Promise(r=>setTimeout(r,15));events.push(2);throw Error('expected')});
 const b=q.run('a',async()=>events.push(3));
 await assert.rejects(a,/expected/);await b;assert.deepEqual(events,[1,2,3]);
});
test('magic item selection rejects consumables, zero quantity and nonmagical items',()=>{
 assert.equal(typeof runtime.isLegendEligible,'function');
 assert.equal(runtime.isLegendEligible({type:'consumable',isMagical:true,quantity:1}),false);
 assert.equal(runtime.isLegendEligible({type:'weapon',isMagical:true,quantity:0}),false);
 assert.equal(runtime.isLegendEligible({type:'weapon',isMagical:false,quantity:1}),false);
 assert.equal(runtime.isLegendEligible({type:'weapon',isMagical:true,quantity:1}),true);
});
