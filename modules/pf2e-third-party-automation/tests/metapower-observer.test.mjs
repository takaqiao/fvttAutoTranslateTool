import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/metapower/observer.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const ID='pf2e-third-party-automation';
test('native use is awaited, original card is stamped, and descriptions outside use are untouched',async()=>{
 assert.equal(typeof api.createMetapowerObserver,'function');const actor={uuid:'Actor.a',id:'a'},item={actor,uuid:'Actor.a.Item.i',id:'i'};const events=[];
 const observer=api.createMetapowerObserver({request:async(method,p)=>{events.push(method);return {nonce:p.nonce,actorUuid:actor.uuid,itemUuid:item.uuid,status:method==='begin'?'reserved':'started'}},id:()=> 'nonce',select:async()=>({})});
 const description={speaker:{actor:'a'},flags:{pf2e:{origin:{uuid:item.uuid}}}};assert.equal(observer.decorate(description),description);assert.equal(description.flags[ID],undefined);
 let calls=0;const result=await observer.observe({actor,item},async()=>{calls++;const data=observer.decorate(structuredClone(description));assert.equal(data.flags[ID].metapowerUse.nonce,'nonce');observer.record([{...data,id:'m',uuid:'ChatMessage.m'}]);return {native:true}});
 assert.deepEqual(result,{native:true});assert.equal(calls,1);assert.deepEqual(events,['begin','start','finish']);
 assert.equal(observer.decorate(structuredClone(description)).flags[ID],undefined);
});
test('native failure is archived uncertain and rethrows without rerunning native',async()=>{
 const events=[],actor={uuid:'a'},item={uuid:'i',actor};const observer=api.createMetapowerObserver({request:async(method,p)=>{events.push([method,p.status]);return {nonce:'n',status:'reserved'}},id:()=> 'n',select:async()=>({})});
 const error=Error('side effect already happened');let calls=0;await assert.rejects(observer.observe({actor,item},async()=>{calls++;throw error}),e=>e===error);assert.equal(calls,1);assert.deepEqual(events.at(-1),['finish','uncertain']);
});
test('native draft or null does not become proof of cancellation or an activation',async()=>{
 const events=[],actor={uuid:'a'},item={uuid:'i',actor};const observer=api.createMetapowerObserver({request:async(method,p)=>{events.push([method,p.status]);return {nonce:p.nonce,status:'reserved'}},id:()=> 'n',select:async()=>({})});
 assert.equal(await observer.observe({actor,item},async()=>null),null);assert.deepEqual(events.at(-1),['finish','uncertain']);
});
test('cancelled branch choice never admits or pays for native use',async()=>{
 let calls=0;const observer=api.createMetapowerObserver({request:()=>{throw Error('must not call')},select:async()=>null});
 assert.equal(await observer.observe({actor:{uuid:'a'},item:{uuid:'i'}},()=>calls++),null);assert.equal(calls,0);
});
test('verified native check dialog cancellation has an explicit no-execution result',async()=>{
 const events=[],observer=api.createMetapowerObserver({request:async(method,p)=>{events.push([method,p.status,p.confirmation]);return {nonce:p.nonce,status:'reserved'}},id:()=> 'n'});
 assert.equal(await observer.observe({actor:{uuid:'a'},entry:'native-check'},async()=>null),null);
 assert.deepEqual(events.at(-1),['finish','cancelled','native-check-no-result']);
});
test('native self-effect card bypassing item.toMessage binds the actual embedded item',async()=>{
 const actor={uuid:'Actor.a',id:'a'},item={uuid:'Actor.a.Item.i',id:'i',actor};let completed;
 const observer=api.createMetapowerObserver({request:async(method,p)=>{if(method==='finish')completed=p;return {nonce:p.nonce,status:'reserved'}},select:async()=>({}),id:()=> 'n'});
 await observer.observe({actor,item},async()=>{const data=observer.decorate({speaker:{actor:'a'},flags:{pf2e:{context:{type:'self-effect',item:'i'}}}});assert.equal(data.flags['pf2e-third-party-automation'].metapowerUse.itemUuid,item.uuid);observer.record([{...data,id:'m',uuid:'ChatMessage.m'}]);});
 assert.equal(completed.status,'committed');
});
test('actual entrance captures targets before asynchronous choices and persists them on original card',async()=>{
 const actor={uuid:'Actor.a',id:'a'},item={uuid:'Actor.a.Item.i',id:'i',actor},targets=['Scene.s.Token.first'];let captured;
 const observer=api.createMetapowerObserver({request:async(_m,p)=>({nonce:p.nonce,status:'reserved'}),captureInput:()=>({targetUuids:targets}),select:async()=>{targets[0]='Scene.s.Token.later';return {}},id:()=> 'n'});
 await observer.observe({actor,item},async()=>{captured=observer.decorate({speaker:{actor:'a'},flags:{pf2e:{origin:{uuid:item.uuid}},'pf2e-third-party-automation':{usageInput:{targetUuids:['Scene.s.Token.later']}}}});observer.record([{...captured,id:'m',uuid:'ChatMessage.m'}]);});
 assert.deepEqual(captured.flags['pf2e-third-party-automation'].usageInput.targetUuids,['Scene.s.Token.first']);
});
