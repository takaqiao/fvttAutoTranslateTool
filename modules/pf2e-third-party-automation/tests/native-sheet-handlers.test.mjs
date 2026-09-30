import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/native-sheet-handlers.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
test('native sheet observers share one libWrapper registration and retain the awaited native action',async()=>{
 assert.equal(typeof api.registerNativeSheetHandlers,'function');let wrapper,registrations=0,removed=0;const calls=[];
 const libWrapper={register(_id,_path,fn){assert.equal(++registrations,1);wrapper=fn;},unregister(){removed++;}};
 const factory=name=>(sheet,handlers)=>{const native=handlers['use-action'];handlers['use-action']=async function(...args){calls.push(name+':start');try{return await native.apply(this,args);}finally{calls.push(name+':end');}};return handlers;};
 const first=api.registerNativeSheetHandlers(libWrapper,'native.sheet',factory('metapower'));
 const second=api.registerNativeSheetHandlers(libWrapper,'native.sheet',factory('usage'));
 const native=()=>({'use-action':async()=>{calls.push('native');return 42;}});
 const handlers=wrapper.call({},native);assert.equal(await handlers['use-action'](),42);
 assert.deepEqual(calls,['usage:start','metapower:start','native','metapower:end','usage:end']);
 second();assert.equal(removed,0);first();assert.equal(removed,1);
});
