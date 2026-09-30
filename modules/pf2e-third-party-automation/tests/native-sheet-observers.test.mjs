import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {registerNativeSheetHandlers} from '../scripts/native-sheet-handlers.mjs';
import {registerUsageEvents} from '../scripts/usage-events.mjs';
import {wrapSheetHandlers} from '../scripts/metapower/entrances.mjs';
import {createMetapowerObserver} from '../scripts/metapower/observer.mjs';
import {createMetapowerLedger} from '../scripts/metapower/lifecycle.mjs';

const ID='pf2e-third-party-automation';
test('main shared native sheet observers make one real ledger lease and await one original Use',async t=>{
 const previousConfig=globalThis.CONFIG;t.after(()=>{globalThis.CONFIG=previousConfig});
 const gm={id:'gm',targets:new Set()},owner={id:'owner',targets:new Set()},requests=[],before=[];let n=0,nativeCalls=0,registrations=0;
 const actor={id:'a',uuid:'Actor.a',type:'character',items:new Map(),flags:{},testUserPermission:()=>true,async update(data){this.flags[ID]={metapower:structuredClone(data[`flags.${ID}.metapower`])}}};
 const item={id:'f',uuid:'Actor.a.Item.f',type:'feat',actor,sourceId:'custom-supported',system:{}};actor.items.set(item.id,item);
 const users=new Map([[gm.id,gm],[owner.id,owner]]);users.activeGM=gm;
 const game={user:gm,users,actors:new Map([[actor.id,actor]]),scenes:new Map(),messages:new Map(),combats:new Map()},documents=new Map([[actor.uuid,actor],[item.uuid,item]]),fromUuid=async uuid=>documents.get(uuid);
 const ledger=createMetapowerLedger({game,fromUuid}),observer=createMetapowerObserver({id:()=>`n${++n}`,request:async(method,payload)=>{requests.push(method);return ledger[method](payload,owner)},onError:error=>{throw error}}),observe=(context,native)=>observer.observe(context,native);
 // Execute the actual main callback, so losing the native-sheet entry condition
 // reproduces two begin calls and the durable pending-lease rejection.
 const source=readFileSync(new URL('../scripts/main.mjs',import.meta.url),'utf8'),start=source.indexOf('observeItemUse:')+'observeItemUse:'.length,end=source.indexOf(',captureUsage:',start);
 assert.ok(start>0&&end>start);
 const observeItemUse=Function('prayer','familiar','halflingLuck','scar','metapower',`return (${source.slice(start,end)})`)(
  {resolveAction:()=>true,beforeUse:()=>before.push('prayer')},
  {beforeUse:()=>before.push('familiar')},{beforeUse:async()=>before.push('halfling')},{beforeUse:()=>before.push('scar')},{observe});
 class Sheet{activateClickListener(){}}globalThis.CONFIG={Actor:{sheetClasses:{character:{'pf2e.Native':{cls:Sheet}}}}};
 const paths=new Map(),path='CONFIG.Actor.sheetClasses.character["pf2e.Native"].cls.prototype.activateClickListener';
 const libWrapper={register(_module,key,fn){assert.equal(paths.has(key),false);paths.set(key,fn);if(key===path)registrations++},unregister(_module,key){paths.delete(key)}};
 const release=registerNativeSheetHandlers(libWrapper,path,(sheet,handlers)=>wrapSheetHandlers(sheet,handlers,observe,()=>true));
 const unregister=registerUsageEvents({game,Hooks:{on:()=>1,off(){}},libWrapper,fromUuid,resolveAction:()=> 'managed',tracksFrequency:()=>false,executeUsage:async()=>{},observeItemUse});
 t.after(()=>{unregister();release()});
 const native=async()=>{
  nativeCalls++;await Promise.resolve();
  const data=observer.decorate({speaker:{actor:actor.id},author:owner,flags:{pf2e:{origin:{uuid:item.uuid}}}}),message={...data,id:'m',uuid:'ChatMessage.m'};
  documents.set(message.uuid,message);observer.record([message]);return message;
 };
 const app=new Sheet();app.actor=actor;const handlers=paths.get(path).call(app,()=>({'use-action':native}));
 const button={closest:selector=>selector==='[data-item-id]'?{dataset:{itemId:item.id}}:null},result=await handlers['use-action']({},button);
 assert.equal(result.uuid,'ChatMessage.m');assert.equal(registrations,1);assert.equal(nativeCalls,1);assert.deepEqual(requests,['begin','finish']);assert.deepEqual(before,['prayer','familiar','halfling','scar']);
 assert.deepEqual(Object.values(actor.flags[ID].metapower.receipts).map(receipt=>receipt.status),['committed']);assert.equal(actor.flags[ID].metapower.pending,null);
 // A hotbar entry still owns its sole observation. It does not pass the sheet
 // entry marker and therefore must not lose normal metapower payment/card proof.
 const receipt=Object.values(actor.flags[ID].metapower.receipts)[0];await ledger.delivery({actorUuid:actor.uuid,nonce:receipt.nonce,status:'done'},gm);
 await observeItemUse(item,native);assert.equal(nativeCalls,2);assert.deepEqual(requests,['begin','finish','begin','finish']);
 assert.deepEqual(Object.values(actor.flags[ID].metapower.receipts).map(entry=>entry.status),['committed','committed']);
});
