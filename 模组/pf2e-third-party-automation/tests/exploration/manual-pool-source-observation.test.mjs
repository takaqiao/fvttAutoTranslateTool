import {test} from 'node:test';
import assert from 'node:assert/strict';
import {getNativeActionEvents} from '../../scripts/native-action-events.mjs';
import {createManualEvents} from '../../scripts/exploration/manual-events.mjs';
import {manualEvidenceFixture,flush,M} from './manual-evidence-fixture.mjs';

test('only the branded original native Use registers its exact check and result source',async t=>{
 const f=manualEvidenceFixture(),observed=[],begun=[];f.game.user=f.users.get('HUSER');
 f.healer.items.some=()=>false;
 class Variant{async use(params){
  const check={id:'NC',isCheckRoll:true,author:f.game.user,speaker:{actor:'H'},rolls:[{_evaluated:true,total:24}],flags:{pf2e:{context:{type:'skill-check',origin:{actor:'Actor.H'},target:{actor:'Actor.P'},options:params.rollOptions,outcome:'success'}}}};
  const data={flags:check.flags};f.handlers.get('preCreateChatMessage')(check,data);f.messages.set(check.id,check);f.handlers.get('createChatMessage')(check);
  const result={id:'ND',isCheckRoll:false,author:f.game.user,speaker:{actor:'H'},rolls:[{_evaluated:true,total:9}],flags:{...structuredClone(check.flags),pf2e:{...structuredClone(check.flags.pf2e),origin:{messageId:check.id}}},async update(changes){this.flags.pf2e.context.options=changes['flags.pf2e.context.options'];return this}};
  f.messages.set(result.id,result);f.handlers.get('createChatMessage')(result);
  return [{actor:f.healer,message:check,outcome:'success'}];
 }}
 const action={slug:'treat-wounds',variants:new Map(),toActionVariant:()=>new Variant(),use(params){return this.toActionVariant().use(params)}};
 f.game.pf2e={actions:new Map([['treat-wounds',action]])};const nativeActions=getNativeActionEvents({game:f.game});
 const sources={async beginNative(scope,meta){begun.push({scope,meta});return {id:'private-use'}},async nativeCheck(use,row){observed.push({type:'check',use,row})},async nativeResult(message){observed.push({type:'result',message})}};
 const recorder=createManualEvents({...f.options,nativeActions,isAuthority:()=>false,manualPoolSources:sources});recorder.start();nativeActions.register();t.after(()=>{recorder.stop();nativeActions.cleanup()});
 await action.use({actors:[f.healer],target:f.patient});await flush();
 assert.equal(begun.length,1);assert.equal(observed.filter(e=>e.type==='check').length,1);assert.equal(observed.find(e=>e.type==='check').row.message,f.messages.get('NC'));
 assert.equal(observed.find(e=>e.type==='result').message,f.messages.get('ND'));
 const forged=structuredClone(f.messages.get('NC').flags);f.handlers.get('createChatMessage')({id:'FORGED',flags:forged});await flush();
 assert.equal(begun.length,1);assert.equal(observed.filter(e=>e.type==='check').length,1);
});

test('ordinary shared enrollment records its actual immutable pool before any application claim',async()=>{
 const f=manualEvidenceFixture(),recorder=createManualEvents({...f.options,hpPools:{discover:()=>({ready:true,poolUUID:'Actor.M',memberUUIDs:['Actor.P','Actor.M'],provider:'pf2e-toolbelt'})}});
 recorder.start();await recorder.observe(f.event);const activity=await f.ledger.getActivity('manual:W');recorder.stop();
 assert.deepEqual(activity.hpPoolUUIDs,['Actor.M']);assert.equal(activity.state,'awaiting-evidence');
 assert.ok(activity.options.missing.includes('shared-hp-completion-unavailable'));
});
