import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createManualEvents} from '../../scripts/exploration/manual-events.mjs';
import {manualEvidenceFixture,flush,M} from './manual-evidence-fixture.mjs';

function nativeFixture({user='PUSER',author='G',canUpdate=false}={}){
 const f=manualEvidenceFixture(),results=[],updates=[];
 f.game.user=f.users.get(user);
 const native={tag:'exploration-manual:N',useId:'N',patientUUID:'Actor.P',continualRecovery:false};
 const check={id:'NC',isCheckRoll:true,author:f.users.get(author),speaker:{actor:'H'},rolls:[{_evaluated:true}],flags:{[M]:{explorationManualNative:native},pf2e:{context:{type:'skill-check',origin:{actor:'Actor.H'},options:[native.tag],outcome:'success'}}}};
 const result={id:'ND',isCheckRoll:false,author:check.author,speaker:{actor:'H'},rolls:[{_evaluated:true}],flags:{[M]:{explorationManualNative:native},pf2e:{origin:{messageId:check.id},context:{origin:{actor:'Actor.H'},options:[native.tag]}}},
  canUserModify(current,action){assert.equal(current,f.game.user);assert.equal(action,'update');return canUpdate},
  async update(changes){updates.push(changes);if(!canUpdate)throw Error('ChatMessage update requires ownership');this.flags.pf2e.context.options=changes['flags.pf2e.context.options'];return this}};
 const recorder=createManualEvents({...f.options,isAuthority:()=>f.game.user.isGM,manualPoolSources:{async nativeResult(message){results.push(message.id)}}});
 recorder.start();
 return {...f,recorder,check,result,results,updates};
}

test('a player observing another author native result does not submit a chat update',async t=>{
 const f=nativeFixture();t.after(()=>f.recorder.stop());
 await f.fire(f.check);await f.fire(f.result);
 assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
 assert.deepEqual(f.result.flags.pf2e.context.options,['exploration-manual:N']);
 assert.deepEqual(f.results,['ND']);assert.deepEqual((await f.ledger.snapshot('S')).activities,[]);
});

for(const [user,author] of [['G','HUSER'],['HUSER','HUSER']])test(`${user==='G'?'GM':'the player author'} can mark a native result`,async t=>{
 const f=nativeFixture({user,author,canUpdate:true});t.after(()=>f.recorder.stop());
 await f.fire(f.check);await f.fire(f.result);
 assert.deepEqual(f.errors,[]);assert.equal(f.updates.length,1);
 assert.deepEqual(f.result.flags.pf2e.context.options,['exploration-manual:N',`${M}:source:ND:0`]);
 assert.deepEqual(f.results,['ND']);
 if(user==='G')assert.deepEqual((await f.ledger.getActivity('manual:NC')).proof.resultIds,['ND']);
});

test('a refused annotation still lets the GM record the native result evidence',async t=>{
 const f=nativeFixture({user:'G'});t.after(()=>f.recorder.stop());
 await f.fire(f.check);await f.fire(f.result);
 assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
 assert.deepEqual((await f.ledger.getActivity('manual:NC')).proof.resultIds,['ND']);
});

test('Workbench observation continues when the current player cannot annotate its result',async()=>{
 const f=manualEvidenceFixture(),results=[],updates=[];f.game.user=f.users.get('PUSER');
 f.result.canUserModify=(user,action)=>{assert.equal(user,f.game.user);assert.equal(action,'update');return false};
 f.result.update=async changes=>{updates.push(changes);throw Error('ChatMessage update requires ownership')};
 const recorder=createManualEvents({...f.options,isAuthority:()=>false,manualPoolSources:{async workbenchResult(source,message){results.push({source,messageId:message.id});return 'observed'}}});
 assert.equal(await recorder.observeWorkbenchResult('source',f.result),'observed');await flush();
 assert.deepEqual(updates,[]);assert.deepEqual(f.errors,[]);
 assert.deepEqual(results,[{source:'source',messageId:'W'}]);
});

test('the player author can annotate its Workbench result',async()=>{
 const f=manualEvidenceFixture(),results=[];f.game.user=f.users.get('HUSER');
 f.result.flags.pf2e={context:{options:[]}};
 f.result.canUserModify=(user,action)=>{assert.equal(user,f.game.user);assert.equal(action,'update');return true};
 f.result.update=async changes=>{f.result.flags.pf2e.context={options:changes['flags.pf2e.context.options']};return f.result};
 const recorder=createManualEvents({...f.options,isAuthority:()=>false,manualPoolSources:{async workbenchResult(_source,message){results.push(message.id);return 'observed'}}});
 assert.equal(await recorder.observeWorkbenchResult('source',f.result),'observed');
 assert.deepEqual(f.result.flags.pf2e.context.options,[`${M}:source:W:0`]);
 assert.deepEqual(results,['W']);assert.deepEqual(f.errors,[]);
});
