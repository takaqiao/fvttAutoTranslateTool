import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';
import {verifyManualPoolProviders} from '../scripts/exploration/manual-pool-provider.mjs';
import {manualPoolBatchModel,MANUAL_POOL_BATCH_SOURCE_SHA} from '../scripts/exploration/manual-pool-model.mjs';
import {createManualPoolSources} from '../scripts/exploration/manual-pool-source.mjs';
import {MODULE_ID} from '../scripts/exploration/schema.mjs';

const hash=value=>createHash('sha256').update(value).digest('hex');
function input(name,sha){
 assert.ok(process.env[name],name+' is required');const bytes=fs.readFileSync(process.env[name]);
 assert.equal(hash(bytes),sha);return bytes;
}
const native=input('PF2E_NATIVE_BUNDLE',MANUAL_POOL_BATCH_SOURCE_SHA);
const toolbelt=input('TOOLBELT_MANUAL_SOURCE','2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f');

async function fixture({binding=false}={}){
 const user={id:'G',active:true,isGM:true},users=new Map([['G',user]]);users.activeGM=user;
 const actors=new Map(['H','P','M'].map(id=>[id,{id,uuid:'Actor.'+id,testUserPermission:owner=>owner===user}]));
 const healer=actors.get('H'),patient=actors.get('P'),messages=new Map(),hooks=new Map(),initializers=[];
 const enrolled=[],recorded=[],resolved=[],updated=[],errors=[];
 const variant={use(){}},action={slug:'treat-wounds',use(){},toActionVariant:()=>variant};
 const game={user,users,actors,messages,time:{worldTime:100},system:{id:'pf2e',version:'8.5.1'},
  pf2e:{actions:new Map([['treat-wounds',action]])},modules:new Map([
   ['patreon-v3',{active:true,version:'3.2.29'}],['pf2e-toolbelt',{active:true,version:'3.56.5'}]])};
 const source=Buffer.concat([native,Buffer.from('\n// unrelated native provenance change\n')]);
 const bridged=buildNativeBridge({source,version:game.system.version});
 const pair=buildSharedManualPair({pf2eSource:bridged.buffer,toolbeltSource:toolbelt,
  pf2eVersion:game.system.version,toolbeltVersion:game.modules.get('pf2e-toolbelt').version});
 const context=vm.createContext({game,Hooks:{once:(name,fn)=>initializers.push(fn)},
  FlatModifierRuleElement:class {beforePrepareData(){}},Y:class {resolveValue(){}resolveInjectedProperties(){}},
  Hn:class {test(){}},AutomaticBonusProgression$1:{}});
 const text=pair.pf2e.toString('utf8');
 vm.runInContext(text.slice(text.indexOf('const __nativeReceiverStacking='),text.indexOf('async function applyDamageFromMessage('))+'\n__nativeManualPoolBatch.install();',context);
 vm.runInContext(pair.toolbelt.toString('utf8').split('/* end toolbelt manual pool */')[0],context);
 // Install the unchanged legacy observer, rather than inventing a descriptor.
 vm.runInContext(fs.readFileSync(new URL('../tools/patreon-manual-immunity/observer.js',import.meta.url),'utf8'),context);
 for(const initialize of initializers)initialize();
 const proof=await verifyManualPoolProviders({game,pf2eSource:pair.pf2e,toolbeltSource:pair.toolbelt,hash});
 assert.equal(proof.ready,true,proof.reason);
 const batch=game.pf2e.thirdPartyManualPoolBatch;
 assert.equal(batch.descriptor.version,2);assert.equal(manualPoolBatchModel(batch.descriptor),true);
 assert.notEqual(batch.descriptor.baseSourceSHA256,MANUAL_POOL_BATCH_SOURCE_SHA);
 const paid=game.modules.get('patreon-v3').api.explorationManualImmunity.descriptor;
 assert.equal(paid.version,1);assert.equal(paid.pf2eSourceSHA256,MANUAL_POOL_BATCH_SOURCE_SHA);
 const session={id:'S',manual:true,status:'recording',startedAt:100,actorUUIDs:['Actor.H','Actor.P','Actor.M']};
 function message(id,isCheckRoll,flags){return {id,isCheckRoll,isReroll:false,author:user,speaker:{actor:'H'},
  rolls:[{_evaluated:true,total:isCheckRoll?24:9}],flags,toObject(){return {id,isCheckRoll,author:user.id,
   speaker:this.speaker,rolls:this.rolls,flags:this.flags}},async update(changes){
   updated.push({id,changes:structuredClone(changes)});
   for(const [key,value] of Object.entries(changes)){const parts=key.split('.');let at=this;
    for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=structuredClone(value)}return this;
  }};}
 const check=message('C',true,{pf2e:{context:{type:'skill-check',origin:{actor:healer.uuid},target:{actor:patient.uuid},
  options:['action:treat-wounds','exploration-manual:U']}},[MODULE_ID]:{explorationManualNative:{useId:'U',
   tag:'exploration-manual:U',patientUUID:patient.uuid,riskySurgery:false,startedAt:100}}});
 const result=message('D',false,{pf2e:{origin:{messageId:'C'}}});messages.set('C',check);messages.set('D',result);
 const saveBinding=()=>{check.flags[MODULE_ID].explorationManualNative.patreonImmunity={invocationId:'I',messageId:'C',
  useId:'U',tag:'exploration-manual:U',actorUUID:healer.uuid,patientUUID:patient.uuid,sourceUserId:user.id,startedAt:100,
  recordingSessionId:'S',targetSnapshot:{type:'check-context',actorUUID:patient.uuid,tokenUUID:'Scene.S.Token.T'}};};
 if(binding)saveBinding();
 const sources=createManualPoolSources({game,Hooks:{on:(name,fn)=>{hooks.set(name,fn);return name},off:name=>hooks.delete(name)},
  ledger:{async recordManualPoolSource(value){recorded.push(structuredClone(value))}},
  fromUuid:async uuid=>{resolved.push(uuid);return [...actors.values()].find(actor=>actor.uuid===uuid)??null},
  hpPools:{discover:()=>({ready:true,poolUUID:'Actor.M',memberUUIDs:['Actor.M','Actor.P']})},clientNonce:'client',
  getSession:async()=>session,isIssuer:()=>true,onEnroll:async value=>enrolled.push(structuredClone(value)),
  getBatchProvider:()=>batch,onError:error=>errors.push(error)});
 sources.start();const handle=await sources.beginNative({slug:'treat-wounds',actors:[healer],user,action,variant,
  params:{target:patient}},{useId:'U',tag:'exploration-manual:U'});assert.ok(handle);resolved.length=0;
 const publish=async()=>{await sources.nativeCheck(handle,{actor:healer,message:check});await sources.nativeResult(result)};
 return {sources,enrolled,recorded,resolved,updated,errors,publish,saveBinding};
}

test('native-v2 provenance cannot bypass the exact legacy patient-binding wait',async()=>{
 const f=await fixture();try{await f.publish();
  assert.deepEqual(f.resolved,[]);assert.deepEqual(f.updated,[]);
  assert.deepEqual(f.enrolled,[]);assert.deepEqual(f.recorded,[]);assert.deepEqual(f.errors,[]);
 }finally{f.sources.stop()}
});

test('the saved original legacy binding publishes once with a qualified native-v2 provider',async()=>{
 const f=await fixture({binding:true});try{await f.publish();await f.publish();
  assert.equal(f.enrolled.length,1);assert.equal(f.recorded.length,1);assert.equal(f.updated.length,1);
  assert.equal(f.recorded[0].checkId,'C');assert.equal(f.recorded[0].resultId,'D');assert.deepEqual(f.errors,[]);
 }finally{f.sources.stop()}
});

test('a later original binding releases the pending source without another native action',async()=>{
 const f=await fixture();try{await f.publish();assert.equal(f.recorded.length,0);
  f.saveBinding();await f.publish();await f.publish();assert.equal(f.recorded.length,1);
  assert.equal(f.updated.length,1);assert.deepEqual(f.errors,[]);
 }finally{f.sources.stop()}
});
