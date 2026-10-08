import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {fixture} from './manual-pool-provider-fixture.mjs';
import {createManualPoolSources} from '../../scripts/exploration/manual-pool-source.mjs';
import {createManualPoolProof} from '../../scripts/exploration/manual-pool-proof.mjs';
import {createManualEvents,registerWorkbenchObservation,WORKBENCH_SOURCE_SHA,IMMUNITY_SOURCES} from '../../scripts/exploration/manual-events.mjs';
import {canonicalJSON} from '../../scripts/exploration/revision-codec.mjs';
import {MODULE_ID as M} from '../../scripts/exploration/schema.mjs';

const sha='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';
const flush=async()=>{for(let i=0;i<30;i++)await new Promise(resolve=>setImmediate(resolve))};
async function settled(f,predicate){for(let i=0;i<250;i++){const activity=await f.activity();if(predicate(activity))return activity;await new Promise(resolve=>setTimeout(resolve,1))}assert.fail(JSON.stringify({activity:await f.activity(),errors:f.errors.map(error=>error.message)}))}
async function sealFixture(t,{remote=false,sealed=true,immunity=false}={}){
 const f=await fixture(remote,{activityId:'manual:R',sourceType:'workbench'});t.after(()=>f.close());const {game,patient}=f.gm,check=f.messages.get('C'),result=f.messages.get('R');
 check.flags={pf2e:{context:{options:['exploration-manual-use:U']}}};result.flags={[M]:{explorationManual:{lexicalSource:true,sourceSHA:WORKBENCH_SOURCE_SHA,useId:'U',actorUUID:'Actor.H',patientUUID:'Actor.P',kind:'treatment',riskySurgery:false,checkIds:['C'],stageIds:[]}},treat_wounds_battle_medicine:{id:'T',healerId:'H',dos:2,healing:9},pf2e:{context:{options:[`${M}:source:R:0`]}}};
 result.rolls[0].formula='2d8[healing]';
 for(const doc of [check,result,f.receipt])doc.toObject=function(){return structuredClone({id:this.id,author:this.author,speaker:this.speaker,rolls:this.rolls??[],flags:this.flags??{}})};
 patient.items=new Map();const hooks=new Map(),Hooks={on:(event,fn)=>{hooks.set(event,fn);return fn},off:(event,fn)=>{if(hooks.get(event)===fn)hooks.delete(event)}};
 const provider={descriptor:{version:1,providerId:'pf2e',providerVersion:'8.5.1',protocol:`${M}:manual-pool-batch:1`,model:'numeric-empty-reception.v1',baseSourceSHA256:sha},subscribe(){return()=>{}}};game.system={id:'pf2e',version:'8.5.1'};game.pf2e={actions:new Map(),thirdPartyManualPoolBatch:provider};
 const sources=createManualPoolSources({game,Hooks,ledger:f.ledger,fromUuid:async uuid=>f.gm.actors.get(uuid),hpPools:f.gm.pools,clientNonce:'cold-read',getSession:()=>f.ledger.getSession('S'),isIssuer:()=>false});sources.start();t.after(()=>sources.stop());
 const command=fs.readFileSync(process.env.WORKBENCH_MANUAL_SOURCE??'C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/treat-wounds-actual-command.txt','utf8'),macro={name:'XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine',command,execute(){throw Error('lookup must not execute a macro')},clone(){throw Error('lookup must not clone a macro')}};
 game.packs=new Map([['xdy-pf2e-workbench.asymonous-benefactor-macros-internal',{getDocuments:async()=>[macro]}]]);const cleanup=await registerWorkbenchObservation({game,recorder:{observeWorkbenchProvider:sources.observeWorkbenchProvider}});t.after(cleanup);
 const source={version:2,sourceVersion:'8.5.1',sessionId:'S',activityId:'manual:R',actorUUID:'Actor.H',patientUUID:'Actor.P',sourceType:'workbench',useId:'U',checkId:'C',resultId:'R',rollIndex:0,worldTime:0,sourceUserId:'O',sourceClientNonce:'O1',sourceNonce:'actual-source-observation',provider:{id:'xdy-pf2e-workbench',sourceSHA:WORKBENCH_SOURCE_SHA},documentsDigest:createHash('sha256').update(canonicalJSON([check.toObject(true),result.toObject(true)])).digest('hex')};
 // Source registration has its own original-callback tests. This private GM
 // fixture seeds that sealed boundary, then obtains the terminal from the real
 // broker and the fixed original Toolbelt local/socket Promise branches.
 await f.ledger.recordManualPoolSource(source,{evidenceGuard:()=>game.messages.get('C')===check&&game.messages.get('R')===result});
 if(sealed){const grant=await f.owner.broker.claim(f.request),pending=f.owner.broker.withApplication(grant,f.owner.patient,f.operation);pending.catch(()=>{});await Promise.race([f.writeStarted,pending]);f.finish();await pending}
 const old=await f.ledger.getActivity('manual:R');await f.ledger.transitionActivity(old.id,{expected:['awaiting-evidence'],patch:{proof:{...old.proof,receiptIds:sealed?[f.receipt.id]:[]}}});
 if(immunity)patient.items.set('I',{id:'I',uuid:'Actor.P.Item.I',type:'effect',actor:patient,parent:patient,sourceId:'Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5',flags:{[M]:{explorationManualImmunity:{messageId:'R',useId:'U',patientUUID:patient.uuid,kind:'treatment',sourceSHA:IMMUNITY_SOURCES['XDY DO_NOT_IMPORT TW Immunity CD'].sha,creatorId:'O'}}}});
 const proof=createManualPoolProof({game,fromUuid:async uuid=>f.gm.actors.get(uuid),hpPools:f.gm.pools,resolveSource:sources.resolveSource});
 const recorder=createManualEvents({game,Hooks,ledger:f.ledger,hpPools:f.gm.pools,manualPoolProof:proof,fromUuid:async uuid=>f.gm.actors.get(uuid),sessionId:()=> 'S',isAuthority:()=>true,onChange:change=>{if(change.error)f.errors.push(Error(change.error))}});t.after(()=>recorder.stop());
 return {...f,game,patient,check,result,source,provider,macro,proof,recorder,hooks,activity:()=>f.ledger.getActivity('manual:R')};
}

for(const remote of [false,true])test(`saved ${remote?'GM socket':'local'} master plus original receipt seal removes only the shared gap on hydrate`,async t=>{
 const f=await sealFixture(t,{remote});assert.ok(await f.proof.evidence(await f.activity()));const writes=f.writes.length,packets=f.packets.length;f.recorder.start();const a=await settled(f,activity=>!activity.options.missing.includes('shared-hp-completion-unavailable'));
 assert.equal(a.state,'awaiting-evidence');assert.ok(!a.options.missing.includes('shared-hp-completion-unavailable'));assert.ok(a.options.missing.includes('native-immunity-receipt'));assert.equal(f.writes.length,writes);assert.equal(f.packets.length,packets);
});
test('an existing independent Workbench immunity and both HP halves confirm on a read-only hydrate',async t=>{
 const f=await sealFixture(t,{immunity:true});f.recorder.start();await settled(f,activity=>activity.state==='confirmed');assert.equal((await f.activity()).state,'confirmed');assert.equal(f.writes.length,1);
});
test('document markers without a settled application never create a shared proof',async t=>{
 const f=await sealFixture(t,{sealed:false});f.result.flags[M].explorationManualPoolParticipation={sessionId:'S',useId:'U',poolUUID:'Actor.M'};f.messages.set(f.receipt.id,f.receipt);assert.equal(await f.proof.evidence(await f.activity()),null);assert.equal(f.writes.length,0);
});
test('a legacy pending source cannot reuse its old application seal or write during proof lookup',async t=>{
 const f=await sealFixture(t),read=f.ledger.getActivity;
 f.ledger.getActivity=async id=>{const activity=await read(id);activity.proof.manualPoolSource.version=1;delete activity.proof.manualPoolSource.sourceVersion;return activity};
 const writes=f.writes.length,packets=f.packets.length,activity=await f.activity();
 assert.equal(activity.state,'awaiting-evidence');assert.equal(await f.proof.evidence(activity),null);
 assert.equal(f.writes.length,writes);assert.equal(f.packets.length,packets);
});
for(const [name,change] of [['deleted source',f=>f.messages.delete('R')],['changed roll',f=>{f.result.rolls[0].total=30}],['edited receipt',f=>{f.receipt.flags.pf2e.appliedDamage.isReverted=true}],['lost patient OWNER',f=>{f.patient.testUserPermission=()=>false}],['lost master writer',f=>{f.gm.master.testUserPermission=()=>false}],['pool rebind',f=>{f.game.toolbelt.api.shareData.getMasterInMemory=()=>f.patient}],['time drift',f=>{f.game.time.worldTime=1}],['provider changed',f=>{f.macro.command='changed'}],['provider pack replaced',f=>{f.game.packs.clear()}],['patient collection replaced',f=>{f.game.actors.set('P',{...f.patient})}],['master collection replaced',f=>{f.game.actors.set('M',{...f.gm.master})}],['a second HP receipt',f=>{f.messages.set('receipt2',{...f.receipt,id:'receipt2'})}]])test(`a settled seal is unavailable after ${name}`,async t=>{
 const f=await sealFixture(t);assert.ok(await f.proof.evidence(await f.activity()));change(f);assert.equal(await f.proof.evidence(await f.activity()),null);
});
function holdConfirmation(t){let entered,release;const seen=new Promise(resolve=>{entered=resolve}),resume=new Promise(resolve=>{release=resolve}),original=crypto.subtle.digest;let armed=true;
 crypto.subtle.digest=async function(...args){const result=await original.apply(this,args);let body;try{body=JSON.parse(new TextDecoder().decode(args[1]))}catch{}if(armed&&body?.changes?.activities?.['manual:R']?.state==='confirmed'){armed=false;entered();await resume}return result};t.after(()=>{release();crypto.subtle.digest=original});return {seen,release};
}
for(const [name,change] of [['Stop',f=>f.client('stopper').updateSession('S',{status:'stopped'})],['receipt revert',f=>{f.receipt.flags.pf2e.appliedDamage.isReverted=true}],['pool rebind',f=>{f.game.toolbelt.api.shareData.getMasterInMemory=()=>f.patient}],['source roll change',f=>{f.result.rolls[0].total=80}],['provider change',f=>{f.macro.command='changed'}]])test(`shared confirmation cannot cross the revision submission boundary after ${name}`,async t=>{
 const f=await sealFixture(t,{immunity:true}),hold=holdConfirmation(t);f.recorder.start();await hold.seen;await change(f);hold.release();await settled(f,()=>f.errors.length>0);assert.equal((await f.activity()).state,'awaiting-evidence');assert.equal(f.writes.length,1);
});
