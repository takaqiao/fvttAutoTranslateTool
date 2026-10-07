import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {authorityFixture} from './authority-fixture.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createManualEvents,registerWorkbenchObservation} from '../../scripts/exploration/manual-events.mjs';
import {createManualPoolSources} from '../../scripts/exploration/manual-pool-source.mjs';
import {getNativeActionEvents} from '../../scripts/native-action-events.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';

const flush=async()=>{for(let i=0;i<30;i++)await new Promise(resolve=>setImmediate(resolve))};
const sha='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve}};
export async function sourceFixture(options={}){
 const storage=await authorityFixture(),ledger=storage.client('issuer');await ledger.createSession({id:'S',manual:true,status:'recording',startedAt:100,budgetEndsAt:700,actorUUIDs:['Actor.H','Actor.P','Actor.M']});
 const users=new Map(['G','O','X'].map(id=>[id,{id,active:true,isGM:id==='G'}]));users.activeGM=users.get('G');
 const actors=new Map(['H','P','M'].map(id=>[id,{id,uuid:`Actor.${id}`,items:[],owners:new Set(['G','O']),testUserPermission(user){return this.owners.has(user.id)}}]));
 const healer=actors.get('H'),patient=actors.get('P'),token={id:'T',uuid:'Scene.S.Token.T',actor:patient},messages=new Map(),clients=[],errors=[],packets=[];token.parent={tokens:new Map([['T',token]])};let privateReads=0,serial=0;
 function doc(id,data,user){const message={id,author:user,speaker:{actor:'H'},rolls:[],flags:{},...data,toObject(){return structuredClone({id:this.id,author:this.author.id,speaker:this.speaker,flags:this.flags,rolls:this.rolls.map(roll=>roll.toJSON?.()??roll)})},async update(changes){for(const [path,value]of Object.entries(changes)){const keys=path.split('.');let target=this;for(const key of keys.slice(0,-1))target=target[key]??={};target[keys.at(-1)]=structuredClone(value)}for(const client of clients)client.hooks.get('updateChatMessage')?.(this);return this}};return message}
 const roll=(total,formula='{2d8[healing]}')=>({_evaluated:true,total,formula,options:{degreeOfSuccess:2},toJSON(){return {_evaluated:true,total:this.total,formula:this.formula}}});
 function make(id,nonce,issuer){
  const clientLedger=id==='G'?(options.createLedger?.(storage,nonce)??storage.client(nonce)):ledger,recorderLedger=options.recorderLedger?.(clientLedger,nonce)??clientLedger,sourceLedger=options.sourceLedger?.(clientLedger,nonce)??clientLedger;
  const hooks=new Map(),listeners=new Set(),scenes=new Map();scenes.active={tokens:new Map([['T',token]])};
  const game={user:users.get(id),users,actors,messages,time:{worldTime:100},system:{version:'8.5.1'},scenes,pf2e:{actions:new Map()},socket:{on:(_,fn)=>listeners.add(fn),off:(_,fn)=>listeners.delete(fn),emit(_channel,packet,options){packets.push({nonce,packet:structuredClone(packet)});for(const client of clients.filter(client=>options.recipients.includes(client.game.user.id)))for(const receive of client.listeners)receive(structuredClone(packet),id)}}};
  const Hooks={on:(event,fn)=>{hooks.set(event,fn);return event},off:event=>hooks.delete(event)},hpPools={discover:()=>({ready:true,provider:'pf2e-toolbelt',poolUUID:'Actor.M',memberUUIDs:['Actor.M','Actor.P']})};let gate;
  const batchProvider={descriptor:{version:1,providerId:'pf2e',providerVersion:'8.5.1',protocol:'pf2e-third-party-automation:manual-pool-batch:1',model:'numeric-empty-reception.v1',baseSourceSHA256:sha},subscribe(_observer,{authorizeBatch}){gate=authorizeBatch;return()=>{}}};
  const nativeActions=getNativeActionEvents({game}),sources=createManualPoolSources({game,Hooks,ledger:id==='G'?sourceLedger:undefined,fromUuid:async uuid=>actors.get(uuid.split('.').at(-1))??(uuid===token.uuid?token:null),hpPools,clientNonce:nonce,isIssuer:()=>issuer,getSession:()=>id==='G'?clientLedger.getSession('S'):(privateReads++,Promise.reject(Error('player-ledger-read'))),getBatchProvider:()=>batchProvider,onEnroll:source=>options.onEnroll?options.onEnroll(source,value):value.recorder.observePoolSource(source),onError:error=>errors.push(error),timeoutMs:100});
  const observedSources=options.beforeNativeResult?{...sources,nativeResult:async message=>{await options.beforeNativeResult(nonce,message);return sources.nativeResult(message)}}:sources;
  const recorder=createManualEvents({game,Hooks,ledger:recorderLedger,nativeActions,hpPools,manualPoolSources:observedSources,fromUuid:async uuid=>actors.get(uuid.split('.').at(-1)),sessionId:()=>id==='G'?'S':null,onChange:change=>{if(change.error)errors.push(Error(change.error))}});
  const value={game,Hooks,hooks,listeners,sources,recorder,nativeActions,ledger:clientLedger,nonce,gate:event=>gate(event)};clients.push(value);sources.start({claim:async()=>{throw Error('claim-not-used-in-source-test')}});recorder.start();return value;
 }
 const gm=make('G','issuer',true),owner=make('O','owner',false),peer=make('G','peer',false);
 function create(data,user=owner.game.user){const id=`msg${++serial}`,message=doc(id,data,user);owner.hooks.get('preCreateChatMessage')?.(message,data);messages.set(id,message);for(const client of clients)client.hooks.get('createChatMessage')?.(message);return message}
 class Variant{async use(params){const check=create({isCheckRoll:true,rolls:[roll(24,'1d20')],flags:{pf2e:{context:{type:'skill-check',origin:{actor:healer.uuid},target:params.target?{actor:patient.uuid}:null,outcome:'success',options:[...(params.rollOptions??[]),'action:treat-wounds']}}}});if(options.afterCheck)await options.afterCheck(check);create({isCheckRoll:false,rolls:[roll(9)],flags:{...structuredClone(check.flags),pf2e:{...structuredClone(check.flags.pf2e),origin:{messageId:check.id}}}});return [{actor:healer,message:check,outcome:'success'}]}}
 const action={slug:'treat-wounds',variants:new Map(),toActionVariant:()=>new Variant(),use(params){return this.toActionVariant().use(params)}};owner.game.pf2e.actions.set('treat-wounds',action);gm.game.pf2e.actions.set('treat-wounds',action);owner.nativeActions.register();
 return {storage,ledger,gm,owner,peer,actors,healer,patient,token,messages,clients,errors,packets,roll,create,action,privateReads:()=>privateReads,close(){for(const client of clients){client.sources.stop();client.recorder.stop();client.nativeActions.cleanup()}}};
}
async function nativeSource(f){
 await f.action.use({actors:[f.healer],target:f.patient});
 const result=[...f.messages.values()].find(message=>message.isCheckRoll===false),deadline=performance.now()+2000;
 for(;;){
  const activity=await f.ledger.getActivity(`manual:${result.flags.pf2e.origin.messageId}`);
  if(activity?.proof.manualPoolSource&&f.owner.gate({phase:'admit',batch:{message:result}}).isCurrent())return result;
  assert.ok(performance.now()<deadline,'Native source registration did not finish: '+f.errors.map(error=>error.message).join('; '));
  await new Promise(resolve=>setTimeout(resolve,1));
 }
}
export async function workbenchSource(f,options={}){
 const command=fs.readFileSync(process.env.WORKBENCH_MANUAL_SOURCE??'C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/treat-wounds-actual-command.txt','utf8');assert.equal(createHash('sha256').update(command).digest('hex'),'b3bac907654522fda62b80da182fc253f7147493a465a82081219fb2a0f1308f');
 const getHealSuccess=Function(`${command.slice(command.indexOf('const getHealSuccess ='),command.indexOf('/**\n * Perform a roll'))}\nreturn getHealSuccess;`)();
 const skill={async roll(args){const check=f.create({isCheckRoll:true,rolls:[f.roll(24,'1d20')],flags:{pf2e:{context:{type:'skill-check',origin:{actor:'Actor.H'},options:args.extraRollOptions}}}});await args.callback(check.rolls[0],'success',check);return check}};
 const ChatMessage={create:options.afterResult?async data=>{const message=f.create(data);if(data.flags?.treat_wounds_battle_medicine)await options.afterResult(message);return message}:async data=>f.create(data),getSpeaker:()=>({actor:'H'})};
 class DamageRoll{constructor(formula){Object.assign(this,f.roll(9,formula));this._total=9}async roll(){return this}async toMessage(data){const message=f.create({...data,isCheckRoll:false,rolls:[this],flags:{...data.flags,pf2e:{context:{origin:{actor:'Actor.H'},options:[]}}}});if(options.afterResult&&data.flags?.treat_wounds_battle_medicine)await options.afterResult(message);return message}}
 const macro={name:'XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine',command,clone(changes){return {...this,...changes}},async execute(input){
  const observed=this.command.slice(this.command.indexOf('const explorationBases ='),this.command.indexOf('async function applyChanges'));
  const call=Function('game','token','ChatMessage','DamageRoll','CheckRoll','event','getHealSuccess','getRollOptions','dsnHook','explorationManualTarget',`${observed};return rollTreatWounds;`)(f.owner.game,{actor:f.healer},ChatMessage,DamageRoll,class{},null,getHealSuccess,()=>[],callback=>callback(),input.explorationManualTarget);
  return call({DC:15,bonus:0,skillUsed:skill,isRiskySurgery:false,isRightHandBlood:false,useMortalHealing:false,useMagicHands:false,assurance:false,bmtw:'Treat Wounds',target:f.token,immunityEffect:{name:'Treat Wounds',system:{duration:{value:60,unit:'minutes'}}},usedBattleMedicsBaton:false,spellStitcherBonus:0,immunityMacroLink:''});
 }};
 const gmMacro={...macro};f.gm.game.packs=new Map([['xdy-pf2e-workbench.asymonous-benefactor-macros-internal',{getDocuments:async()=>[gmMacro],getDocument:async()=>gmMacro}]]);await registerWorkbenchObservation({game:f.gm.game,recorder:f.gm.recorder});
 f.owner.game.packs=new Map([['xdy-pf2e-workbench.asymonous-benefactor-macros-internal',{getDocuments:async()=>[macro],getDocument:async()=>macro}]]);
 await registerWorkbenchObservation({game:f.owner.game,recorder:f.owner.recorder});await macro.execute({});await flush();const result=[...f.messages.values()].find(message=>message.flags?.treat_wounds_battle_medicine);
 for(let attempt=0;attempt<100;attempt++){if((await f.ledger.getActivity(`manual:${result.id}`))?.proof.manualPoolSource)break;await new Promise(resolve=>setTimeout(resolve,1))}return result;
}
test('the exact native Use callback registers an authenticated source in the creating GM tab',async t=>{
 const f=await sourceFixture();t.after(()=>f.close());const result=await nativeSource(f),activity=await f.ledger.getActivity(`manual:${result.flags.pf2e.origin.messageId}`);
 assert.equal(activity.proof.manualPoolSource.sourceUserId,'O');assert.equal(activity.proof.manualPoolSource.sourceClientNonce,'owner');assert.equal(activity.proof.manualPoolSource.resultId,result.id);
 assert.deepEqual(activity.hpPoolUUIDs,['Actor.M']);assert.equal(f.owner.gate({phase:'admit',batch:{message:result}}).isCurrent(),true);assert.equal(f.privateReads(),0);
 await flush();assert.deepEqual(f.errors,[]);assert.equal((await f.ledger.snapshot('S')).activities.length,1);
});
test('the fixed original Workbench check callback and delayed roll result register one distinct source',async t=>{
 const f=await sourceFixture();t.after(()=>f.close());const result=await workbenchSource(f),activity=await f.ledger.getActivity(`manual:${result.id}`);
 assert.ok(activity.proof.manualPoolSource,f.errors.map(error=>error.stack).join('\n'));assert.equal(activity.proof.manualPoolSource.sourceType,'workbench');assert.equal(activity.proof.manualPoolSource.resultId,result.id);assert.equal(activity.proof.manualPoolSource.checkId,activity.proof.checkIds[0]);assert.equal(f.privateReads(),0);
});
test('a saved default-target native result never falls back while original patient metadata is pending',async t=>{
 const f=await sourceFixture();t.after(()=>f.close());await f.action.use({actors:[f.healer]});const result=[...f.messages.values()].find(message=>message.isCheckRoll===false),check=f.messages.get(result.flags.pf2e.origin.messageId),pending=f.owner.gate({phase:'admit',batch:{message:result}});
 assert.equal(pending.status,'participating');assert.equal(pending.isCurrent(),false);const meta=check.flags[MODULE_ID].explorationManualNative;
 await check.update({[`flags.${MODULE_ID}.explorationManualNative`]:{...meta,patientUUID:f.patient.uuid,patreonImmunity:{messageId:check.id,useId:meta.useId,actorUUID:f.healer.uuid,patientUUID:f.patient.uuid,targetSnapshot:{type:'patreon-single-target',actorUUID:f.patient.uuid,tokenUUID:f.token.uuid}}}});
 for(let attempt=0;attempt<100;attempt++){if((await f.ledger.getActivity(`manual:${check.id}`))?.proof.manualPoolSource)break;await new Promise(resolve=>setTimeout(resolve,1))}
 assert.ok((await f.ledger.getActivity(`manual:${check.id}`)).proof.manualPoolSource);assert.equal(f.owner.gate({phase:'admit',batch:{message:result}}).isCurrent(),true);assert.equal(f.privateReads(),0);
});
test('an unknown recording-source ACK preserves the original action and leaves its card unavailable',async t=>{
 const f=await sourceFixture();t.after(()=>f.close());f.owner.game.socket.emit=()=>{};await f.action.use({actors:[f.healer],target:f.patient});const result=[...f.messages.values()].find(message=>message.isCheckRoll===false),answer=f.owner.gate({phase:'admit',batch:{message:result}});
 assert.equal(result.rolls[0].total,9);assert.equal(answer.status,'participating');assert.equal(answer.isCurrent(),false);await flush();assert.equal((await f.ledger.getActivity(`manual:${result.flags.pf2e.origin.messageId}`))?.proof.manualPoolSource,undefined);
});
test('a known inactive recording session retains the ordinary unregistered native path',async t=>{
 const f=await sourceFixture();t.after(()=>f.close());await f.ledger.updateSession('S',{status:'stopped'});await f.action.use({actors:[f.healer],target:f.patient});const result=[...f.messages.values()].find(message=>message.isCheckRoll===false);assert.equal(f.owner.gate({phase:'admit',batch:{message:result}}).status,'unregistered');assert.equal(f.privateReads(),0);
});
test('an explicit native target waits for the fixed original provider metadata before taking its source snapshot',async t=>{
 const f=await sourceFixture();t.after(()=>f.close());for(const c of [f.owner,f.gm])c.game.modules=new Map([['patreon-v3',{active:true,version:'3.2.29',api:{explorationManualImmunity:{descriptor:{version:1,providerVersion:'3.2.29',pf2eSourceSHA256:sha}}}}]]);
 await f.action.use({actors:[f.healer],target:f.patient});const result=[...f.messages.values()].find(message=>message.isCheckRoll===false),check=f.messages.get(result.flags.pf2e.origin.messageId);assert.equal(f.owner.gate({phase:'admit',batch:{message:result}}).isCurrent(),false);
 const meta=check.flags[MODULE_ID].explorationManualNative;await check.update({[`flags.${MODULE_ID}.explorationManualNative`]:{...meta,patreonImmunity:{messageId:check.id,useId:meta.useId,actorUUID:f.healer.uuid,patientUUID:f.patient.uuid,targetSnapshot:{type:'native-check-target',actorUUID:f.patient.uuid}}}});
 for(let attempt=0;attempt<100;attempt++){if((await f.ledger.getActivity(`manual:${check.id}`))?.proof.manualPoolSource)break;await new Promise(resolve=>setTimeout(resolve,1))}assert.ok((await f.ledger.getActivity(`manual:${check.id}`)).proof.manualPoolSource);assert.equal(f.owner.gate({phase:'admit',batch:{message:result}}).isCurrent(),true);
});
test('copied document flags have no executable source and peer GM tabs cannot issue a scope',async t=>{
 const f=await sourceFixture();t.after(()=>f.close());const result=await nativeSource(f),copy=f.create({isCheckRoll:false,rolls:[f.roll(9)],flags:structuredClone(result.flags)});await flush();
 const answer=f.owner.gate({phase:'admit',batch:{message:copy}});assert.equal(answer.status,'participating');assert.equal(answer.isCurrent(),false);
 const request={sessionId:'S',activityId:'forged',actorUUID:'Actor.H',sourceType:'native-action',useId:'forged',checkId:'C',resultId:copy.id,rollIndex:0,stage:'healing',poolUUID:'Actor.M',patientUUIDs:['Actor.P'],batchId:'forged',ownerClientNonce:'owner',attemptNonce:'forged'};
 assert.equal(await f.owner.sources.resolveSource(request,{callerId:'O',owner:true}),null);
 const denied=await f.peer.sources.beginNative({slug:'treat-wounds',actors:[f.healer],user:f.peer.game.user,action:f.action,variant:{use(){}},params:{}},{useId:'peer',tag:'exploration-manual:peer'});
 assert.ok(f.peer.sources.poolMarker(denied));assert.equal(await f.peer.sources.resolveSource(request,{callerId:'G',owner:true}),null);
 const peerAdmission=f.peer.gate({phase:'admit',batch:{message:copy}});assert.equal(peerAdmission.status,'participating');assert.equal(peerAdmission.isCurrent(),false);
});
for(const [name,change] of [['deleted result',(f,r)=>f.messages.delete(r.id)],['changed roll',(_f,r)=>{r.rolls[0].total=100}],['source OWNER',(f)=>f.healer.owners.delete('O')],['time',(f)=>{f.owner.game.time.worldTime=101}],['provider',(f)=>f.owner.game.pf2e.actions.clear()]])test(`an observed source loses current qualification after ${name}`,async t=>{
 const f=await sourceFixture();t.after(()=>f.close());const result=await nativeSource(f),answer=f.owner.gate({phase:'admit',batch:{message:result}});assert.equal(answer.isCurrent(),true);change(f,result);assert.equal(answer.isCurrent(),false);
});
test('cold reload retains an old participating card as unavailable and never invokes a claim',async t=>{
 const f=await sourceFixture();t.after(()=>f.close());const result=await nativeSource(f);f.owner.sources.stop();let gate,claims=0;
 const fresh=createManualPoolSources({game:f.owner.game,Hooks:f.owner.Hooks,hpPools:{discover:()=>({ready:true,poolUUID:'Actor.M'})},clientNonce:'reload',isIssuer:()=>false,getSession:()=>{throw Error('no player ledger')},getBatchProvider:()=>({descriptor:{version:1,providerId:'pf2e',providerVersion:'8.5.1',protocol:'pf2e-third-party-automation:manual-pool-batch:1',model:'numeric-empty-reception.v1',baseSourceSHA256:sha},subscribe(_observer,options){gate=options.authorizeBatch;return()=>{}}})});fresh.start({claim:()=>{claims++}});t.after(()=>fresh.stop());
 const answer=gate({phase:'admit',batch:{message:result}});assert.equal(answer.status,'participating');assert.equal(answer.isCurrent(),false);assert.equal(claims,0);
});

async function orderedSourceFixture(t,{order='peer-first',native=true,pendingResult=false,legacyObservations=true,...options}={}){
 const roles=['issuer','peer'],arrivals=Object.fromEntries(roles.map(role=>[role,deferred()])),releases=Object.fromEntries(roles.map(role=>[role,deferred()])),completed=Object.fromEntries(roles.map(role=>[role,deferred()])),insertCounts={issuer:0,peer:0},resultRelease=deferred(),enrollRows=[];let f;
 async function barrier(message){
  await Promise.all(roles.map(role=>arrivals[role].promise));
  assert.equal(f.packets.filter(({packet})=>packet.status==='source').length,0);
  assert.equal(f.messages.size,native?1:2);
  const first=order==='peer-first'?'peer':'issuer',second=first==='peer'?'issuer':'peer';
  releases[first].resolve();assert.equal(await completed[first].promise,'inserted');
  releases[second].resolve();assert.equal(await completed[second].promise,legacyObservations?'duplicate-activity':'inserted');
  for(const client of [f.gm,f.peer])assert.equal(await client.recorder.observe({kind:'treatment'}),null);
  await options.afterRegistration?.(f,message);
 }
 f=await sourceFixture({...options,...native?{afterCheck:barrier}:{},
  recorderLedger(ledger,nonce){
   if(!roles.includes(nonce))return ledger;
   // Strict legacy insertion keeps duplicate recovery reachable; atomic
   // observation forwards the current recorder's ledger options.
   return {...ledger,async insertActivity(input,options){
    const first=++insertCounts[nonce]===1;
    if(first){arrivals[nonce].resolve();await releases[nonce].promise}
    try{const value=await ledger.insertActivity(input,legacyObservations?undefined:options);if(first)completed[nonce].resolve('inserted');return value}
    catch(error){if(first)completed[nonce].resolve(error.message);throw error}
   }};
  },
  ...pendingResult?{beforeNativeResult:nonce=>roles.includes(nonce)?resultRelease.promise:undefined}:{},
  async onEnroll(source,client){
   enrollRows.push(await client.ledger.getActivity(source.activityId));
   return options.onEnroll?options.onEnroll(source,client,f):client.recorder.observePoolSource(source);
  }
 });
 t.after(()=>{resultRelease.resolve();for(const role of roles)releases[role].resolve();f.close()});
 assert.notEqual(f.gm.ledger,f.peer.ledger);
 assert.deepEqual((await f.ledger.getSession('S')).manualPoolIssuer,{userId:'G',clientNonce:'issuer'});
 for(const client of [f.gm,f.peer]){
  assert.equal(await client.recorder.observe({kind:'treatment'}),null);
  assert.equal((await client.ledger.snapshot('S')).activities.length,0);
 }
 return {f,barrier,insertCounts,enrollRows,releaseResult:()=>resultRelease.resolve()};
}
async function runOrderedNative(f){
 await f.action.use({actors:[f.healer],target:f.patient}).catch(error=>assert.equal(error.message,'duplicate-activity'));
 await flush();return [...f.messages.values()].find(message=>message.isCheckRoll===false);
}
async function assertSourceAccepted(f,result){
 const requests=f.packets.filter(({packet})=>packet.kind==='continuation'&&packet.status==='source');assert.equal(requests.length,1);
 const source=requests[0].packet.command,replies=f.packets.filter(({nonce,packet})=>nonce==='issuer'&&packet.requestId===requests[0].packet.requestId);
 assert.equal(replies.length,1);assert.equal(replies[0].packet.kind,'continuation-ack');assert.equal(replies[0].packet.status,'accepted');assert.deepEqual(replies[0].packet.proof,source);
 const saved=await f.ledger.snapshot('S');assert.equal(saved.activities.length,1);assert.deepEqual(saved.activities[0].proof.manualPoolSource,source);
 assert.equal(source.resultId,result.id);assert.equal(f.privateReads(),0);assert.equal(f.errors.some(error=>error.message==='manual-pool-source-unconfirmed'),false);
 return saved.activities[0];
}

for(const order of ['peer-first','issuer-first'])test(`native source ACK survives ${order} legacy registration from distinct GM ledgers`,async t=>{
 const {f,enrollRows}=await orderedSourceFixture(t,{order});const result=await runOrderedNative(f);await assertSourceAccepted(f,result);
 assert.equal(enrollRows.length,1);assert.ok(f.errors.some(error=>error.message==='duplicate-activity'));
});
for(const native of [true,false])for(const order of ['peer-first','issuer-first'])test(`${native?'native':'Workbench'} source ACK survives ${order} atomic observation from distinct GM ledgers`,async t=>{
 const {f,barrier,enrollRows}=await orderedSourceFixture(t,{native,order,legacyObservations:false});
 const result=native?await runOrderedNative(f):await workbenchSource(f,{afterResult:barrier}),activity=await assertSourceAccepted(f,result);
 assert.deepEqual(activity.proof.checkIds,[activity.proof.manualPoolSource.checkId]);assert.deepEqual(activity.proof.resultIds,[result.id]);
 assert.equal(enrollRows.length,1);assert.deepEqual(f.errors,[]);
});
test('peer-first native source registers its original result while ordinary result observers are still pending',async t=>{
 const {f,enrollRows,releaseResult}=await orderedSourceFixture(t,{pendingResult:true});const result=await runOrderedNative(f);const activity=await assertSourceAccepted(f,result);
 assert.deepEqual(enrollRows[0].proof.resultIds,[]);assert.deepEqual(activity.proof.resultIds,[result.id]);releaseResult();await flush();
});
test('Workbench source ACK survives peer-first registration at the original result-created boundary',async t=>{
 const {f,barrier}=await orderedSourceFixture(t,{native:false});const result=await workbenchSource(f,{afterResult:barrier});const activity=await assertSourceAccepted(f,result);
 assert.equal(activity.source.adapter,'target-callback-instrumentation-v1');assert.equal(activity.source.sourceSHA,'b3bac907654522fda62b80da182fc253f7147493a465a82081219fb2a0f1308f');
});

async function assertSourceDenied(f){
 const requests=f.packets.filter(({packet})=>packet.kind==='continuation'&&packet.status==='source');assert.equal(requests.length,1);
 const replies=f.packets.filter(({nonce,packet})=>nonce==='issuer'&&packet.requestId===requests[0].packet.requestId);
 assert.equal(replies.length,1);assert.equal(replies[0].packet.kind,'denied');assert.equal(replies[0].packet.status,'unavailable');
 const row=await f.ledger.getActivity(requests[0].packet.command.activityId);assert.equal(row?.proof.manualPoolSource,undefined);return row;
}
const collisions=[
 ['activity ID',row=>{row.id='manual:FOREIGN'}],['session',row=>{row.sessionId='OTHER'}],['provider',row=>{row.providerId='automatic'}],
 ['kind',row=>{row.kind='battle-medicine'}],['state',row=>{row.state='uncertain'}],['manual source',row=>{row.source.manual=false}],
 ['checkpoint reservation',row=>{row.temporalSource={type:'checkpoint-reservation'}}],['source type',row=>{row.source.type='workbench'}],
 ['source message',row=>{row.source.messageId='FOREIGN'}],['native tag',row=>{row.source.tag='exploration-manual:FOREIGN'}],
 ['use',row=>{row.proof.useId='FOREIGN'}],['check',row=>{row.proof.checkIds=['FOREIGN']}],['extra check',row=>{row.proof.checkIds.push('FOREIGN')}],
 ['foreign result',row=>{row.proof.resultIds=['FOREIGN']}],['healer',row=>{row.actorUUID='Actor.M'}],['patient',row=>{row.patientUUIDs=['Actor.M']}],
 ['extra patient',row=>{row.patientUUIDs.push('Actor.H')}],['pool',row=>{row.hpPoolUUIDs=['Actor.P']}],['extra pool',row=>{row.hpPoolUUIDs.push('Actor.P')}],
 ['start',row=>{row.startedAt=101}],['end',row=>{row.endsAt=701}],['duration',row=>{row.durationSeconds=601}],
 ['immunity duration',row=>{row.treatmentImmunitySeconds=600}],['group',row=>{row.groupId='FOREIGN'}],['group proof',row=>{row.groupProof='FOREIGN'}],
 ['empty missing evidence',row=>{row.options.missing=[]}]
];
for(const [name,change] of collisions)test(`duplicate source enrollment rejects a persisted collision in ${name}`,async t=>{
 let duplicate=false,before;
 const {f}=await orderedSourceFixture(t,{pendingResult:true,
  afterRegistration:async(f,check)=>{await f.storage.storage('collision').transact(state=>{change(state.activities[`manual:${check.id}`])});before=await f.ledger.getActivity(`manual:${check.id}`)},
  async onEnroll(source,client){try{return await client.recorder.observePoolSource(source)}catch(error){duplicate=error.message==='duplicate-activity';throw error}}
 });
 await runOrderedNative(f);const row=await assertSourceDenied(f);assert.equal(duplicate,name!=='state');assert.deepEqual(row,before);
});
test('legacy duplicate enrollment rejects an activity that no longer awaits evidence',async t=>{
 let before,enrolled=false;
 const {f}=await orderedSourceFixture(t,{pendingResult:true,
  afterRegistration:async(f,check)=>{await f.storage.storage('collision').transact(state=>{state.activities[`manual:${check.id}`].state='uncertain'});before=await f.ledger.getActivity(`manual:${check.id}`)},
  onEnroll(){enrolled=true;throw Error('duplicate-activity')}
 });
 await runOrderedNative(f);const row=await assertSourceDenied(f);assert.equal(enrolled,true);assert.deepEqual(row,before);
});

for(const [name,change] of [
 ['source SHA',row=>{row.source.sourceSHA='foreign'}],['lexical source',row=>{row.source.lexicalSource=false}],
 ['adapter SHA',row=>{row.source.adapterSHA='foreign'}],['adapter',row=>{row.source.adapter='foreign'}],
 ['foreign stage',row=>{row.proof.resultIds.push('FOREIGN')}],['group proof',row=>{row.groupProof='foreign'}]
])test(`peer-first Workbench source rejects a persisted ${name} collision`,async t=>{
 const {f,barrier}=await orderedSourceFixture(t,{native:false,afterRegistration:async(f,result)=>f.storage.storage('collision').transact(state=>{change(state.activities[`manual:${result.id}`])})});
 await workbenchSource(f,{afterResult:barrier});await assertSourceDenied(f);
});

for(const native of [true,false])test(`${native?'native':'Workbench'} duplicate recovery preserves original continual-recovery immunity and group`,async t=>{
 const {f,barrier}=await orderedSourceFixture(t,{native});f.healer.items.push({slug:'continual-recovery'},...native?[]:[{slug:'ward-medic'}]);
 const result=native?await runOrderedNative(f):await workbenchSource(f,{afterResult:barrier}),row=await assertSourceAccepted(f,result);
 assert.equal(row.treatmentImmunitySeconds,600);
 if(native){assert.equal(row.groupId,`manual:${row.proof.checkIds[0]}`);assert.equal(row.groupProof,undefined)}
 else{const meta=result.flags[MODULE_ID].explorationManual;assert.equal(row.groupId,`wb:${meta.useId}:Actor.H`);assert.equal(row.groupProof,`lexical:b3bac907654522fda62b80da182fc253f7147493a465a82081219fb2a0f1308f:${meta.useId}`)}
});

const freshness=[
 ['OWNER',(f)=>f.healer.owners.delete('O')],['document',(f)=>{const result=[...f.messages.values()].find(message=>message.isCheckRoll===false);f.messages.set(result.id,{...result})}],
 ['provider',(f)=>f.gm.game.pf2e.actions.clear()],['world time',(f)=>{f.gm.game.time.worldTime=101}],['stopped session',(_f,state)=>{state.sessions.S.status='stopped'}]
];
for(const boundary of ['read','commit','source-commit'])for(const [name,change] of freshness)test(`duplicate recovery rejects ${name} drift at the ${boundary} boundary`,async t=>{
 let f,armed=false,mutations=0,appendCalls=0,sourceCalls=0;
 const ordered=await orderedSourceFixture(t,{pendingResult:true,
  createLedger(storage,nonce){
   if(nonce!=='issuer')return storage.client(nonce);
   const real=storage.storage(nonce);
   return createLedger({...real,isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:nonce}),transact:(fn,options)=>{
    let state;
    return real.transact((current,context)=>{state=current;return fn(current,context)},options?.validateCommit?{validateCommit(){
     if(armed){armed=false;mutations++;change(f,state)}return options.validateCommit();
    }}:options);
   }});
  },
  sourceLedger(ledger,nonce){if(nonce!=='issuer')return ledger;return {...ledger,
   async getActivity(id){const row=await ledger.getActivity(id);if(boundary==='read'){mutations++;if(name==='stopped session')await ledger.updateSession('S',{status:'stopped'});else change(f)}return row},
   async appendManualEvidence(...args){appendCalls++;if(boundary==='commit')armed=true;return ledger.appendManualEvidence(...args)},
   async recordManualPoolSource(...args){sourceCalls++;if(boundary==='source-commit')armed=true;return ledger.recordManualPoolSource(...args)}
  }}
 });f=ordered.f;
 await runOrderedNative(f);const row=await assertSourceDenied(f);assert.equal(mutations,1);
 assert.equal(appendCalls,boundary==='read'&&name!=='stopped session'?0:1);assert.equal(sourceCalls,boundary==='source-commit'?1:0);
 assert.deepEqual(row.proof.resultIds,boundary==='source-commit'?['msg2']:[]);
});

for(const [name,change] of collisions.filter(([name])=>['duration','immunity duration','group','group proof','foreign result'].includes(name)))test(`duplicate recovery rechecks latest ${name} during append commit`,async t=>{
 let armed=false,changes=0;
 const {f}=await orderedSourceFixture(t,{pendingResult:true,
  createLedger(storage,nonce){if(nonce!=='issuer')return storage.client(nonce);const real=storage.storage(nonce);return createLedger({...real,isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:nonce}),transact:(fn,options)=>{
   let row;return real.transact((state,context)=>{const value=fn(state,context);row=state.activities['manual:msg1'];return value},options?.validateCommit?{validateCommit(){if(armed){armed=false;changes++;change(row)}return options.validateCommit()}}:options);
  }})},
  sourceLedger(ledger,nonce){return nonce==='issuer'?{...ledger,appendManualEvidence(...args){armed=true;return ledger.appendManualEvidence(...args)}}:ledger}
 });await runOrderedNative(f);const row=await assertSourceDenied(f);assert.equal(changes,1);assert.deepEqual(row.proof.resultIds,[]);
});

test('duplicate recovery preserves newer ordinary IDs, options and protected claim state',async t=>{
 let captured,latest;
 const {f}=await orderedSourceFixture(t,{pendingResult:true,sourceLedger(ledger,nonce){return nonce==='issuer'?{...ledger,
  async getActivity(id){captured=await ledger.getActivity(id);await f.storage.storage('concurrent').transact(state=>{
   const row=state.activities[id];row.proof.receiptIds=['PRIOR'];row.proof.immunityIds=['Actor.P.Item.PRIOR'];row.proof.poolApplications={effect:{state:'unknown',permitNonce:'prior-permit'}};row.options.label='Original treatment';row.options.missing.push('prior-review');latest=structuredClone(row);
  });return captured}
 }:ledger}});
 const result=await runOrderedNative(f),row=await assertSourceAccepted(f,result);
 assert.deepEqual(captured.proof.receiptIds,[]);assert.deepEqual(row.proof.receiptIds,['PRIOR']);assert.deepEqual(row.proof.immunityIds,['Actor.P.Item.PRIOR']);assert.deepEqual(row.proof.poolApplications,latest.proof.poolApplications);assert.deepEqual(row.options,latest.options);assert.equal(row.state,'awaiting-evidence');
});

test('duplicate recovery preserves the identical source registered after its initial activity read',async t=>{
 let input,captured;
 const {f}=await orderedSourceFixture(t,{pendingResult:true,
  onEnroll(source,client){input=source;return client.recorder.observePoolSource(source)},
  sourceLedger(ledger,nonce){return nonce==='issuer'?{...ledger,async getActivity(id){
   captured=await ledger.getActivity(id);
   await ledger.appendManualEvidence(id,{activity:captured,proof:{resultIds:[input.resultId]},resolveOptions:row=>row.options});
   await ledger.recordManualPoolSource(input,{evidenceGuard:()=>true});return captured;
  }}:ledger}
 });const result=await runOrderedNative(f),row=await assertSourceAccepted(f,result);
 assert.equal(captured.proof.manualPoolSource,undefined);assert.deepEqual(row.proof.manualPoolSource,input);assert.deepEqual(row.proof.resultIds,['msg2']);
});

for(const mode of ['nonduplicate','missing-append','missing-read','missing-source','failed-read','unknown-append','unknown-source'])test(`source enrollment refuses ${mode} without another source request or replacement write`,async t=>{
 let appendCalls=0,sourceCalls=0,unknownWrites=0,f;
 const ordered=await orderedSourceFixture(t,{pendingResult:true,
  ...mode==='nonduplicate'?{async onEnroll(source,client){await client.recorder.observePoolSource(source).catch(error=>assert.equal(error.message,'duplicate-activity'));throw Error('enrollment-unavailable')}}:{},
  sourceLedger(ledger,nonce){if(nonce!=='issuer')return ledger;return {...ledger,
   getActivity:mode==='missing-read'?undefined:mode==='failed-read'?async()=>{throw Error('source-read-unavailable')}:ledger.getActivity,
   appendManualEvidence:mode==='missing-append'?undefined:async(...args)=>{appendCalls++;if(mode==='unknown-append')f.storage.setAcknowledgement(ack=>{unknownWrites++;return {...ack,result:[]}});return ledger.appendManualEvidence(...args)},
   recordManualPoolSource:mode==='missing-source'?undefined:async(...args)=>{sourceCalls++;if(mode==='unknown-source')f.storage.setAcknowledgement(ack=>{unknownWrites++;return {...ack,result:[]}});return ledger.recordManualPoolSource(...args)}
  }}
 });f=ordered.f;await runOrderedNative(f);
 const requests=f.packets.filter(({packet})=>packet.kind==='continuation'&&packet.status==='source');assert.equal(requests.length,1);
 const reply=f.packets.find(({nonce,packet})=>nonce==='issuer'&&packet.requestId===requests[0].packet.requestId).packet;assert.equal(reply.kind,'denied');
 const row=await f.ledger.getActivity(requests[0].packet.command.activityId);
 assert.equal(appendCalls,['unknown-append','unknown-source','missing-source'].includes(mode)?1:0);assert.equal(sourceCalls,mode==='unknown-source'?1:0);assert.equal(unknownWrites,mode.startsWith('unknown-')?1:0);
 if(mode==='unknown-source')assert.deepEqual(row.proof.manualPoolSource,requests[0].packet.command);else assert.equal(row.proof.manualPoolSource,undefined);
});

for(const [name,change] of freshness)test(`source refuses ${name} drift after ordinary append without rolling back those IDs`,async t=>{
 let f,sourceCalls=0;
 const ordered=await orderedSourceFixture(t,{pendingResult:true,sourceLedger(ledger,nonce){return nonce==='issuer'?{...ledger,
  async appendManualEvidence(...args){const row=await ledger.appendManualEvidence(...args);if(name==='stopped session')await ledger.updateSession('S',{status:'stopped'});else change(f);return row},
  recordManualPoolSource(...args){sourceCalls++;return ledger.recordManualPoolSource(...args)}
 }:ledger}});f=ordered.f;await runOrderedNative(f);const row=await assertSourceDenied(f);
 assert.deepEqual(row.proof.resultIds,['msg2']);assert.equal(row.state,'awaiting-evidence');assert.equal(sourceCalls,name==='stopped session'?1:0);
});
