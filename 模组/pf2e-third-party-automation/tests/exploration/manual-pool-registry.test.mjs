import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {authorityFixture} from './authority-fixture.mjs';
import {createManualEvents,registerWorkbenchObservation} from '../../scripts/exploration/manual-events.mjs';
import {createManualPoolSources} from '../../scripts/exploration/manual-pool-source.mjs';
import {getNativeActionEvents} from '../../scripts/native-action-events.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';

const flush=async()=>{for(let i=0;i<30;i++)await new Promise(resolve=>setImmediate(resolve))};
const sha='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';
export async function sourceFixture(){
 const storage=await authorityFixture(),ledger=storage.client('issuer');await ledger.createSession({id:'S',manual:true,status:'recording',startedAt:100,budgetEndsAt:700,actorUUIDs:['Actor.H','Actor.P','Actor.M']});
 const users=new Map(['G','O','X'].map(id=>[id,{id,active:true,isGM:id==='G'}]));users.activeGM=users.get('G');
 const actors=new Map(['H','P','M'].map(id=>[id,{id,uuid:`Actor.${id}`,items:[],owners:new Set(['G','O']),testUserPermission(user){return this.owners.has(user.id)}}]));
 const healer=actors.get('H'),patient=actors.get('P'),token={id:'T',uuid:'Scene.S.Token.T',actor:patient},messages=new Map(),clients=[],errors=[];token.parent={tokens:new Map([['T',token]])};let privateReads=0,serial=0;
 function doc(id,data,user){const message={id,author:user,speaker:{actor:'H'},rolls:[],flags:{},...data,toObject(){return structuredClone({id:this.id,author:this.author.id,speaker:this.speaker,flags:this.flags,rolls:this.rolls.map(roll=>roll.toJSON?.()??roll)})},async update(changes){for(const [path,value]of Object.entries(changes)){const keys=path.split('.');let target=this;for(const key of keys.slice(0,-1))target=target[key]??={};target[keys.at(-1)]=structuredClone(value)}for(const client of clients)client.hooks.get('updateChatMessage')?.(this);return this}};return message}
 const roll=(total,formula='{2d8[healing]}')=>({_evaluated:true,total,formula,options:{degreeOfSuccess:2},toJSON(){return {_evaluated:true,total:this.total,formula:this.formula}}});
 function make(id,nonce,issuer){
  const clientLedger=id==='G'?storage.client(nonce):ledger;
  const hooks=new Map(),listeners=new Set(),scenes=new Map();scenes.active={tokens:new Map([['T',token]])};
  const game={user:users.get(id),users,actors,messages,time:{worldTime:100},system:{version:'8.5.1'},scenes,pf2e:{actions:new Map()},socket:{on:(_,fn)=>listeners.add(fn),off:(_,fn)=>listeners.delete(fn),emit(_channel,packet,options){for(const client of clients.filter(client=>options.recipients.includes(client.game.user.id)))for(const receive of client.listeners)receive(structuredClone(packet),id)}}};
  const Hooks={on:(event,fn)=>{hooks.set(event,fn);return event},off:event=>hooks.delete(event)},hpPools={discover:()=>({ready:true,provider:'pf2e-toolbelt',poolUUID:'Actor.M',memberUUIDs:['Actor.M','Actor.P']})};let gate;
  const batchProvider={descriptor:{version:1,providerId:'pf2e',providerVersion:'8.5.1',protocol:'pf2e-third-party-automation:manual-pool-batch:1',model:'numeric-empty-reception.v1',baseSourceSHA256:sha},subscribe(_observer,{authorizeBatch}){gate=authorizeBatch;return()=>{}}};
  const nativeActions=getNativeActionEvents({game}),sources=createManualPoolSources({game,Hooks,ledger:id==='G'?clientLedger:undefined,fromUuid:async uuid=>actors.get(uuid.split('.').at(-1))??(uuid===token.uuid?token:null),hpPools,clientNonce:nonce,isIssuer:()=>issuer,getSession:()=>id==='G'?clientLedger.getSession('S'):(privateReads++,Promise.reject(Error('player-ledger-read'))),getBatchProvider:()=>batchProvider,onEnroll:source=>value.recorder.observePoolSource(source),onError:error=>errors.push(error),timeoutMs:100});
  const recorder=createManualEvents({game,Hooks,ledger:clientLedger,nativeActions,hpPools,manualPoolSources:sources,fromUuid:async uuid=>actors.get(uuid.split('.').at(-1)),sessionId:()=>id==='G'?'S':null,onChange:change=>{if(change.error)errors.push(Error(change.error))}});
  const value={game,Hooks,hooks,listeners,sources,recorder,nativeActions,gate:event=>gate(event)};clients.push(value);sources.start({claim:async()=>{throw Error('claim-not-used-in-source-test')}});recorder.start();return value;
 }
 const gm=make('G','issuer',true),owner=make('O','owner',false),peer=make('G','peer',false);
 function create(data,user=owner.game.user){const id=`msg${++serial}`,message=doc(id,data,user);owner.hooks.get('preCreateChatMessage')?.(message,data);messages.set(id,message);for(const client of clients)client.hooks.get('createChatMessage')?.(message);return message}
 class Variant{async use(params){const check=create({isCheckRoll:true,rolls:[roll(24,'1d20')],flags:{pf2e:{context:{type:'skill-check',origin:{actor:healer.uuid},target:params.target?{actor:patient.uuid}:null,outcome:'success',options:[...(params.rollOptions??[]),'action:treat-wounds']}}}});create({isCheckRoll:false,rolls:[roll(9)],flags:{...structuredClone(check.flags),pf2e:{...structuredClone(check.flags.pf2e),origin:{messageId:check.id}}}});return [{actor:healer,message:check,outcome:'success'}]}}
 const action={slug:'treat-wounds',variants:new Map(),toActionVariant:()=>new Variant(),use(params){return this.toActionVariant().use(params)}};owner.game.pf2e.actions.set('treat-wounds',action);gm.game.pf2e.actions.set('treat-wounds',action);owner.nativeActions.register();
 return {storage,ledger,gm,owner,peer,actors,healer,patient,token,messages,clients,errors,roll,create,action,privateReads:()=>privateReads,close(){for(const client of clients){client.sources.stop();client.recorder.stop();client.nativeActions.cleanup()}}};
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
export async function workbenchSource(f){
 const command=fs.readFileSync(process.env.WORKBENCH_MANUAL_SOURCE??'C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/treat-wounds-actual-command.txt','utf8');assert.equal(createHash('sha256').update(command).digest('hex'),'b3bac907654522fda62b80da182fc253f7147493a465a82081219fb2a0f1308f');
 const getHealSuccess=Function(`${command.slice(command.indexOf('const getHealSuccess ='),command.indexOf('/**\n * Perform a roll'))}\nreturn getHealSuccess;`)();
 const skill={async roll(args){const check=f.create({isCheckRoll:true,rolls:[f.roll(24,'1d20')],flags:{pf2e:{context:{type:'skill-check',origin:{actor:'Actor.H'},options:args.extraRollOptions}}}});await args.callback(check.rolls[0],'success',check);return check}};
 const ChatMessage={create:async data=>f.create(data),getSpeaker:()=>({actor:'H'})};
 class DamageRoll{constructor(formula){Object.assign(this,f.roll(9,formula));this._total=9}async roll(){return this}async toMessage(data){return f.create({...data,isCheckRoll:false,rolls:[this],flags:{...data.flags,pf2e:{context:{origin:{actor:'Actor.H'},options:[]}}}})}}
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
