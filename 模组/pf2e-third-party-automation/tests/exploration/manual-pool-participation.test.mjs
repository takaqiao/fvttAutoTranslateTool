import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {sourceFixture,workbenchSource} from './manual-pool-registry.test.mjs';
import {createExplorationRuntime} from '../../scripts/exploration/runtime.mjs';
import {getNativeActionEvents} from '../../scripts/native-action-events.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';
import {OWNER_TRANSPORT_CHANNEL} from '../../scripts/exploration/owner-transport.mjs';
import {samePermit} from '../../scripts/exploration/owner-command.mjs';
import {patchNativeManualPoolBatch} from '../../tools/native-manual-pool-batch/patch.mjs';

const hash=value=>createHash('sha256').update(value).digest('hex');
const input=(name,sha)=>{assert.ok(process.env[name],`${name} is required`);const source=fs.readFileSync(process.env[name],'utf8');assert.equal(hash(source),sha,name);return source};
const native=input('PF2E_MANUAL_POOL_BATCH_SOURCE','d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');
const command=input('WORKBENCH_MANUAL_SOURCE','b3bac907654522fda62b80da182fc253f7147493a465a82081219fb2a0f1308f');
const seam=fs.readFileSync(new URL('../../tools/native-manual-pool-batch/seam.test.mjs',import.meta.url),'utf8');
function region(source,start,end){const a=source.indexOf(start),b=source.indexOf(end,a);assert.ok(a>=0&&b>a);return source.slice(a,b)}
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject}}
const flush=async()=>{for(let i=0;i<30;i++)await new Promise(setImmediate)};
const makeButton=Function('vm','assert','region','deferred','return ('+region(seam,'function fixture(source,','\n\ntest(')+')')(vm,assert,region,deferred);
const patched=patchNativeManualPoolBatch(Buffer.from(native)).bytes.toString('utf8');

async function deniedButton(result,gate){
 const admission=gate({phase:'admit',batch:{message:result}});
 assert.equal(admission.status,'participating');assert.equal(admission.isCurrent(),false);
 assert.ok(result.flags[MODULE_ID].explorationManualPoolParticipation);
 const button=makeButton(patched);button.result.flags=structuredClone(result.flags);button.select([button.token('P')]);
 const off=button.subscribe(event=>gate({phase:event.phase,batch:{...event.batch,message:result}}));
 await assert.rejects(button.run(),/evidence-changed/);
 await assert.rejects(button.run(),/source-already-attempted/);
 off();await assert.rejects(button.run(),/source-already-attempted/);
 assert.equal(button.nativeCalls.length,0);assert.equal(button.frames.length,0);
}

for(const kind of ['native','workbench'])for(const change of ['time','provider','provider throw'])test(`${kind} preserves deny-only participation when ${change} changes after the accepted ticket`,async t=>{
 const f=await sourceFixture();t.after(()=>f.close());let changed=false,scope,macro;
 if(kind==='native'){const begin=f.owner.sources.beginNative;f.owner.sources.beginNative=(...args)=>{scope=args[0];return begin(...args)}}
 else{const observe=f.owner.recorder.observeWorkbenchProvider;f.owner.recorder.observeWorkbenchProvider=witness=>{macro=witness.macro;return observe(witness)}}
 const emit=f.gm.game.socket.emit;
 f.gm.game.socket.emit=(channel,packet,...args)=>{
  if(packet.status==='accepted'&&packet.proof?.sourceNonce&&!packet.proof?.resultId){
   changed=true;
   if(change==='time')for(const client of f.clients)client.game.time.worldTime=101;
   else if(change==='provider throw'){
    if(kind==='native')f.owner.game.pf2e.actions.values=()=>{throw Error('provider-changed')};
    else Object.defineProperty(macro,'command',{get(){throw Error('provider-changed')}});
   }else if(kind==='native')scope.variant.use=async()=>[];
   else macro.command+=' ';
  }
  return emit(channel,packet,...args);
 };
 if(kind==='native')await f.action.use({actors:[f.healer],target:f.patient});else await workbenchSource(f);
 await flush();assert.equal(changed,true);
 const result=[...f.messages.values()].find(message=>kind==='native'?message.isCheckRoll===false:message.flags?.treat_wounds_battle_medicine);
 assert.ok(result,'the original source must still save its real result');assert.equal(result.rolls[0].total,9);
 await deniedButton(result,f.owner.gate);
 const snapshot=await f.ledger.snapshot('S');assert.ok(snapshot.activities.every(activity=>!activity.proof.manualPoolSource&&Object.keys(activity.proof.poolApplications??{}).length===0));
});

// Reuse the document/transport boundary from the public runtime fixture. The
// resolver includes the actual patient/master collection, and socketlib routes
// retain the authenticated caller rather than calling a private recorder API.
const publicFixture=fs.readFileSync(new URL('./native-owner-public-runtime.test.mjs',import.meta.url),'utf8');
let runtimeFixture=region(publicFixture,'function publicRuntimeFixture() {','\nfor(const mapped of');
const resolver='fromUuid:async uuid=>uuid===actor.uuid?actor:null',registration='await runtime.bind({});await runtime.register({socket});';
assert.equal(runtimeFixture.split(resolver).length,2);assert.equal(runtimeFixture.split(registration).length,2);
runtimeFixture=runtimeFixture.replace(resolver,"fromUuid:async uuid=>game.actors.get(uuid.split('.').at(-1))??null").replace(registration,`await configureTab(tab);await runtime.bind({});await runtime.register({socket:{register(name,fn){tab.rpcHandlers??=new Map();tab.rpcHandlers.set(name,fn)},executeAsGM(name,...args){const target=tabs.find(peer=>peer.game.user.isGM);return target.rpcHandlers.get(name).apply({socketdata:{userId}},args)}}});`);
async function runtimeSources({recording=true}={}){
 const messages=new Map(),gates=new Map(),controls=new Map();let sequence=0,masterWrites=0;
 const roll=total=>({_evaluated:true,total,options:{degreeOfSuccess:2},toJSON(){return {_evaluated:true,total:this.total}}});
 function create(tab,data){
  const message={id:'late'+(++sequence),author:tab.game.user,speaker:{actor:'H'},...data,toObject(){return structuredClone({id:this.id,author:this.author.id,speaker:this.speaker,flags:this.flags,rolls:this.rolls.map(roll=>roll.toJSON())})},async update(changes){for(const [key,value]of Object.entries(changes)){const keys=key.split('.');let at=this;for(const part of keys.slice(0,-1))at=at[part]??={};at[keys.at(-1)]=structuredClone(value)}return this}};
  for(const [event,fn]of tab.hooks.values())if(event==='preCreateChatMessage')fn(message,data);
  messages.set(message.id,message);for(const peer of f.tabs)for(const [event,fn]of peer.hooks.values())if(event==='createChatMessage')fn(message);return message;
 }
 async function hold(tab){const control=controls.get(tab);if(control){control.entered.resolve();await control.release.promise}}
 async function configure(tab){
  const {game}=tab,healer=game.actors.get('H');healer.items=[];healer.system.attributes.hp.temp=0;
  for(const id of ['P','M'])game.actors.set(id,{...healer,id,uuid:'Actor.'+id,items:[],system:structuredClone(healer.system),modules:{},async update(){masterWrites++;return this}});
  const patient=game.actors.get('P'),master=game.actors.get('M');patient.modules={'pf2e-toolbelt':{shareData:{data:{health:true}}}};
  game.modules.set('pf2e-toolbelt',{active:true,version:'3.56.5'});game.messages=messages;
  game.toolbelt={api:{shareData:{getMasterInMemory:actor=>actor===patient?master:null,getSlavesInMemory:()=>[patient]}}};
  const setting=game.settings.get;game.settings.get=(scope,key)=>scope==='pf2e-toolbelt'&&key==='shareData.enabled'?true:setting(scope,key);
  game.pf2e.thirdPartyManualPoolBatch={descriptor:{version:1,providerId:'pf2e',providerVersion:'8.5.1',protocol:'pf2e-third-party-automation:manual-pool-batch:1',model:'numeric-empty-reception.v1',baseSourceSHA256:hash(native)},subscribe(_fn,opts){gates.set(tab,opts.authorizeBatch);return()=>gates.delete(tab)}};
  class Variant{async use(params){
   await hold(tab);
   const check=create(tab,{isCheckRoll:true,rolls:[roll(24)],flags:{pf2e:{context:{type:'skill-check',origin:{actor:healer.uuid},target:{actor:patient.uuid},options:[...params.rollOptions,'action:treat-wounds'],outcome:'success'}}}});
   create(tab,{isCheckRoll:false,rolls:[roll(9)],flags:{...structuredClone(check.flags),pf2e:{...structuredClone(check.flags.pf2e),origin:{messageId:check.id}}}});
   return [{actor:params.actors[0],message:check,outcome:'success'}];
  }}
  game.pf2e.actions.set('treat-wounds',{slug:'treat-wounds',variants:new Map(),toActionVariant:()=>new Variant(),use(params){return this.toActionVariant().use(params)}});
  const token={id:'T',uuid:'Scene.S.Token.T',actor:patient},ChatMessage={create:async data=>create(tab,data),getSpeaker:()=>({actor:'H'})};token.parent={tokens:new Map([['T',token]])};game.scenes=new Map();game.scenes.active={tokens:token.parent.tokens};
  const skill={async roll(args){await hold(tab);const check=create(tab,{isCheckRoll:true,rolls:[roll(24)],flags:{pf2e:{context:{type:'skill-check',origin:{actor:healer.uuid},options:args.extraRollOptions}}}});await args.callback(check.rolls[0],'success',check);return check}};
  class DamageRoll{constructor(formula){Object.assign(this,roll(9));this.formula=formula;this._total=9}async roll(){return this}toJSON(){return {_evaluated:true,total:this.total,formula:this.formula}}async toMessage(data){return create(tab,{...data,isCheckRoll:false,rolls:[this],flags:{...data.flags,pf2e:{context:{origin:{actor:healer.uuid},options:[]}}}})}}
  const getHealSuccess=Function(region(command,'const getHealSuccess =','/**\n * Perform a roll')+'return getHealSuccess;')();
  const macro={name:'XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine',command,clone(changes){return {...this,...changes}},async execute(input){
   const observed=region(this.command,'const explorationBases =','async function applyChanges');
   const call=Function('game','token','ChatMessage','DamageRoll','CheckRoll','event','getHealSuccess','getRollOptions','dsnHook','explorationManualTarget',observed+';return rollTreatWounds;')(game,{actor:healer},ChatMessage,DamageRoll,class{},null,getHealSuccess,()=>[],callback=>callback(),input.explorationManualTarget);
   return call({DC:15,bonus:0,skillUsed:skill,isRiskySurgery:false,isRightHandBlood:false,useMortalHealing:false,useMagicHands:false,assurance:false,bmtw:'Treat Wounds',target:token,immunityEffect:{name:'Treat Wounds',system:{duration:{value:60,unit:'minutes'}}},usedBattleMedicsBaton:false,spellStitcherBonus:0,immunityMacroLink:''});
  }};
  tab.macro=macro;game.packs.set('xdy-pf2e-workbench.asymonous-benefactor-macros-internal',{getDocuments:async()=>[macro],getDocument:async()=>macro});
 }
 // The fixture factory closes over configureTab; its source continues to invoke
 // the real imported createExplorationRuntime, bind and register methods.
 const factory=Function('assert','createExplorationRuntime','MODULE_ID','OWNER_TRANSPORT_CHANNEL','samePermit','configureTab','return ('+runtimeFixture+')')(assert,createExplorationRuntime,MODULE_ID,OWNER_TRANSPORT_CHANNEL,samePermit,configure);
 const f=factory();const gm=await f.addTab('GM','G');await f.provision(gm);const owner=await f.addTab('OWNER','O');
 const session=recording?await gm.runtime.api.start({manual:true,actorUUIDs:['Actor.H','Actor.P','Actor.M']}):null;
 return {f,gm,owner,messages,gates,session,masterWrites:()=>masterWrites,hold(tab){const entered=deferred(),release=deferred();controls.set(tab,{entered,release});return {entered:entered.promise,release:()=>release.resolve()}},use(tab,kind){return kind==='native'?tab.game.pf2e.actions.get('treat-wounds').use({actors:[tab.game.actors.get('H')],target:tab.game.actors.get('P')}):tab.macro.execute({})},close(){for(const tab of f.tabs)getNativeActionEvents({game:tab.game}).cleanup();f.dispose()}};
}

for(const kind of ['native','workbench'])for(const trigger of ['GM Stop','OWNER reset'])test(`real runtime ${trigger} preserves the pending ${kind} source as denied after replacement`,async t=>{
 const r=await runtimeSources();t.after(()=>r.close());const tab=trigger==='GM Stop'?r.gm:r.owner;
 let work;
 if(trigger==='GM Stop'){
  const hold=r.hold(tab);work=r.use(tab,kind);work.catch(()=>{});await hold.entered;
  await r.gm.runtime.api.stop(r.session.id,'test-stop');assert.equal((await r.gm.runtime.api.snapshot(r.session.id)).session.status,'paused');hold.release();
 }else{
  const emit=r.gm.game.socket.emit;let reset=false;
  r.gm.game.socket.emit=(channel,packet,...args)=>{if(packet.operationId==='manual-pool-source'&&packet.status==='accepted'&&packet.proof?.sourceNonce&&!packet.proof?.resultId){reset=true;for(const listener of tab.listeners.get('disconnect')??[])listener()}return emit(channel,packet,...args)};
  work=r.use(tab,kind);await work;assert.equal(reset,true);
 }
 await work;await flush();const result=[...r.messages.values()].find(message=>kind==='native'?message.isCheckRoll===false:message.flags?.treat_wounds_battle_medicine);
 assert.ok(result);assert.equal(result.rolls[0].total,9);await deniedButton(result,event=>r.gates.get(tab)(event));
 const snapshot=await r.gm.runtime.api.snapshot(r.session.id);assert.ok(snapshot.activities.every(activity=>!activity.proof.manualPoolSource&&Object.keys(activity.proof.poolApplications??{}).length===0));
 assert.equal(r.masterWrites(),0);
 assert.equal(r.f.packets.some(({packet})=>packet.operationId==='manual-pool-hp'),false);
});

for(const kind of ['native','workbench'])for(const condition of ['inactive','no recording','nonshared'])test(`${kind} retains ordinary unregistered admission for explicit ${condition}`,async t=>{
 if(condition!=='inactive'){
  const r=await runtimeSources({recording:condition!=='no recording'});t.after(()=>r.close());
  if(condition==='nonshared')for(const tab of r.f.tabs)tab.game.actors.get('P').modules={};
  await r.use(r.owner,kind);await flush();const result=[...r.messages.values()].find(message=>kind==='native'?message.isCheckRoll===false:message.flags?.treat_wounds_battle_medicine);
  assert.equal(r.gates.get(r.owner)({phase:'admit',batch:{message:result}}).status,'unregistered');
  const button=makeButton(patched);button.result.flags=structuredClone(result.flags);button.select([button.token('P')]);button.subscribe(event=>r.gates.get(r.owner)({phase:event.phase,batch:{...event.batch,message:result}}));await button.run();assert.equal(button.nativeCalls.length,1);assert.equal(button.frames[0].callFrame,null);return;
 }
 const f=await sourceFixture();t.after(()=>f.close());
 if(condition==='inactive')await f.ledger.updateSession('S',{status:'stopped'});
 if(kind==='native')await f.action.use({actors:[f.healer],target:f.patient});else await workbenchSource(f);await flush();const result=[...f.messages.values()].find(message=>kind==='native'?message.isCheckRoll===false:message.flags?.treat_wounds_battle_medicine);
 assert.equal(f.owner.gate({phase:'admit',batch:{message:result}}).status,'unregistered');
 const button=makeButton(patched);button.result.flags=structuredClone(result.flags);button.select([button.token('P')]);button.subscribe(event=>f.owner.gate({phase:event.phase,batch:{...event.batch,message:result}}));await button.run();assert.equal(button.nativeCalls.length,1);assert.equal(button.frames[0].callFrame,null);
});
