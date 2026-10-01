import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import * as patcher from './patch.mjs';

const sourcePath=process.env.PATREON_MANUAL_SOURCE;
const fixedSource=sourcePath?fs.readFileSync(sourcePath,'utf8'):null;
const pf2ePath=process.env.PF2E_MANUAL_SOURCE,pf2eSource=pf2ePath?fs.readFileSync(pf2ePath,'utf8'):null;
const candidate=fixedSource&&patcher?patcher.patchPatreonManualImmunity(Buffer.from(fixedSource)).bytes.toString('utf8'):fixedSource;
const M='pf2e-third-party-automation';
const turn=()=>new Promise(resolve=>setImmediate(resolve));
function region(source,start,end){const from=source.indexOf(start),to=source.indexOf(end,from)+end.length;assert.ok(from>=0&&to>from,'fixed native function region');return source.slice(from,to)}
function fixture(source,{direct=true,hold=false,reject=false}={}){
 let resolve,rejectWrite;const gate=new Promise((a,b)=>{resolve=a;rejectWrite=b});gate.catch(()=>{});
 const writes=[],events=[],sent=[],hooks=new Map(),module={api:{existing:7}},users=new Map(['HUSER','PUSER','GM'].map(id=>[id,{id,isGM:id==='GM'}]));users.activeGM=users.get('GM');
 const healer={id:'H',uuid:'Actor.H',testUserPermission:u=>u?.isGM||u?.id==='HUSER'};
 const patient={id:'P',uuid:'Actor.P',items:new Map(),conditions:{wounded:{async decrease(){writes.push(['wounded'])}}},
  testUserPermission:u=>u?.isGM||u?.id==='PUSER'||direct&&u?.id==='HUSER',canUserModify:u=>u?.isGM||direct,
  createEmbeddedDocuments(type,data){writes.push(['create',type,structuredClone(data)]);return (hold?gate:reject?Promise.reject(Error('create-rejected')):Promise.resolve()).then(()=>{const item={...structuredClone(data[0]),id:'I',uuid:'Actor.P.Item.I',actor:patient,parent:patient};item.system.start={value:game.time.worldTime,initiative:null};item.sourceId='Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5';patient.items.set(item.id,item);return [item]})}};
 const binding={invocationId:'FORGED',messageId:'C',useId:'U',tag:'exploration-manual:U',actorUUID:healer.uuid,patientUUID:patient.uuid,sourceUserId:'HUSER',startedAt:100};
 const check={id:'C',isCheckRoll:true,isReroll:false,author:users.get('HUSER'),speaker:{actor:'H'},actor:healer,rolls:[{_evaluated:true}],
  flags:{[M]:{explorationManualNative:{tag:'exploration-manual:U',useId:'U',patientUUID:patient.uuid,startedAt:100,riskySurgery:false}},pf2e:{modifiers:[],context:{type:'skill-check',origin:{actor:healer.uuid},target:{actor:patient.uuid},options:['action:treat-wounds','exploration-manual:U'],outcome:'success'}}},
  async update(changes){for(const [key,value] of Object.entries(changes)){const keys=key.split('.');let target=this;for(const part of keys.slice(0,-1))target=target[part]??={};target[keys.at(-1)]=structuredClone(value)}return this}};
 const token={actor:patient,document:{uuid:'Scene.S.Token.T'}},targets=new Set([token]);targets.first=()=>[...targets][0];users.get('HUSER').targets=targets;
 const game={user:users.get('HUSER'),users,system:{version:'8.5.1'},time:{worldTime:100},messages:new Map([['C',check]]),actors:new Map([['H',healer],['P',patient]]),modules:new Map([['patreon-v3',module]]),socket:{emit(channel,payload){sent.push(structuredClone(payload))},on(channel,fn){hooks.set(channel,fn)}}};
 class CheckRoll{}
 const context=vm.createContext({game,Hooks:{once(name,fn){hooks.set(name,fn)}},crypto:{randomUUID:()=> 'INV'},h:{CheckRoll},u:'patreon-v3',Aa:'module.patreon-v3',$t:'addItemToActor',r:()=>{},k:()=>game.user===users.activeGM,
  fromUuid:async uuid=>uuid==='Actor.P'?patient:uuid==='Actor.H'?healer:{toObject:()=>({type:'effect',system:{rules:[],duration:{unit:'hours',value:1,expiry:'turn-start',sustained:false},context:{}},flags:{}})},
  foundry:{utils:{mergeObject:(left,right)=>({...left,...right})}},ui:{notifications:{info(){}}},console,structuredClone});
 const observerStart=source.indexOf('const __patreonManualImmunity=');
 if(observerStart>=0)vm.runInContext(source.slice(observerStart,source.indexOf('async function Ma(',observerStart)),context);
 for(const [start,end] of [['async function p(','r(p,"addItemToActor");'],['function x(','r(x,"executeAsGM");'],['function Ia(','r(Ia,"socketListener");'],['async function b(','r(b,"createItemObjectUuid");'],['async function Ma(','r(Ma,"treatWounds");']])vm.runInContext(region(source,start,end),context);
 vm.runInContext(region(source,'B=class{','this.content=e?.content}};'),context);
 hooks.get('init')?.();module.api.explorationManualImmunity?.subscribe(event=>events.push(event));
 const invoke=()=>context.Ma(new context.B(check,null,new Set(check.flags.pf2e.context.options)),patient);
 const receive=()=>{game.user=users.get('GM');context.Ia();const socket=hooks.get('module.patreon-v3');return socket(sent[0])};
 const forge=()=>context.p(patient,{type:'effect',system:{duration:{unit:'hours',value:1}},flags:{[M]:{explorationManualPatreonImmunity:binding}}});
 return {game,module,patient,healer,check,targets,writes,events,sent,invoke,receive,forge,resolve,rejectWrite,context};
}
test('fixed original native writer produces a same-call completion observer',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);await f.invoke();assert.ok(f.module.api.explorationManualImmunity,'native writer observer API');assert.equal(f.events.length,1);
 const proof=await f.events[0].terminalPromise;assert.equal(proof.itemUUID,'Actor.P.Item.I');assert.equal(proof.creatorId,'HUSER');assert.equal(proof.binding.messageId,'C');assert.equal(proof.binding.useId,'U');assert.equal(proof.start,100);assert.equal(proof.expiresAt,3700);assert.equal(f.module.api.existing,7);
 assert.equal(f.writes.filter(w=>w[0]==='create').length,1);assert.equal(f.writes.filter(w=>w[0]==='wounded').length,1);
});
test('original direct create Promise owns completion, not the action return',{skip:!fixedSource},async()=>{
 const f=fixture(candidate,{hold:true});const work=f.invoke();await turn();assert.equal(f.events.length,1);let complete=false;f.events[0].terminalPromise.then(()=>{complete=true});await turn();assert.equal(complete,false);f.resolve();await work;await f.events[0].terminalPromise;
});
test('GM original socket writer receives the sealed real check and reports its own creator',{skip:!fixedSource},async()=>{
 const f=fixture(candidate,{direct:false,hold:true});await f.invoke();assert.equal(f.sent.length,1);assert.equal(f.writes.filter(w=>w[0]==='create').length,0);assert.equal(f.events.length,0,'emit cannot complete an immunity');
 const work=f.receive();await turn();assert.equal(f.events.length,1);assert.equal(f.events[0].binding.actorUUID,'Actor.H');assert.equal(f.events[0].binding.patientUUID,'Actor.P');let complete=false;f.events[0].terminalPromise.then(()=>{complete=true});await turn();assert.equal(complete,false);f.resolve();await work;
 const proof=await f.events[0].terminalPromise;assert.equal(proof.creatorId,'GM');assert.equal(f.writes.filter(w=>w[0]==='create').length,1);
});
test('a forged item marker cannot create source completion',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);await f.forge();assert.equal(f.writes.filter(w=>w[0]==='create').length,1,'provider keeps its native writer');assert.equal(f.events.length,0);
});
for(const change of [f=>{f.check.flags.pf2e.context.target.actor='Actor.Q'},f=>{f.game.messages.delete('C')},f=>{f.check.author={id:'OUTSIDER'}},f=>{f.check.flags.pf2e.context.options=[]}])test('changed native check cannot lend its same-call source to the original writer',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);change(f);await f.invoke();assert.equal(f.writes.filter(w=>w[0]==='create').length,1);assert.equal(f.events.length,0);
});
test('source change before GM delivery rejects completion but never adds another writer',{skip:!fixedSource},async()=>{
 const f=fixture(candidate,{direct:false});await f.invoke();assert.equal(f.sent.length,1);f.check.flags.pf2e.context.target.actor='Actor.Q';await f.receive();assert.equal(f.events.length,0);assert.equal(f.writes.filter(w=>w[0]==='create').length,1);
});
test('native write rejection rejects its observer without replay',{skip:!fixedSource},async()=>{
 const f=fixture(candidate,{reject:true});await assert.rejects(f.invoke(),/create-rejected/);assert.equal(f.events.length,1);await assert.rejects(f.events[0].terminalPromise,/create-rejected/);assert.equal(f.writes.filter(w=>w[0]==='create').length,1);
});
test('Patreon branch parity retains original duration and wounded calls',{skip:!fixedSource},async()=>{
 const before=fixture(fixedSource),after=fixture(candidate);await before.invoke();await after.invoke();await turn();
 assert.deepEqual(after.writes.map(w=>w[0]),before.writes.map(w=>w[0]));assert.deepEqual(after.writes[0][2][0].system,before.writes[0][2][0].system);
});
test('source patch rejects foreign bytes instead of instrumenting guessed handlers',{skip:!patcher},()=>{
 assert.throws(()=>patcher.patchPatreonManualImmunity(Buffer.from('async function Ma(a,e){}')),/source-sha-mismatch/);
});
test('paid output is rejected anywhere inside the complete checkout before reading source',()=>{
 const checkout=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../../..');
 for(const output of [path.join(checkout,'private-patreon'),path.join(checkout,'模组/pf2e-third-party-automation/tools/private-patreon')])
  assert.throws(()=>patcher.preparePatreonManualImmunity('not-opened',output),/paid-output-in-workspace/);
 if(process.platform==='win32')assert.throws(()=>patcher.preparePatreonManualImmunity('not-opened',path.join(checkout.toUpperCase(),'private-patreon')),/paid-output-in-workspace/);
});
test('failed source metadata leaves the original treatment running without completion',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);f.check.update=async()=>{throw Error('metadata-denied')};await f.invoke();
 assert.equal(f.events.length,0);assert.deepEqual(f.writes.map(w=>w[0]),['create','wounded']);
});
test('an unknown marker shape cannot interrupt an unrelated original native writer',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);f.check.flags.pf2e.context.options={includes:undefined};await f.forge();
 assert.equal(f.writes.filter(w=>w[0]==='create').length,1);assert.equal(f.events.length,0);
});
test('observer exceptions preserve the provider original calls',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);const extra=fixture(candidate);extra.module.api.explorationManualImmunity?.subscribe(()=>{throw Error('observer-error')});
 await f.invoke();await extra.invoke();assert.deepEqual(extra.writes.map(w=>w[0]),f.writes.map(w=>w[0]));
});
test('failed creator permission observation preserves the original native writer',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);f.patient.testUserPermission=()=>{throw Error('permission-observation-failed')};
 await f.invoke();assert.equal(f.events.length,0);assert.deepEqual(f.writes.map(w=>w[0]),['create','wounded']);
});
test('an unobserved native synchronous failure is never called twice',{skip:!fixedSource},async()=>{
 const f=fixture(candidate),error=Error('native-sync-failure');f.game.messages.delete('C');
 f.patient.createEmbeddedDocuments=()=>{f.writes.push(['create']);throw error};
 await assert.rejects(f.invoke(),failure=>failure===error);assert.equal(f.writes.length,1);assert.equal(f.events.length,0);
});
test('the manual source patch retains the complete existing v5 time handler bytes',{skip:!fixedSource||!patcher},()=>{
 const from=fixedSource.indexOf('const __patreonTimeCompletion='),end=fixedSource.indexOf('r(Mn,"handleFastHealingTime");',from)+'r(Mn,"handleFastHealingTime");'.length;
 assert.ok(from>=0&&end>from);assert.equal(candidate.slice(candidate.indexOf('const __patreonTimeCompletion='),candidate.indexOf('r(Mn,"handleFastHealingTime");')+'r(Mn,"handleFastHealingTime");'.length),fixedSource.slice(from,end));
});

test('an empty native check target binds only the original MessageForHandling single target',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);f.check.flags.pf2e.context.target=null;f.check.flags[M].explorationManualNative.patientUUID=null;
 await f.invoke();assert.equal(f.events.length,1);const proof=await f.events[0].terminalPromise;
 assert.deepEqual(JSON.parse(JSON.stringify(proof.binding.targetSnapshot)),{type:'patreon-single-target',actorUUID:'Actor.P',tokenUUID:'Scene.S.Token.T'});
 assert.equal(f.check.flags[M].explorationManualNative.patientUUID,'Actor.P');assert.equal(proof.binding.patientUUID,'Actor.P');
 assert.equal(f.writes.filter(w=>w[0]==='create').length,1);
});
test('a real check target retains its matching original single token snapshot',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);f.check.flags.pf2e.context.target.token='Scene.S.Token.T';
 await f.invoke();const proof=await f.events[0].terminalPromise;
 assert.ok(proof.binding.targetSnapshot,'native check target snapshot');
 assert.deepEqual(JSON.parse(JSON.stringify(proof.binding.targetSnapshot)),{type:'check-context',actorUUID:'Actor.P',tokenUUID:'Scene.S.Token.T'});
});
test('GM socket uses the original saved target snapshot even when GM targets differ',{skip:!fixedSource},async()=>{
 const f=fixture(candidate,{direct:false});f.check.flags.pf2e.context.target=null;f.check.flags[M].explorationManualNative.patientUUID=null;
 await f.invoke();f.game.users.get('GM').targets=new Set([{actor:{uuid:'Actor.DECOY'}}]);await f.receive();
 assert.equal(f.events.length,1);const proof=await f.events[0].terminalPromise;assert.equal(proof.binding.patientUUID,'Actor.P');assert.equal(proof.creatorId,'GM');
});
test('the native terminal retains the recording session observed before the real action',{skip:!fixedSource},async()=>{
 const f=fixture(candidate);f.check.flags[M].explorationManualNative.recordingSessionId='S';
 await f.invoke();const proof=await f.events[0].terminalPromise;assert.equal(proof.binding.recordingSessionId,'S');
});
for(const [name,change] of [
 ['missing target snapshot',f=>f.targets.clear()],
 ['multiple original targets',f=>f.targets.add({actor:{uuid:'Actor.Q'},document:{uuid:'Scene.S.Token.Q'}})],
 ['unknown original token identity',f=>[...f.targets][0].document.uuid=null],
])test(`an empty check target with ${name} keeps native writes unconfirmed`,{skip:!fixedSource},async()=>{
 const f=fixture(candidate);f.check.flags.pf2e.context.target=null;f.check.flags[M].explorationManualNative.patientUUID=null;change(f);
 await f.invoke();assert.equal(f.events.length,0);assert.equal(f.writes.filter(w=>w[0]==='create').length,1);
});

test('fixed PF2e DefaultUse leaves its absent target to the native check helper',{skip:!pf2eSource},async()=>{
 assert.equal(createHash('sha256').update(pf2eSource).digest('hex'),'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');
 const targets=[],action={modifiers:[],notes:[],rollOptions:['action:treat-wounds'],statistic:'medicine',name:'Treat Wounds'};
 class Ns{}class TokenPF2e{}class BaseActionVariant{constructor(){this.name=null;this.cost=null;this.traits=[]}}
 const context=vm.createContext({BaseActionVariant,Ns,TokenPF2e,_loc:x=>x,J:class{},Wi:class{},getActionGlyph:()=>null,toRollNoteSource:x=>x,isValidDifficultyClass:()=>false,
  G:{async simpleRollActionCheck(options){targets.push(options.target());options.callback({message:{id:'C'}})}}});
 vm.runInContext('Number.isNumeric=()=>false;',context);
 const from=pf2eSource.indexOf('var SingleCheckActionVariant = class'),to=pf2eSource.indexOf('}, SingleCheckAction = class',from);
 assert.ok(from>=0&&to>from);vm.runInContext(pf2eSource.slice(from,to+1)+';',context);
 const variant=new context.SingleCheckActionVariant(action),result=await variant.use({});
 assert.equal(targets[0],null);assert.equal(result[0].message.id,'C');
 const patient=new Ns();await variant.use({target:patient});assert.equal(targets[1].actor,patient);
});

for(const outcome of ['failure','success'])test(`fixed PF2e ${outcome} callback retains the actual native stage boundary`,{skip:!pf2eSource},async()=>{
 const stages=[],dice=[],actor={items:[],id:'H'},message={id:'C',flags:{[M]:{explorationManualNative:{useId:'U',tag:'exploration-manual:U'}},pf2e:{modifiers:[],context:{options:['exploration-manual:U']}}},toObject(){return {flags:structuredClone(this.flags)}}};
 const merge=(a,b)=>{for(const [key,value] of Object.entries(b)){if(value&&typeof value==='object'&&!Array.isArray(value))a[key]=merge(a[key]??{},value);else a[key]=value}return a};
 const context=vm.createContext({CheckFeat:(a,slug)=>a.items.some(i=>i.slug===slug),_loc:x=>x,foundry:{utils:{mergeObject:(a,b)=>merge(a,b)}},
  ChatMessagePF2e:{getSpeaker:()=>({actor:'H'}),create:data=>stages.push(data)},cn:class{constructor(formula){dice.push(formula)}async roll(){return this}toJSON(){return {_evaluated:true}}}});
 vm.runInContext(region(pf2eSource,'async function treatWoundsMacroCallback(','}\nvar BaseStatistic').replace(/\nvar BaseStatistic$/,''),context);
 await context.treatWoundsMacroCallback({actor,bonus:0,message,outcome});
 assert.equal(stages.length,outcome==='failure'?0:1);assert.equal(dice.length,stages.length);
 if(stages.length){assert.equal(stages[0].flags.pf2e.origin.messageId,'C');assert.equal(stages[0].flags[M].explorationManualNative.useId,'U')}
});
