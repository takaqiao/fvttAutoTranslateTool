import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {patchToolbeltManualPool} from './patch.mjs';

const input=fs.readFileSync(process.env.TOOLBELT_MANUAL_SOURCE??'C:/Users/Taka/Desktop/fvtt/output/exploration-quality-goal-20260930/qa/runtime/Data/modules/pf2e-toolbelt/scripts/main.js');
const original=input.toString('utf8');
const turn=()=>new Promise(resolve=>setImmediate(resolve));
function methods(source){const start=source.indexOf('ShareDataTool'),g=source.indexOf('#g({master:',start),h=source.indexOf('async#h(',start);return source.slice(g,source.indexOf('async#p(',g))+' '+source.slice(h,source.indexOf('#b(',h))}
export function seamFixture({patched=true,remote=false}={}){
 const source=(patched?patchToolbeltManualPool(input).bytes:input).toString('utf8'),clients=[],writes=[],events=[];let finish,reject,markWrite,alterForward=value=>value;
 const writeStarted=new Promise(resolve=>{markWrite=resolve});
 const masterPromise=new Promise((resolve,fail)=>{finish=resolve;reject=fail});masterPromise.catch(()=>{});
 const users=new Map([['G',{id:'G',active:true,isGM:true}],['O',{id:'O',active:true,isGM:false}]]);users.activeGM=users.get('G');
 function client(id){
  const hooks=new Map(),socketHandlers=[],master={id:'M',uuid:'Actor.M',isOwner:id==='G'||!remote,update(fields){writes.push({id,fields});markWrite();return masterPromise}},patient={id:'P',uuid:'Actor.P'};
  for(const actor of [master,patient])actor.system={attributes:{hp:{value:1,max:30,temp:0}}};
  const game={user:{...users.get(id),isActiveGM:id==='G'},userId:id,users,actors:new Map([['M',master]]),modules:new Map([['pf2e-toolbelt',{version:'3.56.5'}]])};
  const context=vm.createContext({game,Hooks:{once:(event,fn)=>hooks.set(event,fn)},console,Promise,Object,JSON,Error,TypeError,Number,Map,Set,
   a:fn=>fn,M:{id:'pf2e-toolbelt'},foundry:{utils:{getProperty:(obj,key)=>obj[key],deleteProperty:(obj,key)=>delete obj[key]}},
   u:{pipe:(value,...fns)=>fns.reduce((v,fn)=>fn(v),value),map:fn=>items=>items.map(fn),filter:fn=>items=>items.filter(fn),isDefined:x=>x!==undefined,fromEntries:Object.fromEntries,isPlainObject:()=>true},
   Qo(){},IA:value=>({...value,master:value.master.uuid}),Xo:payload=>{for(const other of clients.filter(c=>c.game.user.isActiveGM))for(const handler of other.socketHandlers)void handler({...alterForward(payload),master:other.master},id)},
   vA:async value=>{delete value.__type__;return value},Ne:()=>true,Ly:fn=>socketHandlers.push(fn),Uy:()=>{},ui:{notifications:{error(){}}},ie:{shared:x=>x}});
  const observer=source.includes('const __toolbeltManualPool=')?source.slice(source.indexOf('const __toolbeltManualPool='),source.indexOf('/* end toolbelt manual pool */')+'/* end toolbelt manual pool */'.length):'';
  const emitter=original.slice(original.indexOf('function _e('),original.indexOf('a(_e,"createEmitable")'));
  vm.runInContext(observer+emitter+`;globalThis.Tool=class{#t=_e('master',this.#g.bind(this));constructor(){this.#t.activate()} get key(){return 'shareData'}getMasterInMemory(){return game.actors.get('M')}getMasterId(){return 'M'}isValidMaster(m){return !!m}#v(e,key){return key==='health'}${methods(source)} pre(patient,fields,options={}){return this.#h(patient,async()=>patient,fields,options,game.user.id)}}`,context);
  hooks.get('ready')?.();const tool=vm.runInContext('new Tool()',context),nativeAPI=game.modules.get('pf2e-toolbelt').api?.explorationManualPool;
  // The extracted bundle lives in a VM realm; the application normally shares
  // its realm. Preserve document references and clone only returned JSON proof.
  const api=nativeAPI&&{descriptor:nativeAPI.descriptor,subscribe:fn=>nativeAPI.subscribe(event=>fn(event.terminalPromise?{...event,terminalPromise:event.terminalPromise.then(proof=>structuredClone(proof))}:event))};
  const result={game,context,tool,master,patient,api,socketHandlers};clients.push(result);return result;
 }
 const gm=client('G'),owner=client('O');return {gm,owner,writes,events,masterPromise,writeStarted,tamperForward:fn=>{alterForward=fn},finish:()=>finish(remote?gm.master:owner.master),reject,turn};
}

test('fixed original local and socket chains return the patient before the unawaited master',async()=>{
 for(const remote of [false,true]){const f=seamFixture({patched:false,remote});let returned=false;await f.owner.tool.pre(f.owner.patient,{'system.attributes.hp.value':20}).then(()=>returned=true);await turn();assert.equal(returned,true);assert.equal(f.writes.length,1);f.finish()}
});

for(const remote of [false,true])test(`original ${remote?'authenticated socket':'local OWNER'} master Promise is the only terminal`,async()=>{
 const f=seamFixture({remote}),binding={permitNonce:'permit',applicationNonce:'application',ownerUserId:'O',patientUUID:'Actor.P',poolUUID:'Actor.M'};let terminal,fulfilled=false;
 assert.ok(f.owner.api,'fixed provider must expose observation');
 f.owner.api.subscribe(event=>{if(event.phase==='prepare')return {binding,validate:()=>true,beforeWrite:()=>true};if(event.phase==='write'){terminal=event.terminalPromise;terminal.then(()=>fulfilled=true)}});
 f.gm.api.subscribe(event=>{if(event.phase==='authorize'){assert.equal(event.senderId,'O');return {validate:()=>true,beforeWrite:()=>true}}if(event.phase==='write'){terminal=event.terminalPromise;terminal.then(()=>fulfilled=true)}});
 await f.owner.tool.pre(f.owner.patient,{'system.attributes.hp.value':20});await turn();assert.equal(f.writes.length,1);assert.equal(fulfilled,false);f.finish();
 const proof=await terminal;assert.equal(proof.poolUUID,'Actor.M');assert.equal(proof.writerUserId,remote?'G':'O');assert.equal(fulfilled,true);assert.equal(f.writes[0].fields.__explorationManualPool,undefined);
 assert.deepEqual(Object.keys(f.owner.api).sort(),['descriptor','subscribe']);
});
