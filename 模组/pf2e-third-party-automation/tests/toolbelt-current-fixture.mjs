import fs from 'node:fs';
import vm from 'node:vm';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';

export const currentSocket=JSON.parse(fs.readFileSync(new URL('./fixtures/toolbelt-socket-3.57.1.json',import.meta.url)));
export function currentToolbelt(input=fs.readFileSync(process.env.TOOLBELT_MANUAL_SOURCE)){
 let text=input.toString();
 for(const [start,end,key]of [['function _e(t,e)','a(_e,"createEmitable")','socket'],['function Ly(t)','var yA;','transport'],['async function vA(t)','var yk=','conversion']]){
  const a=text.indexOf(start),b=text.indexOf(end,a);if(a<0||b<a)throw Error('current-socket-fixture-boundary');
  text=text.slice(0,a)+currentSocket.regions[key]+text.slice(b);
 }
 return Buffer.from(text);
}
export function currentPair(){
 const pf2e=buildNativeBridge({source:fs.readFileSync(process.env.PF2E_NATIVE_BUNDLE),version:'8.5.1'});
 return buildSharedManualPair({pf2eSource:pf2e.buffer,toolbeltSource:currentToolbelt(),pf2eVersion:'8.5.1',toolbeltVersion:'unlisted-label'});
}
export function socketFixture({remote=true,patched=true}={}){
 const source=(patched?currentPair().toolbelt:currentToolbelt()).toString(),clients=[],writes=[],packets=[];
 const users=new Map([['G',{id:'G',active:true,isGM:true}],['O',{id:'O',active:true,isGM:false}]]);users.activeGM=users.get('G');
 let finish,reject,alter=packet=>packet;
 const masterPromise=new Promise((resolve,fail)=>{finish=resolve;reject=fail});masterPromise.catch(()=>{});
 const start=source.indexOf('ShareDataTool'),g=source.indexOf('#g({master:',start),h=source.indexOf('async#h(',start);
 const methods=source.slice(g,source.indexOf('async#p(',g))+' '+source.slice(h,source.indexOf('#b(',h));
 function client(id){
  class Actor{constructor(actorId){this.id=actorId;this.uuid='Actor.'+actorId;this.isOwner=id==='G'||!remote;this.system={attributes:{hp:{value:1,max:30,temp:0}}};}update(fields){writes.push({id,fields});return masterPromise}}
  const master=new Actor('M'),patient=new Actor('P'),handlers=new Set(),hooks=[];
  const game={user:{...users.get(id),isActiveGM:id==='G'},userId:id,users,actors:new Map([['M',master]]),modules:new Map([['pf2e-toolbelt',{active:true,version:'unlisted-label'}]])};
  game.socket={on:(_channel,fn)=>handlers.add(fn),off:(_channel,fn)=>handlers.delete(fn),emit:(_channel,packet)=>{
   const sent=structuredClone(alter(packet));packets.push({packet:sent,senderId:id});
   for(const other of clients)for(const handler of other.handlers)void handler(structuredClone(sent),id);
  }};
  const context=vm.createContext({game,Actor,fromUuid:async uuid=>uuid===master.uuid?master:undefined,Hooks:{once:(_event,fn)=>hooks.push(fn)},
   a:fn=>fn,M:{id:'pf2e-toolbelt'},Qo(){},Ne:()=>true,qo:()=>false,zx:()=>false,ui:{notifications:{error(){}}},ie:{shared:x=>x},
   foundry:{abstract:{Document:Actor},utils:{getProperty:(obj,key)=>obj[key],deleteProperty:(obj,key)=>delete obj[key]}},
   u:{entries:Object.entries,isArray:Array.isArray,mapValues:(obj,fn)=>Object.fromEntries(Object.entries(obj).map(([key,value])=>[key,fn(value,key)])),
    pipe:(value,...fns)=>fns.reduce((v,fn)=>fn(v),value),map:fn=>items=>items.map(fn),filter:fn=>items=>items.filter(fn),isDefined:x=>x!==undefined,fromEntries:Object.fromEntries,isPlainObject:obj=>!!obj&&typeof obj==='object'&&!Array.isArray(obj)}});
  const observer=patched?source.slice(source.indexOf('const __toolbeltManualPool='),source.indexOf('/* end toolbelt manual pool */')):'';
  vm.runInContext(observer+Object.values(currentSocket.regions).join('\n')+`;globalThis.emitter=_e('direct',(...args)=>globalThis.directCallback(...args));globalThis.Tool=class{#t=_e('master',this.#g.bind(this));constructor(){this.#t.activate()}get key(){return 'shareData'}getMasterInMemory(){return game.actors.get('M')}getMasterId(){return 'M'}isValidMaster(m){return !!m}#v(e,key){return key==='health'}${methods}pre(patient,fields,options={}){return this.#h(patient,async()=>patient,fields,options,game.user.id)}}`,context);
  for(const fn of hooks)fn();const tool=vm.runInContext('new Tool()',context),api=game.modules.get('pf2e-toolbelt').api?.explorationManualPool;
  const row={game,master,patient,handlers,context,tool,api};clients.push(row);return row;
 }
 const gm=client('G'),owner=client('O');
 return {gm,owner,writes,packets,masterPromise,finish:()=>finish(remote?gm.master:owner.master),reject,tamper:fn=>alter=fn,
  turn:()=>new Promise(resolve=>setImmediate(resolve)),replay:()=>{for(const fn of gm.handlers)void fn(structuredClone(packets[0].packet),packets[0].senderId)}};
}
