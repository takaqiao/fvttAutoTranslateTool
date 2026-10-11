export function sourceText(source){
 if(typeof source==='string')return source;
 if(!(source instanceof Uint8Array))throw Error('native-source-seam-input');
 try{return new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(source)}catch{throw Error('native-source-seam-encoding')}
}
export function sourceRegion(source,start,end,reason='native-source-seam'){
 if(source.split(start).length!==2)throw Error(`${reason}-ambiguous`);
 const from=source.indexOf(start),to=source.indexOf(end,from+start.length);
 if(to<0)throw Error(`${reason}-end`);
 return source.slice(from,to);
}
export const nativeMethod=source=>sourceRegion(source,'\tasync applyDamage({','\n\tasync undoDamage(').slice(1);
export const batchRegion=source=>sourceRegion(source,'async function applyDamageFromMessage(','async function shiftAdjustDamage(');
export const flatRegion=source=>sourceRegion(source,'\tbeforePrepareData() {\n\t\tif (this.ignored) return;\n\t\tlet e = this.getReducedLabel()','\tasync afterRoll(');
export const stackingRegion=source=>sourceRegion(source,'var HIGHER_BONUS =','var StatisticModifier =');
export const TOOL_RECEIVE='#g({master:e,...n}){this.isValidMaster(e)&&e.update(n)}';
export const TOOL_RECEIVE_PATCHED='#g({master:e,__explorationManualPool:x,...n},s){return __toolbeltManualPool.receive(e,n,x,s,()=>this.isValidMaster(e)&&e.update(n))}';
export const TOOL_FORWARD='m.isOwner?m.update(g):this.#t.emit({master:m,...g})';
export const TOOL_FORWARD_PATCHED='let x=__toolbeltManualPool.forward(e,m,g,r,()=>m.update(g),p=>this.#t.emit(p));if(x)await x';
export function toolRegions(source){
 const socket=sourceRegion(source,'function _e(t,e)','a(_e,"createEmitable")','toolbelt-source-seam');
 const current=socket.includes('await yA(o)');
 const latest=socket.includes('await IA(o)'),transportEnd=latest?'a(Yo,"socketEmit");':'a(Xo,"socketEmit");';
 return {
 receive:sourceRegion(source,'#g({master:e,','async#p(','toolbelt-source-seam'),
 forward:sourceRegion(source,'async#h(e,n,i,r,o)','#b(e,n,i,r)','toolbelt-source-seam'),
 socket,
 transport:sourceRegion(source,latest?'function Ny(t)':current?'function zy(t)':'function Ly(t)',transportEnd,'toolbelt-source-seam')+transportEnd,
 conversion:sourceRegion(source,latest?'async function IA(t)':current?'async function yA(t)':'async function vA(t)',latest?'var Sk=':'var yk=','toolbelt-source-seam'),
 validity:sourceRegion(source,'isValidActor(e){return!!e&&e instanceof Actor','isValidSlave(e)','toolbelt-source-seam'),
 binding:sourceRegion(source,'#t=_e(this.path("master")','get key(){return"shareData"}','toolbelt-source-seam')
}}
export function bridgeStatement(source){
 const matches=source.match(/\tstatic thirdPartyNativeIWRBridge[^\n]+/g)??[];
 if(matches.length!==1)throw Error('native-bridge-seam-descriptor');return matches[0];
}
export function automaticDescriptor(statement){
 const match=statement.match(/Object\.freeze\((\{.*\})\)/);
 if(!match)throw Error('native-source-seam-descriptor');
 try{return JSON.parse(match[1].replace(',"applyDamage":this.prototype.applyDamage',''))}catch{throw Error('native-source-seam-descriptor-json')}
}
export function observerRegion(source,kind){
 const start=kind==='pf2e'?'const __nativeReceiverStacking=(()=>{\n':'const __toolbeltManualPool=(()=>{';
 const end=kind==='pf2e'?'async function applyDamageFromMessage(':'/* end toolbelt manual pool */';
 const region=sourceRegion(source,start,end,kind==='pf2e'?'native-shared-seam':'toolbelt-source-seam');
 const header=region.match(/ const descriptor=Object\.freeze\([^\n]+\);/g)??[];
 if(header.length!==1)throw Error('native-source-seam-descriptor');
 const normalized=region.replace(header[0],' const descriptor=Object.freeze({});').replace("if(module?.version!=='3.56.5')return;","if(!module)return;");
 return {region,normalized,statement:header[0]};
}
