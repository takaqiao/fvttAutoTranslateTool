import {createShapeBinding} from './source-shape-match.mjs';

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
// The Toolbelt shared-pool seam, verbatim from the 3.59.0 bundle. A build is
// compared with it by rename-equivalence, so the minified names below only
// have to agree with each other. In the patched forms x (the pool binding) is
// the one name the regions do not already use; it takes a name the build leaves
// free. s (the sender id) and p follow the renaming like the rest.
const forwardHead='async#h(e,n,i,r,o){let s=i.flags?.[A.id]?.[this.key],c=s?s["==data"]??s.data:void 0,l=c?.master;if(l!==void 0){let f=this.getMasterInMemory(e);f&&f.id!==l&&(r.previousShareMasterId=f.id)}let d=l??this.getMasterId(e),m=d?game.actors.get(d):void 0;if(this.isValidMaster(m)){let f=[];(c?.health??this.#v(e,"health"))&&f.push("system.attributes.hp.value","system.attributes.hp.sp.value","system.attributes.hp.temp"),(c?.heroPoints??this.#v(e,"heroPoints"))&&f.push("system.resources.heroPoints.value"),(c?.spellcasting??this.#v(e,"spellcasting"))&&f.push("system.resources.focus.value");let p=u.pipe(f,u.map(g=>{let h=foundry.utils.getProperty(i,g);if(h!==void 0)return foundry.utils.deleteProperty(i,g),[g,h]}),u.filter(u.isDefined));if(p.length){let g=u.fromEntries(p);';
const forwardTail='}}return n(i,r,o)}';
const forwardCall='m.isOwner?m.update(g):this.#t.emit({master:m,...g})';
const forwardCallPatched='let x=__toolbeltManualPool.forward(e,m,g,r,()=>m.update(g),p=>this.#t.emit(p));if(x)await x';
export const TOOL_SHAPE=Object.freeze({
 socket:'function _e(t,e){let n=!1,i=a(async(o,s)=>{if(o.__type__!==t||!game.user.isActiveGM)return;let c=await AA(o);e(c,s)},"onSocket"),r=a(o=>{if(!game.users.activeGM){ui.notifications.error(ne.shared("emiting.noGm"));return}ea();let s=UA(o);s.__type__=t,Xo(s)},"emit");return{get enabled(){return n},async call(o){if(u.isPlainObject(o)){if(game.user.isActiveGM)return e(o,game.userId);r(o)}},emit:r,activate(){this.enabled||!Ne()||(n=!0,Gy(i))},disable(){this.enabled&&(n=!1,Ky(i))},toggle(o){o??=!this.enabled,o?this.activate():this.disable()}}}a(_e,"createEmitable")',
 transport:'function Gy(t){game.socket.on(`module.${A.id}`,t)}a(Gy,"socketOn");function Ky(t){game.socket.off(`module.${A.id}`,t)}a(Ky,"socketOff");function Xo(t){game.socket.emit(`module.${A.id}`,t)}a(Xo,"socketEmit");',
 conversion:'async function AA(t){let e=t.__converter__,n=t.__source__;delete t.__converter__,delete t.__source__,delete t.__type__,await Promise.all(u.entries(t).map(async()=>{}));let i=n==="array"?[]:{};return await Promise.all(u.entries(t).map(async([r,o])=>{i[r]=await MA(o,e[r])})),i}a(AA,"convertToCallOptions");async function MA(t,e){switch(e){case"document":return fromUuid(t);case"target":return RA(t);case"token":return(await fromUuid(t))?.object;default:return t}}a(MA,"convertToCallOption");async function RA(t){let e=await fromUuid(t.actor);if(e instanceof Actor)return{actor:e,token:t.token&&await fromUuid(t.token)||void 0}}a(RA,"convertTargetFromPacket");function zA({actor:t,token:e}){return{actor:t.uuid,token:e?.uuid}}a(zA,"convertTargetToPacket");function UA(t){let e={},n=u.mapValues(t,(i,r)=>i instanceof foundry.abstract.Document?(e[r]="document",i.uuid):Ko(i)?(e[r]="token",i.document.uuid):Jx(i)?(e[r]="target",zA(i)):i);return n.__converter__=e,n.__source__=u.isArray(t)?"array":"object",n}a(UA,"convertToEmitOptions");',
 binding:'static{a(this,"ShareDataTool")}#t=_e(this.path("master"),this.#g.bind(this));#e=_e(this.path("slave-turn"),this.#f.bind(this));#n=_e(this.path("slave-initiative"),this.#c.bind(this));get key(){return"shareData"}',
 validity:'isValidActor(e){return!!e&&e instanceof Actor&&!e.pack&&game.actors.has(e.id)&&e.isOfType("character","npc")}isValidMaster(e,n){if(!this.isValidActor(e)||n&&e.id===n||this.getMasterInMemory(e))return!1;let i=this.getMasterId(e);return!i||!game.actors.get(i)}',
 receive:'#g({master:e,...n}){this.isValidMaster(e)&&e.update(n)}',
 receivePatched:'#g({master:e,__explorationManualPool:x,...n},s){return __toolbeltManualPool.receive(e,n,x,s,()=>this.isValidMaster(e)&&e.update(n))}',
 forward:forwardHead+forwardCall+forwardTail,
 forwardPatched:forwardHead+forwardCallPatched+forwardTail
});
// The bundle functions the regions call without defining them, each with the name esbuild retains for it.
const TOOL_HELPERS={ea:'displayEmiting',Ne:'userIsGM',Ko:'isTokenObject',Jx:'isValidTargetDocuments'};
const NAME='[A-Za-z_$][\\w$]*',TOOL_SEAM='toolbelt-source-seam',CLASS_MEMBERS=new Set(['binding','validity','receive','forward']);
/** Qualify the Toolbelt shared-pool seam of `source`. Returns whether the patch
 * is `installed`, and the source without it (`original`) and with it (`patched`).
 * Regions are located from the function names esbuild retains, string literals
 * and member names; a minified name is only ever read back from a match.
 * Throws `toolbelt-source-seam-*` when the seam is not the reference shape. */
export function toolSeams(source){
 const single=(pattern,text=source,reason='ambiguous')=>{const found=[...text.matchAll(pattern)];if(found.length!==1)throw Error(`${TOOL_SEAM}-${reason}`);return found[0]};
 // The one place where `text` is declared rather than reached through a member access.
 const declared=(text,from=0,to=source.length)=>{
  const found=[];for(let i=source.indexOf(text,from);i>=0&&i<to;i=source.indexOf(text,i+1))if(source[i-1]!=='.')found.push(i);
  if(found.length!==1)throw Error(TOOL_SEAM+'-ambiguous');return found[0];
 };
 const registered=(label,reason)=>single(new RegExp(`${NAME}\\((${NAME}),"${label}"\\)`,'g'),source,reason)[1];
 // esbuild opens every named class with this registration, so the next one ends the class.
 const head=single(new RegExp(`static\\{${NAME}\\(this,"ShareDataTool"\\)\\}`,'g')),following=new RegExp(`static\\{${NAME}\\(this,"`,'g');
 following.lastIndex=head.index+head[0].length;
 const from=head.index,to=following.exec(source)?.index??source.length,body=source.slice(from,to);
 const emitter=single(new RegExp(`${NAME}\\(this\\.path\\("master"\\),this\\.(#${NAME})\\.bind\\(this\\)\\)`,'g'),body);
 const wrapper=single(new RegExp(`${NAME}\\("_preUpdate",this\\.(#${NAME})\\)`,'g'),body,'wrapper');
 const starts={
  socket:declared(`function ${registered('createEmitable')}(`),transport:declared(`function ${registered('socketOn')}(`),conversion:declared(`async function ${registered('convertToCallOptions')}(`),
  binding:from,validity:declared('isValidActor(',from,to),receive:declared(emitter[1]+'(',from,to),forward:declared(`async${wrapper[1]}(`,from,to)
 };
 const shapes=createShapeBinding({fixed:['ui']});
 // Match one region against its accepted forms; a refusal reports the form that got furthest.
 function region(key,...references){
  // A region opens a statement or class member, so no modifier can precede it.
  if(!'{};'.includes(source[starts[key]-1]))throw Error(`${TOOL_SEAM}-${key}`);
  let closest;
  for(const [variant,reference]of references.entries()){
   try{return {variant,end:shapes.match(reference,source,{at:starts[key],within:CLASS_MEMBERS.has(key)?'class':undefined})}}
   catch(error){if(error.message!=='source-shape-mismatch')throw error;if(!(closest?.index>=error.index))closest=error}
  }
  throw Error(`${TOOL_SEAM}-${key} (token ${closest.index}: expected ${JSON.stringify(closest.expected)}, found ${JSON.stringify(closest.found)})`);
 }
 for(const key of ['socket','transport','conversion','binding','validity'])region(key,TOOL_SHAPE[key]);
 // A renamed helper is still the same helper only if esbuild registered it under the same name.
 for(const [name,label]of Object.entries(TOOL_HELPERS))if(shapes.render(name)!==registered(label,'helper'))throw Error(TOOL_SEAM+'-helper');
 const spans={receive:region('receive',TOOL_SHAPE.receive,TOOL_SHAPE.receivePatched),forward:region('forward',TOOL_SHAPE.forward,TOOL_SHAPE.forwardPatched)};
 if(spans.receive.variant!==spans.forward.variant)throw Error(TOOL_SEAM+'-partial');
 const installed=spans.receive.variant===1;
 function rewritten(form){
  let text=source;
  for(const key of Object.keys(spans).sort((a,b)=>starts[b]-starts[a]))text=text.slice(0,starts[key])+shapes.render(TOOL_SHAPE[key+form],{within:'class'})+text.slice(spans[key].end);
  return text;
 }
 return installed?{installed,original:rewritten(''),patched:source}:{installed,original:source,patched:rewritten('Patched')};
}
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
