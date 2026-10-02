import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {PATREON_SOURCE_SEAMS,isPatreonSourceQualified} from '../../scripts/exploration/patreon-source-qualification.mjs';

const hash=value=>createHash('sha256').update(value).digest('hex');
const legacySHA='6b664250f325838850a716c791b3a41df4265899d39377fb8b9b10ad083b10a0';
const legacyDescriptor={version:2,markedCommitOwnership:'private-prepare.v1',providerId:'patreon-v3',providerVersion:'3.2.29',
 baseSourceSHA256:'89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9',
 pf2eSourceSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157'};
const marker='/*pf2e-third-party-automation:patreon-original-seams.v1:';
const regions={
 time:['Hooks.on("updateWorldTime",','r(Mn,"handleFastHealingTime");'],
 manualImmunity:['async function Ma(','r(Ma,"treatWounds");'],
 createItem:['async function p(','r(p,"addItemToActor");']
};
const timeChanges=[
 ['Hooks.on("updateWorldTime",(a,e,t,i)=>{k()&&(y("fastHealingTime")&&Mn(e),Rn())});',
  'Hooks.on("updateWorldTime",(a,e,t,i)=>{if(k()&&__patreonTimeCompletion.owned()){const x=__patreonTimeCompletion.begin(a,e,t,i),work=[];if(!__patreonTimeCompletion.authorize(x))return;try{y("fastHealingTime")&&work.push(Mn(e,x));work.push(Rn(x));}catch(error){work.push(Promise.reject(error));__patreonTimeCompletion.publish(x,work);throw error;}__patreonTimeCompletion.publish(x,work);}});'],
 ['function Rn(){game.actors.map','function Rn(x){return __patreonTimeCompletion.branches(game.actors.map'],
 ['.forEach(async e=>{let t=e.getFlag(u,"decrementPeriod");','.map(async e=>{let t=e.getFlag(u,"decrementPeriod");'],
 ['await e.update({"system.start.value":game.time.worldTime}),e.decrease())})}',
  'await __patreonTimeCompletion.update(x,e,{"system.start.value":game.time.worldTime},"effect-start"),__patreonTimeCompletion.decrease(x,e))}),x)}'],
 ['async function Mn(a){','async function Mn(a,x){'],
 ['.filter(l=>l.test())','.filter(l=>__patreonTimeCompletion.predicate(x,l,l.test()))'],
 ['let c=l.resolveValue(l.value);','let c=__patreonTimeCompletion.resolved(x,l,l.resolveValue(l.value),"eligibility");'],
 ['let s=0,l=game.actors.get(o);if(','let s=0,l=game.actors.get(o);__patreonTimeCompletion.member(x,o,l);if('],
 ['let m=await new h.DamageRoll(`{(${f.resolveValue(f.value)})[healing]}`).evaluate();',
  'let m=__patreonTimeCompletion.roll(x,f,await new h.DamageRoll(`{(${__patreonTimeCompletion.resolved(x,f,f.resolveValue(f.value),"roll")})[healing]}`).evaluate());'],
 ['l.update({"system.attributes.hp.value":c}):l.update({"system.attributes.hp.value":l.system.attributes.hp.max})',
  '__patreonTimeCompletion.update(x,l,{"system.attributes.hp.value":c},"actor-hp"):__patreonTimeCompletion.update(x,l,{"system.attributes.hp.value":l.system.attributes.hp.max},"actor-hp")']
];
const immunityChanges=[
 ['async function Ma(a,e){','async function Ma(a,e){const x=__patreonManualImmunity.bind(a,e);'],
 ['await p(e,t),','await p(e,await __patreonManualImmunity.mark(t,x)),']
];
const itemChanges=[['let i=await a.createEmbeddedDocuments("Item",[e]);',
 'let i=await __patreonManualImmunity.create(a,e,()=>a.createEmbeddedDocuments("Item",[e]));']];

function fail(reason){const error=Error(reason);error.code=reason;throw error}
function once(source,before,after,name){
 if(!before||source.split(before).length!==2)fail('patreon-'+name+'-seam-ambiguous');
 return source.replace(before,after);
}
function region(source,name){
 const [start,end]=regions[name];
 if(source.split(start).length!==2||source.split(end).length!==2)fail('patreon-'+name+'-seam-ambiguous');
 const from=source.indexOf(start),to=source.indexOf(end,from);
 if(to<from)fail('patreon-'+name+'-seam-unavailable');
 return source.slice(from,to+end.length);
}
function qualify(source){
 const found={};
 for(const name of Object.keys(regions)){
  found[name]=region(source,name);
  if(hash(found[name])!==PATREON_SOURCE_SEAMS[name])fail('patreon-'+name+'-seam-changed');
 }
 return found;
}
function change(source,pairs,name,inverse=false){
 for(const [before,after] of inverse?[...pairs].reverse():pairs)source=once(source,inverse?after:before,inverse?before:after,name);
 return source;
}
function readObserver(file,expected){
 const text=fs.readFileSync(new URL('../'+file,import.meta.url),'utf8');
 if(hash(text)!==expected)fail('patreon-observer-source-changed');
 return text;
}
function legacyObservers(){
 return {time:readObserver('patreon-time-completion/completion-observer-v5.js','cf0bba86ed87b71d01e5e04602f03cbabae36581c1197ab3a739de562970da3e'),
  immunity:readObserver('patreon-manual-immunity/observer.js','ac63615e78ea0320df58ebf5c3ac5c900ee2e0d7fc56a9a51afc9b4b534d6f79')};
}
function observer(text,descriptor){
 const declaration=text.match(/^ const descriptor=Object\.freeze\([^\n]+\);$/m)?.[0];
 const json=JSON.stringify(descriptor),seams=JSON.stringify(descriptor.seams);
 return once(text,declaration,' const descriptor=Object.freeze('+json.replace('"seams":'+seams,'"seams":Object.freeze('+seams+')')+');','observer');
}
function observers(descriptor){
 const original=legacyObservers(),time=observer(original.time,descriptor);
 const immunity=once(observer(original.immunity,descriptor),
  'installed&&game.system?.version==="8.5.1"&&patient?.uuid','installed&&patient?.uuid','immunity-version');
 return {time,immunity};
}
function descriptorFor(source,version,pf2eVersion){
 return {version:3,providerId:'patreon-v3',providerVersion:typeof version==='string'&&version?version:'unknown',
  pf2eVersion:typeof pf2eVersion==='string'&&pf2eVersion?pf2eVersion:'unknown',sourceSHA256:hash(source),
  qualification:'patreon-original-seams.v1',seams:{...PATREON_SOURCE_SEAMS},markedCommitOwnership:'private-prepare.v1'};
}
const qualificationMarker=descriptor=>marker+Buffer.from(JSON.stringify(descriptor)).toString('base64')+'*/';
function compose(source,observed,prefix=''){
 const found=qualify(source);
 source=once(source,found.time,prefix+observed.time+change(found.time,timeChanges,'time'),'time');
 source=once(source,found.manualImmunity,observed.immunity+change(found.manualImmunity,immunityChanges,'manualImmunity'),'manualImmunity');
 return once(source,found.createItem,change(found.createItem,itemChanges,'createItem'),'createItem');
}
const patch=(source,descriptor)=>compose(source,observers(descriptor),qualificationMarker(descriptor));
function recoverOriginal(source,observed,prefix=''){
 let original=once(source,prefix+observed.time,'','patched-time-observer');
 original=once(original,observed.immunity,'','patched-immunity-observer');
 for(const [name,pairs] of [['time',timeChanges],['manualImmunity',immunityChanges],['createItem',itemChanges]]){
  const found=region(original,name);
  original=once(original,found,change(found,pairs,name,true),name);
 }
 return original;
}
function recover(source){
 if(source.split(marker).length!==2)fail('patreon-partial-patch');
 const from=source.indexOf(marker),to=source.indexOf('*/',from),encoded=source.slice(from+marker.length,to);
 if(to<0||!(/^[A-Za-z0-9+/]+={0,2}$/).test(encoded))fail('patreon-patched-descriptor-invalid');
 let descriptor;
 try{descriptor=JSON.parse(Buffer.from(encoded,'base64').toString('utf8'))}catch{fail('patreon-patched-descriptor-invalid')}
 if(!isPatreonSourceQualified(descriptor))fail('patreon-patched-descriptor-invalid');
 const original=recoverOriginal(source,observers(descriptor),qualificationMarker(descriptor));
 // sourceSHA256 records the original at patch creation. Region qualification,
 // rather than that historical whole-file hash, governs later startups.
 if(patch(original,descriptor)!==source)fail('patreon-patched-source-changed');
 return descriptor;
}
function recoverLegacy(source){
 const observed=legacyObservers(),original=recoverOriginal(source,observed);
 if(compose(original,observed)!==source)fail('patreon-patched-source-changed');
 return original;
}

/** Build both observations together. The caller owns backup and atomic replacement. */
export function buildPatreonSource({source,version,pf2eVersion}){
 if(typeof source!=='string'&&!Buffer.isBuffer(source)&&!(source instanceof Uint8Array))fail('patreon-source-unavailable');
 const buffer=Buffer.from(source),text=buffer.toString('utf8');
 if(!Buffer.from(text).equals(buffer))fail('patreon-source-encoding');
 if(hash(buffer)===legacySHA&&version==='3.2.29'&&pf2eVersion==='8.5.1')return {status:'unchanged',buffer,descriptor:{...legacyDescriptor}};
 if(text.includes(marker)){
  return {status:'unchanged',buffer,descriptor:recover(text)};
 }
 if(text.includes('__patreonTimeCompletion')||text.includes('__patreonManualImmunity')){
  const original=recoverLegacy(text),descriptor=descriptorFor(original,version,pf2eVersion);
  return {status:'patch',buffer:Buffer.from(patch(original,descriptor)),descriptor};
 }
 const descriptor=descriptorFor(buffer,version,pf2eVersion);
 return {status:'patch',buffer:Buffer.from(patch(text,descriptor)),descriptor};
}
