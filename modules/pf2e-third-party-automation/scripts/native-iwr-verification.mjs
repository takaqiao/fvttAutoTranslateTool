import {NATIVE_IWR_PROFILES} from './native-iwr-profiles.mjs';

const issued=new WeakSet(),snapshots=new WeakMap();
const fields=['version','sourceSHA256','protocol','applyDamage'];
const defaultBridge=()=>globalThis.CONFIG?.Actor?.documentClass?.thirdPartyNativeIWRBridge;
const diagnostic=(reason,profile,details={})=>Object.freeze({ready:false,reason,systemVersion:profile?.systemVersion??null,...details});
async function browserHash(source){
 if(!globalThis.crypto?.subtle)throw Error('Browser SHA-256 is unavailable.');
 const input=typeof source==='string'?new TextEncoder().encode(source):source;
 const bytes=await globalThis.crypto.subtle.digest('SHA-256',input);
 return Array.from(new Uint8Array(bytes),b=>b.toString(16).padStart(2,'0')).join('');
}
async function digest(hash,text){
 const result=await hash(text);
 if(typeof result!=='string'||!/^[a-f0-9]{64}$/i.test(result))throw Error('Hash provider did not return a SHA-256 hexadecimal digest.');
 return result.toLowerCase();
}
function descriptor(bridge){
 if(!bridge||typeof bridge!=='object'||!Object.isFrozen(bridge))return null;
 const descriptors=Object.getOwnPropertyDescriptors(bridge),values={};
 for(const key of fields){
  const property=descriptors[key];
  if(!property||!Object.hasOwn(property,'value'))return null;
  values[key]=property.value;
 }
 return values;
}
const sameDescriptor=(a,b)=>a===null||b===null?a===b:fields.every(key=>a[key]===b[key]);
const matchesProfile=(d,p)=>d&&d.version===p.systemVersion&&d.sourceSHA256===p.originalSHA256&&d.protocol===p.protocol&&typeof d.applyDamage==='function';
function currentSystem(snapshot){
 return snapshot.game?.system===snapshot.system&&snapshot.system?.id==='pf2e'&&snapshot.system.version===snapshot.profile.systemVersion;
}

/** Verify the actual HTTP response bytes and retained original native function.
 * Pass new Uint8Array(await response.arrayBuffer()) in the browser. Decoding the
 * body first would discard a UTF-8 BOM and would not attest exact served bytes.
 * No system source is evaluated, no runtime wrapper is installed, and failures
 * are local diagnostics rather than exceptions from the module ready hook. */
export async function verifyNativeIWRBridge(options={}){
 let profile;
 try{
  const {game,source,getNativeBridge=defaultBridge,hash=browserHash}=options??{};
  const system=game?.system,version=system?.version;
  profile=system?.id==='pf2e'&&typeof version==='string'&&Object.hasOwn(NATIVE_IWR_PROFILES,version)?NATIVE_IWR_PROFILES[version]:null;
  if(!profile)return diagnostic('unsupported-system-profile',null);
  if(typeof source!=='string'&&!(source instanceof Uint8Array))return diagnostic('system-source-unavailable',profile);
  const sourceSnapshot=typeof source==='string'?source:new Uint8Array(source);
  let bridge;
  try{bridge=getNativeBridge();}catch(error){return diagnostic('native-bridge-unavailable',profile,{detail:String(error.message??error)});}
  const retained=descriptor(bridge),snapshot={game,system,profile,bridge,retained};
  const changed=()=>{
   if(!currentSystem(snapshot))return 'system-profile-changed';
   let current;
   try{current=getNativeBridge();}catch{return 'native-bridge-changed';}
   return current!==bridge||!sameDescriptor(descriptor(current),retained)?'native-bridge-changed':null;
  };
  let sourceSHA256;
  try{sourceSHA256=await digest(hash,sourceSnapshot);}catch(error){return diagnostic('hash-failed',profile,{detail:String(error.message??error)});}
  let reason=changed();if(reason)return diagnostic(reason,profile);
  if(sourceSHA256===profile.originalSHA256)return diagnostic('bridge-not-installed',profile,{actualSHA256:sourceSHA256});
  if(sourceSHA256!==profile.patchedSHA256)return diagnostic('unknown-system-source',profile,{actualSHA256:sourceSHA256,expectedSHA256:profile.patchedSHA256});
  if(!matchesProfile(retained,profile))return diagnostic('native-bridge-invalid',profile);
  const methodSource=Function.prototype.toString.call(retained.applyDamage);
  let applyDamageSHA256;
  try{applyDamageSHA256=await digest(hash,methodSource);}catch(error){return diagnostic('hash-failed',profile,{detail:String(error.message??error)});}
  reason=changed();if(reason)return diagnostic(reason,profile);
  if(applyDamageSHA256!==profile.applyDamageSHA256)return diagnostic('native-method-sha-mismatch',profile,{actualSHA256:applyDamageSHA256,expectedSHA256:profile.applyDamageSHA256});
  const proof=Object.freeze({ready:true,reason:'verified',bridge,systemVersion:profile.systemVersion,patchedSHA256:profile.patchedSHA256,applyDamageSHA256});
  snapshots.set(proof,Object.freeze(snapshot));issued.add(proof);
  return proof;
 }catch(error){return diagnostic('verification-error',profile,{detail:String(error.message??error)});}
}

/** Synchronous use-site check. Cloned diagnostics/JSON/caller-made proof objects
 * cannot acquire the private issuance capability. No hashing on the damage path. */
export function isVerifiedNativeIWRBridge(proof,game,bridge){
 try{
  if(!issued.has(proof))return false;
  const snapshot=snapshots.get(proof);
  return !!(snapshot&&snapshot.game===game&&snapshot.bridge===bridge&&proof.bridge===bridge&&Object.isFrozen(proof)&&currentSystem(snapshot)&&sameDescriptor(descriptor(bridge),snapshot.retained)&&matchesProfile(snapshot.retained,snapshot.profile));
 }catch{return false;}
}
