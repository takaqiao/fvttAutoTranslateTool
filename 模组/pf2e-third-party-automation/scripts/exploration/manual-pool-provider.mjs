import {SHARED_MANUAL_SHAPE as shape,TOOL_SOCKET_CONTRACTS,NATIVE_DAMAGE_SHAPES,isAutomaticBatchDescriptor,isAutomaticToolDescriptor} from '../native-iwr-profiles.mjs';
import {sourceText,observerRegion,automaticDescriptor,batchRegion,flatRegion,stackingRegion,toolRegions,nativeMethod,TOOL_RECEIVE_PATCHED,TOOL_RECEIVE,TOOL_FORWARD_PATCHED,TOOL_FORWARD} from '../native-source-shapes.mjs';

const batches=new WeakMap(),providers=new WeakMap();
const LEGACY_TOOL='2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f';
const legacyTool=d=>d?.version===1&&d.hpBaselineGuardVersion===1&&d.sourceSHA256===LEGACY_TOOL;
function data(value){
 if(!value||!Object.isFrozen(value))throw Error('manual-pool-source-seam-descriptor');
 const out={};for(const [key,p]of Object.entries(Object.getOwnPropertyDescriptors(value))){if(!Object.hasOwn(p,'value'))throw Error('manual-pool-source-seam-descriptor');out[key]=p.value}return out;
}
const same=(a,b)=>Object.keys(a).length===Object.keys(b).length&&Object.keys(a).every(k=>a[k]===b[k]);
const retained=(object,key,value)=>{const p=Object.getOwnPropertyDescriptor(object??{},key);return p&&p.value===value&&p.writable===false&&p.configurable===false};
async function browserHash(source){const bytes=await crypto.subtle.digest('SHA-256',typeof source==='string'?new TextEncoder().encode(source):source);return Array.from(new Uint8Array(bytes),b=>b.toString(16).padStart(2,'0')).join('')}
function current(s){return s.game.system===s.system&&s.system.id==='pf2e'&&s.system.version===s.systemVersion&&s.game.pf2e===s.api&&s.api.thirdPartyManualPoolBatch===s.batch&&s.batch.descriptor===s.batchDescriptor&&s.game.modules.get('pf2e-toolbelt')===s.module&&s.module.active===true&&s.module.version===s.toolVersion&&s.module.api===s.toolAPI&&s.toolAPI.explorationManualPool===s.provider&&s.provider.descriptor===s.toolDescriptor}
export function isQualifiedManualPoolBatch(descriptor){const s=batches.get(descriptor);return !!s&&current(s)}
export function isManualPoolProvider(provider){return legacyTool(provider?.descriptor)||!!providers.get(provider)&&current(providers.get(provider))}

/** Attest actual served seams and current installed read-only APIs before
 * exploration subscribes. The private issuance is never a caller JSON DTO. */
export async function verifyManualPoolProviders({game,pf2eSource,toolbeltSource,hash=browserHash}={}){
 const unavailable=reason=>Object.freeze({ready:false,reason});
 try{
  const system=game?.system,api=game?.pf2e,batch=api?.thirdPartyManualPoolBatch,module=game?.modules?.get('pf2e-toolbelt'),toolAPI=module?.api,provider=toolAPI?.explorationManualPool;
  if(system?.id!=='pf2e'||module?.active!==true||!batch||!provider)return unavailable('manual-pool-provider-unavailable');
  const batchDescriptor=batch.descriptor,toolDescriptor=provider.descriptor,a=data(batchDescriptor),b=data(toolDescriptor);
  batches.delete(batchDescriptor);providers.delete(provider);
  const snapshot={game,system,systemVersion:system.version,api,batch,batchDescriptor,module,toolVersion:module.version,toolAPI,provider,toolDescriptor};
  if(!Object.isFrozen(batch)||!Object.isFrozen(provider)||!retained(api,'thirdPartyManualPoolBatch',batch)||!retained(toolAPI,'explorationManualPool',provider)||typeof batch.currentCall!=='function'||typeof batch.subscribe!=='function'||typeof provider.subscribe!=='function')return unavailable('manual-pool-api-unqualified');
  const pf=sourceText(pf2eSource),tb=sourceText(toolbeltSource),po=observerRegion(pf,'pf2e'),to=observerRegion(tb,'toolbelt');
  const digest=async value=>{const result=await hash(value);if(!current(snapshot)||!/^[a-f0-9]{64}$/i.test(result))throw Error('manual-pool-source-changed');return result.toLowerCase()};
  if(a.version===2){if(!isAutomaticBatchDescriptor(a)||!same(a,automaticDescriptor(po.statement)))return unavailable('manual-pool-native-descriptor-mismatch')}
  else if(a.version!==1||a.providerVersion!=='8.5.1'||a.baseSourceSHA256!=='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157')return unavailable('manual-pool-native-descriptor-mismatch');
  if(b.version===2){if(!isAutomaticToolDescriptor(b)||!same(b,automaticDescriptor(to.statement)))return unavailable('manual-pool-tool-descriptor-mismatch')}
  else if(!legacyTool(b))return unavailable('manual-pool-tool-descriptor-mismatch');
  if(await digest(po.normalized)!==shape.nativeObserver||await digest(to.normalized)!==shape.toolObserver)return unavailable('manual-pool-observer-seam-mismatch');
  const registration=' __nativeManualPoolStaticReceiver.register(construct,this,r,this.actor.synthetics.modifiers[r]);';
  if(pf.split(registration).length!==2||pf.split('jm.onInit();__nativeManualPoolBatch.install();').length!==2||tb.split(TOOL_RECEIVE_PATCHED).length!==2||tb.split(TOOL_FORWARD_PATCHED).length!==2)return unavailable('manual-pool-source-seam-partial');
  if(await digest(batchRegion(pf))!==shape.batchPatched)return unavailable('manual-pool-native-batch-seam-mismatch');
  const native=pf.replace(po.region,''),flat=flatRegion(native).replace(registration,'');
  if(await digest(flat)!==shape.flatOriginal||await digest(stackingRegion(native))!==shape.stacking)return unavailable('manual-pool-native-receiver-seam-mismatch');
  const methodSHA=await digest(nativeMethod(native));if(!NATIVE_DAMAGE_SHAPES.some(p=>p.applyDamageSHA256===methodSHA))return unavailable('manual-pool-native-bridge-seam-mismatch');
  const tool=tb.replace(to.region+'/* end toolbelt manual pool */\n','').replace(TOOL_RECEIVE_PATCHED,TOOL_RECEIVE).replace(TOOL_FORWARD_PATCHED,TOOL_FORWARD);
  const regions=toolRegions(tool),socketSHA=await digest(regions.socket),socketContract=TOOL_SOCKET_CONTRACTS.find(contract=>contract.socket===socketSHA);
  if(!socketContract)return unavailable('manual-pool-tool-seam-mismatch');
  for(const [key,value]of Object.entries(regions))if(await digest(value)!==(socketContract[key]??shape['tool'+key[0].toUpperCase()+key.slice(1)]))return unavailable('manual-pool-tool-seam-mismatch');
  if(!tool.includes('r("_preUpdate",this.#h)')||!current(snapshot))return unavailable('manual-pool-source-changed');
  const pf2eSHA256=await digest(pf2eSource),toolbeltSHA256=await digest(toolbeltSource);
  if(a.version===2)batches.set(batchDescriptor,snapshot);if(b.version===2)providers.set(provider,snapshot);
  return Object.freeze({ready:true,reason:'verified-source-seams',pf2eSHA256,toolbeltSHA256});
 }catch(error){return unavailable(error.message??'manual-pool-source-unavailable')}
}
