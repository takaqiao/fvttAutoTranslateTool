const __patreonManualImmunity=(()=>{
 const namespace="pf2e-third-party-automation";
 const descriptor=Object.freeze({version:1,providerId:"patreon-v3",providerVersion:"3.2.29",baseSourceSHA256:"89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9",pf2eSourceSHA256:"d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157"});
 const observers=new Set(),writes=new Map();let installed=false;
 const copy=value=>JSON.parse(JSON.stringify(value));
 const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 const author=message=>message.author?.id??message.author??message.user?.id??message.user;
 function source(binding,patient,pendingTarget=false){
  try{
  const message=game.messages.get(binding?.messageId),context=message?.flags?.pf2e?.context,meta=message?.flags?.[namespace]?.explorationManualNative;
  const healer=message?.actor,user=game.users.get(binding?.sourceUserId);
  const snapshot=binding?.targetSnapshot,token=snapshot&&snapshot.actorUUID===patient?.uuid&&typeof snapshot.tokenUUID==="string"&&snapshot.tokenUUID.startsWith("Scene.")&&snapshot.tokenUUID.includes(".Token.");
  const target=snapshot?token&&(snapshot.type==="patreon-single-target"?context?.target==null:snapshot.type==="check-context"&&context?.target?.actor===patient.uuid&&(!context.target.token||context.target.token===snapshot.tokenUUID)):context?.target?.actor===patient?.uuid;
  return installed&&game.system?.version==="8.5.1"&&patient?.uuid===binding?.patientUUID&&message?.isCheckRoll===true&&!message.isReroll
   &&message.rolls?.[0]?._evaluated===true&&context?.type==="skill-check"&&Array.isArray(context.options)&&context.options.includes("action:treat-wounds")&&context.options.includes(binding.tag)
   &&context.origin?.actor===binding.actorUUID&&target&&message.speaker?.actor===healer?.id&&healer?.uuid===binding.actorUUID
   &&author(message)===binding.sourceUserId&&user&&healer.testUserPermission?.(user,"OWNER")===true
   &&meta?.useId===binding.useId&&meta.tag===binding.tag&&(meta.patientUUID===patient.uuid||pendingTarget&&meta.patientUUID==null)&&meta.startedAt===binding.startedAt
   &&(!binding.recordingSessionId||binding.recordingSessionId===meta.recordingSessionId)
   &&meta.riskySurgery===false&&!context.options.includes("risky-surgery")&&Array.isArray(message.flags.pf2e.modifiers)&&!message.flags.pf2e.modifiers.some(m=>m.slug==="risky-surgery"&&m.enabled)
   &&Number.isFinite(binding.startedAt)&&binding.startedAt===game.time.worldTime?message:null;
  }catch{return null}
 }
 function bind(input,patient){
  try{
  const message=game.messages.get(input?.messageId),meta=message?.flags?.[namespace]?.explorationManualNative;
  if(!meta||input.mainActor!==message.actor||input.messageFlags!==message.flags)return null;
  const binding={invocationId:crypto.randomUUID(),messageId:message.id,useId:meta.useId,tag:meta.tag,actorUUID:input.mainActor.uuid,patientUUID:patient?.uuid,sourceUserId:author(message),startedAt:meta.startedAt};
  if(typeof meta.recordingSessionId==="string"&&meta.recordingSessionId)binding.recordingSessionId=meta.recordingSessionId;
  const targets=Array.from(input.tokenTargets??[]),token=targets.length===1?targets[0]:null,tokenUUID=token?.document?.uuid;
  if(token?.actor!==patient||typeof tokenUUID!=="string")return null;
  binding.targetSnapshot={type:message.flags.pf2e.context.target==null?"patreon-single-target":"check-context",actorUUID:patient.uuid,tokenUUID};
  return source(binding,patient,true)&&author(message)===game.user.id?binding:null;
  }catch{return null}
 }
 async function mark(data,binding){
  if(!binding)return data;
  try{
   const patient=await fromUuid(binding.patientUUID),message=source(binding,patient,true);
   if(!message?.update||message.flags[namespace].explorationManualNative.patreonImmunity)return data;
   await message.update({[`flags.${namespace}.explorationManualNative.patientUUID`]:binding.patientUUID,[`flags.${namespace}.explorationManualNative.patreonImmunity`]:copy(binding)});
   if(!source(binding,patient)||!same(message.flags[namespace].explorationManualNative.patreonImmunity,binding))return data;
   data.flags={...data.flags,[namespace]:{...data.flags?.[namespace],explorationManualPatreonImmunity:copy(binding)}};
  }catch{/* Observation failure leaves the original treatment unchanged. */}
  return data;
 }
 function create(patient,data,native){
  let binding,message,creator,invocation;
  try{
   const marked=data.flags?.[namespace]?.explorationManualPatreonImmunity;binding=marked&&copy(marked);message=binding&&source(binding,patient);
   if(!message||!same(message.flags[namespace].explorationManualNative.patreonImmunity,binding))binding=null;
   creator=game.user;if(binding&&!creator?.isGM&&patient.testUserPermission?.(creator,"OWNER")!==true)binding=null;
   if(binding){
    data.flags[namespace].explorationManualPatreonImmunity={...binding,creatorId:creator.id};
    const previous=writes.get(binding.invocationId);if(previous)previous.duplicate=true;
    invocation={duplicate:!!previous};writes.set(binding.invocationId,invocation);
   }
  }catch{binding=null}
  if(!binding)return native();
  let result;try{result=native()}catch(error){result=Promise.reject(error)}
  const terminalPromise=(async()=>{
   if(!result||typeof result.then!=="function")throw Error("manual-immunity-native-promise-unavailable");
   const saved=await result;
   if(invocation.duplicate||writes.get(binding.invocationId)!==invocation||!source(binding,patient))throw Error("manual-immunity-source-changed");
   const item=saved?.length===1?saved[0]:null,duration=item?.system?.duration,start=item?.system?.start?.value;
   const seconds=duration?.value*({minutes:60,hours:3600}[duration?.unit]??NaN);
   if(!item?.uuid||patient.items?.get(item.id)!==item||item.actor!==patient||!same(item.flags?.[namespace]?.explorationManualPatreonImmunity,{...binding,creatorId:creator.id})
    ||start!==binding.startedAt||!Number.isFinite(seconds)||seconds<=0)throw Error("manual-immunity-native-item-unproven");
   return Object.freeze({descriptor,binding:Object.freeze(copy(binding)),creatorId:creator.id,itemUUID:item.uuid,start,duration:Object.freeze(copy(duration)),expiresAt:start+seconds});
  })();
  terminalPromise.catch(()=>{});
  const event=Object.freeze({descriptor,binding:Object.freeze(copy(binding)),terminalPromise});
  for(const observer of [...observers])try{observer(event)}catch{}
  return result;
 }
 function install(){
  const module=game.modules.get("patreon-v3"),api=module?.api===undefined?{}:module?.api;
  if(!module||!api||typeof api!=="object"||Array.isArray(api)||"explorationManualImmunity"in api)return;
  const observation=Object.freeze({descriptor,subscribe(observer){if(typeof observer!=="function")throw TypeError("manual-immunity-observer-required");observers.add(observer);return()=>observers.delete(observer)}});
  try{Object.defineProperty(api,"explorationManualImmunity",{value:observation,enumerable:true});if(module.api===undefined)module.api=api;installed=module.api.explorationManualImmunity===observation}catch{}
 }
 return {bind,mark,create,install};
})();
Hooks.once("init",()=>__patreonManualImmunity.install());
