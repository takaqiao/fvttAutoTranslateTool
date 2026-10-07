import {MODULE_ID,sameCheckpoint} from './schema.mjs';
import {normalizeManualActivity} from './manual-time.mjs';
import {canonicalJSON} from './revision-codec.mjs';
import {immunityExpiry} from './capabilities.mjs';
export const WORKBENCH_SOURCE_SHA='b3bac907654522fda62b80da182fc253f7147493a465a82081219fb2a0f1308f';
export const IMMUNITY_SOURCES=Object.freeze({
 'XDY DO_NOT_IMPORT TW Immunity CD':{kind:'treatment',sha:'aa3aa174524021b06e38f9128fd29196ac5f5da863bd818068a9b2fa0e699d20'},
 'XDY DO_NOT_IMPORT BM Immunity CD':{kind:'battle-medicine',sha:'44febb1cb29c920da5d21f6d959cf44e35ffadca28a32c84fdd22bcd0f548b33'},
 'TW Immunity CD':{forward:true,sha:'1c6cfd6c83acffd1007c9ab20397a7d6ac49336c61d7938314a78c49f2315555'},
 'BM Immunity CD':{forward:true,sha:'19720ed1fa733f729d8a4bcef56204429f012bddd07d30256fea3b6476a9a0f3'}
 ,'Treat Wounds and Battle Medicine':{forward:true,sha:'3da2e9e0fb2227fa21d2cc2798e66cb63b6305891a70a683c578320c2da75797'}
});
export function observeWorkbenchCommand(command,verifiedSHA){
 if(verifiedSHA!==WORKBENCH_SOURCE_SHA)throw Error('unverified-workbench-source');
 const entry=/const rollTreatWounds = async \(\{[\s\S]*?\}\) => \{/;
 if(!entry.test(command))throw Error('unknown-workbench-callback');
 return command.replace(entry,match=>`const explorationBases = {ChatMessage, DamageRoll, CheckRoll};\n${match}\n const explorationScope = await explorationManualTarget({target, bmtw, skillUsed, isRiskySurgery, healer:token.actor}, explorationBases);\n const {ChatMessage, DamageRoll, CheckRoll} = explorationScope;\n skillUsed = explorationScope.skill;`);
}
const facade=(object,overrides)=>new Proxy(Object.create(Object.getPrototypeOf(object)),{get:(_,key)=>key in overrides?overrides[key]:typeof object[key]==='function'?object[key].bind(object):object[key]});
/** Patch only the known compendium document. Lexical target scopes survive delayed callbacks. */
export async function registerWorkbenchObservation({game,Hooks,recorder,ChatMessage=globalThis.ChatMessage,CONFIG=globalThis.CONFIG}){
 const pack=game.packs.get('xdy-pf2e-workbench.asymonous-benefactor-macros-internal');
 if(!pack)return ()=>{};
 const decorated=new WeakSet(),restores=[];
 async function instrument(macro){
 if(!macro||!['XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine',...Object.keys(IMMUNITY_SOURCES)].includes(macro.name)||decorated.has(macro))return macro;
 const immunity=IMMUNITY_SOURCES[macro.name],expectedSHA=immunity?.sha??WORKBENCH_SOURCE_SHA;
 const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(macro.command.replace(/\r\n/g,'\n'))))).map(x=>x.toString(16).padStart(2,'0')).join('');
 // Source snapshots may retain CRLF; accept only the exact pinned byte form too.
 const byteHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(macro.command)))).map(x=>x.toString(16).padStart(2,'0')).join('');
 if(hash!==expectedSHA&&byteHash!==expectedSHA){recorder.diagnostic?.('workbench-source-unverified');return macro}
 if(immunity){const sourceCommand=macro.command,original=macro.execute;let command;
  if(immunity.forward){if(!sourceCommand.includes('macro.execute();'))return macro;command=sourceCommand.replace('macro.execute();','await explorationExecuteImmunity(macro);')}
  else{const entry=/const message = game\.messages\.contents\.reverse\(\)\.find\([^\n]+;/;if(!entry.test(sourceCommand)||!sourceCommand.includes('await token.actor.createEmbeddedDocuments'))return macro;command=sourceCommand.replace(entry,line=>`${line}\n const explorationImmunity = explorationManualImmunity(message, token);`).replace('await token.actor.createEmbeddedDocuments','await explorationImmunity.createEmbeddedDocuments')}
  const execute=async function(input={}){if(this!==macro||macro.command!==sourceCommand)return original.call(this,input);const clone=macro.clone({command},{keepId:true});return original.call(clone,{...input,explorationExecuteImmunity:async target=>{await instrument(target);return target.execute()},explorationManualImmunity:(message,token)=>recorder.bindImmunity({message,token,kind:immunity.kind,sourceSHA:expectedSHA})})};
  macro.execute=execute;decorated.add(macro);restores.push(()=>{if(macro.execute===execute)macro.execute=original});return macro;
 }
 const command=observeWorkbenchCommand(macro.command,WORKBENCH_SOURCE_SHA),original=macro.execute;
 const adapterSHA=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(command)))).map(x=>x.toString(16).padStart(2,'0')).join(''),sourceCommand=macro.command;
 const execute=async function(input={}){
  if(this!==macro||macro.command!==sourceCommand)return original.call(this,input);
  const useId=crypto.randomUUID(),clone=macro.clone({command},{keepId:true});
  const explorationManualTarget=async({target,bmtw,skillUsed,isRiskySurgery,healer},bases)=>{
   const kind=bmtw==='Treat Wounds'?'treatment':'battle-medicine',ward=kind==='treatment'&&healer.items.some(i=>i.slug==='ward-medic');
   const reservation=await recorder.beginCheckpointSource?.({sourceType:'workbench',kind,useId,actorUUID:healer.uuid,patientUUID:target.actor.uuid,riskySurgery:!!isRiskySurgery});
   const context={sourceSHA:WORKBENCH_SOURCE_SHA,adapterSHA,lexicalSource:true,useId,actorUUID:healer.uuid,patientUUID:target.actor.uuid,kind,continualRecovery:healer.items.some(i=>i.slug==='continual-recovery'),riskySurgery:!!isRiskySurgery,checkIds:[],stageIds:[],...reservation?{checkpointReservation:reservation}:{},...(ward?{groupId:`wb:${useId}:${healer.uuid}`,groupProof:`lexical:${WORKBENCH_SOURCE_SHA}:${useId}`}:{})};
   const poolSource=await recorder.beginWorkbenchSource?.({context,healer,target,isCurrent:()=>macro.execute===execute&&macro.command===sourceCommand});
   const observeCheck=async message=>{if(message?.id&&!context.checkIds.includes(message.id))context.checkIds.push(message.id);await recorder.observeWorkbenchCheck?.(poolSource,message)};
   const observeResult=message=>recorder.observeWorkbenchResult?.(poolSource,message);
   const mark=data=>{const marker=recorder.poolMarker?.(poolSource);return {...data,flags:{...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],explorationManual:{...context},...marker?{explorationManualPoolParticipation:marker}:{}}}}};
   const create=async(data,...args)=>{const m=await bases.ChatMessage.create(mark(data),...args);if(!data.flags?.treat_wounds_battle_medicine&&m?.id)await observeCheck(m);else await observeResult(m);return m};
   const skill=facade(skillUsed,{roll:async args=>skillUsed.roll({...args,extraRollOptions:[...args.extraRollOptions??[],`exploration-manual-use:${useId}`],callback:async(roll,outcome,message,...rest)=>{await observeCheck(message);return args.callback?.(roll,outcome,message,...rest)}})});
   class DamageRoll extends bases.DamageRoll{async toMessage(data,...args){const m=await super.toMessage(mark(data),...args);if(m?.id&&!data.flags?.treat_wounds_battle_medicine)context.stageIds.push(m.id);await observeResult(m);return m}}
   // Assurance check creation uses this target's own ChatMessage facade.
   return {skill,DamageRoll,CheckRoll:bases.CheckRoll,ChatMessage:facade(bases.ChatMessage,{create})};
  };
  return original.call(clone,{...input,explorationManualTarget});
 };
 macro.execute=execute;decorated.add(macro);recorder.observeWorkbenchProvider?.({macro,isCurrent:()=>game.packs.get('xdy-pf2e-workbench.asymonous-benefactor-macros-internal')===pack&&macro.execute===execute&&macro.command===sourceCommand});restores.push(()=>{if(macro.execute===execute)macro.execute=original});return macro;
 }
 const packs=[pack,game.packs.get('xdy-pf2e-workbench.asymonous-benefactor-macros')].filter(Boolean);
 for(const watchedPack of packs)for(const method of ['getDocument','getDocuments']){
  const original=watchedPack[method];if(typeof original!=='function')continue;
  const wrapper=async function(...args){const result=await original.apply(this,args);if(this!==watchedPack)return result;if(Array.isArray(result)){await Promise.all(result.map(instrument));return result}return instrument(result)};
  watchedPack[method]=wrapper;restores.push(()=>{if(watchedPack[method]===wrapper)watchedPack[method]=original});
 }
 await pack.getDocuments({name:'XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine'});
 return ()=>{for(const restore of restores.reverse())restore()};
}
export function workbenchFacts(message){
 const f=message.flags?.treat_wounds_battle_medicine;if(!f)return null;
 const roll=message.rolls?.[0],formula=roll?.toJSON?.().formula??roll?.formula??'';
 return {patientTokenId:f.id,healerActorId:f.healerId,sourceDegree:f.dos,rolledHealing:Number.isFinite(f.healing)?f.healing:null,bmBatonUsed:f.bmBatonUsed,
  effectiveOutcome:/4d8/.test(formula)&&/healing/.test(formula)?'criticalSuccess':/healing/.test(formula)?'success':f.dos===0?'criticalFailure':f.dos===1?'failure':null,actualFormula:formula};
}
/** Read-only recorder. No dice, damage application, or immunity writer is injected. */
export function createManualEvents({game,Hooks,ledger,nativeActions,hpPools,manualPoolSources,manualPoolProof,patreonImmunity,observePatreonTerminal=true,bindRecordingSource,fromUuid,isAuthority=()=>game.user?.isGM&&game.users?.activeGM?.id===game.user.id,sessionId=()=>null,onChange=()=>{},reserveSource,checkpointOptions,onUnknownSource}){
 const seen=new Set(),orders=new Map(),records=new Map(),sourceToActivity=new Map(),sourceMarks=new Set();let evidenceTail=Promise.resolve(),hydratedSession;let unsubscribe,patreonDispose,hook,preHook,itemHook;const scopes=new Map();
 const bound=activity=>activity.temporalSource?.type==='checkpoint-reservation';
 async function evidenceOptions(activity){if(!bound(activity))return {};const options=await checkpointOptions?.(activity.checkpointBinding);if(!options)throw Error('session-driver-required');return options}
 const activityForMessage=messageId=>game.messages.get(messageId)?.flags?.[MODULE_ID]?.explorationManual?.checkpointReservation?.activityId??`manual:${messageId}`;
 const beginCheckpointSource=intent=>typeof reserveSource==='function'?reserveSource(null,intent):Promise.resolve(null);
 async function hydrate(sid){if(!sid||hydratedSession===sid||!isAuthority())return;const saved=await ledger.snapshot?.(sid);seen.clear();orders.clear();records.clear();sourceToActivity.clear();for(const a of saved?.activities??[]){if(!a.source?.manual)continue;seen.add(a.source.messageId??a.id.replace(/^manual:/,''));records.set(a.id,a);const next=orders.get(a.actorUUID)??0;orders.set(a.actorUUID,a.temporalSource?.type==='user-declared'?next+1:Math.max(next,(a.order??0)+1));for(const id of a.proof.resultIds??[])sourceToActivity.set(id,a.id)}hydratedSession=sid;await restoreImmunities();for(const activity of records.values())if(activity.state==='awaiting-evidence'&&Object.values(activity.proof.poolApplications??{}).some(claim=>claim.state==='settled'))await applyEvidence(activity.id,old=>({proof:{...old.proof}}))}
 async function trustedEnrollment(e,session){
  if(Array.isArray(session?.actorUUIDs)&&(!session.actorUUIDs.includes(e.actorUUID)||(e.patientUUIDs??[]).some(uuid=>!session.actorUUIDs.includes(uuid))))return false;
  if(!game.users||!['treatment','battle-medicine'].includes(e.kind))return true;
  const message=game.messages?.get(e.id),uid=message?.author?.id??message?.author??message?.user?.id??message?.user,user=game.users.get(uid),actor=await fromUuid?.(e.actorUUID);
  if(!message||!user||!actor?.testUserPermission?.(user,'OWNER')||message.speaker?.actor!==actor.id)return false;
  if([...records.values()].some(a=>(a.options.sourceMessageId??a.source.messageId)!==e.id&&a.actorUUID===actor.uuid&&a.proof.useId===e.useId&&a.patientUUIDs.some(uuid=>e.patientUUIDs.includes(uuid))&&!bound(a)))return false;
  if(e.source?.type==='native-action'){const meta=message.flags?.[MODULE_ID]?.explorationManualNative,c=message.flags?.pf2e?.context;return message.rolls?.[0]?._evaluated===true&&c?.type==='skill-check'&&meta?.useId===e.useId&&meta.tag===e.source.tag&&c.origin?.actor===actor.uuid&&c.options?.includes(meta.tag)&&e.patientUUIDs.includes(meta.patientUUID)&&(!meta.recordingSessionId||meta.recordingSessionId===session?.id)}
  if(e.source?.type!=='workbench')return false;
  const meta=message.flags?.[MODULE_ID]?.explorationManual,f=message.flags?.treat_wounds_battle_medicine,token=game.scenes?.get(message.speaker?.scene)?.tokens?.get(f?.id)??game.scenes?.active?.tokens?.get(f?.id);
  if(!meta?.lexicalSource||meta.sourceSHA!==WORKBENCH_SOURCE_SHA||meta.useId!==e.useId||meta.actorUUID!==actor.uuid||meta.kind!==e.kind||f?.healerId!==actor.id||token?.actor?.uuid!==meta.patientUUID||!e.patientUUIDs.includes(meta.patientUUID)||!meta.checkIds?.length)return false;
  return meta.checkIds.every(id=>{const check=game.messages.get(id),author=check?.author?.id??check?.author??check?.user?.id??check?.user,c=check?.flags?.pf2e?.context,lexical=check?.flags?.[MODULE_ID]?.explorationManual;return author===uid&&check.speaker?.actor===actor.id&&check.rolls?.[0]?._evaluated===true&&(c?.options?.includes(`exploration-manual-use:${e.useId}`)||lexical?.useId===e.useId&&lexical.actorUUID===actor.uuid&&lexical.sourceSHA===WORKBENCH_SOURCE_SHA)});
 }
 function observe(e){
  e=structuredClone(e);
  if(e.kind==='activity'){
   if(e.source?.type!=='user-record'||typeof e.source.userId!=='string'||!e.source.userId)throw Error('manual-source-identity-unproven');
   e={...normalizeManualActivity(e),id:e.id,expectedSessionId:e.expectedSessionId,kind:'activity',patientUUIDs:[],hpPoolUUIDs:[],source:{type:'user-record',userId:e.source.userId,unverified:true},temporalSource:{type:'user-declared',userId:e.source.userId,recordedAt:game.time.worldTime},missing:['manual-source-requires-review'],checkIds:[],resultIds:[],receiptIds:[],immunityIds:[]};
  }
  const task=evidenceTail.then(()=>enroll(e));evidenceTail=task.catch(()=>{});return task;
 }
 async function enroll(e){
  const reservation=e.checkpointReservation,sid=reservation?.checkpointBinding?.sessionId??sessionId();if(!sid||!isAuthority()||!e.id){if(e.kind==='activity')throw Error('manual-session-changed');return null}
  const session=await ledger.getSession?.(sid);if(ledger.getSession&&session?.status!=='recording'&&!(session?.status==='running'&&reservation)){if(e.kind==='activity')throw Error('manual-session-changed');if(session?.status==='running'&&session.manualCheckpoint&&['treatment','battle-medicine'].includes(e.kind))await onUnknownSource?.(sid);return null}await hydrate(sid);if(!await trustedEnrollment(e,session)){if(e.kind==='activity')throw Error('manual-actor-not-allowed');onChange({error:'manual-source-identity-unproven'});return null}
  if(reservation){
   const old=await ledger.getActivity?.(reservation.activityId),meta=game.messages.get(e.id)?.flags?.[MODULE_ID]?.explorationManual;
   if(!old||old.state!=='awaiting-evidence'||!bound(old)||old.source.reservationId!==reservation.reservationId||reservation.reservationId!==reservation.activityId||!sameCheckpoint(old.checkpointBinding,reservation.checkpointBinding)||!sameCheckpoint(session.manualCheckpoint,reservation.checkpointBinding)||meta?.checkpointReservation?.activityId!==old.id||meta.checkpointReservation.reservationId!==old.id||!sameCheckpoint(old.checkpointBinding,meta.checkpointReservation.checkpointBinding)||e.kind!==old.kind||e.useId!==old.proof.useId||e.actorUUID!==old.actorUUID||e.patientUUIDs?.length!==1||e.patientUUIDs[0]!==old.patientUUIDs[0]||e.source?.type!=='workbench'||e.treatmentImmunitySeconds!==old.treatmentImmunitySeconds)return null;
   if(old.options.sourceMessageId&&old.options.sourceMessageId!==e.id)throw Error('manual-source-already-enrolled');
   if(old.options.sourceMessageId===e.id&&['checkIds','resultIds','receiptIds','immunityIds'].every(key=>(e[key]??[]).every(id=>old.proof[key].includes(id))))return old;
   const proof={...old.proof};for(const key of ['checkIds','resultIds','receiptIds','immunityIds'])proof[key]=[...new Set([...proof[key],...e[key]??[]])];
   const patch={proof,options:{...old.options,sourceMessageId:e.id,sourceDegree:e.sourceDegree??null,effectiveOutcome:e.effectiveOutcome??null,rolledHealing:e.rolledHealing??null,missing:[...e.missing??[],'checkpoint-time-confirmation']}};
   await ledger.transitionActivity(old.id,{expected:['awaiting-evidence'],patch,...await evidenceOptions(old)});const updated={...old,...patch};records.set(old.id,updated);seen.add(e.id);for(const id of proof.resultIds)sourceToActivity.set(id,old.id);onChange(updated);return updated;
  }
  if(e.kind==='activity'){
   if(e.expectedSessionId!==undefined&&e.expectedSessionId!==sid||sessionId()!==sid)throw Error('manual-session-changed');
   if(e.dependsOn.length){
    const saved=await ledger.snapshot?.(sid);if(!saved)throw Error('manual-dependency-unavailable');
    if(e.dependsOn.some(id=>!saved.activities.some(a=>a.id===id&&a.sessionId===sid)))throw Error('manual-dependency-not-in-session');
   }
   const user=game.users?.get(e.source.userId),actor=await fromUuid?.(e.actorUUID),current=await ledger.getSession?.(sid);
   if(!isAuthority()||sessionId()!==sid||e.expectedSessionId!==undefined&&e.expectedSessionId!==sid||ledger.getSession&&current?.status!=='recording')throw Error('manual-session-changed');
   if(!user?.active||!actor?.testUserPermission?.(user,'OWNER')||!current?.actorUUIDs?.includes(actor.uuid))throw Error('manual-actor-not-allowed');
  }
  if(hpPools&&['treatment','battle-medicine'].includes(e.kind)){const missing=new Set(e.missing??[]),pools=new Set();for(const uuid of e.patientUUIDs??[]){try{const actor=await fromUuid(uuid),pool=hpPools.discover(actor);if(!pool.ready)missing.add('hp-pool-source-unavailable');else{pools.add(pool.poolUUID);if(pool.poolUUID!==uuid||pool.memberUUIDs?.some(member=>member!==uuid))missing.add('shared-hp-completion-unavailable')}}catch{missing.add('hp-pool-source-unavailable')}}e={...e,hpPoolUUIDs:[...pools],missing:[...missing]}}
  const old=await ledger.getActivity?.(`manual:${e.id}`)??records.get(`manual:${e.id}`);if(old&&(old.state!=='awaiting-evidence'||old.kind==='activity'||e.kind==='activity'))return null;
  if(seen.has(e.id)&&!old)return null;
  seen.add(e.id);const nextOrder=orders.get(e.actorUUID)??0,order=old?.order??(e.kind==='activity'?(e.order??nextOrder):nextOrder);
  const start=old?.startedAt??game.time.worldTime,duration=Number.isFinite(e.durationSeconds)?e.durationSeconds:0;
  try{const input={id:`manual:${e.id}`,sessionId:sid,providerId:'manual',actorUUID:e.actorUUID,patientUUIDs:e.patientUUIDs??[],hpPoolUUIDs:e.hpPoolUUIDs??[],groupId:e.groupId??`manual:${e.id}`,state:'awaiting-evidence',startedAt:start,endsAt:start+duration,
   kind:e.kind,order,durationSeconds:duration,treatmentImmunitySeconds:e.treatmentImmunitySeconds,groupProof:e.groupProof,source:{...e.source,messageId:e.id,manual:true},
   ...e.kind==='activity'?{durationSource:e.durationSource,temporalSource:e.temporalSource,dependsOn:e.dependsOn,...Object.fromEntries(['notBefore','observedStart','observedEnd'].filter(key=>e[key]!==undefined).map(key=>[key,e[key]]))}:{},
   options:{label:e.label??null,sourceDegree:e.sourceDegree??null,effectiveOutcome:e.effectiveOutcome??null,rolledHealing:e.rolledHealing??null,missing:e.missing??[]},proof:{useId:e.useId??null,checkIds:e.checkIds??[],resultIds:e.resultIds??[e.id],receiptIds:e.receiptIds??[],immunityIds:e.immunityIds??[]}};const saved=await ledger.insertActivity(input,{manualObservation:true})??input;records.set(saved.id,saved);orders.set(e.actorUUID,Math.max(nextOrder,(saved.order??order)+1));for(const id of saved.proof.resultIds)sourceToActivity.set(id,saved.id);onChange(saved);return saved}catch(error){seen.delete(e.id);throw error}
 }

 async function applyEvidence(identity,edit){if(!isAuthority())return;const id=typeof identity==='function'?identity():identity;if(!id)return;const old=await ledger.getActivity?.(id)??records.get(id);if(!old||old.state!=='awaiting-evidence')return;const patch=await edit(structuredClone(old));if(!patch)return;
  // Persisted proof IDs are history; completion must use the currently authorized native receipts.
  const candidate={...old,...patch},native=candidate.source.type==='native-action',patients=new Map();let immunity,poolProof,receipts=[];
  const sourceDocuments=[...candidate.proof.checkIds,...candidate.proof.resultIds].map(id=>game.messages.get(id));
  const sourceSnapshot=()=>canonicalJSON(sourceDocuments.map(message=>message?({speaker:message.speaker,flags:message.flags,rolls:message.rolls?.map(roll=>roll.toJSON?.()??roll)}):null));
  const sourceBefore=sourceSnapshot();
  if(['treatment','battle-medicine'].includes(candidate.kind)){
   if(candidate.source.type==='native-action'){
    const immunity=await patreonImmunity?.evidence(candidate.proof.nativeImmunity,candidate),missing=new Set(candidate.options.missing.filter(m=>m!=='native-immunity-receipt'));
    if(!immunity)missing.add('native-immunity-receipt');
    for(const uuid of candidate.patientUUIDs)patients.set(uuid,await fromUuid?.(uuid));
    candidate.options={...candidate.options,missing:[...missing]};
   }
   for(const receiptId of candidate.proof.receiptIds)receipts.push(await resolveReceipt(game.messages.get(receiptId)));
   for(const resolved of receipts)if(resolved?.patient)patients.set(resolved.patientUUID,resolved.patient);
   if(candidate.source.type==='workbench')for(const uuid of candidate.patientUUIDs)if(!patients.has(uuid))patients.set(uuid,await fromUuid?.(uuid));
   patch.options=currentApplicationEvidence(candidate,old.options.missing.includes('native-application-receipt'),receipts);
   if(candidate.source.type==='native-action'){
    immunity=await patreonImmunity?.evidence(candidate.proof.nativeImmunity,candidate);
   }
  }
  const options=await evidenceOptions(old);
  if(!bound(candidate))poolProof=await manualPoolProof?.evidence(candidate);
  function poolMissing(activity,missing){
   missing.delete('shared-hp-completion-unavailable');missing.delete('hp-pool-source-unavailable');
   for(const uuid of activity.patientUUIDs){const patient=patients.get(uuid),pool=patient&&hpPools?.discover(patient);if(!hpPools)continue;if(!pool?.ready||!activity.hpPoolUUIDs.includes(pool.poolUUID))missing.add('hp-pool-source-unavailable');else if(pool.poolUUID!==uuid||pool.memberUUIDs?.some(member=>member!==uuid)){if(poolProof?.poolUUID!==pool.poolUUID||poolProof.isCurrent(activity)!==true)missing.add('shared-hp-completion-unavailable')}}
  }
  function currentNativeOptions(activity){
   const missing=new Set(activity.options.missing.filter(m=>m!=='native-immunity-receipt'));
   if(immunity?.isCurrent(activity)!==true)missing.add('native-immunity-receipt');
   poolMissing(activity,missing);
   return currentApplicationEvidence({...activity,options:{...activity.options,missing:[...missing]}},old.options.missing.includes('native-application-receipt'),receipts);
  }
  function currentOrdinaryOptions(activity){
   if(native)return currentNativeOptions(activity);
   const missing=new Set(activity.options.missing),ids=[...activity.proof.checkIds,...activity.proof.resultIds];
   if(!sourceDocuments.slice(0,activity.proof.checkIds.length).every(Boolean)||sourceDocuments.every(Boolean)&&(ids.length!==sourceDocuments.length||ids.some((id,index)=>game.messages.get(id)!==sourceDocuments[index])||sourceSnapshot()!==sourceBefore))missing.add('native-source-identity-unproven');
   poolMissing(activity,missing);
   for(const uuid of activity.proof.immunityIds){const patient=patients.get(uuid.split('.Item.')[0]),item=patient?.items?.get(uuid.split('.Item.')[1]);if(!item||!immunityEvidence(item,item.flags?.[MODULE_ID]?.explorationManualImmunity?.creatorId))missing.add('native-immunity-receipt')}
   return currentApplicationEvidence({...activity,options:{...activity.options,missing:[...missing]}},old.options.missing.includes('native-application-receipt'),receipts);
  }
  const ordinary=!bound(candidate)&&['native-action','workbench'].includes(candidate.source.type)&&['treatment','battle-medicine'].includes(candidate.kind);
  if(ordinary)patch.options=currentOrdinaryOptions({...candidate,options:patch.options??candidate.options});
  if(['treatment','battle-medicine'].includes(candidate.kind)){if(!patch.options.missing.length)patch.state='confirmed';else if(patch.state==='confirmed')delete patch.state}
  // The private guard is evaluated inside the ledger mutation and immediately
  // before revision submission. No later UUID lookup can reopen this boundary.
  const ordinaryProof=Object.fromEntries(Object.entries(patch.proof??{}).filter(([key])=>['useId','checkIds','resultIds','receiptIds','immunityIds','nativeImmunity'].includes(key)));
  const updated=ordinary?await ledger.appendManualEvidence(id,{activity:old,proof:ordinaryProof,
   resolveOptions:activity=>currentOrdinaryOptions({...activity,options:{...activity.options,...patch.options}})})
   :await ledger.transitionActivity?.(id,{expected:['awaiting-evidence'],patch,...options})??{...old,...patch};
  records.set(id,updated);onChange(updated);return updated}
 function appendEvidence(identity,edit){const task=evidenceTail.then(async()=>{if(!isAuthority())return;await hydrate(sessionId());return applyEvidence(identity,edit)});evidenceTail=task.catch(()=>{});return task}
 async function observePoolSource(source){
  const message=game.messages.get(source.sourceType==='native-action'?source.checkId:source.resultId);let event;
  if(source.sourceType==='native-action'){event=nativeCheckEvent(message);if(event)event={...event,resultIds:[source.resultId]}}
  else{const meta=message?.flags?.[MODULE_ID]?.explorationManual,facts=message&&workbenchFacts(message);if(meta&&facts)event={id:message.id,actorUUID:meta.actorUUID,patientUUIDs:[meta.patientUUID],kind:meta.kind,durationSeconds:meta.kind==='treatment'?600:6,treatmentImmunitySeconds:meta.kind==='treatment'?(meta.continualRecovery?600:3600):undefined,useId:meta.useId,groupId:meta.groupId,groupProof:meta.groupProof,checkIds:[source.checkId],resultIds:[source.resultId],source:{type:'workbench',sourceSHA:meta.sourceSHA,adapterSHA:meta.adapterSHA,lexicalSource:true,adapter:'target-callback-instrumentation-v1'},...facts,missing:['native-application-receipt','native-immunity-receipt']}}
  if(!event)return null;const activity=await observe(event);return activity??ledger.getActivity(source.activityId);
 }
 function nativeCheckEvent(message){
  const meta=message?.flags?.[MODULE_ID]?.explorationManualNative,c=message?.flags?.pf2e?.context;
  if(game.messages.get(message?.id)!==message||message?.isCheckRoll===false||!meta?.patientUUID||!c?.options?.includes(meta.tag)||c.origin?.actor!==`Actor.${message.speaker?.actor}`)return null;
  return {id:message.id,actorUUID:c.origin.actor,patientUUIDs:[meta.patientUUID],kind:'treatment',durationSeconds:600,treatmentImmunitySeconds:meta.continualRecovery?600:3600,
   effectiveOutcome:c.outcome,useId:meta.useId,checkIds:[message.id],resultIds:[],source:{type:'native-action',tag:meta.tag},missing:['native-application-receipt','native-immunity-receipt']};
 }
 function recordPatreonImmunity(proof){
  const task=evidenceTail.then(async()=>{
   if(!isAuthority())return;await hydrate(sessionId());
   if(proof.binding.recordingSessionId&&proof.binding.recordingSessionId!==sessionId())return;
   const event=nativeCheckEvent(game.messages.get(proof.binding.messageId));if(!event)return;
   // Patreon may finish after action.use returns. Its original terminal supplies
   // the saved patient association; the native call is never replayed here.
   const old=await ledger.getActivity?.(`manual:${event.id}`)??await enroll(event);
   if(!old||bound(old)||old.state!=='awaiting-evidence'||old.source.type!=='native-action')return;
   const candidate={...old,proof:{...old.proof,nativeImmunity:proof}};
   if(!await patreonImmunity.evidence(proof,candidate))return;
   const healer=await fromUuid(candidate.actorUUID),results=new Set(candidate.proof.resultIds),receipts=new Set(candidate.proof.receiptIds);
   for(const child of game.messages?.contents??game.messages?.values?.()??[])if(nativeResultIsTrusted(child,candidate,healer))results.add(child.id);
   candidate.proof.resultIds=[...results];
   for(const message of game.messages?.contents??game.messages?.values?.()??[])if(message.flags?.pf2e?.context?.type==='damage-taken'&&await trustedReceipt(message,candidate))receipts.add(message.id);
   return applyEvidence(old.id,()=>({proof:{...candidate.proof,receiptIds:[...receipts],immunityIds:[...new Set([...old.proof.immunityIds,proof.itemUUID])]},options:old.options}));
  });evidenceTail=task.catch(()=>{});return task;
 }
 async function markResult(message){const options=[...(message.flags?.pf2e?.context?.options??[])].filter(o=>!o.startsWith(`${MODULE_ID}:source:`));options.push(`${MODULE_ID}:source:${message.id}:0`);const changes={'flags.pf2e.context.options':options};if(message.update&&message.canUserModify?.(game.user,'update',changes)!==false)await message.update(changes);return options}
 function queueResultMark(message){const task=markResult(message);sourceMarks.add(task);task.then(()=>sourceMarks.delete(task),()=>sourceMarks.delete(task));return task}
 async function resolveReceipt(message){
  if(!message?.id||game.messages?.get(message.id)!==message)return null;
  const patientUUID=`Actor.${message?.speaker?.actor}`,uid=message?.author?.id??message?.author??message?.user?.id??message?.user;
  const sources=message.flags?.pf2e?.context?.options?.filter(o=>o.startsWith(`${MODULE_ID}:source:`))??[],sourceId=sources.length===1?sources[0].slice(`${MODULE_ID}:source:`.length).replace(/:0$/,''):null;
  const sourceMessage=sourceId&&game.messages.get(sourceId),sourceActor=sourceMessage&&await fromUuid?.(`Actor.${sourceMessage.speaker?.actor}`);
  return {message,patientUUID,uid,sourceMessage,sourceActor,patient:await fromUuid?.(patientUUID)};
 }
 function nativeResultIsTrusted(message,activity,actor){
  if(!message)return false;
  const parentId=message?.flags?.pf2e?.origin?.messageId,parent=game.messages.get(parentId),meta=message?.flags?.[MODULE_ID]?.explorationManualNative,root=parent?.flags?.[MODULE_ID]?.explorationManualNative;
  const userId=message?.author?.id??message?.author??message?.user?.id??message?.user,user=game.users?.get(userId),context=message?.flags?.pf2e?.context;
  return game.messages.get(message?.id)===message&&message.rolls?.[0]?._evaluated===true&&message.isCheckRoll===false
   &&activity.proof.checkIds.includes(parentId)&&parent?.isCheckRoll===true&&!parent.isReroll&&parent.rolls?.[0]?._evaluated===true
   &&actor?.uuid===activity.actorUUID&&user&&actor.testUserPermission?.(user,'OWNER')===true
   &&message.speaker?.actor===parent.speaker?.actor&&`Actor.${message.speaker?.actor}`===activity.actorUUID&&context?.origin?.actor===activity.actorUUID
   &&meta?.useId===activity.proof.useId&&meta.tag===activity.source.tag&&context.options?.includes(meta.tag)
   &&root?.useId===meta.useId&&root.tag===meta.tag&&(!meta.patientUUID||activity.patientUUIDs.includes(meta.patientUUID));
 }
 function receiptIsTrusted(resolved,activity){
  if(!resolved)return false;const {message,patientUUID,uid,patient,sourceMessage,sourceActor}=resolved,user=game.users?.get(uid);
  const currentAuthor=message.author?.id??message.author??message.user?.id??message.user;
  if(game.messages?.get(message.id)!==message||`Actor.${message.speaker?.actor}`!==patientUUID||currentAuthor!==uid||!user||patient?.uuid!==patientUUID||!user.isGM&&!patient.testUserPermission?.(user,'OWNER'))return false;
  const c=message.flags?.pf2e?.context,applied=message.flags?.pf2e?.appliedDamage,sources=c?.options?.filter(o=>o.startsWith(`${MODULE_ID}:source:`))??[];
  if(c?.type!=='damage-taken'||applied?.isReverted||sources.length!==1||!activity.patientUUIDs.includes(patientUUID)||applied&&applied.uuid!==patientUUID)return false;
  const sourceId=sources[0].slice(`${MODULE_ID}:source:`.length).replace(/:0$/,'');
  if(activity.source.type==='native-action'&&(game.messages.get(sourceId)!==sourceMessage||!nativeResultIsTrusted(sourceMessage,activity,sourceActor)))return false;
  return sources[0]===`${MODULE_ID}:source:${sourceId}:0`&&activity.proof.resultIds.includes(sourceId)&&game.messages.get(sourceId)?.rolls?.[0]?._evaluated===true;
 }
 async function trustedReceipt(message,activity){return receiptIsTrusted(await resolveReceipt(message),activity)}
 function currentApplicationEvidence(activity,previouslyMissing,receipts){
  const applied=new Set();for(const resolved of receipts)if(receiptIsTrusted(resolved,activity))for(const option of resolved.message.flags.pf2e.context.options)applied.add(option);
  // A retained non-roll failure card needs no HP application. A deleted/unevaluated result is unknown, not an empty requirement.
  const required=activity.proof.resultIds.filter(id=>{const result=game.messages.get(id);return !result||result.rolls?.length>0});
  const expected=previouslyMissing||required.length>0||activity.proof.receiptIds.length>0,missing=activity.options.missing.filter(m=>m!=='native-application-receipt');
  const stages=activity.source.type==='native-action'?Array.from(game.messages?.contents??game.messages?.values?.()??[]).filter(child=>activity.proof.checkIds.includes(child.flags?.pf2e?.origin?.messageId)):[];
  const nativeStageCount=activity.source.type==='native-action'&&(required.length!==1||stages.length!==1||stages[0]?.id!==required[0])&&!patreonImmunity?.noApplication(activity);
  const wb=game.messages.get(required[0])?.flags?.[MODULE_ID]?.explorationManual;
  const singleWorkbench=activity.source.type==='workbench'&&!bound(activity)&&activity.patientUUIDs.length===1&&required.length===1&&wb?.riskySurgery===false&&!wb.stageIds?.length;
  const nativeReceiptCount=(activity.source.type==='native-action'||singleWorkbench)&&(activity.proof.receiptIds.length!==1||required.some(id=>{
   const resolved=receipts.find(receipt=>receipt?.sourceMessage?.id===id);if(!resolved)return true;
   const current=Array.from(game.messages?.contents??game.messages?.values?.()??[]).filter(message=>{
    const uid=message?.author?.id??message?.author??message?.user?.id??message?.user;
    return receiptIsTrusted({...resolved,message,uid},activity)&&message.flags.pf2e.context.options.includes(`${MODULE_ID}:source:${id}:0`);
   });
   return current.length!==1||current[0].id!==activity.proof.receiptIds[0];
  }));
  if((nativeStageCount||nativeReceiptCount||expected&&(!required.length||required.some(id=>!applied.has(`${MODULE_ID}:source:${id}:0`))))&&!patreonImmunity?.noApplication(activity))missing.push('native-application-receipt');
  return {...activity.options,missing};
 }
 function captureReceipt(message){const c=message.flags?.pf2e?.context;if(game.messages?.get(message?.id)!==message||c?.type!=='damage-taken'||message.flags.pf2e.appliedDamage?.isReverted)return;
  const sources=c.options?.filter(o=>o.startsWith(`${MODULE_ID}:source:`))??[];if(sources.length!==1)return;const sourceId=sources[0].slice(`${MODULE_ID}:source:`.length).replace(/:0$/,'');if(game.messages.get(sourceId)?.rolls?.[0]?._evaluated!==true)return;
  void appendEvidence(()=>sourceToActivity.get(sourceId),async old=>{if(!await trustedReceipt(message,old))return;return {proof:{...old.proof,receiptIds:[...new Set([...old.proof.receiptIds,message.id])]}}}).catch(error=>onChange({error:error.message}));
 }

 function bindImmunity({message,token,kind,sourceSHA}){
  const meta=message?.flags?.[MODULE_ID]?.explorationManual,expected=Object.values(IMMUNITY_SOURCES).find(s=>s.kind===kind)?.sha;
  const trusted=expected===sourceSHA&&game.messages.get(message?.id)===message&&meta?.lexicalSource&&meta.sourceSHA===WORKBENCH_SOURCE_SHA&&meta.kind===kind&&meta.patientUUID===token?.actor?.uuid&&message.flags.treat_wounds_battle_medicine?.id===token.id&&meta.actorUUID===`Actor.${message.flags.treat_wounds_battle_medicine.healerId}`;
  const nativeCreate=token?.actor?.createEmbeddedDocuments?.bind(token.actor);
  return {async createEmbeddedDocuments(type,data,...args){if(!nativeCreate)throw Error('native-immunity-create-unavailable');if(!trusted)return nativeCreate(type,data,...args);
   const marked=data.map(item=>({...item,flags:{...item.flags,[MODULE_ID]:{...item.flags?.[MODULE_ID],explorationManualImmunity:{messageId:message.id,useId:meta.useId,patientUUID:meta.patientUUID,kind,sourceSHA,creatorId:game.user.id}}}}));
   const saved=await nativeCreate(type,marked,...args);if(type!=='Item'||saved.length!==1||!saved[0].uuid?.startsWith(`${meta.patientUUID}.Item.`))return saved;
   await appendEvidence(activityForMessage(message.id),old=>{if(old.kind!==kind||old.proof.useId!==meta.useId||!old.patientUUIDs.includes(meta.patientUUID))return;const missing=old.options.missing.filter(m=>m!=='native-immunity-receipt');return {proof:{...old.proof,immunityIds:[...new Set([...old.proof.immunityIds,saved[0].uuid])]},options:{...old.options,missing},...missing.length===0?{state:'confirmed'}:{}}});return saved;
  }};
 }

 function immunityEvidence(item,userId){
  const f=item.flags?.[MODULE_ID]?.explorationManualImmunity,actor=item.actor??item.parent,source=(item.sourceId??item.flags?.core?.sourceId??'').replace('.Item.','.');
  if(!f||f.creatorId!==userId||actor?.uuid!==f.patientUUID||item.type!=='effect'||source!==(f.kind==='treatment'?'Compendium.pf2e.feat-effects.Lb4q2bBAgxamtix5':'Compendium.pf2e.feat-effects.2XEYQNZTCGpdkyR6'))return;
  const creator=game.users?.get(userId);if(!creator||!creator.isGM&&!actor.testUserPermission?.(creator,'OWNER'))return;
  const message=game.messages.get(f.messageId),meta=message?.flags?.[MODULE_ID]?.explorationManual;
  if(!meta?.lexicalSource||meta.sourceSHA!==WORKBENCH_SOURCE_SHA||meta.useId!==f.useId||meta.patientUUID!==f.patientUUID||meta.kind!==f.kind||Object.values(IMMUNITY_SOURCES).find(s=>s.kind===f.kind)?.sha!==f.sourceSHA)return;
  return {id:activityForMessage(message.id),edit:old=>{
   if(old.kind!==f.kind||old.proof.useId!==f.useId||!old.patientUUIDs.includes(f.patientUUID))return;
   const missing=old.options.missing.filter(m=>m!=='native-immunity-receipt');
   return {proof:{...old.proof,immunityIds:[...new Set([...old.proof.immunityIds,item.uuid])]},options:{...old.options,missing},...missing.length===0?{state:'confirmed'}:{}};
  }};
 }
 function captureImmunity(item,options,userId){const evidence=immunityEvidence(item,userId);if(evidence)void appendEvidence(evidence.id,evidence.edit).catch(error=>onChange({error:error.message}))}
 async function restoreImmunities(){
  const patients=new Set([...records.values()].filter(a=>a.state==='awaiting-evidence'&&a.options.missing.includes('native-immunity-receipt')).flatMap(a=>a.patientUUIDs));
  for(const uuid of patients){if(!isAuthority())return;const actor=await fromUuid?.(uuid);if(actor?.uuid!==uuid)continue;
   for(const item of actor.items?.values?.()??actor.items??[]){const evidence=immunityEvidence(item,item.flags?.[MODULE_ID]?.explorationManualImmunity?.creatorId);if(evidence)await applyEvidence(evidence.id,evidence.edit)}
  }
 }
 async function flushCheckpoint(binding,options){
  await Promise.all([...sourceMarks]);const task=evidenceTail.then(()=>flushCheckpointEvidence(binding,options));evidenceTail=task.catch(()=>{});return task;
 }
 async function flushCheckpointEvidence(binding,{afterAdvance=false}={}){
  await hydrate(binding.sessionId);
  const options=await checkpointOptions?.(binding);if(!options)throw Error('session-driver-required');
  const data=await ledger.snapshot(binding.sessionId),rows=data.activities.filter(a=>bound(a)&&sameCheckpoint(a.checkpointBinding,binding));
  if(rows.length!==1)return {status:'awaiting-evidence',missing:['native-manual-source-unavailable']};const missingAll=new Set();
  for(const old of rows){
   const message=game.messages.get(old.options.sourceMessageId),meta=message?.flags?.[MODULE_ID]?.explorationManual;
   const missing=new Set(old.options.missing.filter(m=>!['native-application-receipt','native-immunity-receipt','native-source-identity-unproven','native-immunity-timing-unconfirmed','hp-pool-source-unavailable','shared-hp-completion-unavailable','ambiguous-native-application'].includes(m)));
   const sourceValid=message&&meta?.checkpointReservation?.activityId===old.id&&meta.checkpointReservation.reservationId===old.source.reservationId&&sameCheckpoint(old.checkpointBinding,meta.checkpointReservation.checkpointBinding)&&meta.continualRecovery===old.options.continualRecovery&&await trustedEnrollment({id:message.id,actorUUID:old.actorUUID,patientUUIDs:old.patientUUIDs,kind:old.kind,useId:old.proof.useId,source:old.source},data.session)&&meta.checkIds.length===old.proof.checkIds.length&&meta.checkIds.every(id=>old.proof.checkIds.includes(id));
   if(!sourceValid)missing.add('native-source-identity-unproven');
   const patient=await fromUuid(old.patientUUIDs[0]),pool=patient&&hpPools?.discover(patient);
   if(!patient||!pool?.ready)missing.add('hp-pool-source-unavailable');else if(pool.poolUUID!==patient.uuid||pool.poolUUID!==old.hpPoolUUIDs[0])missing.add('shared-hp-completion-unavailable');
   const receipts=[];for(const receiptId of old.proof.receiptIds)receipts.push(await resolveReceipt(game.messages.get(receiptId)));
   const application=currentApplicationEvidence({...old,options:{...old.options,missing:[...missing]}},true,receipts);
   // A failure card without a rolled result still needs its real check and immunity.
   if(old.options.effectiveOutcome==='failure'&&!old.options.riskySurgery&&!old.proof.resultIds.some(id=>game.messages.get(id)?.rolls?.length))application.missing=application.missing.filter(m=>m!=='native-application-receipt');
   for(const value of application.missing)missing.add(value);
   for(const id of old.proof.resultIds.filter(id=>game.messages.get(id)?.rolls?.length))if(receipts.filter(receipt=>receiptIsTrusted(receipt,old)&&receipt.message.flags.pf2e.context.options.includes(`${MODULE_ID}:source:${id}:0`)).length!==1)missing.add('ambiguous-native-application');
   const expectedExpiry=binding.from+old.treatmentImmunitySeconds,seal=old.proof.checkpointImmunity;let immunity;
   if(old.proof.immunityIds.length===1){
    const item=await fromUuid(old.proof.immunityIds[0]),creatorId=item?.flags?.[MODULE_ID]?.explorationManualImmunity?.creatorId,evidence=item&&immunityEvidence(item,creatorId),duration=item?.system?.duration,start=item?.system?.start?.value;
    const seconds=duration?.value*({minutes:60,hours:3600}[duration?.unit]??NaN),expiry=start+seconds,actualExpiry=item&&immunityExpiry(item,game.time.worldTime),explicitExpiry=item?.flags?.[MODULE_ID]?.salubriousKiss?.expiresAt;
    const nativeDuration=Number.isFinite(duration?.value)&&duration.value>0&&duration.expiry==='turn-start'&&duration.sustained===false&&(old.options.continualRecovery?duration.unit==='minutes'&&duration.value===10:seconds===3600);
    if(evidence?.id===old.id&&patient?.items?.get?.(item.id)===item&&start===binding.from&&nativeDuration&&seconds===old.treatmentImmunitySeconds&&expiry===expectedExpiry&&(!Number.isFinite(explicitExpiry)||explicitExpiry===expectedExpiry)&&(actualExpiry===expectedExpiry||afterAdvance&&expectedExpiry<=game.time.worldTime&&actualExpiry===null)){
     immunity={itemUUID:item.uuid,messageId:old.options.sourceMessageId,useId:old.proof.useId,patientUUID:patient.uuid,sourceSHA:IMMUNITY_SOURCES['XDY DO_NOT_IMPORT TW Immunity CD'].sha,creatorId,start,duration:{unit:duration.unit,value:duration.value,expiry:duration.expiry},expiresAt:expiry,checkpointId:binding.id};
     if(seal&&canonicalJSON(immunity)!==canonicalJSON(seal))immunity=null;
    }else if(afterAdvance&&!item&&seal&&seal.expiresAt===expectedExpiry&&expectedExpiry<=game.time.worldTime)immunity=seal;
   }
   if(!immunity)missing.add('native-immunity-timing-unconfirmed');
   // Recheck authorized HP receipts after all asynchronous item lookups.
   const finalApplication=currentApplicationEvidence({...old,options:{...old.options,missing:[...missing]}},old.options.effectiveOutcome!=='failure'||!!old.options.riskySurgery,receipts);
   for(const value of finalApplication.missing)missing.add(value);
   const pending=[...missing].filter(value=>value!=='checkpoint-time-confirmation'),proof={...old.proof,...!pending.length&&immunity&&!seal?{checkpointImmunity:immunity}:{}};
   await ledger.transitionActivity(old.id,{expected:['awaiting-evidence'],patch:{proof,options:{...old.options,missing:[...missing]}},...options});records.set(old.id,{...old,proof,options:{...old.options,missing:[...missing]}});for(const value of pending)missingAll.add(value);
  }
  onChange(binding.sessionId);return {status:missingAll.size?'awaiting-evidence':'ready',missing:[...missingAll],activityIds:rows.map(a=>a.id)};
 }
 function start(){if(unsubscribe)return;
  if(observePatreonTerminal)patreonDispose=patreonImmunity?.subscribe(proof=>{void recordPatreonImmunity(proof).catch(error=>onChange({error:error.message}))});
  const restored=evidenceTail.then(async()=>{await hydrate(sessionId());for(const message of game.messages?.contents??game.messages?.values?.()??[])if(message.flags?.pf2e?.context?.type==='damage-taken')captureReceipt(message)});evidenceTail=restored.catch(error=>onChange({error:error.message}));
  itemHook=Hooks?.on('createItem',captureImmunity);
  unsubscribe=nativeActions?.addMiddleware(async(scope,next)=>{
   if(scope.slug!=='treat-wounds'||scope.params.rollOptions?.some(o=>o.startsWith('exploration-activity:')))return next();
    const uuid=globalThis.crypto.randomUUID(),tag=`exploration-manual:${uuid}`;
    // The first checkpoint supports only the pinned Workbench source. Keep
    // ordinary native actions on their existing recording path outside it.
    if(reserveSource)await reserveSource(null,{sourceType:'native-action',kind:'treatment',useId:uuid,actorUUID:scope.params.actors?.[0]?.uuid,patientUUID:(scope.params.target?.actor??scope.params.target)?.uuid});
    let recordingSource;try{recordingSource=await bindRecordingSource?.(scope.actors?.[0]?.uuid??scope.params.actors?.[0]?.uuid)}catch{}
   scope.tagRollOption(tag);let poolSource;try{poolSource=await manualPoolSources?.beginNative(scope,{useId:uuid,tag})}catch(error){onChange({error:error.message})}
   scopes.set(tag,{scope,uuid,poolSource,startedAt:recordingSource?.startedAt??game.time.worldTime,recordingSessionId:recordingSource?.sessionId});
   try{const result=await next();for(const row of result??[]){const m=row.message,c=m?.flags?.pf2e?.context;if(game.messages.get(m?.id)!==m||!c?.options?.includes(tag)||row.actor?.uuid!==c.origin?.actor||m.speaker?.actor!==row.actor.id)continue;
    try{await manualPoolSources?.nativeCheck(poolSource,row)}catch(error){onChange({error:error.message})}
    const patientUUID=m.flags?.[MODULE_ID]?.explorationManualNative?.patientUUID??(scope.params.target?.actor??scope.params.target)?.uuid;
    const event=nativeCheckEvent(m);if(event)await observe(event);else if(patientUUID)await observe({id:m.id,actorUUID:row.actor.uuid,patientUUIDs:[patientUUID],kind:'treatment',durationSeconds:600,treatmentImmunitySeconds:row.actor.items?.some(i=>i.slug==='continual-recovery')?600:3600,useId:uuid,checkIds:[m.id],resultIds:[],source:{type:'native-action',tag},effectiveOutcome:row.outcome,missing:['native-application-receipt','native-immunity-receipt']});}return result;
   }finally{scopes.delete(tag)}
  });
  preHook=Hooks?.on('preCreateChatMessage',(m,data)=>{
   try{
   const c=(data.flags??m.flags)?.pf2e?.context,tag=c?.options?.find(o=>scopes.has(o));if(!tag||c.type&&c.type!=='skill-check')return;const current=scopes.get(tag),target=current.scope.params.target?.actor??current.scope.params.target;
   const actualTarget=typeof c.target?.actor==='string'?c.target.actor:null,patientUUID=actualTarget&&(!target?.uuid||target.uuid===actualTarget)?actualTarget:target?.uuid??null;
   const flags=data.flags??=m.flags,marker=manualPoolSources?.poolMarker?.(current.poolSource);flags[MODULE_ID]={...flags[MODULE_ID],...marker?{explorationManualPoolParticipation:marker}:{},explorationManualNative:{tag,useId:current.uuid,patientUUID,startedAt:current.startedAt,...current.recordingSessionId?{recordingSessionId:current.recordingSessionId}:{},riskySurgery:current.scope.params.selection?!!current.scope.params.selection.feats?.['risky-surgery']:!!c.options?.includes('risky-surgery')||!!flags.pf2e?.modifiers?.some(i=>i.slug==='risky-surgery'&&i.enabled),continualRecovery:current.scope.params.actors?.[0]?.items?.some?.(i=>i.slug==='continual-recovery')??m.actor?.items?.some?.(i=>i.slug==='continual-recovery')??false}};m.updateSource?.({flags});
   }catch{if(data.flags?.[MODULE_ID])delete data.flags[MODULE_ID].explorationManualNative}
  });
  hook=Hooks?.on('createChatMessage',m=>{
   const native=m.flags?.[MODULE_ID]?.explorationManualNative,c=m.flags?.pf2e?.context;
   if(c?.type==='damage-taken'){captureReceipt(m);return}
   if(native&&c?.options?.includes(native.tag)&&c.origin?.actor===`Actor.${m.speaker?.actor}`){
    const origin=m.flags.pf2e.origin?.messageId;if(origin){const root=game.messages.get(origin);if(game.messages.get(m.id)!==m||root?.isCheckRoll!==true||root.flags?.[MODULE_ID]?.explorationManualNative?.useId!==native.useId||root.speaker?.actor!==m.speaker?.actor)return;void markResult(m).then(async()=>{await manualPoolSources?.nativeResult(m);return appendEvidence(`manual:${origin}`,old=>{sourceToActivity.set(m.id,old.id);return {proof:{...old.proof,resultIds:[...new Set([...old.proof.resultIds,m.id])]}}})}).catch(error=>onChange({error:error.message}));return}
    const event=nativeCheckEvent(m);if(event)void observe(event).catch(error=>onChange({error:error.message}));return}
   const f=m.flags?.[MODULE_ID]?.explorationManual;if(!f?.lexicalSource||f.sourceSHA!==WORKBENCH_SOURCE_SHA)return;
    if(m.rolls?.[0]?._evaluated)void queueResultMark(m).catch(error=>onChange({error:error.message}));
   const facts=workbenchFacts(m);if(!facts)return;
   const noApplication=facts.effectiveOutcome==='failure'&&!f.riskySurgery&&!(f.stageIds?.length)&&!(m.rolls?.length);
    void observe({id:m.id,actorUUID:f.actorUUID,patientUUIDs:[f.patientUUID],kind:f.kind,durationSeconds:f.kind==='treatment'?600:6,treatmentImmunitySeconds:f.kind==='treatment'?(f.continualRecovery?600:3600):undefined,useId:f.useId,groupId:f.groupId,groupProof:f.groupProof,checkpointReservation:f.checkpointReservation,checkIds:f.checkIds??[],resultIds:[...(f.stageIds??[]),m.id],source:{type:'workbench',sourceSHA:f.sourceSHA,adapterSHA:f.adapterSHA,lexicalSource:true,adapter:'target-callback-instrumentation-v1'},...facts,missing:[...noApplication?[]:['native-application-receipt'],'native-immunity-receipt']}).catch(error=>onChange({error:error.message}));
  });
 }
 return {start,stop(){unsubscribe?.();unsubscribe=null;patreonDispose?.();patreonDispose=null;if(hook)Hooks.off('createChatMessage',hook);if(preHook)Hooks.off('preCreateChatMessage',preHook);if(itemHook)Hooks.off('createItem',itemHook);hook=null;preHook=null;itemHook=null},observe,observeNativeImmunity:recordPatreonImmunity,bindImmunity,beginCheckpointSource,flushCheckpoint,
  observePoolSource,refreshPoolEvidence:id=>appendEvidence(id,async old=>{
   const receipts=new Set(old.proof.receiptIds);
   for(const claim of Object.values(old.proof.poolApplications??{}))if(claim.state==='settled'){
    const message=game.messages.get(claim.terminal?.receiptId);
    if(message&&await trustedReceipt(message,old))receipts.add(message.id);
   }
   return {proof:{...old.proof,receiptIds:[...receipts]}};
  }),
  poolMarker:source=>manualPoolSources?.poolMarker?.(source),observeWorkbenchProvider:witness=>manualPoolSources?.observeWorkbenchProvider?.(witness),beginWorkbenchSource:async source=>{try{return await manualPoolSources?.beginWorkbench(source)}catch(error){onChange({error:error.message});return null}},observeWorkbenchCheck:async(source,message)=>{try{return await manualPoolSources?.workbenchCheck(source,message)}catch(error){onChange({error:error.message})}},observeWorkbenchResult:async(source,message)=>{try{await queueResultMark(message);return await manualPoolSources?.workbenchResult(source,message)}catch(error){onChange({error:error.message})}}};
}
