import {MODULE_ID} from './schema.mjs';
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
 return command.replace(entry,match=>`const explorationBases = {ChatMessage, DamageRoll, CheckRoll};\n${match}\n const explorationScope = explorationManualTarget({target, bmtw, skillUsed, isRiskySurgery, healer:token.actor}, explorationBases);\n const {ChatMessage, DamageRoll, CheckRoll} = explorationScope;\n skillUsed = explorationScope.skill;`);
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
  const explorationManualTarget=({target,bmtw,skillUsed,isRiskySurgery,healer},bases)=>{
   const kind=bmtw==='Treat Wounds'?'treatment':'battle-medicine',ward=kind==='treatment'&&healer.items.some(i=>i.slug==='ward-medic');
   const context={sourceSHA:WORKBENCH_SOURCE_SHA,adapterSHA,lexicalSource:true,useId,actorUUID:healer.uuid,patientUUID:target.actor.uuid,kind,continualRecovery:healer.items.some(i=>i.slug==='continual-recovery'),riskySurgery:!!isRiskySurgery,checkIds:[],stageIds:[],...(ward?{groupId:`wb:${useId}:${healer.uuid}`,groupProof:`lexical:${WORKBENCH_SOURCE_SHA}:${useId}`}:{})};
   const mark=data=>({...data,flags:{...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],explorationManual:{...context}}}});
   const create=async(data,...args)=>{const m=await bases.ChatMessage.create(mark(data),...args);if(!data.flags?.treat_wounds_battle_medicine&&m?.id)context.checkIds.push(m.id);return m};
   const skill=facade(skillUsed,{roll:async args=>skillUsed.roll({...args,extraRollOptions:[...args.extraRollOptions??[],`exploration-manual-use:${useId}`],callback:async(roll,outcome,message,...rest)=>{if(message?.id)context.checkIds.push(message.id);return args.callback?.(roll,outcome,message,...rest)}})});
   class DamageRoll extends bases.DamageRoll{async toMessage(data,...args){const m=await super.toMessage(mark(data),...args);if(m?.id&&!data.flags?.treat_wounds_battle_medicine)context.stageIds.push(m.id);return m}}
   // Assurance check creation uses this target's own ChatMessage facade.
   return {skill,DamageRoll,CheckRoll:bases.CheckRoll,ChatMessage:facade(bases.ChatMessage,{create})};
  };
  return original.call(clone,{...input,explorationManualTarget});
 };
 macro.execute=execute;decorated.add(macro);restores.push(()=>{if(macro.execute===execute)macro.execute=original});return macro;
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
export function createManualEvents({game,Hooks,ledger,nativeActions,hpPools,fromUuid,isAuthority=()=>game.user?.isGM&&game.users?.activeGM?.id===game.user.id,sessionId=()=>null,onChange=()=>{}}){
 const seen=new Set(),orders=new Map(),records=new Map(),sourceToActivity=new Map();let evidenceTail=Promise.resolve(),hydratedSession;let unsubscribe,hook,preHook,itemHook;const scopes=new Map();
 async function hydrate(sid){if(!sid||hydratedSession===sid||!isAuthority())return;const saved=await ledger.snapshot?.(sid);seen.clear();orders.clear();records.clear();sourceToActivity.clear();for(const a of saved?.activities??[]){if(!a.source?.manual)continue;seen.add(a.source.messageId??a.id.replace(/^manual:/,''));records.set(a.id,a);orders.set(a.actorUUID,Math.max(orders.get(a.actorUUID)??0,(a.order??0)+1));for(const id of a.proof.resultIds??[])sourceToActivity.set(id,a.id)}hydratedSession=sid}
 async function trustedEnrollment(e,session){
  if(Array.isArray(session?.actorUUIDs)&&(!session.actorUUIDs.includes(e.actorUUID)||(e.patientUUIDs??[]).some(uuid=>!session.actorUUIDs.includes(uuid))))return false;
  if(!game.users||!['treatment','battle-medicine'].includes(e.kind))return true;
  const message=game.messages?.get(e.id),uid=message?.author?.id??message?.author??message?.user?.id??message?.user,user=game.users.get(uid),actor=await fromUuid?.(e.actorUUID);
  if(!message||!user||!actor?.testUserPermission?.(user,'OWNER')||message.speaker?.actor!==actor.id)return false;
  if([...records.values()].some(a=>a.source.messageId!==e.id&&a.actorUUID===actor.uuid&&a.proof.useId===e.useId&&a.patientUUIDs.some(uuid=>e.patientUUIDs.includes(uuid))))return false;
  if(e.source?.type==='native-action'){const meta=message.flags?.[MODULE_ID]?.explorationManualNative,c=message.flags?.pf2e?.context;return message.rolls?.[0]?._evaluated===true&&c?.type==='skill-check'&&meta?.useId===e.useId&&meta.tag===e.source.tag&&c.origin?.actor===actor.uuid&&c.options?.includes(meta.tag)&&e.patientUUIDs.includes(meta.patientUUID)}
  if(e.source?.type!=='workbench')return false;
  const meta=message.flags?.[MODULE_ID]?.explorationManual,f=message.flags?.treat_wounds_battle_medicine,token=game.scenes?.get(message.speaker?.scene)?.tokens?.get(f?.id)??game.scenes?.active?.tokens?.get(f?.id);
  if(!meta?.lexicalSource||meta.sourceSHA!==WORKBENCH_SOURCE_SHA||meta.useId!==e.useId||meta.actorUUID!==actor.uuid||meta.kind!==e.kind||f?.healerId!==actor.id||token?.actor?.uuid!==meta.patientUUID||!e.patientUUIDs.includes(meta.patientUUID)||!meta.checkIds?.length)return false;
  return meta.checkIds.every(id=>{const check=game.messages.get(id),author=check?.author?.id??check?.author??check?.user?.id??check?.user,c=check?.flags?.pf2e?.context,lexical=check?.flags?.[MODULE_ID]?.explorationManual;return author===uid&&check.speaker?.actor===actor.id&&check.rolls?.[0]?._evaluated===true&&(c?.options?.includes(`exploration-manual-use:${e.useId}`)||lexical?.useId===e.useId&&lexical.actorUUID===actor.uuid&&lexical.sourceSHA===WORKBENCH_SOURCE_SHA)});
 }
 function observe(e){const task=evidenceTail.then(()=>enroll(e));evidenceTail=task.catch(()=>{});return task}
 async function enroll(e){
  const sid=sessionId();if(!sid||!isAuthority()||!e.id)return null;
  const session=await ledger.getSession?.(sid);if(ledger.getSession&&session?.status!=='recording')return null;await hydrate(sid);if(!await trustedEnrollment(e,session)){onChange({error:'manual-source-identity-unproven'});return null}
  if(hpPools&&['treatment','battle-medicine'].includes(e.kind)){const missing=new Set(e.missing??[]);for(const uuid of e.patientUUIDs??[]){try{const actor=await fromUuid(uuid),pool=hpPools.discover(actor);if(!pool.ready)missing.add('hp-pool-source-unavailable');else if(pool.poolUUID!==uuid)missing.add('shared-hp-completion-unavailable')}catch{missing.add('hp-pool-source-unavailable')}}e={...e,missing:[...missing]}}
  if(seen.has(e.id)){const old=records.get(`manual:${e.id}`)??await ledger.getActivity?.(`manual:${e.id}`);if(!old||old.state!=='awaiting-evidence')return null;const proof={...old.proof};for(const key of ['checkIds','resultIds','receiptIds','immunityIds'])proof[key]=[...new Set([...(proof[key]??[]),...(e[key]??[])])];const options={...old.options};for(const key of ['sourceDegree','effectiveOutcome','rolledHealing'])if(e[key]!==undefined)options[key]=e[key];const patch={proof,options,...e.treatmentImmunitySeconds!==undefined?{treatmentImmunitySeconds:e.treatmentImmunitySeconds}:{}};const updated={...old,...patch};await ledger.transitionActivity?.(old.id,{expected:['awaiting-evidence'],patch});records.set(old.id,updated);for(const id of proof.resultIds)sourceToActivity.set(id,old.id);return updated}
  seen.add(e.id);const order=orders.get(e.actorUUID)??0;orders.set(e.actorUUID,order+1);
  const start=game.time.worldTime,duration=Number.isFinite(e.durationSeconds)?e.durationSeconds:0;
  try{const input={id:`manual:${e.id}`,sessionId:sid,providerId:'manual',actorUUID:e.actorUUID,patientUUIDs:e.patientUUIDs??[],hpPoolUUIDs:e.hpPoolUUIDs??[],groupId:e.groupId??`manual:${e.id}`,state:'awaiting-evidence',startedAt:start,endsAt:start+duration,
   kind:e.kind,order,durationSeconds:duration,treatmentImmunitySeconds:e.treatmentImmunitySeconds,groupProof:e.groupProof,source:{...e.source,messageId:e.id,manual:true},
   options:{label:e.label??null,sourceDegree:e.sourceDegree??null,effectiveOutcome:e.effectiveOutcome??null,rolledHealing:e.rolledHealing??null,missing:e.missing??[]},proof:{useId:e.useId??null,checkIds:e.checkIds??[],resultIds:e.resultIds??[e.id],receiptIds:e.receiptIds??[],immunityIds:e.immunityIds??[]}};await ledger.insertActivity(input);records.set(input.id,input);for(const id of input.proof.resultIds)sourceToActivity.set(id,input.id);onChange(input);return input}catch(error){seen.delete(e.id);throw error}
 }

 function appendEvidence(identity,edit){const task=evidenceTail.then(async()=>{if(!isAuthority())return;await hydrate(sessionId());const id=typeof identity==='function'?identity():identity;if(!id)return;const old=records.get(id)??await ledger.getActivity?.(id);if(!old||old.state!=='awaiting-evidence')return;const patch=edit(structuredClone(old));if(!patch)return;const updated={...old,...patch};await ledger.transitionActivity?.(id,{expected:['awaiting-evidence'],patch});records.set(id,updated);onChange(updated);return updated});evidenceTail=task.catch(()=>{});return task}
 async function markResult(message){const options=[...(message.flags?.pf2e?.context?.options??[])].filter(o=>!o.startsWith(`${MODULE_ID}:source:`));options.push(`${MODULE_ID}:source:${message.id}:0`);if(message.update)await message.update({'flags.pf2e.context.options':options});return options}
 function captureReceipt(message){const c=message.flags?.pf2e?.context;if(c?.type!=='damage-taken'||message.flags.pf2e.appliedDamage?.isReverted)return;
  const sources=c.options?.filter(o=>o.startsWith(`${MODULE_ID}:source:`))??[];if(sources.length!==1)return;const sourceId=sources[0].slice(`${MODULE_ID}:source:`.length).replace(/:0$/,'');if(game.messages.get(sourceId)?.rolls?.[0]?._evaluated!==true)return;
  void appendEvidence(()=>sourceToActivity.get(sourceId),old=>{const patientUUID=`Actor.${message.speaker?.actor}`;if(!old.patientUUIDs.includes(patientUUID)||message.flags.pf2e.appliedDamage&&message.flags.pf2e.appliedDamage.uuid!==patientUUID)return;const proof={...old.proof,receiptIds:[...new Set([...old.proof.receiptIds,message.id])]},applied=new Set(proof.receiptIds.flatMap(r=>game.messages.get(r)?.flags?.pf2e?.context?.options??[]));const required=proof.resultIds.filter(r=>game.messages.get(r)?.rolls?.[0]?._evaluated);const missing=old.options.missing.filter(m=>m!=='native-application-receipt');if(required.some(r=>!applied.has(`${MODULE_ID}:source:${r}:0`)))missing.push('native-application-receipt');return {proof,options:{...old.options,missing},...missing.length===0?{state:'confirmed'}:{}}}).catch(error=>onChange({error:error.message}));
 }

 function bindImmunity({message,token,kind,sourceSHA}){
  const meta=message?.flags?.[MODULE_ID]?.explorationManual,expected=Object.values(IMMUNITY_SOURCES).find(s=>s.kind===kind)?.sha;
  const trusted=expected===sourceSHA&&game.messages.get(message?.id)===message&&meta?.lexicalSource&&meta.sourceSHA===WORKBENCH_SOURCE_SHA&&meta.kind===kind&&meta.patientUUID===token?.actor?.uuid&&message.flags.treat_wounds_battle_medicine?.id===token.id&&meta.actorUUID===`Actor.${message.flags.treat_wounds_battle_medicine.healerId}`;
  const nativeCreate=token?.actor?.createEmbeddedDocuments?.bind(token.actor);
  return {async createEmbeddedDocuments(type,data,...args){if(!nativeCreate)throw Error('native-immunity-create-unavailable');if(!trusted)return nativeCreate(type,data,...args);
   const marked=data.map(item=>({...item,flags:{...item.flags,[MODULE_ID]:{...item.flags?.[MODULE_ID],explorationManualImmunity:{messageId:message.id,useId:meta.useId,patientUUID:meta.patientUUID,kind,sourceSHA,creatorId:game.user.id}}}}));
   const saved=await nativeCreate(type,marked,...args);if(type!=='Item'||saved.length!==1||!saved[0].uuid?.startsWith(`${meta.patientUUID}.Item.`))return saved;
   await appendEvidence(`manual:${message.id}`,old=>{if(old.kind!==kind||old.proof.useId!==meta.useId||!old.patientUUIDs.includes(meta.patientUUID))return;const missing=old.options.missing.filter(m=>m!=='native-immunity-receipt');return {proof:{...old.proof,immunityIds:[...new Set([...old.proof.immunityIds,saved[0].uuid])]},options:{...old.options,missing},...missing.length===0?{state:'confirmed'}:{}}});return saved;
  }};
 }

 function captureImmunity(item,options,userId){
  const f=item.flags?.[MODULE_ID]?.explorationManualImmunity,actor=item.actor??item.parent,source=(item.sourceId??item.flags?.core?.sourceId??'').replace('.Item.','.');
  if(!f||f.creatorId!==userId||actor?.uuid!==f.patientUUID||item.type!=='effect'||source!==(f.kind==='treatment'?'Compendium.pf2e.feat-effects.Lb4q2bBAgxamtix5':'Compendium.pf2e.feat-effects.2XEYQNZTCGpdkyR6'))return;
  if(game.users&&!actor.testUserPermission?.(game.users.get(userId),'OWNER'))return;
  const message=game.messages.get(f.messageId),meta=message?.flags?.[MODULE_ID]?.explorationManual;
  if(!meta?.lexicalSource||meta.sourceSHA!==WORKBENCH_SOURCE_SHA||meta.useId!==f.useId||meta.patientUUID!==f.patientUUID||meta.kind!==f.kind||Object.values(IMMUNITY_SOURCES).find(s=>s.kind===f.kind)?.sha!==f.sourceSHA)return;
  void appendEvidence(`manual:${message.id}`,old=>{if(old.proof.useId!==f.useId||!old.patientUUIDs.includes(f.patientUUID))return;const missing=old.options.missing.filter(m=>m!=='native-immunity-receipt');return {proof:{...old.proof,immunityIds:[...new Set([...old.proof.immunityIds,item.uuid])]},options:{...old.options,missing},...missing.length===0?{state:'confirmed'}:{}}}).catch(error=>onChange({error:error.message}));
 }
 function start(){if(unsubscribe)return;
  const restored=evidenceTail.then(async()=>{await hydrate(sessionId());for(const message of game.messages?.contents??game.messages?.values?.()??[])if(message.flags?.pf2e?.context?.type==='damage-taken')captureReceipt(message)});evidenceTail=restored.catch(error=>onChange({error:error.message}));
  itemHook=Hooks?.on('createItem',captureImmunity);
  unsubscribe=nativeActions?.addMiddleware(async(scope,next)=>{
   if(scope.slug!=='treat-wounds'||scope.params.rollOptions?.some(o=>o.startsWith('exploration-activity:')))return next();
   const uuid=globalThis.crypto.randomUUID(),tag=`exploration-manual:${uuid}`;scope.tagRollOption(tag);scopes.set(tag,{scope,uuid});
   try{const result=await next();for(const row of result??[]){const m=row.message,c=m?.flags?.pf2e?.context;if(game.messages.get(m?.id)!==m||!c?.options?.includes(tag)||row.actor?.uuid!==c.origin?.actor||m.speaker?.actor!==row.actor.id)continue;
    const patient=scope.params.target?.actor??scope.params.target;await observe({id:m.id,actorUUID:row.actor.uuid,patientUUIDs:patient?.uuid?[patient.uuid]:[],kind:'treatment',durationSeconds:600,treatmentImmunitySeconds:row.actor.items?.some(i=>i.slug==='continual-recovery')?600:3600,useId:uuid,checkIds:[m.id],resultIds:[],source:{type:'native-action',tag},effectiveOutcome:row.outcome,missing:['native-application-receipt','native-immunity-receipt']});}return result;
   }finally{scopes.delete(tag)}
  });
  preHook=Hooks?.on('preCreateChatMessage',(m,data)=>{
   const c=(data.flags??m.flags)?.pf2e?.context,tag=c?.options?.find(o=>scopes.has(o));if(!tag||c.type&&c.type!=='skill-check')return;const current=scopes.get(tag),target=current.scope.params.target?.actor??current.scope.params.target;
   const flags=data.flags??=m.flags;flags[MODULE_ID]={...flags[MODULE_ID],explorationManualNative:{tag,useId:current.uuid,patientUUID:target?.uuid??null,continualRecovery:current.scope.params.actors?.[0]?.items?.some?.(i=>i.slug==='continual-recovery')??m.actor?.items?.some?.(i=>i.slug==='continual-recovery')??false}};m.updateSource?.({flags});
  });
  hook=Hooks?.on('createChatMessage',m=>{
   const native=m.flags?.[MODULE_ID]?.explorationManualNative,c=m.flags?.pf2e?.context;
   if(c?.type==='damage-taken'){captureReceipt(m);return}
   if(native&&c?.options?.includes(native.tag)&&c.origin?.actor===`Actor.${m.speaker?.actor}`){
    const origin=m.flags.pf2e.origin?.messageId;if(origin){const root=game.messages.get(origin);if(root?.isCheckRoll!==true||root.flags?.[MODULE_ID]?.explorationManualNative?.useId!==native.useId||root.speaker?.actor!==m.speaker?.actor)return;void markResult(m).then(()=>appendEvidence(`manual:${origin}`,old=>{sourceToActivity.set(m.id,old.id);return {proof:{...old.proof,resultIds:[...new Set([...old.proof.resultIds,m.id])]}}})).catch(error=>onChange({error:error.message}));return}
    if(m.isCheckRoll===false)return;
    void observe({id:m.id,actorUUID:c.origin.actor,patientUUIDs:native.patientUUID?[native.patientUUID]:[],kind:'treatment',durationSeconds:600,treatmentImmunitySeconds:native.continualRecovery?600:3600,effectiveOutcome:c.outcome,useId:native.useId,checkIds:[m.id],resultIds:[],source:{type:'native-action',tag:native.tag},missing:['native-application-receipt','native-immunity-receipt']}).catch(error=>onChange({error:error.message}));return}
   const f=m.flags?.[MODULE_ID]?.explorationManual;if(!f?.lexicalSource||f.sourceSHA!==WORKBENCH_SOURCE_SHA)return;
   if(m.rolls?.[0]?._evaluated)void markResult(m).catch(error=>onChange({error:error.message}));
   const facts=workbenchFacts(m);if(!facts)return;
   const noApplication=facts.effectiveOutcome==='failure'&&!f.riskySurgery&&!(f.stageIds?.length)&&!(m.rolls?.length);
   void observe({id:m.id,actorUUID:f.actorUUID,patientUUIDs:[f.patientUUID],kind:f.kind,durationSeconds:f.kind==='treatment'?600:6,treatmentImmunitySeconds:f.kind==='treatment'?(f.continualRecovery?600:3600):undefined,useId:f.useId,groupId:f.groupId,groupProof:f.groupProof,checkIds:f.checkIds??[],resultIds:[...(f.stageIds??[]),m.id],source:{type:'workbench',sourceSHA:f.sourceSHA,adapterSHA:f.adapterSHA,lexicalSource:true,adapter:'target-callback-instrumentation-v1'},...facts,missing:[...noApplication?[]:['native-application-receipt'],'native-immunity-receipt']}).catch(error=>onChange({error:error.message}));
  });
 }
 return {start,stop(){unsubscribe?.();unsubscribe=null;if(hook)Hooks.off('createChatMessage',hook);if(preHook)Hooks.off('preCreateChatMessage',preHook);if(itemHook)Hooks.off('createItem',itemHook);hook=null;preHook=null;itemHook=null},observe,bindImmunity};
}
