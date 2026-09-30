import {MODULE_ID} from './schema.mjs';
export const WORKBENCH_SOURCE_SHA='b3bac907654522fda62b80da182fc253f7147493a465a82081219fb2a0f1308f';
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
 const docs=await pack.getDocuments({name:'XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine'});const macro=docs?.length===1?docs[0]:null;if(!macro)return ()=>{};
 const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(macro.command.replace(/\r\n/g,'\n'))))).map(x=>x.toString(16).padStart(2,'0')).join('');
 // Source snapshots may retain CRLF; accept only the exact pinned byte form too.
 const byteHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(macro.command)))).map(x=>x.toString(16).padStart(2,'0')).join('');
 if(hash!==WORKBENCH_SOURCE_SHA&&byteHash!==WORKBENCH_SOURCE_SHA){recorder.diagnostic?.('workbench-source-unverified');return ()=>{}}
 const command=observeWorkbenchCommand(macro.command,WORKBENCH_SOURCE_SHA),original=macro.execute;
 const execute=async function(input={}){
  if(this!==macro)return original.call(this,input);
  const useId=crypto.randomUUID(),clone=macro.clone({command},{keepId:true});
  const explorationManualTarget=({target,bmtw,skillUsed,isRiskySurgery,healer},bases)=>{
   const kind=bmtw==='Treat Wounds'?'treatment':'battle-medicine',ward=kind==='treatment'&&healer.items.some(i=>i.slug==='ward-medic');
   const context={sourceSHA:WORKBENCH_SOURCE_SHA,lexicalSource:true,useId,actorUUID:healer.uuid,patientUUID:target.actor.uuid,kind,continualRecovery:healer.items.some(i=>i.slug==='continual-recovery'),checkIds:[],stageIds:[],...(ward?{groupId:`wb:${useId}:${healer.uuid}`,groupProof:`lexical:${WORKBENCH_SOURCE_SHA}:${useId}`}:{})};
   const mark=data=>({...data,flags:{...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],explorationManual:{...context}}}});
   const create=async(data,...args)=>{const m=await bases.ChatMessage.create(mark(data),...args);if(!data.flags?.treat_wounds_battle_medicine&&m?.id)context.checkIds.push(m.id);return m};
   const skill=facade(skillUsed,{roll:async args=>skillUsed.roll({...args,callback:async(roll,outcome,message,...rest)=>{if(message?.id)context.checkIds.push(message.id);return args.callback?.(roll,outcome,message,...rest)}})});
   class DamageRoll extends bases.DamageRoll{async toMessage(data,...args){const m=await super.toMessage(mark(data),...args);if(m?.id&&!data.flags?.treat_wounds_battle_medicine)context.stageIds.push(m.id);return m}}
   // Assurance check creation uses this target's own ChatMessage facade.
   return {skill,DamageRoll,CheckRoll:bases.CheckRoll,ChatMessage:facade(bases.ChatMessage,{create})};
  };
  return original.call(clone,{...input,explorationManualTarget});
 };
 macro.execute=execute;return ()=>{if(macro.execute===execute)macro.execute=original};
}
export function workbenchFacts(message){
 const f=message.flags?.treat_wounds_battle_medicine;if(!f)return null;
 const roll=message.rolls?.[0],formula=roll?.formula??roll?.toJSON?.().formula??'';
 return {patientTokenId:f.id,healerActorId:f.healerId,sourceDegree:f.dos,rolledHealing:Number.isFinite(f.healing)?f.healing:null,bmBatonUsed:f.bmBatonUsed,
  effectiveOutcome:/4d8/.test(formula)&&/healing/.test(formula)?'criticalSuccess':/healing/.test(formula)?'success':f.dos===0?'criticalFailure':f.dos===1?'failure':null,actualFormula:formula};
}
/** Read-only recorder. No dice, damage application, or immunity writer is injected. */
export function createManualEvents({game,Hooks,ledger,nativeActions,isAuthority=()=>game.user?.isGM&&game.users?.activeGM?.id===game.user.id,sessionId=()=>null,onChange=()=>{}}){
 const seen=new Set(),orders=new Map();let unsubscribe,hook;const scopes=new Map();
 async function observe(e){
  const sid=sessionId();if(!sid||!isAuthority()||!e.id||seen.has(e.id))return null;
  seen.add(e.id);const order=orders.get(e.actorUUID)??0;orders.set(e.actorUUID,order+1);
  const start=game.time.worldTime,duration=Number.isFinite(e.durationSeconds)?e.durationSeconds:0;
  try{const a=await ledger.insertActivity({id:`manual:${e.id}`,sessionId:sid,providerId:'manual',actorUUID:e.actorUUID,patientUUIDs:e.patientUUIDs??[],hpPoolUUIDs:e.hpPoolUUIDs??[],groupId:e.groupId??`manual:${e.id}`,state:'awaiting-evidence',startedAt:start,endsAt:start+duration,
   kind:e.kind,order,durationSeconds:duration,treatmentImmunitySeconds:e.treatmentImmunitySeconds,groupProof:e.groupProof,source:{...e.source,messageId:e.id,manual:true},
   options:{sourceDegree:e.sourceDegree??null,effectiveOutcome:e.effectiveOutcome??null,rolledHealing:e.rolledHealing??null,missing:e.missing??[]},proof:{useId:e.useId??null,checkIds:e.checkIds??[],resultIds:e.resultIds??[e.id],receiptIds:e.receiptIds??[],immunityIds:e.immunityIds??[]}});onChange(a);return a}catch(error){seen.delete(e.id);throw error}
 }
 function start(){if(unsubscribe)return;
  unsubscribe=nativeActions?.addMiddleware(async(scope,next)=>{
   if(scope.slug!=='treat-wounds'||scope.params.rollOptions?.some(o=>o.startsWith('exploration-activity:'))||!sessionId())return next();
   const uuid=globalThis.crypto.randomUUID(),tag=`exploration-manual:${uuid}`;scope.tagRollOption(tag);scopes.set(tag,{scope,uuid});
   try{const result=await next();for(const row of result??[]){const m=row.message,c=m?.flags?.pf2e?.context;if(game.messages.get(m?.id)!==m||!c?.options?.includes(tag)||row.actor?.uuid!==c.actor)continue;
    const patient=scope.params.target?.actor??scope.params.target;await observe({id:m.id,actorUUID:row.actor.uuid,patientUUIDs:patient?.uuid?[patient.uuid]:[],kind:'treatment',durationSeconds:600,treatmentImmunitySeconds:row.actor.items?.some(i=>i.slug==='continual-recovery')?600:3600,useId:uuid,checkIds:[m.id],resultIds:[],source:{type:'native-action',tag},missing:['native-application-receipt']});}return result;
   }finally{scopes.delete(tag)}
  });
  hook=Hooks?.on('createChatMessage',m=>{
   const f=m.flags?.[MODULE_ID]?.explorationManual;if(!f?.lexicalSource||f.sourceSHA!==WORKBENCH_SOURCE_SHA)return;
   const facts=workbenchFacts(m);if(!facts)return;
   void observe({id:m.id,actorUUID:f.actorUUID,patientUUIDs:[f.patientUUID],kind:f.kind,durationSeconds:f.kind==='treatment'?600:6,treatmentImmunitySeconds:f.kind==='treatment'?(f.continualRecovery?600:3600):undefined,useId:f.useId,groupId:f.groupId,groupProof:f.groupProof,checkIds:f.checkIds??[],resultIds:[...(f.stageIds??[]),m.id],source:{type:'workbench',sourceSHA:f.sourceSHA,lexicalSource:true,adapter:'target-callback-instrumentation-v1'},...facts,missing:['native-application-receipt']}).catch(error=>onChange({error:error.message}));
  });
 }
 return {start,stop(){unsubscribe?.();unsubscribe=null;if(hook)Hooks.off('createChatMessage',hook);hook=null},observe};
}
