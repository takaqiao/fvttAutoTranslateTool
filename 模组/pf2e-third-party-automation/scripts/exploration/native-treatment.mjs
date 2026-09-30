import {MODULE_ID} from './schema.mjs';
import {TREAT_WOUNDS_IMMUNITY} from '../salubrious-kiss-rules.mjs';
import {cooldown} from './capabilities.mjs';
const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const marker=id=>`exploration-activity:${id}`;
const author=m=>m.author?.id??m.author??m.user?.id??m.user;
export function classifyResult(roll,outcome) {
  const formula=roll?.toJSON?.().formula?.replace(/\s/g,'');
  if(!roll?._evaluated||!Number.isFinite(roll.total)||!formula)throw Error('unpersisted-native-roll');
  if(formula==='{1d8[slashing]}')return 'surgery';
  if(outcome==='criticalFailure'&&formula==='{(1d8)}')return 'failure-damage';
  if(['success','criticalSuccess'].includes(outcome)&&/\[healing\]/.test(formula))return 'healing';
  throw Error('unproven-native-result-stage');
}
export function medicStackingEffect({medicBonus,sourceActorUUID}) {
  if(!Number.isFinite(medicBonus)||medicBonus<=0)return null;
  return {name:'医师治疗加值',type:'effect',system:{duration:{value:-1,unit:'unlimited'},rules:[
    {key:'FlatModifier',selector:'healing-received',slug:'exploration-medic-baked-adjustment',type:'untyped',value:-medicBonus},
    {key:'FlatModifier',selector:'healing-received',slug:'exploration-medic',type:'circumstance',value:medicBonus}
  ],context:{origin:{actor:sourceActorUUID,token:null,item:null,rollOptions:[]},target:null,roll:null}}};
}
export function createNativeTreatment({game,Hooks,fromUuid,ownerOperations,checkScope,damageGuard,hpPools,apply,createMessage=data=>globalThis.ChatMessage.create(data),timeoutMs=15000}) {
  const applications=new Set();
  const valid=(a,ctx)=>{if(!ownerOperations.isActivityContext(ctx,a.id))throw Error('private-activity-context-required');ctx.validate?.();if(game.time.worldTime<a.endsAt)throw Error('activity-not-finished')};
  function waitFor(read,event) {
    const current=read();if(current)return Promise.resolve(current);
    return new Promise((resolve,reject)=>{let hook,timer;const finish=(error,value)=>{Hooks.off(event,hook);clearTimeout(timer);error?reject(error):resolve(value)};hook=Hooks.on(event,()=>{try{const value=read();if(value)finish(null,value)}catch(e){finish(e)}});timer=setTimeout(()=>finish(Error('native-evidence-uncertain-no-retry')),timeoutMs);const value=read();if(value)finish(null,value)});
  }
  async function applySavedResult(activity,result,ctx) {
    valid(activity,ctx);const {message,patient,stage}=result;
    if(game.messages.get(message.id)!==message||message.rolls.length!==1)throw Error('result-not-persisted');
    const roll=message.rolls[0],application=`${MODULE_ID}:exploration-apply:${activity.id}:${message.id}:${patient.uuid}`;
    if(applications.has(application))throw Error('application-already-started');
    const source=`${MODULE_ID}:source:${message.id}:0`;
    const foundToken=patient.getActiveTokens?.(false,true)?.[0];
    if(!apply&&!foundToken&&!patient.token)throw Error('native-application-token-required');
    const params={damage:stage==='healing'?-roll.total:roll,token:foundToken?.document??foundToken??patient.token??null,...result.item?{item:result.item}:{},
      skipIWR:stage==='healing',final:false,shieldBlockRequest:false,outcome:result.outcome,
      rollOptions:new Set([...(message.flags.pf2e.context?.options??[]).filter(o=>o!=='skip-handling-message'),source,application])};
    const effects=[];if(stage==='healing'&&result.medicBonus){const effect=medicStackingEffect({medicBonus:result.medicBonus,sourceActorUUID:activity.actorUUID});if(effect)effects.push(effect)}
    const recipient=effects.length?patient.getContextualClone?.([],effects):patient;
    if(!recipient)throw Error('native-medic-stacking-context-unavailable');
    const request={activity,ctx,message,patient,recipient,stage,source,application,params};
    const revoke=await damageGuard.authorizeExploration(request);valid(activity,ctx);applications.add(application);
    let receipt;const privacyHook=Hooks.on('preCreateChatMessage',(m,data)=>{
      const c=(data.flags??m.flags)?.pf2e?.context;if(c?.type!=='damage-taken'||!c.options?.includes(application))return;
      const sourceAudience=[...(message.whisper??[])],nativeAudience=[...(data.whisper??m.whisper??[])];
      const audience=sourceAudience.length&&nativeAudience.length?sourceAudience.filter(id=>nativeAudience.includes(id)):sourceAudience.length?sourceAudience:nativeAudience;
      if(sourceAudience.length&&nativeAudience.length&&!audience.length)return false;
      data.whisper=audience;data.blind=!!message.blind||!!data.blind||!!m.blind;m.updateSource?.({whisper:audience,blind:data.blind});
    });
    const hook=Hooks.on('createChatMessage',m=>{
      const c=m.flags?.pf2e?.context;if(c?.type!=='damage-taken'||!c.options?.includes(application))return;
      if(receipt)throw Error('duplicate-application-receipt');
      if(!c.options.includes(source)||author(m)!==game.user.id||m.speaker?.actor!==patient.id||m.flags.pf2e.appliedDamage&&m.flags.pf2e.appliedDamage.uuid!==patient.uuid)throw Error('invalid-application-receipt');receipt=m;
    });
    try{
      const applicationResult=await hpPools.withNativeApplication(activity,patient,async()=>({nativeResult:await (apply?apply(request):recipient.applyDamage(params)),receipt}));result.poolReceipt=applicationResult.poolReceipt;valid(activity,ctx);
      const confirmed=await waitFor(()=>receipt&&game.messages.get(receipt.id)===receipt?receipt:null,'createChatMessage');valid(activity,ctx);return confirmed.id;
    }finally{Hooks.off('createChatMessage',hook);Hooks.off('preCreateChatMessage',privacyHook);revoke()}
  }
  async function single(activity,patient,ctx) {
    valid(activity,ctx);const healer=await fromUuid(activity.actorUUID);valid(activity,ctx);let check;
    const proof={useId:activity.id,checkIds:[],resultIds:[],receiptIds:[],immunityIds:[],poolReceipts:[]};
    const captured=new Map(),live=marker(activity.id);
    const belongs=m=>m.flags?.pf2e?.context?.options?.includes(live)&&author(m)===game.user.id&&(m.actor===healer||m.speaker?.actor===healer.id);
    const pre=Hooks.on('preCreateChatMessage',(message,data)=>{
      if(!belongs(message)&&!belongs(data))return;
      const origin=(data.flags??message.flags)?.pf2e?.origin?.messageId,original=origin&&game.messages.get(origin);
      if(original&&belongs(original)){data.blind=!!original.blind;data.whisper=[...(original.whisper??[])];message.updateSource?.({blind:data.blind,whisper:data.whisper})}
      const flags=data.flags??=message.flags??{};flags.pf2e??={};flags.pf2e.context??={};
      flags.pf2e.context.options=[...new Set([...(flags.pf2e.context.options??[]),'skip-handling-message'])];flags.pf2e.suppressDamageButtons=true;
      flags[MODULE_ID]={...flags[MODULE_ID],exploration:{activityId:activity.id,patientUUID:patient.uuid}};
      message.updateSource?.({flags});
    });
    const post=Hooks.on('createChatMessage',m=>{if(belongs(m))captured.set(m.id,m)});
    try{
      const result=await checkScope.runExploration({activity,healer,patient,ctx},()=>game.pf2e.actions.get('treat-wounds').use({actors:[healer],target:patient,
        selection:{skill:activity.options.skill??'medicine',rank:activity.options.rank??'trained',modifier:0,feats:{'risky-surgery':!!activity.options.riskySurgery,'mortal-healing':false}},rollOptions:[live],message:{create:true}}));
      valid(activity,ctx);check=result?.[0];
      if(check?.actor!==healer||!check.message||game.messages.get(check.message.id)!==check.message||!belongs(check.message)||check.message.flags.pf2e.context.outcome!==check.outcome)throw Error('native-check-unconfirmed');
      if(activity.options.assurance&&(!check.message.flags.pf2e.context.substitutions?.some(s=>s.slug==='assurance'&&s.selected)||check.roll?.dice?.length!==0||check.roll?.terms?.[0]?.number!==10))throw Error('native-assurance-unconfirmed');
      const actualRisky=check.message.flags.pf2e.modifiers?.some(m=>m.slug==='risky-surgery'&&m.enabled)===true;
      const expected=(actualRisky?1:0)+(check.outcome==='failure'?0:1);
      const results=await waitFor(()=>{
        const list=[...captured.values()].filter(m=>m.flags?.pf2e?.origin?.messageId===check.message.id);
        if(list.length>expected)throw Error('ambiguous-native-results');return list.length===expected?list:null;
      },'createChatMessage');valid(activity,ctx);
      if(activity.options.riskySurgery&&!actualRisky){
        if(!healer.items?.some(i=>i.type==='feat'&&(i.slug??i.system?.slug)==='risky-surgery'))throw Error('risky-surgery-feat-unavailable');
        // 8.5.1 keys the callback cut on an enabled circumstance modifier.
        // Suppression does not cancel the surgery declared at begin. Generate
        // its one missing native DamageRoll, linked to this saved check.
        const DamageRoll=game.pf2e.DamageRoll??globalThis.CONFIG?.Dice?.rolls?.find(cls=>cls.name==='DamageRoll');if(!DamageRoll)throw Error('native-damage-roll-unavailable');
        const cut=await new DamageRoll('{1d8[slashing]}').evaluate();valid(activity,ctx);
        const flags=structuredClone(check.message.flags);flags.pf2e.origin={...flags.pf2e.origin,messageId:check.message.id};
        const surgery=await createMessage({author:game.user.id,speaker:structuredClone(check.message.speaker??{actor:healer.id}),blind:!!check.message.blind,whisper:[...(check.message.whisper??[])],flags,rolls:[cut.toJSON()],flavor:'激进手术：割伤（原生加值被抑制时的兼容阶段）'});
        valid(activity,ctx);if(game.messages.get(surgery.id)!==surgery)throw Error('surgery-result-unpersisted');results.push(surgery);
      }
      const medicBonus=healer.items?.some?.(i=>i.type==='feat'&&(i.slug??i.system?.slug)==='medic-dedication')?({expert:5,master:10,legendary:15}[activity.options.rank]??0):0;
      const stages=results.map(message=>({message,patient,outcome:check.outcome,medicBonus,stage:classifyResult(message.rolls[0],check.outcome)}));
      if(stages.filter(r=>r.stage==='surgery').length!==((actualRisky||activity.options.riskySurgery)?1:0)||new Set(stages.map(r=>r.stage)).size!==stages.length)throw Error('native-stages-mismatch');
      stages.sort((a,b)=>(a.stage==='surgery'?-1:1)-(b.stage==='surgery'?-1:1));const receiptIds=[];
      for(const stage of stages){if(patient.isDead)throw Error('patient-died-during-treatment');receiptIds.push(await applySavedResult(activity,stage,ctx));if(stage.poolReceipt)proof.poolReceipts.push(stage.poolReceipt)}
      Object.assign(proof,{checkIds:[check.message.id],resultIds:results.map(m=>m.id),receiptIds});
      const expiresAt=cooldown({startedAt:activity.startedAt,finishedAt:activity.endsAt,continualRecovery:activity.options.continualRecovery}).expiresAt;
      if(expiresAt>game.time.worldTime){
        const template=await fromUuid(TREAT_WOUNDS_IMMUNITY);valid(activity,ctx);if(!template?.toObject)throw Error('native-immunity-template-unavailable');
        const data=template.toObject();delete data._id;data.system.duration={value:(expiresAt-game.time.worldTime)/60,unit:'minutes',expiry:'turn-start',sustained:false};data.system.start={value:game.time.worldTime,initiative:null};
        data.system.context={origin:{actor:activity.actorUUID,token:null,item:null,rollOptions:[]},target:{actor:patient.uuid,token:null},roll:null};data.flags={...data.flags,[MODULE_ID]:{exploration:{activityId:activity.id,expiresAt,kind:'immunity'}}};
        const saved=await patient.createEmbeddedDocuments('Item',[data]);valid(activity,ctx);if(saved.length!==1)throw Error('immunity-unconfirmed');proof.immunityIds.push(saved[0].uuid);
      }
      if(['success','criticalSuccess'].includes(check.outcome)&&patient.hasCondition?.('wounded')){await patient.decreaseCondition('wounded',{forceRemove:true});valid(activity,ctx);if(patient.hasCondition('wounded'))throw Error('wounded-removal-unconfirmed')}
      return {status:'confirmed',proof,sourceDegree:outcomes.indexOf(check.message.flags.pf2e.context.unadjustedOutcome??check.outcome),effectiveOutcome:check.outcome,rolledHealing:stages.find(r=>r.stage==='healing')?.message.rolls[0].total??null,medicBonus,expiresAt,resourceReceiptIds:[],patientUUID:patient.uuid};
    }catch(error){
      const checkId=check?.message?.id;error.proof={...proof,checkIds:checkId?[checkId]:[],resultIds:[...new Set([...proof.resultIds,...[...captured.values()].filter(m=>m.flags?.pf2e?.origin?.messageId===checkId).map(m=>m.id)])],receiptIds:[...game.messages.values()].filter(m=>m.flags?.pf2e?.context?.type==='damage-taken'&&m.flags.pf2e.context.options?.some(o=>o.startsWith(`${MODULE_ID}:exploration-apply:${activity.id}:`))).map(m=>m.id)};throw error;
    }finally{Hooks.off('preCreateChatMessage',pre);Hooks.off('createChatMessage',post)}
  }
  async function run(activity,ctx) {
    valid(activity,ctx);const completions=[];
    try{for(const uuid of activity.patientUUIDs){const patient=await fromUuid(uuid);valid(activity,ctx);completions.push(await single(activity,patient,ctx))}}
    catch(error){error.proof={useId:activity.id,...Object.fromEntries(['checkIds','resultIds','receiptIds','immunityIds','poolReceipts'].map(k=>[k,[...new Set([...completions.flatMap(c=>c.proof[k]??[]),...error.proof?.[k]??[]])]]))};error.results=completions;throw error}
    const first=completions[0];if(!first)throw Error('missing-patient');
    return {...first,results:completions,proof:{useId:activity.id,...Object.fromEntries(['checkIds','resultIds','receiptIds','immunityIds','poolReceipts'].map(k=>[k,completions.flatMap(c=>c.proof[k]??[])]))}};
  }
  async function extend(activity,original,ctx){
    valid(activity,ctx);if(original.state!=='confirmed'||activity.startedAt!==original.endsAt||activity.endsAt!==original.startedAt+3600)throw Error('extension-origin-unconfirmed');
    const receipts=[];
    for(const result of original.results??[original]){
      if(!['success','criticalSuccess'].includes(result.effectiveOutcome))continue;
      const message=result.proof.resultIds.map(id=>game.messages.get(id)).find(m=>m&&classifyResult(m.rolls[0],result.effectiveOutcome)==='healing');
      if(!message||message.rolls[0].total!==result.rolledHealing)throw Error('original-healing-roll-unconfirmed');
      const patient=await fromUuid(result.patientUUID??activity.patientUUIDs[0]);
      receipts.push(await applySavedResult(activity,{message,patient,stage:'healing',outcome:result.effectiveOutcome,medicBonus:result.medicBonus},ctx));
    }
    if(!receipts.length)throw Error('extension-without-success');
    return {status:'confirmed',proof:{useId:original.proof.useId,checkIds:original.proof.checkIds,resultIds:original.proof.resultIds,receiptIds:receipts,immunityIds:[]},effectiveOutcome:original.effectiveOutcome,rolledHealing:original.rolledHealing};
  }
  return {run,extend,applySavedResult,reconcile:async a=>({status:'uncertain',reason:'persisted-native-evidence-requires-review',proof:a.proof})};
}
