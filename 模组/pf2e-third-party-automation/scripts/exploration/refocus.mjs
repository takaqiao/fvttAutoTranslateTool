import {MODULE_ID} from './schema.mjs';
import {sourceId,values} from '../salubrious-kiss-rules.mjs';
export const LAY_ON_HANDS='Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS';
export function healingVariant(item){
  if(sourceId(item)!==LAY_ON_HANDS)return null;
  const ids=Object.keys(item.system?.overlays??{});
  const variants=ids.map(id=>item.loadVariant?.({overlayIds:[id],castRank:item.rank})).filter(v=>v&&[...v.damageKinds??[]].length===1&&v.damageKinds.has('healing'));
  return variants.length===1?variants[0]:null;
}
export function focusFinishSatisfied(goal,actor) {return !goal.requireFullFocus||actor.focus.value>=actor.focus.max}
export function createRefocusAdapter({game,canvas,fromUuid,ownerOperations,timeoutMs=15000}) {
  const scopes=new Map();
  function getCurrent(actor) {const scope=scopes.get(actor.uuid);if(!scope)return null;scope.ctx.validate?.();return ownerOperations.isActivityContext(scope.ctx,scope.activity.id)?scope.activity:null}
  function capture(event) {
    const scope=scopes.get(event.actor.uuid);if(!scope)return;
    if(event.proof.nonce!==scope.activity.id||event.proof.startedAt!==scope.activity.startedAt||!Number.isFinite(event.proof.before)||event.proof.after<event.proof.before)throw Error('refocus-receipt-unconfirmed');
    scope.receipt={id:event.proof.nonce,focusBefore:event.proof.before,focusAfter:event.proof.after};scope.resolve(scope.receipt);
  }
  async function complete(activity,ctx) {
    if(!ownerOperations.isActivityContext(ctx,activity.id)||game.time.worldTime<activity.endsAt)throw Error('private-refocus-context-required');ctx.validate?.();
    const actor=await fromUuid(activity.actorUUID),controlled=canvas.tokens?.controlled??[];ctx.validate?.();
    let restore=null;
    if(controlled.length!==1||controlled[0].actor!==actor){
      const token=actor.getActiveTokens?.().find(t=>t.scene?.id===canvas.scene?.id||t.document?.parent?.id===canvas.scene?.id);
      if(!token?.control)throw Error('refocus-controlled-actor-mismatch');
      const previous=[...controlled];token.control({releaseOthers:true});restore=()=>{token.release();for(const t of previous)t.control({releaseOthers:false})};
      if(canvas.tokens.controlled?.length!==1||canvas.tokens.controlled[0].actor!==actor){restore();throw Error('refocus-controlled-actor-mismatch')}
    }
    if(typeof game.PF2eWorkbench?.refocus!=='function')throw Error('native-refocus-unavailable');
    if(scopes.has(actor.uuid))throw Error('refocus-already-running');
    let resolve,reject,timer;const signal=new Promise((r,j)=>{resolve=r;reject=j});signal.catch(()=>{});
    const scope={activity,ctx,resolve,reject};scopes.set(actor.uuid,scope);
    try{timer=setTimeout(()=>reject(Error('refocus-evidence-uncertain-no-retry')),timeoutMs);await game.PF2eWorkbench.refocus([actor]);const receipt=await signal;ctx.validate?.();return receipt}
    finally{clearTimeout(timer);scopes.delete(actor.uuid);restore?.()}
  }
  return {complete,getCurrent,capture};
}
export function createRefocusProvider({game,ledger,capabilities,refocusEvents,salubriousKiss,ownerOperations}) {
  const completed=new Map(),running=new Map();
  async function begin(activity,ctx) {
    const actor=await capabilities.discover(activity.actorUUID);
    if(actor.isDead||actor.unconscious)return {status:'blocked',reason:'actor-cannot-refocus'};
    if(activity.options.threePecks){if(!actor.threePecks)return {status:'blocked',reason:'three-pecks-unavailable'};return salubriousKiss.claimActivity(activity,ctx)}
    if(actor.focus.value>=actor.focus.max)return {status:'blocked',reason:'focus-already-full'};
    return {status:'started'};
  }
  async function complete(activity,ctx) {
    if(!ownerOperations.isActivityContext(ctx,activity.id))throw Error('private-refocus-context-required');
    if(completed.has(activity.id))return completed.get(activity.id);if(running.has(activity.id))return running.get(activity.id);
    const task=(async()=>{
      const stored=await ledger.getActivity(activity.id);if(!['started','completing'].includes(stored?.state))return {status:'uncertain',reason:'refocus-activity-in-flight'};
      const receipt=await refocusEvents.complete(activity,ctx);ctx.validate?.();
      const healing=activity.options.threePecks?await salubriousKiss.completeActivity(activity,ctx):null;
      const result={status:healing?.status??'confirmed',proof:{useId:activity.id,checkIds:healing?.proof?.checkIds??[],resultIds:healing?.proof?.resultIds??[],receiptIds:[receipt.id,...healing?.proof?.receiptIds??[]],immunityIds:healing?.proof?.immunityIds??[]},focusBefore:receipt.focusBefore,focusAfter:receipt.focusAfter,...healing?{treatment:healing}:{}};
      completed.set(activity.id,result);return result;
    })();running.set(activity.id,task);try{return await task}finally{running.delete(activity.id)}
  }
  return {id:'refocus',describe:capabilities.discover,begin,complete,propose:async()=>[],observe:async()=>null,reconcile:async activity=>completed.get(activity.id)??{status:'uncertain',reason:'refocus-context-lost'}};
}
/** Only a verified healing overlay is enrolled. Native consume remains the
 * single payer, including its exact atomic focus commit and cast-card binding. */
export function createFocusHealingProvider({game,Hooks,fromUuid,ownerOperations,castEvents,nativeTreatment}) {
  const scopes=new Map(),attempted=new Set();let registered=false;
  const valid=(activity,ctx)=>{if(!ownerOperations.isActivityContext(ctx,activity.id)||game.time.worldTime<activity.endsAt)throw Error('private-focus-healing-context-required');ctx.validate?.()};
  function register(){
    if(registered)return;registered=true;
    castEvents.addMatcher(item=>sourceId(item)===LAY_ON_HANDS);
    castEvents.addCapture('explorationFocus',item=>{const scope=scopes.get(item.uuid);return scope?{activityId:scope.activity.id}:undefined});
    castEvents.addConsumePolicy(async(context,next)=>{
      const scope=scopes.get(context.item.uuid);if(!scope)return next();valid(scope.activity,scope.ctx);
      if(context.actor.uuid!==scope.activity.actorUUID||sourceId(context.item)!==LAY_ON_HANDS||context.payload.focusPoints!==1||!healingVariant(scope.original))throw Error('native-focus-payment-source-changed');
      context.expectFocusCommit({before:context.actor.system.resources.focus.value,cost:1,changes:proof=>({[`flags.${MODULE_ID}.explorationFocusCommits.${scope.activity.id}`]:{...proof,activityId:scope.activity.id}})});
      const paid=await next();valid(scope.activity,scope.ctx);return paid;
    });
  }
  async function begin(activity){
    const actor=await fromUuid(activity.actorUUID),item=await fromUuid(activity.options.itemUUID);
    const patients=await Promise.all(activity.patientUUIDs.map(fromUuid));
    if(actor?.isDead||actor?.hasCondition?.('unconscious')||actor?.system?.resources?.focus?.value<1||item?.actor!==actor||!healingVariant(item)||patients.length!==1||patients[0]?.modeOfBeing!=='living')return {status:'blocked',reason:'focus-healing-unqualified'};
    return {status:'started'};
  }
  async function complete(activity,ctx){
    valid(activity,ctx);if(attempted.has(activity.id))throw Error('focus-healing-already-started');
    const original=await fromUuid(activity.options.itemUUID),variant=healingVariant(original);valid(activity,ctx);
    if(!variant||variant.actor.uuid!==activity.actorUUID||variant.system.cast.focusPoints!==1)throw Error('healing-overlay-unavailable');
    const patient=await fromUuid(activity.patientUUIDs[0]);if(patient.modeOfBeing!=='living')throw Error('living-patient-required');
    const actor=variant.actor;let card,damage,phase='cast';const scope={activity,ctx,original};
    const pre=Hooks.on('preCreateChatMessage',(message,data)=>{
      if(phase!=='roll'||(data.flags??message.flags)?.pf2e?.origin?.uuid!==variant.uuid)return;
      const flags=data.flags??=message.flags;flags.pf2e.context??={};flags.pf2e.context.options=[...(flags.pf2e.context.options??[]),'skip-handling-message',`exploration-activity:${activity.id}`];flags.pf2e.suppressDamageButtons=true;flags.pf2e.origin.messageId=card.id;
      flags[MODULE_ID]={...flags[MODULE_ID],exploration:{activityId:activity.id,patientUUID:patient.uuid}};data.blind=!!card.blind;data.whisper=[...(card.whisper??[])];message.updateSource?.({flags,blind:data.blind,whisper:data.whisper});
    });
    const post=Hooks.on('createChatMessage',message=>{
      if(phase==='cast'&&message.flags?.[MODULE_ID]?.explorationFocus?.activityId===activity.id){if(card)throw Error('duplicate-native-cast-card');card=message}
      if(phase==='roll'&&message.flags?.[MODULE_ID]?.exploration?.activityId===activity.id){if(damage)throw Error('duplicate-native-focus-roll');damage=message}
    });
    scopes.set(original.uuid,scope);attempted.add(activity.id);
    try{
      await variant.spellcasting.cast(variant,{consume:true,message:true,rank:variant.rank});valid(activity,ctx);
      // The cast may return void; resolve only the current atomic nonce's card.
      const commit=actor.flags?.[MODULE_ID]?.explorationFocusCommits?.[activity.id];
      card??=values(game.messages).find(m=>m.flags?.[MODULE_ID]?.explorationFocus?.activityId===activity.id&&m.flags?.[MODULE_ID]?.nativeCast?.id===commit?.castNonce);
      if(!commit||commit.cost!==1||commit.after!==commit.before-1||!card||game.messages.get(card.id)!==card||card.flags[MODULE_ID].nativeCast?.id!==commit.castNonce)throw Error('native-focus-cast-unconfirmed');
      const resourceReceipt=await castEvents.ensurePaid({actor,item:variant,message:card,user:game.user});valid(activity,ctx);
      if(resourceReceipt.state!=='used'||resourceReceipt.id!==commit.castNonce||resourceReceipt.messageId!==card.id)throw Error('native-focus-payment-unconfirmed');
      phase='roll';
      const originalGetDamage=variant.getDamage;
      if(originalGetDamage)variant.getDamage=async function(options){
        if(this!==variant)throw Error('private-healing-variant-changed');valid(activity,ctx);
        const data=await originalGetDamage.call(this,{...options,skipDialog:true});valid(activity,ctx);
        if(!data?.context?.options?.add)throw Error('native-healing-context-unavailable');
        data.context.options.add(`exploration-activity:${activity.id}`);data.context.options.add('skip-handling-message');data.context.skipDialog=true;
        data.context.messageMode=card.blind?'blind':card.whisper?.length?'gm':'public';return data;
      };
      const event=globalThis.MouseEvent?new MouseEvent('click',{shiftKey:!!game.user.settings.showCheckDialogs}):{target:null,shiftKey:false};
      const roll=await variant.rollDamage(event);valid(activity,ctx);
      if(!damage||game.messages.get(damage.id)!==damage||!roll?._evaluated||JSON.stringify(roll.toJSON())!==JSON.stringify(damage.rolls[0].toJSON()))throw Error('native-focus-roll-unconfirmed');
      const receipt=await nativeTreatment.applySavedResult(activity,{message:damage,patient,stage:'healing',outcome:null,item:variant},ctx);valid(activity,ctx);
      return {status:'confirmed',proof:{useId:activity.id,checkIds:[],resultIds:[card.id,damage.id],receiptIds:[receipt],immunityIds:[]},resourceReceiptIds:[resourceReceipt.id],rolledHealing:roll.total};
    }finally{scopes.delete(original.uuid);Hooks.off('preCreateChatMessage',pre);Hooks.off('createChatMessage',post)}
  }
  return {id:'focus-healing',begin,complete,register,describe:async()=>[],propose:async()=>[],observe:async()=>null,reconcile:async a=>({status:'uncertain',reason:'native-focus-context-lost',proof:a.proof})};
}
