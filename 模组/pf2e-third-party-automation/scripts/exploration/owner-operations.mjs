import {MODULE_ID,clone} from './schema.mjs';
import {isActiveGM} from './document-store.mjs';
export function createExplorationOwnerOperations({game,fromUuid,ledger,sharedOwnerOperations,timeoutMs=60000}) {
  const contexts=new WeakMap(),operations=new Map(),inflightActors=new Set(),cancelled=new Set();let socket;
  const authority=caller=>{if(caller!==game.users.activeGM?.id||!game.users.get(caller)?.isGM||!game.users.get(caller)?.active)throw Error('active-gm-required');if(game.combat?.started)throw Error('encounter-started')};
  async function ownerExecute(payload,callerId){
    authority(callerId);const key=payload?.activity?.actorUUID;if(typeof key!=='string')throw Error('actor-required');
    if(inflightActors.has(key))throw Error('owner-activity-already-in-flight');inflightActors.add(key);
    try{return await executeClaim(payload,callerId)}finally{inflightActors.delete(key)}
  }
  async function executeClaim(payload,callerId) {
    authority(callerId);const handler=operations.get(payload?.operationId);if(!handler)throw Error('unknown-native-operation');
    const activity=clone(payload.activity),actor=await fromUuid(activity?.actorUUID);authority(callerId);
    if(!actor?.testUserPermission(game.user,'OWNER'))throw Error('original-owner-required');
    if(activity.state!=='completing'||game.time.worldTime<activity.endsAt||activity.endsAt<activity.startedAt)throw Error('invalid-activity-time');
    const records=clone(actor.flags?.[MODULE_ID]?.explorationExecutions??{});
    if(records[activity.id])throw Error('activity-already-executed');
    const identity={activityId:activity.id,operationId:payload.operationId,gmId:callerId,userId:game.user.id,actorUUID:actor.uuid};
    records[activity.id]={...identity,state:'started'};
    await actor.update({[`flags.${MODULE_ID}.explorationExecutions`]:records});authority(callerId);
    const ctx=Object.freeze({actor,validate:()=>{authority(callerId);if(cancelled.has(activity.id))throw Error('activity-stopped');if(game.time.worldTime!==activity.endsAt)throw Error('external-world-time-change');if(!actor.testUserPermission(game.user,'OWNER'))throw Error('owner-changed')}});contexts.set(ctx,activity.id);
    try{
      ctx.validate();const result=await handler(activity,ctx);ctx.validate();
      const current=clone(actor.flags?.[MODULE_ID]?.explorationExecutions??{});
      if(current[activity.id]?.state!=='started')throw Error('owner-claim-changed');current[activity.id]={...identity,state:'done',result:clone(result)};
      await actor.update({[`flags.${MODULE_ID}.explorationExecutions`]:current});authority(callerId);return result;
    }catch(error){
      const current=clone(actor.flags?.[MODULE_ID]?.explorationExecutions??{});current[activity.id]={...identity,state:'uncertain',reason:String(error.message),...error.proof?{proof:clone(error.proof)}:{}};
      if(game.user.id===identity.userId)await actor.update({[`flags.${MODULE_ID}.explorationExecutions`]:current}).catch(()=>{});throw error;
    }finally{contexts.delete(ctx)}
  }
  async function runActivityWithOwner(activity,operationId) {
    if(!isActiveGM(game))throw Error('active-gm-required');
    const saved=await ledger.getActivity(activity.id);if(JSON.stringify(saved)!==JSON.stringify(activity)||saved.state!=='completing')throw Error('activity-claim-changed');
    if(sharedOwnerOperations?.runExploration)return sharedOwnerOperations.runExploration(activity,operationId);
    const actor=await fromUuid(activity.actorUUID);if(!isActiveGM(game))throw Error('gm-changed');
    // An active GM is a native OWNER; shared HP forwarding must stay on a client
    // that owns the master. Remote execution is used only for explicit ownerId.
    const ownerId=activity.source?.ownerId??game.user.id;
    if(ownerId===game.user.id)return ownerExecute({activity,operationId},game.user.id);
    if(!actor?.testUserPermission(game.users.get(ownerId),'OWNER')||!game.users.get(ownerId)?.active||!socket)throw Error('original-owner-offline');
    let timer;try{
      const response=await Promise.race([socket.executeAsUser('exploration:execute',ownerId,{activity,operationId}),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('owner-response-uncertain-no-retry')),timeoutMs)})]);
      if(!isActiveGM(game))throw Error('gm-changed');if(!response?.ok)throw Error(response?.error??'owner-response-uncertain-no-retry');return response.value;
    }finally{clearTimeout(timer)}
  }
  async function reconcile(activity){
    if(!isActiveGM(game))throw Error('active-gm-required');const actor=await fromUuid(activity.actorUUID),record=actor?.flags?.[MODULE_ID]?.explorationExecutions?.[activity.id],operation=activity.options?.extensionOf?'treatment-extension':activity.providerId;
    if(record?.state!=='done'||record.activityId!==activity.id||record.actorUUID!==activity.actorUUID||record.operationId!==operation||record.result?.status!=='confirmed')return {status:'uncertain',reason:'saved-owner-completion-unavailable'};
    const result=clone(record.result),proof=result.proof;if(!proof||proof.useId!==(activity.options?.extensionOf??activity.id))return {status:'uncertain',reason:'saved-owner-proof-mismatch'};
    for(const key of ['checkIds','resultIds'])if(!Array.isArray(proof[key])||proof[key].some(id=>!game.messages.get(id)))return {status:'uncertain',reason:'saved-native-message-unavailable'};
    if((proof.poolReceipts??[]).some(p=>!activity.hpPoolUUIDs.includes(p.actorUUID)||p.activityId&&p.activityId!==activity.id))return {status:'uncertain',reason:'saved-pool-proof-mismatch'};
    if(!isActiveGM(game))throw Error('active-gm-required');return result;
  }
  return {ownerExecute,runActivityWithOwner,reconcile,cancelActivity:activity=>cancelled.add(activity.id),isActivityContext:(ctx,id)=>contexts.get(ctx)===id&&!cancelled.has(id),
    createActivityContext:async activity=>{
      if(!isActiveGM(game))throw Error('active-gm-required');const stored=await ledger.getActivity(activity.id);
      if(stored?.state!=='planned'||JSON.stringify(stored)!==JSON.stringify(activity)||game.time.worldTime!==activity.startedAt)throw Error('activity-begin-claim-required');
      const ctx=Object.freeze({validate:()=>{if(!isActiveGM(game))throw Error('gm-changed')}});contexts.set(ctx,activity.id);return ctx;
    },
    registerOperation:(id,handler)=>{if(!/^[-a-z0-9]+$/.test(id)||operations.has(id)||typeof handler!=='function')throw Error('invalid-operation');operations.set(id,handler)},
    register:({socket:api})=>{if(socket)throw Error('duplicate-exploration-socket');socket=api;socket?.register('exploration:execute',async function(payload){try{return {ok:true,value:await ownerExecute(payload,this.socketdata?.userId)}}catch(e){return {ok:false,error:String(e.message)}}})}
  };
}
