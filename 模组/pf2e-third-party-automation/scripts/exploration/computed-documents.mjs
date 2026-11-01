import {MODULE_ID} from './schema.mjs';
import {canonicalJSON} from './revision-codec.mjs';
import {TREAT_WOUNDS_IMMUNITY,values} from '../salubrious-kiss-rules.mjs';

const markerPath=`flags.${MODULE_ID}.computedRecoveryBatch`;
const marker=actor=>actor?.flags?.[MODULE_ID]?.computedRecoveryBatch;
const read=(actor,path)=>path.split('.').reduce((value,key)=>value?.[key],actor);
const same=(a,b)=>a===undefined||b===undefined?a===b:canonicalJSON(a)===canonicalJSON(b);
const itemId=()=>crypto.randomUUID().replaceAll('-','').slice(0,16);
export function computedActorStamp(actor){return JSON.stringify({modified:actor._stats?.modifiedTime,hp:actor.system?.attributes?.hp,focus:actor.system?.resources?.focus,canAct:actor.canAct,dead:actor.isDead,items:values(actor.items).map(item=>[item.id??item._id,item._stats?.modifiedTime,item.isExpired])})}

/** A persisted plan owns its samples. This adapter only applies those values. */
export function createComputedDocuments({game,fromUuid,Actor=globalThis.CONFIG?.Actor?.documentClass,getHpPool}) {
 const attemptedActors=new Set(),attemptedItems=new Set();
 async function prepare(batch) {
  const before=new Map(batch.beforeActors.map(actor=>[actor.actorUUID,actor])),rows=new Map(),items=[];
  const row=async uuid=>{
   if(!rows.has(uuid)){
    const actor=await fromUuid(uuid);if(!actor||actor.isOwner===false)throw Error('computed-actor-unavailable');
    rows.set(uuid,{uuid,id:actor.id,before:{},changes:{},stages:[]});
   }
   return rows.get(uuid);
  };
  for(const after of batch.afterActors){
   const previous=before.get(after.actorUUID);if(!previous||!after.pool?.ready||after.pool.poolUUID!==previous.pool.poolUUID)throw Error('computed-pool-changed');
   if(after.hp.value!==previous.hp.value){
    const target=await row(after.pool.poolUUID),path='system.attributes.hp.value';
    if(path in target.changes&&target.changes[path]!==after.hp.value)throw Error('computed-inconsistent-pool');
    target.before[path]=previous.hp.value;target.before['system.attributes.hp.max']=previous.hp.max;target.before['system.attributes.hp.temp']=previous.hp.temp??0;
    target.changes[path]=after.hp.value;
   }
   if(after.focus.value!==previous.focus.value){const target=await row(after.actorUUID);target.before['system.resources.focus.value']=previous.focus.value;target.before['system.resources.focus.max']=previous.focus.max;target.changes['system.resources.focus.value']=after.focus.value}
   const lastImmunity=batch.events.filter(event=>event.patientUUID===after.actorUUID&&Number.isFinite(event.cooldownExpiresAt)).at(-1);
   const additions=[];
   if(lastImmunity?.cooldownExpiresAt>batch.to){
    const source=await fromUuid(TREAT_WOUNDS_IMMUNITY);if(!source)throw Error('computed-immunity-source-unavailable');
    const data=source.toObject();data._id=itemId();
    data._stats={...data._stats,compendiumSource:TREAT_WOUNDS_IMMUNITY};
    data.system.duration={value:(lastImmunity.cooldownExpiresAt-batch.to)/60,unit:'minutes',expiry:'turn-start',sustained:false};
    data.system.start={value:batch.to,initiative:null};
    data.system.context={origin:{actor:lastImmunity.actorUUID,token:null,item:null,rollOptions:[]},target:{actor:after.actorUUID,token:null},roll:null};
    data.flags={...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],computedRecoveryBatch:batch.id,salubriousKiss:{kind:'immunity',nonce:batch.id,startedAt:lastImmunity.startedAt,expiresAt:lastImmunity.cooldownExpiresAt}}};
    additions.push(data);
   }
   const effects=batch.events.filter(event=>event.patientUUID===after.actorUUID).flatMap(event=>event.effects??[]).filter(effect=>effect.expiresAt>batch.to);
   for(const effect of effects){
    const source=await fromUuid(effect.sourceUUID);if(!source)throw Error('computed-effect-source-unavailable');
    const data=source.toObject();data._id=itemId();data.system.start={value:batch.to,initiative:null};
    data.system.duration={value:(effect.expiresAt-batch.to)/6,unit:'rounds',expiry:'turn-start',sustained:false};
    data.flags={...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],computedRecoveryBatch:batch.id}};additions.push(data);
   }
   if(additions.length||previous.wounded&&!after.wounded){await row(after.actorUUID);items.push({actorUUID:after.actorUUID,create:additions,removeWounded:previous.wounded&&!after.wounded,woundedBefore:previous.wounded})}
  }
  const damagedPools=new Set(batch.events.filter(event=>event.hpStages?.some(stage=>stage.damageTaken>0)).map(event=>event.poolUUID??before.get(event.patientUUID)?.pool.poolUUID));
  for(const pool of damagedPools){
   const target=await row(pool),actor=await fromUuid(pool),stages=batch.events.filter(event=>(event.poolUUID??before.get(event.patientUUID)?.pool.poolUUID)===pool).flatMap(event=>(event.hpStages??[]).map(stage=>({...stage,patientUUID:event.patientUUID})));
   target.stages=stages.filter(stage=>stage.before!==stage.after).map(stage=>({...stage}));
   for(const stage of target.stages){
    if(stage.after!==0||stage.damageTaken<=0)continue;
    const mode=game.settings?.get('pf2e','automation.actorsDeadAtZero');
    const instant=actor.type==='npc'&&actor.modeOfBeing==='undead'||stage.damageTaken>=actor.system.attributes.hp.max*2;
    if(!(actor.type==='npc'&&['npcsOnly','both'].includes(mode)||actor.type==='character'&&mode==='both'&&instant))continue;
    const patient=await fromUuid(stage.patientUUID),found=patient.getActiveTokens?.(false,true)?.[0],token=found?.document??found??patient.token;
    if(token?.combatant&&!token.combatant.isDefeated)stage.defeat={combatantUUID:token.combatant.uuid,mode};
   }
   if(target.stages.length){target.before['system.attributes.hp.value']??=target.stages[0].before;target.changes['system.attributes.hp.value']=target.stages.at(-1).after}
  }
  const actors=[...rows.values()].map(target=>({...target,changes:{...target.changes,[markerPath]:{id:batch.id,at:batch.to,userId:game.user.id,fields:JSON.stringify(target.changes)}}}));
  return {id:batch.id,at:batch.to,actors,items,pools:batch.beforeActors.map(actor=>({actorUUID:actor.actorUUID,poolUUID:actor.pool.poolUUID,...actor.executionStamp?{executionStamp:actor.executionStamp}:{}}))};
 }
 function atTime(plan){if(game.time.worldTime!==plan.at)throw Error('computed-world-time-changed')}
 async function validate(plan,{afterActors=false}={}){
  atTime(plan);
  for(const expected of plan.pools){const actor=await fromUuid(expected.actorUUID);if(!afterActors&&expected.executionStamp&&computedActorStamp(actor)!==expected.executionStamp)throw Error('computed-actor-changed');if(!getHpPool)continue;const pool=getHpPool(actor);if(!pool?.ready||pool.poolUUID!==expected.poolUUID)throw Error('computed-pool-changed')}
  for(const target of plan.actors){
   const actor=await fromUuid(target.uuid);if(!actor||actor.isOwner===false)throw Error('computed-actor-unavailable');
   if(!afterActors)for(const stage of target.stages)if(stage.defeat){
    const combatant=await fromUuid(stage.defeat.combatantUUID);
    if(!combatant||combatant.isDefeated||game.settings?.get('pf2e','automation.actorsDeadAtZero')!==stage.defeat.mode)throw Error('computed-combatant-changed');
   }
   const fields=afterActors?{...target.before,...target.changes}:target.before;
   for(const [path,value]of Object.entries(fields))if(!same(read(actor,path),value))throw Error('computed-actor-changed');
  }
  atTime(plan);return true;
 }
 async function applyActors(plan,{guard=()=>true}={}){
  if(attemptedActors.has(plan.id))throw Error('computed-batch-already-applied');
  await validate(plan);if(guard()!==true)throw Error('computed-driver-changed');
  for(const target of plan.actors)if(marker(await fromUuid(target.uuid))?.id===plan.id)throw Error('computed-batch-already-applied');
  attemptedActors.add(plan.id);
  const ordinary=plan.actors.filter(target=>!target.stages.length),world=ordinary.filter(target=>/^Actor\.[^.]+$/.test(target.uuid));
  if(world.length&&Actor?.updateDocuments)await Actor.updateDocuments(world.map(target=>({_id:target.id,...target.changes})),{render:false});
  else for(const target of world){if(guard()!==true)throw Error('computed-driver-changed');await (await fromUuid(target.uuid)).update(target.changes,{render:false})}
  for(const target of ordinary.filter(target=>!world.includes(target))){if(guard()!==true)throw Error('computed-driver-changed');await (await fromUuid(target.uuid)).update(target.changes,{render:false})}
  for(const target of plan.actors.filter(target=>target.stages.length)){
   const actor=await fromUuid(target.uuid);
   for(const [index,stage]of target.stages.entries()){
    if(guard()!==true)throw Error('computed-driver-changed');atTime(plan);
    if(actor.system.attributes.hp.value!==stage.before)throw Error('computed-actor-changed');
    const last=index===target.stages.length-1;
    await actor.update({...last&&!stage.defeat?target.changes:{[markerPath]:{id:plan.id,at:plan.at,stage:index}},'system.attributes.hp.value':stage.after},{render:false,damageTaken:stage.damageTaken,finePowder:false});
    if(stage.defeat){
     if(guard()!==true)throw Error('computed-driver-changed');atTime(plan);
     const combatant=await fromUuid(stage.defeat.combatantUUID);
     if(!combatant)throw Error('computed-combatant-changed');
     if(actor.isDead&&!combatant.isDefeated)await combatant.toggleDefeated({overlayIcon:true});
     if(last){if(guard()!==true)throw Error('computed-driver-changed');atTime(plan);await actor.update(target.changes,{render:false})}
    }
   }
  }
  await validate(plan,{afterActors:true});
  return {actors:plan.actors.map(target=>({actorUUID:target.uuid,batchId:plan.id,after:target.changes}))};
 }
 async function applyItems(plan,{guard=()=>true}={}){
  if(attemptedItems.has(plan.id))throw Error('computed-items-already-applied');
  await validate(plan,{afterActors:true});if(guard()!==true)throw Error('computed-driver-changed');attemptedItems.add(plan.id);
  const receipts=[];
  for(const target of plan.items){
   if(guard()!==true)throw Error('computed-driver-changed');atTime(plan);const actor=await fromUuid(target.actorUUID);
   if(target.create.some(row=>values(actor.items).some(item=>item.id===row._id||item._id===row._id)))throw Error('computed-items-already-applied');
   if(target.removeWounded&&actor.hasCondition('wounded')!==true)throw Error('computed-wounded-changed');
   if(target.create.length){const saved=await actor.createEmbeddedDocuments('Item',target.create,{keepId:true,render:false});if(saved?.length!==target.create.length)throw Error('computed-items-unconfirmed');receipts.push(...saved.map(item=>({actorUUID:actor.uuid,itemUUID:item.uuid,batchId:plan.id})))}
   if(target.removeWounded){await actor.decreaseCondition('wounded',{forceRemove:true});if(actor.hasCondition('wounded'))throw Error('computed-wounded-unconfirmed')}
  }
  return {items:receipts,wounded:plan.items.filter(row=>row.removeWounded).map(row=>row.actorUUID)};
 }
 async function inspect(plan){
  const actors=[];
  for(const target of plan.actors){
   const actor=await fromUuid(target.uuid),receipt=marker(actor);let fields;
   if(receipt?.id===plan.id&&typeof receipt.fields==='string')try{fields=JSON.parse(receipt.fields)}catch{}
   const expected=Object.fromEntries(Object.entries(target.changes).filter(([path])=>path!==markerPath));
   const confirmed=receipt?.id===plan.id&&fields!==undefined&&same(fields,expected)&&Object.entries(target.changes).every(([path,value])=>same(read(actor,path),value));
   actors.push({actorUUID:target.uuid,state:confirmed?'confirmed':'unknown',...receipt?.id===plan.id&&Number.isInteger(receipt.stage)?{stage:receipt.stage}:{}});
  }
  const items=[];for(const target of plan.items){const actor=await fromUuid(target.actorUUID);items.push({actorUUID:target.actorUUID,state:target.create.every(row=>values(actor?.items).some(item=>(item.id??item._id)===row._id&&item.flags?.[MODULE_ID]?.computedRecoveryBatch===plan.id))&&(!target.removeWounded||!actor?.hasCondition('wounded'))?'confirmed':'unknown'})}
  return {actors,items};
 }
 return {prepare,validate,applyActors,applyItems,inspect};
}
