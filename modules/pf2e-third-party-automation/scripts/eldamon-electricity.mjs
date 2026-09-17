import {SerialActions} from './runtime.mjs';
import {sourceUuid} from './metapower/rules.mjs';
import {genericReactionAvailable} from './reaction-budget.mjs';

export const ELECTRICITY_MODULE_ID='pf2e-third-party-automation';
export const ELECTRICITY_SOURCES=Object.freeze({
 charged:'Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR',
 shocked:'Compendium.battlezoo-eldamon-pf2e.conditions.Item.1fZbuJEbVmE3J4XL',
 shell:'Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.FBc2TxqWSRomT9fC',
 surge:'Compendium.battlezoo-eldamon-pf2e.powers.Item.veFrnrxYjlqca13w',
 anvil:'Compendium.battlezoo-eldamon-pf2e.powers.Item.hQOa1yaP9C6wajNn',
 static:'Compendium.battlezoo-eldamon-pf2e.powers.Item.KWQgx7RMeY3RKW6J',
 shot:'Compendium.battlezoo-eldamon-pf2e.powers.Item.QIYppaP0zcGvb5Bd',
 chain:'Compendium.battlezoo-eldamon-pf2e.powers.Item.fzV5Ly3a9nEsfcAJ',
});
const ID=ELECTRICITY_MODULE_ID,S=ELECTRICITY_SOURCES,copy=x=>structuredClone(x),values=c=>Array.from(c?.values?.()??c??[]);
export const ELECTRICITY_APPLY_PREFIX=`${ID}:electricity-apply:`;
export const ELECTRICITY_SOURCE_PREFIX=`${ID}:electricity-source:`;
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const tokenUuid=s=>s?.scene&&s?.token?`Scene.${s.scene}.Token.${s.token}`:null;
const normalized=s=>s?.replace(/(\.conditions)\.(?!Item\.)/,'$1.Item.');
export const electricityEffects=(actor,source)=>values(actor?.items).filter(i=>normalized(sourceUuid(i))===source);
export const electricityState=actor=>copy(actor?.flags?.[ID]?.electricity??{version:1,damage:{},operations:{},pendingShocks:{}});
const liveToken=t=>t?.documentName==='Token'&&t.actor&&t.parent?.tokens?.get(t.id)===t;
const ownCharge=i=>i?.flags?.[ID]?.electricityCharge?.ownPower===true;
const shell=actor=>values(actor.items).some(i=>sourceUuid(i)===S.shell);
const siphon=r=>r?.snapshot?.siphon?.applies===true;
const frame=combat=>combat?.started?`${combat.id}:${combat.round}:${combat.turn}`:null;
/** Viewed combat is UI state. Only one actual started membership can provide
 * initiative timing; absent or ambiguous membership has no automated clock. */
export function electricityEncounter(game,actorUuid,tokenUuid=null){
 const combats=game.combats?values(game.combats):game.combat?[game.combat]:[];
 const candidates=combats.filter(c=>c.started&&values(c.combatants??c.turns).some(t=>t.actor?.uuid===actorUuid&&(!tokenUuid||t.token?.uuid===tokenUuid)));
 return candidates.length===1?candidates[0]:null;
}

export function classifyElectricityDamage(roll){
 const instances=roll?.instances;if(!Array.isArray(instances))return 'none';
 const direct=instances.filter(i=>!i.persistent&&Number(i.total)>0);
 if(!direct.some(i=>i.type==='electricity'))return 'none';
 return direct.every(i=>i.type==='electricity')?'pure':'mixed';
}
export function electricityRemovalPlan({charged=0,ownPowerCharge=false,shell=false}={}){
 if(charged>0&&!(shell&&ownPowerCharge))return {charged:Math.max(0,charged-1),removeShocked:false};
 return {charged,removeShocked:true};
}
export function sourceTurnExpiry(combat,actorUuid,{rounds=1,phase='end',tokenUuid=null}={}){
 const index=combat?.turns?.findIndex(c=>c.actor?.uuid===actorUuid&&(!tokenUuid||c.token?.uuid===tokenUuid))??-1;
 if(!combat?.started||index<0)return null;
 return {combatId:combat.id,combatantId:combat.turns[index].id,round:combat.round+(index<=combat.turn?rounds:rounds-1),phase};
}
export function expiryReached(expiry,combat,combatant,phase){
 return !!expiry&&combat?.id===expiry.combatId&&combatant?.id===expiry.combatantId&&
  (combat.round>expiry.round||combat.round===expiry.round&&phase===expiry.phase);
}
export function chainEligibility({triggerDamage,sourceDistance,targetDistance,adjacentCaster,enemy,shocked,hitBySameEffect,reactionAvailable,discharge=false,siphoning=false}){
 return Number.isFinite(triggerDamage)&&triggerDamage>0&&Number.isFinite(sourceDistance)&&sourceDistance<=30&&Number.isFinite(targetDistance)&&targetDistance<=30&&
  enemy===true&&hitBySameEffect===false&&reactionAvailable===true&&(shocked===true||discharge===true&&!siphoning&&adjacentCaster===true);
}
export function receiptMatches(message,record){
 const pf=message?.flags?.pf2e,options=pf?.context?.options??[];
 return !!message?.id&&author(message)===record.userId&&message.speaker?.actor===record.actorUuid.split('.').at(-1)&&tokenUuid(message.speaker)===record.tokenUuid&&
  pf?.context?.type==='damage-taken'&&options.filter(o=>typeof o==='string'&&o.startsWith(ELECTRICITY_APPLY_PREFIX)).length===1&&options.includes(ELECTRICITY_APPLY_PREFIX+record.nonce)&&
  (pf.origin?.uuid??null)===record.sourceItemUuid&&(!pf.appliedDamage||pf.appliedDamage.uuid===record.actorUuid&&!pf.appliedDamage.isHealing&&!pf.appliedDamage.isReverted);
}
export function receiptElectricityAmount(message,record){
 if(record.kind!=='pure'||!receiptMatches(message,record))return null;
 const fact=message.flags?.[ID]?.electricityApplied;
 return fact?.nonce===record.nonce&&Number.isFinite(fact.amount)&&fact.amount>=0?fact.amount:null;
}
const fingerprint=m=>JSON.stringify({author:author(m),speaker:m.speaker,pf:m.flags?.pf2e,electricity:m.flags?.[ID]?.electricityApplied});
const sourceFingerprint=m=>JSON.stringify({pf:m.flags?.pf2e,source:m.flags?.[ID]?.electricitySource,rolls:m.rolls?.map(r=>r.toJSON?.()??{options:r.options,instances:r.instances})});

/** Active-GM ledger. Document effects retain their published rules and GrantItem
 * links. An interrupted mutation is never inferred from unrelated HP changes. */
export function createElectricityLedger({game,fromUuid,queue=new SerialActions(),reactionAvailable=genericReactionAvailable}={}){
 const gm=()=>{if(game.user?.id!==game.users.activeGM?.id)throw Error('Electricity lifecycle requires the active GM.');};
 const owner=(actor,user)=>{gm();if(!user||game.users.get(user.id)!==user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('Current actor owner permission is required.');};
 const save=async(actor,state)=>{gm();await actor.update({[`flags.${ID}.electricity`]:state});};
 const mutate=(actor,fn)=>queue.run(actor.uuid,async()=>{gm();return fn(electricityState(actor));});
 async function original(payload,user){
  const actor=await fromUuid(payload.actorUuid);owner(actor,user);
  const r=actor.flags?.[ID]?.metapower?.receipts?.[payload.nonce],m=await fromUuid(payload.messageUuid),item=await fromUuid(r?.itemUuid);
  if(!r||r.status!=='committed'||r.messageUuid!==m?.uuid||game.messages.get(m?.id)!==m||r.actorUuid!==actor.uuid||r.userId!==user.id||author(m)!==user.id||m.speaker?.actor!==actor.id||
   m.flags?.[ID]?.metapowerUse?.nonce!==r.nonce||m.flags?.pf2e?.origin?.uuid!==item?.uuid||item?.actor!==actor||actor.items.get(item.id)!==item||sourceUuid(item)!==r.sourceUuid)throw Error('Original committed electricity channel is required.');
  return {actor,receipt:r,message:m,item};
 }
 async function effectSource(source,flags,context){
  const template=await fromUuid(source);if(!template||template.type!=='effect')throw Error('Published electricity effect is unavailable.');
  const data=template.toObject();delete data._id;data._stats={...data._stats,compendiumSource:source};
  data.flags={...data.flags,[ID]:{...data.flags?.[ID],...flags}};data.system.context=context;return data;
 }
 async function charge(actor,state,key,{gain=false,clear=false,context=null}={}){
  let op=state.operations[key];if(op?.status==='done')return;
  const effects=electricityEffects(actor,S.charged).filter(i=>i.system.badge?.value>0);if(effects.length>1)throw Error('Multiple Charged parents require GM reconciliation.');
  let item=effects[0];
  if(!op){
   const before=item?.system.badge?.value??0,after=clear?0:gain?Math.min(3,before+1):Math.max(0,before-1);
   op=state.operations[key]={status:'started',itemId:item?.id??null,before,after,gain};await save(actor,state);
  }
  const proof=item?.flags?.[ID]?.electricityCharge;
  if(proof?.operation!==key){
   if(op.itemId&&(!item||item.id!==op.itemId||item.system.badge.value!==op.before))throw Error('Interrupted Charged mutation requires GM reconciliation.');
   if(!op.itemId&&item)throw Error('Charged changed while its gain was being committed.');
   if(op.after>0&&!item){
    const data=await effectSource(S.charged,{electricityCharge:{operation:key,ownPower:true}},context);data.system.badge.value=op.after;
    gm();await actor.createEmbeddedDocuments('Item',[data]);
   }else if(item&&op.before!==op.after){
    gm();await item.update({'system.badge.value':op.after,[`flags.${ID}.electricityCharge`]:{operation:key,ownPower:gain?(op.before===0||ownCharge(item)):ownCharge(item)}});
   }
  }
  op.status='done';await save(actor,state);
 }
 async function remove(actor,state,key,{onlyIds=null}={}){
  if(state.operations[key]?.status==='done')return;
  const charged=electricityEffects(actor,S.charged).find(i=>i.system.badge?.value>0),plan=electricityRemovalPlan({charged:charged?.system.badge?.value??0,ownPowerCharge:ownCharge(charged),shell:shell(actor)});
  if(charged&&plan.charged<charged.system.badge.value){await charge(actor,state,key);return;}
  const ids=electricityEffects(actor,S.shocked).filter(i=>!i.flags?.pf2e?.grantedBy?.id&&(!onlyIds||onlyIds.includes(i.id))).map(i=>i.id);
  state.operations[key]={status:'started',deleteIds:ids};await save(actor,state);
  if(ids.length){gm();await actor.deleteEmbeddedDocuments('Item',ids);}
  state.operations[key].status='done';await save(actor,state);
 }
 async function applyShock(actor,state,pending){
  const key=`shock:${pending.key}`;if(state.operations[key]?.status==='done')return;
  if(!electricityEffects(actor,S.shocked).some(i=>i.flags?.[ID]?.electricityShock?.key===pending.key)){
   const data=await effectSource(S.shocked,{electricityShock:copy(pending)},pending.context);
   // The original unlimited effect must not expire on the recipient's initiative.
   data.system.duration={value:-1,unit:'unlimited',expiry:null,sustained:false};
   gm();await actor.createEmbeddedDocuments('Item',[data]);
  }
  state.operations[key]={status:'done'};delete state.pendingShocks[pending.key];await save(actor,state);
 }
 async function validateSource(record){
  const message=await fromUuid(record.sourceMessageUuid),roll=message?.rolls?.[record.rollIndex];
  if(game.messages.get(message?.id)!==message||message.flags?.pf2e?.context?.type!=='damage-roll'||roll?.options?.[ID]?.electricitySource?.nonce!==record.sourceNonce||
   classifyElectricityDamage(roll)!==record.kind||message.flags?.[ID]?.electricitySource?.effectKey!==record.effectKey||record.sourceFingerprint&&sourceFingerprint(message)!==record.sourceFingerprint)throw Error('Electricity damage source changed or is not current.');
  return message;
 }
 async function verified(record){
  try{await validateSource(record);}catch{return false;}const m=await fromUuid(record.receiptUuid);
  return !!m&&game.messages.get(m.id)===m&&receiptMatches(m,record)&&fingerprint(m)===record.fingerprint;
 }
 async function chainFacts({actor,selection,kind},record){
  const evidence=selection.electricityEvidence,source=await fromUuid(record.tokenUuid),target=await fromUuid(evidence?.targetUuid),caster=await fromUuid(evidence?.sourceTokenUuid);
  const sourceCombat=electricityEncounter(game,source?.actor?.uuid,source?.uuid),casterCombat=electricityEncounter(game,actor.uuid,caster?.uuid);
  if(!liveToken(source)||!liveToken(target)||!liveToken(caster)||caster.actor.uuid!==actor.uuid||source.parent!==target.parent||target.parent!==caster.parent||source.uuid===target.uuid||
   !sourceCombat||casterCombat?.id!==sourceCombat.id||selection.targetUuids?.length!==1||selection.targetUuids[0]!==target.uuid||record.frame!==frame(sourceCombat)||record.status!=='confirmed'||record.receiptUuid!==evidence.receiptUuid||record.effectKey!==evidence.effectKey||!await verified(record))return false;
  const all=values(source.parent.tokens).flatMap(t=>Object.values(electricityState(t.actor).damage));
  const sourceMessage=await fromUuid(record.sourceMessageUuid),manifest=sourceMessage.flags?.[ID]?.electricitySource?.targetUuids??[];
  // Conservatively reserve the source's entire original target set while an area
  // application is in flight. A second target cannot sneak in between receipts.
  const hit=manifest.includes(target.uuid)||all.some(r=>r.effectKey===record.effectKey&&r.tokenUuid===target.uuid&&(r.status==='pending'||r.electricityAmount>0));
  return chainEligibility({triggerDamage:record.electricityAmount,sourceDistance:caster.object?.distanceTo?.(source.object),targetDistance:source.object?.distanceTo?.(target.object),
   adjacentCaster:caster.object?.distanceTo?.(target.object)<=5,enemy:!!actor.alliance&&!!target.actor.alliance&&actor.alliance!==target.actor.alliance,
   shocked:electricityEffects(target.actor,S.shocked).length>0,hitBySameEffect:hit,reactionAvailable:reactionAvailable(actor,{combat:casterCombat,modules:game.modules,messages:game.messages,users:game.users}),discharge:selection.discharge,siphoning:kind==='siphoning'});
 }
 return {
  async channel(payload,user){const {actor,receipt:r,message,item}=await original(payload,user);
   if(siphon(r)||r.selection?.discharge||![S.surge,S.anvil,S.static,S.shot].includes(r.sourceUuid))return;
   return mutate(actor,state=>charge(actor,state,`channel:${r.nonce}`,{gain:true,context:{origin:{actor:actor.uuid,token:tokenUuid(message.speaker),item:item.uuid,spellcasting:null,rollOptions:[]},target:null,roll:null}}));
  },
  async beginDamage(payload,user){
   const actor=await fromUuid(payload.actorUuid),token=await fromUuid(payload.tokenUuid);owner(actor,user);
   if(!liveToken(token)||token.actor.uuid!==actor.uuid||!/^[A-Za-z0-9_-]{8,100}$/.test(payload.nonce??'')||!['pure','mixed'].includes(payload.kind))throw Error('Invalid native electricity application.');
   const source=await validateSource(payload);
   return mutate(actor,async state=>{if(state.damage[payload.nonce])throw Error('Electricity application nonce was already used.');
    const r={...copy(payload),sourceFingerprint:sourceFingerprint(source),userId:user.id,status:'pending',frame:frame(electricityEncounter(game,actor.uuid,token.uuid))};state.damage[r.nonce]=r;await save(actor,state);return copy(r);});
  },
  async finishDamage(payload,user){
   const actor=await fromUuid(payload.actorUuid);owner(actor,user);
   return mutate(actor,async state=>{
    const r=state.damage[payload.nonce];if(!r||r.userId!==user.id)throw Error('Electricity application ownership mismatch.');
    if(r.status==='confirmed'&&r.lifecycleDone)return copy(r);
    const m=await fromUuid(payload.receiptUuid);await validateSource(r);
    if(game.messages.get(m?.id)!==m||!receiptMatches(m,r)||values(game.messages).filter(x=>receiptMatches(x,r)).length!==1)throw Error('Exactly one authentic native electricity receipt is required.');
    r.receiptUuid=m.uuid;r.fingerprint=fingerprint(m);r.electricityAmount=r.attribution?.amount??receiptElectricityAmount(m,r);r.status=r.electricityAmount===null?'needs-attribution':'confirmed';
    await save(actor,state);
    if(r.electricityAmount>0)await remove(actor,state,`damage:${r.nonce}`);
    for(const pending of Object.values(state.pendingShocks))if(pending.effectKey===r.effectKey)await applyShock(actor,state,pending);
    r.lifecycleDone=true;await save(actor,state);
    return copy(r);
   });
  },
  async confirmMixed(payload,user){
   gm();if(user!==game.users.activeGM)throw Error('Only the active GM can attribute mixed electricity damage.');
   const actor=await fromUuid(payload.actorUuid);
   return mutate(actor,async state=>{const r=state.damage[payload.nonce];
    if(!r||r.kind!=='mixed'||r.status!=='needs-attribution'||!await verified(r)||payload.receiptUuid!==r.receiptUuid||payload.confirmed!==true||!Number.isFinite(payload.amount)||payload.amount<0)throw Error('Confirm an exact electricity amount for this recorded mixed receipt.');
    const native=(await fromUuid(r.receiptUuid)).flags?.[ID]?.electricityApplied;
    if(native?.nonce!==r.nonce||!Number.isFinite(native.amount)||payload.amount>native.amount)throw Error('Electricity amount cannot exceed the actual native total.');
    r.electricityAmount=payload.amount;r.status='confirmed';r.lifecycleDone=false;r.attribution={userId:user.id,amount:payload.amount};await save(actor,state);
    if(r.electricityAmount>0)await remove(actor,state,`damage:${r.nonce}`);r.lifecycleDone=true;await save(actor,state);return copy(r);
   });
  },
  async check(message){
   gm();const c=message?.flags?.pf2e?.context;if(game.messages.get(message?.id)!==message||!message.isCheckRoll||!message.rolls?.[0]?._evaluated)return;
   const markers=(c?.options??[]).filter(o=>o.startsWith(`${ID}:metapower:`));if(markers.length!==1)return;
   const [cardId,nonce]=markers[0].slice(`${ID}:metapower:`.length).split(':'),card=game.messages.get(cardId),sourceActor=await fromUuid(card?.flags?.[ID]?.metapowerUse?.actorUuid);
   const r=sourceActor?.flags?.[ID]?.metapower?.receipts?.[nonce];
   if(!r||r.status!=='committed'||r.messageUuid!==card?.uuid||message.flags?.pf2e?.origin?.uuid!==r.itemUuid||siphon(r))return;
   const anvil=r.sourceUuid===S.anvil&&c.type==='saving-throw',staticShock=r.sourceUuid===S.static&&c.type==='attack-roll';if(!anvil&&!staticShock)return;
   const target=await fromUuid(anvil?tokenUuid(message.speaker):c.target?.token),source=await fromUuid(tokenUuid(card.speaker));
   if(!liveToken(target)||!liveToken(source)||target.parent!==source.parent||!r.selection?.targetUuids?.includes(target.uuid))return;
   if(anvil&&(!sourceActor.alliance||!target.actor.alliance||sourceActor.alliance===target.actor.alliance))return;
   const outcome=c.outcome,qualifies=anvil?['failure','criticalFailure'].includes(outcome):['criticalSuccess','success','failure'].includes(outcome);
   const effectKey=`channel:${sourceActor.uuid}:${nonce}`,key=`${effectKey}:${target.uuid}`;
   return mutate(target.actor,async state=>{
    if(!qualifies){delete state.pendingShocks[key];await save(target.actor,state);return;}
    const combat=electricityEncounter(game,sourceActor.uuid,source.uuid),expires=sourceTurnExpiry(combat,sourceActor.uuid,{rounds:staticShock&&outcome==='criticalSuccess'?2:1,tokenUuid:source.uuid});
    if(!expires)return; // No invented initiative clock outside an encounter.
    const pending={key,effectKey,sourceActorUuid:sourceActor.uuid,checkUuid:message.uuid,expires,context:{origin:{actor:sourceActor.uuid,token:source.uuid,item:r.itemUuid,spellcasting:null,rollOptions:[]},target:{actor:target.actor.uuid,token:target.uuid},roll:{total:message.rolls[0].total,degreeOfSuccess:message.rolls[0].degreeOfSuccess}}};
    state.pendingShocks[key]=pending;await save(target.actor,state);
    if(Object.values(state.damage).some(d=>d.effectKey===effectKey&&d.receiptUuid))await applyShock(target.actor,state,pending);
   });
  },
  async validateSelection({actor,item,selection,kind,user}){
   if(sourceUuid(item)!==S.chain)return;owner(actor,user);
   const evidence=selection.electricityEvidence,recipient=await fromUuid(evidence?.actorUuid),r=electricityState(recipient).damage[evidence?.nonce];
   if(!r||!await chainFacts({actor,selection,kind},r)||selection.triggerDamage!==r.electricityAmount)throw Error('Reactive Chain requires a current source-bound electricity receipt and eligible target.');
  },
  async candidates(payload,user){
   const actor=await fromUuid(payload.actorUuid);owner(actor,user);const caster=await fromUuid(payload.sourceTokenUuid);if(!liveToken(caster)||caster.actor.uuid!==actor.uuid)return [];
   const results=[];for(const token of values(caster.parent.tokens))for(const r of Object.values(electricityState(token.actor).damage)){
    const evidence={actorUuid:token.actor.uuid,nonce:r.nonce,receiptUuid:r.receiptUuid,effectKey:r.effectKey,targetUuid:payload.selection?.targetUuids?.[0],sourceTokenUuid:caster.uuid};
    const selection={...payload.selection,electricityEvidence:evidence};if(await chainFacts({actor,selection,kind:payload.kind},r))results.push({evidence,amount:r.electricityAmount,label:`${token.name}: ${r.electricityAmount}`});
   }return results;
  },
  async expire({combat,combatant,phase,ended=false,actors=[]}){
   gm();for(const actor of actors)await mutate(actor,async state=>{
    const expired=electricityEffects(actor,S.shocked).filter(i=>{const e=i.flags?.[ID]?.electricityShock?.expires;return ended?e?.combatId===combat.id:expiryReached(e,combat,combatant,phase);});
    for(const item of expired){const key=`expiry:${item.id}`;await remove(actor,state,key,{onlyIds:[item.id]});if(actor.items.get(item.id)&&!item.flags?.pf2e?.grantedBy?.id)await actor.deleteEmbeddedDocuments('Item',[item.id]);}
    for(const [key,p]of Object.entries(state.pendingShocks))if(ended?p.expires?.combatId===combat.id:expiryReached(p.expires,combat,combatant,phase))delete state.pendingShocks[key];
    if(ended&&values(combat.combatants).some(c=>c.actor?.uuid===actor.uuid))await charge(actor,state,`encounter:${combat.id}:charge`,{clear:true});await save(actor,state);
   });
  },
  async refresh({actor,nonce}){gm();if(values(game.combats).some(c=>c.started&&values(c.combatants).some(x=>x.actor?.uuid===actor.uuid)))return;
   return mutate(actor,state=>charge(actor,state,`refresh:${nonce}`,{clear:true}));
  },
  async interact(payload,user){const actor=await fromUuid(payload.actorUuid);owner(actor,user);if(payload.confirmed!==true||!payload.nonce)throw Error('Confirm the completed Interact action.');return mutate(actor,state=>remove(actor,state,`interact:${payload.nonce}`));},
 };
}
