import {MODULE_ID,hasSource} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM} from './native-context.mjs';
import {reactionEpoch} from './reaction-budget.mjs';

export const DISARMING_BLOCK_SOURCE='Compendium.pf2e.feats-srd.Item.dSSwRyuhKTq1VubX';
const TITAN='Compendium.pf2e.feats-srd.Item.KxaYlC50zzHysJj8',BONUS='Compendium.pf2e.other-effects.Item.EpvyTaklBQAOr1eT',OFF_GUARD='Compendium.pf2e.conditionitems.Item.AJh5ex99aV6VTggg';
const values=c=>Array.from(c?.values?.()??c??[]),own=d=>d?.flags?.[MODULE_ID]?.disarmingBlock??{},item=(a,id)=>values(a?.items).find(i=>i.id===id);
const feature=a=>values(a?.items).find(i=>hasSource(i,DISARMING_BLOCK_SOURCE));
const held=w=>w?.type==='weapon'&&w.system?.equipped?.carryType==='held'&&w.system.equipped.handsHeld>0;
const legalWeapon=w=>held(w)&&w.system.category!=='unarmed'&&!w.system.traits?.value?.includes('free-hand');
const acting=a=>a?.canAct!==false&&a?.isDead!==true&&!['unconscious','paralyzed'].some(c=>a?.hasCondition?.(c));
const marker=nonce=>`${MODULE_ID}:disarming-block:${nonce}`;
const terminal=new Set(['done','declined','cancelled','ineligible']);
export const disarmWeaponOption=weapon=>`${MODULE_ID}:disarm-weapon:${typeof weapon==='string'?weapon:weapon.uuid}`;
export const getWeakenedGrasp=(actor,weaponUuid)=>values(actor?.items).find(i=>i.type==='effect'&&own(i).kind==='weakened-grasp'&&own(i).weaponUuid===weaponUuid&&i.isExpired!==true)??null;
const attackIds=(actor,weapon)=>actor.type==='npc'?values(actor.items).filter(i=>i.type==='melee'&&i.flags?.pf2e?.linkedWeapon===weapon.id).map(i=>i.id).sort():[weapon.id];
const ownTurn=(a,g)=>!!g.combat?.started&&g.combat.turns?.[g.combat.turn]?.actor?.uuid===a.uuid;
const turnKey=(a,g)=>ownTurn(a,g)?`${g.combat.id}:${g.combat.round}:${g.combat.turn}`:null;
const size=a=>{const s=a?.size?.value??a?.size??a?.system?.traits?.size?.value;return ['tiny','sm','med','lg','huge','grg'].indexOf(s)};
const rank=a=>a?.getStatistic?.('athletics')?.rank??a?.skills?.athletics?.rank??0;
function performingWeapon(actor,game){
 const disarm=w=>w?.traits?.has?.('disarm')??w?.system?.traits?.value?.includes('disarm');
 // PF2e 8.5 ActionMacroHelpers: keep prepared Strike order, highest native
 // potency, first on ties. ABP gives every candidate the same attack potency.
 const candidates=actor.type==='character'&&Array.isArray(actor.system?.actions)?actor.system.actions.filter(s=>s.ready&&disarm(s.item)).map(s=>s.item):values(actor.items).filter(w=>w.type==='weapon'&&(w.isEquipped??held(w))&&disarm(w));
 const abp=!!actor.flags?.pf2e&&game.pf2e.settings?.variants?.abp!==undefined&&game.pf2e.settings.variants.abp!=='noABP'&&!actor.flags.pf2e.disableABP;
 const potency=w=>abp?actor.synthetics?.weaponPotency?.['strike-attack-roll']?.[0]?.bonus??0:w.system.runes?.potency??0;
 return candidates.reduce((best,w)=>!best||potency(w)>potency(best)?w:best,null);
}
const effect=(name,kind,context,rules,duration={value:-1,unit:'unlimited',expiry:null,sustained:false})=>({name,type:'effect',img:'icons/skills/melee/sword-damaged-broken-glow-red.webp',system:{duration,rules,context,tokenIcon:{show:true}},flags:{[MODULE_ID]:{disarmingBlock:{kind}}}});

/** A confirmed observer event is authoritative only through the injected validator. */
export function createDisarmingBlock({game,canvas=globalThis.canvas,fromUuid=globalThis.fromUuid,choose,validateConfirmed,onError=console.error}={}){
 const queue=new SerialActions(),effectQueue=new SerialActions(),rolling=new Set(),activeOffers=new Set(),tracked=new Map();let socket;
 const gm=()=>{if(!isActiveGM(game))throw Error('卸武格挡必须由当前主GM结算。')};
 const receipt=(actor,nonce)=>own(actor).uses?.find(r=>r.nonce===nonce);
 const save=async(actor,record)=>{gm();const records=[...(own(actor).uses??[]).filter(r=>r.nonce!==record.nonce),structuredClone(record)],recent=new Set(records.filter(r=>terminal.has(r.status)).slice(-64));await actor.update({[`flags.${MODULE_ID}.disarmingBlock.uses`]:records.filter(r=>!terminal.has(r.status)||recent.has(r))});tracked.set(actor.uuid,actor);};
 const currentTrigger=(actor,e)=>{if(e.epoch!=null&&e.epoch!==reactionEpoch(actor,game))throw Error('原盾牌格挡的回合已经结束，旧触发不能再次使用。')};
 async function putEffect(actor,key,data){
  gm();const existing=values(actor.items).filter(i=>i.type==='effect'&&i.flags?.[MODULE_ID]?.nativeEffectKey===key),next=structuredClone(data);next.flags[MODULE_ID].nativeEffectKey=key;
  if(!existing.length)return (await actor.createEmbeddedDocuments('Item',[next]))[0];
  await existing[0].update(next);gm();if(existing.length>1)await actor.deleteEmbeddedDocuments('Item',existing.slice(1).map(i=>i.id));return existing[0];
 }
 async function documents(e){
  const [actor,token,target,targetToken,attack,weapon]=await Promise.all([e.actorUuid,e.tokenUuid,e.attackerActorUuid,e.attackerTokenUuid,e.attackItemUuid,e.weaponUuid].map(u=>fromUuid(u)));
  if(!actor||!target||actor.uuid===target.uuid||token?.actor?.uuid!==actor.uuid||targetToken?.actor?.uuid!==target.uuid||token.parent?.id!==targetToken.parent?.id||attack?.actor?.uuid!==target.uuid||weapon?.actor?.uuid!==target.uuid)throw Error('卸武格挡的角色、Token或武器来源不一致。');
  return {actor,token,target,targetToken,attack,weapon};
 }
 function requirements(d,{ordinary=false}={}){
  const {actor,token,target,targetToken,attack,weapon}=d;
  const statistic=actor.getStatistic?.('athletics')??actor.skills?.athletics;
  if(!ordinary&&!feature(actor)||!acting(actor)||actor.type==='character'&&rank(actor)<1||!statistic||statistic.proficient===false)throw Error('缴械需要对应专长、运动受训且能够行动。');
  const performing=performingWeapon(actor,game);
  if(ordinary&&!(actor.handsFree>0)&&!performing)throw Error('普通缴械需要空手或持握具有缴械特征的武器。');
  if(!legalWeapon(weapon)||attack.type==='melee'&&attack.flags?.pf2e?.linkedWeapon!==weapon.id||attack.type!=='melee'&&attack.uuid!==weapon.uuid)throw Error('本次攻击武器已不再持握或不能被缴械。');
  const a=size(actor),b=size(target),titan=values(actor.items).some(i=>hasSource(i,TITAN)),maximum=titan?(rank(actor)>=4?3:2):1;
  if(a<0||b<0||b-a>maximum)throw Error('目标体型超过当前缴械范围。');
  const distance=token.object?.distanceTo?.(targetToken.object),reach=actor.getReach?.(performing?{action:'attack',weapon:performing}:{action:'interact'});
  if(!Number.isFinite(distance)||!Number.isFinite(reach)||distance>reach)throw Error('目标不在本次缴械的触及范围。');
 }
 async function authentic(e){if(typeof validateConfirmed!=='function'||await validateConfirmed(e)!==true)throw Error('没有本次真实盾牌格挡的已验证来源。');gm();}
 function checkProof(card,r){
  const c=card?.flags?.pf2e?.context,options=c?.options??[],author=card?.author?.id??card?.user?.id??card?.user;
  return !!card&&card.actor?.uuid===r.actorUuid&&author===r.ownerId&&card.rolls?.length===1&&c?.type==='skill-check'&&['criticalFailure','failure','success','criticalSuccess'].includes(c.outcome)&&options.includes('action:disarm')&&options.includes(marker(r.nonce))&&options.includes(disarmWeaponOption(r.weaponUuid))&&options.includes('skip-handling-message')&&c.target?.actor===r.attackerActorUuid&&c.target?.token===r.attackerTokenUuid;
 }
 function findCheck(r){return values(game.messages).find(m=>checkProof(m,r));}
 function captureCheckTiming(card){
  const options=card?.flags?.pf2e?.context?.options;if(!Array.isArray(options))return;
  const marks=options.filter(o=>typeof o==='string'&&o.startsWith(`${MODULE_ID}:disarming-block:`));if(marks.length!==1)return;
  const actor=card.actor,r=receipt(actor,marks[0].slice(`${MODULE_ID}:disarming-block:`.length));
  if(!r||!['claimed','uncertain'].includes(r.status)||r.ownerId!==game.user.id||!actor.testUserPermission?.(game.user,'OWNER')||!checkProof(card,r))return;
  // Native contextual actors can add the root of a dotted UUID (Actor/Scene)
  // alongside the complete option. It is not a second weapon; every other
  // weapon option still invalidates this exact-weapon timing proof.
  const exact=disarmWeaponOption(r.weaponUuid),parent=disarmWeaponOption(r.weaponUuid.split('.')[0]);
  if(options.some(o=>typeof o==='string'&&o.startsWith(`${MODULE_ID}:disarm-weapon:`)&&o!==exact&&o!==parent))return;
  const c=game.combat,index=c?.started?c.turns?.findIndex(t=>t.actor?.uuid===actor.uuid&&t.token?.uuid===r.tokenUuid)??-1:-1,combatant=index>=0?c.turns[index]:null;
  const timing={nonce:r.nonce,actorUuid:actor.uuid,tokenUuid:r.tokenUuid,worldTime:game.time?.worldTime??0,initiative:combatant?.initiative??actor.combatant?.initiative??null,rounds:index>=0&&index>c.turn?0:1,combat:combatant?{id:c.id,round:c.round,turn:c.turn,combatantId:combatant.id}:null};
  card.updateSource({[`flags.${MODULE_ID}.disarmingBlock.timing`]:timing});
 }
 function failureTiming(card,r){
  const t=own(card).timing;
  if(!t||t.nonce!==r.nonce||t.actorUuid!==r.actorUuid||t.tokenUuid!==r.tokenUuid||!Number.isFinite(t.worldTime)||![0,1].includes(t.rounds)||t.combat&&(!t.combat.id||!t.combat.combatantId||!Number.isInteger(t.combat.round)))throw Error('缺少真实缴械检定创建时的回合快照，不能重新开始措手不及计时。');
  const end=t.worldTime+t.rounds*6,now=game.time?.worldTime??0;
  if(!t.combat)return {timing:t,expired:now>=end};
  const c=game.combat?.id===t.combat.id?game.combat:game.combats?.get?.(t.combat.id),index=c?.turns?.findIndex(turn=>turn.id===t.combat.combatantId&&turn.actor?.uuid===t.actorUuid&&turn.token?.uuid===t.tokenUuid)??-1;
  return {timing:t,expired:!c?.started||index<0||now>end||c.round>t.combat.round+t.rounds||c.round===t.combat.round+t.rounds&&c.turn>=index};
 }
 async function clearGrasp(actor,weaponUuid){
  const effects=values(actor.items).filter(i=>i.type==='effect'&&own(i).kind==='weakened-grasp'&&own(i).weaponUuid===weaponUuid),weapon=await fromUuid(weaponUuid);
  if(effects.length){
   // Record the completed regrip/loss of hold on the exact physical item before
   // deleting its effect. Recovery must never re-create a cleared old result.
   if(weapon){gm();await weapon.update({[`flags.${MODULE_ID}.disarmingBlock.clearedChecks`]:[...new Set([...(own(weapon).clearedChecks??[]),...effects.map(i=>own(i).checkId)])]})}
   gm();await actor.deleteEmbeddedDocuments('Item',effects.map(i=>i.id));
  }
 }
 async function applyResult(actor,r,card){
  return effectQueue.run('disarm-effects',async()=>{
  gm();if(!checkProof(card,r))throw Error('缴械检定卡与原始格挡认领不一致。');
  if(r.kind!=='ordinary')await authentic(r);
  const d=await documents(r);gm();const {target,weapon}=d,outcome=card.flags.pf2e.context.outcome;
  const context={origin:{actor:actor.uuid,item:feature(actor)?.uuid??null,token:r.tokenUuid}},bound={weaponUuid:weapon.uuid,nonce:r.nonce,checkId:card.id};
  if(outcome==='success'&&legalWeapon(weapon)&&!own(weapon).clearedChecks?.includes(card.id)){
   const ids=attackIds(target,weapon),rules=[{key:'FlatModifier',selector:ids.map(id=>`${id}-attack`),type:'circumstance',value:-2},{key:'EphemeralEffect',affects:'origin',selectors:['skill-check'],uuid:BONUS,predicate:['action:disarm',disarmWeaponOption(weapon)]}];
   const data=effect('缴械：握持削弱','weakened-grasp',context,rules);Object.assign(data.flags[MODULE_ID].disarmingBlock,bound,{attackIds:ids,handsHeld:weapon.system.equipped.handsHeld});
   gm();await putEffect(target,`disarming-block:grasp:${weapon.uuid}`,data);tracked.set(target.uuid,target);
  }else if(outcome==='criticalSuccess'){
   // The guard is committed first: a disconnected client cannot retain a usable
   // NPC attack after the native item update succeeds but its response is lost.
   if(target.type==='npc'){gm();await target.update({[`flags.${MODULE_ID}.disarmingBlock.droppedWeapons`]:{...(own(target).droppedWeapons??{}),[weapon.id]:bound}});tracked.set(target.uuid,target)}
   if(held(weapon)&&!own(weapon).droppedChecks?.includes(card.id)){gm();await weapon.update({'system.equipped.carryType':'dropped','system.equipped.handsHeld':0,[`flags.${MODULE_ID}.disarmingBlock.droppedChecks`]:[...(own(weapon).droppedChecks??[]),card.id]})}
   await clearGrasp(target,weapon.uuid);
  }else if(outcome==='criticalFailure'){
   const {timing,expired}=failureTiming(card,r);
   if(!expired){
    const data=effect('缴械大失败：措手不及','critical-failure',context,[{key:'GrantItem',uuid:OFF_GUARD,inMemoryOnly:true}],{value:timing.rounds,unit:'rounds',expiry:'turn-start',sustained:false});
    data.system.start={value:timing.worldTime,initiative:timing.initiative};Object.assign(data.flags[MODULE_ID].disarmingBlock,bound);
    gm();const created=await putEffect(actor,`disarming-block:failure:${r.nonce}`,data);gm();
    // Native Effect._preCreate always resets start to now. Restore the actual
    // check's captured clock after creation, never restart it on reconnection.
    if(created.system.start?.value!==timing.worldTime||created.system.start?.initiative!==timing.initiative)await created.update({'system.start':data.system.start});
   }
  }
  gm();await save(actor,{...r,status:'done',checkId:card.id,outcome});return {status:'done',checkId:card.id,outcome};
  });
 }
 async function ownerRoll({actorUuid,nonce},sender){
  if(sender?.id!==game.users.activeGM?.id)throw Error('只有当前主GM能够发起卸武格挡检定。');
  const actor=await fromUuid(actorUuid),r=receipt(actor,nonce);
  if(!r||r.status!=='claimed'||r.ownerId!==game.user.id||!actor.testUserPermission?.(game.user,'OWNER'))throw Error('卸武格挡的拥有者或认领无效。');
  const previous=findCheck(r);if(previous)return {checkId:previous.id};
  if(rolling.has(nonce))throw Error('此卸武格挡检定已开始，不会重掷。');
  const d=await documents(r);requirements(d);
  if(sender.id!==game.users.activeGM?.id)throw Error('主GM已更换，旧检定请求失效。');
  if(turnKey(actor,game)!==r.turnKey)throw Error('缴械检定前回合已改变，需要重新确认MAP。');
  // PF2e 8.5 simpleRollActionCheck selects the first native origin, then
  // StatisticCheck reselects the target from its actor. Passing a Token alone
  // does not bind either choice, so reject mismatches before any native roll.
  const origin=actor.getActiveTokens?.(false,true)?.[0],target=d.target.getActiveTokens?.(true,true)?.find(t=>t.actor?.isOfType?.('army','creature','hazard')??['character','npc','army','hazard'].includes(t.actor?.type));
  if(origin?.uuid!==r.tokenUuid||origin?.actor?.uuid!==actor.uuid||target?.uuid!==r.attackerTokenUuid||target?.actor?.uuid!==d.target.uuid)throw Error('原生缴械将使用其他Token，本次准确格挡目标无法绑定，未掷骰。');
  const native=game.pf2e.actions.get('disarm');if(!native?.toActionVariant)throw Error('缺少可等待的原生Disarm动作。');
  rolling.add(nonce);
  const result=await native.toActionVariant({cost:'free'}).use({actors:[actor],target:d.targetToken.object,multipleAttackPenalty:r.map,rollOptions:[marker(nonce),disarmWeaponOption(r.weaponUuid),'skip-handling-message'],event:{ctrlKey:false,metaKey:false,shiftKey:game.user.settings?.showCheckDialogs??true}});
  const card=result?.[0]?.message;if(!card)return {cancelled:true};
  if(!checkProof(card,r))throw Error('原生缴械返回卡无法与本次武器认领核对。');return {checkId:card.id};
 }
 async function handleConfirmed(event){
  gm();await authentic(event);const actor=await fromUuid(event.actorUuid);gm();if(!actor)throw Error('卸武格挡角色已不存在。');
  let record=await queue.run(actor.uuid,async()=>{
   gm();const existing=receipt(actor,event.nonce);
   if(existing&&existing.status!=='offered'){if(['claimed','uncertain'].includes(existing.status)){const card=findCheck(existing);if(card)await applyResult(actor,existing,card)}return null}
   if(activeOffers.has(event.nonce))return null;
   try{currentTrigger(actor,event);requirements(await documents(event))}catch(error){gm();await save(actor,{...structuredClone(event),kind:'block',ownerId:existing?.ownerId??event.userId,status:'ineligible',reason:error.message});return null}
   gm();let user=game.users.get(existing?.ownerId??event.userId);if(!user?.active||!actor.testUserPermission?.(user,'OWNER'))user=values(game.users).find(u=>u.active&&!u.isGM&&actor.testUserPermission?.(u,'OWNER'))??game.user;
   const r={...structuredClone(event),kind:'block',ownerId:user.id,status:'offered'};await save(actor,r);activeOffers.add(event.nonce);return r;
  });
  if(!record)return {status:receipt(actor,event.nonce)?.status??'ignored'};
  try{
  const mapChoices=()=>[0,1,2].map(map=>({value:`use:${map}`,label:`卸武格挡（${map===0?'无MAP':map===1?'第二次攻击':'第三次攻击'}）`}));let choiceTurn=turnKey(actor,game);
  let choices=choiceTurn?mapChoices():[{value:'use:0',label:'使用卸武格挡（自由动作）'}];choices.push({value:'decline',label:'不使用卸武格挡'});
  let selected=await choose({actor,user:game.users.get(record.ownerId),title:'卸武格挡：缴械本次攻击所用的武器？',choices});
  if(selected?.startsWith('use:')&&turnKey(actor,game)&&turnKey(actor,game)!==choiceTurn){gm();choiceTurn=turnKey(actor,game);choices=[...mapChoices(),{value:'decline',label:'不使用卸武格挡'}];selected=await choose({actor,user:game.users.get(record.ownerId),title:'缴械现在发生于自己回合：请选择当前MAP',choices})}
  await authentic(event);
  record=await queue.run(actor.uuid,async()=>{gm();const current=receipt(actor,event.nonce);if(current?.status!=='offered')return null;if(selected==null||selected==='decline'){await save(actor,{...current,status:'declined'});return null}if(!choices.some(c=>c.value===selected))throw Error('卸武格挡选择无效。');try{currentTrigger(actor,event);requirements(await documents(event))}catch(error){gm();await save(actor,{...current,status:'ineligible',reason:error.message});return null}gm();const currentTurn=turnKey(actor,game);if(currentTurn&&currentTurn!==choiceTurn)throw Error('缴械确认期间回合已变化，未使用新的MAP。');const map=currentTurn?Number(selected.split(':')[1]):0;const r={...current,status:'claimed',map,turnKey:currentTurn};await save(actor,r);return r});
  if(!record)return {status:receipt(actor,event.nonce)?.status};
  gm();let result;
  try{
   if(record.ownerId===game.user.id)result=await ownerRoll({actorUuid:actor.uuid,nonce:event.nonce},game.user);
   else {if(!socket)throw Error('缺少拥有者原生缴械通讯。');const response=await socket.executeAsUser('disarming-block:roll',record.ownerId,{actorUuid:actor.uuid,nonce:event.nonce});if(!response?.ok)throw Error(response?.error??'拥有者原生缴械未完成。');result=response.value}
  }catch(error){if(isActiveGM(game))await queue.run(actor.uuid,async()=>{const current=receipt(actor,event.nonce),card=findCheck(record);if(card)return applyResult(actor,current,card);if(current?.status==='claimed')await save(actor,{...current,status:'uncertain',error:String(error.message??error)})});throw error}
  gm();return queue.run(actor.uuid,async()=>{gm();const current=receipt(actor,event.nonce);if(current?.status==='done')return {status:'done',checkId:current.checkId};const card=game.messages.get(result?.checkId)??findCheck(record);if(card)return applyResult(actor,current,card);await save(actor,{...current,status:result?.cancelled?'cancelled':'uncertain'});return {status:result?.cancelled?'cancelled':'uncertain'}});
  }finally{activeOffers.delete(event.nonce)}
 }
 async function maintain(actor){
  if(!isActiveGM(game)||!actor?.items)return;
  return queue.run(actor.uuid,async()=>{
   await effectQueue.run('disarm-effects',async()=>{gm();for(const i of values(actor.items).filter(i=>own(i).kind==='weakened-grasp')){
    const w=await fromUuid(own(i).weaponUuid);gm();
    if(!held(w)||Number.isFinite(own(i).handsHeld)&&w.system.equipped.handsHeld>own(i).handsHeld){await clearGrasp(actor,own(i).weaponUuid);continue}
    const ids=attackIds(actor,w),changes={};
    if(JSON.stringify(ids)!==JSON.stringify(own(i).attackIds)){const rules=structuredClone(i.system.rules);rules.find(r=>r.key==='FlatModifier').selector=ids.map(id=>`${id}-attack`);changes['system.rules']=rules;changes[`flags.${MODULE_ID}.disarmingBlock.attackIds`]=ids}
    // Releasing one hand is free and is not the Interact that ends Disarm.
    // Persist the remaining grip so adding that hand again is recognized.
    if(w.system.equipped.handsHeld!==own(i).handsHeld)changes[`flags.${MODULE_ID}.disarmingBlock.handsHeld`]=w.system.equipped.handsHeld;
    if(Object.keys(changes).length){gm();await i.update(changes)}
   }});
   for(const r of own(actor).uses??[])if(['claimed','uncertain'].includes(r.status)){const card=findCheck(r);if(card)await applyResult(actor,r,card)}
  });
 }
 async function completeRegrip(actor,weaponUuid,user){
  gm();if(!user||!actor.testUserPermission?.(user,'OWNER'))throw Error('没有改握角色的所有者权限。');const weapon=await fromUuid(weaponUuid);gm();if(weapon?.actor?.uuid!==actor.uuid||!held(weapon))throw Error('改握武器必须是当前角色持握的准确物品。');return queue.run(actor.uuid,()=>effectQueue.run('disarm-effects',()=>clearGrasp(actor,weaponUuid)));
 }
 async function clearRegripFromCard({actor,weaponUuid,user,graspCheckId,cardId}){
  const valid=()=>{gm();const card=game.messages.get(cardId),proof=card?.flags?.[MODULE_ID]?.disarmRegrip;
   if(!card||card.actor?.uuid!==actor?.uuid||card.author?.id!==user?.id||!actor.testUserPermission?.(user,'OWNER')||!proof?.nonce||proof.status!=='claimed'||proof.actorUuid!==actor.uuid||proof.weaponUuid!==weaponUuid||proof.graspCheckId!==graspCheckId||proof.userId!==user.id)throw Error('没有本次真实Interact改握动作卡的准确认领。');
  };valid();
  return queue.run(actor.uuid,()=>effectQueue.run('disarm-effects',async()=>{
   valid();const current=getWeakenedGrasp(actor,weaponUuid);if(current&&own(current).checkId!==graspCheckId)return {status:'superseded'};
   if(current)await clearGrasp(actor,weaponUuid);return {status:'cleared'};
  }));
 }
 async function prepareOrdinary(payload,user){
  gm();const actor=await fromUuid(payload?.actorUuid);gm();if(!actor?.testUserPermission?.(user,'OWNER')||!user.active)throw Error('普通缴械的角色所有者无效。');
  const record={kind:'ordinary',nonce:`ordinary-${globalThis.crypto.randomUUID().replaceAll('-','')}`,actorUuid:actor.uuid,tokenUuid:payload.tokenUuid,attackerActorUuid:payload.attackerActorUuid,attackerTokenUuid:payload.attackerTokenUuid,weaponUuid:payload.weaponUuid,attackItemUuid:payload.weaponUuid,ownerId:user.id,userId:user.id,status:'claimed'};
  const d=await documents(record);gm();requirements(d,{ordinary:true});const {target,weapon}=d;if(!legalWeapon(weapon)||!values(target.items).some(i=>own(i).kind==='weakened-grasp'&&i.isExpired!==true))throw Error('普通缴械没有需要精确绑定的握持削弱目标。');
  await queue.run(actor.uuid,()=>save(actor,record));gm();return {nonce:record.nonce,rollOptions:[marker(record.nonce),disarmWeaponOption(weapon),'skip-handling-message']};
 }
 async function settleOrdinary(payload,user){
  gm();const actor=await fromUuid(payload?.actorUuid);gm();if(!actor?.testUserPermission?.(user,'OWNER'))throw Error('普通缴械所有者无效。');
  return queue.run(actor.uuid,async()=>{gm();const r=receipt(actor,payload.nonce);if(r?.kind!=='ordinary'||r.ownerId!==user.id)throw Error('普通缴械认领无效。');if(r.status==='done'){if(payload.checkId!==r.checkId)throw Error('普通缴械结果与原认领不一致。');return {status:'done',checkId:r.checkId}}const card=game.messages.get(payload.checkId);return applyResult(actor,r,card)});
 }
 async function asGM(name,payload,local){
  if(isActiveGM(game))return local(payload,game.user);
  if(!socket||!game.users.activeGM)throw Error('缺少当前主GM缴械通讯。');const response=await socket.executeAsUser(name,game.users.activeGM.id,payload);if(!response?.ok)throw Error(response?.error??'缴械通讯未完成。');return response.value;
 }
 const prepareOrdinaryDisarm=payload=>asGM('disarming-block:ordinary-prepare',payload,prepareOrdinary);
 const settleOrdinaryDisarm=payload=>asGM('disarming-block:ordinary-settle',payload,settleOrdinary);
 function wrapStrike(strike,actor){
  if(actor?.type!=='npc'||strike?.item?.type!=='melee')return strike;
  const weaponId=strike.item.flags?.pf2e?.linkedWeapon;if(!weaponId)return strike;
  const guard=()=>{if(own(actor).droppedWeapons?.[weaponId]&&!held(item(actor,weaponId)))throw Error('这把已被缴械的武器尚未重新持握。')};
  if(own(actor).droppedWeapons?.[weaponId]&&!held(item(actor,weaponId))){strike.ready=false;strike.canAttack=false}
  for(const variant of strike.variants??[]){const original=variant.roll;if(typeof original!=='function'||original.disarmingBlockWrapped)continue;const wrapped=async function(...args){guard();return original.apply(this,args)};wrapped.disarmingBlockWrapped=true;variant.roll=wrapped}
  for(const key of['attack','roll','damage','critical']){const original=strike[key];if(typeof original!=='function'||original.disarmingBlockWrapped)continue;const wrapped=async function(...args){if(!(['damage','critical'].includes(key)&&args[0]?.getFormula===true))guard();return original.apply(this,args)};wrapped.disarmingBlockWrapped=true;strike[key]=wrapped}
  return strike;
 }
 function register({Hooks,socket:api,libWrapper}={}){
  socket=api;socket?.register('disarming-block:roll',async function(payload){try{return {ok:true,value:await ownerRoll(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  for(const[name,handler]of [['disarming-block:ordinary-prepare',prepareOrdinary],['disarming-block:ordinary-settle',settleOrdinary]])socket?.register(name,async function(payload){try{return {ok:true,value:await handler(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  const ids=[],on=(n,f)=>ids.push([n,Hooks.on(n,(...args)=>Promise.resolve().then(()=>f(...args)).catch(onError))]);
  ids.push(['preCreateChatMessage',Hooks.on('preCreateChatMessage',captureCheckTiming)]);
  const npcPath='CONFIG.PF2E.Actor.documentClasses.npc.prototype.prepareDerivedData';
  if(libWrapper)libWrapper.register(MODULE_ID,npcPath,function(wrapped,...args){const result=wrapped(...args);for(const strike of this.system.actions??[])wrapStrike(strike,this);return result},'WRAPPER');
  // Current unlinked Token actors may already be prepared before ready. Cover
  // those existing actions without resetting documents or constructing tokens
  // from other scenes; the same pass handles cached actors on scene entry.
  const guardCanvas=()=>{for(const actor of new Set(values(canvas?.tokens?.placeables).map(t=>t.actor)))if(actor?.type==='npc')for(const strike of actor.system.actions??[])wrapStrike(strike,actor)};
  guardCanvas();on('canvasReady',guardCanvas);
  for(const n of['updateItem','deleteItem','createItem'])on(n,i=>{if(i?.type==='weapon'||i?.type==='melee')return maintain(i.actor)});
  on('createChatMessage',m=>{const options=m?.flags?.pf2e?.context?.options,marked=Array.isArray(options)&&options.some(o=>typeof o==='string'&&o.startsWith(`${MODULE_ID}:disarming-block:`));if(marked&&isActiveGM(game))return maintain(m.actor)});
  for(const a of values(game.actors)){
   if(a.type==='npc'&&Object.keys(own(a).droppedWeapons??{}).length)for(const strike of a.system.actions??[])wrapStrike(strike,a);
   if(own(a).uses?.length||values(a.items).some(i=>own(i).kind==='weakened-grasp'))tracked.set(a.uuid,a);
  }
  on('deleteActor',actor=>tracked.delete(actor.uuid));
  on('deleteToken',token=>{if(token?.actorLink===false&&token.actor?.uuid)tracked.delete(token.actor.uuid)});
  on('updateUser',()=>{if(isActiveGM(game))for(const a of tracked.values())void maintain(a).catch(onError)});
  return()=>{for(const[n,id]of ids)Hooks.off(n,id);if(libWrapper)libWrapper.unregister(MODULE_ID,npcPath)};
 }
 return {handleConfirmed,maintain,wrapStrike,completeRegrip,clearRegripFromCard,prepareOrdinaryDisarm,settleOrdinaryDisarm,register};
}
