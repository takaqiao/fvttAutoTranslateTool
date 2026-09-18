import {MODULE_ID as M} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
export const SCAR_SLOWED='Compendium.pf2e.conditionitems.Item.xYTAsEpcJE1Ccni3';
const values=c=>Array.from(c?.values?.()??c??[]),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const order=c=>c.turns.map(t=>({id:t.id,initiative:Number.isFinite(t.initiative)?t.initiative:null}));
export function spiritualScarExpiryFor(token,game){
 const matches=values(game.combats).filter(c=>c.started&&c.turns.some(t=>t.token===token&&t.actor===token.actor));
 if(matches.length!==1)throw Error('精神伤痕持续时间需要唯一的实际来源遭遇。');
 const combat=matches[0],index=combat.turns.findIndex(t=>t.token===token),combatant=combat.turns[index],baseline=combatant.flags?.pf2e?.roundOfLastTurn??null;
 if(!Number.isInteger(combat.round)||combat.round<1||!Number.isInteger(combat.turn)||combat.turn<0||combat.turn>=combat.turns.length||!Number.isFinite(game.time?.worldTime)||baseline!==null&&(!Number.isInteger(baseline)||baseline>combat.round||index>combat.turn&&baseline>=combat.round))throw Error('精神伤痕无法证明来源的下一回合开始。');
 return {schema:1,combatId:combat.id,combatantId:combatant.id,actorUuid:token.actor.uuid,tokenUuid:token.uuid,startRound:combat.round,startTurn:combat.turn,endRound:combat.round+(index>combat.turn?0:1),baselineStarted:baseline,startTime:game.time.worldTime,order:order(combat)};
}
/** Player Core p426: round duration decreases at the creator's turn start.
 * Native duration compares world time, numeric initiative and viewed combat;
 * bind this source to its actual creator's persisted PF2e start-turn receipt. */
export function createSpiritualScarExpiry({game,onError=error=>{console.error(M,error);globalThis.ui?.notifications?.warn?.(error.message)}}={}){
 const queue=new SerialActions();let installation;
 const present=e=>e.actor?.items?.get(e.id)===e;
 function exact(e,r){
  const s=e._source?.system??e.system,rules=s?.rules;
  return r?.schema===1&&['armed','fallback'].includes(r.status)&&/^[A-Za-z0-9-]{1,80}$/.test(r.nonce??'')&&Number.isFinite(r.startTime)&&Number.isInteger(r.startRound)&&r.startRound>0&&Number.isInteger(r.startTurn)&&r.startTurn>=0&&[r.startRound,r.startRound+1].includes(r.endRound)&&(r.baselineStarted===null||Number.isInteger(r.baselineStarted))&&Array.isArray(r.order)&&r.order.some(c=>c.id===r.combatantId)&&e.type==='effect'&&e.actor?.uuid===r.targetActorUuid&&s.slug===`tpa-spiritual-scar-${r.nonce.toLowerCase()}`&&s.context?.origin?.actor===r.actorUuid&&s.context.origin.token===r.tokenUuid&&s.context.origin.item===r.itemUuid&&rules?.length===1&&rules[0].key==='GrantItem'&&rules[0].uuid===SCAR_SLOWED&&rules[0].inMemoryOnly===true&&rules[0].allowDuplicate===true&&rules[0].alterations?.length===1&&rules[0].alterations[0].mode==='override'&&rules[0].alterations[0].property==='badge-value'&&rules[0].alterations[0].value===1;
 }
 async function remove(effect){if(isActiveGM(game)&&present(effect))await effect.actor.deleteEmbeddedDocuments('Item',[effect.id])}
 async function fallback(effect,r,reason){
  if(!isActiveGM(game)||!present(effect))return;
  const initiative=r.order.find(t=>t.id===r.combatantId)?.initiative??null;
  const result=await effect.update({'system.duration':{value:1,unit:'rounds',expiry:'turn-start',sustained:false},'system.start':{value:r.startTime,initiative},[`flags.${M}.spiritualScarExpiry`]:{...r,status:'fallback',reason}});
  if(result!==effect||effect.flags?.[M]?.spiritualScarExpiry?.status!=='fallback')throw Error('精神伤痕有限时长回退未保存，请核对这份缓慢。');
  onError(Error(`精神伤痕原回合记录已改变（${reason}）；这份缓慢已恢复原始起点的有限一轮时长，请核对到期。`));
 }
 async function settle(effect){return queue.run(effect.uuid,async()=>{
  const r=effect.flags?.[M]?.spiritualScarExpiry;if(!isActiveGM(game)||!present(effect)||!exact(effect,r))return;
  const receipt=game.messages.get(r.damageMessageId),combat=game.combats.get(r.combatId);
  if(!receipt||receipt.flags?.pf2e?.appliedDamage?.isReverted||!combat?.started)return remove(effect);
  if(r.status==='fallback')return;
  const combatant=combat.turns.find(c=>c.id===r.combatantId),token=values(game.scenes).flatMap(s=>values(s.tokens)).find(t=>t.uuid===r.tokenUuid);
  if(!combatant||combatant.actor?.uuid!==r.actorUuid||combatant.token!==token||token?.actor?.uuid!==r.actorUuid)return fallback(effect,r,'来源参战者或 token 已改变');
  if(!same(order(combat),r.order))return fallback(effect,r,'先攻顺序或数值已改变');
  if(combat.round<r.startRound||combat.round===r.startRound&&combat.turn<r.startTurn)return fallback(effect,r,'遭遇轮次回退');
  const started=combatant.flags?.pf2e?.roundOfLastTurn;
  if(Number.isInteger(started)&&started>combat.round)return fallback(effect,r,'来源开始记录晚于当前轮次');
  if(Number.isInteger(started)&&started>=r.endRound&&(r.baselineStarted===null||started>r.baselineStarted))return remove(effect);
  const index=combat.turns.indexOf(combatant);
  if((combat.round>r.endRound||combat.round===r.endRound&&combat.turn>index)&&combat.turns.some(c=>c.flags?.pf2e?.roundOfLastTurn===combat.round))return fallback(effect,r,'来源开始回合被跳过');
 })}
 async function reconcile(){
  if(!isActiveGM(game))return;
  const actors=new Map([...values(game.actors),...values(game.scenes).flatMap(s=>values(s.tokens).map(t=>t.actor))].filter(Boolean).map(a=>[a.uuid,a]));
  for(const actor of actors.values())for(const effect of values(actor.items).filter(e=>e.flags?.[M]?.spiritualScarExpiry))await settle(effect);
 }
 function register({Hooks}){
  if(installation)return;const ids=[],changed=()=>reconcile().catch(onError);installation={Hooks,ids};
  for(const event of ['pf2e.startTurn','updateCombat','updateCombatant','createCombatant','deleteCombatant','deleteCombat','deleteToken','deleteScene','updateChatMessage','deleteChatMessage','updateUser'])ids.push([event,Hooks.on(event,changed)]);
  changed();
 }
 function unregister(){if(installation)for(const [event,id]of installation.ids)installation.Hooks.off(event,id);installation=null}
 return {reconcile,settle,register,unregister};
}
