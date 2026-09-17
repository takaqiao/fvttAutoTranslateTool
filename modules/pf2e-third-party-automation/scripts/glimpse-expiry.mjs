import {MODULE_ID as M} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {glimpseEncounter} from './glimpse-source.mjs';
const values=c=>Array.from(c?.values?.()??c??[]),slug=nonce=>`tpa-glimpse-${nonce.toLowerCase()}`;
const order=combat=>combat.turns.map(c=>({id:c.id,initiative:Number.isFinite(c.initiative)?c.initiative:null}));
export function glimpseExpiryFor(enemy,game){const {combat,combatant}=glimpseEncounter(enemy,game),index=combat.turns.indexOf(combatant),baselineEnded=combatant.flags?.pf2e?.roundOfLastTurnEnd??null;if(Number.isInteger(baselineEnded)&&(baselineEnded>combat.round||index>combat.turn&&baselineEnded>=combat.round))throw Error('敌人本轮已结束后又被重排，无法证明下一次结束；请先核对实际轮次。');return {schema:1,combatId:combat.id,combatantId:combatant.id,actorUuid:enemy.actor.uuid,tokenUuid:enemy.uuid,startRound:combat.round,startTurn:combat.turn,endRound:combat.round+(index>combat.turn?0:1),baselineEnded,order:order(combat)}}
const valid=r=>r?.schema===1&&['combatId','combatantId','actorUuid','tokenUuid'].every(k=>typeof r[k]==='string'&&r[k])&&Number.isInteger(r.startRound)&&r.startRound>0&&Number.isInteger(r.startTurn)&&r.startTurn>=0&&[r.startRound,r.startRound+1].includes(r.endRound)&&(r.baselineEnded===null||Number.isInteger(r.baselineEnded))&&Array.isArray(r.order)&&r.order.some(c=>c.id===r.combatantId);
/** Native round durations compare initiative numbers and the viewed combat.
 * Keep the engine's exact GrantItem, but expire our effect against the enemy's
 * actual PF2e end-turn receipt instead, including reload and source deletion. */
export function createGlimpseExpiry({game,onError=error=>{console.error(M,error);globalThis.ui?.notifications?.warn?.(error.message)}}={}){
 const queue=new SerialActions();let installed=false;
 const present=effect=>effect.actor?.items?.get?.(effect.id)===effect;
 function exact(effect,record){const rules=effect?.system?.rules,condition=game.pf2e?.ConditionManager?.conditions?.get('enfeebled')?.uuid;return valid(record)&&/^[A-Za-z0-9-]{1,80}$/.test(record.nonce??'')&&record.effectId===effect.id&&effect.type==='effect'&&effect.actor?.uuid===record.actorUuid&&effect.system.slug===slug(record.nonce)&&effect.system.context?.origin?.actor===record.actorUuid&&effect.system.context?.origin?.token===record.tokenUuid&&rules?.length===1&&rules[0].key==='GrantItem'&&rules[0].uuid===condition&&rules[0].inMemoryOnly===true&&rules[0].alterations?.some(a=>a.mode==='override'&&a.property==='badge-value'&&a.value===2)}
 function scope(record){const combat=game.combats?.get?.(record.combatId),combatant=combat?.turns?.find(c=>c.id===record.combatantId),token=values(game.scenes).flatMap(s=>values(s.tokens)).find(t=>t.uuid===record.tokenUuid);return {combat,combatant,token}}
 async function finite(effect,record,reason){if(!isActiveGM(game)||!present(effect))return;const {combatant}=scope(record);await effect.update({'system.duration':{value:1,unit:'rounds',expiry:'turn-end',sustained:false},'system.start':{value:game.time.worldTime,initiative:combatant?.initiative??null},[`flags.${M}.glimpseExpiry`]:{...record,status:'fallback',reason}});onError(Error(`救赎瞥视的原遭遇或目标已改变（${reason}）；这份衰弱已恢复原生 1 轮有限到期，请核对剩余时长。`))}
 async function settle(effect){return queue.run(effect.uuid??`${effect.actor?.uuid}.${effect.id}`,async()=>{
  const record=effect.flags?.[M]?.glimpseExpiry;if(!isActiveGM(game)||!present(effect)||record?.status!=='armed'||!exact(effect,record))return;
  const {combat,combatant,token}=scope(record);
  if(!combat?.started||!combatant||combatant.actor?.uuid!==record.actorUuid||combatant.token?.uuid!==record.tokenUuid||token?.actor?.uuid!==record.actorUuid)return finite(effect,record,'原遭遇、参战者或 token 不再存在');
  if(JSON.stringify(order(combat))!==JSON.stringify(record.order))return finite(effect,record,'先攻顺序或数值已改变');
  if(combat.round<record.startRound||combat.round===record.startRound&&combat.turn<record.startTurn)return finite(effect,record,'遭遇轮次回退');
  const ended=combatant.flags?.pf2e?.roundOfLastTurnEnd;
  if(Number.isInteger(ended)&&ended>combat.round)return finite(effect,record,'回合结束记录晚于当前轮次');
  if(Number.isInteger(ended)&&ended>=record.endRound&&(record.baselineEnded===null||ended>record.baselineEnded)){if(isActiveGM(game)&&present(effect))await effect.actor.deleteEmbeddedDocuments('Item',[effect.id]);return}
  // A later real end-turn receipt proves this target end was skipped. Merely
  // observing the next round before asynchronous PF2e end hooks finish does not.
  if(combat.turns.some(c=>Number.isInteger(c.flags?.pf2e?.roundOfLastTurnEnd)&&c.flags.pf2e.roundOfLastTurnEnd>record.endRound))return finite(effect,record,'目标结束回合被跳过');
 })}
 async function reconcile(){if(!isActiveGM(game))return;const actors=new Map([...values(game.actors),...values(game.scenes).flatMap(s=>values(s.tokens).map(t=>t.actor))].filter(Boolean).map(a=>[a.uuid,a]));for(const actor of actors.values())for(const effect of values(actor.items).filter(e=>e.flags?.[M]?.glimpseExpiry?.status==='armed'))await settle(effect)}
 async function arm({effect,expiry,nonce}){const record={...structuredClone(expiry),nonce,effectId:effect?.id,status:'armed'};if(!isActiveGM(game)||!present(effect)||!exact(effect,record))throw Error('救赎瞥视到期范围或原生衰弱效果无法验证。');if(effect.flags?.[M]?.glimpseExpiry)throw Error('这份救赎瞥视效果已绑定到期范围。');await effect.update({'system.duration':{value:-1,unit:'unlimited',expiry:null,sustained:false},[`flags.${M}.glimpseExpiry`]:record});await settle(effect)}
 function register({Hooks}){if(installed)return;installed=true;const changed=()=>reconcile().catch(onError);for(const name of ['pf2e.endTurn','updateCombat','createCombatant','deleteCombat','deleteCombatant','deleteToken','deleteScene'])Hooks.on(name,changed);Hooks.on('updateCombatant',(_doc,changes)=>{if(changes.flags?.pf2e?.roundOfLastTurnEnd!==undefined||Object.hasOwn(changes,'flags.pf2e.roundOfLastTurnEnd')||Object.hasOwn(changes,'initiative')||changes.flags?.pf2e?.overridePriority!==undefined)changed()})}
 return {arm,reconcile,register};
}
