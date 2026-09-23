import {getSourceId} from './native-context.mjs';

export const ROARING_APPLAUSE_SOURCE='Compendium.pf2e.spells-srd.Item.czO0wbT1i320gcu9';
const values=c=>Array.from(c?.values?.()??c??[]);
const isSource=item=>item?.type==='spell'&&getSourceId(item)===ROARING_APPLAUSE_SOURCE;
const fail=reason=>({handled:true,eligible:false,reason});
// PF2e stores overrides by initiative; the whole map is not a tie priority.
export const roaringTurnPriority=combatant=>combatant.overridePriority?.(combatant.initiative??0)??combatant.flags?.pf2e?.overridePriority?.[combatant.initiative??0]??null;

/** Admit this original invocation, without enrolling the owner's other spells. */
export function assessRoaringCast({game,actor,item,entry,user=game?.user,options={}}={}){
 if(game?.world?.id!=='ujx5r8oipw7ercdr'||actor?.type!=='character'||actor.isToken||!isSource(item)||options.consume===false||options.message===false)return {handled:false,eligible:false};
 if(game.system?.version!=='8.5.1')return fail('当前系统版本尚未验证轰然喝彩接入。');
 const base=item.original??item;
 if(game.actors?.get(actor.id)!==actor||actor.items?.get(base.id)!==base||item.actor!==actor||base.actor!==actor||item.uuid!==base.uuid||!isSource(base)||!user?.active||game.users?.get(user.id)!==user||actor.testUserPermission?.(user,'OWNER')!==true)return fail('需要原始法术、当前角色与其所有者的实际施法。');
 if(!game.users.activeGM?.active||actor.canAct!==true||actor.isDead===true)return fail('需要在线主GM及能够行动的施法者。');
 if(entry?.type!=='spellcastingEntry'||entry.actor!==actor||actor.items.get(entry.id)!==entry||item.spellcasting!==entry||entry.isSpontaneous!==true||entry.system.prepared?.value!=='spontaneous'||entry.system.tradition?.value!=='occult'||base.system.location?.value!==entry.id||item.system.location?.value!==entry.id||base.system.location?.signature!==true)return fail('当前接入只支持本角色的原生自发施法条目与招牌法术。');
 const rank=options.rank??item.rank,slot=entry.system.slots?.slot3;
 if(rank!==3||!Number.isInteger(slot?.value)||!Number.isInteger(slot?.max)||slot.value<1||slot.value>slot.max||item.atWill||item.isCantrip||(item.system.cast?.focusPoints??0)!==0)return fail('当前接入只支持三环单目标及可用原生法术位。');
 if((options.messageMode??game.settings?.get('core','messageMode'))!=='public'||options.rollMode&&options.rollMode!=='publicroll')return fail('当前接入支持公开施法；私密结果需手工处理。');
 if(values(item.appliedOverlays).length||Object.keys(base.system.overlays??{}).length||base.flags?.['pf2e-toolbelt']?.actionable?.linked||base.flags?.['pf2e-toolbelt']?.linked)return fail('本法术的覆盖或自定义宏尚未验证，请手工处理。');
 const data=base.system;
 if(data.level?.value!==3||data.time?.value!=='2'||data.range?.value!=='60 feet'||data.area||data.duration?.sustained!==true||data.rules?.length||Object.keys(data.damage??{}).length||data.defense?.save?.statistic!=='will'||data.defense.save.basic!==false||values(data.traits?.value).sort().join(',')!=='concentrate,emotion,manipulate,mental')return fail('原法术的动作、射程、特征或豁免配置已改变，请手工核对。');
 return {handled:true,eligible:true,rank,base};
}

/** Bind the original public target; spatial and sensory legality belongs to the table. */
export function validateRoaringTarget({game,actor,token,targets}={}){
 const scene=token?.parent;
 if(!scene||game.scenes?.get(scene.id)!==scene||scene.tokens?.get(token.id)!==token||token.documentName!=='Token'||token.actor!==actor||token.hidden||!token.object)throw Error('需要当前场景中准确的公开施法者Token。');
 if(!Array.isArray(targets)||targets.length!==1)throw Error('当前接入需要一个实际目标Token。');
 const target=targets[0];
 if(target?.documentName!=='Token'||target.parent!==scene||scene.tokens.get(target.id)!==target||!target.object||target.hidden||!['character','npc','familiar'].includes(target.actor?.type)||target.actor.isDead===true)throw Error('目标已改变或不是可确认的公开生物目标。');
 return target;
}

/** Freeze a unique real own turn. The viewed encounter and initiative ties are irrelevant. */
export function roaringOwnTurn({game,actor,token}={}){
 const candidates=values(game?.combats).filter(combat=>combat.started===true&&combat.scene?.id===token?.parent?.id&&values(combat.turns).some(c=>c.token?.uuid===token?.uuid));
 if(candidates.length!==1)throw Error('需要施法者所在的唯一进行中遭遇；当前时间关系请手工处理。');
 const combat=candidates[0],turns=values(combat.turns),matching=turns.filter(c=>c.token?.uuid===token.uuid),combatant=matching[0];
 if(matching.length!==1||combatant.actor!==actor||combatant.token!==token||!Number.isInteger(combat.round)||combat.round<1||!Number.isInteger(combat.turn)||turns[combat.turn]!==combatant||!Number.isFinite(combatant.initiative)||new Set(turns.map(c=>c.id)).size!==turns.length)throw Error('当前接入只处理施法者准确的本人回合。');
 const lastTurnEnd=combatant.flags?.pf2e?.roundOfLastTurnEnd??null;
 if(lastTurnEnd!==null&&(!Number.isInteger(lastTurnEnd)||lastTurnEnd<0||lastTurnEnd>=combat.round))throw Error('施法者本轮已结束或回合记录不明确，请手工核对时长。');
 const order=turns.map(c=>({id:c.id,initiative:Number.isFinite(c.initiative)?c.initiative:null,overridePriority:roaringTurnPriority(c)}));
 if(order.some(c=>typeof c.id!=='string'||!c.id||c.overridePriority!==null&&!Number.isFinite(c.overridePriority)))throw Error('遭遇顺序记录不明确，请手工核对时长。');
 const ended=turns.map(c=>c.flags?.pf2e?.roundOfLastTurnEnd).filter(Number.isInteger);
 return {combatId:combat.id,combatantId:combatant.id,actorUuid:actor.uuid,tokenUuid:token.uuid,started:true,round:combat.round,turn:combat.turn,order,lastTurnEnd,latestTurnEndRound:ended.length?Math.max(...ended):null};
}
