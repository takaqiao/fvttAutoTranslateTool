import {getSourceId} from './native-context.mjs';

export const FORCE_BARRAGE_SOURCE='Compendium.pf2e.spells-srd.Item.gKKqvLohtrSJj3BM';
const values=c=>Array.from(c?.values?.()??c??[]);
const fail=reason=>({handled:true,eligible:false,reason});
export const isForceBarrageItem=item=>item?.type==='spell'&&getSourceId(item)===FORCE_BARRAGE_SOURCE;

/** Admission is call-local. Owning this spell never enrolls other actor casts. */
export function assessForceBarrageCast({game,actor,item,entry,user=game.user,options={}}={}){
 if(game?.world?.id!=='ujx5r8oipw7ercdr'||actor?.type!=='character'||actor.isToken||!isForceBarrageItem(item)||options.consume===false||options.message===false)return {handled:false,eligible:false};
 if(game.system?.version!=='8.5.1')return fail('当前系统版本尚未验证力场飞弹原施法接线。');
 const base=item.original??item;
 if(actor?.type!=='character'||actor.isToken||game.actors?.get(actor.id)!==actor||item.actor!==actor||base.actor!==actor||actor.items?.get(base.id)!==base||item.uuid!==base.uuid||!isForceBarrageItem(base)||!user?.active||game.users?.get(user.id)!==user||actor.testUserPermission?.(user,'OWNER')!==true)return fail('需要现役角色、原始法术和当前所有者的实际施法。');
 if(!game.users.activeGM?.active||actor.canAct!==true||actor.isDead===true)return fail('需要在线主GM及能够行动的施法者。');
 if(entry?.actor!==actor||actor.items.get(entry.id)!==entry||entry.type!=='spellcastingEntry'||entry.isSpontaneous!==true||entry.system.prepared?.value!=='spontaneous'||entry.system.tradition?.value!=='occult'||base.system.location?.value!==entry.id||item.system.location?.value!==entry.id||base.system.location?.signature!==true)return fail('当前桥仅支持本施法条目的自发施法招牌法术。');
 const rank=options.rank??item.rank,slot=entry.system.slots?.[`slot${rank}`];
 if(!Number.isInteger(rank)||rank<1||rank>3||!Number.isInteger(slot?.max)||slot.max<1||!Number.isInteger(slot?.value)||slot.value<1||slot.value>slot.max||item.atWill||item.isCantrip||(item.system.cast?.focusPoints??0)!==0)return fail('需要当前1至3环的可用原生法术位。');
 if((options.messageMode??game.settings?.get('core','messageMode'))!=='public'||options.rollMode&&options.rollMode!=='publicroll')return fail('当前分弹接线只支持公开施法；私密模式请手工处理。');
 if(values(item.appliedOverlays).length||Object.keys(base.system.overlays??{}).length||base.flags?.['pf2e-toolbelt']?.actionable?.linked||base.flags?.['pf2e-toolbelt']?.linked)return fail('法术覆盖或另接宏尚未验证，未接管本次施法。');
 const data=base.system,damage=Object.values(data.damage??{});
 if(data.level?.value!==1||data.time?.value!=='1 to 3'||data.range?.value!=='120 feet'||data.area||data.duration?.sustained||data.duration?.value||data.heightening||data.rules?.length||damage.length!==1)return fail('原法术的施法、伤害或升环数据已改变，需要人工处理。');
 const d=damage[0],traits=values(data.traits?.value).sort();
 if(traits.join(',')!=='concentrate,force,manipulate'||d.type!=='force'||d.formula?.replace(/\s+/g,'')!=='1d4+1'||d.applyMod||d.category||values(d.kinds).join(',')!=='damage'||d.materials?.length)return fail('当前只接管已验证的原始力场伤害配置。');
 return {handled:true,eligible:true,rank,base};
}

/** Explicit sight confirmation supplements geometry; it never turns GM visibility
 * into character sight, admits hidden tokens, or bypasses a blocked ray. */
export function validateForceBarrageTargets({game,actor,token,targets,visibilityConfirmed=false}={}){
 const scene=token?.parent;
 if(!scene||game.scenes?.get(scene.id)!==scene||scene.tokens?.get(token.id)!==token||token.documentName!=='Token'||token.actor!==actor||token.hidden||!token.object||scene.grid?.type!==1||!['ft','feet','foot'].includes(String(scene.grid.units).toLowerCase())||actor.canSee===false||actor.hasCondition?.('blinded')||!visibilityConfirmed)throw Error('需要唯一公开来源Token、方格尺制场景及施法者能看见目标的确认。');
 if(!Array.isArray(targets)||targets.length<1||targets.length>6||new Set(targets.map(t=>t?.uuid)).size!==targets.length)throw Error('需要1至6个不同的实际目标Token。');
 for(const target of targets){
  if(target?.documentName!=='Token'||target.parent!==scene||scene.tokens.get(target.id)!==target||!target.object||target.hidden||!['character','npc','familiar'].includes(target.actor?.type)||target.actor.isDead===true||['invisible','hidden','undetected','unnoticed'].some(c=>target.actor.hasCondition?.(c)))throw Error('目标已改变，或需要GM人工判断其可见性及生物身份。');
  if(!Number.isFinite(token.elevation)||target.elevation!==token.elevation||target.level!==token.level)throw Error('不同高度或楼层的目标需要人工确认三维射线。');
  const distance=token.object.distanceTo?.(target.object);
  if(!Number.isFinite(distance)||distance<0||distance>120)throw Error('目标不在本次原生测得的120尺射程内。');
  if(!target.object.center||token.object.checkCollision?.(target.object.center,{origin:token.object.center,type:'sight',mode:'any'})!==false)throw Error('无法确认施法者至目标的视线。');
 }
 return targets;
}

/** Missile arithmetic belongs to the hash-verified Workbench adapter. */
export function validateForceBarrageAllocation({targets,allocations,missiles}={}){
 if(!Number.isSafeInteger(missiles)||missiles<1||missiles>6||!Array.isArray(targets)||!Array.isArray(allocations)||allocations.length!==targets.length||new Set(targets.map(t=>t?.uuid)).size!==targets.length)throw Error('分弹目标或总弹数无效。');
 const wanted=new Set(targets.map(t=>t.uuid));let total=0;
 for(const a of allocations){if(!wanted.delete(a?.targetUuid)||!Number.isSafeInteger(a.count)||a.count<0||a.count>missiles)throw Error('每个目标的弹数必须是不重复的非负整数。');total+=a.count;}
 if(wanted.size||total!==missiles)throw Error(`本次必须分配全部${missiles}枚飞弹。`);
 return allocations.map(({targetUuid,count})=>({targetUuid,count}));
}
