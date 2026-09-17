import {sameSalubriousPrivacy,validateSalubriousPrivacy,treatmentPrivacyForPatient,salubriousPrivacyData} from './salubrious-privacy.mjs';
import {MODULE_ID} from './rules.mjs';
import {salubriousFeat,treatmentTiers,treatmentImmune,kissState} from './salubrious-kiss-rules.mjs';
export const identityKeys=['nonce','actorUuid','itemUuid','tokenUuid','targetUuid','targetActorUuid','userId','startedAt','dc','tier','skill','refocusNoteId'];
export const sameClaim=(a,b)=>!!a&&!!b&&identityKeys.every(k=>a[k]===b[k])&&sameSalubriousPrivacy(a.privacy,b.privacy);
export const claimOf=(actor,nonce)=>kissState(actor).claims?.find(c=>c.nonce===nonce);
export const marker=(stage,claim)=>`${MODULE_ID}:salubrious-${stage}:${claim.nonce}`;
export const outcomes=['criticalFailure','failure','success','criticalSuccess'];
export const fail=reason=>Error(`仙露三吻：${reason}；未知结果不会自动重复。`);
export function currentToken(token,game){const scene=token?.parent;return token?.documentName==='Token'&&game.scenes.get(scene?.id)===scene&&scene.tokens.get(token.id)===token&&!!token.actor&&(!token.actorLink||game.actors.get(token.actorId)===token.actor);}
export function assertSource({game,actor,item,token,user,privacy}){
 if(actor?.type!=='character'||game.actors.get(actor.id)!==actor||actor.isToken||!currentToken(token,game)||token.actor!==actor||actor.items.get(item?.id)!==item||salubriousFeat(actor)!==item)throw fail('当前角色、已持有专长或唯一来源Token已变');
 if(game.users.get(user?.id)!==user||!user.active||!actor.testUserPermission(user,'OWNER')||actor.isDead||actor.hasCondition?.('unconscious'))throw fail('原拥有者离线、失去权限或无法行动');
 validateSalubriousPrivacy({game,user,token,item,privacy});
}
export function assertPatient({game,actor,token,target,allowImmune=false,user=game.user}){
 if(!currentToken(target,game)||target.hidden&&!user?.isGM||target.parent!==token.parent||target.actor.modeOfBeing!=='living'||target.actor.isDead||target.actor!==actor&&!target.actor.isAllyOf?.(actor))throw fail('需要当前场景中的可见受伤活体自身或盟友');
 const hp=target.actor.hitPoints,negative=hp?.negativeHealing??target.actor.system?.attributes?.hp?.negativeHealing;
 if(!hp||!Number.isFinite(hp.value)||!Number.isFinite(hp.max)||hp.max<=0||negative!==false)throw fail('原生命能治疗资格未知或具有虚能治疗');
 if(hp.value>=hp.max&&!target.actor.hasCondition?.('wounded'))throw fail('患者未受伤');
 if((target.actor.attributes?.immunities??[]).some(i=>['healing','vitality','object-immunities','custom'].includes(i.type)||i.definition||i.exceptions?.length))throw fail('此患者的命能或自定义免疫尚不能由原生负数治疗可靠处理');
 const distance=token===target?0:token.object?.distanceTo?.(target.object);if(!Number.isFinite(distance)||distance>5||distance<0)throw fail('本次患者不在接触距离内');
 if(!allowImmune&&treatmentImmune(target.actor,game.time.worldTime))throw fail('患者仍有医疗暂时免疫');
}
export function assertClaimPrivacy({game,claim,token,item,target,user}){
 if(!claim.privacy)return;
 const note=game.messages.get(claim.refocusNoteId),p=note?.flags?.[MODULE_ID]?.avRefocusNote,author=note?.author?.id??note?.author??note?.user?.id??note?.user;
 if(!note||p?.kind!=='completion'||p.nonce!==claim.nonce||p.actorUuid!==claim.actorUuid||p.userId!==claim.userId||p.tokenUuid!==claim.tokenUuid||author!==claim.userId||note.speaker?.actor!==claim.actorUuid.split('.').at(-1)||`Scene.${note.speaker?.scene}.Token.${note.speaker?.token}`!==claim.tokenUuid)throw fail('没有本次私密模式的原生重新聚能完成卡');
 const original=salubriousPrivacyData({privacy:p.privacy,userId:claim.userId});
 if(!!note.blind!==original.blind||JSON.stringify([...(note.whisper??[])].sort())!==JSON.stringify([...original.whisper].sort()))throw fail('原始重新聚能卡的受众已改变');
 const expected=treatmentPrivacyForPatient({game,user,token,item,target,privacy:p.privacy});
 if(!sameSalubriousPrivacy(expected,claim.privacy))throw fail('患者或原始重新聚能模式要求更严格受众');
}
export async function contextFor({game,fromUuid,claim,allowImmune=false}){
 if(!claim||!/^[A-Za-z0-9_-]{1,80}$/.test(claim.nonce??'')||claim.skill!=='occultism'||!Number.isFinite(claim.startedAt)||game.time.worldTime<claim.startedAt||game.time.worldTime>=claim.startedAt+3600)throw fail('医疗认领或活动时间无效');
 const [actor,item,token,target]=await Promise.all([claim.actorUuid,claim.itemUuid,claim.tokenUuid,claim.targetUuid].map(uuid=>fromUuid(uuid))),user=game.users.get(claim.userId);
 assertSource({game,actor,item,token,user,privacy:claim.privacy});assertPatient({game,actor,token,target,allowImmune,user});assertClaimPrivacy({game,claim,token,item,target,user});
 if(target.actor.uuid!==claim.targetActorUuid||!treatmentTiers(actor).some(t=>t.tier===claim.tier&&t.dc===claim.dc))throw fail('患者或所选神秘等级DC已改变');
 return {actor,item,token,target,user};
}
