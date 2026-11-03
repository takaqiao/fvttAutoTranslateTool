import {MODULE_ID as ID} from './rules.mjs';
import {getSourceId} from './native-context.mjs';

export const FLAT_CHECK_HELPER='pf2e-flatcheck-helper';
const SURE_STRIKE='Compendium.pf2e.spell-effects.Item.fpGDAz2v5PG0zUSl';
const BLIND_FIGHT='Compendium.pf2e.feats-srd.Item.y2XeMe1F18lIyo59';
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const author=message=>message.author?.id??message.author??message.user?.id??message.user;
const clone=data=>structuredClone(data);
const dieTotal=value=>Number.isInteger(value)&&value>=1&&value<=20;

export function assertDualStrikeFlatCheckCompatibility(game){
 if(game.modules?.get('patreon-v3')?.active&&['attack','all'].includes(game.settings.get('patreon-v3','flatCheck')))throw Error('Patreon 的攻击平检会重复投骰；本次双击尚未开始，请GM先处理此平检设置冲突。');
}

function nativeExceptions(actor,options,checks){
 const target=checks.target;if(!target||!['concealed','hidden'].includes(target.type))return;
 const items=values(actor.items);
 const sureStrike=items.some(item=>item.type==='effect'&&getSourceId(item)===SURE_STRIKE&&!item.isExpired&&item.rules?.some(rule=>rule.key==='RollTwice'&&!rule.ignored&&rule.keep==='higher'&&rule.selector.includes('attack-roll')&&rule.test(options)));
 const blindFight=items.some(item=>item.type==='feat'&&getSourceId(item)===BLIND_FIGHT)||options.has('feat:blind-fight');
 if(sureStrike||blindFight&&target.type==='concealed'){delete checks.target;return;}
 if(!blindFight)return;
 // Utility's rendered adjustment list loses override/upgrade/downgrade mode.
 // Only signed additions can be replayed against Blind-Fight's lower base DC.
 const adjustments=target.dcAdjustments??[];
 if(adjustments.some(entry=>typeof entry.value!=='string'||!/^[-+]\d+(?:\.\d+)?$/.test(entry.value)))throw Error('盲斗的平检 DC 覆盖需要GM裁定；尚未攻击，不能重复投骰。');
 target.baseDc=5;target.finalDc=adjustments.reduce((dc,entry)=>dc+Number(entry.value),5);
}

function passedChecks(checks){
 const required=Object.values(checks);
 if(required.some(check=>!Number.isFinite(check.finalDc)))throw Error('平检 DC 尚未确认，需要GM裁定；不能重复投骰。');
 if(required.some(check=>check.finalDc>=21))return false;
 return required.every(check=>{
  if(check.finalDc<=1)return true;
  if(!dieTotal(check.roll))throw Error('平检结果尚未保存；不能重复投骰。');
  let total=check.roll;
  if(check.reroll?.keep==='low'||check.reroll?.keep==='high'){
   if(!dieTotal(check.reroll.oldRoll))throw Error('原平检重投记录无效；不能重复投骰。');
   total=check.reroll.keep==='low'?Math.min(total,check.reroll.oldRoll):Math.max(total,check.reroll.oldRoll);
  }
  return total>=check.finalDc;
 });
}

const binding=({message,operation,actor,request})=>({activityMessageId:message.id,nonce:operation.nonce,scope:operation.scope,actorUuid:actor.uuid,userId:operation.userId,tokenUuid:request.tokenUuid,targetUuid:request.targetUuid,targetActorUuid:request.targetActorUuid,weaponId:request.weaponId,altUsageType:request.altUsageType??'',map:request.map,index:request.flags.dualStrikeAttack.index});

/** The GM accepts only the saved result of this owner's exact Strike attempt. */
export function verifyDualStrikeFlatCheck({game,receipt,message,operation,actor,request,status}){
 const proof=receipt?.flags?.[ID]?.dualStrikeFlatCheck,expected=binding({message,operation,actor,request}),context=receipt?.flags?.pf2e?.context,checks=receipt?.flags?.[FLAT_CHECK_HELPER]?.flatchecks;
 if(!receipt?.id||game.messages.get(receipt.id)!==receipt||author(receipt)!==operation.userId||receipt.speaker?.actor!==actor.id||proof?.status!==status||Object.entries(expected).some(([key,value])=>proof[key]!==value)||context?.target?.token!==request.targetUuid||context?.target?.actor!==request.targetActorUuid||receipt.flags.pf2e.origin?.actor!==actor.uuid||receipt.blind!==true||!receipt.whisper?.length||values(receipt.whisper).some(id=>!game.users.get(id)?.isGM)||!checks||Array.isArray(checks))throw Error('双击平检凭据与本次操作者、来源或目标不符。');
 if(passedChecks(checks)!==(status==='passed'))throw Error('双击平检凭据的判定与原结果不符。');
 if(proof.nativeMessageId){
  const native=game.messages.get(proof.nativeMessageId),flat=native?.flags?.[ID]?.dualStrikeFlatCheckRoll,total=native?.rolls?.[0]?.total;
  if(!native||author(native)!==operation.userId||native.flags.pf2e?.context?.type!=='flat-check'||flat?.receiptId!==receipt.id||flat.nonce!==operation.nonce||native.rolls.length!==1||!dieTotal(total)||Object.values(checks).some(check=>check.finalDc>1&&check.finalDc<21&&check.roll!==total))throw Error('双击平检的原生投骰凭据无效，不能重复投骰。');
 }
 return {flatchecks:clone(checks),privacy:clone(proof.privacy),messageId:receipt.id};
}

/** Use a real non-roll weapon card so Utility Buttons computes its own checks.
 * All checks preventing this Strike share one native flat die at the highest DC. */
export async function prepareDualStrikeFlatChecks({game,Message,message,operation,actor,strike,token,target,request,options,guard}){
 if(!game.modules?.get(FLAT_CHECK_HELPER)?.active)return null;
 const item=strike.item;
 if(item.type!=='weapon'||typeof item.toMessage!=='function')throw Error('双击平检缺少原生武器意图接口，尚未投骰。');
 const raw=await item.toMessage(null,{create:false});guard();
 if(!raw||raw.rolls?.length)throw Error('双击平检没有取得无骰武器意图，尚未投骰。');
 const data=raw.toObject(),basePrivacy={messageMode:game.settings?.get?.('core','messageMode')??'public',blind:data.blind===true,whisper:[...data.whisper??[]]};delete data._id;
 const rollOptions=new Set([...actor.getRollOptions(strike.domains),...item.getRollOptions('item'),...target.actor.getSelfRollOptions('target'),...strike.options??[],...options,`map:increases:${request.map}`]);
 const distance=token.object.distanceTo(target.object);if(Number.isInteger(distance))rollOptions.add(`target:distance:${distance}`);rollOptions.add('target');
 const proof={...binding({message,operation,actor,request}),status:'pending',privacy:basePrivacy,nativeMessageId:null};
 const title=`第${proof.index+1}击：平检`;
 const whisper=values(game.users).filter(user=>user.isGM).map(user=>user.id);if(!whisper.length)throw Error('没有GM可保存双击平检凭据。');
 Object.assign(data,{author:game.user.id,speaker:{...data.speaker,actor:actor.id,scene:token.parent.id,token:token.id},blind:true,whisper,content:`<p>${title}</p>`});
 data.flags={...data.flags,pf2e:{...data.flags?.pf2e,context:{target:{token:target.uuid,actor:target.actor.uuid},options:[...rollOptions]}},[ID]:{usageGenerated:true,dualStrikeFlatCheck:proof},'xdy-pf2e-workbench':{noAutoDamageRoll:true}};
 const receipt=await Message.create(data,{notify:false,scroll:false});guard();
 if(!receipt?.id||game.messages.get(receipt.id)!==receipt)throw Error('双击平检凭据没有保存，尚未攻击，不能重复投骰。');
 const checks=clone(receipt.flags?.[FLAT_CHECK_HELPER]?.flatchecks??{});nativeExceptions(actor,rollOptions,checks);
 const required=Object.values(checks);
 if(required.some(check=>!Number.isFinite(check.finalDc)))throw Error('平检 DC 尚未确认，需要GM裁定；尚未攻击。');
 const secret=required.some(check=>check.secret),mode=secret?(game.user.isGM?'gm':'blind'):basePrivacy.messageMode;
 const privacyData=Message.applyMode({author:game.user.id},mode);proof.privacy={messageMode:mode,blind:privacyData.blind===true,whisper:[...privacyData.whisper??[]]};
 const needsRoll=required.filter(check=>check.finalDc>1&&check.finalDc<21).sort((a,b)=>b.finalDc-a.finalDc);
 if(!required.some(check=>check.finalDc>=21)&&needsRoll.some(check=>check.roll==null)){
  const existing=needsRoll.find(check=>dieTotal(check.roll));let total=existing?.roll;
  if(existing?.reroll?.keep==='low')total=Math.min(total,existing.reroll.oldRoll);else if(existing?.reroll?.keep==='high')total=Math.max(total,existing.reroll.oldRoll);
  if(total==null){
   if(!game.pf2e?.Check?.roll||!game.pf2e?.CheckModifier)throw Error('双击平检缺少原生平检接口，尚未攻击。');
   const dc=needsRoll[0].finalDc,flatOptions=new Set([`${ID}:native-operation:${operation.nonce}`,`${ID}:dual-strike-flat-check:${receipt.id}`,...secret?['secret']:[]]);
   let saved;
   const roll=await game.pf2e.Check.roll(new game.pf2e.CheckModifier('dual-strike-flat-check',{modifiers:[]}),{actor,token,type:'flat-check',title,domains:['flat-check'],options:flatOptions,dc:{value:dc,visible:!secret},target:{actor:target.actor,token:target},skipDialog:true,createMessage:false,messageMode:mode,rollTwice:false,substitutions:[]},null,async(evaluated,_outcome,native)=>{
    guard();if(saved||evaluated?._evaluated!==true||!dieTotal(evaluated.total))throw Error('双击平检结果不唯一或无效，不能重复投骰。');
    const flatData=native.toObject();delete flatData._id;flatData.author=game.user.id;
    flatData.flags={...flatData.flags,[ID]:{usageGenerated:true,dualStrikeFlatCheckRoll:{receiptId:receipt.id,nonce:operation.nonce}}};
    saved=await Message.create(flatData,{notify:false,scroll:false});guard();
    if(!saved?.id||game.messages.get(saved.id)!==saved)throw Error('双击原生平检没有保存，不能重复投骰。');
   });guard();
   if(!roll&&!saved){await receipt.update({[`flags.${ID}.dualStrikeFlatCheck.status`]:'cancelled',content:`<p>${title}已取消</p>`});guard();return {status:'cancelled'};}
   if(!saved||!dieTotal(roll?.total))throw Error('双击原生平检结果未确认，不能重复投骰。');
   total=roll.total;proof.nativeMessageId=saved.id;
   proof.privacy={messageMode:mode,blind:saved.blind===true,whisper:[...saved.whisper??[]]};
  }
  for(const check of needsRoll)if(check.roll==null)check.roll=total;
 }
 const passed=passedChecks(checks);proof.status=passed?'passed':'failed';guard();
 const flatchecks={...checks};
 // Foundry merges flag objects; an omitted target would keep the old check.
 if(receipt.flags[FLAT_CHECK_HELPER]?.flatchecks?.target&&!Object.hasOwn(checks,'target'))flatchecks['-=target']=null;
 await receipt.update({[`flags.${ID}.dualStrikeFlatCheck`]:proof,[`flags.${FLAT_CHECK_HELPER}.flatchecks`]:flatchecks,[`flags.${FLAT_CHECK_HELPER}.revealed`]:true,content:`<p>第${proof.index+1}击：平检${passed?'通过':'未通过'}</p>`});guard();
 const result=verifyDualStrikeFlatCheck({game,receipt,message,operation,actor,request,status:proof.status});
 return {...result,status:passed?'passed':'flat-check-failed'};
}
