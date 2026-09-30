import {MODULE_ID} from './rules.mjs';
import {getDisruptPreyMeleeOptions,isCurrentDisruptToken} from './disrupt-prey-rules.mjs';

const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const optionsOf=message=>new Set(message?.flags?.pf2e?.context?.options??[]);
const author=message=>message?.author?.id??message?.author??message?.user?.id??message?.user;
const marker=(kind,claim)=>`${MODULE_ID}:disrupt-${kind}:${claim.nonce}`;
const event=(game,kind)=>({ctrlKey:false,metaKey:false,shiftKey:game.user.settings?.[kind==='attack'?'showCheckDialogs':'showDamageDialogs']??true});
const proof=claim=>({nonce:claim.nonce,claimKey:claim.claimKey,actorUuid:claim.actorUuid,weaponKey:claim.weaponKey});
const sameProof=(value,claim)=>!!value&&Object.entries(proof(claim)).every(([key,expected])=>value[key]===expected);
const failure=text=>Error(`扰乱狩猎：${text}；不会自动重掷。`);

function checkOwner({game,actor,token,target,claim}){
 if(actor?.type!=='character'||!game.user?.id||game.user.id!==claim?.userId||game.users?.get(claim.userId)!==game.user||!actor.testUserPermission?.(game.user,'OWNER'))throw failure('必须由确切反应拥有者的客户端执行');
 if(!isCurrentDisruptToken(token,game)||!isCurrentDisruptToken(target,game)||token.actor!==actor||token.parent!==target.parent||claim.actorUuid!==actor.uuid||claim.tokenUuid!==token.uuid||claim.targetActorUuid!==target.actor.uuid||(claim.targetTokenUuid??claim.targetUuid)!==target.uuid||claim.targetUuid!=null&&claim.targetUuid!==target.uuid)throw failure('当前角色、Token或认领目标来源不匹配');
 if(typeof claim.nonce!=='string'||!claim.nonce||typeof claim.claimKey!=='string'||!claim.claimKey||typeof claim.weaponKey!=='string'||!Number.isInteger(claim.map)||claim.map<0||claim.map>2)throw failure('反应认领标识或MAP无效');
}
function currentOption(context){
 checkOwner(context);const {claim,option}=context;
 if(option?.key!==claim.weaponKey||option.map!==undefined&&option.map!==claim.map)throw failure('武器选择或MAP与认领不一致');
 const current=getDisruptPreyMeleeOptions(context).find(candidate=>candidate.key===claim.weaponKey);
 if(!current||typeof current.strike.variants?.[claim.map]?.roll!=='function')throw failure('该近战用法已不可用，或猎物、持握及触及发生变化');
 return current;
}
function rollData(roll){
 if(typeof roll==='string')return JSON.parse(roll);
 return roll;
}
function checkCardSource(message,context,current,{damage=false,stored=false}={}){
 const {game,actor,token,target,claim}=context,pf=message?.flags?.pf2e,c=pf?.context,o=optionsOf(message),r=rollData(message?.rolls?.[0]);
 if(stored&&(!message?.id||game.messages?.get(message.id)!==message))throw failure('原生消息没有作为同一真实文档保存');
 if(author(message)!==claim.userId||message.speaker?.actor!==actor.id||message.speaker?.token!==token.id||message.speaker?.scene!==token.parent.id||pf?.origin?.actor!==actor.uuid||pf.origin.uuid!==current.itemUuid||pf.origin.type!==current.strike.item.type||c?.actor!==actor.id||c.token!==token.id||c.target?.actor!==target.actor.uuid||c.target.token!==target.uuid||c.mapIncreases!==claim.map)throw failure('原生卡的作者、武器、角色或目标来源不匹配');
 if(!r||!Number.isFinite(r.total)||r._evaluated!==true&&r.evaluated!==true||!message.rolls.length||!damage&&message.rolls.length!==1)throw failure('没有唯一可确认的原生检定结果');
 if(stored){
  const serialized=typeof r.toJSON==='function'?rollData(r.toJSON()):null;
  if(r._evaluated!==true||serialized?.evaluated!==true||serialized.total!==r.total||(damage?!message.isDamageRoll:!message.isCheckRoll))throw failure('保存后的原生掷骰尚未完成或来源类型不一致');
 }
 const kind=damage?'damage':'attack';
 if(c.type!==`${kind}-roll`||!o.has(marker(kind,claim))||[...o].some(value=>typeof value==='string'&&value.startsWith(`${MODULE_ID}:disrupt-${kind}:`)&&value!==marker(kind,claim))||!o.has('item:melee')||o.has('item:ranged')||!o.has(`item:id:${current.strike.item.id}`))throw failure('原生卡不属于这次确切近战调用');
 if(pf.target&&(pf.target.actor!==target.actor.uuid||pf.target.token!==target.uuid))throw failure('原生卡的额外目标记录不一致');
 const helperTargets=message.flags?.['pf2e-toolbelt']?.targetHelper?.targets;
 if(Array.isArray(helperTargets)&&helperTargets.some(uuid=>uuid!==target.uuid))throw failure('原生卡含有其他伤害目标');
 const degree=outcomes.indexOf(c.outcome);
 if(degree<0||Number.isInteger(r.options?.degreeOfSuccess)&&r.options.degreeOfSuccess!==degree)throw failure('最终原生成功度不一致或未能确认');
 if(damage){
  if(c.sourceType!=='attack'||!pf.strike?.damaging||pf.strike.actor!==actor.uuid||(pf.strike.altUsage??null)!==current.usage||o.has('action:reaction')||o.has('trait:reaction')||!o.has(`${MODULE_ID}:bear-attack:${context.attack.id}`))throw failure('原生伤害来源、用法或收费标记不一致');
 }else if(c.action!=='strike'||c.origin?.actor!==actor.uuid||c.origin.token!==token.uuid||(c.altUsage??null)!==current.usage||!o.has('action:reaction')||!o.has('action:disrupt-prey'))throw failure('原生攻击不是本次反应Strike');
 return degree;
}
function reactionFlavor(flavor){
 let found=false;
 const result=String(flavor??'').replace(/(<h4\b[^>]*\bclass=["'][^"']*\baction\b[^"']*["'][^>]*>)([\s\S]*?)(<\/h4>)/i,(_all,start,body,end)=>{
  found=true;const glyph=/(<span\b[^>]*\bclass=["'][^"']*\baction-glyph\b[^"']*["'][^>]*>)[\s\S]*?(<\/span>)/i;
  return start+(glyph.test(body)?body.replace(glyph,(_all,open,close)=>open+'R'+close):body+'<span class="action-glyph">R</span>')+end;
 });
 if(!found)throw failure('原生攻击标题结构无法确认');return result;
}

/** Publish inside the one final native callback so Knowledge's completion can
 * observe the card. Identical repeats share its promise; conflicting callbacks
 * are uncertain, never a reason to select the latest card or roll again. */
export async function rollDisruptPreyAttack({game,actor,token,target,option,claim,createMessage=data=>globalThis.ChatMessage.create(data)}){
 const context={game,actor,token,target,option,claim:{...claim}},current=currentOption(context);
 let fingerprint,publication,callbackError;
 const callback=async(_roll,_outcome,raw)=>{
  try{
   currentOption(context);
   if(typeof raw?.toObject!=='function'||raw.id&&game.messages.get(raw.id)===raw)throw failure('检定回调没有提供未发布的原生草稿');
   const data=raw.toObject();checkCardSource(data,context,current);delete data._id;
   const key=JSON.stringify(data);
   if(fingerprint!==undefined){if(key!==fingerprint)throw failure('同一次调用出现多个不一致草稿，结果不确定');return publication;}
   fingerprint=key;
   publication=Promise.resolve().then(async()=>{
    if(callbackError)throw callbackError;currentOption(context);
    data.author=context.claim.userId;data.flavor=reactionFlavor(data.flavor);
    data.flags={...data.flags,'xdy-pf2e-workbench':{...data.flags?.['xdy-pf2e-workbench'],noAutoDamageRoll:true},[MODULE_ID]:{...data.flags?.[MODULE_ID],disruptPreyReaction:proof(context.claim)}};
    const message=await createMessage(data);checkOwner(context);checkCardSource(message,context,current,{stored:true});
    if(!sameProof(message.flags?.[MODULE_ID]?.disruptPreyReaction,context.claim)||message.flags?.['xdy-pf2e-workbench']?.noAutoDamageRoll!==true)throw failure('保存后的反应来源证明不一致');
    return message;
   });
   return await publication;
  }catch(error){callbackError=error;throw error;}
 };
 let rolled;
 try{rolled=await current.strike.variants[context.claim.map].roll({target:target.object,altUsage:current.usage??undefined,options:new Set(['action:reaction','action:disrupt-prey','hunted-prey',marker('attack',context.claim)]),event:event(game,'attack'),createMessage:false,callback});}
 catch(error){if(publication)await publication.catch(()=>{});throw error;}
 if(callbackError)throw callbackError;
 if(!rolled||!publication)throw failure('原生攻击取消或没有可确认的结果');
 const message=await publication;checkOwner(context);checkCardSource(message,context,current,{stored:true});return message;
}

/** Native damage creates its own card and full context. Capture only this
 * invocation's nonce while its native promise is running, and always detach. */
export async function rollDisruptPreyDamage({game,actor,token,target,option,claim,attack,Hooks=globalThis.Hooks}){
 const context={game,actor,token,target,option,claim:{...claim},attack},current=currentOption(context);
 const degree=checkCardSource(attack,context,current,{stored:true});
 if(!sameProof(attack.flags?.[MODULE_ID]?.disruptPreyReaction,context.claim)||context.claim.checkId&&context.claim.checkId!==attack.id||![2,3].includes(degree))throw failure('没有属于本认领的已命中原生攻击');
 const method=degree===3?'critical':'damage';if(typeof current.strike[method]!=='function'||!Hooks?.on||!Hooks.off)throw failure('原生伤害方法或消息观察接口不可用');
 const cards=new Set();let captureError;
 const hook=Hooks.on('createChatMessage',message=>{
  if(!optionsOf(message).has(marker('damage',context.claim)))return;
  cards.add(message);
  try{checkCardSource(message,context,current,{damage:true,stored:true});}catch(error){captureError=error;}
 });
 try{
  const roll=await current.strike[method]({target:target.object,checkContext:attack.flags.pf2e.context,mapIncreases:context.claim.map,options:new Set(['hunted-prey',marker('damage',context.claim),`${MODULE_ID}:bear-attack:${attack.id}`]),createMessage:true,event:event(game,'damage')});
  checkOwner(context);
  if(captureError)throw captureError;
  if(!roll||cards.size!==1)throw failure('伤害取消或出现多个结果，无法确认唯一原生伤害卡');
  const [message]=cards;checkCardSource(message,context,current,{damage:true,stored:true});return message;
 }finally{Hooks.off('createChatMessage',hook);}
}
