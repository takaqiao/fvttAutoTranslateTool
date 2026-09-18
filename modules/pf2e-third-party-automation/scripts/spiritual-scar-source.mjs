import {isCurrentDisruptToken} from './disrupt-prey-rules.mjs';

const no=unsupportedReason=>({verified:false,unsupportedReason});
const serial=value=>JSON.stringify(value,(_key,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))):v);
const hash=async text=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),b=>b.toString(16).padStart(2,'0')).join('');
const fiend=actor=>new Set(actor?.traits??actor?.system?.traits?.value??[]).has('fiend');
const currentSourceToken=(token,game)=>!!(token?.documentName==='Token'&&game.scenes?.get(token.parent?.id)===token.parent&&token.parent?.tokens?.get(token.id)===token);
const values=c=>Array.from(c?.values?.()??c??[]);
const currentSourceActor=(actor,game)=>actor?.isToken===true
 ?currentSourceToken(actor.token,game)&&actor.token.actor===actor
 :game.actors?.get(actor?.id)===actor;
const preparedItems=actor=>values(actor.system?.actions).flatMap(strike=>[strike,...values(strike.altUsages)]).map(strike=>strike?.item);
const currentSourceItem=(item,actor)=>item?.actor===actor&&(actor.items?.get(item.id)===item||preparedItems(actor).includes(item));
const privacy=message=>Object.freeze({blind:message.blind===true,whisper:Object.freeze([...(message.whisper??[])])});
function helperEvidence(helper){
 if(!helper)return null;
 const {applied,...stable}=helper;
 const defaults={area:null,author:null,expended:0,item:null,isRegen:false,options:[],private:false,saveVariants:{},splashIndex:-1,traits:[],splashTargets:[],targets:[]};
 for(const[key,value]of Object.entries(defaults))if(stable[key]===undefined)stable[key]=value;
 if(stable.saveVariants)stable.saveVariants=Object.fromEntries(Object.entries(stable.saveVariants).map(([key,value])=>[key,{basic:false,saves:{},...value}]));
 return stable;
}
function evidence(message,index){
 return serial({author:message.author?.id??message.user?.id??message.user,speaker:message.speaker,pf:message.flags?.pf2e,
  privacy:privacy(message),helper:helperEvidence(message.flags?.['pf2e-toolbelt']?.targetHelper),roll:message.rolls[index].toJSON()});
}

/** An actual positive spirit instance, including an evaluated persistent tick.
 * Applying a newly created persistent condition is not taking its future damage. */
export function hasIncomingScarDamage(params){
 const roll=params?.damage;
 return !params?.final&&!params?.skipIWR&&typeof roll==='object'&&roll!==null&&Number.isFinite(roll.total)&&roll.total>0
  &&new Set(params.rollOptions??[]).has('origin:trait:fiend')&&Array.isArray(roll.instances)
  &&roll.instances.some(instance=>instance.type==='spirit'&&Number.isFinite(instance.total)&&instance.total>0&&(!instance.persistent||roll.options?.evaluatePersistent===true));
}

async function inspect({game,fromUuid,actor,token,message,rollIndex}){
 if(!isCurrentDisruptToken(token,game)||token.actor.uuid!==actor?.uuid)return no('recipient-unavailable');
 if(!message?.id||game.messages.get(message.id)!==message||!message.isDamageRoll||!Number.isSafeInteger(rollIndex)||rollIndex<0)return no('native-damage-card-unavailable');
 if(message.flags?.['pf2e-toolbelt']?.betterChat?.mergeDamage)return no('merged-source-unproven');
 const roll=message.rolls?.[rollIndex],pf=message.flags?.pf2e,context=pf?.context,origin=pf?.origin;
 if(!roll||!Number.isFinite(roll.total)||roll.total<0||!Array.isArray(roll.instances)||!roll.instances.length||context?.type!=='damage-roll')return no('native-damage-roll-unavailable');
 const raw=roll.toJSON?.();if(raw?.class!=='DamageRoll'||raw.evaluated!==true)return no('native-roll-evidence-unavailable');
 if(!Array.isArray(message.whisper??[])||(message.whisper??[]).some(id=>typeof id!=='string'))return no('source-privacy-unavailable');
 const before=evidence(message,rollIndex),speaker=message.speaker;
 let sourceToken=null;
 if(speaker?.token){
  if(typeof speaker.scene!=='string')return no('source-token-unavailable');
  sourceToken=await fromUuid(`Scene.${speaker.scene}.Token.${speaker.token}`);
  if(!currentSourceToken(sourceToken,game))return no('source-token-unavailable');
 }
 const sourceActor=origin?.actor?await fromUuid(origin.actor):sourceToken?.actor??(speaker?.actor?await fromUuid(`Actor.${speaker.actor}`):null);
 const sourceCurrent=()=>!!(sourceActor?.uuid&&currentSourceActor(sourceActor,game)&&sourceActor.id===speaker?.actor&&fiend(sourceActor)
  &&(!origin?.actor||origin.actor===sourceActor.uuid)&&(!sourceToken||currentSourceToken(sourceToken,game)&&sourceToken.actor===sourceActor)
  &&(context.actor==null||[sourceActor.id,sourceActor.uuid].includes(context.actor))
  &&(context.token==null||sourceToken&&[sourceToken.id,sourceToken.uuid].includes(context.token)));
 if(!sourceCurrent())return no('fiend-origin-unproven');
 const authorId=message.author?.id??message.user?.id??message.user,user=game.users?.get(authorId);
 const authorized=()=>!!user&&game.users.get(authorId)===user&&(user.isGM===true||sourceActor.testUserPermission?.(user,'OWNER')===true);
 if(!authorized())return no('source-author-unauthorized');
 let item=null;
 if(origin?.uuid){
  item=await fromUuid(origin.uuid);
  // Native unarmed Strikes can own prepared, non-embedded items.
  if(!item){
   const candidates=preparedItems(sourceActor).filter(item=>item?.uuid===origin.uuid&&item.actor===sourceActor),unique=[...new Set(candidates)];
   if(unique.length===1)item=unique[0];
  }
  if(!currentSourceItem(item,sourceActor)||item.uuid!==origin.uuid||origin.type!=null&&item.type!==origin.type)return no('source-item-unavailable');
 }
 const digest=await hash(before);
 if(game.messages.get(message.id)!==message||before!==evidence(message,rollIndex)||!isCurrentDisruptToken(token,game)||token.actor.uuid!==actor.uuid)return no('source-evidence-changed');
 if(!authorized()||!sourceCurrent())return no('source-author-or-fiend-changed');
 if(item&&(!currentSourceItem(item,sourceActor)||item.uuid!==origin.uuid||origin.type!=null&&item.type!==origin.type))return no('source-item-changed');
 const snapshot=Object.freeze({schema:1,kind:'spiritual-scar',damageMessageId:message.id,rollIndex,actorUuid:actor.uuid,tokenUuid:token.uuid,
  sourceActorUuid:sourceActor.uuid,sourceTokenUuid:sourceToken?.uuid??null,itemUuid:item?.uuid??null,evidence:digest});
 return {verified:true,snapshot,actor,token,sourceActor,sourceToken,item,message,privacy:privacy(message)};
}

/** Evidence validation only. The provider must authenticate its still-open
 * native call before spending resources; a caller-supplied snapshot cannot. */
export async function resolveSpiritualScarSource({game,fromUuid=globalThis.fromUuid,actor,params,source}={}){
 try{
  if(!source||!hasIncomingScarDamage(params))return no('native-fiend-spirit-input-unavailable');
  const result=await inspect({game,fromUuid,actor,token:params.token?.document??params.token,message:game.messages.get(source.messageId),rollIndex:source.rollIndex});
  if(result.verified&&(result.item?params.item?.uuid!==result.item.uuid||params.item.actor?.uuid!==result.sourceActor.uuid:params.item!=null))return no('application-item-mismatch');
  return result;
 }catch{return no('source-document-unavailable');}
}
export async function validateSpiritualScarSource({game,fromUuid=globalThis.fromUuid,snapshot}={}){
 try{
  if(snapshot?.schema!==1||snapshot.kind!=='spiritual-scar'||!/^[a-f0-9]{64}$/.test(snapshot.evidence??''))return no('snapshot-invalid');
  const token=await fromUuid(snapshot.tokenUuid),result=await inspect({game,fromUuid,actor:token?.actor,token,message:game.messages.get(snapshot.damageMessageId),rollIndex:snapshot.rollIndex});
  return result.verified&&serial(result.snapshot)===serial(snapshot)?result:no('source-evidence-changed');
 }catch{return no('source-document-unavailable');}
}
