import {isCurrentDisruptToken} from './disrupt-prey-rules.mjs';
const no=unsupportedReason=>({verified:false,unsupportedReason}),values=c=>Array.from(c?.values?.()??c??[]);
const serial=value=>JSON.stringify(value,(_k,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))):v);
const hash=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),n=>n.toString(16).padStart(2,'0')).join('');
function helperEvidence(helper){
 if(!helper)return null;
 const {applied,...stable}=helper;
 const defaults={area:null,author:null,expended:0,item:null,isRegen:false,options:[],private:false,saveVariants:{},splashIndex:-1,traits:[],splashTargets:[],targets:[]};
 for(const [key,value]of Object.entries(defaults))if(stable[key]===undefined)stable[key]=value;
 if(stable.saveVariants)stable.saveVariants=Object.fromEntries(Object.entries(stable.saveVariants).map(([key,v])=>[key,{basic:false,saves:{},...v}]));
 return stable;
}
function evidence(message,index){
 const roll=message.rolls[index]?.toJSON?.();
 return serial({author:message.author?.id??message.user?.id??message.user,speaker:message.speaker,pf:message.flags?.pf2e,helper:helperEvidence(message.flags?.['pf2e-toolbelt']?.targetHelper),roll:typeof roll==='string'?JSON.parse(roll):roll});
}
async function inspect({game,fromUuid,actor,token,message,rollIndex}){
 if(!isCurrentDisruptToken(token,game)||token.actor.uuid!==actor?.uuid)return no('recipient-unavailable');
 if(!message?.id||game.messages.get(message.id)!==message||!message.isDamageRoll||rollIndex!==0)return no('not-primary-attack-damage');
 if(message.flags?.['pf2e-toolbelt']?.betterChat?.mergeDamage)return no('combined-attacks-unsupported');
 const roll=message.rolls?.[rollIndex],pf=message.flags?.pf2e,c=pf?.context,o=pf?.origin;
 if(!roll||!Number.isFinite(roll.total)||roll.total<0||roll.options?.splashOnly||roll.options?.evaluatePersistent||!Array.isArray(roll.instances)||!roll.instances.length||c?.type!=='damage-roll'||c.sourceType!=='attack'||!['weapon','melee','spell'].includes(o?.type))return no('not-native-attack-damage');
 const raw=roll.toJSON?.();if(!raw||raw.evaluated!==true||raw.class!=='DamageRoll')return no('roll-evidence-unavailable');
 for(const target of [c.target,pf.target])if(target&&(target.actor!=null&&target.actor!==actor.uuid||target.token!=null&&target.token!==token.uuid))return no('source-target-mismatch');
 const targets=message.flags?.['pf2e-toolbelt']?.targetHelper?.targets;
 if(Array.isArray(targets)&&targets.length&&!targets.includes(token.uuid))return no('source-target-mismatch');
 const before=evidence(message,rollIndex),s=message.speaker,attacker=await fromUuid(`Scene.${s?.scene}.Token.${s?.token}`);
 if(!isCurrentDisruptToken(attacker,game)||attacker.parent!==token.parent||attacker.actor.id!==s.actor||attacker.actor.uuid!==o.actor||c.actor!=null&&c.actor!==attacker.actor.id||c.token!=null&&c.token!==attacker.id)return no('attacker-provenance-mismatch');
 const authorId=message.author?.id??message.user?.id??message.user,user=game.users?.get?.(authorId);
 const authorized=()=>!!user&&game.users.get(authorId)===user&&(user.isGM===true||attacker.actor.testUserPermission?.(user,'OWNER')===true);
 if(!authorized())return no('attacker-author-unauthorized');
 let item=await fromUuid(o.uuid);
 // Generated native unarmed items are prepared Strike documents, not embedded
 // inventory. Accept only that exact actor-owned UUID, never a display name.
 if(!item){const candidates=values(attacker.actor.system?.actions).flatMap(strike=>[strike,...values(strike.altUsages)]).map(strike=>strike?.item).filter(i=>i?.uuid===o.uuid&&i.actor===attacker.actor);const unique=[...new Set(candidates)];if(unique.length===1)item=unique[0];}
 if(item?.actor?.uuid!==attacker.actor.uuid||item.uuid!==o.uuid||item.type!==o.type)return no('attack-item-mismatch');
 if(o.type!=='spell'&&(pf.strike?.damaging!==true||pf.strike.actor!==attacker.actor.uuid))return no('native-strike-unavailable');
 const digest=await hash(before);
 if(game.messages.get(message.id)!==message||before!==evidence(message,rollIndex))return no('source-evidence-changed');
 if(!authorized())return no('attacker-author-unauthorized');
 const snapshot={schema:1,damageMessageId:message.id,rollIndex,actorUuid:actor.uuid,tokenUuid:token.uuid,attackerActorUuid:attacker.actor.uuid,attackerTokenUuid:attacker.uuid,itemUuid:item.uuid,evidence:digest};
 return {verified:true,snapshot,actor,token,attacker,item,message};
}
/** source is supplied from the real local cycle WeakMap. A serialized source is
 * not authorization to spend; the provider authenticates a live native scope. */
export async function resolveDeflectionSource({game,fromUuid=globalThis.fromUuid,actor,params,source}={}){
 try{
  if(!source||typeof params?.damage==='number'||params?.final||params?.damage?.options?.evaluatePersistent||params?.damage?.options?.splashOnly)return no('native-source-unavailable');
  const result=await inspect({game,fromUuid,actor,token:params.token?.document??params.token,message:game.messages.get(source.messageId),rollIndex:source.rollIndex});
  if(result.verified&&(!params.item||params.item.uuid!==result.item.uuid||params.item.actor?.uuid!==result.item.actor.uuid))return no('application-item-mismatch');
  return result;
 }catch{return no('source-document-unavailable');}
}
export async function validateDeflectionSource({game,fromUuid=globalThis.fromUuid,snapshot}={}){
 try{
  if(snapshot?.schema!==1||!/^[a-f0-9]{64}$/.test(snapshot.evidence??''))return no('snapshot-invalid');
  const token=await fromUuid(snapshot.tokenUuid),result=await inspect({game,fromUuid,actor:token?.actor,token,message:game.messages.get(snapshot.damageMessageId),rollIndex:snapshot.rollIndex});
  return result.verified&&serial(result.snapshot)===serial(snapshot)?result:no('source-evidence-changed');
 }catch{return no('source-document-unavailable');}
}
