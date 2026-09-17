import {isCurrentDisruptToken} from './disrupt-prey-rules.mjs';
import {MODULE_ID} from './rules.mjs';
export const GLIMPSE_SOURCES=Object.freeze({glimpse:'Compendium.pf2e.actionspf2e.Item.tuZnRWHixLArvaIf',aura:'Compendium.pf2e.classfeatures.Item.0x76o5OxgEmvqIDp',resistance:'Compendium.pf2e.feat-effects.Item.DawVHfoPKbPJsz4k',weight:'Compendium.pf2e.feats-srd.Item.2c9awqDem5OLK47S'});
export const glimpseSourceId=item=>item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId;
export const glimpseClaims=combatant=>combatant?.flags?.[MODULE_ID]?.glimpseClaims??[];
export function findGlimpseClaim(game,nonce){
 const matches=Array.from(game.combats?.values?.()??[]).flatMap(combat=>Array.from(combat.turns??[]).flatMap(combatant=>glimpseClaims(combatant).filter(claim=>claim.nonce===nonce).map(claim=>({combat,combatant,claim}))));
 return matches.length===1?matches[0]:null;
}
export function provenGlimpseReactionCard(message,actor,game){
 const proof=message?.flags?.[MODULE_ID]?.glimpseUse,bound=findGlimpseClaim(game,proof?.nonce),claim=bound?.claim;
 const user=game.users?.get?.(message?.author?.id??message?.user?.id??message?.user),origin=message?.flags?.pf2e?.origin;
 if(!claim||!['paid','native','followup','done','uncertain'].includes(claim.status)||game.messages?.get(message.id)!==message||message.rolls?.length||message.speaker?.actor!==actor.id||claim.actorUuid!==actor.uuid||claim.messageId!==message.id||claim.userId!==user?.id||actor.testUserPermission?.(user,'OWNER')!==true||claim.claimKey!==`glimpse:${claim.nonce}`||proof.claimKey!==claim.claimKey||claim.itemUuid!==origin?.uuid||origin.actor!==actor.uuid||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==claim.tokenUuid)return null;
 const item=actor.items.get?.(claim.itemUuid.split('.').at(-1));if(glimpseSourceId(item)!==GLIMPSE_SOURCES.glimpse)return null;
 return {...claim,cost:1,slug:'glimpse-of-redemption',checkId:message.id};
}
export function glimpseEncounter(token,game){
 const matches=Array.from(game.combats?.values?.()??[]).filter(c=>c.started&&c.turns?.some(t=>t.token?.uuid===token?.uuid&&t.actor?.uuid===token?.actor?.uuid));
 if(matches.length!==1)throw Error('救赎瞥视需要唯一的实际进行中遭遇。');
 const combat=matches[0],index=combat.turns.findIndex(t=>t.token?.uuid===token.uuid),combatant=combat.turns[index];
 if(!Number.isInteger(combat.round)||!Number.isInteger(combat.turn))throw Error('救赎瞥视遭遇回合不可验证。');
 return {combat,combatant,epoch:`${combat.id}:${combat.round-(index>combat.turn?1:0)}`};
}
export function glimpseCandidates(context,game){
 if(!context?.verified)return [];
 const {actor:ally,token:victim,attacker:enemy}=context;
 if(Array.from(ally.items?.values?.()??[]).some(i=>glimpseSourceId(i)===GLIMPSE_SOURCES.resistance))return [];
 try{
  const encounter=glimpseEncounter(victim,game).combat;
  if(glimpseEncounter(enemy,game).combat!==encounter)return [];
  return Array.from(victim.parent.tokens.values()).flatMap(token=>{
   const actor=token.actor,items=Array.from(actor?.items?.values?.()??[]),ability=items.find(i=>glimpseSourceId(i)===GLIMPSE_SOURCES.glimpse&&i.system?.actionType?.value==='reaction');
   if(!isCurrentDisruptToken(token,game)||actor?.type!=='character'||actor.level!==5||actor.uuid===ally.uuid||actor.canAct!==true||actor.isDead||actor.hasCondition?.('unconscious')||!ability||!items.some(i=>glimpseSourceId(i)===GLIMPSE_SOURCES.aura)||items.some(i=>glimpseSourceId(i)===GLIMPSE_SOURCES.weight)||actor.isEnemyOf?.(enemy.actor)!==true||actor.isAllyOf?.(ally)!==true)return [];
   const aura=token.auras?.get?.('champions-aura');if(!aura||aura.containsToken?.(enemy)!==true||aura.containsToken?.(victim)!==true)return [];
   try{const bound=glimpseEncounter(token,game);return bound.combat===encounter?[{actor,token,ability,...bound}]:[]}catch{return []}
  });
 }catch{return []}
}
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
 if(!roll||!Number.isFinite(roll.total)||roll.total<0||roll.options?.splashOnly||roll.options?.evaluatePersistent||!Array.isArray(roll.instances)||!roll.instances.length||c?.type!=='damage-roll'||!['attack','save'].includes(c.sourceType)||!['weapon','melee','spell'].includes(o?.type))return no('not-native-attack-damage');
 const raw=roll.toJSON?.();if(!raw||raw.evaluated!==true||raw.class!=='DamageRoll')return no('roll-evidence-unavailable');
 for(const target of [c.target,pf.target])if(target&&(target.actor!=null&&target.actor!==actor.uuid||target.token!=null&&target.token!==token.uuid))return no('source-target-mismatch');
 const targets=message.flags?.['pf2e-toolbelt']?.targetHelper?.targets;
 if(Array.isArray(targets)&&targets.length&&!targets.includes(token.uuid))return no('source-target-mismatch');
 if(![c.target,pf.target].some(t=>t?.token===token.uuid)&&!(Array.isArray(targets)&&targets.includes(token.uuid)))return no('recipient-not-recorded');
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
export async function resolveGlimpseSource({game,fromUuid=globalThis.fromUuid,actor,params,source}={}){
 try{
  if(!source||typeof params?.damage==='number'||params?.final||params?.skipIWR||params?.damage?.options?.evaluatePersistent||params?.damage?.options?.splashOnly)return no('native-source-unavailable');
  const result=await inspect({game,fromUuid,actor,token:params.token?.document??params.token,message:game.messages.get(source.messageId),rollIndex:source.rollIndex});
  if(result.verified&&(!params.item||params.item.uuid!==result.item.uuid||params.item.actor?.uuid!==result.item.actor.uuid))return no('application-item-mismatch');
  return result;
 }catch{return no('source-document-unavailable');}
}
export async function validateGlimpseSource({game,fromUuid=globalThis.fromUuid,snapshot}={}){
 try{
  if(snapshot?.schema!==1||!/^[a-f0-9]{64}$/.test(snapshot.evidence??''))return no('snapshot-invalid');
  const token=await fromUuid(snapshot.tokenUuid),result=await inspect({game,fromUuid,actor:token?.actor,token,message:game.messages.get(snapshot.damageMessageId),rollIndex:snapshot.rollIndex});
  return result.verified&&serial(result.snapshot)===serial(snapshot)?result:no('source-evidence-changed');
 }catch{return no('source-document-unavailable');}
}
