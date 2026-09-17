import {getSourceId} from './native-context.mjs';
import {occupiedTraits} from './eat-fortune.mjs';

export const HALFLING_LUCK_SOURCE='Compendium.pf2e.feats-srd.Item.ZbRVqf14RTJJIZXG';
export const isHalflingLuckItem=item=>item?.type==='feat'&&getSourceId(item)===HALFLING_LUCK_SOURCE;
const no=reason=>({eligible:false,reason}),outcomes=['criticalFailure','failure','success','criticalSuccess'];
const options=value=>value==null?[]:value instanceof Set?[...value]:Array.isArray(value)?value:null;
const canonical=value=>JSON.stringify(value,(_key,entry)=>entry&&typeof entry==='object'&&!Array.isArray(entry)?Object.fromEntries(Object.keys(entry).sort().map(k=>[k,entry[k]])):entry);

/** Admission only: no choice, resource update, dice throw, or consequence replay.
 * `manual-*` is an automation/evidence boundary, not a prohibition in the rules.
 * Pass the original local Check invocation and its native callback draft. A
 * serialized/remote claim or an old card alone is not that invocation proof.
 * requestedCreateMessage must be the original caller's snapshot, taken before
 * the shared pipeline forces an internal draft. No private-route proof field
 * bypasses this boundary: Toolbelt can keep a private row from a public draft.
 */
export function assessHalflingLuck({game,actor,item,user,check,context,roll,card,requestedCreateMessage=context?.createMessage!==false}={}){
 try{
  if(game?.system?.id!=='pf2e'||game.system.version!=='8.5.1')return no('manual-native-compatibility');
  if(actor?.type!=='character'||actor.isToken||game.actors?.get(actor.id)!==actor)return no('actor-unavailable');
  if(!user?.active||game.users?.get(user.id)!==user||actor.testUserPermission?.(user,'OWNER')!==true)return no('not-owner');
  if(actor.canAct!==true||actor.isDead===true)return no('cannot-act');
  if(!isHalflingLuckItem(item)||item.actor!==actor||actor.items?.get(item.id)!==item)return no('item-unavailable');
  const frequency=item.system?.frequency;
  if(item.system?.actionType?.value!=='free'||frequency?.max!==1||frequency.per!=='day'||!item.system.traits?.value?.includes('fortune')||item.system.selfEffect||item.crafting)return no('unsupported-feat');
  // PF preparation supplies value from max. Never infer a usable charge from
  // raw _source or from max when the live prepared value is absent.
  if(frequency.value!==1)return no('daily-use-unavailable');
  const RollClass=globalThis.CONFIG?.Dice?.rolls?.find(C=>C.name==='CheckRoll'),DieClass=globalThis.foundry?.dice?.terms?.Die,MessageClass=globalThis.CONFIG?.ChatMessage?.documentClass,CheckClass=game.pf2e?.CheckModifier;
  const pf=card?.flags?.pf2e?.context;
  if(!RollClass||!DieClass||!MessageClass||!CheckClass||!(check instanceof CheckClass)||!(roll instanceof RollClass)||!(card instanceof MessageClass)||card.isCheckRoll!==true||roll._evaluated!==true||!Number.isFinite(roll.total)||!pf||!context)return no('manual-native-evidence');
  // CheckContext can legitimately prepare a contextual actor clone. The live
  // resource actor is checked above; the original roller must retain its UUID.
  const self=context.origin?.self===true?context.origin:context.target?.self===true?context.target:null;
  const recordedSelf=context.origin?.self===true?pf.origin:pf.target;
  if(context.actor?.uuid!==actor.uuid||self?.actor?.uuid!==actor.uuid||recordedSelf?.actor!==actor.uuid||pf.actor!==actor.id||card.speaker?.actor!==actor.id||card.author?.id!==user.id||roll.options?.rollerId!==user.id||pf.type!==context.type||roll.options.type!==context.type||card.flags.pf2e.modifierName!==check.slug||roll.options.totalModifier!==check.totalModifier)return no('manual-native-evidence');
  const token=context.token;
  if(token){
   if(token.documentName!=='Token'||token.actorLink!==true||token.actor?.uuid!==actor.uuid||game.scenes?.get(token.parent?.id)!==token.parent||token.parent.tokens?.get(token.id)!==token||self.token!==token||pf.token!==token.id||recordedSelf.token!==token.uuid||card.speaker.token!==token.id||card.speaker.scene!==token.parent.id)return no('manual-native-evidence');
  }else if(pf.token||self.token||recordedSelf.token||card.speaker.token)return no('manual-native-evidence');
  if(!Array.isArray(card.rolls)||card.rolls.length!==1||!(card.rolls[0] instanceof RollClass)||typeof roll.toJSON!=='function'||canonical(card.rolls[0].toJSON())!==canonical(roll.toJSON()))return no('manual-native-evidence');
  if(!Array.isArray(context.domains)||!context.domains.length||!context.domains.every(d=>typeof d==='string')||canonical(context.domains)!==canonical(pf.domains)||canonical(context.domains)!==canonical(roll.options.domains))return no('manual-native-evidence');
  const liveOptions=options(context.options),savedOptions=options(pf.options);
  if(!liveOptions||!savedOptions||![...liveOptions,...savedOptions].every(v=>typeof v==='string'))return no('manual-native-evidence');
  const allOptions=new Set([...liveOptions,...savedOptions]),traits=[...context.traits??[],...pf.traits??[]];
  if(requestedCreateMessage!==true)return no('manual-unproven-draft-privacy');
  // A failure-only player prompt can reveal a blind/secret result even before a
  // check message exists. Original Use currently has no private publication proof.
  if(allOptions.has('secret')||traits.some(t=>(typeof t==='string'?t:t?.name)==='secret')||['blind','gm','self'].includes(context.messageMode)||['blind','gm','self'].includes(pf.messageMode)||card.blind===true||card.whisper?.length>0||game.pf2e.settings?.metagame?.results===false)return no('manual-private-check');
  if(context.messageMode!=='public'||pf.messageMode!=='public'||card.blind!==false||!Array.isArray(card.whisper)||game.pf2e.settings?.metagame?.results!==true)return no('manual-unknown-privacy');
  if(!['skill-check','saving-throw'].includes(context.type))return no('not-skill-or-save');
  if(!Number.isFinite(context.dc?.value)||!Number.isFinite(pf.dc?.value))return no('manual-no-dc');
  const degree=roll.options.degreeOfSuccess;
  if(context.dc.value!==pf.dc.value||!Number.isInteger(degree)||degree<0||degree>3||pf.outcome!==outcomes[degree]||context.outcome!==outcomes[degree])return no('manual-native-evidence');
  if(degree>1)return no('not-failed');
  if(context.isReroll||pf.isReroll||roll.isReroll||roll.options.isReroll||allOptions.has('check:reroll'))return no('already-rerolled');
  if(!Array.isArray(context.substitutions)||!Array.isArray(pf.substitutions))return no('manual-native-evidence');
  if([...context.substitutions,...pf.substitutions].some(s=>s.selected===true))return no('manual-substitution');
  // Retain the original prepared actor and matching synthetics: native
  // extractRollTwice collapses simultaneous higher/lower sources into false.
  const occupied=new Set([...occupiedTraits(context),...occupiedTraits({...context,...pf,actor:context.actor,options:savedOptions})]);
  if(occupied.has('misfortune'))return no('manual-fortune-misfortune');
  if(occupied.has('fortune'))return no('fortune-occupied');
  const dice=roll.dice,die=dice?.[0],result=die?.results?.[0];
  if(!Array.isArray(dice)||dice.length!==1||!(die instanceof DieClass)||die.faces!==20||die.number!==1||die._evaluated!==true||!Array.isArray(die.modifiers)||die.modifiers.length!==0||die.results.length!==1||result?.active!==true||result.discarded||!Number.isInteger(result.result)||result.result<1||result.result>20||roll.options.dice!=='1d20'||roll.isRerollable!==true)return no('manual-nonordinary-d20');
  return {eligible:true,reason:null};
 }catch{return no('manual-native-evidence')}
}
