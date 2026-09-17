import {MODULE_ID} from './rules.mjs';
import {isCurrentDisruptToken} from './disrupt-prey-rules.mjs';

const applicationPrefix=`${MODULE_ID}:disrupt-apply:`,sourcePrefix=`${MODULE_ID}:source:`;

/** PF2e 8.5 rules helper, limited to damage-received effects affecting the exact target. */
async function damageReceivedEffects({origin,target,item,options}){
 if(!origin||!target)return [];
 const domains=['damage-received'],test=[...options,...origin.getRollOptions(domains),...target.getSelfRollOptions('target')];
 const resolvables=item?(item.isOfType('spell')?{spell:item}:{weapon:item}):{};
 const deferred=origin.synthetics?.ephemeralEffects?.['damage-received']?.target??[];
 const results=await Promise.all(deferred.map(fn=>fn({test,resolvables})));
 return results.filter(effect=>effect!==null).map(effect=>{
  // Native deferred functions return item sources. Copy before decorating so a
  // module's cached source cannot retain the last application target or duration.
  const copy=structuredClone(effect);
  if(copy.type==='effect'){
   copy.system.context={origin:{actor:origin.uuid,token:null,item:null,spellcasting:null,rollOptions:[]},target:{actor:target.uuid,token:null},roll:null};
   copy.system.duration={value:-1,unit:'unlimited',expiry:null,sustained:false};
  }
  return copy;
 });
}

/**
 * Prepare the PF2e 8.5 damage-card application context for one proven target.
 * This neither applies damage nor claims/settles it. The caller must persist its
 * application claim, serialize actual application and verify its native receipt.
 * Native source: pf2e.mjs 53712-53751, 38277-38312 and 31512-31521.
 */
export async function prepareNativeTargetDamage({game,message,target,rollIndex=0,applicationOption,DamageRoll}={}){
 const fail=reason=>{throw Error(`Cannot prepare native target damage: ${reason}`)};
 const current=()=>{
  if(typeof message?.id!=='string'||!message.id||game?.messages?.get(message.id)!==message)fail('damage message is not the current document');
  if(!isCurrentDisruptToken(target,game))fail('target is not a current scene Token document');
  const context=message.flags?.pf2e?.context;
  if(context?.type!=='damage-roll'||context.target?.token!==target.uuid||context.target?.actor!==target.actor.uuid)fail('damage context does not name this exact target actor and token');
  return context;
 };
 const context=current();
 if(!Number.isInteger(rollIndex)||rollIndex<0)fail('roll index must be a nonnegative integer');
 if(typeof applicationOption!=='string'||!applicationOption.startsWith(applicationPrefix)||!/^[A-Za-z0-9_-]{1,80}$/.test(applicationOption.slice(applicationPrefix.length)))fail('invalid bounded application nonce');
 const options=[...(Array.isArray(context.options)?context.options:context.options==null?[]:fail('damage options must be an array'))];
 if(options.some(option=>typeof option!=='string'))fail('damage options must contain strings');
 const sourceOption=`${sourcePrefix}${message.id}:${rollIndex}`;
 if(options.some(option=>option.startsWith(sourcePrefix)&&option!==sourceOption||option.startsWith(applicationPrefix)&&option!==applicationOption))fail('conflicting damage provenance');
 DamageRoll??=game?.pf2e?.DamageRoll??globalThis.CONFIG?.Dice?.rolls?.find(cls=>typeof cls==='function'&&cls.name==='DamageRoll');
 const roll=message.rolls?.[rollIndex];
 if(typeof DamageRoll!=='function'||!(roll instanceof DamageRoll)||typeof roll.alter!=='function'||!Number.isFinite(roll.total))fail('roll must be an actual evaluated DamageRoll');
 const contextSnapshot=JSON.stringify(context),origin=message.actor,recipient=target.actor,item=message.item;
 const damage=roll.alter(1,0);
 if(!(damage instanceof DamageRoll)||!Number.isFinite(damage.total))fail('native alteration must retain a DamageRoll');
 const originOptions=options.filter(option=>option.startsWith('self:')).map(option=>option.replace(/^self\b/,'origin'));
 const effectOptions=item?.isOfType('affliction','condition','effect')?item.getRollOptions('item'):[];
 if(recipient.alliance&&origin)options.push(`origin:${recipient.alliance===origin.alliance?'ally':'enemy'}`);
 if(!options.some(option=>option.startsWith('target')))options.push(...recipient.getSelfRollOptions('target'));
 const effects=await damageReceivedEffects({origin,target:recipient,item,options});
 // A deferred effect may load a compendium over the network. Do not prepare an
 // application using a card/Token/roll that was removed or replaced meanwhile.
 if(JSON.stringify(current())!==contextSnapshot||message.rolls?.[rollIndex]!==roll||target.actor!==recipient||message.actor!==origin)fail('damage source changed while preparing ephemeral effects');
 const actor=recipient.getContextualClone(originOptions,effects);
 const rollOptions=new Set([
  ...options.filter(option=>!/^(?:self|target)(?::|$)/.test(option)),
  ...effectOptions,...originOptions,...actor.getSelfRollOptions(),sourceOption,applicationOption,
 ]);
 return {actor,params:{damage,token:target,item,skipIWR:false,rollOptions,shieldBlockRequest:false,outcome:context.outcome}};
}
