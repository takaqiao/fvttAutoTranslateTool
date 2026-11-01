import {MODULE_ID} from './rules.mjs';

const special=new Set(['precision','critical-hits','splash-damage']);
let serial=0;
function alternativeOptions(options,type){
 const result=new Set([...options].filter(option=>!option.startsWith('damage:type:')&&!['damage:category:physical','damage:category:energy'].includes(option)));
 result.add(`damage:type:${type}`);result.add('damage:category:physical');return result;
}

export function volatileSparkDamageParts(parts){
 const types=spell=>new Set(parts.filter(part=>(part.item.type==='spell')===spell).flatMap(part=>(part.roll.instances??[]).map(instance=>instance.type)));
 const spells=types(true),weapons=types(false);
 const redirects=category=>parts.filter(part=>part.item.type==='weapon').flatMap(part=>part.roll.options?.bypass?.[category]?.redirect??[]).filter(entry=>['piercing','bludgeoning'].includes(entry.from)&&['piercing','bludgeoning'].includes(entry.to)&&entry.from!==entry.to&&weapons.has(entry.from)&&!spells.has(entry.from)).map(({from,to})=>({from,to}));
 return {types:[...spells].filter(type=>!weapons.has(type)),mixedTypes:[...spells].filter(type=>weapons.has(type)),weaponRedirects:{immunity:redirects('immunity'),resistance:redirects('resistance')}};
}

/** Call only around the final synchronous native applyDamage invocation. PF2e
 * 8.6.1 computes IWR before its first await. The view is removed before its
 * promise is returned, so another damage application cannot inherit it. */
export function createVolatileSparkDamage({game,getRollContext}={}){
 function applyDamage(actor,params,next){
  if(params.final||params.skipIWR||game.pf2e?.settings?.iwr===false||typeof params.damage==='number')return next(params);
  const context=getRollContext?.(params.damage),message=context?game.messages.get(context.messageId):null;
  const proof=message?.flags?.[MODULE_ID]?.volatileSparkSpellDamage,activity=message?.flags?.[MODULE_ID]?.spellCombinationDamage;
  const target=params.token?.document??params.token;
  if(!proof||context.rollIndex!==0||target?.uuid!==proof.targetUuid||target?.actor?.uuid!==actor.uuid||activity?.activityMessageId!==proof.activityMessageId||activity.targetUuid!==proof.targetUuid||!activity.parts?.some(part=>part.itemUuid===proof.spellUuid)||!activity.attacks?.some(attack=>attack.weaponUuid===proof.weaponUuid)||!Array.isArray(proof.types))return next(params);
  const types=new Set(proof.types.filter(type=>!proof.mixedTypes?.includes(type))),attributes=actor.attributes,original=attributes.resistances;
  const redirects=category=>(proof.weaponRedirects?.[category]??[]).filter(entry=>['piercing','bludgeoning'].includes(entry.from)&&['piercing','bludgeoning'].includes(entry.to)&&entry.from!==entry.to&&!types.has(entry.from)&&!proof.mixedTypes?.includes(entry.from));
  const immunityRedirects=redirects('immunity'),resistanceRedirects=redirects('resistance');
  if(!types.size&&!immunityRedirects.length&&!resistanceRedirects.length)return next(params);
  const marker=`${MODULE_ID}:volatile-spark-iwr:${++serial}`,options=new Set(params.rollOptions??[]);options.add(marker);
  const view=original.map(resistance=>new Proxy(resistance,{
   get(target,key){
    if(key==='getDoubledValue')return facts=>{
     const value=target.getDoubledValue(facts);
     if(!facts.has(marker)||special.has(target.type)||![...types].some(type=>facts.has(`damage:type:${type}`)))return value;
     const alternatives=['piercing','bludgeoning'].map(type=>{
      const changed=alternativeOptions(facts,type);
      return Math.max(0,...original.filter(entry=>!special.has(entry.type)&&entry.test(changed)).map(entry=>entry.getDoubledValue(changed)));
     });
     return Math.min(value,...alternatives);
    };
    const value=Reflect.get(target,key,target);return typeof value==='function'?value.bind(target):value;
   },
  }));
  const descriptor=Object.getOwnPropertyDescriptor(attributes,'resistances');
  const rollOptions=params.damage.options,bypassDescriptor=Object.getOwnPropertyDescriptor(rollOptions,'bypass');
  if(immunityRedirects.length||resistanceRedirects.length){
   const bypass=rollOptions.bypass;
   Object.defineProperty(rollOptions,'bypass',{configurable:true,enumerable:true,writable:true,value:{
    immunity:{ignore:[],downgrade:[],...bypass?.immunity,redirect:[...bypass?.immunity?.redirect??[],...immunityRedirects]},
    resistance:{ignore:[],...bypass?.resistance,redirect:[...bypass?.resistance?.redirect??[],...resistanceRedirects]},
   }});
  }
  Object.defineProperty(attributes,'resistances',{...descriptor,value:view});
  try{return next({...params,rollOptions:options});}
  finally{
   Object.defineProperty(attributes,'resistances',descriptor);
   if(bypassDescriptor)Object.defineProperty(rollOptions,'bypass',bypassDescriptor);else delete rollOptions.bypass;
  }
 }
 return {applyDamage};
}
