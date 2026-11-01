import {preparedTreatmentSelections} from './prepared-treatment.mjs';
import {canReceiveVitalityHealing,isSuppressedItem,refocusUnsupported,wardCapacity} from './capabilities.mjs';
import {healingVariant,LAY_ON_HANDS} from './refocus.mjs';
import {salubriousFeat,sourceId,treatmentTiers,values} from '../salubrious-kiss-rules.mjs';
import {canonicalJSON} from './revision-codec.mjs';

const layOnHandsEffect='Compendium.pf2e.spell-effects.Item.lyLMiauxIVUM3oF1';
const unavailable=reason=>({ready:false,reason});
const able=actor=>actor.canAct===true&&!actor.isDead&&!actor.hasCondition?.('unconscious');
const noIWR=actor=>['immunities','resistances','weaknesses'].every(key=>Array.isArray(actor.attributes?.[key])&&actor.attributes[key].length===0);
const entries=value=>Array.isArray(value)?value.length:value&&typeof value==='object'?Object.keys(value).length:0;
const robustHealthSource='Compendium.pf2e.feats-srd.Item.yTLGclKtWVFZLKIz';
const robustHealthRules=[{key:'FlatModifier',predicate:[{or:['action:battle-medicine','action:treat-wounds']}],selector:'healing-received',type:'circumstance',value:'@actor.level'}];

function robustHealthBonus(actor){
 const items=values(actor.items).filter(item=>!isSuppressedItem(item)&&sourceId(item)===robustHealthSource);
 if(items.length!==1||!Number.isSafeInteger(actor.level)||actor.level<0)return null;
 const item=items[0],rules=values(actor.rules).filter(rule=>!rule.ignored&&rule.key==='FlatModifier'&&rule.item===item);
 if(rules.length!==1||rules[0].type!=='circumstance'||rules[0].critical!=null||rules[0].removeAfterRoll===true)return null;
 const selectors=Array.isArray(rules[0].selector)?rules[0].selector:[rules[0].selector];
 if(selectors.length!==1||selectors[0]!=='healing-received')return null;
 try{
  const raw=typeof item.toObject==='function'?item.toObject(true)?.system?.rules:item._source?.system?.rules??item.system?.rules;
  if(canonicalJSON(raw)!==canonicalJSON(robustHealthRules))return null;
 }catch{return null}
 const modifiers=actor.synthetics?.modifiers?.['healing-received'],legacy=actor.synthetics?.statisticsModifiers?.['healing-received'];
 if(!Array.isArray(modifiers)||modifiers.length!==1||entries(legacy)&&legacy!==modifiers)return null;
 return actor.level;
}

function patientModel(actor,master){
 const synth=actor.synthetics??{},domains=['healing-received'];
 const robustHealth=robustHealthBonus(actor),modifiers=(robustHealth===null&&['modifiers','statisticsModifiers'].some(key=>entries(synth[key]?.['healing-received'])))||['damageDice','modifierAdjustments','damageAlterations'].some(key=>domains.some(domain=>entries(synth[key]?.[domain])))||entries(synth.ephemeralEffects?.['damage-received']?.target);
 const healingReason=modifiers?'healing-received-modifier-unverified':null;
 const hp=master.attributes?.hp??master.system?.attributes?.hp;
 const damageReady=noIWR(actor)&&noIWR(master)&&actor.hardness===0&&master.hardness===0&&hp?.temp===0&&(hp.sp===undefined||hp.sp?.max===0);
 return {healingReady:healingReason===null,healingReason,damageReady,damageReason:damageReady?null:'patient-damage-model-unverified',vitalityReady:canReceiveVitalityHealing(actor),vitalityReason:canReceiveVitalityHealing(actor)?null:'patient-vitality-unavailable',...robustHealth!==null?{robustHealth}:{}};
}
function treatmentModels({game,actor,skill,slugs,context,capacity,threePecks=false}){
 const selections=preparedTreatmentSelections({game,actor,skill,slugs,context}),models=[],reasons=[];
 for(const selection of selections){
  if(threePecks&&selection.riskySurgery)continue;
  if(!selection.ready){if(!reasons.includes(selection.reason))reasons.push(selection.reason);continue}
  for(const [index,row] of selection.outcomesByRank.entries()){
   const medicBonus=!threePecks&&slugs.includes('medic-dedication')?[0,5,10,15][index]:0;
   models.push({ready:true,skill,rank:row.rank,dc:row.dc,modifier:selection.modifier,assurance:selection.assurance,riskySurgery:selection.riskySurgery,outcomes:row.cases.map(c=>c.outcome),bonus:[0,10,30,50][index]+medicBonus,medicBonus,durationSeconds:600,cooldownSeconds:!threePecks&&slugs.includes('continual-recovery')?600:3600,capacity:threePecks?1:capacity,sourceVersion:selection.sourceVersion,source:selection.source});
  }
 }
 return {models,reasons};
}
function healingFormula(roll){
 const instances=roll?.instances;
 if(!Array.isArray(instances)||instances.length!==1)return null;
 const instance=instances[0];
 if(instance.type!=='vitality'||instance.kinds?.size!==1||!instance.kinds.has('healing')||instance.persistent!==false||instance.materials?.size!==0)return null;
 // DamageRoll.formula includes localized damage labels. The instance head
 // retains the arithmetic expression used by the prepared, unevaluated roll.
 const raw=instance.head?.expression;
 if(typeof raw!=='string')return null;
 let expression=raw.replace(/\s/g,'');
 while(expression.startsWith('(')&&expression.endsWith(')')){
  let depth=0,enclosed=true;
  for(let i=0;i<expression.length;i++){if(expression[i]==='(')depth++;if(expression[i]===')')depth--;if(depth===0&&i<expression.length-1){enclosed=false;break}}
  if(!enclosed||depth!==0)break;
  expression=expression.slice(1,-1);
 }
 if(!/^(?:\d+d\d+|\d+)(?:\+(?:\d+d\d+|\d+))*$/.test(expression))return null;
 const result={constant:0,dice:[]};
 for(const term of expression.split('+')){
  if(term.includes('d')){const [number,faces]=term.split('d').map(Number);if(!Number.isSafeInteger(number)||number<1||number>100||!Number.isSafeInteger(faces)||faces<2||faces>100)return null;result.dice.push({number,faces})}
  else{const amount=Number(term);if(!Number.isSafeInteger(amount)||amount<0)return null;result.constant+=amount}
 }
 return Number.isSafeInteger(result.constant)&&result.constant+result.dice.reduce((sum,die)=>sum+die.number,0)>0?result:null;
}
async function focusModel(actor,items){
 const spells=items.filter(item=>sourceId(item)===LAY_ON_HANDS);
 if(spells.length!==1)return unavailable('focus-healing-unavailable');
 if(!able(actor))return unavailable('actor-cannot-cast');
 const variant=healingVariant(spells[0]);
 if(!variant||variant.system?.cast?.focusPoints!==1||typeof variant.getDamage!=='function')return unavailable('focus-healing-variant-unverified');
 try{
  const prepared=await variant.getDamage({skipDialog:true}),healing=healingFormula(prepared?.template?.damage?.roll);
  if(!healing)return unavailable('focus-healing-formula-unverified');
  return {ready:true,itemUUID:spells[0].uuid,focusCost:1,durationSeconds:6,healing,effects:[{sourceUUID:layOnHandsEffect,durationSeconds:6,otherOnly:true}]};
 }catch(error){return {...unavailable('focus-healing-preparation-failed'),detail:String(error.message??error)}}
}

/** Attach execution models; estimate-only fallbacks are never promoted. */
export async function enhanceComputedActors({actors,game,fromUuid}){
 const documents=new Map();
 async function read(uuid){if(!documents.has(uuid))documents.set(uuid,await fromUuid(uuid));const actor=documents.get(uuid);if(!actor)throw Error('actor-unavailable');return actor}
 const result=[];
 for(const snapshot of actors){
  const actor=await read(snapshot.actorUUID),master=await read(snapshot.pool.poolUUID),items=values(actor.items).filter(item=>!isSuppressedItem(item)),slugs=items.filter(item=>item.type==='feat').map(item=>item.slug??item.system?.slug);
  const capacity=wardCapacity({wardMedic:slugs.includes('ward-medic'),medicineRank:actor.getStatistic?.('medicine')?.rank??0}),treatments=[],treatmentReasons=[];
  for(const skill of ['medicine',...slugs.includes('natural-medicine')?['nature']:[]]){
   const prepared=treatmentModels({game,actor,skill,slugs,capacity});treatments.push(...prepared.models);treatmentReasons.push(...prepared.reasons);
  }
  const unsupported=refocusUnsupported(items),refocus=!able(actor)?unavailable('actor-cannot-refocus'):unsupported.length?unavailable('refocus-recovery-unadapted'):{ready:true,recovery:1,durationSeconds:600};
  let threePecks=unavailable('three-pecks-unavailable');const feat=salubriousFeat(actor);
  if(feat){
   try{
    const tiers=treatmentTiers(actor),context={item:feat,rollDomains:['all','skill-check','occultism'],traits:['exploration','healing','manipulate','vitality'],extraRollOptions:feat.getRollOptions?.('item')??[]};
    const prepared=treatmentModels({game,actor,skill:'occultism',slugs,context,capacity:1,threePecks:true});
    const models=prepared.models.filter(model=>tiers.some(tier=>tier.dc===model.dc));
    threePecks=refocus.ready&&models.length?{ready:true,treatments:models}:unavailable(refocus.reason??prepared.reasons[0]??'three-pecks-treatment-unverified');
   }catch(error){threePecks={...unavailable('three-pecks-variant-unadapted'),detail:String(error.message??error)}}
  }
  result.push({...snapshot,type:actor.type,canAct:able(actor),computed:{version:1,treatments,treatmentReasons:[...new Set(treatmentReasons)],patient:patientModel(actor,master),focusHealing:await focusModel(actor,items),refocus,threePecks}});
 }
 return result;
}
