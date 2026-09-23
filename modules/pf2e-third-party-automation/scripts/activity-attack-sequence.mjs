import {MODULE_ID} from './rules.mjs';
import {getSourceId} from './native-context.mjs';

const DAMAGE_OPTION=`${MODULE_ID}:activity-damage`;
const outcomes=new Set(['criticalFailure','failure','success','criticalSuccess']);
const miss=outcome=>outcome==='failure'||outcome==='criticalFailure';
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const usageKey=strike=>`${strike.item.id}:${strike.item.altUsageType??''}`;

function twinPair(first,second){
 if(first.id===second.id||!first.traits.has('twin')||!second.traits.has('twin'))return false;
 if(first.base&&second.base)return first.base===second.base;
 return !!first.source&&first.source===second.source;
}

/** Facts belong only to this invocation, not to MAP or an inferred turn history.
 * Each frame freezes its predecessors before rolling, so deferred damage cannot
 * accidentally use later attacks. No documents, listeners or timers are created. */
export function createAttackSequence({actor}){
 const history=[],frames=new WeakMap();
 function begin(strike,target){
  const item=strike.item,valid=item.type==='weapon'&&!!item.id&&item.actor?.uuid===actor.uuid;
  const fact={id:item.id,usage:usageKey(strike),traits:new Set(item.system.traits?.value??[]),base:item.system.baseItem,source:getSourceId(item),target:target?.uuid};
  const prior=valid?history.filter(entry=>entry.fact.id===fact.id):[],attackOptions=new Set(),bonuses=[];
  if(fact.traits.has('backswing')&&miss(prior.at(-1)?.outcome))attackOptions.add('backswing-bonus');
  if(fact.traits.has('sweep')&&fact.target&&prior.some(entry=>entry.fact.target&&entry.fact.target!==fact.target))attackOptions.add('sweep-bonus');
  if(valid&&history.some(entry=>twinPair(entry.fact,fact)))bonuses.push({trait:'twin',slug:'activity-twin',label:'PF2E.Item.Weapon.Twin.SecondPlus',value:'@weapon.system.damage.dice'});
  if(fact.traits.has('forceful')&&prior.length)bonuses.push({trait:'forceful',slug:`activity-forceful-${prior.length===1?'second':'third'}`,label:`PF2E.Item.Weapon.Forceful.${prior.length===1?'Second':'Third'}`,value:prior.length===1?'@weapon.system.damage.dice':'2 * @weapon.system.damage.dice'});
  const state={fact,valid,recorded:false};
  const frame={attackOptions,damage(current){
   const unchanged={strike:current,options:new Set()};
   if(!state.recorded||!bonuses.length||current.item.actor?.uuid!==actor.uuid||usageKey(current)!==fact.usage)return unchanged;
   // Use the Strike's actor so a Spellstrike infusion and alternate usage survive.
   // Distinct slugs avoid native manual (ignored) modifiers shadowing these rules;
   // native circumstance stacking also preserves any larger manual bonus.
   const owner=current.item.actor,copy=owner.clone({items:[...structuredClone(owner._source.items),{
    _id:globalThis.foundry?.utils?.randomID?.()??'ActivityDamage01',
    name:'本次组合活动：武器伤害',type:'effect',system:{duration:{value:-1,unit:'unlimited'},rules:bonuses.map(bonus=>({
     key:'FlatModifier',selector:`${fact.id}-damage`,slug:bonus.slug,label:bonus.label,
     type:'circumstance',value:bonus.value,predicate:[DAMAGE_OPTION,`item:id:${fact.id}`,`item:trait:${bonus.trait}`],
    }))},
   }]},{keepId:true});
   const prepared=values(copy.system.actions).flatMap(action=>[action,...action.altUsages??[]]).find(action=>action.item&&usageKey(action)===fact.usage);
   return prepared?{strike:prepared,options:new Set([DAMAGE_OPTION])}:unchanged;
  }};
  frames.set(frame,state);return frame;
 }
 function record(frame,outcome){
  const state=frames.get(frame);if(!state?.valid||state.recorded||!outcomes.has(outcome))return false;
  state.recorded=true;history.push({fact:state.fact,outcome});return true;
 }
 return {begin,record};
}
