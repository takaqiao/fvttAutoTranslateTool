import {MODULE_ID} from './rules.mjs';

export const SPIRITUAL_SCAR_SOURCE='Compendium.pf2e.actionspf2e.Item.f8vqWQktvmtSRpUb';
const compiled=new WeakMap();
export const spiritualScarMarker=nonce=>`${MODULE_ID}:spiritual-scar:${nonce}`;
const exactList=(value,expected)=>Array.isArray(value)&&value.length===expected.length&&value.every((entry,index)=>entry===expected[index]);
function originalRule(ability){
 const source=ability?.sourceId??ability?._stats?.compendiumSource??ability?.flags?.core?.sourceId;
 const rules=ability?._source?.system?.rules,matches=rules?.filter(rule=>rule.key==='Resistance');
 if(source!==SPIRITUAL_SCAR_SOURCE||ability?.type!=='action'||matches?.length!==1)return null;
 const rule=matches[0];
 return Object.keys(rule).sort().join(',')==='definition,key,label,predicate,type,value'&&rule.type==='custom'
  &&rule.label==='PF2E.IWR.Custom.SpiritFromFiends'&&rule.value==='@actor.abilities.cha.mod + 2*@actor.level'
  &&exactList(rule.definition,['origin:trait:fiend','damage:type:spirit'])&&exactList(rule.predicate,['spiritual-scar'])
  ?{rule,index:rules.indexOf(rule)}:null;
}

/** Compile the original rule through PF2e without merging into an unrelated
 * resistance or changing the live manual toggle. This does not authorize Use. */
export function compileSpiritualScarResistance({actor,ability,nonce,options=[],ResistanceRuleElement=globalThis.game?.pf2e?.RuleElements?.all?.Resistance}){
 const source=originalRule(ability);
 if(!source||ability.actor!==actor||actor.items?.get(ability.id)!==ability||typeof actor.getContextualClone!=='function'
  ||typeof ResistanceRuleElement!=='function'||typeof nonce!=='string'||!/^[A-Za-z0-9_-]{1,100}$/.test(nonce))throw Error('精神伤痕原卡、原生抗力规则或调用范围无法验证。');
 const marker=spiritualScarMarker(nonce),clone=actor.getContextualClone([...new Set([...options,marker])]);
 const item=clone?.items?.get(ability.id),attributes=clone?.system?.attributes,all=clone?.rollOptions?.all;
 if(clone===actor||!item||item===ability||item.actor!==clone||!attributes||attributes===actor.system?.attributes
  ||!Array.isArray(attributes.resistances)||attributes.resistances===actor.attributes?.resistances||!all||all===actor.rollOptions?.all)throw Error('精神伤痕需要完全独立的原生上下文克隆。');
 const previous=attributes.resistances,descriptor=Object.getOwnPropertyDescriptor(all,'spiritual-scar');
 let resistance;
 try{
  all['spiritual-scar']=true;
  // PF2e combines equal custom definitions in place. Prepare the exact rule
  // alone so a higher pre-existing resistance cannot be attributed to Scar.
  attributes.resistances=[];
  const rule=new ResistanceRuleElement(structuredClone(source.rule),{parent:item,sourceIndex:source.index});
  if(rule.test()!==true||rule.ignored||rule.invalid)throw Error('精神伤痕原生规则未能通过准备。');
  rule.afterPrepareData();
  resistance=attributes.resistances[0];
  if(attributes.resistances.length!==1||!resistance||previous.includes(resistance)||actor.attributes?.resistances?.includes(resistance)
   ||compiled.has(resistance)||resistance.type!=='custom'||!Number.isInteger(resistance.value)||resistance.value<=0
   ||typeof resistance.test!=='function'||typeof resistance.getDoubledValue!=='function'
   ||!exactList(resistance.definition,['origin:trait:fiend','damage:type:spirit']))throw Error('精神伤痕未生成独立且有效的原生抗力。');
 }finally{
  attributes.resistances=previous;
  if(descriptor)Object.defineProperty(all,'spiritual-scar',descriptor);else delete all['spiritual-scar'];
 }
 const nativeTest=resistance.test.bind(resistance);
 resistance.test=options=>{const set=new Set(options??[]);return set.has(marker)&&nativeTest(set);};
 compiled.set(resistance,{actor,state:'ready'});
 return resistance;
}

/** One compiled instance, one native call. Cleanup never removes another
 * resistance, including one with identical type, value or displayed label. */
export async function withSpiritualScarResistance(actor,resistance,native){
 const proof=compiled.get(resistance),original=actor?.attributes?.resistances;
 if(proof?.actor!==actor||proof.state!=='ready'||!Array.isArray(original)||original.includes(resistance)||typeof native!=='function')throw Error('精神伤痕的单次原生抗力范围无效或已使用。');
 proof.state='active';
 try{original.push(resistance);return await native();}
 finally{
  proof.state='used';
  for(const list of new Set([original,actor.attributes?.resistances])){
   if(!Array.isArray(list))continue;
   for(let index=list.length-1;index>=0;index--)if(list[index]===resistance)list.splice(index,1);
  }
 }
}
