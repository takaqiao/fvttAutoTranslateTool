const MODULE_ID='pf2e-third-party-automation';
const MARKER='siphonDamage';
const clone=value=>structuredClone(value);
// Stable comparison keeps distinct native instance metadata/materials partitioned.
const comparable=value=>value===null?['null']:Array.isArray(value)?['array',value.map(comparable)]:typeof value==='object'?['object',Object.keys(value).sort().map(key=>[key,comparable(value[key])])]:[typeof value,String(value)];
const stable=value=>JSON.stringify(comparable(value));

function convertedInstance(instance){
 const data=clone(instance.toJSON());
 const flavors=String(data.options?.flavor??'').split(',').filter(v=>v&&![instance.type,'persistent','bleed','healing','damage','untyped'].includes(v));
 data.options={...data.options,flavor:['untyped','damage',...flavors].join(',')};
 delete data.options.evaluatePersistent;
 const result=instance.constructor.fromData(data);
 result.critRule=instance.critRule;
 return result;
}

function mergeCompatible(instances){
 const groups=[];
 for(const instance of instances){
  // Never let merging turn a clamped negative component into a penalty against
  // another component, or change native rounding of pre-existing fractions.
  const raw=instance.head?.total;
  const key=Number.isInteger(raw)&&raw>=0?stable({options:instance.options,critRule:instance.critRule,materials:[...instance.materials].sort()}):null;
  const group=key===null?null:groups.find(g=>g.key===key);
  if(group)group.instances.push(instance);else groups.push({key,instances:[instance]});
 }
 return groups.map(({instances:group})=>{
  if(group.length===1)return group[0];
  const first=group[0],data=clone(first.toJSON());
  // Preserve term flavors and critical subtrees even when every operand is
  // deterministic. PF2e otherwise simplifies some sums to an unmarked number.
  const heads=group.map(instance=>({class:'Grouping',term:clone(instance.head.toJSON()),options:{flavor:'damage'},evaluated:true}));
  data.terms=[heads.reduce((left,right)=>({class:'ArithmeticExpression',operator:'+',operands:[left,right],options:{flavor:'damage'},evaluated:true}))];
  data.total=group.reduce((total,instance)=>total+instance.total,0);
  const result=first.constructor.fromData(data);
  result.critRule=first.critRule;
  result.resetFormula();
  return result;
 });
}

/** Convert a full evaluated native PF2e DamageRoll in place, without evaluating
 * any dice. Invoke at its toMessage boundary, after native modifiers/evaluation
 * and before publication. Both toMessage and pf2e.damageRoll then see this same
 * object. Do not call for snapshots whose siphon.applies is false.
 *
 * Persistent instances and healing-only instances are excluded. Remaining
 * instances become actual untyped damage, preserving dice, fixed modifiers,
 * critical/precision/splash terms, materials and native roll options. Compatible
 * instances are consolidated before native outcome scaling so e.g. (7+3)/2 is 5.
 * Distinct material/metadata partitions retain native per-instance rounding;
 * they must not be merged by granting one component another's material or bypass.
 *
 * This deliberately keeps the full shared base total. Use siphonMultiplier per
 * target on the outcome-scaled roll before IWR; do not globally halve this roll.
 * Constructor discovery is public CONFIG.Dice.rolls, or pass {DamageRoll} for a
 * verified native adapter. No actor/item or previous instance object is mutated.
 */
export function convertSiphonRoll(roll,{DamageRoll=globalThis.CONFIG?.Dice?.rolls?.find(C=>C.name==='DamageRoll')}={}){
 if(typeof DamageRoll!=='function'||!(roll instanceof DamageRoll)||roll._evaluated!==true||!roll.pool||!Array.isArray(roll.instances)||!roll.instances.length)throw Error('Siphoning requires an evaluated native DamageRoll with instances.');
 if(roll.options?.[MODULE_ID]?.[MARKER]?.version===1)return roll;
 const original=roll.instances;
 const converted=mergeCompatible(original.filter(i=>!i.persistent&&i.kinds.has('damage')).map(convertedInstance));
 if(converted.length===0){
  converted.push(original[0].constructor.fromData({class:'DamageInstance',formula:'0',options:{flavor:'untyped,damage'},terms:[{class:'NumericTerm',number:0,evaluated:true}],total:0,evaluated:true}));
 }
 const options=clone(roll.options);
 options[MODULE_ID]={...options[MODULE_ID],[MARKER]:{version:1}};
 const pool=roll.pool.constructor.fromRolls(converted);
 const replacement=DamageRoll.fromTerms([pool],options);
 if(!replacement._evaluated||replacement.instances.some(i=>i.type!=='untyped'||i.persistent||i.kinds.has('healing')))throw Error('Native Siphoning conversion produced an invalid damage roll.');
 // Foundry 14 Roll holds evaluated state in these public/protected properties;
 // PF2e's pool/instances/dice/formula getters derive from terms. Rebuild first so
 // constructor failures cannot leave a partially transformed original object.
 for(const key of ['terms','_formula','_total','_dice','_evaluated','options'])roll[key]=replacement[key];
 return roll;
}
