const __nativeManualPoolStaticReceiver=(()=>{
 const registered=new WeakMap();
 const identities={before:FlatModifierRuleElement.prototype.beforePrepareData,resolveValue:Y.prototype.resolveValue,resolveInjected:Y.prototype.resolveInjectedProperties,predicateTest:Hn.prototype.test,predicateValid:Hn.isValid,predicateArray:Hn.isArray,abp:AutomaticBonusProgression$1.suppressRuleElement,abpEnabled:AutomaticBonusProgression$1.isEnabled};
 const fail=()=>{throw Error('native-manual-batch-static-receiver-unavailable')};
 const copy=value=>JSON.parse(JSON.stringify(value));
 const freeze=value=>{if(value&&typeof value==='object'){for(const child of Object.values(value))freeze(child);Object.freeze(value)}return value};
 const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 const godlessPredicate=[{or:['action:battle-medicine','action:treat-wounds']}];
 const fields=['value','min','max','type','force','critical','battleForm','damageType','damageCategory','removeAfterRoll','ignored','invalid','fromEquipment','selector','predicate'];
 const rawKeys=new Set(['key','value','min','max','type','force','critical','battleForm','damageType','damageCategory','removeAfterRoll','ignored','fromEquipment','selector','predicate','label','slug','priority','requiresEquipped','requiresInvestment','spinoff','tags','hideIfDisabled']);
 function data(value){
  if(value===undefined||value===null||typeof value==='boolean'||typeof value==='number'&&Number.isFinite(value))return;
  if(typeof value==='string'){if(/[{@}]/.test(value))fail();return}
  if(!value||typeof value!=='object'||!Array.isArray(value)&&value.constructor?.name!=='Object')fail();
  for(const key of Reflect.ownKeys(value)){if(key==='length'&&Array.isArray(value))continue;const d=Object.getOwnPropertyDescriptor(value,key);if(typeof key!=='string'||!d||!Object.hasOwn(d,'value'))fail();data(d.value)}
 }
 function array(actor,key){
  const map=actor?.synthetics?.[key];if(!map||typeof map!=='object')fail();const d=Object.getOwnPropertyDescriptor(map,'healing-received');
  if('healing-received'in map&&!d||d&&(!Object.hasOwn(d,'value')||!Array.isArray(d.value)))fail();return d?.value??null;
 }
 function currentMethods(rule){return rule.constructor===FlatModifierRuleElement&&rule.beforePrepareData===identities.before&&rule.resolveValue===identities.resolveValue&&rule.resolveInjectedProperties===identities.resolveInjected&&Hn.prototype.test===identities.predicateTest&&Hn.isValid===identities.predicateValid&&Hn.isArray===identities.predicateArray&&AutomaticBonusProgression$1.suppressRuleElement===identities.abp&&AutomaticBonusProgression$1.isEnabled===identities.abpEnabled&&game.pf2e.variantRules.AutomaticBonusProgression===AutomaticBonusProgression$1}
 function snapshot(rule){
  const raw=rule.item?._source?.system?.rules?.[rule.sourceIndex];
  if(!raw||raw.key!=='FlatModifier'||Object.keys(raw).some(key=>!rawKeys.has(key)))fail();data(raw);
  // Only these exact predicates are unaffected by source and receipt markers
  // added after batch selection. Other native predicates remain unsupported.
  if(!same(raw.predicate===undefined?[]:raw.predicate,[])&&!same(raw.predicate,godlessPredicate))fail();
  const prepared=Object.fromEntries(fields.filter(key=>rule[key]!==undefined).map(key=>[key,key==='predicate'?[...rule.predicate]:rule[key]]));data(prepared);
  if(!currentMethods(rule)||rule.ignored||rule.invalid||!Number.isFinite(raw.value)||rule.value!==raw.value||!same(rule.selector,['healing-received'])||!(['healing-received',JSON.stringify(['healing-received'])].includes(typeof raw.selector==='string'?raw.selector:JSON.stringify(raw.selector))))fail();
  if(!['untyped','status','circumstance','potency','proficiency'].includes(rule.type)||rule.type!==(raw.type??'untyped')||rule.battleForm||rule.damageType!=null||rule.damageCategory!=null||rule.removeAfterRoll||rule.critical!==null||raw.spinoff||!same([...rule.predicate],raw.predicate??[]))fail();
  for(const key of ['min','max'])if(rule[key]!==raw[key]||rule[key]!==undefined&&!Number.isFinite(rule[key]))fail();
  if(rule.min!==undefined&&rule.max!==undefined&&rule.min>rule.max||rule.force&&rule.type==='untyped'||rule.force!==(raw.force??false))fail();
  const predicate=new Hn(copy([...rule.predicate]));if(!predicate.isValid)fail();
  return {raw:copy(raw),prepared:copy(prepared),itemSource:JSON.stringify(rule.item._source)};
 }
 function register(callback,rule,selector,list){
  // Registration observes the original push. Invalid or dynamic rules remain
  // ordinary native rules; only a participating batch asks to qualify them.
  if(selector!=='healing-received')return;
  try{registered.set(callback,{rule,actor:rule.actor,item:rule.item,sourceIndex:rule.sourceIndex,list,index:list.length-1,initial:snapshot(rule)})}catch{}
 }
 function model(actor,params){
  if(typeof params.damage!=='number'||!Number.isFinite(params.damage)||Math.trunc(params.damage)>=0||params.final===true||!params.rollOptions||typeof params.rollOptions[Symbol.iterator]!=='function')fail();
  const options=[...params.rollOptions];if(options.some(option=>typeof option!=='string'))fail();
  const callbacks=array(actor,'modifiers'),dice=array(actor,'damageDice'),adjustments=array(actor,'modifierAdjustments');if(dice?.length||adjustments?.length)fail();
  const ordered=[...callbacks??[]],records=[],entries=[];
  for(let index=0;index<ordered.length;index++){
   const callback=ordered[index],r=registered.get(callback);if(!r||r.actor!==actor||r.list!==callbacks||r.index!==index||r.item.actor!==actor||actor.items?.get(r.item.id)!==r.item||!actor.rules?.includes(r.rule)||r.rule.item!==r.item||r.rule.sourceIndex!==r.sourceIndex||!same(snapshot(r.rule),r.initial))fail();
   const s=r.initial.prepared,value=Math.clamp(s.value,s.min??s.value,s.max??s.value),predicate=new Hn(copy(s.predicate)),critical=params.outcome==='criticalSuccess';
   const enabled=(s.critical===null||s.critical===critical)&&predicate.test(new Set(options));
   entries.push(freeze({itemUUID:r.item.uuid,sourceIndex:r.sourceIndex,registrationOrdinal:index,type:s.type,value,critical:s.critical,predicate:copy(s.predicate),enabled}));
   if(enabled)records.push({modifier:value,type:s.type,force:s.force??false,ignored:false,enabled:true});
  }
  const flatTotal=__nativeReceiverStacking(records),amount=-Math.min(0,Math.trunc(params.damage)-flatTotal),damage=params.damage,outcome=params.outcome,optionSet=params.rollOptions;
  if(!Number.isFinite(flatTotal)||!Number.isFinite(amount))fail();
  const isCurrent=()=>{try{
   if(params.damage!==damage||params.outcome!==outcome||params.final===true||params.rollOptions!==optionSet||!same([...optionSet],options)||array(actor,'modifiers')!==callbacks||array(actor,'damageDice')!==dice||array(actor,'modifierAdjustments')!==adjustments||dice?.length||adjustments?.length||ordered.length!==(callbacks?.length??0)||ordered.some((callback,index)=>callbacks[index]!==callback))return false;
   for(const callback of ordered){const r=registered.get(callback);if(!r||r.actor!==actor||r.rule.item!==r.item||r.item.actor!==actor||actor.items.get(r.item.id)!==r.item||!actor.rules.includes(r.rule)||r.rule.sourceIndex!==r.sourceIndex||!same(snapshot(r.rule),r.initial))return false}return true;
  }catch{return false}};
  return Object.freeze({qualified:true,amount,flatTotal,entries:Object.freeze(entries),isCurrent});
 }
 return {register,model};
})();
