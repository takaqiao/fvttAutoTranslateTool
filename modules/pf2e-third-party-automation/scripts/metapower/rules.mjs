/** Reviewed source identities. Names, slugs and partial IDs are never identities. */
const PACK='Compendium.battlezoo-eldamon-pf2e.';
export const METAPOWER_SOURCES=Object.freeze({
 siphoning:PACK+'actions.Item.4w72ljp4eLBqeZB2',
 widen:PACK+'feats.Item.3YasBiZw3N96rdUW',
 disruptiveSiphon:PACK+'feats.Item.kG0HSsDc6eHYjTU9'
});
const freeze=value=>{if(value&&typeof value==='object'){for(const child of Object.values(value))freeze(child);Object.freeze(value)}return value};
const entries=[
 ['veFrnrxYjlqca13w',{id:'electric-surge',areaType:'line',baseDistance:20,levelArea:true,outcomeMode:'basic-save',save:'reflex',effects:['charged'],dischargeNonDamage:true}],
 ['hQOa1yaP9C6wajNn',{id:'anvil-crawler-lightning',areaType:'cone',baseDistance:30,outcomeMode:'basic-save',save:'fortitude',effects:['charged','shocked'],dischargeNonDamage:true}],
 ['KWQgx7RMeY3RKW6J',{id:'static-shock',outcomeMode:'attack-with-fixed-failure',effects:['charged','shocked']}],
 ['QIYppaP0zcGvb5Bd',{id:'electric-shot',range:40,outcomeMode:'attack-with-target-dependent-failure',effects:['charged'],dischargeNonDamage:true}],
 ['fzV5Ly3a9nEsfcAJ',{id:'reactive-chain',reaction:true,outcomeMode:'basic-save',save:'reflex',damageBasis:'trigger-damage-halved',effects:['charged']}],
 ['geZCat82IOuShmmk',{id:'retributive-shock',reaction:true,outcomeMode:'special-save',save:'reflex',effects:['shocked'],dischargeNonDamage:true}],
 ['9bElF2uVf5FCJtb9',{id:'high-voltage',hasDuration:true,dependentEffect:true,outcomeMode:'delayed-basic-save',save:'reflex',effects:['refresh']}]
];
/** These seven profiles are the reviewed electricity scope, not all Eldamon powers.
 * OutcomeMode describes native delivery; it must never replace native prerequisites,
 * prepared-state checks, target selection, attack/save arithmetic or discharge payment. */
export const POWER_PROFILES=freeze(Object.fromEntries(entries.map(([id,profile])=>{
 const source=PACK+'powers.Item.'+id;
 return [source,{sourceUuid:source,hasDuration:false,damageBasis:'native',...profile}];
})));

/** PF2e document identity first, with source-data and legacy fallbacks. */
export function sourceUuid(item){return item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId??null}
export function metapowerKind(item){const source=sourceUuid(item);return ['siphoning','widen'].find(kind=>METAPOWER_SOURCES[kind]===source)??null}
export function powerProfile(item){const source=sourceUuid(item);return Object.hasOwn(POWER_PROFILES,source)?POWER_PROFILES[source]:null}

/** The package does not establish a Widen cost. Its actual-use entry must supply
 * the table's confirmed cost explicitly; reading a passive feat is not activation. */
export function metapowerActionCost(kind,policy={}){
 if(kind==='siphoning')return 1;
 if(kind!=='widen')throw Error('Unsupported metapower kind.');
 if(![1,2,3,'free'].includes(policy.widenActionCost))throw Error('Widen action cost requires an explicit policy.');
 return policy.widenActionCost;
}

/** Pure base -> final geometry, never feed the previously widened distance back. */
export function widenDistance({type,distance,hasDuration=false}){
 if(!Number.isFinite(distance)||distance<0)throw Error('Area distance must be finite and non-negative.');
 if(hasDuration||distance===0)return distance;
 if(type==='burst')return distance>=10?distance+5:distance;
 if(type==='cone'||type==='line')return distance+(distance<=15?5:10);
 return distance;
}

function selectedArea(profile,{level,discharge,baseDistance}){
 if(!profile.areaType)return null;
 const min=profile.baseDistance*(profile.levelArea&&discharge?2:1);
 const max=profile.levelArea?(20+10*Math.min(4,Math.max(0,Math.floor((level-1)/4))))*(discharge?2:1):discharge?60:30;
 const step=profile.levelArea?(discharge?20:10):5;
 const distance=baseDistance??min;
 if(!Number.isFinite(distance)||distance<min||distance>max||(distance-min)%step!==0)throw Error('Selected base distance is not legal for this power, level and branch.');
 return {type:profile.areaType,baseDistance:distance,distance,hasDuration:profile.hasDuration};
}
const strings=value=>Array.isArray(value)||value instanceof Set?[...value].filter(v=>typeof v==='string'):[];
const items=actor=>Array.isArray(actor?.items)?actor.items:actor?.items?.contents??(actor?.items?.values?[...actor.items.values()]:[]);

/** Immutable per-channel domain snapshot. Lifecycle/ownership/nonce validation is
 * the caller's responsibility. `selection.baseDistance` means the chosen legal
 * native branch before Widen; selection.discharge never waives its normal cost.
 * Unresolved policies are explicit: dischargeNonDamage='retain'|'remove',
 * highVoltage='unaffected'|'convert'. A convert decision still requires the caller
 * to bind High Voltage's delayed trigger to this channel, never immediate damage. */
export function buildChannelSnapshot({kind,item,actor=item?.actor,level=actor?.level??1,selection={},policy={}}){
 if(!['siphoning','widen'].includes(kind))throw Error('Unsupported metapower kind.');
 const profile=powerProfile(item);
 if(!profile)throw Error('Unsupported power source: no reviewed profile.');
 if(!Number.isInteger(level)||level<1)throw Error('Power level must be a positive integer.');
 const discharge=selection.discharge===true;
 if(kind==='siphoning'&&profile.dependentEffect&&!['unaffected','convert'].includes(policy.highVoltage))throw Error('High Voltage requires an explicit dependent-effect policy.');
 if(kind==='siphoning'&&discharge&&profile.dischargeNonDamage&&!['retain','remove'].includes(policy.dischargeNonDamage))throw Error('Siphoning discharge non-damage benefits require an explicit policy.');
 const applies=kind==='siphoning'&&(!profile.dependentEffect||policy.highVoltage==='convert');
 const removeBenefits=applies&&discharge&&policy.dischargeNonDamage==='remove';
 const area=selectedArea(profile,{level,discharge,baseDistance:selection.baseDistance});
 if(area){
  if(removeBenefits)area.distance=profile.levelArea?area.baseDistance/2:profile.baseDistance;
  else if(kind==='widen')area.distance=widenDistance(area);
 }
 const element=actor?.flags?.pf2e?.eldamon?.element??{};
 return freeze({
  version:1,kind,powerId:profile.id,powerSourceUuid:profile.sourceUuid,actorUuid:actor?.uuid??null,itemUuid:item?.uuid??null,level,
  traits:[...new Set(strings(item?.traits??item?.system?.traits?.value))],
  associatedTraits:[...new Set([element.trait,element.traitTwo].filter(t=>typeof t==='string'&&t.length))],
  disruptive:items(actor).some(i=>sourceUuid(i)===METAPOWER_SOURCES.disruptiveSiphon),
  siphon:{applies,reason:kind!=='siphoning'?'different-metapower':applies?'direct-damage':'dependent-effect'},
  area,range:profile.range?(discharge&&!removeBenefits?2:1)*profile.range:null,
  discharge,dischargeCost:discharge?1:0,
  saveDowngrade:profile.id==='retributive-shock'&&discharge&&!removeBenefits?1:0,
  outcomeMode:profile.outcomeMode,damageBasis:profile.damageBasis,
  suppressEffects:applies?[...profile.effects]:[],
  policy:{...(policy.dischargeNonDamage?{dischargeNonDamage:policy.dischargeNonDamage}:{}),...(policy.highVoltage?{highVoltage:policy.highVoltage}:{})}
 });
}

/** Apply this coefficient to the target's native outcome-scaled damage, before
 * IWR. The shared card roll stays at full untyped base. Pass creature traits only:
 * an electricity immunity/resistance or Shocked state is not an associated trait. */
export function siphonMultiplier(snapshot,targetTraits){
 if(snapshot?.kind!=='siphoning'||snapshot?.siphon?.applies!==true)return 1;
 const target=new Set(strings(targetTraits));
 return snapshot.disruptive&&snapshot.associatedTraits?.some(trait=>target.has(trait))?1:0.5;
}
