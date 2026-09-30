/** PF2e simplifies deterministic nested sums unless an operand retains a flavor.
 * Preserve this native boundary so merging cannot flatten away critical terms.
 * Existing precision, splash, and persistent flavors retain their native meaning. */
export function preserveDamagePartForMerge(roll){
 for(const instance of roll.instances??[])if(instance.head){instance.head.options??={};instance.head.options.flavor??='damage';}
 return roll;
}

const MODULE_ID='pf2e-third-party-automation';
const validMax=max=>max===null||max===Infinity||(Number.isFinite(max)&&max>=0);
const literal=(type,types)=>typeof type==='string'&&Object.hasOwn(types??{},type);
const encode=entry=>({type:entry.type,max:entry.max===Infinity?'Infinity':entry.max});
const decode=entry=>({type:entry.type,max:entry.max==='Infinity'?Infinity:entry.max});
const same=(a,b)=>a.type===b.type&&Object.is(a.max,b.max);

function installSafeBypass(roll,entries){
 roll.options??={};
 // PF2e 8.5 reads this exact structure. Strings belong only in our JSON proof;
 // native Roll/ChatMessage JSON serializes numeric Infinity as null.
 roll.options.bypass={immunity:{ignore:[],downgrade:[],redirect:[]},resistance:{ignore:entries.map(e=>({...e})),redirect:[]}};
 roll.options[MODULE_ID]={...roll.options[MODULE_ID],safeDamageBypass:{version:1,ignore:entries.map(encode)}};
 return roll;
}

/** Restore only literal-type ignore entries common to every original contributor
 * of that type. parts must be the original DamageRoll[], never pairwise merges.
 * Pending persistent instances count: their total is normally zero. */
export function preserveMergedDamageBypass(mergedRoll,parts,damageTypes=globalThis.CONFIG?.PF2E?.damageTypes){
 if(!mergedRoll||typeof mergedRoll!=='object')return mergedRoll;
 if(!Array.isArray(parts)||!parts.length||parts.some(p=>!Array.isArray(p?.instances)))return installSafeBypass(mergedRoll,[]);
 const byType=new Map();
 for(const part of parts)for(const type of new Set(part.instances.map(i=>i.type))){
  if(!literal(type,damageTypes))continue;
  const list=byType.get(type)??[];list.push(part);byType.set(type,list);
 }
 const safe=[];
 for(const[type,contributors]of byType){
  const entries=part=>Array.isArray(part.options?.bypass?.resistance?.ignore)?part.options.bypass.resistance.ignore.filter(e=>e?.type===type&&validMax(e.max)):[];
  for(const entry of entries(contributors[0]))if(!safe.some(e=>same(e,entry))&&contributors.every(p=>entries(p).some(e=>same(e,entry))))safe.push({type,max:entry.max});
 }
 return installSafeBypass(mergedRoll,safe);
}

/** Called by the existing single DamageRoll.alter wrapper after native alter.
 * Only our persistent proof is propagated. Positive scaling preserves sources;
 * an addend has no proven source and invalidates its first damage type. */
export function preserveDamageBypassOnAlter(original,result,{multiplier=1,addend=0,damageTypes=globalThis.CONFIG?.PF2E?.damageTypes}={}){
 const proof=original?.options?.[MODULE_ID]?.safeDamageBypass;
 if(!result||typeof result!=='object'||proof?.version!==1||!Array.isArray(proof.ignore))return result;
 if(proof.ignore.some(e=>!e||typeof e!=='object'))return result;
 const entries=proof.ignore.map(decode);
 if(entries.some(e=>!literal(e.type,damageTypes)||!validMax(e.max)))return result;
 const firstType=original.instances?.[0]?.type;
 const safe=Number.isFinite(multiplier)&&multiplier>0&&Number.isFinite(addend)?entries.filter(e=>addend===0||firstType&&e.type!==firstType):[];
 return installSafeBypass(result,safe);
}
