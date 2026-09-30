import {TREAT_WOUNDS_IMMUNITY,sourceId,values} from '../salubrious-kiss-rules.mjs';
export function wardCapacity({wardMedic,medicineRank}) {return wardMedic&&medicineRank>=2?2**(Math.min(4,medicineRank)-1):1}
export function cooldown({startedAt,finishedAt,continualRecovery}) {const expiresAt=startedAt+(continualRecovery?600:3600);return {expiresAt,remainingSeconds:Math.max(0,expiresAt-finishedAt)}}
export function earliestTreatmentStart({now,existingExpiresAt}) {return Math.max(now,existingExpiresAt??now)}
export function immunityExpiry(item,now) {
  if(item.isExpired===true||item.remainingDuration?.expired===true)return null;
  const explicit=item.flags?.['pf2e-third-party-automation']?.salubriousKiss?.expiresAt;
  if(Number.isFinite(explicit))return explicit;
  const start=item.system?.start?.value,duration=item.system?.duration;
  const multiplier={rounds:6,minutes:60,hours:3600,days:86400}[duration?.unit];
  if(Number.isFinite(start)&&Number.isFinite(duration?.value)&&multiplier)return start+duration.value*multiplier;
  if(Number.isFinite(item.remainingDuration?.remaining))return now+item.remainingDuration.remaining;
  return Infinity;
}
export function createCapabilities({game,fromUuid,hpPools}) {
  async function discover(uuid) {
    const actor=await fromUuid(uuid);if(!actor)throw Error('actor-unavailable');
    const items=values(actor.items).filter(i=>!i.isSuppressed&&!i.system?.suppressed);
    const feats=items.filter(i=>i.type==='feat');const slugs=feats.map(i=>i.slug??i.system?.slug);
    const statistics={};for(const skill of ['medicine','nature','occultism']){const stat=actor.getStatistic?.(skill);statistics[skill]={rank:stat?.rank??0,mod:stat?.mod??stat?.check?.mod??null}}
    const assuranceSkills=feats.filter(i=>(i.slug??i.system?.slug)==='assurance').map(i=>i.flags?.pf2e?.rulesSelections?.assurance??i.flags?.system?.rulesSelections?.assurance).filter(Boolean);
    const now=game.time?.worldTime??0,immunities=items.filter(i=>sourceId(i)===TREAT_WOUNDS_IMMUNITY).map(i=>({id:i.uuid,expiresAt:immunityExpiry(i,now),originActorUUID:i.system?.context?.origin?.actor}));
    const pool=hpPools.discover(actor);const master=pool.poolUUID===uuid?actor:await fromUuid(pool.poolUUID);
    return {actorUUID:uuid,name:actor.name,level:actor.level,isDead:!!actor.isDead,unconscious:!!actor.hasCondition?.('unconscious'),wounded:!!actor.hasCondition?.('wounded'),modeOfBeing:actor.modeOfBeing,...statistics,slugs,assuranceSkills,items:items.map(i=>({uuid:i.uuid,sourceId:sourceId(i),slug:i.slug??i.system?.slug,type:i.type})),
      wardCapacity:wardCapacity({wardMedic:slugs.includes('ward-medic'),medicineRank:statistics.medicine.rank}),
      continualRecovery:slugs.includes('continual-recovery'),riskySurgery:slugs.includes('risky-surgery'),threePecks:feats.some(i=>sourceId(i)==='Compendium.pf2e.feats-srd.Item.Qg5M34t95rtT0sOp'),
      hp:structuredClone(master?.system?.attributes?.hp??{}),focus:structuredClone(actor.system?.resources?.focus??{value:0,max:0}),pool,immunities,
      cooldownExpiresAt:Math.max(now,...immunities.map(i=>i.expiresAt)),unsupported:slugs.filter(s=>['mortal-healing'].includes(s))};
  }
  return {discover,snapshot:async uuids=>Promise.all(uuids.map(discover)),activePassiveRules:async()=>{
    const members=values(game.actors?.party?.members);return members.flatMap(actor=>values(actor.rules).filter(r=>['FastHealing','Regeneration'].includes(r.key)&&!r.ignored).map(rule=>{
      let passing;try{passing=typeof rule.test==='function'?rule.test():rule.predicate?.test?.(actor.getRollOptions?.(['all'])??[])??false}catch{passing=true}
      return {actorUUID:actor.uuid,providerId:'pf2e-patreon',key:rule.key,passing};
    }));
  }};
}
