import {TREAT_WOUNDS_IMMUNITY,sourceId,values} from '../salubrious-kiss-rules.mjs';
import {canonicalItemSource} from './source-ids.mjs';
export const improvedRefocusSlugs=new Set(['bloodline','bonded','conflux','devoted','domain','hex','inspirational','link','meditative','primal','wardens'].flatMap(name=>[`${name}-focus`,`${name}-wellspring`]));
export function refocusUnsupported(items){return values(items).filter(i=>!i.isSuppressed&&!i.system?.suppressed&&i.type==='feat'&&improvedRefocusSlugs.has(i.slug??i.system?.slug)).map(i=>i.slug??i.system?.slug)}
export function treatablePatient(patient,healer){return patient.modeOfBeing==='living'||patient.modeOfBeing==='undead'&&healer.slugs.includes('stitch-flesh')}
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
    const statistics={},treatmentEstimate={};for(const skill of ['medicine','nature','occultism']){const stat=actor.getStatistic?.(skill);statistics[skill]={rank:stat?.rank??0,mod:stat?.mod??stat?.check?.mod??null};treatmentEstimate[skill]={ready:false,reason:'native-context-unverified'};
      try{const extra=[...actor.getRollOptions(['all','skill-check',skill]),'action:treat-wounds',`check:statistic:${skill}`,'check:type:skill',...['exploration','healing','manipulate'].flatMap(t=>[t,`item:trait:${t}`])],check=stat.withRollOptions({extraRollOptions:extra}).check,domains=check.domains;
       const rules=values(actor.rules).filter(r=>!r.ignored),dynamicRules=rules.some(r=>typeof r.beforeRoll==='function'&&!(r.key==='ActiveEffectLike'&&r.phase!=='beforeRoll'));
       const dynamic=dynamicRules||check.modifiers.some(m=>m.predicate?.length||m.adjustments?.length)||domains.some(d=>actor.synthetics?.degreeOfSuccessAdjustments?.[d]?.length||actor.synthetics?.rollTwice?.[d]?.length||actor.synthetics?.rollSubstitutions?.[d]?.some(s=>s.required))||slugs.some(s=>['magic-hands','mortal-healing'].includes(s));
       if(!dynamic){const opts=check.createRollOptions({origin:actor,extraRollOptions:extra}),modifier=new game.pf2e.CheckModifier(skill,{modifiers:check.modifiers},[],opts);if(Number.isFinite(modifier.totalModifier))treatmentEstimate[skill]={ready:true,modifier:modifier.totalModifier,domains:[...domains],options:[...opts],sourceVersion:'8.5.1'}}
      }catch{}
    }
    const assuranceSkills=feats.filter(i=>(i.slug??i.system?.slug)==='assurance').map(i=>i.flags?.pf2e?.rulesSelections?.assurance??i.flags?.system?.rulesSelections?.assurance).filter(Boolean);
    const now=game.time?.worldTime??0,immunities=items.filter(i=>canonicalItemSource(sourceId(i))===TREAT_WOUNDS_IMMUNITY).map(i=>({id:i.uuid,expiresAt:immunityExpiry(i,now),originActorUUID:i.system?.context?.origin?.actor}));
    const pool=hpPools.discover(actor);const master=pool.poolUUID===uuid?actor:await fromUuid(pool.poolUUID);
    return {actorUUID:uuid,name:actor.name,level:actor.level,isDead:!!actor.isDead,unconscious:!!actor.hasCondition?.('unconscious'),wounded:!!actor.hasCondition?.('wounded'),modeOfBeing:actor.modeOfBeing,...statistics,treatmentEstimate,healingExpectationReady:!actor.synthetics?.statisticsModifiers?.['healing-received']?.length&&!actor.synthetics?.damageDice?.['healing-received']?.length,slugs,assuranceSkills,items:items.map(i=>({uuid:i.uuid,sourceId:sourceId(i),slug:i.slug??i.system?.slug,type:i.type})),
      wardCapacity:wardCapacity({wardMedic:slugs.includes('ward-medic'),medicineRank:statistics.medicine.rank}),
      continualRecovery:slugs.includes('continual-recovery'),riskySurgery:slugs.includes('risky-surgery'),threePecks:feats.some(i=>sourceId(i)==='Compendium.pf2e.feats-srd.Item.Qg5M34t95rtT0sOp'),
      hasActiveToken:typeof actor.getActiveTokens==='function'?actor.getActiveTokens(false,true).length>0:undefined,hp:{value:master?.system?.attributes?.hp?.value??0,max:master?.system?.attributes?.hp?.max??0,temp:master?.system?.attributes?.hp?.temp??0},focus:structuredClone(actor.system?.resources?.focus??{value:0,max:0}),pool,immunities,
      cooldownExpiresAt:immunities.length?Math.max(...immunities.map(i=>i.expiresAt)):null,refocusUnsupported:refocusUnsupported(items),unsupported:slugs.filter(s=>['mortal-healing'].includes(s))};
  }
  return {discover,snapshot:async uuids=>Promise.all(uuids.map(discover)),activePassiveRules:async()=>{
    let enabled=false;try{enabled=game.modules?.get('patreon-v3')?.active===true&&game.settings.get('patreon-v3','fastHealingTime')===true}catch{}if(!enabled)return [];
    const parties=values(game.actors).filter(a=>a.type==='party'),members=[...new Map([...parties.flatMap(a=>values(a.members)),...values(game.actors?.party?.members)].map(a=>[a.uuid,a])).values()];return members.flatMap(actor=>values(actor.rules).filter(r=>['FastHealing','Regeneration'].includes(r.key)&&!r.ignored).map(rule=>{
      let passing;try{passing=typeof rule.test==='function'?rule.test():rule.predicate?.test?.(actor.getRollOptions?.(['all'])??[])??false}catch{passing=true}
      return {actorUUID:actor.uuid,providerId:'pf2e-patreon',key:rule.key,passing};
    }));
  }};
}
