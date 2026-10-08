const ranks=['trained','expert','master','legendary'];
const dcs=[15,20,30,40];
const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const clamp=value=>Math.max(0,Math.min(3,value));
const invalid=()=>{throw Error('invalid-treatment-context')};

function validateAdjustments(adjustments){
 if(!Array.isArray(adjustments))invalid();
 for(const entry of adjustments){
  if(!entry||typeof entry.adjustments!=='object'||entry.adjustments===null||Array.isArray(entry.adjustments)||entry.predicate!==undefined&&typeof entry.predicate?.test!=='function')invalid();
  for(const [key,value] of Object.entries(entry.adjustments)){
   if(!['all',...outcomes].includes(key)||!value||typeof value.label!=='string'||!value.label||![-2,-1,0,1,2,...outcomes].includes(value.amount))invalid();
  }
 }
}

function degree(face,total,dc,adjustments){
 let outcome=total>=dc+10?3:total>=dc?2:total<=dc-10?0:1;
 outcome=clamp(outcome+(face===20?1:face===1?-1:0));
 for(const key of ['all',...outcomes]){
  const adjustment=adjustments[key],amount=adjustment?.amount;
  if(!amount||!adjustment.label||outcome===3&&amount===1||outcome===0&&amount===-1)continue;
  if(key!=='all'&&outcomes.indexOf(key)!==outcome)continue;
  return typeof amount==='string'?outcomes.indexOf(amount):clamp(outcome+amount);
 }
 return outcome;
}

/** Mirror the native degree calculation; no dice or native action runs. */
export function treatmentOutcomeRows({rank,modifier,assurance=false,options=new Set(),adjustments=[]}){
 if(!Number.isInteger(rank)||rank<1||rank>4||!Number.isFinite(modifier)||typeof assurance!=='boolean'||!(options instanceof Set))invalid();
 validateAdjustments(adjustments);
 return ranks.slice(0,rank).map((name,index)=>{
  const dc=dcs[index],faces=assurance?[10]:Array.from({length:20},(_,i)=>i+1);
  const cases=faces.map(face=>{
   const total=face+modifier,natural=assurance?undefined:face;
   const totals=[`check:total:${total}`,`check:total:natural:${natural}`,`check:roll:total:natural:${natural}`,`check:total:delta:${total-dc}`],facts=new Set([...options,...totals]);
   const selected={};
   for(const entry of adjustments){
    if(entry.predicate&&!entry.predicate.test(entry.options?new Set([...totals,...entry.options]):facts))continue;
    for(const key of ['all',...outcomes])if(entry.adjustments[key])selected[key]={...entry.adjustments[key]};
   }
   return {weight:assurance?1:0.05,outcome:degree(face,total,dc,selected)};
  });
  return {rank:name,dc,cases};
 });
}
