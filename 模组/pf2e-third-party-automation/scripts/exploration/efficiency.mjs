const ranks=['trained','expert','master','legendary'],dcs=[15,20,30,40],bonuses=[0,10,30,50];
const distributions=new Map();
function meanCapped(dice,bonus,deficit){if(!distributions.has(dice)){let dist=new Map([[0,1]]);for(let i=0;i<dice;i++){const next=new Map();for(const [sum,p]of dist)for(let n=1;n<=8;n++)next.set(sum+n,(next.get(sum+n)??0)+p/8);dist=next}distributions.set(dice,dist)}return [...distributions.get(dice)].reduce((n,[sum,p])=>n+Math.min(deficit,sum+bonus)*p,0)}
/** This gate excludes dynamic native modifiers and outcomes. No future roll is sampled. */
export function selectTreatmentRank(healer,patients,options,deficit){
 const skill=options.skill??'medicine',skillRank=healer[skill]?.rank??0,requested=options.treatmentRank??'trained';
 if(requested!=='auto')return ranks.indexOf(requested)<skillRank?{rank:requested,estimate:'fixed-selected-dc',expectedNetHealing:patients.reduce((n,p)=>n+Math.min(deficit(p),9),0)}:null;
 const context=healer.treatmentEstimate?.[skill];
 if(!context?.ready||options.riskySurgery||options.assurance||patients.some(p=>!p.healingExpectationReady))return skillRank>=1?{rank:'trained',estimate:'fixed-dc-unverified-context',expectedNetHealing:patients.reduce((n,p)=>n+Math.min(deficit(p),9),0)}:null;
 let best;
 for(let i=0;i<skillRank;i++){let net=0;const bonus=bonuses[i]+(healer.slugs.includes('medic-dedication')?[0,5,10,15][i]:0);
  for(let face=1;face<=20;face++){const total=face+context.modifier;let degree=total>=dcs[i]+10?3:total>=dcs[i]?2:total<=dcs[i]-10?0:1;degree=Math.max(0,Math.min(3,degree+(face===20?1:face===1?-1:0)));net+=patients.reduce((n,p)=>n+(degree===0?-4.5:degree>=2?meanCapped(degree===3?4:2,bonus,deficit(p)):0),0)/20}
  if(!best||net>best.expectedNetHealing+1e-8)best={rank:ranks[i],estimate:'verified-unconditional-native-context',expectedNetHealing:net};
 }return best;
}
