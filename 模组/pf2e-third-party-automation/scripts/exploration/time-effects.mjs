export function createTimeEffects({capabilities,completionAdapters=[]}) {
 const checkpoints=new Map(),rests=[];let last=null;
 async function beforeAdvance(checkpoint){
  const rules=(await capabilities.activePassiveRules()).filter(r=>r.passing),selected=new Set();
  for(const rule of rules){let adapter;for(const a of completionAdapters)if(await a.matches(rule,checkpoint)){adapter=a;break}
   if(!adapter)return last={status:'blocked',reason:'passive-completion-unavailable',rule};selected.add(adapter);
  }
  for(const a of selected){const result=await a.beforeAdvance(checkpoint);if(result.status!=='ready')return last=result}
  checkpoints.set(checkpoint.id,{selected,rules});return last={status:'ready'};
 }
 async function settle(checkpoint){
  const pending=checkpoints.get(checkpoint.id);if(!pending)return {status:'uncertain',reason:'missing-passive-checkpoint'};
  const current=(await capabilities.activePassiveRules()).filter(r=>r.passing);
  if(current.some(r=>!pending.rules.some(p=>JSON.stringify(p)===JSON.stringify(r))))return last={status:'uncertain',reason:'passive-rules-changed'};
  const proof=[];for(const a of pending.selected){const result=await a.settle(checkpoint);if(result.status!=='ready'||!result.proof?.length)return last={status:'uncertain',reason:result.reason??'passive-completion-unproven'};proof.push(...result.proof)}
  checkpoints.delete(checkpoint.id);return last={status:'ready',proof};
 }
 return {beforeAdvance,settle,observeRest(event){const result={status:event.invocationId&&event.timeReceipt?.invocationId===event.invocationId?'observed':'uncertain',reason:'external-rest-time-authority',event};rests.push(result);return result},diagnostic:()=>({last,rests:[...rests]})};
}
