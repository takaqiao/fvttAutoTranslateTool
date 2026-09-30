export function createClock({game,Hooks,ledger,timeEffects,isAuthority,confirmationTimeoutMs=10000}){
 let busy=false,close;const authority=()=>{if(!isAuthority())throw Error('active-gm-required')};
 async function reconcile(commit){const prior=await ledger.getClockCommit(commit.id);if(prior?.state==='confirmed')return {status:'confirmed',commit:prior};const exact=prior?.nativeResolved===true&&prior.effectsSettled===true&&prior.evidence.some(r=>{const e=r.options?.pf2eThirdPartyAutomation?.exploration;return e?.checkpointId===prior.id&&e.sessionId===prior.sessionId&&e.gmId===prior.gmId&&e.expectedFrom===prior.from&&e.expectedTo===prior.to&&r.userId===prior.gmId&&r.worldTime===prior.to&&r.dt===prior.to-prior.from});if(exact){authority();const saved=await ledger.transitionClockCommit(prior.id,{expected:['started','uncertain'],patch:{state:'confirmed',reason:null,reconciled:true}});return {status:'confirmed',commit:saved}}return {status:'uncertain',reason:'clock-never-replayed',commit:prior??commit}}
 async function advanceTo(input){
  const prior=await ledger.getClockCommit(input.id);if(prior)return reconcile(input);
  if(busy)return {status:'blocked',reason:'clock-in-flight'};
  busy=true;let committed=false,hook,timer,deadlineTimer;
  const commit={...input,gmId:game.user.id,state:'started',evidence:[]};
  try{
   authority();if(!Number.isFinite(commit.from)||!Number.isFinite(commit.to)||commit.to<commit.from||game.time.worldTime!==commit.from)return {status:'blocked',reason:'world-time-conflict'};
   const passive=await timeEffects.beforeAdvance(commit);authority();if(passive.status!=='ready')return passive;
   await ledger.upsertClockCommit(commit);committed=true;authority();if(game.time.worldTime!==commit.from)throw Error('world-time-conflict-after-claim');
   let receipt,foreign=false,resolve;const observed=new Promise(yes=>{resolve=yes});
   const finish=value=>{clearTimeout(timer);resolve(value)};close=()=>finish(null);
   hook=Hooks.on('updateWorldTime',(worldTime,dt,options,userId)=>{
    const e=options?.pf2eThirdPartyAutomation?.exploration;
    if(!isAuthority()||e?.checkpointId!==commit.id||e.sessionId!==commit.sessionId||e.expectedFrom!==commit.from||e.expectedTo!==commit.to||e.gmId!==commit.gmId||userId!==commit.gmId||worldTime!==commit.to||dt!==commit.to-commit.from){foreign=true;finish(null);return}
    receipt={worldTime,dt,options,userId};finish(receipt);
   });
   timer=setTimeout(()=>finish(null),confirmationTimeoutMs);
   const options={pf2eThirdPartyAutomation:{exploration:{sessionId:commit.sessionId,checkpointId:commit.id,expectedFrom:commit.from,expectedTo:commit.to,gmId:commit.gmId}}};
   // The hook receipt and native operation must both settle, within the same bounded attempt.
   const call=Promise.resolve(game.time.advance(commit.to-commit.from,options));call.catch(()=>{});
   const result=await Promise.race([Promise.all([call,observed]).then(([,r])=>r),new Promise(yes=>{deadlineTimer=setTimeout(()=>yes(null),confirmationTimeoutMs)})]);
   authority();if(!result||foreign||game.time.worldTime!==commit.to)throw Error('world-time-source-unconfirmed');
   await ledger.transitionClockCommit(commit.id,{expected:['started'],patch:{nativeResolved:true,evidence:[receipt]}});authority();
   const effects=await timeEffects.settle(commit);authority();if(effects.status!=='ready'||game.time.worldTime!==commit.to)throw Error(effects.reason??'time-effects-unconfirmed');
   const saved=await ledger.transitionClockCommit(commit.id,{expected:['started'],patch:{state:'confirmed',effectsSettled:true,evidence:[receipt,...effects.proof??[]]}});authority();return {status:'confirmed',commit:saved};
  }catch(error){if(committed&&isAuthority())await ledger.transitionClockCommit(commit.id,{expected:['started'],patch:{state:'uncertain',reason:error.message}}).catch(()=>{});return {status:committed?'uncertain':'blocked',reason:error.message,commit}}
  finally{if(hook)Hooks.off('updateWorldTime',hook);clearTimeout(timer);clearTimeout(deadlineTimer);close=null;busy=false}
 }
 return {advanceTo,reconcile,stop(){close?.()}};
}
