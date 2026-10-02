export function createClock({game,Hooks,ledger,timeEffects,isAuthority,confirmationTimeoutMs=10000}){
 let active=null;const attempts=new Set(),authority=()=>{if(!isAuthority())throw Error('active-gm-required')};
 async function reconcile(commit){const prior=await ledger.getClockCommit(commit.id);if(prior?.state==='confirmed')return {status:'confirmed',commit:prior};const exact=prior?.nativeResolved===true&&prior.effectsSettled===true&&prior.evidence.some(r=>{const e=r.options?.pf2eThirdPartyAutomation?.exploration;return e?.checkpointId===prior.id&&e.sessionId===prior.sessionId&&e.gmId===prior.gmId&&e.expectedFrom===prior.from&&e.expectedTo===prior.to&&r.userId===prior.gmId&&r.worldTime===prior.to&&r.dt===prior.to-prior.from});if(exact){authority();const saved=await ledger.transitionClockCommit(prior.id,{expected:['started','uncertain'],patch:{state:'confirmed',reason:null,reconciled:true},reconcile:true});return {status:'confirmed',commit:saved}}return {status:'uncertain',reason:'clock-never-replayed',commit:prior??commit}}
 async function advanceTo(input,{leaseNonce,guard}={}){
  const slot={sessionId:input.sessionId,leaseNonce,nativeIssued:false,close:null,cancelReason:null};attempts.add(slot);let committed=false,nativeIssued=false,hook,timer,deadlineTimer;
  const commit={...input,gmId:game.user.id,state:'started',evidence:[]};
  const current=()=>{authority();if(slot.cancelReason!==null)throw Error(slot.cancelReason);if(game.combat?.started)throw Error('encounter-started')};
  const scope={leaseNonce};
  let latestSession;
  // The from-time guard ends at issuance; late receipts and effects keep their original settlement path.
  const preNative=(session=latestSession)=>{current();if(game.time.worldTime!==commit.from)throw Error('world-time-conflict');if(guard!==undefined){if(typeof guard!=='function')throw Error('synchronous-checkpoint-guard-required');if(guard(session)!==true)throw Error('activity-checkpoint-changed')}return true};
  const eligible=async()=>{current();if(typeof ledger.getSession==='function'){const session=await ledger.getSession(commit.sessionId);current();if(ledger.atomic===true&&!ledger.ownsSession(session,scope))throw Error('session-driver-required');if(session?.status!=='running')throw Error('recovery-session-stopped');latestSession=session}if(!nativeIssued)preNative()};
  try{
   const prior=await ledger.getClockCommit(input.id);if(prior)return reconcile(input);
   if(active)return {status:'blocked',reason:'clock-in-flight'};
   active=slot;
   await eligible();if(!Number.isFinite(commit.from)||!Number.isFinite(commit.to)||commit.to<commit.from||game.time.worldTime!==commit.from)return {status:'blocked',reason:'world-time-conflict'};
   const passive=await timeEffects.beforeAdvance(commit);await eligible();if(passive.status!=='ready')return passive;
   await ledger.upsertClockCommit(commit,{...scope,guard:preNative});committed=true;await eligible();if(game.time.worldTime!==commit.from)throw Error('world-time-conflict-after-claim');
   let receipt,foreign=false,resolve;const observed=new Promise(yes=>{resolve=yes});
   const finish=value=>{clearTimeout(timer);resolve(value)};slot.close=()=>finish(null);
   hook=Hooks.on('updateWorldTime',(worldTime,dt,options,userId)=>{
    const e=options?.pf2eThirdPartyAutomation?.exploration;
    if(!isAuthority()||e?.checkpointId!==commit.id||e.sessionId!==commit.sessionId||e.expectedFrom!==commit.from||e.expectedTo!==commit.to||e.gmId!==commit.gmId||userId!==commit.gmId||worldTime!==commit.to||dt!==commit.to-commit.from){foreign=true;finish(null);return}
    receipt={worldTime,dt,options,userId};finish(receipt);
   });
   timer=setTimeout(()=>finish(null),confirmationTimeoutMs);
   const options={pf2eThirdPartyAutomation:{exploration:{sessionId:commit.sessionId,checkpointId:commit.id,expectedFrom:commit.from,expectedTo:commit.to,gmId:commit.gmId}}};
   // The hook receipt and native operation must both settle, within the same bounded attempt.
   preNative();nativeIssued=true;slot.nativeIssued=true;const call=Promise.resolve(game.time.advance(commit.to-commit.from,options));call.catch(()=>{});
   const result=await Promise.race([Promise.all([call,observed]).then(([,r])=>r),new Promise(yes=>{deadlineTimer=setTimeout(()=>yes(null),confirmationTimeoutMs)})]);
   authority();if(!result||foreign||game.time.worldTime!==commit.to)throw Error('world-time-source-unconfirmed');
   await ledger.transitionClockCommit(commit.id,{expected:['started'],patch:{nativeIssued:true,nativeResolved:true,evidence:[receipt]},...scope});await eligible();
   const effects=await timeEffects.settle(commit);await eligible();if(effects.status!=='ready'||game.time.worldTime!==commit.to)throw Error(effects.reason??'time-effects-unconfirmed');
   const saved=await ledger.transitionClockCommit(commit.id,{expected:['started'],patch:{state:'confirmed',effectsSettled:true,evidence:[receipt,...effects.proof??[]]},...scope});authority();return {status:'confirmed',commit:saved};
  }catch(error){if(committed&&isAuthority())await ledger.transitionClockCommit(commit.id,{expected:['started'],patch:{state:'uncertain',nativeIssued,reason:error.message},...scope}).catch(()=>{});return {status:committed?'uncertain':'blocked',reason:error.message,commit}}
  finally{attempts.delete(slot);if(hook)Hooks.off('updateWorldTime',hook);clearTimeout(timer);clearTimeout(deadlineTimer);if(active===slot)active=null}
 }
 return {advanceTo,reconcile,stop(reason='clock-stopped',scope){for(const slot of attempts){if(scope&&(slot.sessionId!==scope.sessionId||scope.leaseNonce!==undefined&&slot.leaseNonce!==scope.leaseNonce))continue;slot.cancelReason=reason;slot.close?.();if(active===slot&&!slot.nativeIssued)active=null}}};
}
