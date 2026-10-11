export function exactNativeClock(clock){
 return clock.nativeResolved===true&&clock.evidence?.some(receipt=>{
  const source=receipt.options?.pf2eThirdPartyAutomation?.exploration;
  return source?.checkpointId===clock.id&&source.sessionId===clock.sessionId&&source.gmId===clock.gmId&&source.expectedFrom===clock.from&&source.expectedTo===clock.to&&receipt.userId===clock.gmId&&receipt.worldTime===clock.to&&receipt.dt===clock.to-clock.from;
 })===true;
}

const canAcceptClock=clock=>clock.state==='uncertain'&&clock.reason==='time-effects-unconfirmed'&&exactNativeClock(clock);

export function clockBlocksRecovery(state,clock){
 if(clock.state==='confirmed')return false;
 const session=state.sessions[clock.sessionId],review=session?.recoveryReview;
 return !(session?.status==='closed'&&session.manual!==true&&review?.disposition==='accepted-current-state'&&typeof review.userId==='string'&&review.userId&&Number.isFinite(review.at)&&Array.isArray(review.clockIds)&&review.clockIds.includes(clock.id)&&canAcceptClock(clock));
}

export function currentTimeRecoveryReason(state,sessionId){
 const session=state.sessions[sessionId];
 if(!session||session.manual===true||!['paused','closed'].includes(session.status))return 'paused-automatic-session-required';
 if(session.recoveryReview)return 'current-time-already-accepted';
 if(Object.values(state.sessions).some(row=>row.manual!==true&&row.status==='running'))return 'recovery-session-already-running';
 for(const activity of Object.values(state.activities).filter(row=>row.sessionId===sessionId)){
  if(!['confirmed','cancelled','blocked'].includes(activity.state)||activity.executor&&activity.executor.state!=='settled'||Object.values(activity.proof?.poolApplications??{}).some(claim=>claim.state!=='settled')||[activity.proof?.battleMedicineAttempt,activity.proof?.medicBypass].some(claim=>claim&&!['used','released'].includes(claim.state)))return 'unresolved-activity';
 }
 if(Object.values(state.clocks).some(clock=>clock.sessionId===sessionId&&clock.state!=='confirmed'&&!canAcceptClock(clock)))return 'unresolved-clock';
 return null;
}
