/** Roaring Applause's current rank-3, single-target lifecycle only.
 * Pure JSON in/out: callers authenticate native/GM facts and verify owned grants.
 * `ended` is a logical terminal, never a claim that document deletion succeeded.
 * Commands are source-scoped intentions; the executor owns durable operations.
 */
const SOURCE='Compendium.pf2e.spells-srd.Item.czO0wbT1i320gcu9';
const OUTCOMES=['criticalSuccess','success','failure','criticalFailure'];
const EVENTS=['save-confirmed','save-unverified','sustain-use','sustain-settled','reconcile','turn-end','target-start','continuity-unverified','clock','own-child-deleted','end-fascination','own-parent-deleted'];
const OBSERVED=['save-confirmed','save-unverified','sustain-use','sustain-settled','reconcile','turn-end','target-start','continuity-unverified'];
const clone=value=>structuredClone(value);
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const own=(object,key)=>Object.hasOwn(object,key);
function demand(value,message){if(!value)throw Error(`Roaring lifecycle: ${message}`)}
function text(value,name){demand(typeof value==='string'&&value.length>0&&value.length<=512,`invalid ${name}`);return value}
function key(value,name){text(value,name);demand(/^[A-Za-z0-9-]{1,80}$/.test(value)&&!['constructor','prototype'].includes(value),`unsafe ${name}`);return value}
function time(value,name){demand(Number.isFinite(value),`invalid ${name}`);return value}
function integer(value,name,min=0){demand(Number.isSafeInteger(value)&&value>=min,`invalid ${name}`);return value}
function optionalRound(value,name){return value===null||value===undefined?null:integer(value,name)}
function frame(value){
  demand(value&&typeof value==='object','missing actual turn');
  const f={combatId:text(value.combatId,'combatId'),combatantId:text(value.combatantId,'combatantId'),actorUuid:text(value.actorUuid,'turn actorUuid'),tokenUuid:text(value.tokenUuid,'turn tokenUuid'),started:value.started,round:integer(value.round,'round',1),turn:integer(value.turn,'turn'),lastTurnEnd:optionalRound(value.lastTurnEnd,'lastTurnEnd'),latestTurnEndRound:optionalRound(value.latestTurnEndRound,'latestTurnEndRound')};
  demand(typeof f.started==='boolean'&&Array.isArray(value.order)&&value.order.length>0,'invalid encounter/order');
  f.order=value.order.map(c=>{const initiative=c.initiative??null,overridePriority=c.overridePriority??null;demand((initiative===null||Number.isFinite(initiative))&&(overridePriority===null||Number.isFinite(overridePriority)),'invalid initiative/priority');return {id:text(c.id,'order id'),initiative,overridePriority}});
  demand(f.turn<f.order.length&&new Set(f.order.map(c=>c.id)).size===f.order.length,'invalid/ambiguous turn order');
  return f;
}
function normalTurn(f,scope){
  demand(f.started&&f.actorUuid===scope.casterActorUuid&&f.tokenUuid===scope.casterTokenUuid&&f.order[f.turn]?.id===f.combatantId,'not the caster normal own turn');
  demand(f.lastTurnEnd===null||f.lastTurnEnd<f.round,'this caster turn already ended');
}
function finite(value,worldTime,f){
  const start=value?.start,duration=value?.duration;
  demand(start&&duration&&Number.isFinite(start.value)&&start.value<=worldTime,'invalid/future finite start');
  demand((start.initiative===null||Number.isFinite(start.initiative))&&start.initiative===f.order.find(c=>c.id===f.combatantId)?.initiative,'finite initiative differs from caster');
  demand(duration.value===1&&duration.unit==='rounds'&&duration.expiry==='turn-end'&&duration.sustained===false,'expected frozen one-round finite envelope');
  return {start:{value:start.value,initiative:start.initiative},duration:{value:1,unit:'rounds',expiry:'turn-end',sustained:false}};
}
function deadline(f){return {combatId:f.combatId,combatantId:f.combatantId,actorUuid:f.actorUuid,tokenUuid:f.tokenUuid,anchorRound:f.round,anchorTurn:f.turn,endRound:f.round+1,baselineEnded:f.lastTurnEnd,order:clone(f.order),turnKey:`${f.combatId}/${f.combatantId}/${f.round}`}}
function validSource(s){
  demand(s?.schema===1&&['awaiting-save','active','ended'].includes(s.status),'invalid source state');
  key(s.sourceNonce,'sourceNonce');
  demand(s.source?.sourceId===SOURCE&&s.source.rank===3&&s.hardStopAt===s.completedWorldTime+600&&Number.isFinite(s.completedWorldTime)&&Number.isFinite(s.clockHighWater),'invalid source scope/cap');
  demand(['exact','manual-finite'].includes(s.timing?.mode)&&s.timing.deadline?.endRound===s.timing.deadline?.anchorRound+1,'invalid source deadline');
  demand(s.timing.deadline.actorUuid===s.source.casterActorUuid&&s.timing.deadline.tokenUuid===s.source.casterTokenUuid,'source deadline identity changed');
  demand(s.clapReceipts===undefined||s.clapReceipts&&typeof s.clapReceipts==='object'&&!Array.isArray(s.clapReceipts),'invalid clap receipts');
}

/** Call only after an exact paid original Cast has completed in this normal turn.
 * completedWorldTime is that native completion fact, not later save/card time. */
export function createRoaringSource(input){
  const sourceNonce=key(input?.sourceNonce,'sourceNonce'),source={};
  for(const name of ['castNonce','sourceId','itemUuid','entryUuid','casterActorUuid','casterTokenUuid','targetActorUuid','targetTokenUuid','originalMessageUuid'])source[name]=text(input[name],name);
  key(source.castNonce,'castNonce');source.rank=input.rank;
  demand(source.sourceId===SOURCE&&source.rank===3,'unsupported source/rank');
  const completedWorldTime=time(input.completedWorldTime,'completedWorldTime'),f=frame(input.turn);normalTurn(f,source);
  demand(Number.isFinite(completedWorldTime+600)&&completedWorldTime+600>completedWorldTime,'unrepresentable maximum duration');
  return {schema:1,revision:0,sourceNonce,source,status:'awaiting-save',completedWorldTime,hardStopAt:completedWorldTime+600,clockHighWater:completedWorldTime,
    timing:{mode:'exact',deadline:deadline(f),finiteEnvelope:finite(input.finiteEnvelope,completedWorldTime,f),lastFrame:f,reason:null},
    result:null,manualReview:null,tombstones:{slowed:null,fascinated:null},sustainUses:{},clapReceipts:{},termination:null};
}

/** This is the source's desired rule state, not verified document presence or a
 * guarantee that an external reaction consumer currently enforces the fact. */
export function projectRoaringConditions(source){
  validSource(source);
  const active=source.status==='active',outcome=active?source.result?.outcome:null,failed=['failure','criticalFailure'].includes(outcome),fascinated=outcome==='criticalFailure'&&!source.tombstones.fascinated;
  return {sourceNonce:source.sourceNonce,noReactions:active&&['success','failure','criticalFailure'].includes(outcome),slowed:failed&&!source.tombstones.slowed?1:0,fascinated,clap:failed,
    subject:fascinated?{actorUuid:source.source.casterActorUuid,tokenUuid:source.source.casterTokenUuid}:null,manualReview:!!source.manualReview||source.timing.mode==='manual-finite',timingMode:source.timing.mode};
}

/** Every event carries sourceNonce and observation.worldTime. Events needing
 * current scope carry observation.turn (null means source structure is gone).
 * `sustain-use.turn` is the frozen invocation frame, distinct from current facts.
 * `target-start.targetTurn.lastTurnStart` is the native roundOfLastTurn flag;
 * its authentic new update is verified by the caller, never inferred here.
 * Persist the resulting clap receipt before delivering its once-only prompt.
 * Callers must validate sender, live cards and bilateral grant ownership first. */
export function reduceRoaringSource(source,event){
  validSource(source);demand(EVENTS.includes(event?.type),'unknown event');demand(event.sourceNonce===source.sourceNonce,'event source mismatch');
  const worldTime=time(event.observation?.worldTime,'observed worldTime');
  if(OBSERVED.includes(event.type))demand(own(event.observation,'turn'),'missing current source turn observation');
  const s=clone(source),commands=[];let decision='unchanged',reason=null;
  const command=(type,data={})=>commands.push({type,sourceNonce:s.sourceNonce,...clone(data)});
  const reject=why=>{decision='rejected';reason=why};
  const manual=why=>{decision='manual';reason=why;if(!s.manualReview){s.manualReview={reason:why,receiptId:event.receiptId??event.verdictId??null};command('manual-review',{reason:why})}};
  const end=why=>{if(s.status==='ended')return;s.status='ended';s.termination={reason:why,worldTime,receiptId:event.receiptId??event.verdictId??null};decision='applied';reason=why;command('source-end',{reason:why})};
  const elapsedFinite=()=>worldTime>s.timing.finiteEnvelope.start.value+6;
  const degrade=why=>{
    if(elapsedFinite()){end('finite-fallback-expired');return}
    if(s.timing.mode==='manual-finite')return;
    s.timing.mode='manual-finite';s.timing.reason=why;decision='manual';reason=why;
    command('restore-finite',{finiteEnvelope:s.timing.finiteEnvelope,hardStopAt:s.hardStopAt,reason:why});command('manual-review',{reason:why});
  };
  function observe(){
    if(s.status==='ended')return;
    const previous=s.clockHighWater;s.clockHighWater=Math.max(previous,worldTime);
    if(s.clockHighWater>=s.hardStopAt){end('maximum-duration');return}
    if(worldTime<previous){degrade('clock-rewind');return}
    if(s.timing.mode==='manual-finite'){if(elapsedFinite())end('finite-fallback-expired');return}
    if(!own(event.observation,'turn'))return;
    const raw=event.observation.turn;
    if(raw===null){degrade('source-structure-missing');return}
    let f;
    try{f=frame(raw)}catch{degrade('unprovable-turn-frame');return}
    const d=s.timing.deadline,previousFrame=s.timing.lastFrame;
    if(!f.started||f.combatId!==d.combatId||f.combatantId!==d.combatantId||f.actorUuid!==d.actorUuid||f.tokenUuid!==d.tokenUuid||!f.order.some(c=>c.id===d.combatantId)){degrade('source-structure-changed');return}
    if(!same(f.order,d.order)){degrade('turn-order-changed');return}
    if(f.round<previousFrame.round||f.round===previousFrame.round&&f.turn<previousFrame.turn){degrade('turn-rewind');return}
    if(f.lastTurnEnd!==null&&f.lastTurnEnd>f.round){degrade('future-end-receipt');return}
    if(previousFrame.lastTurnEnd!==null&&(f.lastTurnEnd===null||f.lastTurnEnd<previousFrame.lastTurnEnd)){degrade('end-receipt-rewind');return}
    if(f.lastTurnEnd!==null&&f.lastTurnEnd>=d.endRound&&(d.baselineEnded===null||f.lastTurnEnd>d.baselineEnded)){end('caster-next-turn-ended');return}
    if(f.latestTurnEndRound!==null&&f.latestTurnEndRound>d.endRound){degrade('source-end-skipped');return}
    s.timing.lastFrame=f;
  }
  observe();
  if(s.status!=='ended'){
    switch(event.type){
      case 'continuity-unverified':
        degrade('turn-continuity-unverified');break;
      case 'target-start': {
        const p=projectRoaringConditions(s);
        if(!p.clap||p.manualReview){reject('source cannot prompt clap');break}
        const t=event.targetTurn,receipt={combatId:key(t?.combatId,'target combatId'),combatantId:key(t?.combatantId,'target combatantId'),round:integer(t?.round,'target round',1),actorUuid:text(t?.actorUuid,'target actorUuid'),tokenUuid:text(t?.tokenUuid,'target tokenUuid'),lastTurnStart:integer(t?.lastTurnStart,'target start receipt')};
        const f=s.timing.lastFrame;
        if(receipt.combatId!==s.timing.deadline.combatId||receipt.actorUuid!==s.source.targetActorUuid||receipt.tokenUuid!==s.source.targetTokenUuid||receipt.round!==f.round||receipt.lastTurnStart!==receipt.round||f.order[f.turn]?.id!==receipt.combatantId){reject('target start scope is not current');break}
        const receiptKey=`${receipt.combatId}/${receipt.combatantId}/${receipt.round}`;
        s.clapReceipts??={};
        if(own(s.clapReceipts,receiptKey)){decision=same(s.clapReceipts[receiptKey],receipt)?'duplicate':'rejected';if(decision==='rejected')reason='target start receipt already bound';break}
        s.clapReceipts[receiptKey]=receipt;decision='applied';command('clap-prompt',{receiptKey,targetTurn:receipt});break;
      }
      case 'save-confirmed': {
        const result={revision:integer(event.revision,'result revision',1),receiptId:text(event.receiptId,'save receiptId'),outcome:event.outcome};
        demand(OUTCOMES.includes(result.outcome),'invalid native outcome');
        if(s.result&&same(s.result,result)){decision='duplicate';break}
        if(s.result||s.manualReview||result.revision!==1||s.timing.mode!=='exact'){manual('save-revision-unverified');break}
        s.result=result;decision='applied';
        if(result.outcome==='criticalSuccess')end('save-no-effect');
        else {s.status='active';command('condition-sync',{conditions:projectRoaringConditions(s)})}
        break;
      }
      case 'save-unverified':
        text(event.receiptId,'unverified receiptId');text(event.reason,'unverified reason');manual(event.reason);break;
      case 'sustain-use': {
        const useNonce=key(event.useNonce,'useNonce'),f=frame(event.turn);
        const use={useNonce,userId:text(event.userId,'Use userId'),messageUuid:text(event.messageUuid,'Use messageUuid'),invocationId:key(event.invocationId,'invocationId'),turn:f,finiteEnvelope:finite(event.finiteEnvelope,worldTime,f),candidateDeadline:deadline(f),status:'use-recorded',verdict:null};
        if(own(s.sustainUses,useNonce)){
          const old=s.sustainUses[useNonce];decision=same({...old,status:'use-recorded',verdict:null},use)?'duplicate':'rejected';if(decision==='rejected')reason='Use nonce already bound';break;
        }
        if(s.status!=='active'||s.timing.mode!=='exact'){reject('source cannot renew');break}
        normalTurn(f,s.source);
        const current=s.timing.lastFrame;
        if(!same(f,current)){reject('Use turn no longer current');break}
        s.sustainUses[useNonce]=use;decision='applied';break;
      }
      case 'sustain-settled': {
        const useNonce=key(event.useNonce,'useNonce'),verdict={verdictId:text(event.verdictId,'verdictId'),gmId:text(event.gmId,'GM id'),terminal:event.terminal};
        demand(['completed','disrupted'].includes(verdict.terminal),'invalid Sustain terminal');
        if(!own(s.sustainUses,useNonce)){reject('unknown native Use');break}
        const use=s.sustainUses[useNonce];
        if(use.verdict){if(same(use.verdict,verdict))decision='duplicate';else manual('conflicting-sustain-verdict');break}
        if(s.status!=='active'||verdict.terminal==='completed'&&s.timing.mode!=='exact'){reject('source cannot renew');break}
        // The recorded Use was admitted in its own actual turn. Adjudication
        // before the source's deadline may be later; never anchor at that time.
        use.verdict=verdict;use.status=verdict.terminal;decision='applied';
        if(verdict.terminal==='disrupted'){end('sustain-disrupted');break}
        if(use.candidateDeadline.endRound>s.timing.deadline.endRound){
          s.timing.deadline=clone(use.candidateDeadline);s.timing.finiteEnvelope=clone(use.finiteEnvelope);
          command('renew-source',{deadline:s.timing.deadline,finiteEnvelope:s.timing.finiteEnvelope,hardStopAt:s.hardStopAt});
        }
        break;
      }
      case 'own-child-deleted': {
        demand(['slowed','fascinated'].includes(event.condition),'unknown owned condition');
        const tombstone={receiptId:text(event.receiptId,'child receiptId'),itemUuid:text(event.itemUuid,'child itemUuid'),reason:'confirmed-own-child-deletion'};
        if(s.tombstones[event.condition])decision='duplicate';else {s.tombstones[event.condition]=tombstone;decision='applied'}
        break;
      }
      case 'end-fascination': {
        const receiptId=text(event.receiptId,'fascination receiptId'),gmId=text(event.gmId,'GM id');
        if(s.tombstones.fascinated){decision='duplicate';break}
        s.tombstones.fascinated={receiptId,gmId,reason:'gm-fascination-ended'};decision='applied';command('end-fascination',{receiptId});break;
      }
      case 'own-parent-deleted':
        text(event.itemUuid,'parent itemUuid');text(event.receiptId,'parent receiptId');end('confirmed-own-parent-deletion');break;
    }
  }
  if(!same(source,s))s.revision=source.revision+1;
  return {source:s,commands,decision,reason};
}
