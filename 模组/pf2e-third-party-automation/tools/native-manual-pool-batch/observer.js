const __nativeManualPoolBatch=(()=>{
 const descriptor=Object.freeze({version:1,protocol:'pf2e-third-party-automation:manual-pool-batch:1',providerId:'pf2e',providerVersion:'8.5.1',baseSourceSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157',model:'numeric-empty-reception.v1'});
 const observers=new Map(),attempted=new Set(),frames=new WeakMap();let owned=false;
 const fail=reason=>{throw Error(`native-manual-batch-${reason}`)};
 function frozenJSON(value){
  function check(value){if(value===null||typeof value==='string'||typeof value==='boolean'||typeof value==='number'&&Number.isFinite(value))return;if(!value||typeof value!=='object'||!Array.isArray(value)&&value.constructor?.name!=='Object')fail('invalid-json');for(const field of Object.values(value))check(field)}
  check(value);const copy=JSON.parse(JSON.stringify(value));
  function freeze(value){if(value&&typeof value==='object'){for(const field of Object.values(value))freeze(field);Object.freeze(value)}return value}return freeze(copy);
 }
 const documentSource=document=>document===null||document===undefined?null:JSON.stringify(document.toObject(true));
 const paramsSource=params=>JSON.stringify({damage:params.damage,tokenUUID:params.token?.uuid,itemUUID:params.item?.uuid??null,skipIWR:params.skipIWR,rollOptions:[...params.rollOptions],outcome:params.outcome,shieldBlockRequest:params.shieldBlockRequest});
 const sourceKey=input=>JSON.stringify([input.message.uuid??input.message.id,input.rollIndex]);
 function publish(type,inv,extra={}){
  const event=Object.freeze({type,descriptor,batch:inv.view,...extra});for(const entry of [...observers.values()])try{entry.observer(event)}catch{}
 }
 function sourceView(input,batchId){
  return Object.freeze({batchId,message:input.message,roll:input.roll,rollIndex:input.rollIndex,multiplier:input.multiplier,addend:input.addend,item:input.item,
   targets:Object.freeze(input.targets.map((token,targetOrdinal)=>Object.freeze({targetOrdinal,token,patient:token.actor}))) });
 }
 function admit(input){
  const key=sourceKey(input);if(attempted.has(key))fail('source-already-attempted');
  const view=sourceView(input,crypto.randomUUID()),participants=[];
  for(const entry of [...observers.values()])if(entry.authorizeBatch){
   let answer;try{answer=entry.authorizeBatch(Object.freeze({phase:'admit',descriptor,batch:view}))}catch(error){attempted.add(key);throw error}
   if(answer?.then){Promise.resolve(answer).catch(()=>{});attempted.add(key);fail('synchronous-admission-required')}
   if(answer?.status==='participating'){attempted.add(key);participants.push({entry,answer})}else if(answer?.status!=='unregistered'){attempted.add(key);fail('source-admission-unavailable')}
  }
  if(participants.length===0)return null;
  attempted.add(key);if(participants.length!==1)fail('gate-ambiguous');
  const {entry,answer}=participants[0];if(typeof answer.isCurrent!=='function')fail('source-admission-unavailable');
  const binding=frozenJSON(answer.sourceBinding);
  const inv={input,view,entry,binding,isCurrent:answer.isCurrent,closed:false,userId:game.user.id,worldTime:game.time.worldTime,
   messageSource:documentSource(input.message),itemSource:documentSource(input.item),rollSource:JSON.stringify(input.roll.toJSON()),
   targets:input.targets.map(token=>({token,patient:token.actor,source:documentSource(token.actor)}))};
  current(inv);publish('batch-admitted',inv);return inv;
 }
 function current(inv){
  let valid=false;try{valid=!inv.closed&&owned&&observers.get(inv.entry.observer)===inv.entry&&game.user.id===inv.userId&&game.user.active===true&&game.time.worldTime===inv.worldTime&&inv.isCurrent()===true&&
   game.messages.get(inv.input.message.id)===inv.input.message&&inv.input.message.rolls.at(inv.input.rollIndex)===inv.input.roll&&inv.input.roll._evaluated===true&&
   inv.input.message.item===inv.input.item&&documentSource(inv.input.message)===inv.messageSource&&documentSource(inv.input.item)===inv.itemSource&&JSON.stringify(inv.input.roll.toJSON())===inv.rollSource;
  }catch{}if(!valid)fail('evidence-changed');
 }
 function reception(actor){
  const synthetics=actor?.synthetics;if(!synthetics||typeof synthetics!=='object')fail('reception-unavailable');
  for(const key of ['damageDice','modifiers','modifierAdjustments']){
   const map=synthetics[key];if(!map||typeof map!=='object')fail('reception-unavailable');
   const field=Object.getOwnPropertyDescriptor(map,'healing-received');
   if('healing-received'in map&&!field)fail('reception-unavailable');
   if(field&&(!Object.hasOwn(field,'value')||!Array.isArray(field.value)||field.value.length!==0))fail('reception-unavailable');
  }
 }
 function candidateCurrent(inv,candidate){
  const original=inv.targets[candidate.targetOrdinal];
  if(!original||original.token!==candidate.token||candidate.token.actor!==original.patient||candidate.patient!==original.patient||documentSource(candidate.patient)!==original.source||
   documentSource(candidate.contextualActor)!==candidate.actorSource||candidate.contextualActor.applyDamage!==candidate.applyDamage||candidate.params.rollOptions!==candidate.rollOptions||paramsSource(candidate.params)!==candidate.paramsSource)fail('evidence-changed');
  reception(candidate.contextualActor);
 }
 async function selection(inv){
  let timer;const work=Promise.resolve().then(()=>inv.entry.authorizeBatch(Object.freeze({phase:'select',descriptor,batch:inv.view})));work.catch(()=>{});
  try{return await Promise.race([work,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('native-manual-batch-selection-unknown')),inv.entry.timeoutMs)})])}finally{clearTimeout(timer)}
 }
 function select(inv,answer){
  if(answer?.status!=='selected'||answer.batchId!==inv.view.batchId||!/^[a-f0-9]{64}$/.test(answer.sourceDigest??'')||!Array.isArray(answer.selections)||answer.selections.length<1||answer.selections.length>8)fail('selection-unavailable');
  const claimed=new Set(),selected=new Map(),pools=new Set();
  for(const raw of answer.selections){
   const {grant,...selection}=raw,fields=frozenJSON(selection),grantSnapshot=frozenJSON(grant),entry={...fields,grant,grantSource:JSON.stringify(grantSnapshot)};
   if(typeof entry.poolUUID!=='string'||!entry.poolUUID||typeof entry.effectKey!=='string'||!entry.effectKey||pools.has(entry.poolUUID)||!Array.isArray(entry.patientUUIDs)||entry.patientUUIDs.length<1||
    new Set(entry.patientUUIDs).size!==entry.patientUUIDs.length||!Number.isSafeInteger(entry.selectedOrdinal)||typeof entry.grant?.permitNonce!=='string'||!entry.grant.permitNonce)fail('selection-unavailable');
   const members=inv.candidates.filter(candidate=>entry.patientUUIDs.includes(candidate.patient.uuid));
   if(members.length===0||entry.patientUUIDs.some(uuid=>!members.some(candidate=>candidate.patient.uuid===uuid)))fail('selection-mismatch');
   // This model has equal amounts; the original target order decides the tie.
   if(members[0].targetOrdinal!==entry.selectedOrdinal)fail('selection-mismatch');
   for(const member of members){if(claimed.has(member.targetOrdinal))fail('selection-mismatch');claimed.add(member.targetOrdinal);selected.set(member.targetOrdinal,{entry,apply:member.targetOrdinal===entry.selectedOrdinal})}
   pools.add(entry.poolUUID);
  }
  if(claimed.size!==inv.candidates.length)fail('selection-mismatch');return {selected,sourceDigest:answer.sourceDigest};
 }
 async function run(inv,prepare){
  try{
   current(inv);if(inv.input.multiplier>=0||typeof inv.input.damage!=='number'||!Number.isFinite(inv.input.damage)||Math.trunc(inv.input.damage)>=0||inv.targets.length<1||inv.targets.length>8)fail('model-unavailable');
   const candidates=await prepare();current(inv);
   if(!Array.isArray(candidates)||candidates.length<1||candidates.length!==inv.targets.length)fail('targets-unavailable');
   inv.candidates=candidates.map((candidate,targetOrdinal)=>{
    if(candidate.token!==inv.targets[targetOrdinal].token||candidate.patient!==inv.targets[targetOrdinal].patient||!candidate.contextualActor||candidate.params.damage!==inv.input.damage||candidate.params.token!==candidate.token||candidate.params.item!==inv.input.item||candidate.params.skipIWR!==true)fail('targets-unavailable');
    const captured={...candidate,targetOrdinal,actorSource:documentSource(candidate.contextualActor),applyDamage:candidate.contextualActor.applyDamage,rollOptions:candidate.params.rollOptions,paramsSource:paramsSource(candidate.params)};candidateCurrent(inv,captured);return captured;
   });
   inv.view=Object.freeze({...inv.view,sourceBinding:inv.binding,candidates:Object.freeze(inv.candidates.map(candidate=>Object.freeze({targetOrdinal:candidate.targetOrdinal,token:candidate.token,patient:candidate.patient,contextualActor:candidate.contextualActor,
    paramsSnapshot:frozenJSON({damage:candidate.params.damage,skipIWR:candidate.params.skipIWR,rollOptions:[...candidate.params.rollOptions],...(candidate.params.outcome===undefined?{}:{outcome:candidate.params.outcome}),shieldBlockRequest:candidate.params.shieldBlockRequest})}))) });
   publish('batch-prepared',inv);const answer=await selection(inv);current(inv);for(const candidate of inv.candidates)candidateCurrent(inv,candidate);
   const plan=select(inv,answer);
   for(const candidate of inv.candidates){
    current(inv);const choice=plan.selected.get(candidate.targetOrdinal);
    if(!choice.apply){publish('member-linked',inv,{targetOrdinal:candidate.targetOrdinal,poolUUID:choice.entry.poolUUID,effectKey:choice.entry.effectKey,permitNonce:choice.entry.grant.permitNonce});continue}
    candidateCurrent(inv,candidate);
    if(JSON.stringify(choice.entry.grant)!==choice.entry.grantSource)fail('evidence-changed');
    const callFrame=Object.freeze({descriptor,batchId:inv.view.batchId,callId:crypto.randomUUID(),targetOrdinal:candidate.targetOrdinal,message:inv.input.message,roll:inv.input.roll,rollIndex:inv.input.rollIndex,item:inv.input.item,
     token:candidate.token,patient:candidate.patient,contextualActor:candidate.contextualActor,multiplier:inv.input.multiplier,addend:inv.input.addend,stage:'healing',sourceBinding:inv.binding,
     sourceDigest:plan.sourceDigest,poolUUID:choice.entry.poolUUID,effectKey:choice.entry.effectKey,selectedPatientUUID:candidate.patient.uuid,grant:choice.entry.grant,paramsSnapshot:inv.view.candidates[candidate.targetOrdinal].paramsSnapshot,
     isCurrent(){try{current(inv);if(candidate.token.actor!==candidate.patient||candidate.contextualActor.applyDamage!==candidate.applyDamage||candidate.params.rollOptions!==candidate.rollOptions||paramsSource(candidate.params)!==candidate.paramsSource)return false;reception(candidate.contextualActor);return true}catch{return false}}});
    let promise;frames.set(candidate.params,{actor:candidate.contextualActor,callFrame});
    try{promise=candidate.contextualActor.applyDamage(candidate.params)}finally{frames.delete(candidate.params)}
    await promise;publish('native-call-returned',inv,{callFrame});
   }
   publish('batch-returned',inv);
  }catch(error){publish('batch-rejected',inv,{reason:String(error.message??error)});throw error}finally{inv.closed=true}
 }
 function install(){
  const api=game.pf2e;if(!api||typeof api!=='object'||'thirdPartyManualPoolBatch'in api)fail('api-collision');
  const observation=Object.freeze({descriptor,subscribe(observer,{authorizeBatch,timeoutMs=10000}={}){
   if(typeof observer!=='function'||authorizeBatch!==undefined&&typeof authorizeBatch!=='function'||!Number.isSafeInteger(timeoutMs)||timeoutMs<1||timeoutMs>10000)fail('invalid-subscription');
   let entry=observers.get(observer);if(entry&&(entry.authorizeBatch!==authorizeBatch||entry.timeoutMs!==timeoutMs))fail('subscription-conflict');
   if(!entry){entry={observer,authorizeBatch,timeoutMs};observers.set(observer,entry)}return()=>{if(observers.get(observer)===entry)observers.delete(observer)};
  },currentCall(actor,params){const frame=frames.get(params);return frame?.actor===actor?frame.callFrame:null}});
  Object.defineProperty(api,'thirdPartyManualPoolBatch',{value:observation,enumerable:true});owned=api.thirdPartyManualPoolBatch===observation;
 }
 return {admit,run,install};
})();
