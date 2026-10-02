const __toolbeltManualPool=(()=>{
 const descriptor=Object.freeze({version:1,hpBaselineGuardVersion:1,providerId:'pf2e-toolbelt',providerVersion:'3.56.5',sourceSHA256:'2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f'});
 const observers=new Set(),seen=new Set();let installed=false;
 const copy=value=>JSON.parse(JSON.stringify(value));
 function notify(event){const answers=[];for(const observer of observers)try{const answer=observer(Object.freeze(event));if(answer!==undefined)answers.push(answer)}catch{}return answers}
 function fields(value){const keys=Object.keys(value);if(!keys.length||keys.some(key=>!['system.attributes.hp.value','system.attributes.hp.sp.value','system.attributes.hp.temp'].includes(key)||!Number.isFinite(value[key])||value[key]<0))throw Error('manual-pool-native-fields-mismatch');return Object.freeze(copy(value))}
 function valid(binding,master){return installed&&binding&&['permitNonce','applicationNonce','ownerUserId','patientUUID','poolUUID'].every(key=>typeof binding[key]==='string'&&binding[key])&&binding.poolUUID===master?.uuid}
 function hpSnapshot(value){
  if(value===null||['undefined','string','boolean'].includes(typeof value))return [typeof value,value];
  if(typeof value==='number'&&Number.isFinite(value))return ['number',value];
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error('manual-pool-native-hp-shape');
  return ['object',Object.keys(value).sort().map(key=>{const field=Object.getOwnPropertyDescriptor(value,key);if(!field||!('value' in field))throw Error('manual-pool-native-hp-shape');return [key,hpSnapshot(field.value)]})];
 }
 function unchangedHP(master,updates){
  try{
   const raw=master._source?.system?.attributes?.hp,hp=master.system?.attributes?.hp;
   if(!raw||!hp||typeof raw!=='object'||typeof hp!=='object')return null;
   for(const [path,value] of Object.entries(updates))for(const source of [raw,hp]){
    let current=source;
    for(const part of path.slice('system.attributes.hp.'.length).split('.')){const field=current&&Object.getOwnPropertyDescriptor(current,part);if(!field||!('value' in field))return null;current=field.value}
    if(current!==value)return null;
   }
   const prepared=Object.fromEntries(['value','max','temp','sp'].filter(key=>Object.prototype.hasOwnProperty.call(hp,key)).map(key=>[key,hp[key]]));
   return JSON.stringify([hpSnapshot(raw),hpSnapshot(prepared)]);
  }catch{return null}
 }
 function write(master,updates,binding,authorization,native){
  const exact=fields(updates),key=binding.applicationNonce;
  // Consume the private HP baseline synchronously at the original call. The
  // later validation still checks provenance, not the pre-write HP value.
  if(!valid(binding,master)||!master.isOwner||seen.has(key)||authorization.validate?.()!==true||authorization.beforeWrite?.()!==true)throw Error('manual-pool-native-authorization-changed');
  const hp=master.system?.attributes?.hp??{},before=copy(Object.fromEntries(['value','max','temp','sp'].filter(key=>hp[key]!==undefined).map(key=>[key,hp[key]])));
  const unchangedBefore=unchangedHP(master,exact);
  seen.add(key);const writer=game.user;
  const result=native();
  const terminalPromise=(async()=>{if(!result||typeof result.then!=='function')throw Error('manual-pool-native-promise-required');const saved=await result;
   const unchanged=saved===undefined&&unchangedBefore!==null&&unchangedHP(master,exact)===unchangedBefore;
   if((saved!==master&&!unchanged)||game.user!==writer||!master.isOwner||authorization.validate()!==true)throw Error('manual-pool-native-terminal-unavailable');
   return Object.freeze({binding:Object.freeze(copy(binding)),poolUUID:master.uuid,writerUserId:writer.id,fields:exact,before,terminal:'fulfilled',...(unchanged?{updateOutcome:'unchanged'}:{})});
  })();terminalPromise.catch(()=>{});notify({phase:'write',descriptor,binding:Object.freeze(copy(binding)),master,fields:exact,terminalPromise});return result;
 }
 function forward(patient,master,updates,options,local,remote){
  const answers=notify({phase:'prepare',descriptor,patient,master,fields:Object.freeze(copy(updates)),options});
  if(!answers.length){master.isOwner?local():remote({master,...updates});return null}
  const task=(async()=>{if(answers.length!==1)throw Error('manual-pool-native-observer-ambiguous');const authorization=await answers[0],binding=copy(authorization?.binding);
   if(!valid(binding,master)||binding.patientUUID!==patient.uuid||binding.ownerUserId!==game.user.id||authorization.validate?.()!==true)throw Error('manual-pool-native-authorization-changed');fields(updates);
   if(master.isOwner){write(master,updates,binding,authorization,local);return}
   remote({master,...updates,__explorationManualPool:binding});
  })();task.catch(()=>{});return task;
 }
 function receive(master,updates,binding,senderId,native){
  if(binding===undefined)return native();
  const task=(async()=>{
   if(!game.user.isActiveGM||!valid(binding,master)||binding.ownerUserId!==senderId)throw Error('manual-pool-native-sender-mismatch');fields(updates);
   const answers=notify({phase:'authorize',descriptor,master,fields:Object.freeze(copy(updates)),binding:Object.freeze(copy(binding)),senderId});
   if(answers.length!==1)throw Error('manual-pool-native-authorization-required');const authorization=await answers[0];
   if(!game.user.isActiveGM||authorization?.validate?.()!==true)throw Error('manual-pool-native-authorization-changed');write(master,updates,binding,authorization,native);
  })();task.catch(()=>{});return task;
 }
 Hooks.once('ready',()=>{
  const module=game.modules.get('pf2e-toolbelt');if(module?.version!=='3.56.5')return;const api=module.api??{};
  if('explorationManualPool' in api)return;
  const value=Object.freeze({descriptor,subscribe(observer){if(typeof observer!=='function')throw TypeError('manual-pool-observer-required');observers.add(observer);return()=>observers.delete(observer)}});
  Object.defineProperty(api,'explorationManualPool',{value,enumerable:true});if(module.api===undefined)module.api=api;installed=module.api?.explorationManualPool===value;
 });
 return {forward,receive};
})();
/* end toolbelt manual pool */
