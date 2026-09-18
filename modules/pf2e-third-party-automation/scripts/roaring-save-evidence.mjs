import {isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';

const ID='pf2e-third-party-automation', TOOL='pf2e-toolbelt';
const CLAIM='roaring-save:candidate', PROOF='roaring-save:proof';
const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const clone=value=>JSON.parse(JSON.stringify(value));
const canonical=value=>JSON.stringify(value,(_key,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
const equal=(a,b)=>canonical(a)===canonical(b);
const bounded=value=>typeof value==='string'&&/^[A-Za-z0-9-]{1,80}$/.test(value);
const author=message=>message?.author?.id??message?.user?.id??message?.user;
const publicMessage=m=>m?.blind===false&&Array.isArray(m.whisper)&&m.whisper.length===0;
const helper=m=>m?.flags?.[TOOL]?.targetHelper;
const saveRow=(m,s)=>helper(m)?.saveVariants?.null?.saves?.[s.targetUuid.split('.').at(-1)]??null;
const sourceKeys=['sourceNonce','castNonce','originalMessageUuid','casterActorUuid','casterTokenUuid','entryUuid','itemUuid','targetUuid','targetActorUuid','dc','rank','gmId'];
const identity=s=>Object.fromEntries(sourceKeys.map(k=>[k,s[k]]));
async function fingerprint(value){const bytes=await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(canonical(value)));return [...new Uint8Array(bytes)].map(n=>n.toString(16).padStart(2,'0')).join('');}

// Same narrowly confirmed DSN fields as the native Force witness. Do not drop
// Roll.options.type, arbitrary keys, mechanical results or role fields at other
// levels. Both original raw snapshots remain in the proof.
function mechanicalRoll(value,result=false){
 if(Array.isArray(value))return value.map(v=>mechanicalRoll(v,result));
 if(!value||typeof value!=='object')return value;
 const term=typeof value.class==='string'&&!value.class.endsWith('Roll')&&value.class!=='DamageInstance';
 return Object.fromEntries(Object.entries(value).filter(([k])=>!(result&&k==='indexThrow')).map(([k,v])=>{
  if(k==='options'&&term&&v&&typeof v==='object')return [k,mechanicalRoll(Object.fromEntries(Object.entries(v).filter(([name])=>name!=='type'&&!(value.class==='Die'&&['dsnRole','dsnRoleManaged'].includes(name)))))];
  return [k,mechanicalRoll(v,k==='results')];
 }));
}

// Toolbelt 3.56.2's persisted row schema adds these defaults. Preserve all
// mechanical fields; do not compare just total/outcome or a subset of dice.
function normalizeRow(data){
 const d=clone(data),r={...d,dosAdjustments:d.dosAdjustments??{},modifiers:d.modifiers??[],notes:d.notes??[],significantModifiers:d.significantModifiers??[],unadjustedOutcome:d.unadjustedOutcome??null};
 r.notes=r.notes.map(n=>Object.fromEntries(Object.entries({...n,outcome:n.outcome??[]}).filter(([k])=>['selector','title','text','predicate','outcome','visibility'].includes(k))));
 r.modifiers=r.modifiers.map(m=>({...m,excluded:m.excluded??false}));
 return r;
}

/** First-save evidence only. lookupSource is a synchronous, trusted provider
 * lookup of the live paid source/card pair, never a caller-supplied attestation.
 * Call track on every client while its original row is still empty. Native
 * hooks are captured synchronously; RPC and persistence can arrive either way.
 * Reloaded bare rows, all rerolls and ambiguous candidates remain manual.
 */
export function createRoaringSaveEvidence({game,fromUuid=globalThis.fromUuid,lookupSource,onVerified,onManual,onError=()=>{},randomId=()=>globalThis.crypto.randomUUID()}={}){
 const records=new Map(),local=new Map(),pending=new Set(),queue=new SerialActions();
 let socket,Hooks,installed=false,generation=0;const hooks=[];
 function compatible(){return game.system?.id==='pf2e'&&game.system.version==='8.5.1'&&game.modules?.get(TOOL)?.active===true&&game.modules.get(TOOL).version==='3.56.2';}
 function sourceFor(message){
  if(!compatible()||game.messages?.get(message?.id)!==message||!publicMessage(message))return null;
  const s=lookupSource(message);if(!s||typeof s.then==='function'||!['awaiting-save','active'].includes(s.status)||!bounded(s.sourceNonce)||!bounded(s.castNonce)||s.originalMessageUuid!==message.uuid||s.rank!==3||!Number.isFinite(s.dc)||s.gmId!==game.users.activeGM?.id||sourceKeys.some(k=>s[k]===undefined))return null;
  const h=helper(message),v=h?.saveVariants?.null,origin=message.flags?.pf2e?.origin;
  if(h?.type!=='spell'||h.private!==false||h.item!=null&&h.item!==s.itemUuid||!equal(h.targets,[s.targetUuid])||!v||v.statistic!=='will'||v.basic!==false||v.dc!==s.dc||Object.keys(h.saveVariants).length!==1||origin?.uuid!==s.itemUuid||origin.actor!==s.casterActorUuid||origin.castRank!==3||message.flags?.[ID]?.nativeCast?.id!==s.castNonce)return null;
  return s;
 }
 function current(record){const s=sourceFor(record.message);return installed&&s&&equal(identity(s),record.source)&&record.gmId===game.users.activeGM?.id?s:null;}
 function defer(fn){const task=Promise.resolve().then(fn).catch(onError);pending.add(task);task.finally(()=>pending.delete(task));return task;}
 async function manual(record,reason){
  if(record.status==='manual')return;record.status='manual';record.reason=reason;
  if(isActiveGM(game))await onManual?.({...record.source,reason,priorProof:record.proof?clone(record.proof):null});
 }
 function track(message){
  const s=sourceFor(message);if(!s)return false;
  if(records.has(message.uuid))return current(records.get(message.uuid))?true:false;
  if(records.size>=128)return false;
  const record={message,source:clone(identity(s)),gmId:s.gmId,status:'awaiting',candidates:new Map(),announced:new Set(),proof:null,observedRow:null};records.set(message.uuid,record);
  if(saveRow(message,s)!==null)defer(()=>manual(record,'unproven-existing-row'));
  return true;
 }
 function nativeSnapshot(record,event,reroll=false){
  const s=current(record),{roll,rollMessage:card,target,data,message}=event,user=game.user;
  if(!s||record.message!==message||!user?.active||game.users.get(user.id)!==user||target?.uuid!==s.targetUuid||target.actor?.uuid!==s.targetActorUuid||target.actor.testUserPermission?.(user,'OWNER')!==true||game.scenes?.get(target.parent?.id)?.tokens?.get(target.id)!==target)throw Error('unproven-roller-or-source');
  if(reroll)return {kind:'reroll',data:clone(data),target,userId:user.id};
  const Roll=globalThis.CONFIG?.Dice?.rolls?.find(C=>C.name==='CheckRoll'),Message=globalThis.CONFIG?.ChatMessage?.documentClass,pf=card?.flags?.pf2e?.context;
  if(!Roll||!Message||!(roll instanceof Roll)||!(card instanceof Message)||!roll._evaluated||card.isCheckRoll!==true||!Number.isFinite(roll.total)||!publicMessage(card)||game.pf2e?.settings?.metagame?.results!==true||data?.private!==false||data.rerolled||pf?.isReroll||roll.options?.isReroll)throw Error('unproven-native-or-private');
  const degree=roll.options.degreeOfSuccess;
  if(!Number.isInteger(degree)||degree<0||degree>3||data.success!==outcomes[degree]||pf.outcome!==data.success||data.statistic!=='will'||pf.type!=='saving-throw'||roll.options.type!==pf.type||pf.dc?.value!==s.dc||pf.messageMode!=='public'||pf.options?.includes('secret')||pf.traits?.some(t=>(typeof t==='string'?t:t?.name)==='secret'))throw Error('unproven-native-result');
  if(author(card)!==user.id||roll.options.rollerId!==user.id||pf.actor!==target.actor.id||pf.token!==target.id||card.speaker?.actor!==target.actor.id||card.speaker.token!==target.id||card.speaker.scene!==target.parent.id||pf.origin?.actor!==s.casterActorUuid||pf.origin.token!==s.casterTokenUuid||pf.target?.actor!==s.targetActorUuid||pf.target.token!==s.targetUuid||card.flags.pf2e.origin?.uuid!==s.itemUuid||card.flags.pf2e.origin.actor!==s.casterActorUuid)throw Error('unproven-native-context');
  const json=clone(roll.toJSON());
  if(!Array.isArray(card.rolls)||card.rolls.length!==1||!(card.rolls[0] instanceof Roll)||!equal(mechanicalRoll(card.rolls[0].toJSON()),mechanicalRoll(json))||!equal(mechanicalRoll(JSON.parse(data.roll)),mechanicalRoll(json))||data.value!==roll.total||data.die!==roll.terms?.[0]?.total||data.unadjustedOutcome!==(pf.unadjustedOutcome??null))throw Error('unproven-native-roll');
  return {kind:'first',row:normalizeRow(data),rollJSON:json,draftContext:clone(pf),target,userId:user.id};
 }
 function capture(event,reroll=false){
  const record=records.get(event.message?.uuid);if(!record||!current(record))return;
  try{
   const evidence=nativeSnapshot(record,event,reroll),invocationId=randomId();if(!bounded(invocationId))throw Error('invalid-invocation');
   // Store the actual local documents and synchronous snapshots before any
   // hashing/RPC await. A caller cannot revive this proof from serialized data.
   const scope={record,event,evidence,invocationId,generation,gmId:record.gmId};local.set(invocationId,scope);
   defer(async()=>{
    const payload={originalMessageUuid:record.message.uuid,sourceNonce:record.source.sourceNonce,invocationId};
    const response=isActiveGM(game)?await accept(payload,game.user.id):await socket.executeAsUser(CLAIM,scope.gmId,payload);
    if(response?.ok===false)throw Error(response.error);
   });
  }catch(error){defer(()=>manual(record,error.message));}
 }
 async function prove(payload,sender){
  if(!installed||sender!==game.users.activeGM?.id)throw Error('proof-active-GM-required');
  const s=local.get(payload.invocationId);
  if(!s||s.generation!==generation||s.gmId!==sender||s.record.message.uuid!==payload.originalMessageUuid||s.record.source.sourceNonce!==payload.sourceNonce||!current(s.record))throw Error('proof-scope-unavailable');
  const ambiguous=s.evidence.kind==='first'&&[...local.values()].filter(v=>v.record===s.record&&v.evidence.kind==='first').length!==1;
  const now=nativeSnapshot(s.record,s.event,s.evidence.kind==='reroll');
  if(now.userId!==s.evidence.userId||now.kind!==s.evidence.kind||now.kind==='first'&&(!equal(now.row,s.evidence.row)||!equal(mechanicalRoll(now.rollJSON),mechanicalRoll(s.evidence.rollJSON))))throw Error('proof-native-evidence-changed');
  const proof={schema:1,invocationId:s.invocationId,rollerUserId:now.userId,gmId:sender,source:clone(s.record.source),kind:ambiguous?'ambiguous':now.kind};
  if(now.kind==='first')Object.assign(proof,{beforeRow:null,row:clone(s.evidence.row),rollJSON:clone(s.evidence.rollJSON),rowFingerprint:await fingerprint(s.evidence.row),rollFingerprint:await fingerprint(s.evidence.rollJSON),dc:s.record.source.dc});
  if(!current(s.record)||game.users.activeGM?.id!==sender)throw Error('proof-GM-changed');
  return proof;
 }
 async function settle(record){
  if(!isActiveGM(game)||!current(record)||['manual','delivering'].includes(record.status))return;
  const row=saveRow(record.message,record.source);
  if(record.status==='verified'){
   if(!row||!equal(normalizeRow(row),record.proof.row))await manual(record,'unverified-result-revision');return;
  }
  if(!row)return;
  if(row.private!==false){await manual(record,'private-row-requires-manual-review');return;}
  if(row.rerolled){await manual(record,'reroll-requires-manual-review');return;}
  if(record.candidates.size!==1){if(record.candidates.size>1||record.announced.size>1)await manual(record,'ambiguous-native-candidates');return;}
  const proof=[...record.candidates.values()][0];
  if(!equal(normalizeRow(row),proof.row)){await manual(record,'persisted-row-does-not-match-native');return;}
  const target=await fromUuid(record.source.targetUuid),user=game.users.get(proof.rollerUserId);
  if(!current(record)||!isActiveGM(game)||!user?.active||target?.actor?.uuid!==record.source.targetActorUuid||target.actor.testUserPermission?.(user,'OWNER')!==true||!equal(normalizeRow(saveRow(record.message,record.source)),proof.row))return;
  record.status='delivering';record.proof=clone(proof);
  try{await onVerified({...record.source,adjustedOutcome:proof.row.success,revision:1,proof:clone(proof)});if(record.status==='delivering')record.status='verified';}
  catch(error){await manual(record,'verified-delivery-uncertain');throw error;}
 }
 async function accept(payload,sender){
  if(!isActiveGM(game)||!bounded(payload?.invocationId))throw Error('candidate-active-GM-required');
  const record=records.get(payload.originalMessageUuid),user=game.users.get(sender);
  if(!record||!current(record)||record.source.sourceNonce!==payload.sourceNonce||!user?.active)throw Error('candidate-scope-unavailable');
  const target=await fromUuid(record.source.targetUuid);
  if(target?.actor?.uuid!==record.source.targetActorUuid||target.actor.testUserPermission?.(user,'OWNER')!==true)throw Error('candidate-not-owner');
  const result=sender===game.user.id?{ok:true,value:await prove(payload,game.user.id)}:await socket.executeAsUser(PROOF,sender,payload);
  if(!result?.ok)throw Error(result?.error??'candidate-proof-unavailable');
  const proof=result.value;
  if(!isActiveGM(game)||!current(record)||proof?.rollerUserId!==sender||proof.gmId!==game.user.id||proof.invocationId!==payload.invocationId||!equal(proof.source,record.source))throw Error('candidate-proof-mismatch');
  // A failed/spoofed request must not mutate an authentic source, including its
  // ambiguity set. Only this authenticated native scope may announce a result.
  if(proof.kind==='first'&&(proof.beforeRow!==null||proof.rowFingerprint!==await fingerprint(proof.row)||proof.rollFingerprint!==await fingerprint(proof.rollJSON)))throw Error('candidate-fingerprint-mismatch');
  if(!['first','reroll','ambiguous'].includes(proof.kind))throw Error('candidate-kind-unknown');
  record.announced.add(`${sender}:${payload.invocationId}`);
  return queue.run(record.message.uuid,async()=>{
   if(!isActiveGM(game)||!current(record))throw Error('candidate-GM-changed');
   if(proof.kind==='reroll'){await manual(record,'reroll-requires-manual-review');return {ok:true};}
   if(proof.kind==='ambiguous'){await manual(record,'ambiguous-native-candidates');return {ok:true};}
   if(proof.kind!=='first'||proof.beforeRow!==null||proof.rowFingerprint!==await fingerprint(proof.row)||proof.rollFingerprint!==await fingerprint(proof.rollJSON))throw Error('candidate-fingerprint-mismatch');
   if(record.candidates.has(proof.invocationId))return {ok:true};
   record.candidates.set(proof.invocationId,clone(proof));
   if(record.candidates.size>1||record.announced.size>1)await manual(record,'ambiguous-native-candidates');
   else await settle(record);
   return {ok:true};
  });
 }
 function observe(message){
  const record=records.get(message.uuid);if(!record)return;
  // Capture every actual row transition synchronously, before later updates can
  // overwrite the live Document. A changed-then-restored row is not a first save.
  const row=saveRow(message,record.source),encoded=row===null?null:canonical(normalizeRow(row));
  if(record.observedRow!==null&&record.observedRow!==encoded){const task=manual(record,'unverified-result-revision');defer(()=>task);return;}
  record.observedRow=encoded;
  if(!current(record)){const task=manual(record,'source-or-card-changed');defer(()=>task);return;}
  defer(()=>queue.run(message.uuid,()=>settle(record)));
 }
 function register({Hooks:api,socket:rpc}){
  if(installed)return cleanup;installed=true;generation++;Hooks=api;socket=rpc;
  const on=(name,fn)=>hooks.push([name,Hooks.on(name,fn)]);
  on('pf2e-toolbelt.rollSave',event=>capture(event));on('pf2e-toolbelt.rerollSave',event=>capture(event,true));on('updateChatMessage',observe);
  socket.register(PROOF,async function(payload){try{return {ok:true,value:await prove(payload,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}});
  socket.register(CLAIM,async function(payload){try{return await accept(payload,this.socketdata?.userId)}catch(error){return {ok:false,error:error.message}}});
  return cleanup;
 }
 function cleanup(){for(const[name,id]of hooks.splice(0))Hooks.off(name,id);installed=false;generation++;local.clear();records.clear();}
 return {register,track,cleanup,inspect:uuid=>{const r=records.get(uuid);return r?{status:r.status,reason:r.reason??null,candidates:r.candidates.size,proof:r.proof?clone(r.proof):null}:null;},async whenIdle(){while(pending.size)await Promise.all([...pending]);}};
}
