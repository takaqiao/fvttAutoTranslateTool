import {MODULE_ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';
import {hasDisruptPrey,isCurrentDisruptToken,isDisruptPreyTarget,getDisruptPreyMeleeOptions} from './disrupt-prey-rules.mjs';
import {prepareDisruptMovementCheckpoints} from './disrupt-prey-movement.mjs';

const values=c=>Array.from(c?.values?.()??c??[]),doc=t=>t?.document??t;
const randomId=()=>globalThis.foundry?.utils?.randomID?.(24)??globalThis.crypto.randomUUID();
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const turn=game=>game.combat?.started?`${game.combat.id}:${game.combat.round}:${game.combat.turn}`:null;
const position=t=>({x:t._source?.x??t.x,y:t._source?.y??t.y,elevation:t._source?.elevation??t.elevation??0});
const samePosition=(a,b)=>['x','y','elevation'].every(k=>(a?.[k]??0)===(b?.[k]??0));
const active=actor=>actor?.canAct!==false&&!actor?.isDead&&!actor?.hasCondition?.('unconscious');
const eventFields=['eventId','nonce','actorUuid','tokenUuid','sourceActorUuid','sourceTokenUuid','sourceUserId','kind','phase'];
const sameEvent=(a,b)=>eventFields.every(k=>typeof a?.[k]==='string'&&a[k]===b?.[k]);
const physicalCard=m=>!!(m?.id&&!m.rolls?.length&&/<[^>]*class=["'][^"']*\baction-glyph\b[^"']*["'][^>]*>\s*1\s*</.test(m.flavor??'')&&/data-slug=["']move["']/.test(m.flavor??''));

// Core resumes checkpoint chains shortly before the previous animation ends.
// PF2e distanceTo reads those animated bounds, so finish that visual prefix
// before asking native reach/roll code to use the authoritative current square.
async function settleVisualMovement(token){
 const object=token?.object,promise=object?.movementAnimationPromise;
 if(!promise?.then)return;
 const page=globalThis.document;let onHidden;
 try{await new Promise((resolve,reject)=>{
  onHidden=()=>{if(page?.hidden)try{object.stopAnimation({reset:true});resolve();}catch(error){reject(error);}};
  page?.addEventListener?.('visibilitychange',onHidden);promise.then(resolve,reject);onHidden();
 });}finally{page?.removeEventListener?.('visibilitychange',onHidden);}
}

/** Await actual native sources. Socket payloads carry identities, never continuations. */
export function createDisruptPreyEvents({game,fromUuid=globalThis.fromUuid,handleConfirmed,onSourceStopped=async()=>{},onError=console.error}={}){
 const targets=new Map(),indexed=new Map(),tracked=new Map(),actorTokens=new Map(),scopes=new Map(),movementScopes=new Map(),gmRequests=new Map(),gmActive=new Map(),brands=new WeakMap();
 const restores=[],hookIds=[];let installed=false,socket,registeredHooks;
 const report=e=>{try{onError(e)}catch{/* Diagnostics cannot continue an uncertain action. */}};
 const sourceOwner=(actor,user=game.user)=>!!(user?.active&&actor?.testUserPermission?.(user,'OWNER'));
 const renderedScene=()=>game.scenes.current??globalThis.canvas?.scene;
 function removeToken(uuid){for(const targetUuid of indexed.get(uuid)??[]){const list=targets.get(targetUuid);list?.delete(uuid);if(!list?.size)targets.delete(targetUuid);}indexed.delete(uuid);const prior=tracked.get(uuid);if(prior){const tokens=actorTokens.get(prior.actorUuid);tokens?.delete(uuid);if(!tokens?.size)actorTokens.delete(prior.actorUuid);tracked.delete(uuid);}}
 function indexToken(token){
  removeToken(token.uuid);
  // Token.actor can lazily construct a synthetic NPC. Verify an already
  // rendered current-scene object before touching that getter.
  if(token.parent!==renderedScene()||token.object?.document!==token||!isCurrentDisruptToken(token,game))return;
  const actor=token.actor,tokens=actorTokens.get(actor.uuid)??new Map();tokens.set(token.uuid,token);actorTokens.set(actor.uuid,tokens);tracked.set(token.uuid,{actorUuid:actor.uuid});
  if(!hasDisruptPrey(actor))return;
  const marks=[];for(const [targetUuid,labels]of token.actor.synthetics?.tokenMarks??[]){
   if(!labels.includes('hunted-prey'))continue;
   const list=targets.get(targetUuid)??new Map();list.set(token.uuid,{actor:token.actor,token});targets.set(targetUuid,list);marks.push(targetUuid);
  }indexed.set(token.uuid,marks);
 }
 function rebuildIndex(){targets.clear();indexed.clear();tracked.clear();actorTokens.clear();for(const token of values(renderedScene()?.tokens))indexToken(token);}
 function refreshActor(actor){if(actor?.uuid)for(const token of values(actorTokens.get(actor.uuid)))indexToken(token);}
 const possible=token=>values(targets.get(token?.uuid)).filter(r=>isCurrentDisruptToken(r.token,game)&&r.token.parent===token.parent&&isDisruptPreyTarget(r.actor,token,game));
 const isPreyActor=actor=>values(actor?.getActiveTokens?.(true,true)).map(doc).some(token=>possible(token).length);
 function clearLine(token,target){
  // Native V14 checkCollision uses the document position, independent of an
  // in-flight animation. A move barrier blocks this ordinary melee reach.
  const point=target.getCenterPoint?.();return !!point&&typeof token.object?.checkCollision==='function'&&token.object.checkCollision(point,{type:'move',mode:'any'})===false;
 }
 const eligible=(reactor,target,choice=null)=>{
  const options=getDisruptPreyMeleeOptions({...reactor,target,game});return options.some(o=>!choice||o.key===(choice.key??choice.weaponKey))&&clearLine(reactor.token,target);
 };
 function sourceToken(actor){
  const controlled=values(game.user.getActiveTokens?.()).map(doc).filter(t=>t.actor?.uuid===actor.uuid&&isCurrentDisruptToken(t,game));
  const candidates=controlled.length?controlled:values(actor.getActiveTokens?.(true,true)).map(doc).filter(t=>isCurrentDisruptToken(t,game)&&(!game.scenes.current||t.parent===game.scenes.current));
  return candidates.length===1?candidates[0]:null;
 }
 function openScope(kind,actor,token,extra={}){
  if(!sourceOwner(actor)||!isCurrentDisruptToken(token,game)||token.actor!==actor)throw Error('扰乱狩猎来源操作者或Token已经失效。');
  const scope={id:randomId(),kind,actor,token,user:game.user,leader:game.users.activeGM?.id,turn:turn(game),events:new Map(),active:true,stage:'ready',...extra};scopes.set(scope.id,scope);return scope;
 }
 function closeScope(scope){scope.active=false;scopes.delete(scope.id);if(movementScopes.get(scope.token.uuid)===scope)movementScopes.delete(scope.token.uuid);}
 function currentScope(scope){
  return !!(installed&&scope?.active&&scopes.get(scope.id)===scope&&scope.user===game.user&&sourceOwner(scope.actor,scope.user)&&turn(game)===scope.turn&&(!scope.user.isGM||scope.user.id===game.users.activeGM?.id)&&isCurrentDisruptToken(scope.token,game)&&scope.token.actor===scope.actor);
 }
 function castProof(scope){
  const c=scope.cast,receipt=scope.actor.flags?.[MODULE_ID]?.nativeCasts?.find(r=>r.id===c?.castNonce);
  return !!(c?.token===scope.token&&c.tokenUuid===scope.token.uuid&&!c.unsupportedReason&&c.item?.actor===scope.actor&&c.item.type==='spell'&&values(c.item.traits??c.item.system?.traits?.value).includes('manipulate')&&receipt?.state==='paid'&&receipt.userId===scope.user.id&&receipt.nativeCastScope?.castNonce===c.castNonce&&receipt.nativeCastScope.tokenUuid===scope.token.uuid&&['actorUuid','itemUuid','entryUuid','rank','slotId','focusPoints','overlayIds','sourceId'].every(k=>JSON.stringify(receipt[k]??null)===JSON.stringify(c.input?.[k]??null)));
 }
 function scopeProof(scope,event){
  if(!currentScope(scope)||!sameEvent(scope.events.get(event.eventId),event)||!active(scope.actor))return null;
  if(scope.kind==='spell'){if(!castProof(scope))return null;return {scopeId:scope.id,castId:scope.cast.castNonce,input:scope.cast.input,position:position(scope.token),turn:scope.turn};}
  const message=game.messages.get(scope.messageId);
  if(!physicalCard(message)||author(message)!==scope.user.id||message.speaker?.actor!==scope.actor.id||message.speaker.scene!==scope.token.parent.id||message.speaker.token!==scope.token.id||message.flavor!==scope.flavor)return null;
  if(scope.kind==='stand'&&scope.actor.hasCondition?.('prone')||scope.kind==='drop-prone'&&!scope.actor.hasCondition?.('prone'))return null;
  if(scope.kind==='stride'&&(scope.stage!=='departure'||!scope.gate||!samePosition(scope.gate.origin,position(scope.token))))return null;
  return {scopeId:scope.id,messageId:message.id,flavor:scope.flavor,position:position(scope.token),turn:scope.turn,...(scope.kind==='stride'?{movementId:scope.gate.id,planId:scope.plan.id}:{})};
 }
 async function probeSource(event,scopeId){
  if(event.sourceUserId===game.user.id)return scopeProof(scopes.get(scopeId),event);
  if(!socket)return null;
  const response=await socket.executeAsUser('disrupt-prey:verify-source',event.sourceUserId,{event,scopeId});return response?.ok?response.proof:null;
 }
 async function validateSource(event,{actor,token,target,choice}={}){
  if(!isActiveGM(game))return false;
  const request=gmActive.get(event.eventId);if(!request||!sameEvent(request.event,event))return false;
  await Promise.all([settleVisualMovement(token),settleVisualMovement(target)]);
  const proof=await probeSource(event,request.scopeId);if(!isActiveGM(game)||!proof||proof.scopeId!==request.scopeId||proof.turn!==turn(game))return false;
  const user=game.users.get(event.sourceUserId);
  if(!sourceOwner(target?.actor,user)||!active(target.actor)||!isCurrentDisruptToken(target,game)||target.uuid!==event.sourceTokenUuid||target.actor.uuid!==event.sourceActorUuid||actor?.uuid!==event.actorUuid||token?.uuid!==event.tokenUuid||!eligible({actor,token},target,choice)||!samePosition(proof.position,position(target)))return false;
  if(event.kind==='spell'){
   const receipt=target.actor.flags?.[MODULE_ID]?.nativeCasts?.find(r=>r.id===proof.castId);
   if(receipt?.state!=='paid'||receipt.userId!==event.sourceUserId||receipt.nativeCastScope?.tokenUuid!==target.uuid||receipt.nativeCastScope.castNonce!==proof.castId||['actorUuid','itemUuid','entryUuid','rank','slotId','focusPoints','overlayIds','sourceId'].some(k=>JSON.stringify(receipt[k]??null)!==JSON.stringify(proof.input?.[k]??null)))return false;
  }else{
   const message=game.messages.get(proof.messageId);if(!physicalCard(message)||author(message)!==event.sourceUserId||message.speaker?.actor!==target.actor.id||message.speaker.scene!==target.parent.id||message.speaker.token!==target.id||message.flavor!==proof.flavor)return false;
  }
  return true;
 }
 async function receiveEvent(payload,user){
  const event=payload?.event;if(!isActiveGM(game)||!user?.active||event?.sourceUserId!==user.id||typeof payload.scopeId!=='string'||typeof handleConfirmed!=='function')throw Error('没有可等待的扰乱狩猎原生来源。');
  const existing=gmRequests.get(event.eventId);if(existing){if(existing.scopeId!==payload.scopeId||!sameEvent(existing.event,event))throw Error('扰乱狩猎重复来源不一致。');return existing.promise;}
  const record={event:{...event},scopeId:payload.scopeId};gmActive.set(event.eventId,record);
  record.promise=(async()=>{
   try{
    const actor=await fromUuid(event.actorUuid),token=await fromUuid(event.tokenUuid),target=await fromUuid(event.sourceTokenUuid);
    if(!await validateSource(event,{actor,token,target}))throw Error('扰乱狩猎原动作已不在等待，或来源/费用无法核实。');
    if(!isActiveGM(game))throw Error('扰乱狩猎主GM已改变。');
    const result=await handleConfirmed(event);if(!isActiveGM(game))throw Error('扰乱狩猎结算等待期间主GM已改变。');return result;
   }finally{gmActive.delete(event.eventId);gmRequests.delete(event.eventId);}
  })();gmRequests.set(event.eventId,record);return record.promise;
 }
 async function dispatch(scope){
  if(!currentScope(scope))throw Error('原生动作的操作者、回合或来源已改变，不能继续。');
  for(const reactor of possible(scope.token)){
   await settleVisualMovement(reactor.token);
   if(scope.events.has(reactor.token.uuid)||!eligible(reactor,scope.token))continue;
   const event={eventId:`${scope.id}:${reactor.token.uuid}`,nonce:randomId(),actorUuid:reactor.actor.uuid,tokenUuid:reactor.token.uuid,sourceActorUuid:scope.actor.uuid,sourceTokenUuid:scope.token.uuid,sourceUserId:scope.user.id,kind:scope.kind,phase:scope.kind==='spell'?'before-effects':scope.kind==='stride'?'before-departure':'after-action'};
   scope.events.set(reactor.token.uuid,event);scope.events.set(event.eventId,event);
   if(!scopeProof(scope,event))throw Error('原生动作的实际支付或来源无法确认。');
   const result=isActiveGM(game)?await receiveEvent({event,scopeId:scope.id},scope.user):await (async()=>{
    if(!socket||!game.users.activeGM)throw Error('扰乱狩猎需要在线主GM。');
    const response=await socket.executeAsUser('disrupt-prey:event',game.users.activeGM.id,{event,scopeId:scope.id});if(!response?.ok)throw Error(response?.error??'扰乱狩猎原生反应未确认。');return response.value;
   })();
   if(!['done','declined','ineligible'].includes(result?.status))throw Error('扰乱狩猎反应结果不确定，停止原动作。');
   if(!currentScope(scope))throw Error('等待反应期间原生动作的操作者或回合已改变。');
   if(result.disrupted||!active(scope.actor))return {...result,disrupted:true,reason:result.disrupted?'disrupt-prey':'source-incapacitated',eventId:event.eventId};
  }
  return undefined;
 }
 async function paidCast(context){
  if(!installed||!values(context.item?.traits??context.item?.system?.traits?.value).includes('manipulate')||!isPreyActor(context.actor))return;
  const scope=openScope('spell',context.actor,context.token,{cast:context});let terminalAttempted=false;
  try{const result=await dispatch(scope);if(result?.disrupted){terminalAttempted=true;await onSourceStopped(context,result);return result;}}
  catch(error){if(!terminalAttempted)await onSourceStopped(context,{status:'uncertain',disrupted:false,reason:error.message});throw error;}
  finally{closeScope(scope);}
 }
 function walkingSpeed(actor,kind){
  if(!active(actor))throw Error('这个角色当前不能进行步行移动。');
  // Player Core: while prone, Crawl and Stand are the only move actions.
  if(actor.hasCondition?.('prone'))throw Error('俯卧时不能疾行或跨步；请使用起立或爬行。');
  // PF2e 8.5 deletes prepared attributes.speed. Its evaluated land Statistic
  // (or prepared trace) includes modifiers; _source retains only the raw base.
  const speed=Number(actor.movement?.speeds?.land?.value??actor.system?.movement?.speeds?.land?.value);
  if(!Number.isFinite(speed)||speed<=0)throw Error('无法确认本次原生步行速度。');
  if(kind==='step'&&speed<10)throw Error('普通跨步要求陆地速度至少10尺。');
  return speed;
 }
 function stopOwnedPlan(scope){
  const token=scope.token,movement=token.movement;
  // stopMovement belongs to the initiating client even if actor OWNER or the
  // combat turn has changed. Never cancel a replacement or another user's plan.
  if(scope.plan?.id&&movement?.id===scope.plan.id&&movement.state==='planned'&&movement.user?.id===scope.user.id&&game.user===scope.user&&token.parent?.tokens?.get(token.id)===token)token.stopMovement();
 }
 async function actionUse(native,self,params,kind){
  const actors=params?.actors?(Array.isArray(params.actors)?params.actors:[params.actors]):values(game.user.getActiveTokens?.()).map(t=>doc(t)?.actor);
  if(!actors.length&&game.user.character)actors.push(game.user.character);
  const proneBefore=new Map(actors.filter(Boolean).map(actor=>[actor.uuid,!!actor.hasCondition?.('prone')]));
  // Reject invalid movement before its normal action card can be charged. A
  // caller requesting an unpublished draft does not start a movement plan.
  if(['stride','step'].includes(kind)&&params?.message?.create!==false)for(const actor of actors){const token=sourceToken(actor);if(token&&possible(token).length)walkingSpeed(actor,kind);}
  const results=await native.call(self,params);
  if(params?.message?.create===false)return results;
  for(const result of results??[]){
   const actor=result.actor,token=sourceToken(actor),message=result.message;
   if(!token||!possible(token).length||!physicalCard(message)||game.messages.get(message.id)!==message)continue;
   if(kind==='stand'&&proneBefore.get(actor.uuid)!==true||kind==='drop-prone'&&proneBefore.get(actor.uuid)!==false)continue;
   const scope=openScope(kind,actor,token,{messageId:message.id,flavor:message.flavor});
   try{
    if(kind==='stride'||kind==='step'){
     if(movementScopes.has(token.uuid))throw Error('这个Token已有尚未完成的原生移动动作。');
     movementScopes.set(token.uuid,scope);scope.stage='planning';
     const speed=walkingSpeed(actor,kind);
     // Ordinary Step is 5 feet, costs at most 5 feet, and uses land Speed only.
     // Native terrain measurement plus preventDrop rejects difficult terrain;
     // no inferred feat exception extends this ordinary action's distance.
     const plan=await token.object.planMovement({allowedActions:['walk'],maxCost:kind==='step'?5:speed,...(kind==='step'?{maxDistance:5,direct:true}:{}),preventDrop:true,moveOptions:{[MODULE_ID]:{disruptMovement:scope.id}}});
     if(!plan)continue;scope.plan=plan;
     if(!currentScope(scope)||!active(actor))throw Error('选路期间移动来源已经失效。');
     walkingSpeed(actor,kind);scope.stage='moving';
     // startMovement resolves when the plan starts, not when every checkpoint
     // finishes. The original plan's public finished promise follows its chain.
     const finished=token.movement.finished;
     if(!finished?.then)throw Error('无法取得这次原生移动的完整结束回执。');
     if(await token.startMovement(plan.id))await finished;
     if(scope.error)throw scope.error;
    }else await dispatch(scope);
   }finally{try{stopOwnedPlan(scope);}finally{closeScope(scope);}}
  }return results;
 }
 async function preUpdate(native,token,changes,operation,user){
  const scope=movementScopes.get(token.uuid),tag=operation?.[MODULE_ID]?.disruptMovement;
  if(scope?.kind==='stride'&&currentScope(scope)&&tag===scope.id){
   const move=operation.movement?.[token.id];
   if(move?.waypoints?.length){
    const path=token.getCompleteMovementPath([{...position(token),action:'walk'},...move.waypoints]).slice(1);
    // Preserve the operation itself: core writes its continuation promises and
    // result side-channel onto this exact object during the awaited call.
    operation.movement={...operation.movement,[token.id]:{...move,waypoints:prepareDisruptMovementCheckpoints({game,token,path,reactors:possible(token)})}};
   }
  }return native(changes,operation,user);
 }
 async function preMovement(native,token,movement,operation){
  const allowed=await native(movement,operation);if(allowed===false)return false;
  const scope=movementScopes.get(token.uuid);
  if(!scope||scope.kind!=='stride'||!scope.plan||movement.id!==scope.plan.id&&!movement.chain?.includes(scope.plan.id))return allowed;
  if(!movement.passed?.waypoints?.some(w=>!samePosition(w,movement.origin)))return allowed;
  // An exact Stride uses walk only. A relocation/teleport inserted into its
  // operation is not a new move-action trigger and must not inherit its scope.
  if(movement.passed.waypoints.some(w=>w.action!=='walk'))return allowed;
  scope.stage='departure';scope.gate=movement;
  try{
   await settleVisualMovement(token);
   if(!samePosition(position(token),movement.origin))throw Error('等待移动检查点期间来源位置已改变。');
   const result=await dispatch(scope);if(result?.disrupted)return false;return allowed;
  }
  catch(error){scope.error=error;report(error);return false;}
  finally{scope.gate=null;scope.stage='moving';}
 }
 function register({Hooks=globalThis.Hooks,libWrapper=globalThis.libWrapper,socket:socketApi,castEvents=getNativeCastEvents({game,fromUuid})}={}){
  if(installed)return unregister;installed=true;socket=socketApi;registeredHooks=Hooks;rebuildIndex();
  socket?.register('disrupt-prey:verify-source',function({event,scopeId}={}){if(this.socketdata?.userId!==game.users.activeGM?.id)return {ok:false};const proof=scopeProof(scopes.get(scopeId),event);return {ok:!!proof,proof};});
  socket?.register('disrupt-prey:event',async function(payload){try{return {ok:true,value:await receiveEvent(payload,game.users.get(this.socketdata?.userId))};}catch(error){return {ok:false,error:error.message};}});
  castEvents.addActorMatcher(actor=>installed&&isPreyActor(actor));castEvents.addPaidCastPolicy(paidCast);
  const prototypes=new Map();
  for(const kind of ['stand','drop-prone','stride','step']){
   const action=game.pf2e.actions.get(kind);if(!action)continue;
   const original=action.toActionVariant,variant=original.call(action),prototype=Object.getPrototypeOf(variant),set=prototypes.get(prototype)??new Set();set.add(kind);prototypes.set(prototype,set);
   for(const v of values(action.variants))brands.set(v,kind);
   const factory=function(...args){const v=original.apply(this,args);brands.set(v,kind);return v;};action.toActionVariant=factory;restores.push(()=>{if(action.toActionVariant===factory)action.toActionVariant=original;});
  }
  for(const [prototype,kinds]of prototypes){
   const descriptor=Object.getOwnPropertyDescriptor(prototype,'use'),native=prototype.use;
   const wrapper=function(params={}){const kind=brands.get(this);return installed&&kinds.has(kind)&&this.slug===kind&&this.cost===1&&this.traits?.includes('move')?actionUse(native,this,params,kind):native.call(this,params);};
   Object.defineProperty(prototype,'use',{configurable:true,writable:true,value:wrapper});restores.push(()=>{if(prototype.use===wrapper){if(descriptor)Object.defineProperty(prototype,'use',descriptor);else delete prototype.use;}});
  }
  for(const [path,fn]of [['CONFIG.Token.documentClass.prototype._preUpdate',function(native,changes,operation,user){return preUpdate(native,this,changes,operation,user)}],['CONFIG.Token.documentClass.prototype._preUpdateMovement',function(native,movement,operation){return preMovement(native,this,movement,operation)}]]){
   libWrapper?.register(MODULE_ID,path,fn,'MIXED');restores.push(()=>libWrapper?.unregister(MODULE_ID,path));
  }
  const on=(name,fn)=>hookIds.push([name,Hooks?.on(name,fn)]);
  for(const name of ['createItem','updateItem','deleteItem'])on(name,item=>refreshActor(item.actor));
  on('updateActor',refreshActor);on('createToken',indexToken);on('drawToken',token=>indexToken(token.document));on('updateToken',indexToken);on('deleteToken',token=>removeToken(token.uuid));on('canvasReady',rebuildIndex);on('deleteScene',rebuildIndex);
  return unregister;
 }
 function unregister(){installed=false;for(const scope of scopes.values())scope.active=false;scopes.clear();movementScopes.clear();for(const restore of restores.splice(0).reverse())restore();for(const [name,id]of hookIds.splice(0))registeredHooks?.off(name,id);targets.clear();indexed.clear();tracked.clear();actorTokens.clear();}
 return {register,validateSource,refreshActor,rebuildIndex};
}
