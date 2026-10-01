import {canonicalJSON} from './revision-codec.mjs';

const providerId='patreon-v3';
const baseSourceSHA256='89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9';
const pf2eSourceSHA256='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';
const copy=value=>JSON.parse(canonicalJSON(value));
const binding=checkpoint=>({id:checkpoint.id,sessionId:checkpoint.sessionId,from:checkpoint.from,to:checkpoint.to,gmId:checkpoint.gmId});
function same(a,b){try{return canonicalJSON(a)===canonicalJSON(b)}catch{return false}}
function sameFields(a,b){
 try{
  if(!same(a,b))return false;
  if(a===null||typeof a!=='object')return true;
  const keys=Reflect.ownKeys(a);
  return keys.length===Reflect.ownKeys(b).length&&keys.every(key=>Object.hasOwn(b,key)&&sameFields(a[key],b[key]));
 }catch{return false}
}
function supported(descriptor){
 return same(descriptor,{version:2,providerId,providerVersion:'3.2.29',baseSourceSHA256,pf2eSourceSHA256,markedCommitOwnership:'private-prepare.v1'});
}

export function createPatreonTimeCompletion({game,runtimeIdentity,getDriverScope,timeoutMs=10000}){
 let pending=null;
 const api=()=>game.modules.get(providerId)?.api?.explorationTimeCompletion;
 function hasScope(slot){
  const identity=runtimeIdentity(),scope=getDriverScope(slot.checkpoint.sessionId);
  return pending===slot&&api()===slot.api&&game.modules.get(providerId)?.active===true&&supported(slot.api.descriptor)
   &&game.user.id===slot.checkpoint.gmId&&game.users.activeGM?.id===slot.checkpoint.gmId
   &&same(identity,slot.identity)&&scope?.leaseNonce===slot.leaseNonce;
 }
 const owns=slot=>!slot.result&&hasScope(slot);
 function matchesOptions(slot,options){
  if(sameFields(options,slot.options))return true;
  const modifiedTime=options&&Object.getOwnPropertyDescriptor(options,'modifiedTime')?.value;
  // Foundry appends the Setting update options before updateWorldTime.
  return Number.isSafeInteger(modifiedTime)&&modifiedTime>=0&&modifiedTime<=8640000000000000
   &&sameFields(options,{...slot.options,action:'update',documentName:'Setting',modifiedTime,diff:true,recursive:true,render:true,parent:null});
 }
 function matchesInvocation(slot,invocation){
  return invocation&&typeof invocation.invocationId==='string'&&!!invocation.invocationId.trim()
   &&matchesOptions(slot,invocation.options)
   &&sameFields(invocation,{invocationId:invocation.invocationId,worldTime:slot.checkpoint.to,delta:slot.checkpoint.to-slot.checkpoint.from,
    userId:slot.checkpoint.gmId,handlerUserId:slot.checkpoint.gmId,activeGMId:slot.checkpoint.gmId,options:invocation.options});
 }
 function finish(slot,result){
  if(slot.result)return;
  clearTimeout(slot.timer);try{slot.dispose?.()}catch{result={status:'uncertain',reason:'patreon-dispose-failed'}}
  slot.result=result;slot.resolve(result);
 }
 const uncertain=(slot,reason)=>finish(slot,{status:'uncertain',reason});
 function authorize(slot,invocation){
  // This must run in the provider callback before either original handler starts.
  if(!owns(slot)){uncertain(slot,'patreon-private-scope-lost');return false}
  if(slot.invocation||!matchesInvocation(slot,invocation))return false;
  slot.invocation=copy(invocation);return true;
 }
 function observe(slot,event){
  if(pending!==slot||slot.result)return;
  if(!owns(slot)){uncertain(slot,'patreon-private-scope-lost');return}
  if(!slot.invocation||!same(event?.descriptor,slot.descriptor)||!sameFields(event?.invocation,slot.invocation))return;
  if(slot.observed){uncertain(slot,'patreon-duplicate-invocation');return}
  slot.observed=true;
  if(typeof event.terminalPromise?.then!=='function'){uncertain(slot,'patreon-terminal-unavailable');return}
  event.terminalPromise.then(proof=>{
   if(slot.result)return;
   if(!owns(slot)){uncertain(slot,'patreon-private-scope-lost');return}
   try{
    if(!sameFields(proof?.invocation,slot.invocation))throw Error('invalid-native-proof');
    const saved=copy(proof);
    if(!same(saved.descriptor,slot.descriptor)||!sameFields(saved.invocation,slot.invocation)
     ||!['members','rules','rolls','writes','errors'].every(key=>Array.isArray(saved[key]))||saved.errors.length
     ||saved.writes.some(write=>write.status!=='fulfilled'||!['actor-hp','effect-start','effect-decrease'].includes(write.kind)
      ||typeof write.documentUUID!=='string'||!write.documentUUID||!Object.hasOwn(write,'data')))throw Error('invalid-native-proof');
    finish(slot,{status:'ready',proof:[saved]});
   }catch{uncertain(slot,'patreon-completion-unproven')}
  },()=>uncertain(slot,'patreon-native-rejected'));
 }
 async function beforeAdvance(checkpoint){
  if(pending?.result?.status==='uncertain'&&!hasScope(pending)&&pending.checkpoint.id!==checkpoint.id)pending=null;
  if(pending)return {status:'blocked',reason:'patreon-checkpoint-pending'};
  const provider=api();
  // The prepared v4 seam exposes version 1 and cannot choose one active-GM tab.
  if(!supported(provider?.descriptor))return {status:'blocked',reason:'patreon-marked-ownership-unavailable'};
  const identity=runtimeIdentity(),scope=getDriverScope(checkpoint.sessionId);
  if(game.modules.get(providerId)?.active!==true||game.user.id!==checkpoint.gmId||game.users.activeGM?.id!==checkpoint.gmId
   ||identity?.userId!==checkpoint.gmId||!identity.clientNonce||!scope?.leaseNonce
   ||!checkpoint.id||!checkpoint.sessionId||!Number.isFinite(checkpoint.from)||!Number.isFinite(checkpoint.to)
   ||checkpoint.to<=checkpoint.from||game.time.worldTime!==checkpoint.from)return {status:'blocked',reason:'patreon-private-scope-unavailable'};
  const slot={checkpoint:copy(binding(checkpoint)),identity:copy(identity),leaseNonce:scope.leaseNonce,api:provider,descriptor:copy(provider.descriptor),
   options:{pf2eThirdPartyAutomation:{exploration:{sessionId:checkpoint.sessionId,checkpointId:checkpoint.id,expectedFrom:checkpoint.from,expectedTo:checkpoint.to,gmId:checkpoint.gmId}}}};
  slot.completion=new Promise(resolve=>slot.resolve=resolve);pending=slot;
  try{
   slot.dispose=provider.subscribe(event=>observe(slot,event),{authorizeMarkedCommit:invocation=>authorize(slot,invocation)});
   if(typeof slot.dispose!=='function')throw Error('missing-disposer');
   slot.timer=setTimeout(()=>uncertain(slot,'patreon-completion-timeout'),timeoutMs);
   return {status:'ready'};
  }catch{uncertain(slot,'patreon-subscription-unavailable');return {status:'blocked',reason:'patreon-subscription-unavailable'}}
 }
 async function settle(checkpoint){
  const slot=pending;
  if(!slot||!same(slot.checkpoint,binding(checkpoint)))return {status:'uncertain',reason:'patreon-checkpoint-unavailable'};
  const result=await slot.completion;
  if(result.status==='ready'&&!hasScope(slot))return {status:'uncertain',reason:'patreon-private-scope-lost'};
  if(result.status==='ready'&&pending===slot)pending=null;
  return result;
 }
 function cancel(checkpoint,reason){if(pending?.checkpoint.id===checkpoint.id){uncertain(pending,reason);pending=null}}
 function invalidate(reason){if(pending)uncertain(pending,reason);pending=null}
 return {matches:rule=>rule.providerId==='pf2e-patreon',ownershipAvailable:()=>game.modules.get(providerId)?.active===true&&supported(api()?.descriptor),beforeAdvance,settle,cancel,invalidate};
}
