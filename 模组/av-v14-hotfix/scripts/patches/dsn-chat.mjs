import {sha256Fallback} from '../source-hash.mjs';
import {installDsnQueueRecovery} from './dsn-queue.mjs';
import {installDsnModelRecovery} from './dsn-model.mjs';
import {dsnCompatibility} from './dsn-runtime.mjs';

// Audited DsN 6.4.1/6.4.2/6.4.3 functions. Model errors reject and failed batches continue native
// chat reveal, preserving permissions and interactive pending throws.
const hashes={
  renderRolls:'8ed6ad569e58f7a64474f862a9a08a5e27492b2d8cedbe16b9d2ddce76a9caed',
  showForRoll:'a4284a3191192a06efafb1a56b44fb84803e63222cbe5544c404589b392f67b2',
  show:'59da9bc59811eefe5116e4a12a62ff5beca221e9a29286ea85e2b456a47d9fc8',
  _revealMessage:'390f9c83fe67f2fa1ea26661741c8bd5bf45342edd537392977befab903c1034'
};
const installations=new WeakMap(),waiting=new WeakSet();

export function installDsnChatRecovery({g=globalThis,report=()=>{}}={}){
  const finish=result=>{report({feature:'dsnChat',...result});return result;};
  const module=g.game?.modules?.get('dice-so-nice');
  if(!module?.active)return finish({status:'inactive'});
  const reason=dsnCompatibility(g);if(reason)return finish({status:reason});
  const pipeline=g.game.dice3d?.pipeline;
  if(!pipeline){
    if(!waiting.has(g.game)&&g.Hooks?.once){
      waiting.add(g.game);
      g.Hooks.once('diceSoNiceReady',()=>{waiting.delete(g.game);installDsnChatRecovery({g,report});});
    }
    return finish({status:'waiting-dsn'});
  }
  const prior=installations.get(pipeline);if(prior)return finish(prior);
  const proto=Object.getPrototypeOf(pipeline),native={};
  for(const [key,hash]of Object.entries(hashes)){
    const descriptor=Object.getOwnPropertyDescriptor(proto,key);
    if(typeof descriptor?.value!=='function'||Object.hasOwn(pipeline,key)
      ||sha256Fallback(Function.prototype.toString.call(descriptor.value))!==hash)return finish({status:'unsupported-source',detail:key});
    native[key]=descriptor.value;
  }
  if(!Object.isExtensible(pipeline))return finish({status:'unsupported-runtime'});
  const stats={recovered:0},result={status:'installed',stats};
  const recover=error=>{
    stats.recovered++;
    try{g.console?.error('av-v14-hotfix | DsN animation failed; continuing native chat reveal.',error);}catch{}
    try{finish(result);}catch{}
    return false;
  };
  let queueRecovery,readyHook,modelRecovery,modelReadyHook,generation=0,active=true;
  const attachModel=()=>{
    modelRecovery=installDsnModelRecovery({g});
    result.modelStatus=modelRecovery.status;
    result.modelPendingAtInstall=modelRecovery.pendingNativeLoads??0;
  };
  const queue=pipeline.queue,queueProto=queue&&Object.getPrototypeOf(queue),attach=queueProto?.attach;
  const ownsAttach=()=>Object.getOwnPropertyDescriptor(queue??{},'attach')?.value===attachWrapper;
  const attachWrapper=function(...args){
    const value=attach.apply(this,args);
    if(this===queue&&active){attachQueue();finish(result);}
    return value;
  };
  const attachQueue=completedReady=>{
    const attempt=++generation;
    queueRecovery?.restore?.();
    const unchanged=compatible()&&pipeline.queue===queue&&queueProto?.attach===attach
      &&(ownsAttach()||(!Object.hasOwn(queue??{},'attach')&&queue?.attach===attach));
    queueRecovery=unchanged?installDsnQueueRecovery({queue,recover,game:g.game,ticker:g.canvas?.app?.ticker}):{status:'unsupported-queue'};
    // A fulfilled ready promise without an engine is unsupported, not another
    // readiness wait on that same already-fulfilled promise.
    if(completedReady&&queueRecovery.ready===completedReady)queueRecovery={status:'unsupported-queue'};
    result.queueStatus=queueRecovery.status;
    result.queueCompletionStatus=queueRecovery.completionStatus??queueRecovery.status;
    if(queueRecovery.attach===attach&&typeof attach==='function'&&!Object.hasOwn(queue,'attach')&&Object.isExtensible(queue))
      Object.defineProperty(queue,'attach',{value:attachWrapper,writable:true,configurable:true});
    else if(queueRecovery.status==='unsupported-queue'&&ownsAttach())delete queue.attach;
    const {ready}=queueRecovery,box=queue?.box;
    if(ready){
      const current=()=>active&&generation===attempt&&queue.box===box&&box.ready===ready;
      Promise.resolve(ready).then(()=>{
        if(!current())return;
        attachQueue(ready);finish(result);
      },()=>{
        if(!current())return;
        generation++;queueRecovery={status:'unsupported-queue'};result.queueStatus=queueRecovery.status;result.queueCompletionStatus=queueRecovery.status;
        if(ownsAttach())delete queue.attach;
        finish(result);
      });
    }
    return queueRecovery;
  };
  const compatible=()=>g.game.modules.get('dice-so-nice')===module&&!dsnCompatibility(g)
    &&g.game.dice3d?.pipeline===pipeline
    &&Object.entries(native).every(([key,fn])=>proto[key]===fn&&(key==='renderRolls'?pipeline[key]===wrapper:pipeline[key]===fn));
  function wrapper(...args){
    if(this!==pipeline||!compatible())return native.renderRolls.apply(this,args);
    const safely=fn=>{
      try{return Promise.resolve(fn()).catch(recover);}catch(error){return Promise.resolve(recover(error));}
    };
    const queueViews=new WeakMap();
    const view=new Proxy(pipeline,{get(target,key){
      if(key==='showForRoll')return (...values)=>safely(()=>native.showForRoll.apply(view,values));
      if(key==='show')return (...values)=>safely(()=>native.show.apply(view,values));
      if(key==='queue'){
        const queue=target.queue;
        if(!queue||typeof queue!=='object')return queue;
        if(!queueViews.has(queue))queueViews.set(queue,new Proxy(queue,{get(owner,field){
          const value=Reflect.get(owner,field,owner);
          if(typeof value!=='function')return value;
          return field==='enqueue'||field==='deferHidden'
            ?(...values)=>safely(()=>value.apply(owner,values)):value.bind(owner);
        }}));
        return queueViews.get(queue);
      }
      const value=Reflect.get(target,key,target);
      return typeof value==='function'?value.bind(target):value;
    }});
    return native.renderRolls.apply(view,args);
  }
  Object.defineProperty(pipeline,'renderRolls',{value:wrapper,writable:true,configurable:true});
  attachModel();
  if(modelRecovery.status==='waiting-dsn'&&g.Hooks?.once){
    modelReadyHook=g.Hooks.once('diceSoNiceReady',()=>{
      if(active&&compatible()){attachModel();finish(result);}
    });
  }
  attachQueue();
  if(!queue?.box&&queueRecovery.status==='waiting-dsn'&&g.Hooks?.once){
    const attempt=generation;
    readyHook=g.Hooks.once('diceSoNiceReady',()=>{
      if(active&&generation===attempt){attachQueue();finish(result);}
    });
  }
  result.restore=()=>{
    if(!active)return;
    active=false;generation++;
    modelRecovery.restore?.();
    queueRecovery.restore?.();
    if(readyHook!==undefined)g.Hooks?.off?.('diceSoNiceReady',readyHook);
    if(modelReadyHook!==undefined)g.Hooks?.off?.('diceSoNiceReady',modelReadyHook);
    if(ownsAttach())delete queue.attach;
    if(Object.getOwnPropertyDescriptor(pipeline,'renderRolls')?.value===wrapper)delete pipeline.renderRolls;
    installations.delete(pipeline);
  };
  installations.set(pipeline,result);
  return finish(result);
}
