import {sha256Fallback} from '../source-hash.mjs';

const hashes={
  attach:'33c3ae427c8bc94015e787bff10dc2a288436f6e27baec9dec708e249191f144',
  accumulator:'37b4b5b33974705d477aedbc8c84656813c92103391aff8551686d0d4a30a25d',
  boxStart:'a022226b8666a5de569bd756c3936933c137a23ab56b84b925491e01c4dcd993',
  engineStart:'91b92ea708def66da9c335f384df9bb01d398285fd6f6a7736155af56b600f21',
  engineComplete:'a1f4e7f93c8ea5928254b1f37e0baef5f0db9a38e6a5b060b65f6d1a46157fa7',
  engineEffects:'c4b4d67ee75da62d5f8ce473671ee61dcaa3e9864e62a3a7d857e80f7933e0dd',
  workerExec:'9f910bea44df0fe32a3ad4153b4525e556b5ace83ea99ac47124ee8165aa5e91',
  effectsConsumer:'b7d1c7ff49fbd90ed61837a4cd745aa757da7ec1c613f7b3cf7d1590baf3df1d',
  cleanupConsumer:'64fe5498b3b65f88d0f6cc22b10a083567297911d7f81c35a034c33da3d1bbe5'
};
const tickerConsumers={
  spawnPersistentDie:'2cf3df32d2d41e2e04aeecaf17ebf1101d220532b980115e18047affa9a84652',
  removePersistentDie:'30572115ec823800376c44a6f32b59fe7ae7ea85bb91e3db4ae1c8e65cabd5d1',
  fadeOutEphemeral:'624cbc8456b7ccaab6fde43a6acc8eb4c18367471acd033a5899abd71af2150a',
  fadeOutPersistentDie:'280fdd16bf37dd51748004fa9e03a9d7447019ed3edb2e02ad5eb925371cf4c6',
  clearAll:'c57ee0a73ea771033dfd5d7c0e468639598af90854ef7e804a65b74750d50d88',
  clearScene:'cf3b8fc698b2efaccad912d005f29ffeed3f63cefe9b1e917c3d7dbcc9d2dd8b'
};
// The captured PersistentDice adapter installs these together. Keep its actual
// function values so its own restore comparisons still work.
const bridgeHashes={
  spawnPersistentDie:'d5e1bead9c55dd00887bb905b2db0b0395ea785ea46b84c307656294d6d7492f',
  clearScene:'811288073cc835a47f1f9e8000d3c18a2078941da39b5c1dc77894a02b3f20be',
  handlePersistentThrowCompletion:'da8629905de5cce5f1c194baebc8f261cf3b7121b3909ce79698c1ba3ed5f162',
  exec:'f5d4ddf7ec2834991653d054dfb962f65498d57af099e9c52823e309244e3afd'
};
const profiles=[{
  queue:'7253acd493cfc6fd612934ece4c2b30d84f1e04d51827eae9cd945e7a4270f77',
  onEnd:'70de7b37350e4e244dd05a35672dec84e2ebf7e15e482b16d41970a1d620dc7a',
  boxClass:'b5b291d969cbf3c58ecaa2d0b508af9c68823462b95f0d727a646cee4e837935',
  boxAnimate:'d0f022f7cfe344f8bcc596d007a1ec7d259a926c940d25484958f0e74e43f2ce'
},{
  queue:'6d28b4d7e4a9e6c998c04630536e705b1696213cc9e5c941d5eac36599f2f865',
  onEnd:'6ce7da61fbe7c5b249e4fd80011dbf30d87437e903b6b680ae4c4dca3f753b9a',
  boxClass:'bdaa8ba1b22781ac9bf5a7adae39ab8bac12d96cd568fc8f7b6a902a3cfa3a4d',
  boxAnimate:'c722cddba3608fade37e2b526ece24ce64d32ca144ce69bd39bebc072f5f3d12'
}];
const sourceHashes=new WeakMap(),ownedCompletions=new WeakMap();
const digest=fn=>{
  if(typeof fn!=='function')return null;
  if(!sourceHashes.has(fn))sourceHashes.set(fn,sha256Fallback(Function.prototype.toString.call(fn)));
  return sourceHashes.get(fn);
};

// Legacy dispatch loses rejections; current dispatch catches them but leaves
// failed physics behind. Repair only batches from the audited Accumulator.
export function installDsnQueueRecovery({queue,recover,game=globalThis.game,ticker=globalThis.canvas?.app?.ticker}){
  const accumulator=queue?.nextAnimation,box=queue?.box,engine=box?.throwEngine;
  const attach=queue&&Object.getPrototypeOf(queue)?.attach;
  const onEnd=Object.getOwnPropertyDescriptor(accumulator??{},'_onEnd');
  const boxProto=box&&Object.getPrototypeOf(box),engineProto=engine&&Object.getPrototypeOf(engine);
  const start=boxProto?.startUnifiedBatch;
  const profile=profiles.find(value=>value.queue===digest(queue?.constructor));
  if(!profile||digest(accumulator?.constructor)!==hashes.accumulator
    ||digest(attach)!==hashes.attach||digest(onEnd?.value)!==profile.onEnd||!onEnd?.writable)return {status:'unsupported-queue'};
  if(!box)return {status:'waiting-dsn',attach};
  if(digest(start)!==hashes.boxStart||!Object.isExtensible(box)||Object.hasOwn(box,'startUnifiedBatch'))return {status:'unsupported-queue'};
  const animate=boxProto.animateThrow,animateHash=digest(animate);
  if(profiles.some(value=>value!==profile&&value.boxAnimate===animateHash))return {status:'unsupported-queue'};
  // Native attach runs before initialize finishes. Recheck this specific box.
  if(!engine&&typeof box.ready?.then==='function')return {status:'waiting-dsn',attach,ready:box.ready};
  const engineStart=engineProto?.startUnifiedBatch;
  if(digest(engineStart)!==hashes.engineStart||Object.hasOwn(engine??{},'startUnifiedBatch'))return {status:'unsupported-queue'};
  const completion=engineProto.handlePersistentThrowCompletion,effects=engineProto.handleSpecialEffectsInit;
  const consumers=Object.fromEntries(Object.keys(tickerConsumers).map(key=>[key,boxProto[key]]));
  const consumerKeys=Object.keys(consumers),consumerEntries=Object.entries(consumers);
  let current=null,playbackTail=null,active=true;
  const dataValue=(owner,key)=>Object.getOwnPropertyDescriptor(owner,key)?.value;
  const bridgeSupported=()=>game?.modules?.get('pf2e-dsn-persistent-bridge')?.active===true
    &&digest(dataValue(box,'spawnPersistentDie'))===bridgeHashes.spawnPersistentDie
    &&digest(dataValue(box,'clearScene'))===bridgeHashes.clearScene
    &&digest(dataValue(engine,'handlePersistentThrowCompletion'))===bridgeHashes.handlePersistentThrowCompletion;
  const initialWorker=box.physicsWorker,workerProto=initialWorker&&Object.getPrototypeOf(initialWorker);
  const initialExec=bridgeSupported()?(digest(workerProto?.exec)===hashes.workerExec?workerProto.exec:null):initialWorker?.exec;
  const releasedCompletion=fn=>{
    const owned=ownedCompletions.get(fn);
    return owned?.engine===engine&&owned.native===completion&&!owned.active();
  };
  const nativeCompletion=fn=>fn===completion||releasedCompletion(fn);
  const validConsumers=()=>consumerEntries.every(([key,fn])=>boxProto[key]===fn
    &&(box[key]===fn&&!Object.hasOwn(box,key)
      ||bridgeHashes[key]&&bridgeSupported()&&dataValue(box,key)===box[key]));
  const sourcesIntact=()=>Object.getPrototypeOf(box)===boxProto&&Object.getPrototypeOf(engine)===engineProto
    &&boxProto.startUnifiedBatch===start&&engineProto.startUnifiedBatch===engineStart
    &&engine.startUnifiedBatch===engineStart&&!Object.hasOwn(engine,'startUnifiedBatch')
    &&boxProto.animateThrow===animate&&engineProto.handlePersistentThrowCompletion===completion
    &&engineProto.handleSpecialEffectsInit===effects&&engine.handleSpecialEffectsInit===effects
    &&!Object.hasOwn(engine,'handleSpecialEffectsInit')&&validConsumers();
  const sourceProfile=()=>{
    if(!sourcesIntact())return null;
    if(bridgeSupported())return 'bridge';
    const value=engine.handlePersistentThrowCompletion;
    return value===completionWrapper||releasedCompletion(value)
      ||value===completion&&!Object.hasOwn(engine,'handlePersistentThrowCompletion')?'native':null;
  };
  const refreshTail=tail=>{
    const profile=sourceProfile();
    if(!profile)return false;
    const nextCompletion=engine.handlePersistentThrowCompletion,nextExec=tail.worker?.exec;
    const unchanged=consumerKeys.every(key=>tail.consumers[key]===box[key]);
    if(unchanged&&nextCompletion===tail.completion&&nextExec===tail.workerExec)return true;
    const next=Object.fromEntries(consumerKeys.map(key=>[key,box[key]]));
    // ready/dispose replace a complete bridge on the same physical owner.
    // A single unknown replacement never authorizes another worker executor.
    const ordinaryUnchanged=Object.entries(next).every(([key,fn])=>bridgeHashes[key]||tail.consumers[key]===fn);
    const bridgeInstalled=profile==='bridge'&&ordinaryUnchanged
      &&next.spawnPersistentDie!==tail.consumers.spawnPersistentDie&&next.clearScene!==tail.consumers.clearScene
      &&nextCompletion!==tail.completion&&digest(nextExec)===bridgeHashes.exec;
    const bridgeRemoved=tail.profile==='bridge'&&profile==='native'&&ordinaryUnchanged
      &&(nextExec===tail.workerExec||tail.nativeExec&&nextExec===tail.nativeExec);
    const restored=!active&&unchanged&&tail.completion===completionWrapper
      &&nativeCompletion(nextCompletion)&&nextExec===tail.workerExec;
    if(!bridgeInstalled&&!bridgeRemoved&&!restored)return false;
    tail.profile=profile;tail.consumers=next;tail.completion=nextCompletion;tail.workerExec=nextExec;
    if(profile==='native')tail.nativeExec=nextExec;
    return true;
  };
  const pendingTail=tail=>current===tail.scope&&tail.scope.tail===tail;
  const callbackIntact=tail=>{
    const owned=engine.callback===tail.once&&engine.throws===tail.throws;
    if(owned)tail.attached=true;
    return owned;
  };
  const attachedOwns=tail=>queue.box===box&&box.throwEngine===engine
    &&callbackIntact(tail)
    &&box.physicsWorker===tail.worker&&engine.physicsWorker===tail.worker
    &&(box.startUnifiedBatch===wrapper||!active&&box.startUnifiedBatch===start&&!Object.hasOwn(box,'startUnifiedBatch'))
    &&(box.animateThrow===tickerWrapper||!active&&box.animateThrow===animate)
    &&refreshTail(tail);
  const owns=tail=>pendingTail(tail)&&attachedOwns(tail);
  const startupOwns=tail=>pendingTail(tail)&&queue.box===box&&box.throwEngine===engine
    &&box.physicsWorker===tail.worker&&engine.physicsWorker===tail.worker
    &&(callbackIntact(tail)||!tail.attached&&engine.callback===null&&engine.throws===null)
    &&(box.startUnifiedBatch===wrapper||!active&&box.startUnifiedBatch===start&&!Object.hasOwn(box,'startUnifiedBatch'))
    &&(!completionSupported||box.animateThrow===tickerWrapper||!active&&box.animateThrow===animate)
    &&boxProto.startUnifiedBatch===start&&engine.startUnifiedBatch===engineStart
    &&engineProto.startUnifiedBatch===engineStart&&!Object.hasOwn(engine,'startUnifiedBatch')
    &&(completionSupported?refreshTail(tail):tail.worker?.exec===tail.workerExec);
  // A retained native ticker can run while start awaits spawn or simulation.
  // Its callback is still null until that same native startup acquires it.
  const frameOwns=tail=>tail.starting&&box._preparingThrow?startupOwns(tail):attachedOwns(tail);
  const audited=()=>active&&box.animateThrow===tickerWrapper&&sourceProfile()!==null;
  const failed=(tail,error)=>{
    recover(error);
    if(pendingTail(tail))tail.scope.failed=true;
  };
  const abandon=tail=>{
    failed(tail,tail.replaced);
    if(completionSupported)ticker.remove(tickerWrapper,box);
    tail.once(tail.throws);
  };
  // Intercept only the two native consumers. External await/catch retain the
  // original Promise. The bridge's outer Promise receives the same treatment.
  const intercept=(promise,tail,consumer,handle)=>{
    const then=promise.then;
    Object.defineProperty(promise,'then',{configurable:true,writable:true,value:function(onFulfilled,onRejected){
      if(this!==promise||onRejected!==undefined||digest(onFulfilled)!==consumer||!owns(tail))
        return then.call(this,onFulfilled,onRejected);
      return handle(then,onFulfilled);
    }});
    return promise;
  };
  const adapt=(promise,tail)=>intercept(promise,tail,hashes.effectsConsumer,(then,onEffects)=>{
    tail.completing=true;
    const pending=then.call(promise,()=>owns(tail)?onEffects():undefined);
    return intercept(pending,tail,hashes.cleanupConsumer,(then,onCleanup)=>then
      .call(pending,undefined,error=>failed(tail,error))
      .then(async()=>{
        try{
          if(owns(tail)){
            tail.cleaning=true;
            return await onCleanup();
          }
        }catch(error){failed(tail,error);}
        finally{
          tail.cleaning=false;
          if(pendingTail(tail)){
            if(owns(tail))engine.rolling=false;else tail.scope.failed=true;
            tail.once(tail.throws);
          }
        }
      }));
  });
  const completionWrapper=function(...args){
    const promise=completion.apply(this,args),tail=current?.tail;
    return this===engine&&tail&&audited()&&owns(tail)?adapt(promise,tail):promise;
  };
  const guard=tail=>{
    if(tail.cleaning&&!attachedOwns(tail)||tail.ticking&&!frameOwns(tail))throw tail.replaced;
  };
  function createView(tail){
    const engineView=new Proxy(engine,{
      get(target,key){
        guard(tail);
        if(key==='handlePersistentThrowCompletion')return (...args)=>{
          const promise=tail.completion.apply(engine,args);
          return owns(tail)?adapt(promise,tail):promise;
        };
        const value=Reflect.get(target,key,target);
        return typeof value==='function'?value.bind(target):value;
      },
      set(target,key,value){guard(tail);return Reflect.set(target,key,value,target);}
    });
    return new Proxy(box,{
      get(target,key){guard(tail);return key==='throwEngine'?engineView:Reflect.get(target,key,target);},
      set(target,key,value){guard(tail);return Reflect.set(target,key,value,target);}
    });
  }
  const tickerWrapper=function(...args){
    const tail=playbackTail;
    if(this!==box||!tail||!pendingTail(tail))return animate.apply(this,args);
    // A normal frame can arrive while stale cleanup is awaiting the worker.
    // Skip only that captured batch; unknown native business exceptions escape.
    if(!frameOwns(tail)){
      if(!tail.completing)abandon(tail);
      return;
    }
    tail.ticking=true;
    try{return animate.apply(tail.view,args);}
    catch(error){
      if(error!==tail.replaced)throw error;
      if(!tail.completing)abandon(tail);
    }finally{tail.ticking=false;}
  };
  const callback=async function(items){
    const prior=current,scope={items,failed:false};current=scope;
    const tracked=items.map(item=>({...item,resolve:value=>item.resolve(scope.failed?false:value)}));
    try{return await onEnd.value.call(this,tracked);}finally{current=prior;}
  };
  const wrapper=async function(...args){
    const scope=current;
    if(this!==box||!scope||queue.box!==box)return start.apply(this,args);
    const complete=args[2];let completed=false;
    const once=(...values)=>{
      if(completed)return;
      completed=true;
      if(scope.tail===tail)scope.tail=null;
      return complete(...values);
    };
    // The worker RPC function may be bridged. Capture its current identity for
    // this batch instead of pinning the installation's exec function.
    const profile=completionSupported?sourceProfile():null;
    const tail={scope,once,throws:args[0],worker:box.physicsWorker,workerExec:box.physicsWorker?.exec,profile,starting:true,
      nativeExec:profile==='native'?box.physicsWorker?.exec:box.physicsWorker===initialWorker?initialExec:null,
      completion:engine.handlePersistentThrowCompletion,
      consumers:Object.fromEntries(Object.keys(consumers).map(key=>[key,box[key]])),
      replaced:Error('DsN cleanup batch was replaced')};
    scope.tail=tail;
    if(box.throwEngine!==engine||boxProto.startUnifiedBatch!==start||engine.startUnifiedBatch!==engineStart
      ||engineProto.startUnifiedBatch!==engineStart||Object.hasOwn(engine,'startUnifiedBatch')
      ||completionSupported&&!audited()){
      abandon(tail);return;
    }
    if(completionSupported&&audited()){tail.view=createView(tail);playbackTail=tail;}
    try{
      const value=await start.call(box,args[0],args[1],once);
      if(completionSupported&&pendingTail(tail)&&!attachedOwns(tail)&&!tail.completing){
        abandon(tail);
      }
      return value;
    }
    catch(error){
      recover(error);
      if(completed)return;
      scope.failed=true;let ghostified;
      try{
        if(startupOwns(tail)){
          ghostified=engine._ghostifiedIds;
          if(ghostified?.length)await tail.workerExec.call(tail.worker,'setCollisionResponse',{ids:ghostified,enabled:true});
        }
      }catch(cleanupError){recover(cleanupError);}
      finally{
        if(startupOwns(tail)&&engine._ghostifiedIds===ghostified){
          box._preparingThrow=false;
          engine.rolling=false;engine.running=false;engine.callback=null;engine._ghostifiedIds=[];
        }
      }
      try{queue._settleDroppedBinds(scope.items);}catch(cleanupError){recover(cleanupError);}
      finally{once();}
    }finally{tail.starting=false;}
  };
  const completionSupported=digest(box.constructor)===profile.boxClass&&animateHash===profile.boxAnimate
    &&digest(completion)===hashes.engineComplete&&digest(effects)===hashes.engineEffects
    &&Object.entries(tickerConsumers).every(([key,hash])=>digest(consumers[key])===hash)
    &&(!bridgeSupported()||initialExec!==null)
    &&!Object.hasOwn(box,'animateThrow')&&sourcesIntact()
    &&(engine.handlePersistentThrowCompletion===completion&&!Object.hasOwn(engine,'handlePersistentThrowCompletion')
      ||releasedCompletion(dataValue(engine,'handlePersistentThrowCompletion'))||bridgeSupported())
    &&Object.isExtensible(engine)&&typeof ticker?.add==='function'&&typeof ticker?.remove==='function';
  const registered=fn=>{
    for(let node=ticker?._head?.next;node;node=node.next)if(node.fn===fn&&node.context===box)return true;
    return false;
  };
  if(completionSupported){
    if(nativeCompletion(engine.handlePersistentThrowCompletion)){
      ownedCompletions.set(completionWrapper,{engine,native:completion,active:()=>active});
      Object.defineProperty(engine,'handlePersistentThrowCompletion',{value:completionWrapper,writable:true,configurable:true});
    }
    const migrate=registered(animate);
    Object.defineProperty(box,'animateThrow',{value:tickerWrapper,writable:true,configurable:true});
    if(migrate){ticker.remove(animate,box);ticker.add(tickerWrapper,box);}
  }
  Object.defineProperty(accumulator,'_onEnd',{...onEnd,value:callback});
  Object.defineProperty(box,'startUnifiedBatch',{value:wrapper,writable:true,configurable:true});
  return {status:'installed',completionStatus:completionSupported?'installed':'unsupported-source',attach,restore(){
    if(!active)return;
    const migrate=completionSupported&&registered(tickerWrapper),ownAnimate=dataValue(box,'animateThrow')===tickerWrapper;
    active=false;
    if(dataValue(engine,'handlePersistentThrowCompletion')===completionWrapper)delete engine.handlePersistentThrowCompletion;
    if(ownAnimate)delete box.animateThrow;
    if(migrate){
      ticker.remove(tickerWrapper,box);
      if(ownAnimate&&box.animateThrow===animate&&queue.box===box&&box.throwEngine===engine&&sourcesIntact())ticker.add(animate,box);
    }
    if(accumulator._onEnd===callback)Object.defineProperty(accumulator,'_onEnd',onEnd);
    if(dataValue(box,'startUnifiedBatch')===wrapper)delete box.startUnifiedBatch;
  }};
}
