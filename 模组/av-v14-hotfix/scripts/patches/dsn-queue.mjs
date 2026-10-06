import {sha256Fallback} from '../source-hash.mjs';

const hashes={
  attach:'33c3ae427c8bc94015e787bff10dc2a288436f6e27baec9dec708e249191f144',
  accumulator:'37b4b5b33974705d477aedbc8c84656813c92103391aff8551686d0d4a30a25d',
  boxStart:'a022226b8666a5de569bd756c3936933c137a23ab56b84b925491e01c4dcd993',
  engineStart:'91b92ea708def66da9c335f384df9bb01d398285fd6f6a7736155af56b600f21',
  engineComplete:'a1f4e7f93c8ea5928254b1f37e0baef5f0db9a38e6a5b060b65f6d1a46157fa7',
  engineEffects:'c4b4d67ee75da62d5f8ce473671ee61dcaa3e9864e62a3a7d857e80f7933e0dd',
  effectsConsumer:'b7d1c7ff49fbd90ed61837a4cd745aa757da7ec1c613f7b3cf7d1590baf3df1d',
  cleanupConsumer:'64fe5498b3b65f88d0f6cc22b10a083567297911d7f81c35a034c33da3d1bbe5'
};
const profiles=[{
  queue:'7253acd493cfc6fd612934ece4c2b30d84f1e04d51827eae9cd945e7a4270f77',
  onEnd:'70de7b37350e4e244dd05a35672dec84e2ebf7e15e482b16d41970a1d620dc7a',
  boxAnimate:'d0f022f7cfe344f8bcc596d007a1ec7d259a926c940d25484958f0e74e43f2ce'
},{
  queue:'6d28b4d7e4a9e6c998c04630536e705b1696213cc9e5c941d5eac36599f2f865',
  onEnd:'6ce7da61fbe7c5b249e4fd80011dbf30d87437e903b6b680ae4c4dca3f753b9a',
  boxAnimate:'c722cddba3608fade37e2b526ece24ce64d32ca144ce69bd39bebc072f5f3d12'
}];
const digest=fn=>typeof fn==='function'?sha256Fallback(Function.prototype.toString.call(fn)):null;

// Legacy dispatch loses rejections; current dispatch catches them but leaves
// failed physics behind. Repair only batches from the audited Accumulator.
export function installDsnQueueRecovery({queue,recover}){
  const accumulator=queue?.nextAnimation,box=queue?.box,engine=box?.throwEngine;
  const attach=queue&&Object.getPrototypeOf(queue)?.attach;
  const onEnd=Object.getOwnPropertyDescriptor(accumulator??{},'_onEnd');
  const boxProto=box&&Object.getPrototypeOf(box),engineProto=engine&&Object.getPrototypeOf(engine);
  const start=boxProto?.startUnifiedBatch;
  const worker=box?.physicsWorker;
  const profile=profiles.find(value=>value.queue===digest(queue?.constructor));
  if(!profile||digest(accumulator?.constructor)!==hashes.accumulator
    ||digest(attach)!==hashes.attach||digest(onEnd?.value)!==profile.onEnd||!onEnd?.writable)return {status:'unsupported-queue'};
  if(!box)return {status:'waiting-dsn',attach};
  if(digest(start)!==hashes.boxStart||!Object.isExtensible(box)||Object.hasOwn(box,'startUnifiedBatch'))return {status:'unsupported-queue'};
  const animate=boxProto.animateThrow,animateHash=digest(animate);
  if(profiles.some(value=>value!==profile&&value.boxAnimate===animateHash))return {status:'unsupported-queue'};
  // Native attach runs before initialize finishes. The texture-ready hook does
  // not await the box; recheck all sources after this specific box is ready.
  if(!engine&&typeof box.ready?.then==='function')return {status:'waiting-dsn',attach,ready:box.ready};
  if(digest(engineProto?.startUnifiedBatch)!==hashes.engineStart||Object.hasOwn(engine??{},'startUnifiedBatch'))return {status:'unsupported-queue'};
  // Replacing _onEnd affects the next Accumulator dispatch. The current one
  // retains its original closure. In particular, queue.idle() can resolve just
  // before Accumulator's finally clears _isProcessing during a native resize.
  let current=null,active=true;
  const callback=async function(items){
    const prior=current,scope={items,failed:false};current=scope;
    const tracked=items.map(item=>({...item,resolve:value=>item.resolve(scope.failed?false:value)}));
    try{return await onEnd.value.call(this,tracked);}finally{current=prior;}
  };
  const wrapper=async function(...args){
    const scope=current;
    if(this!==box||!scope||queue.box!==box||box.throwEngine!==engine
      ||boxProto.startUnifiedBatch!==start||engine.startUnifiedBatch!==engineProto.startUnifiedBatch
      ||Object.hasOwn(engine,'startUnifiedBatch')||digest(engineProto.startUnifiedBatch)!==hashes.engineStart)
      return start.apply(this,args);
    const complete = args[2];
    let completed = false;
    const once = (...values) => {
      if (completed) return;
      completed = true;
      if (scope.tail === tail) scope.tail = null;
      return complete(...values);
    };
    const tail = {scope, once, throws:args[0]};
    scope.tail = tail;
    // Keep the native ticker function and bundle closures. Its context carries
    // this batch's ownership check through each await in the native cleanup.
    const view=completionSupported?new Proxy(box,{get(target,key){
      if(tail.cleaning&&!attachedOwns(tail))throw new Error('DsN cleanup batch was replaced');
      return Reflect.get(target,key,target);
    }}):box;
    try{return await start.call(view,args[0],args[1],once);}
    catch(error){
      recover(error);
      // A callback already fired may have started the next batch. Do not reset
      // that later batch's physics or settle this one a second time.
      if(completed)return;
      scope.failed=true;
      try{
        if(pendingTail(tail)&&queue.box===box&&box.throwEngine===engine
          &&(engine.callback===null||engine.callback===once)
          &&(engine.throws===null||engine.throws===tail.throws)
          &&boxProto.startUnifiedBatch===start&&engine.startUnifiedBatch===engineProto.startUnifiedBatch
          &&!Object.hasOwn(engine,'startUnifiedBatch')&&digest(engineProto.startUnifiedBatch)===hashes.engineStart){
          box._preparingThrow=false;
          engine.rolling=false;engine.running=false;engine.callback=null;
          const ghostified=engine._ghostifiedIds;
          engine._ghostifiedIds=[];
          if(ghostified?.length)await engine.physicsWorker.exec('setCollisionResponse',{ids:ghostified,enabled:true});
        }
      }catch(cleanupError){recover(cleanupError);}
      try{queue._settleDroppedBinds(scope.items);}catch(cleanupError){recover(cleanupError);}
      finally{once();}
    }
  };
  const completion = engineProto.handlePersistentThrowCompletion;
  const effects = engineProto.handleSpecialEffectsInit;
  const pendingTail = (tail) => current === tail.scope && tail.scope.tail === tail;
  const attachedOwns = (tail) =>
    queue.box === box &&
    box.throwEngine === engine &&
    engine.callback === tail.once &&
    engine.throws === tail.throws &&
    box.physicsWorker === worker &&
    engine.physicsWorker === worker &&
    box.animateThrow === animate &&
    boxProto.animateThrow === animate &&
    engineProto.handlePersistentThrowCompletion === completion &&
    (engine.handlePersistentThrowCompletion === completionWrapper ||
      (!active && engine.handlePersistentThrowCompletion === completion)) &&
    engine.handleSpecialEffectsInit === effects &&
    engineProto.handleSpecialEffectsInit === effects;
  const owns = (tail) => pendingTail(tail) && attachedOwns(tail);
  const audited = () =>
    active &&
    box.animateThrow === animate &&
    boxProto.animateThrow === animate &&
    engine.handlePersistentThrowCompletion === completionWrapper &&
    engineProto.handlePersistentThrowCompletion === completion &&
    engine.handleSpecialEffectsInit === effects &&
    engineProto.handleSpecialEffectsInit === effects;
  const failed = (tail, error) => {
    recover(error);
    if (pendingTail(tail)) tail.scope.failed = true;
  };
  // Intercept only the two audited consumers. External await/catch calls keep
  // the original Promise, and the native closures retain their bundle aliases.
  const intercept = (promise, tail, consumer, handle) => {
    const then = promise.then;
    Object.defineProperty(promise, 'then', {
      configurable: true,
      writable: true,
      value: function (onFulfilled, onRejected) {
        if (
          this !== promise ||
          onRejected !== undefined ||
          digest(onFulfilled) !== consumer ||
          !owns(tail)
        )
          return then.call(this, onFulfilled, onRejected);
        return handle(then, onFulfilled);
      }
    });
    return promise;
  };
  const completionWrapper = function (...args) {
    const promise = completion.apply(this, args);
    const tail = current?.tail;
    if (this !== engine || !tail || !audited() || !owns(tail)) return promise;
    return intercept(promise, tail, hashes.effectsConsumer, (then, onEffects) => {
      const pending = then.call(promise, () => (owns(tail) ? onEffects() : undefined));
      return intercept(pending, tail, hashes.cleanupConsumer, (then, onCleanup) =>
        then
          .call(pending, undefined, (error) => failed(tail, error))
          .then(async () => {
            try {
              if (owns(tail)) {
                tail.cleaning=true;
                return await onCleanup();
              }
            } catch (error) {
              failed(tail, error);
            } finally {
              tail.cleaning=false;
              if (pendingTail(tail)) {
                if (owns(tail)) engine.rolling = false;
                else tail.scope.failed = true;
                tail.once(tail.throws);
              }
            }
          })
      );
    });
  };
  const completionSupported =
    animateHash === profile.boxAnimate &&
    digest(completion) === hashes.engineComplete &&
    digest(effects) === hashes.engineEffects &&
    !Object.hasOwn(box, 'animateThrow') &&
    !Object.hasOwn(engine, 'handlePersistentThrowCompletion') &&
    !Object.hasOwn(engine, 'handleSpecialEffectsInit') &&
    Object.isExtensible(engine);
  if (completionSupported)
    Object.defineProperty(engine, 'handlePersistentThrowCompletion', {
      value: completionWrapper,
      writable: true,
      configurable: true
    });
  Object.defineProperty(accumulator,'_onEnd',{...onEnd,value:callback});
  Object.defineProperty(box,'startUnifiedBatch',{value:wrapper,writable:true,configurable:true});
  return {status:'installed',completionStatus:completionSupported?'installed':'unsupported-source',attach,restore(){
    active = false;
    if (
      Object.getOwnPropertyDescriptor(engine, 'handlePersistentThrowCompletion')?.value === completionWrapper
    ) {
      delete engine.handlePersistentThrowCompletion;
    }
    if(accumulator._onEnd===callback)Object.defineProperty(accumulator,'_onEnd',onEnd);
    if(Object.getOwnPropertyDescriptor(box,'startUnifiedBatch')?.value===wrapper)delete box.startUnifiedBatch;
  }};
}
