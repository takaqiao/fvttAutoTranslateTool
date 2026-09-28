import {sha256Fallback} from '../source-hash.mjs';

const hashes={
  queue:'7253acd493cfc6fd612934ece4c2b30d84f1e04d51827eae9cd945e7a4270f77',
  attach:'33c3ae427c8bc94015e787bff10dc2a288436f6e27baec9dec708e249191f144',
  accumulator:'37b4b5b33974705d477aedbc8c84656813c92103391aff8551686d0d4a30a25d',
  boxStart:'a022226b8666a5de569bd756c3936933c137a23ab56b84b925491e01c4dcd993',
  engineStart:'91b92ea708def66da9c335f384df9bb01d398285fd6f6a7736155af56b600f21',
  onEnd:'70de7b37350e4e244dd05a35672dec84e2ebf7e15e482b16d41970a1d620dc7a'
};
const digest=fn=>typeof fn==='function'?sha256Fallback(Function.prototype.toString.call(fn)):null;

// The audited queue loses rejections inside an async Promise executor. Repair
// only batches dispatched by its serial Accumulator, using native completion.
export function installDsnQueueRecovery({queue,recover}){
  const accumulator=queue?.nextAnimation,box=queue?.box,engine=box?.throwEngine;
  const attach=queue&&Object.getPrototypeOf(queue)?.attach;
  const onEnd=Object.getOwnPropertyDescriptor(accumulator??{},'_onEnd');
  const boxProto=box&&Object.getPrototypeOf(box),engineProto=engine&&Object.getPrototypeOf(engine);
  const start=boxProto?.startUnifiedBatch;
  if(digest(queue?.constructor)!==hashes.queue||digest(accumulator?.constructor)!==hashes.accumulator
    ||digest(attach)!==hashes.attach||digest(onEnd?.value)!==hashes.onEnd||!onEnd?.writable)return {status:'unsupported-queue'};
  if(!box)return {status:'waiting-dsn',attach};
  if(digest(start)!==hashes.boxStart||!Object.isExtensible(box)||Object.hasOwn(box,'startUnifiedBatch'))return {status:'unsupported-queue'};
  // Native attach runs before initialize finishes. The texture-ready hook does
  // not await the box; recheck all sources after this specific box is ready.
  if(!engine&&typeof box.ready?.then==='function')return {status:'waiting-dsn',attach,ready:box.ready};
  if(digest(engineProto?.startUnifiedBatch)!==hashes.engineStart||Object.hasOwn(engine??{},'startUnifiedBatch'))return {status:'unsupported-queue'};
  // Replacing _onEnd affects the next Accumulator dispatch. The current one
  // retains its original closure. In particular, queue.idle() can resolve just
  // before Accumulator's finally clears _isProcessing during a native resize.
  let current=null;
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
    const complete=args[2];let completed=false;
    const once=(...values)=>{if(completed)return;completed=true;return complete(...values);};
    try{return await start.call(this,args[0],args[1],once);}
    catch(error){
      recover(error);
      // A callback already fired may have started the next batch. Do not reset
      // that later batch's physics or settle this one a second time.
      if(completed)return;
      scope.failed=true;
      box._preparingThrow=false;
      engine.rolling=false;engine.running=false;engine.callback=null;
      const ghostified=engine._ghostifiedIds;
      engine._ghostifiedIds=[];
      try{
        if(ghostified?.length)await engine.physicsWorker.exec('setCollisionResponse',{ids:ghostified,enabled:true});
      }catch(cleanupError){recover(cleanupError);}
      try{queue._settleDroppedBinds(scope.items);}catch(cleanupError){recover(cleanupError);}
      finally{once();}
    }
  };
  Object.defineProperty(accumulator,'_onEnd',{...onEnd,value:callback});
  Object.defineProperty(box,'startUnifiedBatch',{value:wrapper,writable:true,configurable:true});
  return {status:'installed',attach,restore(){
    if(accumulator._onEnd===callback)Object.defineProperty(accumulator,'_onEnd',onEnd);
    if(Object.getOwnPropertyDescriptor(box,'startUnifiedBatch')?.value===wrapper)delete box.startUnifiedBatch;
  }};
}
