// The extra stop relies on these Sound members, plus the private stop queue
// and ended callback, which cannot change while the class stays the same
// object. Each member is pinned by identity at install; once one is replaced,
// _stop takes the original path.
const MEMBERS=['constructor','stop','_stop','play','_play','pause','_pause','_createNodes','_disconnectPipeline',
  'sourceNode','isBuffer','playing','context','gain'];
const installed=new WeakSet();
const descriptorFunction=(prototype,key)=>{const d=Object.getOwnPropertyDescriptor(prototype,key);return d?.value??d?.get;};
const ownValue=(object,key)=>Object.getOwnPropertyDescriptor(object,key)?.value;

export function installSoundStopPatch({moduleId='av-v14-hotfix',runtime=globalThis,report=()=>{}}={}){
  const finish=(status,extra)=>{const result={feature:'soundStop',status,...extra};report(result);return result;};
  const Sound=runtime.foundry?.audio?.Sound,Buffer=runtime.AudioBuffer,Source=runtime.AudioBufferSourceNode;
  const prototype=Sound?.prototype,nodePrototype=Source?.prototype,nodeStop=nodePrototype?.stop;
  if(!prototype||typeof Buffer!=='function'||typeof nodeStop!=='function'||typeof runtime.libWrapper?.register!=='function'
    ||typeof runtime.Hooks?.on!=='function'||typeof runtime.Hooks?.off!=='function')return finish('unsupported-runtime');
  if(installed.has(prototype))return finish('already-installed');
  const native=Object.fromEntries(MEMBERS.map(key=>[key,key==='constructor'?Sound:descriptorFunction(prototype,key)]));
  if(Object.values(native).some(fn=>typeof fn!=='function'))return finish('unsupported-runtime');
  const states=Sound.STATES;
  const unchanged=(includeStop=true)=>runtime.foundry?.audio?.Sound===Sound&&runtime.AudioBuffer===Buffer
    &&runtime.AudioBufferSourceNode===Source&&nodePrototype.stop===nodeStop&&Sound.STATES===states
    &&Object.entries(native).every(([key,fn])=>key==='constructor'||(!includeStop&&key==='_stop')||descriptorFunction(prototype,key)===fn);
  // Only an accessor that yields a different value per read can differ this
  // early; such a member could never match at stop time either.
  if(!unchanged())return finish('source-changed-during-validation');
  const target='foundry.audio.Sound.prototype._stop';
  // Live counter: the entry keeps the reported object and reads it later.
  const stats={extraStops:0};
  let warned=false,conflicted=false,targetId,registeredStop,registeredCallable;
  const stopChainUnchanged=()=>registeredStop===descriptorFunction(prototype,'_stop')&&prototype._stop===registeredCallable;
  const registrations=[];
  const conflict=()=>{if(!conflicted){conflicted=true;report({feature:'soundStop',status:'foreign-wrapper',detail:'Another registration changed the verified stop chain; reload to revalidate.',stats});}};
  const hookId=runtime.Hooks.on('libWrapper.Register',(owner,path,_type,_options,id)=>{
    if(owner===moduleId)return;
    if(path===target||(Number.isSafeInteger(targetId)&&id===targetId))conflict();
    else if(targetId===undefined)registrations.push(id);
  });
  try{targetId=runtime.libWrapper.register(moduleId,target,function(wrapped,...args){
    let node;
    if(!conflicted&&stopChainUnchanged()&&unchanged(false)&&Object.getPrototypeOf(this)===prototype
      &&Object.keys(native).every(key=>key==='constructor'||!Object.hasOwn(this,key))
      &&ownValue(this,'_state')===states.STOPPING&&Number.isFinite(ownValue(this,'startTime'))
      &&ownValue(this,'pausedTime')===undefined&&ownValue(this,'buffer') instanceof Buffer&&!ownValue(this,'element')){
      const candidate=native.sourceNode.call(this);
      if(candidate&&Object.getPrototypeOf(candidate)===nodePrototype&&candidate.stop===nodeStop
        &&candidate.buffer===ownValue(this,'buffer'))node=candidate;
    }
    // Keep native fades/delays and cleanup ordering. Its PLAYING-only condition
    // skips a buffer in STOPPING. A paused or not-yet-started node is excluded.
    const result=wrapped(...args);
    if(node&&!conflicted&&stopChainUnchanged()&&result===undefined&&ownValue(this,'_state')===states.STOPPING&&native.sourceNode.call(this)===node){
      try{nodeStop.call(node,0);stats.extraStops++;}
      catch(error){
        // Do not prevent the original queue from disconnecting and clearing a
        // sound if an unexpected native node cannot be stopped.
        if(!warned){warned=true;report({feature:'soundStop',status:'native-stop-failed',detail:String(error),stats});}
      }
    }
    return result;
  },'WRAPPER');}
  catch(error){runtime.Hooks.off('libWrapper.Register',hookId);throw error;}
  registeredStop=descriptorFunction(prototype,'_stop');
  // libWrapper's property setter can replace its cached callable without
  // changing the descriptor or emitting a registration event.
  registeredCallable=prototype._stop;
  if(Number.isSafeInteger(targetId)&&registrations.includes(targetId))conflict();
  // The returned target ID identifies aliases as well as the canonical path.
  // Without it, the chain cannot be monitored safely.
  if(!Number.isSafeInteger(targetId))conflict();
  installed.add(prototype);
  return finish(conflicted?'foreign-wrapper':'installed',{stats});
}
