import {hashSource} from '../source-hash.mjs';

// Foundry 14.368 source and served bundle: the complete class also covers its
// private stop queue and ended callback. Changed implementations fail closed.
const HASHES={
  constructor:'4dfb4732a049f9164dcc98c3d840464284cd64e23e622293cba628dc79ddcd06',
  stop:'e7f3e8b10e98da4ec02274665c0888ea58c471e2b87d6f7165baa36c36fa639c',
  _stop:'28226e38f07a2bb1c5c05fbab0e5a9f074e0c1fdfaff46a99c1af5c20e8c7964',
  play:'652c782af4c4705be4f0b1592cb1e9939db5ee94d29f6824d9c3a2eeef5fa492',
  _play:'ef17c3aa49ef8287e9457981308ca2212e1be45f7a02e772b01248b30dbff5e6',
  pause:'1cbd47131ed7bd52a2ef4b234663f9292c861872daea74b21d8fb556f840bb59',
  _pause:'d0e02467ed3ef3631483efcbcfec5a7d20b75efe42441957568afee634f5c8c5',
  _createNodes:'f38e7b2b919fb66e64286ebd9562b24556ee28910d7d645e8386d58e794e3b9a',
  _disconnectPipeline:'e8643960cecde825cced890c4f192f59bbb647c61803f0b0900c14465ed628c9',
  sourceNode:'9db29b77e2dd419f85e932bae95db2fec11e80412d8eb51d6ae9264fc54923bf',
  isBuffer:'139a98d4cb0a99d9e514431dbf8534a8d295aaaed588053d81eca22a6577746b',
  playing:'a86abe9529a3a02fad8a196f9537d882a5649fcfd4c3837b966f0b91d9c4a4fc',
  context:'d71f581b2ea9e0773e5124847b0d124885b0f38c374a365565dd48fde250be39',
  gain:'7ead1438ba9eff60f6907416ace04a56a560f3bb6d25228a0c8fb2e7c719f1f7'
};
const installed=new WeakSet();
const descriptorFunction=(prototype,key)=>{const d=Object.getOwnPropertyDescriptor(prototype,key);return d?.value??d?.get;};
const ownValue=(object,key)=>Object.getOwnPropertyDescriptor(object,key)?.value;

export async function installSoundStopPatch({moduleId='av-v14-hotfix',runtime=globalThis,report=()=>{}}={}){
  const finish=status=>{const result={feature:'soundStop',status};report(result);return result;};
  const core14=()=>(runtime.game?.release?.generation??Number.parseInt(runtime.game?.version,10))===14;
  if(!core14())return finish('unsupported-core');
  const Sound=runtime.foundry?.audio?.Sound,Buffer=runtime.AudioBuffer,Source=runtime.AudioBufferSourceNode;
  const prototype=Sound?.prototype,nodePrototype=Source?.prototype,nodeStop=nodePrototype?.stop;
  if(!prototype||typeof Buffer!=='function'||typeof nodeStop!=='function'||typeof runtime.libWrapper?.register!=='function'
    ||typeof runtime.Hooks?.on!=='function'||typeof runtime.Hooks?.off!=='function')return finish('unsupported-runtime');
  if(installed.has(prototype))return finish('already-installed');
  const native=Object.fromEntries(Object.keys(HASHES).map(key=>[key,key==='constructor'?Sound:descriptorFunction(prototype,key)]));
  for(const [key,fn]of Object.entries(native)){
    if(typeof fn!=='function'||await hashSource(Function.prototype.toString.call(fn))!==HASHES[key])return finish('unsupported-source');
  }
  const states=Sound.STATES;
  const unchanged=(includeStop=true)=>core14()&&runtime.foundry?.audio?.Sound===Sound&&runtime.AudioBuffer===Buffer
    &&runtime.AudioBufferSourceNode===Source&&nodePrototype.stop===nodeStop&&Sound.STATES===states
    &&Object.entries(native).every(([key,fn])=>key==='constructor'||(!includeStop&&key==='_stop')||descriptorFunction(prototype,key)===fn);
  if(!unchanged())return finish('source-changed-during-validation');
  const target='foundry.audio.Sound.prototype._stop';
  let warned=false,conflicted=false,targetId,registeredStop,registeredCallable;
  const stopChainUnchanged=()=>registeredStop===descriptorFunction(prototype,'_stop')&&prototype._stop===registeredCallable;
  const registrations=[];
  const conflict=()=>{if(!conflicted){conflicted=true;report({feature:'soundStop',status:'foreign-wrapper',detail:'Another registration changed the verified stop chain; reload to revalidate.'});}};
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
      try{nodeStop.call(node,0);}
      catch(error){
        // Do not prevent the original queue from disconnecting and clearing a
        // sound if an unexpected native node cannot be stopped.
        if(!warned){warned=true;report({feature:'soundStop',status:'native-stop-failed',detail:String(error)});}
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
  return finish(conflicted?'foreign-wrapper':'installed');
}
