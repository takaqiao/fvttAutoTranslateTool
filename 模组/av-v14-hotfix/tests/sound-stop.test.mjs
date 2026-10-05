import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {pathToFileURL} from 'node:url';

const app=process.env.FVTT_NATIVE_APP??'C:/Program Files/Foundry Virtual Tabletop/resources/app';
const soundURL=pathToFileURL(`${app}/client/audio/sound.mjs`);
const patchURL=new URL('../scripts/patches/sound-stop.mjs',import.meta.url);
let sequence=0;
async function fixture(t,{patched=true,accessorDispatcher=false}={}){
  const globals=['AudioBuffer','AudioBufferSourceNode','MediaElementAudioSourceNode','game'];
  const saved=Object.fromEntries(globals.map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
  t.after(()=>{for(const key of globals)saved[key]?Object.defineProperty(globalThis,key,saved[key]):delete globalThis[key];});
  const nodes=[],calls=[],reports=[];
  class AudioNode {connect(){}disconnect(){calls.push(['disconnect',this.kind]);}}
  class Param {value=1;cancelScheduledValues(){}setValueAtTime(value){this.value=value;}linearRampToValueAtTime(value){this.value=value;}exponentialRampToValueAtTime(value){this.value=value;}}
  class Gain extends AudioNode {gain=new Param();kind='gain';}
  class Buffer {constructor({length=48000,sampleRate=48000,kind='timeout'}={}){this.duration=length/sampleRate;this.kind=kind;}}
  class BufferSource extends AudioNode {
    constructor(_context,{buffer}){super();this.buffer=buffer;this.kind=buffer.kind;nodes.push(this);}
    start(){this.started=true;calls.push(['start',this.kind]);}
    stop(){calls.push(['stopAttempt',this.kind]);if(!this.started)throw new DOMException('not started','InvalidStateError');if(this.failStop)throw new Error('node stop failed');calls.push(['stop',this.kind]);this.stopped=true;}
    async finish(){this.ended=true;await this.onended?.();}
  }
  class MediaSource extends AudioNode {constructor(_context,{mediaElement}){super();this.element=mediaElement;this.kind='media';}}
  globalThis.AudioBuffer=Buffer;globalThis.AudioBufferSourceNode=BufferSource;globalThis.MediaElementAudioSourceNode=MediaSource;
  const context={currentTime:0,sampleRate:48000,destination:new AudioNode(),createGain:()=>new Gain()};
  const game={version:'14.368',release:{generation:14},audio:{music:context,playing:new Map(),debug(){}}};globalThis.game=game;
  const {default:Sound}=await import(`${soundURL.href}?test=${++sequence}`);
  let dispatcher,wrapper;
  const hookListeners=new Map();
  const Hooks={on(name,fn){hookListeners.set(name,fn);return fn;},off(name){hookListeners.delete(name);},callAll(name,...args){hookListeners.get(name)?.(...args);}};
  const runtime={game,foundry:{audio:{Sound}},AudioBuffer:Buffer,AudioBufferSourceNode:BufferSource,
    Hooks,libWrapper:{register(id,target,fn,type){assert.equal(target,'foundry.audio.Sound.prototype._stop');assert.equal(type,'WRAPPER');wrapper=fn;const original=Sound.prototype._stop;dispatcher=function(...args){return fn.call(this,original.bind(this),...args);};
      if(accessorDispatcher){let handler=dispatcher;Object.defineProperty(Sound.prototype,'_stop',{configurable:true,get(){return handler;},set(value){handler=value;}});}
      else Sound.prototype._stop=dispatcher;
      Hooks.callAll('libWrapper.Register',id,target,type,{},42);return 42;}}};
  const patch=existsSync(patchURL)?await import(patchURL.href):null;
  const install=()=>patch?.installSoundStopPatch({runtime,report:r=>reports.push(r)});
  if(patched)await install();
  const make=()=>{const sound=new Sound('generated-test-buffer',{context});sound.buffer=new Buffer({kind:'music'});sound._state=Sound.STATES.LOADED;return sound;};
  const count=(operation,kind='music')=>calls.filter(c=>c[0]===operation&&c[1]===kind).length;
  const tick=async()=>{for(let i=0;i<10;i++)await Promise.resolve();};
  const finishTimeout=async()=>{await tick();const n=nodes.findLast(n=>n.kind==='timeout'&&!n.ended&&!n.stopped);assert.ok(n,'scheduled audio timeout');context.currentTime+=n.buffer.duration;await n.finish();await tick();};
  return {Sound,context,game,runtime,patch,install,reports,nodes,calls,make,count,tick,finishTimeout,get wrapper(){return wrapper;},get dispatcher(){return dispatcher;}};
}

test('buffer stop terminates its started source once and preserves native events',async t=>{
  const f=await fixture(t),s=f.make(),events=[];s.addEventListener('stop',()=>events.push('stop'));s.addEventListener('end',()=>events.push('end'));
  await s.play({loop:true});const node=s.sourceNode;assert.equal(await s.stop(),s);await s.stop();
  assert.equal(f.count('stop'),1);assert.equal(node.onended,undefined);assert.equal(s.sourceNode,undefined);assert.equal(f.game.audio.playing.size,0);assert.deepEqual(events,['stop']);
});
test('twenty loop restarts stop all twenty sources',async t=>{const f=await fixture(t),s=f.make();for(let i=0;i<20;i++){await s.play({loop:true});await s.stop();}assert.equal(f.count('start'),20);assert.equal(f.count('stop'),20);});
test('pause keeps its native stop, then resumed playback stops the new node',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});f.context.currentTime=0.25;s.pause();assert.equal(f.count('stop'),1);await s.stop();assert.equal(f.count('stop'),1);await s.play();await s.stop();assert.equal(f.count('stop'),2);
});
test('natural completion keeps one native end event and its original stop event',async t=>{
  const f=await fixture(t),s=f.make(),events=[];s.addEventListener('stop',()=>events.push('stop'));s.addEventListener('end',()=>events.push('end'));
  await s.play({loop:false});await s.sourceNode.finish();assert.deepEqual(events,['stop','end']);assert.equal(s._state,f.Sound.STATES.STOPPED);assert.equal(f.game.audio.playing.size,0);
});
test('fade-out reaches its native boundary before stopping the source',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});const operation=s.stop({fade:100});await f.tick();assert.equal(f.count('stop'),0);await f.finishTimeout();await operation;assert.equal(f.count('stop'),1);
});
test('delayed stop waits for its native timeout',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});const operation=s.stop({delay:0.1});await f.tick();assert.equal(f.count('stop'),0);await f.finishTimeout();await operation;assert.equal(f.count('stop'),1);
});
test('cancelling delayed playback never stops an unstarted source',async t=>{
  const f=await fixture(t),s=f.make();const play=s.play({loop:true,delay:1});await f.tick();const stop=s.stop();await Promise.all([play,stop]);assert.equal(f.count('start'),0);assert.equal(f.count('stop'),0);assert.equal(s.sourceNode,undefined);
});
test('concurrent stop calls do not double-stop a fading source',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});const first=s.stop({fade:100}),second=s.stop({fade:100});await f.finishTimeout();await Promise.all([first,second]);assert.equal(f.count('stop'),1);
});
test('a play requested while stopping preserves the native no-op behavior',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});const stop=s.stop({fade:100});await s.play();await f.finishTimeout();await stop;assert.equal(f.count('start'),1);assert.equal(f.count('stop'),1);
});
test('streamed media keeps native pause and unload and never receives buffer stop',async t=>{
  const f=await fixture(t),s=f.make();s.buffer=null;let paused=0,removed=0;s.element={duration:1000,currentTime:0,play(){},pause(){paused++;},remove(){removed++;},src:'stream'};
  await s.play();await s.stop();assert.equal(paused,1);assert.equal(removed,1);assert.equal(s.element,null);assert.equal(f.count('stop'),0);
});
test('unknown source fingerprints decline installation',async t=>{
  const f=await fixture(t,{patched:false});f.Sound.prototype.stop=async function changedStop(){};const result=await f.install();assert.equal(result?.status,'unsupported-source');assert.equal(f.wrapper,undefined);
});
test('non-v14 core declines installation',async t=>{
  const f=await fixture(t,{patched:false});f.game.release.generation=15;const result=await f.install();assert.equal(result?.status,'unsupported-core');assert.equal(f.wrapper,undefined);
});
test('a later instance override follows the original path',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});s.stop=f.Sound.prototype.stop.bind(s);await s.stop();assert.equal(f.count('stop'),0);
});
test('subclassed Sound stays on its original path',async t=>{
  const f=await fixture(t);class Custom extends f.Sound {}const s=new Custom('custom',{context:f.context});s.buffer=new f.runtime.AudioBuffer({kind:'music'});s._state=f.Sound.STATES.LOADED;await s.play({loop:true});await s.stop();assert.equal(f.count('stop'),0);
});
test('the original wrapped exception remains the same error and skips the extra stop',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});s._state=f.Sound.STATES.STOPPING;const error=new Error('original');assert.throws(()=>f.wrapper.call(s,()=>{throw error}),e=>e===error);assert.equal(f.count('stop'),0);
});
test('unexpected wrapped return values remain unchanged without early stop',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});s._state=f.Sound.STATES.STOPPING;const value=Promise.resolve('foreign');assert.equal(f.wrapper.call(s,()=>value),value);assert.equal(f.count('stop'),0);
});
test('cancelling delayed resume does not stop its unstarted replacement node',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});f.context.currentTime=0.25;s.pause();
  const play=s.play({delay:1});await f.tick();const stop=s.stop();await Promise.all([play,stop]);
  assert.equal(f.count('start'),1);assert.equal(f.count('stopAttempt'),1);assert.equal(s.sourceNode,undefined);
});
test('a source changed while hashing is not wrapped',async t=>{
  const f=await fixture(t,{patched:false});const installation=f.install();f.Sound.prototype.pause=function changedPause(){};
  assert.equal((await installation).status,'source-changed-during-validation');assert.equal(f.wrapper,undefined);
});
test('a later prototype change falls back to the original path',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});const prior=f.Sound.prototype.play;f.Sound.prototype.play=function(...args){return prior.apply(this,args);};
  await s.stop();assert.equal(f.count('stop'),0);
});
test('unexpected node stop failure does not prevent native disconnect and bookkeeping',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});s.sourceNode.failStop=true;
  assert.equal(await s.stop(),s);assert.equal(s.sourceNode,undefined);assert.equal(f.game.audio.playing.size,0);assert.equal(f.reports.at(-1).status,'native-stop-failed');
});
test('installation is idempotent for the same native class',async t=>{
  const f=await fixture(t),original=f.dispatcher;assert.equal((await f.install()).status,'already-installed');assert.equal(f.Sound.prototype._stop,original);
});

test('a later foreign MIXED registration disables extra stop even when its short circuit returns undefined',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});s._state=f.Sound.STATES.STOPPING;
  f.runtime.Hooks.callAll('libWrapper.Register','other-module','Sound.prototype._stop','MIXED',{},42);
  assert.equal(f.wrapper.call(s,()=>undefined),undefined);assert.equal(f.count('stop'),0);
  // Unregistering the foreign module cannot silently reactivate an unverified chain.
  f.runtime.Hooks.callAll('libWrapper.Unregister','other-module','Sound.prototype._stop',42);
  f.wrapper.call(s,()=>undefined);assert.equal(f.count('stop'),0);
});

test('an ordinary later prototype _stop replacement skips the extra stop',async t=>{
  const f=await fixture(t),s=f.make();await s.play({loop:true});
  const dispatcher=f.Sound.prototype._stop;f.Sound.prototype._stop=function(...args){return dispatcher.apply(this,args);};
  await s.stop();assert.equal(f.count('stop'),0);
});

test('unrelated wrapper registrations leave the verified audio path enabled',async t=>{
  const f=await fixture(t),s=f.make();f.runtime.Hooks.callAll('libWrapper.Register','other-module','Other.prototype._stop','MIXED',{},43);
  await s.play({loop:true});await s.stop();assert.equal(f.count('stop'),1);
});

test('a stable libWrapper accessor whose setter replaces its handler disables extra stop',async t=>{
  const f=await fixture(t,{accessorDispatcher:true}),s=f.make();await s.play({loop:true});
  const getter=Object.getOwnPropertyDescriptor(f.Sound.prototype,'_stop').get;
  const handler=f.Sound.prototype._stop;f.Sound.prototype._stop=function(...args){return handler.apply(this,args);};
  assert.equal(Object.getOwnPropertyDescriptor(f.Sound.prototype,'_stop').get,getter);
  await s.stop();assert.equal(f.count('stop'),0);
});
