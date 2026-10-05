import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {installDsnChatRecovery} from '../scripts/patches/dsn-chat.mjs';

const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/dsn-model-native.json',import.meta.url)));
const chat=JSON.parse(fs.readFileSync(new URL('./fixtures/dsn-chat-native.json',import.meta.url)));
const queueFixture=JSON.parse(fs.readFileSync(new URL('./fixtures/dsn-queue-native.json',import.meta.url)));
const settle=()=>new Promise(resolve=>setImmediate(resolve));
const observe=promise=>{
 const result={state:'pending'};
 Promise.resolve(promise).then(value=>Object.assign(result,{state:'resolved',value}),error=>Object.assign(result,{state:'rejected',error}));
 return result;
};

function setup(){
 const requests=[],events=[],items=[],mesh={isMesh:true,userData:{},material:{map:{},normalMap:null,emissiveMap:null,roughnessMap:null,metalnessMap:null},layers:{enableAll(){this.enabled=true;}}};
 const scene={children:[mesh],traverse(fn){fn(mesh);},updateMatrixWorld(){}};
 const manager={itemStart:url=>items.push(['start',url]),itemEnd:url=>items.push(['end',url]),itemError:url=>items.push(['error',url])};
 // File I/O and parsed mesh construction are controlled here. The native
 // GLTF load/parse/error chain and DicePreset success callback run unchanged.
 class FileLoader{
  setPath(){return this;}setResponseType(){return this;}setRequestHeader(){return this;}setWithCredentials(){return this;}
  load(url,onLoad,onProgress,onError){requests.push({url,fail:onError,succeed:()=>onLoad(JSON.stringify({asset:{version:'2.0'}}))});}
 }
 const game={version:'14.368',modules:new Map([['dice-so-nice',{active:true,version:'6.4.1'}]])};
 const shader=function shader(){};
 const context=vm.createContext({game,TextDecoder,ArrayBuffer,FileLoader,ShaderUtils:{applyDiceSoNiceShader:shader},Hooks:{callAll:(...args)=>events.push(args)},LoaderUtils:{extractUrlBase:url=>url.slice(0,url.lastIndexOf('/')+1),resolveURL:(url,base)=>base+url},addUnknownExtensionsToUserData(){},assignExtrasToUserData(){},scene,manager});
 const classes=vm.runInContext(`(()=>{
  class GLTFParser{
   constructor(json,options){this.json=json;this.options=options;this.extensions={};this.cache={removeAll(){}};this.fileLoader=new FileLoader();}
   setExtensions(value){this.extensions=value;}setPlugins(){} _invokeAll(){return [];}
   getDependencies(type){return Promise.resolve(type==='scene'?[scene]:[]);}
   ${fixture.methods.parserParse}
  }
  class GLTFLoader{
   constructor(){this.manager=manager;this.resourcePath='';this.path='';this.requestHeader={};this.pluginCallbacks=[];}
   ${fixture.methods.loaderLoad}
   ${fixture.methods.loaderParse}
  }
  class DicePreset{
   constructor(){this.type='d20';this.shape='d20';this.modelFile='models/dice_20.gltf';this.modelLoaded=false;this.modelLoading=false;this.model=null;}
   unloadModel(){this.modelLoaded=false;this.modelLoading=false;}
   ${fixture.methods.loadModel}
  }
  class Pipeline{${Object.values(chat.methods).join('\n')}}
  return {DicePreset,GLTFLoader,Pipeline};
 })()`,context);
 const preset=new classes.DicePreset(),loader=new classes.GLTFLoader(),pipeline=new classes.Pipeline();pipeline.queue={};
 const factory={systems:new Map([['standard',{dice:new Map([['d20',preset]])}]])};
 game.dice3d={DiceFactory:factory,pipeline,box:{anisotropy:8}};
 const g={game,console:{error(){}}};
 return {g,preset,loader,pipeline,factory,requests,events,items,mesh,scene,shader,context,classes};
}

test('native GLTF onError rejects the shared model promise and permits a later retry',async()=>{
 const f=setup();installDsnChatRecovery({g:f.g});
 const first=f.preset.loadModel(f.loader),concurrent=f.preset.loadModel(f.loader),state=observe(first);
 assert.equal(first,concurrent);assert.equal(f.requests.length,1);
 const error=Error('THREE.GLTFLoader: Failed to load buffer "dice_20.bin".');f.requests[0].fail(error);await settle();
 assert.equal(state.state,'rejected');assert.equal(state.error,error);
 assert.equal(f.preset.modelLoading,false);assert.equal(f.preset.modelLoaded,false);
 const retry=f.preset.loadModel(f.loader);assert.notEqual(retry,first);assert.equal(f.requests.length,2);
 f.requests[1].succeed();const model=await retry;
 assert.equal(f.preset.model,model);assert.equal(f.preset.modelLoaded,true);
});

test('successful native model preparation keeps concurrent and cached promise identity',async()=>{
 const f=setup();installDsnChatRecovery({g:f.g});
 const first=f.preset.loadModel(f.loader);assert.equal(f.preset.loadModel(f.loader),first);
 f.requests[0].succeed();const model=await first;
 assert.equal(f.preset.loadModel(f.loader),first);assert.equal(f.requests.length,1);
 assert.equal(f.preset.model,model);assert.equal(model.scene,f.scene);assert.equal(f.mesh.castShadow,true);
 assert.equal(f.mesh.material.map.anisotropy,8);assert.equal(f.mesh.material.onBeforeCompile,f.shader);assert.equal(f.mesh.layers.enabled,true);
 assert.deepEqual(f.events.map(e=>e[0]),['diceSoNiceOnMaterialReady','diceSoNiceModelLoaded']);
 assert.equal(f.events[0][1],f.mesh.material);assert.equal(f.events[1][1],f.preset);
 assert.deepEqual(f.items,[['start','models/dice_20.gltf'],['end','models/dice_20.gltf']]);
});

test('cached and concurrent model calls do not inspect an unused loader argument',async()=>{
 const f=setup();installDsnChatRecovery({g:f.g});
 const unused=new Proxy({},{getPrototypeOf(){throw Error('unused loader inspected');}});
 const first=f.preset.loadModel(f.loader);
 assert.equal(f.preset.loadModel(unused),first);
 f.requests[0].succeed();await first;assert.equal(f.preset.loadModel(unused),first);
});

test('an exception inside the native model callback rejects through the real GLTF parse error path',async()=>{
 const f=setup();installDsnChatRecovery({g:f.g});
 const error=Error('native material preparation failed');f.scene.traverse=()=>{throw error;};
 const state=observe(f.preset.loadModel(f.loader));f.requests[0].succeed();await settle();
 assert.equal(state.state,'rejected');assert.equal(state.error,error);assert.equal(f.preset.modelLoading,false);
 assert.deepEqual(f.items.map(i=>i[0]),['start','error','end']);
});

test('synchronous loader failure clears only the failed attempt and preserves its error',async()=>{
 const f=setup();installDsnChatRecovery({g:f.g});
 const error=Error('loader setup failed');f.loader.manager.itemStart=()=>{throw error;};
 const state=observe(f.preset.loadModel(f.loader));await settle();
 assert.equal(state.state,'rejected');assert.equal(state.error,error);assert.equal(f.preset.modelLoading,false);
});

test('an older failed model load cannot clear the newer in-flight cache',async()=>{
 const f=setup();installDsnChatRecovery({g:f.g});
 const old=observe(f.preset.loadModel(f.loader));f.preset.unloadModel();
 const current=f.preset.loadModel(f.loader),error=Error('old load failed');f.requests[0].fail(error);await settle();
 assert.equal(old.state,'rejected');assert.equal(old.error,error);assert.equal(f.preset.modelLoading,current);
 f.requests[1].succeed();assert.equal(await current,f.preset.model);
});

test('foreign loader methods keep their native receiver and original two callback arguments',async()=>{
 const f=setup();installDsnChatRecovery({g:f.g});let receiver,args;
 f.loader.load=function(...values){receiver=this;args=values;};
 const promise=f.preset.loadModel(f.loader);
 assert.equal(receiver,f.loader);assert.equal(args.length,2);assert.equal(args[0],f.preset.modelFile);
 assert.equal(f.preset.modelLoading,promise);
});

for(const change of ['version','preset','loader-parse'])test(`unknown ${change} source retains native model loading`,async()=>{
 const f=setup(),native=Object.getPrototypeOf(f.preset).loadModel;
 if(change==='version')f.g.game.modules.get('dice-so-nice').version='6.4.2';
 if(change==='preset')Object.getPrototypeOf(f.preset).loadModel=function foreign(){};
 if(change==='loader-parse')Object.getPrototypeOf(f.loader).parse=function foreign(){};
 const original=f.preset.loadModel;installDsnChatRecovery({g:f.g});
 if(change!=='loader-parse')assert.equal(f.preset.loadModel,original);
 else {
  const state=observe(f.preset.loadModel(f.loader));f.requests[0].fail(Error('foreign loader'));await settle();
  assert.equal(state.state,'pending');
 }
 if(change==='version')assert.equal(f.preset.loadModel,native);
});

test('restore removes only the owned model adapter and repeat install is idempotent',()=>{
 const f=setup(),proto=Object.getPrototypeOf(f.preset),native=proto.loadModel;
 const result=installDsnChatRecovery({g:f.g}),wrapper=proto.loadModel;
 assert.notEqual(wrapper,native);assert.equal(installDsnChatRecovery({g:f.g}),result);
 result.restore();assert.equal(proto.loadModel,native);
 const current=installDsnChatRecovery({g:f.g}),foreign=function foreign(){};proto.loadModel=foreign;
 result.restore();current.restore();assert.equal(proto.loadModel,foreign);
});

test('an already-started native load is reported without replacing its cached promise',async()=>{
 const f=setup(),pending=f.preset.loadModel(f.loader),state=observe(pending);
 const result=installDsnChatRecovery({g:f.g});
 assert.equal(result.modelPendingAtInstall,1);assert.equal(f.preset.loadModel(f.loader),pending);
 f.requests[0].fail(Error('before adapter'));await settle();assert.equal(state.state,'pending');
});

test('diceSoNiceReady installs model error propagation before immediate native preloading',async()=>{
 const f=setup(),dice3d=f.g.game.dice3d,hooks=[];delete f.g.game.dice3d;
 f.g.Hooks={once:(event,fn)=>{hooks.push({event,fn});return hooks.length;},off(){}};
 assert.equal(installDsnChatRecovery({g:f.g}).status,'waiting-dsn');f.g.game.dice3d=dice3d;
 for(const hook of hooks)if(hook.event==='diceSoNiceReady')hook.fn();
 const state=observe(f.preset.loadModel(f.loader));f.requests[0].fail(Error('preload failed'));await settle();
 assert.equal(state.state,'rejected');assert.equal(f.preset.modelLoading,false);
});

test('a native model download failure settles the real AnimationQueue batch before physics simulation',async()=>{
 const f=setup();let hidden=0,simulated=0;
 f.g.game.settings={get:(_module,key)=>key==='maxDiceNumber'?20:false};
 Object.assign(f.context,{setTimeout,clearTimeout,DsnSettings:{isEnabled:()=>true},DiceNotation:{mergeQueuedRollCommands:()=>[[{dice:[{}],dsnConfig:{}}]]},Utils:{removeTicker(){}},canvas:{app:{ticker:{add(){}}}}});
 const classes=vm.runInContext(`(()=>{${queueFixture.accumulator};${queueFixture.queue};return {AnimationQueue,Box:class{${queueFixture.boxStart}},Engine:class{${queueFixture.engineStart}}}})()`,f.context);
 const engine=Object.assign(new classes.Engine(),{rolling:false,running:false,diceList:[],deadDiceList:[],persistentDiceList:[],clearDice(){},getVectors(){},diceScene:{display:{innerWidth:1000,innerHeight:800}},async spawnDiceMesh(){await f.preset.loadModel(f.loader);},physicsWorker:{exec(){simulated++;}}});
 const box=Object.assign(new classes.Box(),{throwEngine:engine,inputHandler:{clearPendingThrowDice(){}},animateThrow(){}});
 const queue=new classes.AnimationQueue({canvasVisibility:{show(){},hide(){hidden++;}},pendingThrows:{}});queue.attach(box);f.pipeline.queue=queue;
 const installed=installDsnChatRecovery({g:f.g}),state=observe(queue.enqueue({throws:[{}]},{}));
 await settle();assert.equal(f.requests.length,1);f.requests[0].fail(Error('buffer download failed'));await settle();
 assert.equal(state.state,'resolved');assert.equal(state.value,false);assert.equal(hidden,1);
 assert.equal(simulated,0);assert.equal(box._preparingThrow,false);assert.equal(engine.rolling,false);
 assert.equal(f.preset.modelLoading,false);assert.equal(installed.stats.recovered,1);await queue.idle();
});
