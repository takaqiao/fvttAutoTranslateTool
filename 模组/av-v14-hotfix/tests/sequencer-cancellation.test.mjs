import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/sequencer-cancellation-native.json',import.meta.url),'utf8'));
const deferred=()=>{let resolve,reject;const promise=new Promise((ok,fail)=>{resolve=ok;reject=fail});return {promise,resolve,reject}};

function setup(version='after'){
 const sprites=[],gates=[],waiters=[],events=[],resolutions=[];
 const point=()=>({set(x,y){this.x=x;this.y=y}});
 class Sprite{
  constructor(){this.destroyed=false;this.position=point();this.anchor=point();this.volumeWrites=[];sprites.push(this);gates.push(deferred());waiters[sprites.length-1]?.resolve(this)}
  activate(){return gates[sprites.indexOf(this)].promise}
  destroy(){this.destroyed=true}
  set volume(value){if(this.destroyed)throw Error('write to destroyed sprite');this.volumeWrites.push(value)}
  preloadVariants(){events.push('preload')}
  addText(){events.push('text');return {anchor:point()}}
 }
 const math=Object.assign(Object.create(Math),{toRadians:value=>value*Math.PI/180,normalizeRadians:value=>value});
 const methods=vm.runInNewContext('({'+[...Object.values(fixture[version]),...Object.values(fixture.lifecycle)].join(',\n')+'})',{
  SequencerSpriteManager:Sprite,Math:math,game:{settings:{get:()=>0.5}},fromUuidSync:()=>({object:{sourceElement:{currentTime:11}}}),
  foundry:{utils:{deepClone:structuredClone}},canvas:{grid:{size:100}},PluginsManager:{createSprite:()=>events.push('plugin')},PIXI:{ColorMatrixFilter:class{}},
  Hooks:{callAll(){}},debug(){},hooksManager:{removeHooks(){}},SequencerAnimationEngine:{endAnimations(){}},clearTimeout,
 });
 Object.setPrototypeOf(methods,{destroy(){this.destroyed=true}});
 const effect={...methods,id:'effect',data:{volume:0.8,opacity:1},_startTime:2,loopDelay:3,loops:4,ready:false,
  _initializeVariables(){this._ended=null;this.spriteContainer={addChild(){},rotation:0};this.rotationContainer={position:point()};this.effectFilters={};this._tickerMethods=[]},
  _addToContainer(){events.push('container')},async _createFile(){events.push('file')},_updateCurrentFilePath(){this._currentFilePath='fixture.webm'},
  _recomputeRenderable(){},updateElevation(){events.push('elevation')},
  _teardownVoidProxy(){},removeChildren(){return []},
  _durationResolve:value=>resolutions.push(['duration',value]),_resolve:value=>resolutions.push(['finish',value]),
 };
 for(const name of ['_calculateDuration','_createShapes','_setupMasks','_transformSprite','_playPresetAnimations','_playCustomAnimations','_setEndTimeout','_registerTickers','_timeoutVisibility','_startEffect'])effect[name]=()=>{events.push(name)};
 const waitForSprite=index=>sprites[index]?Promise.resolve(sprites[index]):(waiters[index]??=deferred()).promise;
 const cancel=()=>{effect._ended=true;if(effect.sprite)effect.sprite.destroyed=true;effect.sprite=null;effect.spriteContainer=null};
 const replace=()=>effect.sprite=new Sprite();
 return {effect,sprites,gates,events,resolutions,waitForSprite,cancel,replace};
}

test('captured Sequencer 4.2.3 reproduces the destroyed-sprite volume error',async()=>{
 const f=setup('before'),run=f.effect._initialize();await f.waitForSprite(0);f.cancel();f.gates[0].resolve();
 await assert.rejects(run,/null.*volume/);assert.equal(f.events.includes('_registerTickers'),false);
});

test('destroying an activating sprite cancels initialization before shapes and tickers',async()=>{
 const f=setup(),run=f.effect._initialize();const sprite=await f.waitForSprite(0);f.cancel();f.gates[0].resolve();
 await run;assert.deepEqual(sprite.volumeWrites,[]);assert.equal(f.effect.ready,false);
 assert.deepEqual(f.events,['container','file']);assert.deepEqual(f.resolutions,[['duration',0],['finish',f.effect.data]]);
});

test('an ended effect does not configure an activating sprite that is still referenced',async()=>{
 const f=setup(),run=f.effect._initialize();const sprite=await f.waitForSprite(0);f.effect._ended=true;f.gates[0].resolve();
 await run;assert.deepEqual(sprite.volumeWrites,[]);assert.deepEqual(f.events,['container','file']);assert.equal(f.effect.ready,false);
});

test('a destroyed sprite still referenced by the effect cannot finish initialization',async()=>{
 const f=setup(),run=f.effect._initialize();const sprite=await f.waitForSprite(0);sprite.destroyed=true;f.gates[0].resolve();
 await run;assert.deepEqual(sprite.volumeWrites,[]);assert.deepEqual(f.events,['container','file']);assert.equal(f.effect.ready,false);
});

test('replacing a sprite during activation leaves its replacement untouched',async()=>{
 const f=setup(),run=f.effect._initialize();const sprite=await f.waitForSprite(0),replacement=f.replace();f.gates[0].resolve();
 await run;assert.equal(f.effect.sprite,replacement);assert.deepEqual(replacement.volumeWrites,[]);assert.deepEqual(sprite.volumeWrites,[]);
 assert.deepEqual(f.events,['container','file']);assert.equal(f.effect.ready,false);
});

test('replacing the sprite container during activation cancels the old initialization',async()=>{
 const f=setup(),run=f.effect._initialize();const sprite=await f.waitForSprite(0);f.effect.spriteContainer={addChild(){},rotation:8};f.gates[0].resolve();
 await run;assert.deepEqual(sprite.volumeWrites,[]);assert.equal(f.effect.spriteContainer.rotation,8);assert.deepEqual(f.events,['container','file']);
});

test('a replacement between sprite completion and initializer resumption stops later setup',async()=>{
 const f=setup(),run=f.effect._initialize();await f.waitForSprite(0);f.gates[0].resolve();queueMicrotask(()=>f.replace());
 await run;assert.equal(f.events.includes('_calculateDuration'),false);assert.equal(f.events.includes('_registerTickers'),false);assert.equal(f.effect.ready,false);
});

test('normal initialization retains volume, timing and downstream setup',async()=>{
 const f=setup(),run=f.effect._initialize();const sprite=await f.waitForSprite(0);f.gates[0].resolve();await run;
 assert.deepEqual(sprite.volumeWrites,[0.4]);assert.equal(sprite.currentTime,2);assert.equal(sprite.loopDelay,3);assert.equal(sprite.loop,4);
 assert.equal(f.effect.ready,true);assert.deepEqual(f.resolutions,[]);
 assert.deepEqual(f.events,['container','file','plugin','elevation','_calculateDuration','_createShapes','_setupMasks','_transformSprite','_playPresetAnimations','_playCustomAnimations','_setEndTimeout','_registerTickers','_timeoutVisibility','_startEffect']);
});

test('normal copy-sprite initialization preserves source time and text configuration',async()=>{
 const f=setup();f.effect.data.copySprite={uuid:'Token.source',offsetX:5,offsetY:6};f.effect.data.text={text:'caption'};
 const run=f.effect._initialize(),sprite=await f.waitForSprite(0);f.gates[0].resolve();await run;
 assert.equal(sprite.currentTime,11);assert.deepEqual(sprite.volumeWrites,[0.4]);assert.equal(f.events.includes('text'),true);assert.equal(f.effect.ready,true);
});

test('a real activation failure retains its original error and native settlement',async()=>{
 const f=setup(),failure=Error('asset failed',{cause:Error('decode failed')}),run=f.effect._initialize();await f.waitForSprite(0);f.gates[0].reject(failure);
 await assert.rejects(run,error=>error===failure);assert.equal(f.effect.ready,false);
 assert.deepEqual(f.resolutions,[['duration',0],['finish',f.effect.data]]);assert.equal(f.events.includes('_registerTickers'),false);
});

test('an older activation finishing after reinitialization cannot modify the new sprite',async()=>{
 const f=setup(),old=f.effect._initialize();const oldSprite=await f.waitForSprite(0);f.cancel();
 const current=f.effect._initialize(),currentSprite=await f.waitForSprite(1);f.gates[1].resolve();await current;const before=[...f.events];
 f.gates[0].resolve();await old;
 assert.equal(f.effect.sprite,currentSprite);assert.equal(f.effect.ready,true);assert.deepEqual(oldSprite.volumeWrites,[]);assert.deepEqual(currentSprite.volumeWrites,[0.4]);
 assert.deepEqual(f.events,before);assert.deepEqual(f.resolutions,[]);
});

test('a rejected older initialization cannot resolve or clear the newer callbacks',async()=>{
 const f=setup(),old=f.effect._initialize();await f.waitForSprite(0);f.cancel();
 const current=f.effect._initialize();await f.waitForSprite(1);
 const duration=()=>assert.fail('old initialization settled new duration'),finish=()=>assert.fail('old initialization settled new finish');
 f.effect._durationResolve=duration;f.effect._resolve=finish;const failure=Error('old asset failed');f.gates[0].reject(failure);
 await assert.rejects(old,error=>error===failure);assert.equal(f.effect._durationResolve,duration);assert.equal(f.effect._resolve,finish);
 f.gates[1].resolve();await current;assert.equal(f.effect.ready,true);assert.deepEqual(f.resolutions,[]);
});

for(const method of ['destroy','_destroyDependencies','endEffect'])test(`native play promises both settle when ${method} cancels sprite activation`,async()=>{
 const f=setup(),initialize=f.effect._initialize;let run;
 f.effect._initialize=function(){return run=initialize.call(this)};
 const playing=f.effect.play(),settled={};playing.duration.then(value=>settled.duration=value);playing.promise.then(value=>settled.finish=value);
 await f.waitForSprite(0);f.effect[method]();f.gates[0].resolve();await run;await new Promise(resolve=>setImmediate(resolve));
 assert.equal(settled.duration,0);assert.equal(settled.finish,f.effect.data);assert.equal(f.effect.ready,false);
 assert.equal(f.events.includes('_registerTickers'),false);
});

test('cancellation does not settle callbacks replaced without reusing the initialization',async()=>{
 const f=setup(),run=f.effect._initialize();await f.waitForSprite(0);f.effect.destroy();
 const duration=()=>assert.fail('cancel settled a replacement duration'),finish=()=>assert.fail('cancel settled a replacement finish');
 f.effect._durationResolve=duration;f.effect._resolve=finish;f.gates[0].resolve();await run;
 assert.equal(f.effect._durationResolve,duration);assert.equal(f.effect._resolve,finish);assert.deepEqual(f.resolutions,[['duration',0]]);
});

test('activation rejection after native destruction settles its finish and preserves the original error',async()=>{
 const f=setup(),failure=Error('destroyed asset failed'),run=f.effect._initialize();await f.waitForSprite(0);f.effect.destroy();f.gates[0].reject(failure);
 await assert.rejects(run,error=>error===failure);assert.deepEqual(f.resolutions,[['duration',0],['finish',f.effect.data]]);
});
