import test from 'node:test';
import assert from 'node:assert/strict';
import {installBobTileCachePatch} from '../scripts/patches/bob-tile-cache.mjs';
import {MiB,createPyramid,tileEnvironment} from './bob-tile-cache-harness.mjs';

const at=(row,column)=>({x:column*1000+500,y:row*1000+500,scale:1});

function setup(limits={}){
  const e=tileEnvironment(),{pyramid}=e;
  for(const root of pyramid.allNodes.filter(node=>node.level===0))pyramid.load(root);
  e.visit=(row,column)=>{e.view(at(row,column));pyramid.load(pyramid.node(2,row,column));return pyramid.node(2,row,column);};
  e.install=()=>installBobTileCachePatch({runtime:e.runtime,report:entry=>(e.reports??=[]).push(entry),idleMs:5000,...limits});
  return e;
}

function assertDerived(node,ancestor,layers=[0,1]){
  for(const layer of layers){
    const shown=node.uniformGroup.uniforms[`sampler${layer}`],base=ancestor.layers[layer].texture.baseTexture;
    assert.equal(node.layers[layer].texture,null);
    assert.equal(shown.baseTexture,base);
    const resolution=base.width/ancestor.rect.width;
    assert.deepEqual({...shown.frame},{x:(node.rect.x-ancestor.rect.x)*resolution,y:(node.rect.y-ancestor.rect.y)*resolution,
      width:node.rect.width*resolution,height:node.rect.height*resolution});
  }
  assert.equal(node.sprite.texture,node.uniformGroup.uniforms.sampler0);
}

test('an inactive Bastion module is reported quietly and starts no timer',()=>{
  const e=tileEnvironment({active:false});
  assert.equal(installBobTileCachePatch({runtime:e.runtime}).status,'inactive');
  assert.equal(e.intervals.length,0);
});

test('reading the roots restores the current view exactly',()=>{
  const e=setup(),{pyramid}=e;e.visit(0,0);
  const before=[...pyramid.children],updates=pyramid.updates;
  const patch=e.install();patch.scan(0);
  assert.deepEqual(pyramid.children,before);
  assert.equal(pyramid.updates,updates+2);
  assert.equal(e.intervals[0].ms,30_000);
  assert.deepEqual(patch.stats,{pyramids:1,loadedMB:24,evictedTiles:0,unloaded:0},'status reads the live counters');
  assert.equal(e.reports.at(-1).stats,patch.stats);
  assert.equal(e.reports.at(-1).status,'installed');
});

test('the tiles out of view longest are unloaded down to the low mark',async()=>{
  const e=setup({highBytes:36*MiB,lowBytes:32*MiB}),{pyramid,unloaded,bases}=e;
  const patch=e.install();
  const far=e.visit(3,7);patch.scan(0);
  const middle=e.visit(3,5);patch.scan(10_000);
  const near=e.visit(0,0);
  const farTextures=far.layers.map(layer=>layer.texture),farBases=farTextures.map(texture=>texture.baseTexture);
  patch.scan(20_000);
  assert.deepEqual(unloaded,far.layers.map(layer=>layer.src));
  assert.equal(far.loaded,false);
  assert(farTextures.every(texture=>texture.destroyed));
  assert(farBases.every(base=>base.destroyed));
  assertDerived(far,pyramid.node(0,0,1));
  for(const kept of [middle,near])assert(kept.layers.every(layer=>layer.texture&&!layer.texture.destroyed));
  assert(bases.has(middle.layers[0].src));
  // Coming back reloads it through the pyramid's own path.
  e.visit(3,7);
  assert(far.layers.every(layer=>layer.texture&&!layer.texture.destroyed));
  assert.equal(far.sprite.texture,far.layers[0].texture);
});

test('a file cached under its cache-busting URL is unloaded under that key',()=>{
  const bust=src=>`https://cdn.example/${src}?cors-retry=1`;
  for(const [label,configure,key] of [
    ['cross-origin',(e,far)=>{
      const busted=new Set(far.layers.map(layer=>layer.src));
      e.runtime.PIXI.Assets.cache={has:key=>key.startsWith('https://')||(e.bases.has(key)&&!busted.has(key))};
      e.runtime.foundry={utils:{getCacheBustURL:bust}};
    },bust],
    // getCacheBustURL is false for same-origin and relative paths.
    ['same-origin',e=>{e.runtime.PIXI.Assets.cache={has:()=>false};e.runtime.foundry={utils:{getCacheBustURL:()=>false}};},src=>src],
    ['no cache index',()=>{},src=>src]
  ]){
    const e=setup({highBytes:36*MiB,lowBytes:32*MiB}),far=e.visit(3,7);configure(e,far);
    const patch=e.install();patch.scan(0);
    e.visit(3,5);patch.scan(10_000);e.visit(0,0);patch.scan(20_000);
    assert.deepEqual(e.unloaded,far.layers.map(layer=>key(layer.src)),label);
  }
});

test('descendants drawing from an evicted tile move to its nearest loaded ancestor',()=>{
  const e=setup({highBytes:30*MiB,lowBytes:28*MiB}),{pyramid}=e;
  const patch=e.install();
  const parent=pyramid.node(1,1,3),child=pyramid.node(2,3,7),sibling=pyramid.node(2,3,6);
  e.view(at(0,0));pyramid.load(parent);patch.scan(0);
  pyramid.load(sibling);patch.scan(1000);
  const previous=child.uniformGroup.uniforms.sampler0;
  assert.equal(previous.baseTexture,parent.layers[0].texture.baseTexture);
  patch.scan(10_000);
  assert.equal(parent.layers[0].texture,null,'the parent is the older of the two');
  assertDerived(parent,pyramid.node(0,0,1));
  assertDerived(child,pyramid.node(0,0,1));
  assert.equal(previous.destroyed,true);
  assert(sibling.layers.every(layer=>layer.texture&&!layer.texture.destroyed),'a descendant with its own tile is untouched');
});

test('tiles in view, their ancestors and the preload ring stay loaded over budget',()=>{
  const e=setup({highBytes:0,lowBytes:0}),{pyramid}=e;
  const patch=e.install();
  const viewed=e.visit(1,1);
  const parent=pyramid.node(1,0,0);pyramid.load(parent);
  const ring=pyramid.node(2,1,2);pyramid.load(ring);
  patch.scan(0);patch.scan(60_000);
  for(const node of [viewed,parent,ring,...pyramid.allNodes.filter(node=>node.level===0)])
    assert(node.layers.every(layer=>layer.texture),`${node.level}_${node.row}_${node.column}`);
});

test('recently seen and still loading tiles are not unloaded',()=>{
  const e=setup({highBytes:0,lowBytes:0}),{pyramid,unloaded}=e;
  const patch=e.install();
  e.view(at(3,7));
  const loading=pyramid.node(2,3,7);pyramid.load(loading,[0]);
  const idle=e.visit(3,5);
  e.view(at(0,0));
  patch.scan(0);
  assert.deepEqual(unloaded,[],'nothing has been idle long enough yet');
  patch.scan(10_000);
  assert(loading.layers[0].texture,'a tile whose other layer is still loading is skipped');
  assert.equal(idle.layers[0].texture,null);
});

// The shore backdrop pyramid draws its second layer from the shore background's files.
function sharedSetup(){
  const e=setup({highBytes:0,lowBytes:0}),{pyramid}=e;
  const backdrop=createPyramid(e,{prefix:'backdrop'});
  for(const node of backdrop.allNodes){
    const [,level,row,column]=node.layers[1].src.match(/(\d+)_(\d+)_(\d+)_1/);
    node.layers[1].src=pyramid.node(+level,+row,+column).layers[1].src;
  }
  for(const root of backdrop.allNodes.filter(node=>node.level===0))backdrop.load(root);
  e.runtime.canvas.primary.children.push(backdrop);
  const tile=e.visit(3,5),copy=backdrop.node(2,3,5);
  backdrop.load(copy);
  assert.equal(copy.layers[1].texture.baseTexture,tile.layers[1].texture.baseTexture);
  const both=position=>{e.view(position);backdrop.updateView(position);};
  both(at(0,0));
  return Object.assign(e,{backdrop,tile,copy,both});
}

test('a file two pyramids share is unloaded once, after both tiles let go',()=>{
  const e=sharedSetup(),shared=e.tile.layers[1].src,base=e.tile.layers[1].texture.baseTexture;
  const patch=e.install();patch.scan(0);patch.scan(10_000);
  assert.equal(e.tile.layers[1].texture,null);assert.equal(e.copy.layers[1].texture,null);
  assert.equal(e.unloaded.filter(src=>src===shared).length,1);
  assert.equal(base.destroyed,true);
  assert.equal(e.reports.at(-1).status,'installed');
});

test('a file a tile in view still shows is not unloaded',()=>{
  const e=sharedSetup(),shared=e.tile.layers[1].src,base=e.tile.layers[1].texture.baseTexture;
  const patch=e.install();patch.scan(0);
  // Only the backdrop looks at the shared tile, as a pyramid with another grid could.
  e.backdrop.updateView(at(3,5));
  patch.scan(10_000);
  assert.equal(e.tile.layers[0].texture,null,'the tile out of view still gives up its own file');
  assert(e.unloaded.includes(e.tile.layers[0].src));
  assert(!e.unloaded.includes(shared));
  assert.equal(base.destroyed,false);
  assert.equal(e.copy.layers[1].texture.baseTexture,base);
});

test('a pyramid whose nodes are not the documented shape is left alone',()=>{
  for(const spoil of [
    node=>{delete node.touchedId;},
    node=>{node.children=null;},
    node=>{node.uniformGroup.uniforms.sampler1=undefined;},
    node=>{node.uniformGroup.uniforms.sampler0={baseTexture:{}};}
  ]){
    const e=setup({highBytes:0,lowBytes:0}),{pyramid,unloaded}=e;
    const far=e.visit(3,7);e.view(at(0,0));spoil(pyramid.node(2,3,6));
    const patch=e.install();patch.scan(0);patch.scan(60_000);
    assert.equal(e.reports.at(-1).status,'unsupported-source',String(spoil));
    assert.deepEqual(unloaded,[]);assert(far.layers[0].texture);
  }
});

test('an instance-level _updateSprite is neither wrapped nor removed',()=>{
  const e=setup({highBytes:0,lowBytes:0}),{pyramid,unloaded}=e;
  e.visit(3,7);e.view(at(0,0));
  const override=function(node){return Object.getPrototypeOf(this)._updateSprite.call(this,node);};
  pyramid._updateSprite=override;
  const patch=e.install();patch.scan(0);patch.scan(60_000);
  assert.equal(e.reports.at(-1).status,'unsupported-source');
  assert.equal(pyramid._updateSprite,override);
  assert.deepEqual(unloaded,[]);
});

test('a pyramid that rebuilt its tiles after they were read stands down',()=>{
  const e=setup({highBytes:0,lowBytes:0}),{pyramid,unloaded,PIXI}=e;
  const far=e.visit(3,7);e.view(at(0,0));
  const patch=e.install();patch.scan(0);
  assert.equal(e.reports.at(-1).status,'installed');
  // The shown tile now draws through a sprite the captured tree has never seen.
  pyramid.node(2,0,0).sprite={texture:PIXI.Texture.EMPTY,parent:null};
  e.view(at(0,0));
  patch.scan(60_000);
  assert.equal(e.reports.at(-1).status,'unsupported-source');
  assert.match(e.reports.at(-1).detail,/replaced its tiles/);
  patch.scan(120_000);
  assert.deepEqual(unloaded,[]);assert(far.layers[0].texture);
});

test('the status recovers once every pyramid in view matches, and an empty scene reports no tile memory',()=>{
  const e=setup({highBytes:0,lowBytes:0});
  e.visit(3,7);e.view(at(0,0));
  const patch=e.install();patch.scan(0);
  assert(e.reports.at(-1).stats.loadedMB>0);
  const spoiled=createPyramid(e,{prefix:'spoiled'});spoiled.node(1,0,0).children=null;
  e.runtime.canvas.primary.children=[spoiled];
  patch.scan(10_000);
  assert.equal(e.reports.at(-1).status,'unsupported-source');
  // The next scene has no pyramid at all, then a well-formed one.
  e.runtime.canvas.primary.children=[{destroyed:false}];
  patch.scan(20_000);
  assert.equal(e.reports.at(-1).status,'installed');
  assert.equal(e.reports.at(-1).detail,undefined);
  assert.deepEqual([patch.stats.pyramids,patch.stats.loadedMB],[0,0]);
  const next=createPyramid(e,{prefix:'next'});
  for(const root of next.allNodes.filter(node=>node.level===0))next.load(root);
  e.runtime.canvas.primary.children=[next];
  patch.scan(30_000);
  assert.equal(patch.stats.pyramids,1);
  assert(patch.stats.loadedMB>0);
  assert.equal(e.reports.at(-1).status,'installed');
});

test('destroyed pyramids, an unready canvas and a missing view position are skipped',()=>{
  const e=setup({highBytes:0,lowBytes:0}),{pyramid,unloaded}=e;
  e.visit(3,7);e.view(at(0,0));
  const patch=e.install();
  e.runtime.canvas.ready=false;patch.scan(0);assert.equal(pyramid.updates,2);
  e.runtime.canvas.ready=true;e.scene._viewPosition={level:'level'};patch.scan(0);assert.equal(pyramid.updates,2);
  pyramid.destroyed=true;e.scene._viewPosition={...at(0,0)};patch.scan(0);patch.scan(60_000);
  assert.equal(pyramid.updates,2);assert.deepEqual(unloaded,[]);
});

test('a failing scan stops the timer and reports the failure',()=>{
  const e=setup();
  e.install();
  e.runtime.canvas.primary={get children(){throw Error('primary unavailable');}};
  e.intervals[0].fn();
  assert.equal(e.intervals[0].cleared,true);
  assert.equal(e.reports.at(-1).status,'failed');
  assert.match(e.reports.at(-1).detail,/primary unavailable/);
});
