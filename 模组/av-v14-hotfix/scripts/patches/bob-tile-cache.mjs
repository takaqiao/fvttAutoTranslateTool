// Bastion of Blasphemies streams its large maps as sprite pyramids: tiles load as
// the camera reaches them and stay loaded until the scene is left. Core only
// expires textures when the canvas is redrawn, so a long session on one map
// keeps every visited tile in memory.
//
// This keeps the loaded tiles of each pyramid under a budget by unloading the
// ones that have been out of view longest. It only uses the documented node
// structure (the SpritePyramidNode typedef) and the public updateView,
// _updateSprite and _invalidateSprites methods; an evicted tile is left in the
// state the pyramid itself starts unloaded tiles in, so the pyramid reloads it
// the next time it comes into view.
const MODULE='pf2e-bastion-of-blasphemies';
const MiB=1024**2;
const DEFAULTS={intervalMs:30_000,highBytes:1024*MiB,lowBytes:768*MiB,idleMs:120_000};
// updateView at this scale selects the lowest level over the whole map: every root.
const OVERVIEW={x:0,y:0,scale:1e-9};

const finite=value=>typeof value==='number'&&Number.isFinite(value);

function isPyramid(object){
  return !!object&&!object.destroyed&&Array.isArray(object.children)&&['updateView','getTexturesToPreload','_invalidateSprites','_updateSprite','_loadPendingNodes']
    .every(name=>typeof object[name]==='function');
}

function validNode(node){
  const {rect,layers,uniformGroup,sprite,children}=node??{};
  return typeof node?.level==='number'&&typeof node.loaded==='boolean'&&typeof node.touchedId==='number'
    &&finite(rect?.x)&&finite(rect?.y)&&finite(rect?.width)&&finite(rect?.height)&&rect.width>0
    &&Array.isArray(layers)&&layers.length>0&&layers.every(layer=>layer&&'src' in layer&&'texture' in layer)
    &&layers.every((layer,index)=>uniformGroup?.uniforms?.[`sampler${index}`]!==undefined)
    &&!!sprite&&typeof sprite==='object'&&Array.isArray(children);
}

/** Install the tile budget for Bastion of Blasphemies sprite pyramids. */
export function installBobTileCachePatch({runtime=globalThis,report=()=>{},...limits}={}){
  const config={...DEFAULTS,...limits};
  const stats={pyramids:0,loadedMB:0,evictedTiles:0,unloaded:0};
  let status='installed',detail;
  const finish=(next=status,extra={})=>{
    status=next;
    const result={feature:'bobTileCache',status,stats,...(detail?{detail}:{}),...extra};
    report(result);return result;
  };
  if(!runtime.game?.modules?.get(MODULE)?.active)return finish('inactive',{reason:'module-inactive'});
  const PIXI=runtime.PIXI,EMPTY=PIXI?.Texture?.EMPTY;
  if(typeof PIXI?.Texture!=='function'||typeof PIXI.Rectangle!=='function'||typeof PIXI.Assets?.unload!=='function'||!EMPTY
    ||typeof runtime.setInterval!=='function'||typeof runtime.clearInterval!=='function')return finish('unsupported-runtime');

  // Pyramid -> tree state, or null when its nodes are not the expected shape.
  const trees=new WeakMap();

  // The texture loader retries a cross-origin file under its cache-busting URL
  // and caches it under that key; look it up the way the loader's getCache does.
  function cacheKey(src){
    if(PIXI.Assets.cache?.has?.(src)!==false)return src;
    return runtime.foundry?.utils?.getCacheBustURL?.(src)||src;
  }

  function viewPosition(){
    const view=runtime.canvas?.scene?._viewPosition;
    if(finite(view?.x)&&finite(view?.y)&&finite(view?.scale)&&view.scale>0)return {x:view.x,y:view.y,scale:view.scale};
    return null;
  }

  // The pyramid keeps its roots private; viewing the whole map once hands them
  // to _updateSprite, and the real view is restored in the same synchronous step.
  function capture(pyramid,position){
    // Someone else's instance-level override is neither wrapped nor removed.
    if(Object.hasOwn(pyramid,'_updateSprite'))return null;
    const original=pyramid._updateSprite,viewed=[];
    Object.defineProperty(pyramid,'_updateSprite',{configurable:true,writable:true,
      value(node){viewed.push(node);return original.call(this,node);}});
    try{
      pyramid._invalidateSprites();
      pyramid.updateView(OVERVIEW);
    }finally{
      delete pyramid._updateSprite;
      pyramid.updateView(position);
    }
    const roots=[...new Set(viewed)],parents=new WeakMap(),nodes=[],stack=[...roots];
    if(!roots.length||roots.some(root=>root?.level!==0))return null;
    while(stack.length){
      const node=stack.pop();
      if(!validNode(node)||node.layers.length!==roots[0].layers.length)return null;
      nodes.push(node);
      for(const child of node.children){
        if(parents.has(child))return null;
        parents.set(child,node);stack.push(child);
      }
    }
    // Unloaded layers must show their nearest loaded ancestor, as the pyramid derives them.
    for(const node of nodes)for(let layer=0;layer<node.layers.length;layer++){
      if(node.layers[layer].texture)continue;
      const shown=node.uniformGroup.uniforms[`sampler${layer}`],ancestor=nearest(parents,node,layer);
      const expected=ancestor?ancestor.layers[layer].texture.baseTexture:null;
      if(expected?(shown?.baseTexture!==expected):(shown!==EMPTY))return null;
    }
    return {pyramid,parents,nodes,sprites:new Set(nodes.map(node=>node.sprite)),lastSeen:new WeakMap()};
  }

  function nearest(parents,node,layer){
    for(let ancestor=parents.get(node);ancestor;ancestor=parents.get(ancestor))if(ancestor.layers[layer].texture)return ancestor;
    return null;
  }

  function derive(ancestor,node,layer){
    if(!ancestor)return EMPTY;
    const base=ancestor.layers[layer].texture.baseTexture,resolution=base.width/ancestor.rect.width;
    return new PIXI.Texture(base,new PIXI.Rectangle((node.rect.x-ancestor.rect.x)*resolution,(node.rect.y-ancestor.rect.y)*resolution,
      node.rect.width*resolution,node.rect.height*resolution));
  }

  function show(node,layer,texture){
    const key=`sampler${layer}`,previous=node.uniformGroup.uniforms[key];
    node.uniformGroup.uniforms[key]=texture;
    if(layer===0)node.sprite.texture=texture;
    return previous;
  }

  // Descendants without their own layer drew from the evicted tile; point them at
  // its nearest loaded ancestor instead, stopping at tiles that have their own.
  function rederive(node,ancestor,layer){
    if(node.layers[layer].texture)return;
    const previous=show(node,layer,derive(ancestor,node,layer));
    if(previous!==EMPTY)previous?.destroy?.();
    for(const child of node.children)rederive(child,ancestor,layer);
  }

  function size(base){
    const buffers=base?.resource?._levelBuffers;
    if(Array.isArray(buffers)&&buffers.length)return buffers.reduce((sum,level)=>sum+(level?.levelBuffer?.byteLength??0),0);
    return (base?.realWidth??0)*(base?.realHeight??0)*4;
  }

  const own=node=>node.layers.filter(layer=>layer.texture);
  const loading=node=>node.loaded&&node.layers.some(layer=>layer.src&&!layer.texture);

  // Returns the released layers; their files are unloaded once no tile holds them.
  function evict(tree,node){
    const released=[];
    for(let layer=0;layer<node.layers.length;layer++){
      const entry=node.layers[layer],texture=entry.texture;
      if(!texture)continue;
      const ancestor=nearest(tree.parents,node,layer);
      released.push({src:entry.src,base:texture.baseTexture});
      show(node,layer,derive(ancestor,node,layer));
      entry.texture=null;
      for(const child of node.children)rederive(child,ancestor,layer);
      texture.destroy();
    }
    node.loaded=false;
    tree.lastSeen.delete(node);
    return released;
  }

  function scan(now=Date.now()){
    const canvas=runtime.canvas;
    if(!canvas?.ready||!canvas.primary?.children)return;
    const live=[];
    for(const child of canvas.primary.children){
      if(!isPyramid(child))continue;
      let tree=trees.get(child);
      if(tree===undefined){
        const position=viewPosition();
        if(!position)return;
        tree=capture(child,position);
        trees.set(child,tree);
        if(!tree){
          detail='A sprite pyramid no longer has the node structure this repair expects.';
          return finish('unsupported-source');
        }
      }
      if(!tree)return;
      // Every shown tile must be one of the captured nodes; a pyramid that rebuilt
      // its tiles in place is no longer the tree the budget counts.
      if(!child.children.every(sprite=>tree.sprites.has(sprite))){
        trees.set(child,null);
        detail='A sprite pyramid replaced its tiles after they were read.';
        return finish('unsupported-source');
      }
      live.push(tree);
    }
    stats.pyramids=live.length;
    // A scene whose pyramids all match again, after one that did not.
    if(status==='unsupported-source'){detail=undefined;finish('installed');}
    if(!live.length){stats.loadedMB=0;return;}

    let bytes=0;
    // Pyramids can share tile files (the shore backdrop draws the shore's second layer).
    const counted=new Set(),references=new Map(),pinned=new Set(),candidates=[];
    for(const tree of live){
      let latest=-Infinity;
      for(const node of tree.nodes)latest=Math.max(latest,node.touchedId);
      // In view, or in the latest view's preload ring, together with every ancestor
      // a visible tile may be drawing from.
      const kept=new Set();
      for(const node of tree.nodes){
        if(node.sprite.parent!==tree.pyramid&&node.touchedId!==latest)continue;
        for(let current=node;current&&!kept.has(current);current=tree.parents.get(current))kept.add(current);
      }
      for(const node of tree.nodes){
        const textures=own(node);
        for(const {texture} of textures){
          const base=texture.baseTexture;
          references.set(base,(references.get(base)??0)+1);
          if(!counted.has(base)){counted.add(base);bytes+=size(base);}
          if(kept.has(node))pinned.add(base);
        }
        if(kept.has(node)||(textures.length&&!tree.lastSeen.has(node)))tree.lastSeen.set(node,now);
        if(!textures.length||kept.has(node)||!tree.parents.has(node)||loading(node))continue;
        if(now-tree.lastSeen.get(node)>=config.idleMs)candidates.push({tree,node});
      }
    }
    if(bytes>config.highBytes){
      candidates.sort((a,b)=>a.tree.lastSeen.get(a.node)-b.tree.lastSeen.get(b.node));
      for(const {tree,node} of candidates){
        if(bytes<=config.lowBytes)break;
        // Nothing would be freed while a tile in view holds the same files.
        if(own(node).every(layer=>pinned.has(layer.texture.baseTexture)))continue;
        stats.evictedTiles++;
        for(const {src,base} of evict(tree,node)){
          const remaining=references.get(base)-1;
          references.set(base,remaining);
          if(remaining>0)continue;
          bytes-=size(base);
          if(!src)continue;
          stats.unloaded++;
          const key=cacheKey(src);
          Promise.resolve(PIXI.Assets.unload(key)).catch(error=>console.warn(`av-v14-hotfix | bobTileCache could not unload ${key}`,error));
        }
      }
    }
    stats.loadedMB=Math.round(bytes/MiB);
  }

  const timer=runtime.setInterval(()=>{
    try{scan();}
    catch(error){
      runtime.clearInterval(timer);
      detail=String(error);
      finish('failed');
    }
  },config.intervalMs);
  return Object.assign(finish('installed'),{scan});
}
