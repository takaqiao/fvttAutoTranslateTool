// A stand-in for a Bastion of Blasphemies sprite pyramid, written from the
// SpritePyramidNode typedef and the pyramid's observable behaviour rather than
// from its code: a three-level quadtree of tiles over an 8000×4000 map with
// private roots; a view that shows the tiles of one level and marks a preload
// ring around them; and tiles without a file of their own for a layer showing
// the matching region of the nearest tile above them that has one.
export const MiB=1024**2;
const WIDTH=8000,HEIGHT=4000,FINEST=2,TILE=1000,WINDOW=1000;

export function pixiEnvironment(){
  const bases=new Map(),unloaded=[];
  class Rectangle {
    constructor(x=0,y=0,width=0,height=0){Object.assign(this,{x,y,width,height});}
    intersects(other){
      return this.x<other.x+other.width&&other.x<this.x+this.width&&this.y<other.y+other.height&&other.y<this.y+this.height;
    }
  }
  class BaseTexture {
    constructor(src,bytes){this.src=src;this.width=1000;this.height=1000;this.realWidth=1000;this.realHeight=1000;
      this.resource={_levelBuffers:[{levelBuffer:{byteLength:bytes}}]};this.destroyed=false;}
    destroy(){this.destroyed=true;this.resource=null;}
  }
  class Texture {
    constructor(baseTexture,frame){this.baseTexture=baseTexture;this.frame=frame;this.destroyed=false;}
    destroy(){this.destroyed=true;}
  }
  Texture.EMPTY=new Texture(null);
  Texture.EMPTY.destroy=()=>{};
  const Assets={unload(src){unloaded.push(src);bases.get(src)?.destroy();bases.delete(src);return Promise.resolve();}};
  return {PIXI:{Rectangle,Texture,Assets},BaseTexture,bases,unloaded};
}

export function createPyramid(pixi,{layers=2,bytes=4*MiB,prefix='tiles'}={}){
  const {PIXI:{Rectangle,Texture},BaseTexture,bases}=pixi,EMPTY=Texture.EMPTY;
  const side=level=>TILE*2**(FINEST-level);
  const id=(level,row,column)=>`${level}/${row}/${column}`;

  function tile(level,row,column){
    const size=side(level),indexes=Array.from({length:layers},(_,index)=>index);
    return {level,row,column,rect:new Rectangle(column*size,row*size,size,size),
      layers:indexes.map(index=>({src:`${prefix}/${level}_${row}_${column}_${index}.ktx2`,fileSize:1,texture:null})),
      uniformGroup:{uniforms:Object.fromEntries(indexes.map(index=>[`sampler${index}`,EMPTY]))},
      sprite:{texture:EMPTY,parent:null},spriteId:-1,touchedId:-1,touchedTime:0,loaded:false,children:[]};
  }

  // The part of a file above a tile that covers the tile, in the file's pixels.
  function regionFor(holder,node,base){
    const pixels=base.width/holder.rect.width;
    return new Rectangle((node.rect.x-holder.rect.x)*pixels,(node.rect.y-holder.rect.y)*pixels,
      node.rect.width*pixels,node.rect.height*pixels);
  }

  class Pyramid {
    #roots;#tiles=new Map();#above=new Map();#generation=0;#views=0;
    constructor(){
      this.children=[];this.destroyed=false;this.updates=0;
      for(let level=0;level<=FINEST;level++){
        const size=side(level);
        for(let row=0;row<HEIGHT/size;row++)for(let column=0;column<WIDTH/size;column++){
          const node=tile(level,row,column),parent=this.#tiles.get(id(level-1,Math.floor(row/2),Math.floor(column/2)));
          this.#tiles.set(id(level,row,column),node);
          if(parent){parent.children.push(node);this.#above.set(node,parent);}
        }
      }
      this.#roots=this.allNodes.filter(node=>node.level===0);
    }
    // Test access only; the real class keeps its tiles private.
    get allNodes(){return [...this.#tiles.values()];}
    node(level,row,column){return this.#tiles.get(id(level,row,column));}
    removeChildren(){for(const child of this.children)child.parent=null;this.children.length=0;}
    addChild(sprite){sprite.parent=this;this.children.push(sprite);}

    // Full size shows the finest level; each halving of the zoom one level coarser.
    #levelFor(scale){
      let level=FINEST;
      for(let zoom=scale;level>0&&zoom<1;zoom*=2)level--;
      return level;
    }
    #around(level,{x,y},width){
      const area=new Rectangle(x-width/2,y-width/2,width,width);
      if(level===0)return this.#roots.filter(node=>node.rect.intersects(area));
      return this.allNodes.filter(node=>node.level===level&&node.rect.intersects(area));
    }
    updateView(position){
      this.updates++;this.#views++;
      const level=this.#levelFor(position.scale),width=WINDOW/position.scale;
      this.removeChildren();
      for(const node of this.#around(level,position,width)){
        if(node.spriteId!==this.#generation){node.spriteId=this.#generation;this._updateSprite(node);}
        this.addChild(node.sprite);
        node.touchedId=this.#views;
      }
      // The preload ring: twice the view at this level, and four times it one level coarser.
      const ring=this.#around(level,position,width*2);
      if(level>0)ring.push(...this.#around(level-1,position,width*4));
      for(const node of ring)node.touchedId=this.#views;
    }
    _invalidateSprites(){this.#generation++;}
    _updateSprite(node){node.sprite.updated=(node.sprite.updated??0)+1;}
    _loadPendingNodes(){}
    *getTexturesToPreload(){for(const node of this.allNodes)for(const layer of node.layers)if(layer.texture)yield layer.src;}

    // A tile's files arriving. A file still in the texture cache comes back as a
    // new Texture on the cached base.
    load(node,only){
      node.loaded=true;
      node.layers.forEach((layer,index)=>{
        if(!layer.src||layer.texture||(only&&!only.includes(index)))return;
        if(!bases.has(layer.src))bases.set(layer.src,new BaseTexture(layer.src,bytes));
        layer.texture=new Texture(bases.get(layer.src));
        this.#display(node,index);
      });
    }
    // What a tile shows for a layer: its own file, else a region of the nearest
    // file above it, else nothing. Tiles below it without their own file follow.
    #display(node,index){
      const own=node.layers[index].texture;
      let holder=null;
      for(let above=this.#above.get(node);!own&&above&&!holder;above=this.#above.get(above))if(above.layers[index].texture)holder=above;
      const base=holder?.layers[index].texture.baseTexture;
      this.#put(node,index,own??(holder?new Texture(base,regionFor(holder,node,base)):EMPTY));
      for(const child of node.children)if(!child.layers[index].texture)this.#display(child,index);
    }
    #put(node,index,texture){
      const name=`sampler${index}`,shown=node.uniformGroup.uniforms[name];
      // Region textures belong to the tile showing them.
      if(shown!==texture&&shown!==EMPTY&&shown!==node.layers[index].texture)shown.destroy();
      node.uniformGroup.uniforms[name]=texture;
      if(index===0)node.sprite.texture=texture;
    }
    destroy(){this.destroyed=true;}
  }
  return new Pyramid();
}

export function tileEnvironment({active=true,layers,bytes}={}){
  const pixi=pixiEnvironment(),pyramid=createPyramid(pixi,{layers,bytes}),intervals=[];
  const scene={_viewPosition:{x:500,y:500,scale:1,level:'level'}};
  const runtime={PIXI:pixi.PIXI,game:{modules:new Map([['pf2e-bastion-of-blasphemies',{active}]])},
    canvas:{ready:true,scene,primary:{children:[{destroyed:false},pyramid]}},
    setInterval:(fn,ms)=>{intervals.push({fn,ms});return intervals.length;},clearInterval:id=>{intervals[id-1].cleared=true;}};
  const view=position=>{scene._viewPosition={...position,level:'level'};pyramid.updateView(position);};
  return {...pixi,pyramid,runtime,scene,intervals,view};
}
