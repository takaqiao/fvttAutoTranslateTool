import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import vm from 'node:vm';
import {captureGridNative,installGrid} from '../scripts/patches/grid.mjs';

const upstream=JSON.parse(readFileSync(new URL('./fixtures/grid-upstream-2.2.2.json',import.meta.url),'utf8'));
const upstream230=JSON.parse(readFileSync(new URL('./fixtures/grid-upstream-2.3.0.json',import.meta.url),'utf8'));
const pf2e=JSON.parse(readFileSync(new URL('./fixtures/grid-native-pf2e-8.5.json',import.meta.url),'utf8'));
const sha=source=>createHash('sha256').update(source).digest('hex');
class Rectangle{
  constructor(x,y,width,height){Object.assign(this,{x,y,width,height});}
  get left(){return this.x;}get right(){return this.x+this.width;}
  get top(){return this.y;}get bottom(){return this.y+this.height;}
}

function setup(release=upstream){
  const upstream=release;
  const callbacks=new Map(),reports=[],registrations=[],timers=[],fetches=[],settings=new Map();
  let nextId=1,blocked=false;
  const Hooks={
    on(name,fn){const id=nextId++;callbacks.set(id,{name,fn,once:false});return id;},
    once(name,fn){const id=nextId++;callbacks.set(id,{name,fn,once:true});return id;},
    off(name,id){if(callbacks.get(id)?.name===name)callbacks.delete(id);},
    emit(name,...args){for(const[id,row]of [...callbacks])if(row.name===name){if(row.once)callbacks.delete(id);row.fn(...args);}}
  };
  const g={Hooks,console:{debug(){}},queueMicrotask,
    game:{ready:false,system:{id:'pf2e',version:'8.5.1'},pf2e:{settings:{automation:{flanking:true}}},
      modules:new Map([['f2e-grid-enhancements',{active:true,version:upstream.provenance.gridVersion}]]),
      settings:{get:(_id,key)=>settings.get(key)??false,register(){}},i18n:{localize:s=>s}},
    canvas:{ready:true,tokens:{placeables:[]},grid:{type:1,isSquare:true,isHexagonal:false,isGridless:false,sizeX:100,sizeY:100,
      getCenterPoint:({i,j})=>({x:j*100+50,y:i*100+50}),
      measurePath:([a,b])=>{assert.equal(a.y,b.y,'horizontal geometry fixture');return{distance:Math.abs(b.x-a.x)/100*5};}},
      dimensions:{size:100,distance:5}},
    foundry:{utils:{debounce:fn=>(...args)=>{timers.push(()=>fn(...args));}},canvas:{geometry:{Ray:class{constructor(A,B){Object.assign(this,{A,B});}}}},
      data:{fields:{NumberField:class{},BooleanField:class{}}}},PIXI:{Rectangle},CONST:{GRID_TYPES:{SQUARE:1}}
  };
  const context=vm.createContext(g);
  for(const[name,source]of Object.entries(pf2e.intersectionHelpers))g.foundry.utils[name]=vm.runInContext(`(${source})`,context);
  vm.runInContext(pf2e.distanceHelpers,context);
  const methods=vm.runInContext(`({${Object.values(pf2e.versions['8.5.1']).map(m=>m.source).join(',')},${upstream.canFlank.source}})`,context);
  class Token{
    constructor(x,y=0){
      this.x=x;this.y=y;this.mechanicalBounds=new Rectangle(x,y,100,100);
      this.document={x,y,elevation:0,hidden:false,width:1,height:1,depth:1,scene:{grid:{size:100,distance:5}},
        center:{x:x+50,y:y+50},getOccupiedGridSpaceOffsets:()=>[{i:y/100,j:x/100}]};
      this.actor={attributes:{flanking:{canFlank:true,flankable:true}},canAttack:true,isOfType:type=>type==='creature',isAllyOf:()=>false,getReach:()=>5};
    }
  }
  Object.assign(Token.prototype,methods);
  class BaseLayer{placeRegion(){}_onDragLeftMove(){}}
  class RegionLayer extends BaseLayer{}
  class Region{snappingMode(){}}
  class Scene{canHaveAuras(){return true;}}
  g.CONFIG={Token:{objectClass:Token},Scene:{documentClass:Scene},Region:{objectClass:Region},
    Canvas:{layers:{regions:{layerClass:RegionLayer}},polygonBackends:{sight:{testCollision:()=>blocked}}}};
  g.foundry.canvas.sources={PointVisionSource:class{}};
  const native=captureGridNative(g);
  g.libWrapper={register(owner,path,fn,type){
    const parts=path.split('.'),key=parts.pop();let target=g;for(const part of parts)target=target[part];
    const previous=target[key];assert.equal(typeof previous,'function',path);
    const value=function(...args){return fn.call(this,previous.bind(this),...args);};
    Object.defineProperty(target,key,{value,configurable:true,writable:true});
    const id=nextId++;registrations.push({owner,path,type,fn,target,value});Hooks.emit('libWrapper.Register',owner,path,type,{},id);return id;
  }};
  for(const name of ['all','square','hex','gridless']){
    const source=upstream.files[`grid/${name}.js`].source;
    const exports=[...source.matchAll(/export function (\w+)/g)].map(m=>m[1]);
    g[name]=vm.runInContext(`(()=>{${source.replace(/^export /gm,'')};return{${exports.join(',')}};})()`,context);
  }
  vm.runInContext(upstream.files['main.js'].source.replace(/^import[^\n]*\n/gm,''),context);
  Hooks.emit('libWrapper.Ready');
  g.fetch=async url=>{const name=url.replace('modules/f2e-grid-enhancements/scripts/','');fetches.push(name);return{ok:!!upstream.files[name],text:async()=>upstream.files[name]?.source};};
  const a=new Token(0),b=new Token(200);g.canvas.tokens.placeables.push(a,b);
  function queueAura(){
    class TokenAura{constructor(token){this.token=token;this.radius=15;this.traits=['visual'];}}
    Object.assign(TokenAura.prototype,vm.runInContext(`({${upstream.containsToken.source}})`,context));
    class AuraRenderer{highlight(){}draw(){}}
    a.document.object=a;b.document.object=b;
    const aura=new TokenAura(a.document);
    a.auras=new Map([['test-aura',new AuraRenderer()]]);
    a.document.auras=new Map([['test-aura',aura]]);
    Hooks.emit('refreshToken',a,{});
    return aura;
  }
  return {g,native,reports,fetches,Token,a,b,registrations,timers,settings,
    queueAura,flushAura(){for(const run of timers.splice(0))run();},
    setBlocked(value){blocked=value;},
    install:()=>installGrid({g,native,hash:async source=>sha(source),report:(...r)=>reports.push(r)})};
}

test('actual Grid 2.2.2 distance subtraction causes PF2e to count reach twice',()=>{
  const h=setup();
  assert.equal(h.a.distanceTo(h.b),10);
  assert.equal(h.a.distanceTo(h.b,{reach:5}),5);
  assert.equal(h.a.canFlank(h.b,{reach:5}),true);
});

test('verified 2.2.2 restores native square reach without owning the fixed flanking method',async()=>{
  const h=setup(),flanking=h.a.onOppositeSides;await h.install();
  assert.equal(h.a.distanceTo(h.b,{reach:5}),10);
  assert.equal(h.a.canFlank(h.b,{reach:5}),false);
  assert.equal(Object.hasOwn(h.a,'onOppositeSides'),false);
  assert.equal(h.a.onOppositeSides,flanking);
  assert.equal(h.a.onOppositeSides(new h.Token(-100),new h.Token(100),h.a),true);
  assert.equal(h.a.onOppositeSides(new h.Token(-100),new h.Token(-200),h.a),false);
  assert.equal(h.reports.at(-1)[1],'installed');
});

test('2.2.2 repair preserves collision/aura calculations and their wall checks',async()=>{
  const h=setup();await h.install();
  assert.equal(h.a.distanceTo(h.b,{reach:5,collision_types:['sight']}),5);
  h.setBlocked(true);
  assert.equal(h.a.distanceTo(h.b,{reach:5,collision_types:['sight']}),Infinity);
  assert.equal(h.a.distanceTo(h.b,{reach:5}),10);
});

test('2.2.2 repair leaves non-square grids on the upstream distance path',async()=>{
  const h=setup();await h.install();h.g.canvas.grid.isSquare=false;h.g.canvas.grid.isHexagonal=true;
  assert.equal(h.a.distanceTo(h.b,{reach:5}),5);
});

for(const name of Object.keys(upstream.files))test(`changed 2.2.2 ${name} fails closed before adapting tokens`,async()=>{
  const h=setup(),fetch=h.g.fetch;
  h.g.fetch=async url=>{const r=await fetch(url);return url.endsWith('/'+name)?{ok:true,text:async()=>upstream.files[name].source+'\n// unreviewed change'}:r;};
  await h.install();
  assert.equal(h.reports.at(-1)[1],'unsupported-grid-source');
  assert.equal(Object.hasOwn(h.a,'distanceTo'),false);
});

test('unavailable 2.2.2 source keeps the module path and reports why',async()=>{
  const h=setup();h.g.fetch=async()=>({ok:false});await h.install();
  assert.equal(h.reports.at(-1)[1],'unsupported-grid-source');
  assert.equal(h.a.distanceTo(h.b,{reach:5}),5);
});

test('a module upgrade during 2.2.2 verification cannot install the stale adaptation',async()=>{
  const h=setup(),fetch=h.g.fetch;h.g.fetch=async url=>{const r=await fetch(url);h.g.game.modules.get('f2e-grid-enhancements').version='2.2.3';return r;};
  await h.install();assert.equal(h.reports.at(-1)[1],'source-changed-during-validation');assert.equal(Object.hasOwn(h.a,'distanceTo'),false);
});

test('newly drawn tokens receive only the verified distance repair',async()=>{
  const h=setup();await h.install();const next=new h.Token(0);h.g.Hooks.emit('drawToken',next);
  assert.equal(next.canFlank(h.b,{reach:5}),false);assert.equal(Object.hasOwn(next,'onOppositeSides'),false);
});

test('aura guard waits for Grid debounce registration and tolerates unrendered source and target',async()=>{
  const h=setup();await h.install();const aura=h.queueAura();
  assert.equal(h.g.CONFIG.F2e.Aura,undefined,'Grid has not yet created its Aura alias');
  h.flushAura();
  h.b.document.object=null;
  assert.equal(aura.containsToken(h.b.document),false);
  h.b.document.object=h.b;h.a.document.object=null;
  assert.equal(aura.containsToken(h.b.document),false);
});

test('aura guard also installs after Grid registered before source validation completed',async()=>{
  const h=setup(),aura=h.queueAura();h.flushAura();await h.install();h.b.document.object=null;
  assert.equal(aura.containsToken(h.b.document),false);
});

test('rendered aura targets retain original collision containment and hidden-token behavior',async()=>{
  const h=setup();await h.install();const aura=h.queueAura();h.flushAura();
  assert.equal(aura.containsToken(h.b.document),true);
  h.setBlocked(true);assert.equal(aura.containsToken(h.b.document),false);
  h.setBlocked(false);h.b.document.hidden=true;assert.equal(aura.containsToken(h.b.document),false);
  h.b.document.hidden=false;h.b.document.object=null;
  assert.equal(aura.containsToken(h.b.document),false);
  h.b.document.object=h.b;assert.equal(aura.containsToken(h.b.document),true);
});

test('a changed Grid bundle cannot authorize its later aura registration',async()=>{
  const h=setup();h.g.fetch=async()=>({ok:false});await h.install();const aura=h.queueAura();h.flushAura();h.b.document.object=null;
  assert.throws(()=>aura.containsToken(h.b.document));
  assert.equal(h.registrations.some(r=>r.owner==='av-v14-hotfix'),false);
});

test('a foreign aura wrapper disables the guard instead of masking its changed contract',async()=>{
  const h=setup();await h.install();const aura=h.queueAura();h.flushAura();
  h.g.libWrapper.register('other-module','CONFIG.F2e.Aura.token.containsToken',function(wrapped,...args){return wrapped(...args);},'WRAPPER');
  h.b.document.object=null;
  assert.throws(()=>aura.containsToken(h.b.document));
  assert.equal(h.reports.some(r=>r[0]==='gridAura'&&r[1]==='disabled-conflict'),true);
});

test('an upgrade before debounced aura registration cannot install the old guard',async()=>{
  const h=setup();await h.install();const aura=h.queueAura();h.g.game.modules.get('f2e-grid-enhancements').version='2.2.3';h.flushAura();
  h.b.document.object=null;assert.throws(()=>aura.containsToken(h.b.document));
  assert.equal(h.registrations.some(r=>r.owner==='av-v14-hotfix'),false);
  assert.equal(h.reports.some(r=>r[0]==='gridAura'&&r[1]==='source-changed'),true);
});

test('verified 2.3.0 retains PF2e reach distances on the actual Foundry square grid',async t=>{
  const app=process.env.FVTT_NATIVE_APP??'C:/Program Files/Foundry Virtual Tabletop/resources/app';
  let SquareGrid;
  try{
    await import(pathToFileURL(app+'/common/primitives/number.mjs'));
    await import(pathToFileURL(app+'/common/primitives/math.mjs'));
    ({default:SquareGrid}=await import(pathToFileURL(app+'/common/grid/square.mjs')));
  }catch(error){if(error.code==='ERR_MODULE_NOT_FOUND')return t.skip('Set FVTT_NATIVE_APP to the installed Foundry resources/app');throw error;}
  const h=setup(upstream230);h.g.canvas.grid=new SquareGrid({size:100,distance:5,diagonals:4});
  assert.equal(h.a.distanceTo(new h.Token(300,300),{reach:10}),20,'upstream differs numerically beyond reach');
  await h.install();assert.equal(h.reports.at(-1)[1],'installed');
  for(const c of [
    {x:200,y:0,reach:5,want:10},{x:400,y:0,reach:10,want:20},
    {x:200,y:200,reach:10,want:10},{x:200,y:200,reach:5,want:15},
    {x:300,y:300,reach:10,want:15}
  ])assert.equal(h.a.distanceTo(new h.Token(c.x,c.y),{reach:c.reach}),c.want,JSON.stringify(c));
  assert.equal(h.a.distanceTo(new h.Token(300,300),{reach:10,collision_types:['sight']}),20,'collision-aware distance remains upstream');
});

test('2.3.0 adapts only distance while retaining upstream native and custom flanking',async()=>{
  const h=setup(upstream230),flanking=h.a.onOppositeSides;
  const left=new h.Token(-100),right=new h.Token(100),sameSide=new h.Token(-200);
  h.settings.set('flanking-square-override',true);h.settings.set('flanking-angle',180);
  assert.equal(h.a.onOppositeSides(left,sameSide,h.a),true,'explicit upstream geometry differs from PF2e');
  h.settings.set('flanking-square-override',false);
  await h.install();assert.equal(h.reports.at(-1)[1],'installed');
  assert.equal(Object.hasOwn(h.a,'distanceTo'),true);
  assert.equal(Object.hasOwn(h.a,'onOppositeSides'),false);assert.equal(h.a.onOppositeSides,flanking);
  assert.equal(h.a.onOppositeSides(left,right,h.a),true);
  assert.equal(h.a.onOppositeSides(left,sameSide,h.a),false);
  h.settings.set('flanking-square-override',true);h.settings.set('flanking-angle',180);
  assert.equal(h.a.onOppositeSides(left,right,h.a),true);
  assert.equal(h.a.onOppositeSides(left,sameSide,h.a),true,'opt-in upstream geometry is preserved');
});

test('2.3.0 preserves collision checks and non-square upstream distance dispatch',async()=>{
  const h=setup(upstream230);await h.install();assert.equal(h.reports.at(-1)[1],'installed');
  assert.equal(h.a.distanceTo(h.b,{reach:5,collision_types:['sight']}),10);
  h.setBlocked(true);assert.equal(h.a.distanceTo(h.b,{reach:5,collision_types:['sight']}),Infinity);
  assert.equal(h.a.distanceTo(h.b,{reach:5}),10);
  for(const kind of ['isHexagonal','isGridless']){
    h.g.canvas.grid.isSquare=false;h.g.canvas.grid.isHexagonal=kind==='isHexagonal';h.g.canvas.grid.isGridless=kind==='isGridless';
    h.g.CONST.TOKEN_SHAPES={ELLIPSE_1:1,ELLIPSE_2:2,RECTANGLE_1:3,RECTANGLE_2:4};
    assert.equal(h.a.distanceTo(h.b,{reach:5,collision_types:['sight']}),Infinity);
    h.setBlocked(false);assert.equal(h.a.distanceTo(h.b,{reach:5}),10);h.setBlocked(true);
  }
});

for(const name of Object.keys(upstream230.files))test(`changed 2.3.0 ${name} cannot authorize distance or aura repairs`,async()=>{
  const h=setup(upstream230),fetch=h.g.fetch;
  h.g.fetch=async url=>{const r=await fetch(url);return url.endsWith('/'+name)?{ok:true,text:async()=>upstream230.files[name].source+'\n// unreviewed change'}:r;};
  await h.install();assert.equal(h.reports.at(-1)[1],'unsupported-grid-source');
  assert.equal(Object.hasOwn(h.a,'distanceTo'),false);
  h.queueAura();h.flushAura();assert.equal(h.registrations.some(r=>r.owner==='av-v14-hotfix'),false);
});

test('a module upgrade during 2.3.0 validation cannot install the old distance repair',async()=>{
  const h=setup(upstream230),fetch=h.g.fetch;
  h.g.fetch=async url=>{const r=await fetch(url);h.g.game.modules.get('f2e-grid-enhancements').version='2.3.1';return r;};
  await h.install();assert.equal(h.reports.at(-1)[1],'source-changed-during-validation');
  assert.equal(Object.hasOwn(h.a,'distanceTo'),false);
});

for(const timing of ['before','after'])test(`2.3.0 aura registration ${timing} validation protects undrawn source and target`,async()=>{
  const h=setup(upstream230),aura=h.queueAura();
  if(timing==='before')h.flushAura();await h.install();if(timing==='after')h.flushAura();
  h.b.document.object=null;assert.equal(aura.containsToken(h.b.document),false);
  h.b.document.object=h.b;h.a.document.object=null;assert.equal(aura.containsToken(h.b.document),false);
});

test('2.3.0 aura guard preserves real-distance radius boundaries, walls, self and hidden targets',async()=>{
  const h=setup(upstream230);await h.install();assert.equal(h.reports.at(-1)[1],'installed');
  const aura=h.queueAura();h.flushAura();
  aura.radius=9;assert.equal(aura.containsToken(h.b.document),false);
  aura.radius=10;assert.equal(aura.containsToken(h.b.document),true);
  assert.equal(aura.containsToken(h.a.document),true);
  h.setBlocked(true);assert.equal(aura.containsToken(h.b.document),false);
  h.setBlocked(false);h.b.document.hidden=true;assert.equal(aura.containsToken(h.b.document),false);
});

test('2.3.0 aura guard stops intervening after a module upgrade',async()=>{
  const h=setup(upstream230);await h.install();const aura=h.queueAura();h.flushAura();
  h.b.document.object=null;assert.equal(aura.containsToken(h.b.document),false);
  h.g.game.modules.get('f2e-grid-enhancements').version='2.3.1';
  assert.throws(()=>aura.containsToken(h.b.document));
});
