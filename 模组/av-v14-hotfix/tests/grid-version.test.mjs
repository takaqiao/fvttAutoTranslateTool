import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
import {captureGridNative,installGrid} from '../scripts/patches/grid.mjs';

const fixture=JSON.parse(readFileSync(new URL('./fixtures/grid-native-pf2e-8.5.json',import.meta.url),'utf8'));
const sha=source=>createHash('sha256').update(source).digest('hex');
class Rectangle{
  constructor(x,y,width,height){Object.assign(this,{x,y,width,height});}
  get left(){return this.x;}get right(){return this.x+this.width;}
  get top(){return this.y;}get bottom(){return this.y+this.height;}
}
function setup(version,sourceVersion=version){
  const hooks=new Map(),reports=[];
  let override=false;
  const g={
    game:{system:{version},modules:new Map([['f2e-grid-enhancements',{active:true,version:'2.2.0'}]]),settings:{get:()=>override}},
    canvas:{ready:true,grid:{type:1,isSquare:true,sizeX:150,sizeY:150},dimensions:{size:150,distance:5},tokens:{placeables:[]}},
    Hooks:{on(name,callback){const list=hooks.get(name)??[];list.push(callback);hooks.set(name,list);}},
    foundry:{canvas:{geometry:{Ray:class{constructor(A,B){Object.assign(this,{A,B});}}}},utils:{}},
    PIXI:{Rectangle},CONST:{GRID_TYPES:{SQUARE:1}}
  };
  const context=vm.createContext(g);
  for(const [name,source] of Object.entries(fixture.intersectionHelpers))g.foundry.utils[name]=vm.runInContext(`(${source})`,context);
  vm.runInContext(fixture.distanceHelpers,context);
  const nativeMethods=vm.runInContext(`({${Object.values(fixture.versions[sourceVersion]).map(m=>m.source).join(',')}})`,context);
  for(const [name,method] of Object.entries(nativeMethods))assert.equal(sha(method.toString()),fixture.versions[sourceVersion][name].sha256);
  class Token{}
  Object.assign(Token.prototype,nativeMethods);
  g.CONFIG={Token:{objectClass:Token}};
  const native=captureGridNative(g);
  // Simulate Grid registering later: capture must keep the original method identities.
  Object.assign(Token.prototype,{distanceTo:()=>99,onOppositeSides:()=> 'grid-fallback'});
  const token=new Token();g.canvas.tokens.placeables.push(token);
  return {g,native,nativeMethods,token,reports,hooks,setOverride(value){override=value;},
    install:()=>installGrid({g,native,hash:async source=>sha(source),report:(...r)=>reports.push(r)}),
    emit(name,...args){for(const callback of hooks.get(name)??[])callback(...args);}};
}
function pointToken(x,y,{tiny=false}={}){
  return {document:{x:x*150,y:y*150,elevation:0},actor:{},
    mechanicalBounds:new Rectangle((tiny?Math.floor(x):x)*150,(tiny?Math.floor(y):y)*150,150,150)};
}
// Independent literal geometry expectations: Tiny visual positions and mechanical squares differ.
const tinyCases=[
  {name:'corrected false positive',a:[-1.5,1.5],b:[1,-1],want:{'8.5.0':true,'8.5.1':false}},
  {name:'corrected false negative',a:[-2,2.5],b:[1,-1],want:{'8.5.0':false,'8.5.1':true}}
];
for(const version of ['8.5.0','8.5.1']){
  test(`${version} installation hashes genuine release methods and retains native Tiny geometry`,async()=>{
    const h=setup(version);await h.install();
    assert.equal(h.reports.at(-1)[1],'installed');
    for(const c of tinyCases){
      const a=pointToken(...c.a,{tiny:true}),b=pointToken(...c.b),target=pointToken(0,0);
      assert.equal(h.token.onOppositeSides(a,b,target),c.want[version],c.name);
    }
    Object.assign(h.token,pointToken(-2,0));
    assert.equal(h.token.distanceTo(pointToken(0,0),{reach:5}),10,'native distance must not deduct reach');
    h.setOverride(true);
    assert.equal(h.token.onOppositeSides(),'grid-fallback','explicit Grid override retained');
    assert.equal(h.token.distanceTo(null,{collision_types:['sight']}),99,'aura collision branch retained');
  });
  test(`${version} cannot borrow the other version's onOppositeSides source`,async()=>{
    const h=setup(version,version==='8.5.0'?'8.5.1':'8.5.0');await h.install();
    assert.equal(h.reports.at(-1)[1],'unsupported-source');
    assert.equal(h.reports.at(-1)[2],'onOppositeSides');
    assert.equal(Object.hasOwn(h.token,'distanceTo'),false);
  });
  test(`${version} rejects a changed distance method and later competing wrapper disables delegation`,async()=>{
    const changed=setup(version);changed.native.distanceTo=function(){return 0;};await changed.install();
    assert.equal(changed.reports.at(-1)[1],'unsupported-source');
    assert.equal(changed.reports.at(-1)[2],'distanceTo');
    assert.equal(Object.hasOwn(changed.token,'distanceTo'),false);
    const h=setup(version);await h.install();
    assert.equal(h.reports.at(-1)[1],'installed');
    h.emit('libWrapper.Register','other-module','CONFIG.Token.objectClass.prototype.onOppositeSides');
    assert.equal(h.reports.at(-1)[1],'disabled-conflict');
    assert.equal(h.token.onOppositeSides(),'grid-fallback');
    assert.equal(h.token.distanceTo(),99);
  });
}
test('Grid 2.2.1 restores the audited PF2e distance and flanking paths',async()=>{
  const h=setup('8.5.1');h.g.game.modules.get('f2e-grid-enhancements').version='2.2.1';await h.install();
  assert.equal(h.reports.at(-1)[1],'installed');
  Object.assign(h.token,pointToken(-2,0));
  assert.equal(h.token.distanceTo(pointToken(0,0),{reach:5}),10);
  assert.equal(h.token.onOppositeSides(pointToken(-1,0),pointToken(-2,0),pointToken(0,0)),false);
  assert.equal(h.token.onOppositeSides(pointToken(-1,0),pointToken(1,0),pointToken(0,0)),true);
  h.setOverride(true);
  assert.equal(h.token.onOppositeSides(),'grid-fallback');
  assert.equal(h.token.distanceTo(null,{collision_types:['sight']}),99);
});

test('unknown PF2e versions and unaudited Grid versions remain uninstalled',async()=>{
  for(const version of ['8.5.2','8.6.0','__proto__','constructor']){
    const h=setup(version,'8.5.1');await h.install();
    assert.equal(h.reports.at(-1)[1],'unsupported-version');
    assert.equal(Object.hasOwn(h.token,'distanceTo'),false);
  }
  for(const version of ['2.2.3','2.3.1','__proto__','constructor']){
    const h=setup('8.5.1');h.g.game.modules.get('f2e-grid-enhancements').version=version;await h.install();
    assert.equal(h.reports.at(-1)[1],'unsupported-version');
    assert.equal(Object.hasOwn(h.token,'distanceTo'),false);
  }
});
