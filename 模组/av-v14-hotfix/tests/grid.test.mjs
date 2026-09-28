import test from 'node:test';
import assert from 'node:assert/strict';
import {adaptGridToken,installGrid,GRID_HASHES} from '../scripts/patches/grid.mjs';
test('square distance and disabled flanking delegate to native with original receiver/options',()=>{
  const calls=[],opts={reach:5};
  const token={__proto__:{distanceTo(){throw Error('broken distance');},onOppositeSides(){throw Error('broken flank');}}};
  const native={distanceTo(...args){calls.push([this,args]);return 10;},onOppositeSides(){return false;}};
  const g={canvas:{ready:true,grid:{isSquare:true}},game:{settings:{get:()=>false}}};
  assert.equal(adaptGridToken(token,native,g),true);
  assert.equal(token.distanceTo('target',opts),10);assert.equal(calls[0][0],token);assert.equal(calls[0][1][1],opts);
  assert.equal(token.onOppositeSides(1,2,3),false);
});
test('explicit override, collision aura and inactive canvas keep existing behavior',()=>{
  const token={__proto__:{distanceTo:()=>99,onOppositeSides:()=>true}};
  const g={canvas:{ready:true,grid:{isSquare:true}},game:{settings:{get:()=>true}}};
  adaptGridToken(token,{distanceTo:()=>10,onOppositeSides:()=>false},g);
  assert.equal(token.distanceTo(null,{collision_types:['sight']}),99);
  assert.equal(token.onOppositeSides(),true);
  g.canvas.ready=false;assert.equal(token.distanceTo(),99);
});
test('existing instance overrides are never replaced; repeat installation is harmless',()=>{
  const token={distanceTo:()=>7},original=token.distanceTo;
  assert.equal(adaptGridToken(token,{distanceTo(){},onOppositeSides(){}}),false);
  assert.equal(token.distanceTo,original);
});
test('a competing wrapper registered while source hashing awaits prevents installation',async()=>{
  const native={distanceTo(){},onOppositeSides(){},conflicts:[]},reports=[],hooks=[];
  const token={__proto__:{distanceTo:()=>99,onOppositeSides:()=>true}};
  const g={game:{modules:new Map([['f2e-grid-enhancements',{active:true,version:'2.2.0'}]]),system:{version:'8.5.0'}},canvas:{tokens:{placeables:[token]}},Hooks:{on:(...a)=>hooks.push(a)}};
  let calls=0;
  await installGrid({g,native,hash:async()=>{await Promise.resolve();native.conflicts.push('another-module');return Object.values(GRID_HASHES)[calls++];},report:(...a)=>reports.push(a)});
  assert.equal(reports.at(-1)[1],'disabled-conflict');assert.equal(Object.hasOwn(token,'distanceTo'),false);
});
