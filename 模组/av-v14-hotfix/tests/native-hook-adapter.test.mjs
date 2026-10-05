import test from 'node:test';
import assert from 'node:assert/strict';
import {nativeHooks} from './hook-harness.mjs';
import {adaptNativeHooks} from '../scripts/native-hook-adapter.mjs';

function install(Hooks, callbacks) {
  const result=adaptNativeHooks({Hooks,callbacks});
  assert.equal(result.status,'installed',JSON.stringify(result));
  return result;
}
const adaptation=(hook,fn,tag,seen)=>({hook,source:fn.toString(),wrap:original=>function(...args){seen?.push(tag);return Reflect.apply(original,this,args);}});

test('two features share native records and removal mapping in either install and restore order',()=>{
  for(const order of [false,true])for(const restoreOrder of [false,true]) {
    const Hooks=nativeHooks(),seen=[];
    function left(...args){seen.push(['left',this.id,...args]);return false;}
    function right(...args){seen.push(['right',this.id,...args]);return 'right';}
    const ids=[Hooks.on('refreshToken',left),Hooks.on('refreshToken',right)];
    const list=Hooks.events.refreshToken,records=[...list],nativeOff=Object.getOwnPropertyDescriptor(Hooks,'off');
    const callbacks=[adaptation('refreshToken',left,'L',seen),adaptation('refreshToken',right,'R',seen)];
    const results=[];for(const i of order?[1,0]:[0,1])results[i]=install(Hooks,[callbacks[i]]);
    assert.equal(Hooks.events.refreshToken,list);assert.deepEqual([...list],records);
    assert.equal(Hooks.call('refreshToken','token',17),false);
    assert.deepEqual(seen,['L',['left',ids[0],'token',17]]);
    const first=restoreOrder?1:0,second=1-first;results[first].restore();
    assert.equal(records[first].fn,first?right:left);assert.notEqual(Hooks.off,nativeOff.value);
    Hooks.off('refreshToken',second?right:left);assert.equal(list.length,1);assert.equal(list[0],records[first]);
    results[second].restore();assert.deepEqual(Object.getOwnPropertyDescriptor(Hooks,'off'),nativeOff);
    results[first].restore();results[second].restore();assert.equal(list.length,1);
  }
});

test('original, replacement and numeric selectors retain first-match order across groups',()=>{
  for(const selector of ['original','replacement','id']) {
    const Hooks=nativeHooks();function first(){}function second(){}
    Hooks.on('refreshToken',first);Hooks.on('refreshToken',second);
    const firstRecord=Hooks.events.refreshToken[0];
    const a=install(Hooks,[adaptation('refreshToken',first)]),b=install(Hooks,[adaptation('refreshToken',second)]);
    const later=Hooks.on('refreshToken',first);
    Hooks.off(selector==='id'?'ignored':'refreshToken',selector==='id'?firstRecord.id:selector==='replacement'?firstRecord.fn:first);
    assert.deepEqual(Array.from(Hooks.events.refreshToken,e=>e.id),[2,later]);
    Hooks.off('refreshToken',first);assert.deepEqual(Array.from(Hooks.events.refreshToken,e=>e.id),[2]);
    a.restore();b.restore();assert.equal(Hooks.events.refreshToken.length,1);
  }
});

test('restore repairs removed in-flight records without resurrection or touching another feature',()=>{
  const Hooks=nativeHooks(),seen=[];let a,b;
  function first(){seen.push('first');}function second(){seen.push('second');}
  Hooks.once('refreshToken',()=>{Hooks.off('refreshToken',first);a.restore();});
  Hooks.on('refreshToken',first);Hooks.on('refreshToken',second);
  a=install(Hooks,[adaptation('refreshToken',first,'adapted-first',seen)]);
  b=install(Hooks,[adaptation('refreshToken',second,'adapted-second',seen)]);
  Hooks.callAll('refreshToken');assert.deepEqual(seen,['first','adapted-second','second']);
  assert.equal(Hooks.events.refreshToken.length,1);b.restore();assert.equal(Hooks.events.refreshToken[0].fn,second);
});

test('later off wrappers and later callback edits survive separate restores',()=>{
  const Hooks=nativeHooks();function first(){}function second(){}function other(){}
  Hooks.on('refreshToken',first);Hooks.on('refreshToken',second);
  const a=install(Hooks,[adaptation('refreshToken',first)]),b=install(Hooks,[adaptation('refreshToken',second)]),off=Hooks.off;
  let calls=0;Hooks.off=function(...args){calls++;return Reflect.apply(off,this,args);};const later=Hooks.off;
  Hooks.events.refreshToken[0].fn=other;a.restore();assert.equal(Hooks.events.refreshToken[0].fn,other);
  Hooks.off('refreshToken',second);assert.equal(Hooks.events.refreshToken.length,1);
  b.restore();assert.equal(Hooks.off,later);Hooks.off('refreshToken',other);assert.equal(calls,2);
  assert.equal(Hooks.events.refreshToken.length,0);
});

test('foreign core methods, duplicate sources and invalid records fail before any group mutation',()=>{
  for(const kind of ['off','call','duplicate','once','frozen','missing']) {
    const Hooks=nativeHooks();function first(){}function second(){}
    Hooks.on('refreshToken',first);Hooks.on('refreshToken',second);
    if(kind==='off')Hooks.off=function(){};
    if(kind==='call')Hooks.call=function(){};
    if(kind==='duplicate')Hooks.on('refreshToken',second);
    if(kind==='once')Hooks.events.refreshToken[1].once=true;
    if(kind==='frozen')Object.freeze(Hooks.events.refreshToken[1]);
    if(kind==='missing')Hooks.off('refreshToken',second);
    const entries=[...Hooks.events.refreshToken],fns=entries.map(e=>e.fn),off=Hooks.off;
    const result=adaptNativeHooks({Hooks,callbacks:[adaptation('refreshToken',first),adaptation('refreshToken',second)]});
    assert.equal(result.status,'skipped',kind);assert.equal(Hooks.off,off);assert.deepEqual(entries.map(e=>e.fn),fns);
  }
});

test('failed second-group installation restores only its partial changes',()=>{
  const Hooks=nativeHooks();function first(){}function second(){}function third(){}
  Hooks.on('refreshToken',first);Hooks.on('refreshToken',second);Hooks.on('refreshToken',third);
  const a=install(Hooks,[adaptation('refreshToken',first)]),ownedOff=Hooks.off,ownedFn=Hooks.events.refreshToken[0].fn;
  const thirdRecord=Hooks.events.refreshToken[2];
  const fail=adaptation('refreshToken',third);fail.wrap=original=>{Object.freeze(thirdRecord);return (...args)=>original(...args);};
  const result=adaptNativeHooks({Hooks,callbacks:[adaptation('refreshToken',second),fail]});
  assert.equal(result.status,'skipped');assert.equal(Hooks.off,ownedOff);assert.equal(Hooks.events.refreshToken[0].fn,ownedFn);
  assert.equal(Hooks.events.refreshToken[1].fn,second);assert.equal(Hooks.events.refreshToken[2].fn,third);
  Hooks.off('refreshToken',first);assert.equal(Hooks.events.refreshToken.length,2);a.restore();
});

test('fallback return, rejection object, exception and receiver are passed through unchanged',async()=>{
  const Hooks=nativeHooks(),error=new Error('original error'),promise=Promise.reject(error);promise.catch(()=>{});
  function native(value){if(value==='throw')throw error;return value==='promise'?promise:this;}
  Hooks.on('refreshToken',native);const entry=Hooks.events.refreshToken[0];
  install(Hooks,[adaptation('refreshToken',native)]);
  assert.equal(entry.fn('promise'),promise);await assert.rejects(entry.fn('promise'),e=>e===error);
  assert.throws(()=>entry.fn('throw'),e=>e===error);assert.equal(entry.fn(),entry);
  assert.throws(()=>Reflect.apply(Hooks.off,{},['refreshToken',native]),error=>error.name==='TypeError');
});

test('unrelated off selectors keep native accessor reads and errors',()=>{
  const Hooks=nativeHooks();function first(){}function unrelated(){}
  Hooks.on('refreshToken',first);Hooks.on('refreshToken',unrelated);
  let reads=0;Object.defineProperty(Hooks.events.refreshToken[1],'fn',{get(){reads++;return unrelated;}});
  install(Hooks,[adaptation('refreshToken',first)]);reads=0;
  Hooks.off('refreshToken',unrelated);assert.equal(reads,1);assert.equal(Hooks.events.refreshToken.length,1);
});

test('later changes to the owned off descriptor are not overwritten by another feature',()=>{
  const Hooks=nativeHooks();function first(){}function second(){}
  Hooks.on('refreshToken',first);Hooks.on('refreshToken',second);
  const installed=install(Hooks,[adaptation('refreshToken',first)]);
  const descriptor=Object.getOwnPropertyDescriptor(Hooks,'off');
  Object.defineProperty(Hooks,'off',{...descriptor,enumerable:!descriptor.enumerable});
  const changed=Object.getOwnPropertyDescriptor(Hooks,'off');
  const result=adaptNativeHooks({Hooks,callbacks:[adaptation('refreshToken',second)]});
  assert.equal(result.status,'skipped');assert.deepEqual(Object.getOwnPropertyDescriptor(Hooks,'off'),changed);
  assert.equal(Hooks.events.refreshToken[1].fn,second);
  installed.restore();assert.deepEqual(Object.getOwnPropertyDescriptor(Hooks,'off'),changed);
  Hooks.off('refreshToken',first);assert.equal(Hooks.events.refreshToken.length,1);
});

test('frozen first-feature records do not prevent either restore from retiring the broker',()=>{
  for(const order of [[0,1],[1,0]]) {
    const Hooks=nativeHooks();function first(){}function second(){}
    Hooks.on('refreshToken',first);Hooks.on('refreshToken',second);
    const nativeOff=Object.getOwnPropertyDescriptor(Hooks,'off'),entries=[...Hooks.events.refreshToken];
    const results=[install(Hooks,[adaptation('refreshToken',first)]),install(Hooks,[adaptation('refreshToken',second)])];
    Object.freeze(entries[0]);const frozen=Object.getOwnPropertyDescriptor(entries[0],'fn');
    for(const index of order)assert.doesNotThrow(()=>results[index].restore());
    assert.deepEqual(Object.getOwnPropertyDescriptor(entries[0],'fn'),frozen);
    assert.equal(entries[1].fn,second);assert.deepEqual(Object.getOwnPropertyDescriptor(Hooks,'off'),nativeOff);
    Hooks.off('refreshToken',entries[0].id);Hooks.off('refreshToken',second);assert.equal(Hooks.events.refreshToken.length,0);
  }
});

test('later writable and enumerable callback edits survive restore and retire their aliases',()=>{
  for(const changed of [{enumerable:false},{writable:false}]) {
    const Hooks=nativeHooks();function first(){}
    Hooks.on('refreshToken',first);const entry=Hooks.events.refreshToken[0],nativeOff=Hooks.off;
    const result=install(Hooks,[adaptation('refreshToken',first)]);
    Object.defineProperty(entry,'fn',changed);const descriptor=Object.getOwnPropertyDescriptor(entry,'fn');
    result.restore();assert.deepEqual(Object.getOwnPropertyDescriptor(entry,'fn'),descriptor);
    assert.equal(Hooks.off,nativeOff);Hooks.off('refreshToken',descriptor.value);assert.equal(Hooks.events.refreshToken.length,0);
  }
});
