import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {calendariaEnvironment,TICK,WORLD} from './calendaria-hud-harness.mjs';
import {adaptNativeHooks} from '../scripts/native-hook-adapter.mjs';

const url=new URL('../scripts/patches/calendaria-hud.mjs',import.meta.url);
const install=existsSync(url)?(await import(url)).installCalendariaHudPatch:()=>({status:'missing'});
function patched(options){
  const env=calendariaEnvironment(options),reports=[];
  const result=install({runtime:env.runtime,report:report=>reports.push(report)});
  assert.equal(result.status,'installed',JSON.stringify(result));
  return {...env,result,reports};
}

test('equal nonempty text keeps the Text node while both original callbacks still update the sky',()=>{
  const env=patched();env.render();const time=env.hud.element.nodes['.time'],date=env.hud.element.nodes['.date'];
  const children=[time.firstChild,date.firstChild],sky=env.state.sky;
  env.tick(1,2);env.world(3);
  assert.deepEqual([time.firstChild,date.firstChild],children);assert.equal(time.writes+date.writes,0);
  assert.equal(env.state.sky,sky+2);assert.equal(env.result.stats.skipped,4);
  assert.equal(Object.hasOwn(time,'textContent'),false);assert.equal(Object.hasOwn(date,'textContent'),false);
});

test('changed, empty and multiple-child text writes retain native replacement behavior',()=>{
  const env=patched();env.render();const time=env.hud.element.nodes['.time'];
  env.state.time='10:00';env.tick();assert.equal(time.textContent,'10:00');assert.equal(time.writes,1);
  time.childNodes=[new env.Text('10:'),new env.Text('00')];env.tick();assert.equal(time.childNodes.length,1);assert.equal(time.writes,2);
  env.state.time='';env.tick();env.tick();assert.equal(time.childNodes.length,0);assert.equal(time.writes,4);
  time.childNodes=[new env.Element('icon')];env.state.time='11:00';env.tick();assert.equal(time.textContent,'11:00');assert.equal(time.writes,5);
});

test('HTML labels remain native and are never suppressed',()=>{
  const env=patched();env.render();env.state.time='<i>moon</i>';env.state.date='<b>Day 1</b>';
  env.tick();env.world();const nodes=env.hud.element.nodes;
  assert.equal(nodes['.time'].htmlWrites,2);assert.equal(nodes['.date'].htmlWrites,2);assert.equal(env.result.stats.skipped,0);
  assert.equal(Object.hasOwn(nodes['.time'],'innerHTML'),false);
});

test('date changes still schedule the native bar render',()=>{
  const env=patched();env.render();env.game.time.components.dayOfMonth=2;env.state.date='Day 2';
  env.world();env.tick();assert.equal(env.state.timers.length,1);env.state.timers[0]();
  assert.equal(JSON.stringify(env.state.renders),'[{"parts":["bar"]}]');assert.equal(env.hud.element.nodes['.date'].textContent,'Day 2');
});

test('callback exceptions clean up only our temporary descriptors',()=>{
  const env=patched();env.render();const time=env.hud.element.nodes['.time'],date=env.hud.element.nodes['.date'],error=new Error('format failed');
  env.state.format=()=>{throw error;};assert.throws(()=>env.tick(),e=>e===error);
  assert.equal(Object.hasOwn(time,'textContent'),false);assert.equal(Object.hasOwn(date,'textContent'),false);
  const foreign={get(){return 'foreign';},set(){},configurable:true};
  env.state.format=()=>{Object.defineProperty(time,'textContent',foreign);throw error;};assert.throws(()=>env.tick(),e=>e===error);
  assert.equal(Object.getOwnPropertyDescriptor(time,'textContent').get,foreign.get);
});

test('own, intermediate and nonextensible text accessors retain their ordinary setter',()=>{
  for(const mode of ['own','intermediate','sealed']){
    const env=patched();env.render();const time=env.hud.element.nodes['.time'];let writes=0;
    if(mode==='sealed')Object.preventExtensions(time);
    else {const owner=mode==='own'?time:Object.create(Object.getPrototypeOf(time));
      if(mode==='intermediate')Object.setPrototypeOf(time,owner);
      Object.defineProperty(owner,'textContent',{get(){return '09:00';},set(){writes++;},configurable:true});}
    env.tick();assert.equal(mode==='sealed'?time.writes:writes,1,mode);
  }
});

test('the current inherited setter is delegated after a third-party Node patch',()=>{
  const env=patched();env.render();const descriptor=Object.getOwnPropertyDescriptor(env.Node.prototype,'textContent');let writes=0;
  Object.defineProperty(env.Node.prototype,'textContent',{...descriptor,set(value){writes++;Reflect.apply(descriptor.set,this,[value]);}});
  env.state.time='12:00';env.tick();assert.equal(writes,1);assert.equal(env.hud.element.nodes['.time'].textContent,'12:00');
});

test('a widget appearing first with the same class is never shadowed',()=>{
  const env=patched();env.render();const widget=new env.Element('time',new env.Element('widget'));widget.append(new env.Text('09:00'));
  env.hud.element.nodes['.time']=widget;env.tick();assert.equal(widget.writes,1);assert.equal(Object.hasOwn(widget,'textContent'),false);
});

test('widget spans with matching local parents still require the native bar ancestor chain',()=>{
  for(const [selector,parent] of [['.time','time-wrapper'],['.date','info-panel']]){
    const env=patched();env.render();const widget=new env.Element(selector.slice(1),new env.Element(parent));
    widget.append(new env.Text(selector==='.time'?'09:00':'Day 1'));env.hud.element.nodes[selector]=widget;
    env.tick();assert.equal(widget.writes,1,selector);assert.equal(Object.hasOwn(widget,'textContent'),false);
  }
});

test('inactive Calendaria is an ordinary skipped feature',()=>{
  const env=calendariaEnvironment();env.game.modules.get('calendaria').active=false;
  const result=install({runtime:env.runtime});assert.equal(result.status,'skipped');assert.equal(result.reason,'module-inactive');
  assert.equal(env.registrations.length,0);
});

test('a DOM rebuild and slice-mode replacement are discovered on the next callback',()=>{
  const env=patched();env.render();const old=env.hud.element;env.hud.element=env.dom().root;
  const time=env.hud.element.nodes['.time'];time.parentElement.className='slice-overlay';env.world();
  assert.equal(time.writes,0);assert.equal(old.nodes['.time'].writes,0);assert.equal(env.result.stats.skipped,2);
});

test('native records, ordering, IDs and duplicate renders survive close and reopen',async()=>{
  const env=patched(),seen=[];env.Hooks.on(TICK,()=>seen.push('before'));env.render();
  const list=env.Hooks.events[TICK],entry=list[1],fn=entry.fn;env.Hooks.on(TICK,()=>seen.push('after'));
  env.state.format=kind=>{if(kind==='hudTime')seen.push('hud');};env.render({isFirstRender:false});
  assert.equal(list[1],entry);assert.equal(entry.fn,fn);env.Hooks.callAll(TICK,'value');assert.deepEqual(seen,['before','hud','after']);
  await env.hud._onClose({});assert.equal(list.includes(entry),false);assert.equal(env.Hooks.events[WORLD].length,0);
  env.render();assert.notEqual(list.at(-1).id,entry.id);assert.equal(env.result.stats.adapted,4);
});

test('late installation reports unowned existing callbacks and does not rewrite them',()=>{
  const env=calendariaEnvironment();env.render();const before=[env.Hooks.events[TICK][0].fn,env.Hooks.events[WORLD][0].fn],reports=[];
  const result=install({runtime:env.runtime,report:r=>reports.push(r)});assert.equal(result.status,'installed');env.render({isFirstRender:false});
  assert.deepEqual([env.Hooks.events[TICK][0].fn,env.Hooks.events[WORLD][0].fn],before);
  assert.ok(reports.some(r=>r.reason==='existing-unowned-hud-callbacks'));
});

test('unknown HUD or Hooks functions cannot install',()=>{
  for(const mode of ['class','render','close','on','events','callAll']){
    const env=calendariaEnvironment();
    if(mode==='class')env.runtime.CALENDARIA.apps.HUD=class extends env.HUD{};
    else if(mode==='render')env.HUD.prototype._onRender=function(){};
    else if(mode==='close')env.HUD.prototype._onClose=async function(){};
    else if(mode==='events')Object.defineProperty(env.Hooks,'events',{get(){return {};}});
    else env.Hooks[mode]=function(){};
    assert.notEqual(install({runtime:env.runtime}).status,'installed',mode);assert.equal(env.registrations.length,0,mode);
  }
});

test('an unknown or asynchronous later render chain remains unadapted',async()=>{
  for(const mode of ['wrapper','listener','method','base','async']){
    const env=patched(),record=env.registrations[0];
    if(mode==='wrapper')record.chain.records.push({fn(){}});
    if(mode==='listener')record.chain.listeners.push({fn(){}});
    if(mode==='base')env.Application.prototype._onRender=function(){};
    if(mode==='method'||mode==='async'){
      const original=record.chain._wrapped;
      record.chain._wrapped=mode==='async'?async function(...args){await 0;return Reflect.apply(original,this,args);}:function(...args){return Reflect.apply(original,this,args);};
    }
    await env.render();const time=env.hud.element.nodes['.time'];env.tick();assert.equal(time.writes,1,mode);assert.equal(env.result.stats.adapted,0,mode);
  }
});

test('ambiguous deltas and modified existing records leave both new callbacks native',()=>{
  for(const mode of ['extra','existing','frozen','once']){
    const env=patched();let old;
    if(mode==='existing'){env.Hooks.on(TICK,()=>{});old=env.Hooks.events[TICK][0];}
    env.state.attach=()=>{
      if(mode==='extra')env.Hooks.on(TICK,()=>{});
      if(old)Object.defineProperty(old,'fn',{enumerable:false});
      if(mode==='frozen'||mode==='once'){
        env.Hooks.on(WORLD,()=>{});env.Hooks.events[WORLD].length=0;
        const list=env.Hooks.events[WORLD],push=list.push;
        list.push=function(entry){if(mode==='frozen')Object.freeze(entry);else entry.once=true;return Reflect.apply(push,this,[entry]);};
      }
    };
    env.render();env.tick();assert.equal(env.hud.element.nodes['.time'].writes,1,mode);assert.equal(env.result.stats.adapted,0,mode);
  }
});

test('a second record write failure restores the first record without changing native registration',()=>{
  const env=patched();env.Hooks.on(WORLD,()=>{});const list=env.Hooks.events[WORLD];list.length=0;
  const push=list.push;list.push=function(entry){return Reflect.apply(push,this,[new Proxy(entry,{defineProperty(){throw new Error('blocked');}})]);};
  env.render();env.tick();assert.equal(env.hud.element.nodes['.time'].writes,1);assert.equal(env.result.stats.adapted,0);
});

test('duplicate installation and restore affect only this patch',()=>{
  const env=patched();env.render();assert.equal(install({runtime:env.runtime}).status,'already-installed');
  const original=env.Hooks.events[TICK][0],foreign=()=>{};env.Hooks.events[WORLD][0].fn=foreign;
  env.result.restore();env.result.restore();env.tick();assert.equal(env.hud.element.nodes['.time'].writes,1);
  assert.equal(env.Hooks.events[TICK][0],original);assert.equal(env.Hooks.events[WORLD][0].fn,foreign);assert.equal(env.registrations[0].unregistered,true);
});

test('native-hook-adapter off broker can install before or after the HUD patch',async()=>{
  for(const first of [true,false]){
    const env=calendariaEnvironment();function other(){}env.Hooks.on('refreshToken',other);
    const adapt=()=>adaptNativeHooks({Hooks:env.Hooks,callbacks:[{hook:'refreshToken',source:other.toString(),wrap:fn=>function(...args){return Reflect.apply(fn,this,args);}}]});
    let broker=first?adapt():null;const result=install({runtime:env.runtime});assert.equal(result.status,'installed');env.render();broker??=adapt();
    assert.equal(broker.status,'installed');env.tick();assert.equal(env.hud.element.nodes['.time'].writes,0);
    await env.hud._onClose({});assert.equal(env.Hooks.events[TICK].length,0);env.Hooks.off('refreshToken',other);
    assert.equal(env.Hooks.events.refreshToken.length,0);result.restore();broker.restore();
  }
});

test('the render wrapper preserves the continuation receiver, arguments, result and exception',()=>{
  const env=patched(),wrapper=env.registrations[0].fn,first={},second={},value={};let seen;
  function continuation(...args){seen={receiver:this,args};return value;}
  assert.equal(Reflect.apply(wrapper,env.hud,[continuation.bind(env.hud),first,second]),value);
  assert.equal(seen.receiver,env.hud);assert.deepEqual(seen.args,[first,second]);
  const error=new Error('render failed');assert.throws(()=>Reflect.apply(wrapper,env.hud,[()=>{throw error;}]),e=>e===error);
});

test('a promised render result stays identical and its new records remain native',async()=>{
  const env=patched(),wrapper=env.registrations[0].fn,promise=Promise.resolve(42);
  const result=Reflect.apply(wrapper,env.hud,[()=>{Reflect.apply(env.registrations[0].chain._wrapped,env.hud,[{},{isFirstRender:true}]);return promise;}]);
  assert.equal(result,promise);assert.equal(await result,42);env.tick();assert.equal(env.hud.element.nodes['.time'].writes,1);
  assert.ok(env.reports.some(r=>r.reason==='async-render'));
});

test('a later foreign chain or hook method leaves already adapted callbacks native',()=>{
  for(const kind of ['wrapper','on','callAll','close','instance']){
    const env=patched();env.render();
    if(kind==='wrapper')env.registrations[0].chain.records.push({fn(){}});
    else if(kind==='close')env.HUD.prototype._onClose=async function(){};
    else if(kind==='instance')env.hud._onRender=function(){};
    else env.Hooks[kind]=function(){};
    env.tick();assert.equal(env.hud.element.nodes['.time'].writes,1,kind);
  }
});

test('restore preserves later descriptor edits and permits a fresh installation',()=>{
  const env=patched();env.render();const entry=env.Hooks.events[TICK][0];
  Object.defineProperty(entry,'fn',{enumerable:false});const descriptor=Object.getOwnPropertyDescriptor(entry,'fn');
  env.result.restore();assert.deepEqual(Object.getOwnPropertyDescriptor(entry,'fn'),descriptor);
  env.tick();assert.equal(env.hud.element.nodes['.time'].writes,1);
  assert.equal(install({runtime:env.runtime}).status,'installed');
});

test('reinstall through the retained empty libWrapper accessor adapts a newly shown HUD',async()=>{
  const env=patched();let result=env.result;
  for(let i=0;i<3;i++){
    env.render();const time=env.hud.element.nodes['.time'],writes=time.writes;
    env.tick();assert.equal(time.writes,writes);assert.equal(result.stats.adapted,2);
    const descriptor=Object.getOwnPropertyDescriptor(env.HUD.prototype,'_onRender');
    result.restore();assert.deepEqual(Object.getOwnPropertyDescriptor(env.HUD.prototype,'_onRender'),descriptor);
    await env.hud._onClose({});
    result=install({runtime:env.runtime});assert.equal(result.status,'installed');
  }
  result.restore();
});

test('reinstall rejects changed retained chains and empty wrappers this installer did not create',()=>{
  for(const kind of ['wrapper','listener','original','accessor','unknown']){
    const env=calendariaEnvironment();
    if(kind==='unknown'){
      const id=env.runtime.libWrapper.register('other','CALENDARIA.apps.HUD.prototype._onRender',wrapped=>wrapped(),'WRAPPER');
      env.runtime.libWrapper.unregister('other',id);
    }else{
      const result=install({runtime:env.runtime});assert.equal(result.status,'installed');result.restore();
      const chain=env.registrations[0].chain;
      if(kind==='wrapper')chain.records.push({fn(){}});
      if(kind==='listener')chain.listeners.push({fn(){}});
      if(kind==='original')chain._wrapped=function(){};
      if(kind==='accessor'){
        const descriptor=Object.getOwnPropertyDescriptor(env.HUD.prototype,'_onRender'),get=function(){return descriptor.get();};
        get._lib_wrapper=chain;Object.defineProperty(env.HUD.prototype,'_onRender',{...descriptor,get});
      }
    }
    const count=env.registrations.length,result=install({runtime:env.runtime});
    assert.equal(result.status,'skipped',kind);assert.equal(result.reason,'hud-source-mismatch',kind);
    assert.equal(env.registrations.length,count,kind);
  }
});
