import test from 'node:test';
import assert from 'node:assert/strict';
import {installHudTrackerPatch} from '../scripts/patches/hud-tracker.mjs';

const STATES={ERROR:-3,CLOSING:-2,CLOSED:-1,NONE:0,RENDERING:1,RENDERED:2};

// Mirrors the parts of pf2e-hud 2.57.0's TrackerPF2eHUD the patch relies on.
class Tracker {
  constructor(){
    this.id='pf2e-hud-tracker';this.state=STATES.RENDERED;this.html='<ol>a</ol>';this.collapsed=false;
    this.content={isConnected:true,innerHTML:''};this.renders=[];this.replaced=0;this.onRendered=0;this.closes=0;
  }
  get rendered(){return this.state===STATES.RENDERED;}
  close(){this.closes++;this.state=STATES.CLOSED;return Promise.resolve(this);}
  render(options,_options){
    this.renders.push([options,_options]);
    this._replaceHTML(this.html,this.content,{collapsed:this.collapsed});
    return Promise.resolve(this._onRender({}, {})).then(()=>this);
  }
  _replaceHTML(e,n,i){n.innerHTML=e,this.replaced++}
  async _onRender(e,n){this.onRendered++}
}

function environment({active=true,instance}={}){
  const timers=[],hooks=[],reports=[];let time=1000;
  const runtime={
    game:{modules:new Map([['pf2e-hud',{active}]])},
    foundry:{applications:{instances:new Map(instance?[[instance.id,instance]]:[]),api:{ApplicationV2:{RENDER_STATES:STATES}}}},
    Hooks:{on:(name,fn)=>{hooks.push({name,fn,id:hooks.length+1});return hooks.length;},off:(name,id)=>{const hook=hooks.find(entry=>entry.id===id);if(hook)hook.off=true;}},
    setTimeout:(fn,delay)=>{timers.push({fn,at:time+delay});},
    performance:{now:()=>time}
  };
  return {runtime,timers,hooks,reports,report:entry=>reports.push(entry),
    advance(ms){time+=ms;for(const timer of timers.splice(0))if(timer.at<=time)timer.fn();else timers.push(timer);},
    get time(){return time;}};
}

const settle=()=>new Promise(resolve=>setImmediate(resolve));

test('an inactive pf2e-hud is reported quietly and nothing is hooked',()=>{
  const e=environment({active:false});
  assert.equal(installHudTrackerPatch({runtime:e.runtime}).status,'inactive');
  assert.equal(e.hooks.length,0);
});

test('the tracker is adopted when it first renders, and only that application',async()=>{
  const e=environment(),app=new Tracker(),other={id:'other'};
  const result=installHudTrackerPatch({runtime:e.runtime,report:e.report});
  assert.equal(result.status,'installed');assert.equal(result.adopted,false);
  const [hook]=e.hooks;assert.equal(hook.name,'renderApplicationV2');
  hook.fn(other);assert.equal(hook.off,undefined);
  hook.fn(app);assert.equal(hook.off,true);
  assert.equal(Object.hasOwn(app,'render'),true);
  assert.equal(e.reports.at(-1).adopted,true);
  const already=environment({instance:app});
  assert.equal(installHudTrackerPatch({runtime:already.runtime}).adopted,true);
  assert.equal(already.hooks.length,0);
});

test('a burst of refreshes renders once at once and once more with the latest state',async()=>{
  const e=environment(),app=new Tracker();e.runtime.foundry.applications.instances.set(app.id,app);
  const result=installHudTrackerPatch({runtime:e.runtime});
  await app.render(true);
  assert.equal(app.renders.length,1);
  const second=app.render(true),third=app.render({force:true});
  assert.equal(second,third);
  assert.equal(app.renders.length,1);
  e.advance(99);assert.equal(app.renders.length,1);
  e.advance(1);
  assert.equal(await second,app);
  assert.deepEqual(app.renders.map(([options])=>options),[true,{force:true}]);
  assert.deepEqual(result.stats,{rendered:2,throttled:2,dropped:0,identical:1});
  // A refresh after the window renders at once again.
  e.advance(100);await app.render(true);assert.equal(app.renders.length,3);
});

test('a close inside the window wins over the pending refresh',async()=>{
  const e=environment(),app=new Tracker();e.runtime.foundry.applications.instances.set(app.id,app);
  const result=installHudTrackerPatch({runtime:e.runtime});
  await app.render(true);
  const pending=app.render(true);await app.close();
  assert.equal(await pending,app);
  e.advance(100);await settle();
  assert.equal(app.renders.length,1);assert.equal(result.stats.dropped,1);assert.equal(app.closes,1);
});

test('a close queued behind a slow render still cancels the merged refresh',async()=>{
  // Render and close share one queue; the close only takes effect after the render in flight.
  class Queued extends Tracker {
    constructor(){super();this.queue=Promise.resolve();this.gates=[];}
    render(options,_options){
      this.renders.push([options,_options]);
      if(this.state<=STATES.NONE&&!(options===true||options?.force))return Promise.resolve(this);
      return this.queue=this.queue.then(()=>new Promise(resolve=>this.gates.push(resolve))).then(()=>{
        this.state=STATES.RENDERED;this._replaceHTML(this.html,this.content,{collapsed:this.collapsed});return this;
      });
    }
    close(){this.closes++;return this.queue=this.queue.then(()=>{this.state=STATES.CLOSED;return this;});}
  }
  const e=environment(),app=new Queued();e.runtime.foundry.applications.instances.set(app.id,app);
  const result=installHudTrackerPatch({runtime:e.runtime});
  const leading=app.render(true);
  const merged=app.render(true);
  const closing=app.close();
  assert.equal(app.state,STATES.RENDERED,'the close is still waiting');
  e.advance(100);
  assert.equal(await merged,app);
  app.gates.shift()();await leading;await closing;await settle();
  assert.equal(app.state,STATES.CLOSED);
  assert.deepEqual(app.renders.map(([options])=>options),[true],'the tracker is not reopened');
  assert.equal(result.stats.dropped,1);
  // Reopening after the close renders at once.
  const reopen=app.render(true);await settle();app.gates.shift()?.();await reopen;
  assert.equal(app.renders.length,2);assert.equal(app.state,STATES.RENDERED);
});

test('a refresh while the tracker is still rendering is merged too',async()=>{
  const e=environment(),app=new Tracker();e.runtime.foundry.applications.instances.set(app.id,app);
  installHudTrackerPatch({runtime:e.runtime});
  await app.render(true);app.state=STATES.RENDERING;
  const merged=app.render(true);assert.equal(app.renders.length,1);
  app.state=STATES.RENDERED;e.advance(100);await merged;assert.equal(app.renders.length,2);
});

test('first renders, other options and other receivers keep the native path',async()=>{
  const e=environment(),app=new Tracker();e.runtime.foundry.applications.instances.set(app.id,app);
  installHudTrackerPatch({runtime:e.runtime});
  await app.render(true);
  for(const [options,_options] of [[{parts:['list']}],[{force:true,position:{}}],[true,{}],[Object.create({force:true})]]){
    const before=app.renders.length;await app.render(options,_options);
    assert.equal(app.renders.length,before+1,JSON.stringify(options));
  }
  app.state=STATES.CLOSED;const before=app.renders.length;await app.render(true);assert.equal(app.renders.length,before+1);
  const other=new Tracker();await app.render.call(other,true);assert.equal(other.renders.length,1);
});

test('identical content leaves the tracker DOM and its render follow-up alone',async()=>{
  const e=environment(),app=new Tracker();e.runtime.foundry.applications.instances.set(app.id,app);
  installHudTrackerPatch({runtime:e.runtime});
  await app.render(true);assert.deepEqual([app.replaced,app.onRendered],[1,1]);
  e.advance(100);await app.render(true);assert.deepEqual([app.replaced,app.onRendered],[1,1]);
  for(const change of [
    ()=>{app.html='<ol>b</ol>';},
    ()=>{app.collapsed=true;},
    ()=>{app.content={isConnected:true,innerHTML:''};},
    ()=>{app.content.isConnected=false;}
  ]){
    change();const before=[app.replaced,app.onRendered];
    e.advance(100);await app.render(true);
    assert.deepEqual([app.replaced,app.onRendered],[before[0]+1,before[1]+1],String(change));
    app.content.isConnected=true;
  }
});

test('a tracker that no longer replaces its content from the string is left alone',()=>{
  for(const Spoiled of [
    class extends Tracker {_replaceHTML(e,n,i){n.replaceChildren(e);}},
    class extends Tracker {_replaceHTML(e,n,i){n.innerHTML=i;}},
    class extends Tracker {_replaceHTML(e,n,i){m.innerHTML=e;}},
    class extends Tracker {get _onRender(){return undefined;}}
  ]){
    const app=new Spoiled(),e=environment({instance:app});
    assert.equal(installHudTrackerPatch({runtime:e.runtime}).status,'unsupported-source',String(Spoiled));
    assert.equal(Object.hasOwn(app,'render'),false);
  }
  // Someone else's instance-level override is not stacked on.
  for(const name of ['render','close']){
    const app=new Tracker();app[name]=function(){};
    assert.equal(installHudTrackerPatch({runtime:environment({instance:app}).runtime}).status,'unsupported-source',name);
  }
});

test('wrappers registered on the class after the tracker was adopted still run',async()=>{
  class Late extends Tracker {}
  const e=environment(),app=new Late();e.runtime.foundry.applications.instances.set(app.id,app);
  installHudTrackerPatch({runtime:e.runtime});
  const calls=[];
  for(const name of ['render','close','_replaceHTML','_onRender']){
    const inner=Tracker.prototype[name];
    Late.prototype[name]=function(...args){calls.push(name);return inner.apply(this,args);};
  }
  await app.render(true);
  assert.deepEqual(calls,['render','_replaceHTML','_onRender']);
  app.html='<ol>b</ol>';e.advance(100);await app.render(true);
  const merged=app.render(true);e.advance(100);await merged;
  assert.equal(calls.filter(name=>name==='render').length,3);
  await app.close();assert.equal(calls.at(-1),'close');
});

test('missing hook or timer APIs stand down',()=>{
  const e=environment();delete e.runtime.setTimeout;
  assert.equal(installHudTrackerPatch({runtime:e.runtime}).status,'unsupported-runtime');
});

await settle();
