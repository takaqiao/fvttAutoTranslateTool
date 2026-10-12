import test from 'node:test';
import assert from 'node:assert/strict';
import {timestampEnvironment} from './timestamps-harness.mjs';
import {installTimestampPatch} from '../scripts/patches/timestamps.mjs';

const TARGET='foundry.applications.sidebar.tabs.ChatLog.prototype.updateTimestamps';
const own=element=>Object.getOwnPropertyDescriptor(element,'textContent');
function installed(e=timestampEnvironment()){
  const reports=[],result=installTimestampPatch({runtime:e.runtime,report:entry=>reports.push(entry)});
  assert.equal(result.status,'installed');
  return {...e,result,reports,stats:result.stats};
}

test('the unpatched refresh rewrites an unchanged label',()=>{
  const e=timestampEnvironment(),{stamp}=e.message(),text=stamp.firstChild;
  e.core();
  assert.equal(stamp.writes,1);assert.notEqual(stamp.firstChild,text);assert.equal(stamp.textContent,'1 min ago');
});

test('installation synchronously registers one WRAPPER on the core refresh',()=>{
  const e=timestampEnvironment(),reports=[];
  const result=installTimestampPatch({moduleId:'owner',runtime:e.runtime,report:entry=>reports.push(entry)});
  assert.ok(!(result instanceof Promise));
  assert.deepEqual(result,{feature:'timestamps',status:'installed',stats:{skipped:0,written:0}});
  assert.equal(reports.length,1);assert.equal(reports[0],result);
  assert.equal(e.registrations.length,1);
  const {moduleId,target,fn,type}=e.registrations[0];
  assert.deepEqual([moduleId,target,type,typeof fn],['owner',TARGET,'WRAPPER','function']);
  const other=timestampEnvironment();installTimestampPatch({runtime:other.runtime});
  assert.equal(other.registrations[0].moduleId,'av-v14-hotfix');
});

test('an unchanged label is not written and keeps its Text node',()=>{
  const e=installed(),{stamp}=e.message(),text=stamp.firstChild;
  e.refresh();e.refresh();
  assert.equal(stamp.writes,0);assert.equal(stamp.firstChild,text);
  assert.deepEqual(e.stats,{skipped:2,written:0});assert.equal(e.result.stats,e.stats);
});

test('a changed label is written once through the inherited setter',()=>{
  const e=installed(),{stamp,age}=e.message(),text=stamp.firstChild;
  age('2 min ago');e.refresh();
  assert.equal(stamp.writes,1);assert.equal(stamp.textContent,'2 min ago');assert.notEqual(stamp.firstChild,text);
  assert.deepEqual(e.stats,{skipped:0,written:1});
  e.refresh();
  assert.equal(stamp.writes,1);assert.deepEqual(e.stats,{skipped:1,written:1});
});

test('stamps share one own accessor during a refresh and keep no own property afterwards',()=>{
  for(const enumerable of [true,false]){
    const e=timestampEnvironment(),inherited=Object.getOwnPropertyDescriptor(e.Node.prototype,'textContent');
    Object.defineProperty(e.Node.prototype,'textContent',{...inherited,enumerable});
    installed(e);
    const a=e.message(),b=e.message('3 min ago'),keys=[a,b].map(({li,stamp})=>[Reflect.ownKeys(li),Reflect.ownKeys(stamp)]);
    let during;
    e.refresh(()=>{during=[own(a.stamp),own(b.stamp),own(a.li),a.stamp.textContent,b.stamp.textContent];e.core();});
    const [first,second,li,...labels]=during;
    assert.deepEqual([first.configurable,first.enumerable,typeof first.get,typeof first.set],[true,enumerable,'function','function']);
    assert.equal(first.get,second.get);assert.equal(first.set,second.set);
    assert.equal(li,undefined);assert.deepEqual(labels,['1 min ago','3 min ago']);
    assert.deepEqual([a,b].map(({li,stamp})=>[Reflect.ownKeys(li),Reflect.ownKeys(stamp)]),keys);
  }
});

test('the core refresh runs exactly once with its arguments, result and error intact',()=>{
  const e=installed(),{stamp}=e.message(),calls=[],value={};
  assert.equal(e.refresh((...args)=>{calls.push(args);return value;},1,'two',value),value);
  assert.deepEqual(calls,[[1,'two',value]]);
  assert.equal(e.refresh(),undefined);
  const failure=Error('core failure');let guarded;
  assert.throws(()=>e.refresh(()=>{guarded=!!own(stamp);throw failure;}),error=>error===failure);
  assert.equal(guarded,true);assert.equal(own(stamp),undefined);
});

test('a foreign own textContent stays in place and receives the write',()=>{
  const e=installed(),a=e.message(),b=e.message(),received=[];
  const foreign={get(){return 'foreign';},set(value){received.push(value);},enumerable:false,configurable:true};
  Object.defineProperty(a.stamp,'textContent',foreign);
  e.refresh();
  assert.deepEqual(own(a.stamp),foreign);assert.deepEqual(received,['1 min ago']);
  assert.equal(b.stamp.writes,0);assert.deepEqual(e.stats,{skipped:1,written:0});
});

test('a stamp that cannot take the accessor is written as core would',()=>{
  const blockers={
    'non-extensible':({stamp})=>Object.preventExtensions(stamp),
    'defineProperty throws':({li,stamp})=>{
      li.childNodes[0]=new Proxy(stamp,{defineProperty(target,key,descriptor){
        if(key==='textContent')throw Error('define failure');
        return Reflect.defineProperty(target,key,descriptor);
      }});
    }
  };
  for(const [name,block] of Object.entries(blockers)){
    const e=installed(),a=e.message(),b=e.message();block(a);
    let calls=0;
    assert.doesNotThrow(()=>e.refresh(()=>{calls++;e.core();}),name);
    assert.equal(calls,1,name);assert.equal(a.stamp.writes,1,name);assert.equal(own(a.stamp),undefined,name);
    assert.equal(b.stamp.writes,0,name);assert.deepEqual(e.stats,{skipped:1,written:0},name);
  }
});

test('only an equal string over a sole Text child is skipped',()=>{
  const shapes={
    'number':[e=>[new e.Text('5')],5,'5'],
    'String object':[e=>[new e.Text('same')],new String('same'),'same'],
    'null':[e=>[new e.Text('')],null,''],
    'two Text nodes':[e=>[new e.Text('sa'),new e.Text('me')],'same','same'],
    'Text then comment':[e=>[new e.Text('same'),Object.assign(new e.Node(8),{data:''})],'same','same'],
    'element child':[e=>[new e.Element('span').append(new e.Text('same'))],'same','same'],
    'comment child':[e=>[Object.assign(new e.Node(8),{data:'same'})],'same','same'],
    'no children':[()=>[],'',''],
    'empty Text node':[e=>[new e.Text('')],'',''],
    'different text':[e=>[new e.Text('same')],'Same','Same']
  };
  for(const [name,[children,value,expected]] of Object.entries(shapes)){
    const e=installed(),{stamp}=e.message();stamp.childNodes=children(e);
    e.refresh(()=>{stamp.textContent=value;});
    assert.equal(stamp.writes,1,name);assert.equal(stamp.textContent,expected,name);
    assert.deepEqual(e.stats,{skipped:0,written:1},name);
  }
});

test('unreadable children fall back to the inherited setter',()=>{
  for(const key of ['firstChild','lastChild','nodeType','data']){
    const e=installed(),{stamp}=e.message();
    const target=key.endsWith('Child')?stamp:stamp.firstChild;
    Object.defineProperty(target,key,{get(){throw Error('unreadable');},configurable:true});
    assert.doesNotThrow(()=>e.refresh(),key);
    assert.equal(stamp.writes,1,key);assert.deepEqual(e.stats,{skipped:0,written:1},key);
  }
});

test('an inherited setter failure propagates and still releases the stamps',()=>{
  const e=installed(),{stamp}=e.message();
  assert.throws(()=>e.refresh(()=>{stamp.textContent=Symbol('label');}),TypeError);
  assert.equal(own(stamp),undefined);assert.equal(stamp.writes,0);assert.deepEqual(e.stats,{skipped:0,written:1});
});

test('an accessor put on Node.prototype after installation is used, not bypassed',()=>{
  const e=installed(),{stamp,age}=e.message(),first=Object.getOwnPropertyDescriptor(e.Node.prototype,'textContent'),calls={get:0,set:0};
  Object.defineProperty(e.Node.prototype,'textContent',{...first,
    get(){calls.get++;return Reflect.apply(first.get,this,[]);},
    set(value){calls.set++;Reflect.apply(first.set,this,[value]);}});
  e.refresh();
  assert.deepEqual(calls,{get:0,set:0});assert.deepEqual(e.stats,{skipped:1,written:0});
  age('2 min ago');let label;
  e.refresh(()=>{label=stamp.textContent;e.core();});
  assert.equal(label,'1 min ago');assert.deepEqual(calls,{get:1,set:1});
  assert.equal(stamp.writes,1);assert.deepEqual(e.stats,{skipped:1,written:1});
});

test('stamps behind an accessor layered on Element.prototype are not shadowed',()=>{
  const e=installed(),{stamp}=e.message(),inherited=Object.getOwnPropertyDescriptor(e.Node.prototype,'textContent'),received=[];
  Object.defineProperty(e.Element.prototype,'textContent',{...inherited,set(value){received.push(value);Reflect.apply(inherited.set,this,[value]);}});
  let shadowed;
  e.refresh(()=>{shadowed=!!own(stamp);e.core();});
  assert.equal(shadowed,false);assert.deepEqual(received,['1 min ago']);
  assert.equal(stamp.writes,1);assert.deepEqual(e.stats,{skipped:0,written:0});
});

test('the refresh runs unguarded when the guard cannot be set up',()=>{
  const breakers={
    'Node removed':e=>{delete e.runtime.Node;},
    'data property':e=>Object.defineProperty(e.Node.prototype,'textContent',{value:'',writable:true,configurable:true}),
    'getter only':e=>Object.defineProperty(e.Node.prototype,'textContent',{get(){return '';},set:undefined,configurable:true}),
    'query throws':e=>{e.runtime.document.querySelectorAll=()=>{throw Error('query failure');};},
    'query result not iterable':e=>{e.runtime.document.querySelectorAll=()=>null;}
  };
  for(const [name,breaker] of Object.entries(breakers)){
    const e=installed(),{stamp}=e.message();breaker(e);
    let calls=0,shadowed;
    assert.equal(e.refresh(()=>{calls++;shadowed=!!own(stamp);return 'done';}),'done',name);
    assert.equal(calls,1,name);assert.equal(shadowed,false,name);
  }
});

test('a re-entrant refresh leaves the outer accessors in place',()=>{
  const e=installed(),a=e.message(),b=e.message();let outer,afterInner;
  b.age('2 min ago');
  e.refresh(()=>{
    outer=own(a.stamp);
    e.refresh();
    afterInner=[own(a.stamp),own(b.stamp)];
    e.core();
  });
  assert.equal(typeof outer.set,'function');assert.deepEqual(afterInner,[outer,outer]);
  assert.equal(own(a.stamp),undefined);assert.equal(own(b.stamp),undefined);
  assert.equal(a.stamp.writes,0);assert.equal(b.stamp.writes,1);
  assert.deepEqual(e.stats,{skipped:3,written:1});
});

test('an own accessor redefined during the refresh is not removed',()=>{
  for(const key of ['get','set']){
    const e=installed(),a=e.message(),b=e.message();let foreign;
    e.refresh(()=>{foreign={...own(a.stamp),[key]:function(){}};Object.defineProperty(a.stamp,'textContent',foreign);});
    assert.deepEqual(own(a.stamp),foreign,key);assert.equal(own(b.stamp),undefined,key);
  }
});

test('an async core refresh is reported once as ineffective and returned as is',()=>{
  const e=installed(),{stamp}=e.message(),pending=Promise.resolve(),thenable={then(){}};
  assert.equal(e.refresh(()=>({then:'not callable'})).then,'not callable');
  assert.equal(e.reports.length,1);
  assert.equal(e.refresh(()=>pending),pending);assert.equal(e.refresh(()=>thenable),thenable);
  assert.deepEqual(e.reports.slice(1),[{feature:'timestamps',status:'ineffective',reason:'async-refresh',stats:{skipped:0,written:0}}]);
  assert.equal(e.reports[1].stats,e.stats);assert.equal(own(stamp),undefined);
});

test('diagnostics cannot replace the refresh result',()=>{
  const e=timestampEnvironment();
  installTimestampPatch({runtime:e.runtime,report:entry=>{if(entry.status==='ineffective')throw Error('report failure');}});
  const pending=Promise.resolve(),hostile={get then(){throw Error('then failure');}};
  assert.equal(e.refresh(()=>hostile),hostile);assert.equal(e.refresh(()=>pending),pending);
});

test('missing runtime capabilities are reported without registering',()=>{
  const breakers={
    'no libWrapper':e=>{delete e.runtime.libWrapper;},
    'no register':e=>{e.runtime.libWrapper={};},
    'no foundry':e=>{delete e.runtime.foundry;},
    'no ChatLog':e=>{delete e.runtime.foundry.applications.sidebar.tabs.ChatLog;},
    'no updateTimestamps':e=>{delete e.ChatLog.prototype.updateTimestamps;},
    'updateTimestamps not a function':e=>{e.ChatLog.prototype.updateTimestamps=15000;},
    'no document':e=>{delete e.runtime.document;},
    'no querySelectorAll':e=>{delete e.runtime.document.querySelectorAll;},
    'no Node':e=>{delete e.runtime.Node;},
    'no textContent':e=>{delete e.Node.prototype.textContent;},
    'data textContent':e=>Object.defineProperty(e.Node.prototype,'textContent',{value:'',writable:true,configurable:true}),
    'getter-only textContent':e=>Object.defineProperty(e.Node.prototype,'textContent',{get(){return '';},set:undefined,configurable:true}),
    'setter-only textContent':e=>Object.defineProperty(e.Node.prototype,'textContent',{get:undefined,set(){},configurable:true})
  };
  for(const [name,breaker] of Object.entries(breakers)){
    const e=timestampEnvironment(),reports=[];breaker(e);
    const result=installTimestampPatch({runtime:e.runtime,report:entry=>reports.push(entry)});
    assert.deepEqual(result,{feature:'timestamps',status:'unsupported-runtime',stats:{skipped:0,written:0}},name);
    assert.equal(reports.length,1,name);assert.equal(reports[0],result,name);assert.equal(e.registrations.length,0,name);
  }
});

test('a ChatLog prototype is patched once',()=>{
  const e=installed(),reports=[];
  for(const runtime of [e.runtime,{...e.runtime}]){
    const again=installTimestampPatch({runtime,report:entry=>reports.push(entry)});
    assert.equal(again.status,'already-installed');assert.equal(reports.at(-1),again);
  }
  assert.equal(e.registrations.length,1);
  const {stamp}=e.message();e.refresh();
  assert.equal(stamp.writes,0);assert.deepEqual(e.stats,{skipped:1,written:0});
  assert.equal(installed().registrations.length,1);
});

test('every reported result survives structuredClone',()=>{
  const e=installed(),unsupported=installTimestampPatch({runtime:{}});
  e.message();e.refresh();e.refresh(()=>Promise.resolve());
  const results=[e.result,...e.reports,unsupported,installTimestampPatch({runtime:e.runtime})];
  assert.deepEqual(results.map(result=>result.status),['installed','installed','ineffective','unsupported-runtime','already-installed']);
  for(const result of results)assert.deepEqual(structuredClone(result),result,result.status);
});
