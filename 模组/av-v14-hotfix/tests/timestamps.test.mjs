import test from 'node:test';
import assert from 'node:assert/strict';
import {timestampEnvironment} from './timestamps-harness.mjs';
import {installTimestampPatch} from '../scripts/patches/timestamps.mjs';
const hookRecords=e=>['renderChatMessageHTML','renderChatLog','renderChatPopout'].flatMap(name=>e.runtime.Hooks.events[name]??[]);

async function install(e){const result=await installTimestampPatch({runtime:e.runtime});assert.equal(result.status,'installed',JSON.stringify(result));return result;}
function ownedHook(e,result,name,...args){const record=result.hooks.find(record=>record.hook===name);assert.ok(record,name);return e.runtime.Hooks.events[name].find(entry=>entry.id===record.id).fn(...args);}

test('identical primitive strings preserve the sole Text node and avoid a native write',async()=>{
  const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);
  const native=Object.getOwnPropertyDescriptor(e.Node.prototype,'textContent'),text=element.firstChild;
  const result=await install(e);element.textContent='1 min ago';
  assert.equal(element.writes,0);assert.equal(element.firstChild,text);assert.equal(element.textContent,'1 min ago');
  assert.deepEqual(Object.getOwnPropertyDescriptor(e.Node.prototype,'textContent'),native);
  result.restore();assert.equal(Object.hasOwn(element,'textContent'),false);
  element.textContent='1 min ago';assert.equal(element.writes,1);assert.notEqual(element.firstChild,text);
});

test('changed strings replace the Text through the native setter rather than updating its data',async()=>{
  const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);await install(e);
  const original=element.firstChild;element.textContent='2 min ago';
  assert.equal(element.writes,1);assert.equal(element.textContent,'2 min ago');assert.notEqual(element.firstChild,original);
  const current=element.firstChild;element.textContent='2 min ago';assert.equal(element.writes,1);assert.equal(element.firstChild,current);
});

test('rich, multiple and empty child shapes keep native replacement even for equal aggregate text',async()=>{
  for(const kind of ['rich','multiple','empty','comment']) {
    const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);await install(e);
    element._children=[];
    if(kind==='rich'){const span=new e.Element();span.append(new e.Text('same'));element.append(span);}
    if(kind==='multiple')element.append(new e.Text('sa'),new e.Text('me'));
    if(kind==='comment'){const child=new e.Node(8);element.append(child);}
    const value=kind==='rich'||kind==='multiple'?'same':'';const prior=element.firstChild;
    element.textContent=value;assert.equal(element.writes,1,kind);assert.equal(element.textContent,value);
    if(value)assert.notEqual(element.firstChild,prior);
  }
});

test('non-string values retain native coercion counts and native failures',async()=>{
  for(const value of [null,undefined,12,false,Symbol('x'),new String('1 min ago')]) {
    const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);await install(e);
    if(typeof value==='symbol')assert.throws(()=>{element.textContent=value;},TypeError);
    else {element.textContent=value;assert.equal(element.writes,1);assert.equal(element.textContent,value===null?'':String(value));}
  }
  const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);await install(e);
  let calls=0;element.textContent={toString(){calls++;return '1 min ago';}};assert.equal(calls,1);assert.equal(element.writes,1);
});

test('existing own or inherited custom textContent descriptors are never replaced',async()=>{
  for(const inherited of [false,true]){
    const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);let calls=0;
    const target=inherited?e.Element.prototype:element;
    Object.defineProperty(target,'textContent',{get(){return 'custom';},set(){calls++;},enumerable:false,configurable:true});
    const original=Object.getOwnPropertyDescriptor(target,'textContent');await install(e);
    assert.deepEqual(Object.getOwnPropertyDescriptor(target,'textContent'),original);if(inherited)assert.equal(Object.hasOwn(element,'textContent'),false);
    element.textContent='custom';assert.equal(calls,1);
  }
});

test('custom Text data or child accessors are not invoked by the guard',async()=>{
  for(const kind of ['data','firstChild','nodeType','nextSibling']) {
    const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);await install(e);let calls=0;
    const target=kind==='firstChild'?element:element.firstChild;
    Object.defineProperty(target,kind,{get(){calls++;throw Error('custom getter');},configurable:true});
    element.textContent='1 min ago';assert.equal(calls,0,kind);assert.equal(element.writes,1,kind);
  }
});

test('later prototype setters are honored and later native descriptor drift disables skipping',async()=>{
  for(const kind of ['textContent','firstChild','matches']) {
    const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);await install(e);
    const text=Object.getOwnPropertyDescriptor(e.Node.prototype,'textContent');let calls=0;
    if(kind==='textContent')Object.defineProperty(e.Node.prototype,'textContent',{...text,set(value){calls++;return Reflect.apply(text.set,this,[value]);}});
    if(kind==='firstChild'){const d=Object.getOwnPropertyDescriptor(e.Node.prototype,'firstChild');Object.defineProperty(e.Node.prototype,'firstChild',{...d,get(){calls++;return d.get.call(this);}});}
    if(kind==='matches'){const native=e.Element.prototype.matches;e.Element.prototype.matches=function(...args){calls++;return native.apply(this,args);};}
    element.textContent='1 min ago';assert.equal(element.writes,1,kind);assert.equal(calls,kind==='textContent'?1:0,kind);
  }
});

test('native detached message render hooks attach without retaining the ChatMessage object',async()=>{
  const e=timestampEnvironment(),result=await install(e),{message:html,element}=e.stamp();
  const message=new e.runtime.foundry.documents.ChatMessage();message.system={renderHTML:async()=>html};
  assert.equal(await message.renderHTML(),html);element.textContent='1 min ago';assert.equal(element.writes,0);
  assert.ok(result.hooks.some(h=>h.hook==='renderChatMessageHTML'));
});

test('local post-hook attachment handles a later synchronous timestamp replacement',async()=>{
  const e=timestampEnvironment(),result=await install(e),{message,element}=e.stamp();
  ownedHook(e,result,'renderChatMessageHTML',{},message);element.remove();const next=new e.Element(['message-timestamp']);next.append(new e.Text('same'));message.append(next);
  await Promise.resolve();next.textContent='same';assert.equal(next.writes,0);
});

test('native ChatLog and ChatPopout class hooks cover popouts and preserve unrelated scopes',async()=>{
  const e=timestampEnvironment(),result=await install(e);
  for(const kind of ['ChatLog','ChatPopout']) {
    const app=new (kind==='ChatLog'?e.runtime.foundry.applications.sidebar.tabs.ChatLog:e.runtime.foundry.applications.sidebar.apps.ChatPopout)();
    const root=new e.Element(),{message,element}=e.stamp();root.append(message);
    await app._doEvent(()=>{}, {hookName:'render',hookArgs:[root]});element.textContent='1 min ago';assert.equal(element.writes,0,kind);
  }
  const unrelated=new e.Element(['message-timestamp']);unrelated.append(new e.Text('same'));e.runtime.document.append(unrelated);
  ownedHook(e,result,'renderChatLog',{},e.runtime.document);unrelated.textContent='same';assert.equal(unrelated.writes,1);
  const {message,element}=e.stamp();ownedHook(e,result,'renderChatMessageHTML',{},message);element.remove();element.textContent='1 min ago';assert.equal(element.writes,1);
});

test('borrowed accessors delegate for other receivers rather than extending patch scope',async()=>{
  const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);await install(e);
  const descriptor=Object.getOwnPropertyDescriptor(element,'textContent'),other=new e.Element();other.append(new e.Text('same'));
  Reflect.apply(descriptor.set,other,['same']);assert.equal(other.writes,1);assert.equal(Reflect.apply(descriptor.get,other,[]),'same');
  assert.throws(()=>Reflect.apply(descriptor.set,{},['same']),TypeError);
});

test('restore removes only owned descriptors and hooks even after freeze or foreign edits',async()=>{
  const e=timestampEnvironment(),first=e.stamp(),second=e.stamp(),third=e.stamp();e.runtime.document.append(first.message,second.message,third.message);
  const result=await install(e),saved=Object.getOwnPropertyDescriptor(first.element,'textContent');Object.freeze(first.element);
  Object.defineProperty(second.element,'textContent',{get:()=>42,configurable:true});const foreign=Object.getOwnPropertyDescriptor(second.element,'textContent');
  third.message.remove();assert.doesNotThrow(()=>result.restore());assert.doesNotThrow(()=>result.restore());
  assert.equal(Object.getOwnPropertyDescriptor(first.element,'textContent').set,saved.set);
  assert.deepEqual(Object.getOwnPropertyDescriptor(second.element,'textContent'),foreign);assert.equal(Object.hasOwn(third.element,'textContent'),false);
  for(const {hook,id}of result.hooks)assert.ok(!e.runtime.Hooks.events[hook].some(entry=>entry.id===id));
});

test('restore prevents pending local attachment and repeat installs stay idempotent',async()=>{
  const e=timestampEnvironment(),result=await install(e);assert.equal(await installTimestampPatch({runtime:e.runtime}),result);
  const {message,element}=e.stamp();ownedHook(e,result,'renderChatMessageHTML',{},message);result.restore();await Promise.resolve();assert.equal(Object.hasOwn(element,'textContent'),false);
  const next=await install(e);assert.notEqual(next,result);next.restore();
});

test('core source and DOM overrides skip without attachment or hooks',async()=>{
  for(const kind of ['source','native','changed-source','changed-native']) {
    const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);
    if(kind==='source')e.runtime.foundry.applications.api.ApplicationV2.prototype._doEvent=function(){};
    if(kind==='native')Object.defineProperty(e.Node.prototype,'textContent',{get(){return '';},set(){},configurable:true});
    const options={runtime:e.runtime};
    if(kind.startsWith('changed-')){const {hashSource}=await import('../scripts/source-hash.mjs');options.hash=async value=>{if(kind==='changed-source')e.runtime.foundry.applications.api.ApplicationV2.prototype._doEvent=function(){};else e.Element.prototype.matches=function(){return true;};return hashSource(value);};}
    const result=await installTimestampPatch(options);assert.notEqual(result.status,'installed',kind);assert.equal(Object.hasOwn(element,'textContent'),false);
    assert.equal(hookRecords(e).length,0,kind);
  }
});

test('later inherited getter/setter return values and exceptions are preserved',async()=>{
  const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);await install(e);
  const failure=Error('foreign setter'),received=[];
  Object.defineProperty(e.Element.prototype,'textContent',{get(){assert.equal(this,element);return 'foreign';},set(value){received.push([this,value]);throw failure;},configurable:true});
  assert.equal(element.textContent,'foreign');assert.throws(()=>{element.textContent='1 min ago';},error=>error===failure);
  assert.deepEqual(received,[[element,'1 min ago']]);assert.equal(element.writes,0);
});

test('later inherited data properties follow ordinary assignment and getters',async()=>{
  for(const writable of [true,false]){
    const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);const result=await install(e);
    Object.defineProperty(e.Element.prototype,'textContent',{value:'foreign',writable,configurable:true});
    assert.equal(element.textContent,'foreign');
    if(writable){element.textContent='next';assert.equal(element.textContent,'next');assert.equal(Object.getOwnPropertyDescriptor(element,'textContent').value,'next');}
    else assert.throws(()=>{element.textContent='next';},TypeError);
    result.restore();assert.equal(element.writes,0);
  }
});

test('descriptor-only foreign changes survive restore and frozen owned setters retire',async()=>{
  const e=timestampEnvironment(),first=e.stamp(),second=e.stamp();e.runtime.document.append(first.message,second.message);const result=await install(e);
  const own=Object.getOwnPropertyDescriptor(first.element,'textContent');Object.defineProperty(first.element,'textContent',{...own,enumerable:!own.enumerable});
  const changed=Object.getOwnPropertyDescriptor(first.element,'textContent');
  Object.defineProperty(second.element,'textContent',{configurable:false});result.restore();
  assert.deepEqual(Object.getOwnPropertyDescriptor(first.element,'textContent'),changed);
  first.element.textContent='1 min ago';second.element.textContent='1 min ago';
  assert.equal(first.element.writes,1);assert.equal(second.element.writes,1);
});

test('a failed hook registration retires already registered hooks',async()=>{
  const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);
  const on=e.runtime.Hooks.on;let calls=0;e.runtime.Hooks.on=function(...args){if(++calls===2)throw Error('registration failed');return Reflect.apply(on,this,args);};
  const result=await installTimestampPatch({runtime:e.runtime});assert.equal(result.status,'installation-failed');
  assert.equal(Object.hasOwn(element,'textContent'),false);assert.equal(hookRecords(e).length,0);
});

test('concurrent installs create one set of callbacks',async()=>{
  const e=timestampEnvironment(),results=await Promise.all([installTimestampPatch({runtime:e.runtime}),installTimestampPatch({runtime:e.runtime})]);
  assert.equal(results[0].status,'installed');assert.equal(results[0],results[1]);
  assert.equal(hookRecords(e).length,3);results[0].restore();
});

test('a wrapped message renderer getter is not evaluated or required to subscribe to its public hook',async()=>{
  const e=timestampEnvironment();let getterReads=0;
  Object.defineProperty(e.runtime.foundry.documents.ChatMessage.prototype,'renderHTML',{get(){getterReads++;throw Error('libWrapper boundary must not be read');},configurable:true});
  const result=await install(e),{message,element}=e.stamp();ownedHook(e,result,'renderChatMessageHTML',{},message);
  element.textContent='1 min ago';assert.equal(element.writes,0);assert.equal(getterReads,0);result.restore();
});

test('native-looking comments in ordinary functions never satisfy DOM native guards',async()=>{
  for(const kind of ['getter','method','constructor']){
    const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.document.append(message);
    if(kind==='getter')Object.defineProperty(e.Node.prototype,'textContent',{get(){/* [native code] */return '';},set(){/* [native code] */},configurable:true});
    if(kind==='method')e.Element.prototype.matches=function(){/* [native code] */return true;};
    if(kind==='constructor'){function Replacement(){/* [native code] */}Replacement.prototype=e.Node.prototype;e.runtime.Node=Replacement;}
    const result=await installTimestampPatch({runtime:e.runtime});assert.equal(result.status,'unsupported-runtime',kind);
    assert.equal(Object.hasOwn(element,'textContent'),false,kind);assert.equal(hookRecords(e).length,0,kind);
  }
});

test('matching timestamp contracts avoid redundant writes on later core generations',async()=>{
 const e=timestampEnvironment(),{message,element}=e.stamp();e.runtime.game.version='15.1';e.runtime.document.append(message);
 const result=await install(e);element.textContent='1 min ago';assert.equal(element.writes,0);
 element.textContent='2 min ago';assert.equal(element.writes,1);result.restore();
});
