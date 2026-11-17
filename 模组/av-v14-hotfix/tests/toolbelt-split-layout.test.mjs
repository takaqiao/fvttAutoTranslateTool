import test from 'node:test';
import assert from 'node:assert/strict';
import {nativeHooks} from './hook-harness.mjs';
import {installToolbeltSplitLayoutPatch} from '../scripts/patches/toolbelt-split-layout.mjs';

// A page that counts synchronous layouts the way a browser forces them: reading innerText after a
// DOM write. textContent never needs layout.
function makePage(){
  const page={writes:0,layouts:0,dirty:false};
  class Element{
    constructor(text=''){this.textContent=text;this.children=[];}
    get innerText(){if(page.dirty){page.layouts++;page.dirty=false;}return this.textContent;}
    set innerText(value){this.textContent=String(value);page.writes++;page.dirty=true;}
    replaceChildren(...nodes){this.children=nodes;this.textContent=nodes.map(node=>node.textContent).join('');page.writes++;page.dirty=true;}
  }
  page.Element=Element;
  return page;
}

// The markup PF2e 8.7 item-line.hbs renders: <span class="quantity">…<span>{{item.quantity}}</span>…</span>.
function makeSheet(page,quantities,{owner=true,inventory=true}={}){
  const spans=quantities.map(quantity=>new page.Element(String(quantity)));
  const tab={querySelectorAll:selector=>selector==='.items [data-item-id] .quantity span'?spans:[]};
  const root={querySelector:selector=>selector==='.tab.inventory'&&inventory?tab:null};
  // A freshly inserted sheet has not been laid out yet.
  page.dirty=true;
  return {app:{actor:{isOwner:owner}},html:{0:root,length:1},spans};
}

// Toolbelt 3.61.3 betterInventory #a, splitItem branch, as written: read innerText, replace children.
function toolbeltRender(page,calls){
  return (app,html)=>{
    calls.push(app);
    if(!app.actor.isOwner)return;
    const tab=html[0].querySelector('.tab.inventory');
    if(!tab)return;
    for(const span of tab.querySelectorAll('.items [data-item-id] .quantity span')){
      const quantity=Number(span.innerText);
      if(isNaN(quantity)||quantity<=1)continue;
      span.replaceChildren(new page.Element(span.innerText));
    }
  };
}

function makeRuntime({active=true,split=true}={}){
  const Hooks=nativeHooks(),tasks=[];
  const settings={get(ns,key){
    if(ns==='pf2e-toolbelt'&&key==='betterInventory.splitItem'){if(split instanceof Error)throw split;return split;}
    throw new Error(`${ns}.${key} is not registered`);
  }};
  return {Hooks,tasks,queueMicrotask:fn=>tasks.push(fn),flush(){for(const fn of tasks.splice(0))fn();},
    game:{modules:{get:id=>id==='pf2e-toolbelt'?{active}:undefined},settings}};
}
const own=span=>Object.hasOwn(span,'innerText');

test('without the patch every stacked row forces its own layout',()=>{
  const page=makePage(),runtime=makeRuntime(),sheet=makeSheet(page,[3,1,5,2]);
  runtime.Hooks.on('renderActorSheet',toolbeltRender(page,[]));
  runtime.Hooks.callAll('renderActorSheet',sheet.app,sheet.html,{});
  assert.equal(page.layouts,3);
});

test('with the patch the split links are the same and nothing forces layout',()=>{
  const page=makePage(),runtime=makeRuntime(),reports=[];
  const result=installToolbeltSplitLayoutPatch({runtime,report:data=>reports.push(data)});
  assert.equal(result.status,'installed');assert.equal(reports.at(-1).feature,'toolbeltSplitLayout');
  // Toolbelt registers its hook in ready, after the Hotfix setup install.
  const calls=[];runtime.Hooks.on('renderActorSheet',toolbeltRender(page,calls));
  const sheet=makeSheet(page,[3,1,5,2]);
  runtime.Hooks.callAll('renderActorSheet',sheet.app,sheet.html,{});
  assert.equal(calls.length,1);assert.equal(page.layouts,0);
  assert.deepEqual(sheet.spans.map(span=>span.children.map(link=>link.textContent)),[['3'],[],['5'],['2']]);
  assert.deepEqual(sheet.spans.map(span=>span.textContent),['3','1','5','2']);
  assert.deepEqual(result.stats,{renders:1,spans:4});
});

test('the override is gone once the render hook chain has finished',()=>{
  const page=makePage(),runtime=makeRuntime();
  installToolbeltSplitLayoutPatch({runtime});
  let during;runtime.Hooks.on('renderActorSheet',(app,html)=>{during=html[0].querySelector('.tab.inventory').querySelectorAll('.items [data-item-id] .quantity span').map(own);});
  const sheet=makeSheet(page,[4,2]);
  runtime.Hooks.callAll('renderActorSheet',sheet.app,sheet.html,{});
  assert.deepEqual(during,[true,true]);assert.equal(runtime.tasks.length,1);
  runtime.flush();
  assert.deepEqual(sheet.spans.map(own),[false,false]);
  page.dirty=true;assert.equal(sheet.spans[0].innerText,'4');assert.equal(page.layouts,1);
});

test('two renders before the microtask queue one cleanup that covers both',()=>{
  const page=makePage(),runtime=makeRuntime();
  installToolbeltSplitLayoutPatch({runtime});
  const first=makeSheet(page,[2]),second=makeSheet(page,[3]);
  runtime.Hooks.callAll('renderActorSheet',first.app,first.html,{});
  runtime.Hooks.callAll('renderActorSheet',second.app,second.html,{});
  assert.equal(runtime.tasks.length,1);runtime.flush();
  assert.deepEqual([...first.spans,...second.spans].map(own),[false,false]);
});

test('writing innerText during the window goes to the native setter',()=>{
  const page=makePage(),runtime=makeRuntime();
  installToolbeltSplitLayoutPatch({runtime});
  runtime.Hooks.on('renderActorSheet',(app,html)=>{html[0].querySelector('.tab.inventory').querySelectorAll('.items [data-item-id] .quantity span')[0].innerText='7';});
  const sheet=makeSheet(page,[2,3]),writes=page.writes;
  runtime.Hooks.callAll('renderActorSheet',sheet.app,sheet.html,{});
  assert.equal(sheet.spans[0].textContent,'7');assert.equal(page.writes,writes+1);assert.equal(own(sheet.spans[0]),false);
  runtime.flush();assert.equal(own(sheet.spans[1]),false);
});

test('a quantity with surrounding whitespace reads like innerText would',()=>{
  const page=makePage(),runtime=makeRuntime();
  installToolbeltSplitLayoutPatch({runtime});
  let read;runtime.Hooks.on('renderActorSheet',(app,html)=>{read=html[0].querySelector('.tab.inventory').querySelectorAll('.items [data-item-id] .quantity span')[0].innerText;});
  const sheet=makeSheet(page,['\n  12 ']);
  runtime.Hooks.callAll('renderActorSheet',sheet.app,sheet.html,{});
  assert.equal(read,'12');assert.equal(page.layouts,0);
});

for(const [name,options] of [['split off',{split:false}],['the setting missing',{split:new Error('not registered')}]])
  test(`with ${name} the spans are left alone`,()=>{
    const page=makePage(),runtime=makeRuntime(options),result=installToolbeltSplitLayoutPatch({runtime});
    let during;runtime.Hooks.on('renderActorSheet',(app,html)=>{during=html[0].querySelector('.tab.inventory').querySelectorAll('.items [data-item-id] .quantity span').map(own);});
    const sheet=makeSheet(page,[3]);
    runtime.Hooks.callAll('renderActorSheet',sheet.app,sheet.html,{});
    assert.deepEqual(during,[false]);assert.equal(runtime.tasks.length,0);assert.deepEqual(result.stats,{renders:0,spans:0});
  });

for(const [name,options] of [['a sheet the user does not own',{owner:false}],['a sheet without an inventory tab',{inventory:false}]])
  test(`${name} is not touched`,()=>{
    const page=makePage(),runtime=makeRuntime(),result=installToolbeltSplitLayoutPatch({runtime});
    const sheet=makeSheet(page,[3],options);
    runtime.Hooks.callAll('renderActorSheet',sheet.app,sheet.html,{});
    assert.equal(own(sheet.spans[0]),false);assert.equal(runtime.tasks.length,0);assert.deepEqual(result.stats,{renders:0,spans:0});
  });

test('an HTMLElement passed instead of jQuery works the same',()=>{
  const page=makePage(),runtime=makeRuntime();
  installToolbeltSplitLayoutPatch({runtime});
  let during;runtime.Hooks.on('renderActorSheet',(app,html)=>{during=own(html.querySelector('.tab.inventory').querySelectorAll('.items [data-item-id] .quantity span')[0]);});
  const sheet=makeSheet(page,[3]);
  runtime.Hooks.callAll('renderActorSheet',sheet.app,sheet.html[0],{});
  assert.equal(during,true);
});

test('Toolbelt inactive: reported as skipped and no hook is registered',()=>{
  const runtime=makeRuntime({active:false}),result=installToolbeltSplitLayoutPatch({runtime});
  assert.equal(result.status,'skipped');assert.equal(result.reason,'module-inactive');
  assert.equal(runtime.Hooks.events.renderActorSheet,undefined);
});

test('a second install on the same Hooks does not register twice',()=>{
  const runtime=makeRuntime();
  installToolbeltSplitLayoutPatch({runtime});
  assert.equal(installToolbeltSplitLayoutPatch({runtime}).status,'already-installed');
  assert.equal(runtime.Hooks.events.renderActorSheet.length,1);
});
