import test from 'node:test';
import assert from 'node:assert/strict';
import {directoryEnvironment} from './token-directory-harness.mjs';

const baseline=process.env.TOKEN_DIRECTORY_BASELINE==='1';
const install=baseline?()=>{}:(await import('../scripts/patches/token-directory-performance.mjs')).installTokenDirectoryPerformancePatch;

async function installed(options){
  const e=directoryEnvironment(options);e.add('a');e.add('b');
  e.result=install({runtime:e.runtime});
  await e.render({force:true});
  e.events.length=0;
  return e;
}

test('coordinate-only updates keep existing directory entries and reapply viewed filters',async()=>{
  const e=await installed(),[token,other]=e.documents,original=e.elements[0];
  token.viewed=false;token.object.controlled=true;
  await e.render();
  assert.equal(e.elements[0],original,'movement must not rebuild existing entries');
  assert.equal(e.elements[0].hidden,true);
  assert.equal(token.object.controlled,false);
  assert.deepEqual([...e.tab._filterState.visible],[other]);
  assert.equal(e.group.meta.textContent,'(1)');
  assert.deepEqual(e.events.map(event=>event[0]),['release','flags','palette']);
});

test('x, y and batched coordinates accept accompanying document metadata',async()=>{
  for(const changes of [[{_id:'a',x:4000}],[{_id:'a',y:0}],[{_id:'a',x:0,y:200,_stats:{modifiedTime:1}}],
    [{_id:'a',x:200},{_id:'b',y:400}]]){
    const e=await installed(),entry=e.elements[0];
    await e.render(e.movement(changes));
    assert.equal(e.elements[0],entry);
  }
});

test('directory-affecting, mixed and unknown changes rebuild entries',async()=>{
  for(const change of [
    {_id:'a',name:'Renamed'}, {_id:'a',hidden:true}, {_id:'a',locked:true}, {_id:'a',level:'other'},
    {_id:'a',elevation:10}, {_id:'a',sort:5}, {_id:'a',rotation:90}, {_id:'a',_regions:['region']},
    {_id:'a',x:10,name:'Renamed'}, {_id:'a',x:10,flags:{}}, {_id:'a',x:10,'texture.src':'portrait'},
    {_id:'a'}, {_id:'a',_stats:{modifiedTime:1}}, {_id:'a',x:NaN}, {_id:'a',x:'10'}, {x:10}, {_id:'missing',x:10}
  ]){
    const e=await installed(),entry=e.elements[0];
    await e.render(e.movement([change]));
    assert.notEqual(e.elements[0],entry,JSON.stringify(change));
  }
  for(const changes of [[],null,{},[{_id:'a',x:1},{_id:'b',name:'Mixed'}]]){
    const e=await installed(),entry=e.elements[0];await e.render(e.movement(changes));assert.notEqual(e.elements[0],entry);
  }
});

test('ordinary, forced, initial and create/delete renders keep the native path',async()=>{
  for(const options of [{},{force:true},{renderContext:'createtokens'}, {renderContext:'deletetokens'},
    {renderContext:'updatetokens',renderData:[{_id:'a',x:1}],force:true},
    {renderContext:'updatetokens',renderData:[{_id:'a',x:1}],window:{detach:true}},
    {renderContext:'updatetokens',renderData:[{_id:'a',x:1}],parts:['tab']}]){
    const e=await installed(),entry=e.elements[0];await e.render(options);assert.notEqual(e.elements[0],entry);
  }
  const e=await installed(),entry=e.elements[0];e.directory.rendered=false;
  await e.render(e.movement(undefined,{force:true}));assert.notEqual(e.elements[0],entry);
});

test('search, elevation, hidden and locked filters retain the same results without replacing entries',async()=>{
  for(const configure of [
    e=>{e.tab._searchFilter.rgx=/^b$/;},
    e=>{e.documents[0].elevation=0;e.documents[1].elevation=20;e.tab._filterState.elevation.bottom=10;},
    e=>{e.documents[0].hidden=true;e.tab._filterState.hidden=false;},
    e=>{e.documents[0].locked=true;e.tab._filterState.locked=false;}
  ]){
    const e=await installed(),entry=e.elements[0];configure(e);await e.render();
    assert.equal(e.elements[0],entry);assert.deepEqual(e.elements.map(row=>row.hidden),[true,false]);
    assert.deepEqual([...e.tab._filterState.visible],[e.documents[1]]);
  }
});

test('a changed visible membership rebuilds the directory for GM and player',async()=>{
  for(const gm of [true,false]){
    const e=await installed({gm}),entry=e.elements[0];
    Object.defineProperty(e.documents[1],'visible',{get(){return this.x<100;}});
    e.documents[1].x=200;
    await e.render(e.movement([{_id:'b',x:200}]));
    assert.notEqual(e.elements[0],entry);assert.deepEqual(e.elements.map(row=>row.dataset.entryId),['a']);
  }
  const e=await installed({gm:false});e.documents[1].hidden=true;await e.render({force:true});
  const entry=e.elements[0];e.documents[1].hidden=false;await e.render();
  assert.notEqual(e.elements[0],entry);assert.deepEqual(e.elements.map(row=>row.dataset.entryId),['a','b']);
});

test('unknown or duplicate DOM membership falls back to rebuilding',async()=>{
  for(const damage of [e=>e.elements.push({...e.elements[0]}),e=>e.elements[1].dataset.entryId='unknown',e=>e.elements.pop()]){
    const e=await installed();damage(e);const entry=e.elements[0];await e.render();assert.notEqual(e.elements[0],entry);
  }
});

test('the first render after a scene switch is never suppressed, including a revisited scene',async()=>{
  const e=await installed(),first=e.scene;
  const next={...first,id:'other',apps:{[e.directory.id]:e.directory}};
  for(const scene of [next,first]){
    e.canvas.scene=scene;const entry=e.elements[0];await e.render();assert.notEqual(e.elements[0],entry);
  }
});

test('popouts, other layers, custom classes and disconnected tabs retain the original result',async()=>{
  for(const change of [
    e=>{e.directory.options.window.frame=true;},e=>{e.directory.popout={rendered:true};},
    e=>{e.canvas.activeLayer={options:{name:'tiles'}};},e=>{e.directory.tabGroups.sheet='tiles';e.CONFIG.Canvas.layers.tiles={layerClass:{documentName:'Token'}};},
    e=>{e.tab.options.collectionName='tiles';},e=>{e.tab.options.directory={};},e=>{e.canvas.tokens={};},
    e=>{e.CONFIG.Token.sidebar.applicationClass=class {};},e=>{e.CONFIG.ui.placeables=class {};},
    e=>{Object.setPrototypeOf(e.directory,Object.create(e.classes.PlaceableDirectory.prototype));},
    e=>{Object.setPrototypeOf(e.tab,Object.create(e.classes.TokenTab.prototype));},
    e=>{e.tab.element.isConnected=false;},e=>{e.directory.element.isConnected=false;},
    e=>{e.canvas.ready=false;},e=>{delete e.scene.apps[e.directory.id];},e=>{e.tab.rendered=false;}
  ]){
    const e=await installed();change(e);assert.equal(e.canRender(),true);
  }
});

test('other wrappers run exactly once before the decision, with arguments and non-boolean results intact',async()=>{
  const e=await installed(),wrapper=e.registrations[0]?.fn;
  if(baseline)return;
  const options=e.movement();let calls=0;
  assert.equal(wrapper.call(e.directory,actual=>{calls++;assert.equal(actual,options);return true;},options),false);
  assert.equal(calls,1);
  for(const result of [false,undefined,null,1,'yes',Promise.resolve(true)]){
    calls=0;assert.equal(wrapper.call(e.directory,()=>{calls++;return result;},options),result);assert.equal(calls,1);
  }
  const failure=Error('upstream');
  assert.throws(()=>wrapper.call(e.directory,()=>{throw failure;},options),error=>error===failure);
  let observed;
  wrapper.call(e.directory,()=>{observed=e.events.length;return true;},options);
  assert.equal(e.events.length,observed+1);
});

test('unknown preparation and render consumers prevent installation or delegate after replacement',async()=>{
  if(baseline)return;
  for(const [owner,name] of [['PlaceableDirectory','_canRender'],['PlaceableDirectory','_prepareContext'],
    ['PlaceableDirectory','_onRender'],['PlaceableDirectory','_renderTab'],['PlaceableTab','_prepareEntry'],
    ['PlaceableTab','_prepareDirectoryContext'],['PlaceableTab','_getEntryLabel'],['PlaceableTab','_onRender'],
    ['TokenTab','_prepareEntry'],['TokenTab','_prepareDirectoryContext'],['PlaceableTab','_applyFilters'],
    ['AbstractSidebarTab','render'],['ApplicationV2','render']]){
    const e=directoryEnvironment();e.classes[owner].prototype[name]=function(){return true;};
    assert.equal(install({runtime:e.runtime}).status,'unsupported-source',`${owner}.${name}`);
    assert.equal(e.registrations.length,0);
    if(name==='_canRender')continue;
    const after=await installed();after.classes[owner].prototype[name]=function(){return true;};
    assert.equal(after.canRender(),true,`${owner}.${name} changed after installation`);
  }
  const e=await installed();e.tab._getEntryLabel=()=> 'coordinate label';assert.equal(e.canRender(),true);
});

test('comments, whitespace and version labels do not disable recognized sources',async()=>{
  const e=await installed({mutateSource:source=>source.replace('if ( game.settings.get','/* upstream format */\n    if (game.settings.get')});
  e.runtime.game.version='99.999';const entry=e.elements[0];await e.render();assert.equal(e.elements[0],entry);
});

test('new overrides of inherited render or presentation methods keep their side effects',async()=>{
  if(baseline)return;
  for(const [owner,name] of [['TokenTab','_getEntryLabel'],['TokenTab','_applyFilters'],['TokenTab','render'],
    ['PlaceableDirectory','render'],['TokenTab','_preparePartContext'],['TokenTab','_prepareSearchContext'],
    ['TokenTab','_configureRenderParts']]){
    const e=directoryEnvironment();e.classes[owner].prototype[name]=function(){return 'custom';};
    assert.equal(install({runtime:e.runtime}).status,'unsupported-source',`${owner}.${name}`);
    const after=await installed();after.tab[name]=function(){return 'custom';};
    if(owner==='TokenTab')assert.equal(after.canRender(),true,`${owner}.${name} instance override`);
  }
});

test('registration, status and duplicate installation report the actual boundary',async()=>{
  if(baseline)return;
  const e=directoryEnvironment(),reports=[];
  const result=install({moduleId:'owner',runtime:e.runtime,report:entry=>reports.push(entry)});
  assert.equal(result.feature,'tokenDirectory');assert.equal(result.status,'installed');assert.equal(reports[0],result);
  assert.deepEqual([e.registrations[0].moduleId,e.registrations[0].target,e.registrations[0].type],
    ['owner','foundry.applications.sidebar.tabs.PlaceableDirectory.prototype._canRender','WRAPPER']);
  assert.equal(install({runtime:e.runtime}).status,'already-installed');assert.equal(e.registrations.length,1);
  assert.equal(install({runtime:{}}).status,'unsupported-runtime');
});

test('name, hidden, locked and level updates preserve native labels, order and grouping',async()=>{
  const e=await installed();
  Object.assign(e.documents[0],{name:'Zulu',hidden:true,locked:true,level:'upper'});
  const levels=new Map([['lower',{id:'lower',name:'Lower'}],['upper',{id:'upper',name:'Upper'}]]);
  levels.reverseSorted=[levels.get('upper'),levels.get('lower')];e.scene.levels=levels;
  await e.render(e.movement([{_id:'a',name:'Zulu',hidden:true,locked:true,level:'upper'}]));
  assert.deepEqual(e.elements.map(row=>[row.label,row.documentHidden,row.locked,row.levelId]),
    [['b',false,false,'level'],['Zulu',true,true,'upper']]);
  const prepared=e.events.find(([event])=>event==='tab-render')[1];
  assert.deepEqual(Array.from(prepared.groups,group=>[group.label,Array.from(group.entries,entry=>entry.id)]),
    [['Upper',['a']],['SIDEBAR.PLACEABLES.Categories.NoLevel',['b']]]);
});

test('unfiltered selected entries remain selected and first installation does not suppress a render',async()=>{
  const e=directoryEnvironment(),token=e.add('a');token.object.controlled=true;e.canvas.tokens.controlled=[token.object];
  const original=e.elements[0];const result=install({runtime:e.runtime});await e.render();
  assert.notEqual(e.elements[0],original);assert.equal(e.elements[0].classList.contains('active'),true);
  const selected=e.elements[0];await e.render();assert.equal(e.elements[0],selected);
  assert.equal(token.object.controlled,true);assert.equal(e.elements[0].classList.contains('active'),true);
  if(!baseline)assert.deepEqual(result.stats,{skipped:1,delegated:1});
});

test('guard-read failures delegate, and native filter failures still propagate',async()=>{
  if(baseline)return;
  const e=await installed(),wrapper=e.registrations[0].fn,entry=e.elements[0];
  const query=e.tab.element.querySelectorAll;
  e.tab.element.querySelectorAll=()=>{throw Error('DOM unavailable');};
  assert.equal(e.canRender(),true);e.tab.element.querySelectorAll=query;
  const failure=Error('filter state unavailable');
  Object.defineProperty(e.tab._filterState,'elevation',{get(){throw failure;}});
  assert.throws(()=>wrapper.call(e.directory,()=>true,e.movement()),error=>error===failure);
  assert.equal(e.elements[0],entry);
});
