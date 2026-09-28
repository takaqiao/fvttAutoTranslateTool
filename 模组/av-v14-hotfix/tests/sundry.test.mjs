import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {installSundryPatch} from '../scripts/patches/sundry.mjs';

const original = await readFile(new URL('./fixtures/sundry-1.10.2-tokenEffectHider.js.txt', import.meta.url), 'utf8');
const constants = await readFile(new URL('./fixtures/sundry-1.10.2-const.js.txt', import.meta.url), 'utf8');
const coreHooks = JSON.parse(await readFile(new URL('./fixtures/sundry-core-hooks.json', import.meta.url), 'utf8'));
const originalCode = constants.replaceAll('export const ', 'const ') + '\n'
  + original.replace(/^import[\s\S]*?;\r?\n/gm, '').replaceAll('export ', '') + '\nsetupHideTokenEffects';
const currentOriginal=await readFile(new URL('./fixtures/sundry-1.10.3-tokenEffectHider.js.txt',import.meta.url),'utf8');
const currentConstants=await readFile(new URL('./fixtures/sundry-1.10.3-const.js.txt',import.meta.url),'utf8');
const currentCode=currentConstants.replaceAll('export const ','const ')+'\n'
  +currentOriginal.replace(/^import[\s\S]*?;\r?\n/gm,'').replaceAll('export ','')+'\nsetupHideTokenEffects';

function hookAPI() {
  return vm.runInNewContext(`${coreHooks.findSplice};
    Object.defineProperty(Array.prototype,'findSplice',{value:findSplice});
    ${coreHooks.Hooks}; Hooks$1`,{CONFIG:{debug:{hooks:false}},CONST:{vtt:'Foundry'},console});
}

async function environment({version = '1.10.2', active = true, mode = 'relevant', types = 'all', modules = [], beforeHooks, systemVersion = '8.5.0'} = {}) {
  const settings = new Map([['hide.effects.token.surface', mode], ['hide.effects.token.enabled-for', types]]);
  const g = {Hooks: hookAPI(), canvas: {tokens: {placeables: [], highlightObjects: false}},
    game: {version:'14.368', system:{id:'pf2e',version:systemVersion}, modules: new Map([['sundry', {version, active}], ...modules.map(id => [id, {active: true}])]),
      settings: {get(namespace, key) {assert.equal(namespace, 'sundry'); return settings.get(key);}}}};
  beforeHooks?.(g.Hooks);
  await vm.runInNewContext(version==='1.10.3'?currentCode:originalCode, {...g, MODULE_ID: 'sundry', getSetting: key => g.game.settings.get('sundry', key)})(true);
  return {g, settings};
}

function fixture(count = 0, info = []) {
  let reads = 0;
  const bg = {visible: true};
  const overlay = {visible: true};
  const icons = Array.from({length: count}, (_, index) => ({visible: index % 2 === 0}));
  const token = {hover: false, actor: {type: 'character', get appliedEffects() {reads++; return info;}},
    effects: {bg, overlay, children: [bg, ...icons, overlay]}};
  return {token, reads: () => reads, view: () => ({background: bg.visible, overlay: overlay.visible, icons: icons.map(icon => icon.visible)})};
}

// Native Sprite's visible property is an own boolean data field in the audited
// Foundry PIXI runtime. Unknown display classes intentionally keep the slow path.
class Sprite {constructor(visible=false){this.visible=visible;}}
function nativeFixture(g, visible=[false,false], info=[]) {
  g.PIXI={Sprite};
  const f=fixture(0,info),icons=visible.map(value=>new Sprite(value));
  f.token.effects.children=[f.token.effects.bg,...icons,f.token.effects.overlay];
  return {...f,icons,view:()=>({background:f.token.effects.bg.visible,icons:icons.map(i=>i.visible)})};
}

test('native ordinary icons already at the target visibility avoid document construction',async()=>{
 for(const systemVersion of ['8.5.0','8.5.1']) {
  const {g}=await environment({systemVersion});const f=nativeFixture(g);
  installSundryPatch({g});g.Hooks.callAll('refreshToken',f.token);
  assert.equal(f.reads(),0);assert.deepEqual(f.view(),{background:false,icons:[false,false]});
  f.token.hover=true;for(const icon of f.icons)icon.visible=true;
  g.Hooks.callAll('refreshToken',f.token);assert.equal(f.reads(),0);
  assert.deepEqual(f.view(),{background:true,icons:[true,true]});
 }
});

test('installed Sundry 1.10.3 avoids unchanged effects and preserves a subsequent real visibility change',async()=>{
 const {g}=await environment({version:'1.10.3',systemVersion:'8.5.1'}),f=nativeFixture(g,[false]);
 installSundryPatch({g});g.Hooks.callAll('refreshToken',f.token);
 assert.equal(f.reads(),0);assert.deepEqual(f.view().icons,[false]);
 f.token.hover=true;g.Hooks.callAll('refreshToken',f.token);
 assert.equal(f.reads(),1);assert.deepEqual(f.view().icons,[true]);
});

test('native shortcut matches upstream across current values, modes, hover and background owners',async()=>{
 const info=[{slug:'off-guard'},{slug:'custom',duration:{secondsRemaining:45}}];
 for(const systemVersion of ['8.5.0','8.5.1'])for(const mode of ['none','relevant','under-1-hour'])for(const hover of [false,true])for(const highlighted of [false,true])for(const modules of [[],['pf2e-effects-halo']])for(const visible of [[false,false],[true,true],[false,true]]){
  const left=await environment({mode,modules,systemVersion}),right=await environment({mode,modules,systemVersion});
  const a=nativeFixture(left.g,visible,info),b=nativeFixture(right.g,visible,info);
  a.token.hover=b.token.hover=hover;left.g.canvas.tokens.highlightObjects=right.g.canvas.tokens.highlightObjects=highlighted;
  installSundryPatch({g:right.g});left.g.Hooks.callAll('refreshToken',a.token);right.g.Hooks.callAll('refreshToken',b.token);
  assert.deepEqual(b.view(),a.view());
 }
});

test('new native icons and changing durations fall back whenever any visibility differs',async()=>{
 for(const systemVersion of ['8.5.0','8.5.1']) {
 const {g}=await environment({mode:'under-1-hour',systemVersion}),info=[{slug:'custom',duration:{secondsRemaining:4000}}];
 const f=nativeFixture(g,[false],info);installSundryPatch({g});
 g.Hooks.callAll('refreshToken',f.token);assert.equal(f.reads(),0);
 // Effect redraw replaces the actual child, including when no special flag is supplied.
 const replacement=new Sprite(true);f.token.effects.children=[f.token.effects.bg,replacement,f.token.effects.overlay];
 info[0].duration.secondsRemaining=45;g.Hooks.callAll('refreshToken',f.token,{refreshState:true});assert.equal(replacement.visible,true);assert.equal(f.reads(),1);
 info[0].duration.secondsRemaining=4000;g.Hooks.callAll('refreshToken',f.token);assert.equal(replacement.visible,false);assert.equal(f.reads(),2);
 g.Hooks.callAll('refreshToken',f.token);assert.equal(f.reads(),2);
 }
});

test('accessors, non-native display objects and unsupported PF2e use original populated logic',async()=>{
 for(const systemVersion of ['8.5.0','8.5.1'])for(const kind of ['accessor','subclass','system']){
  const {g}=await environment({systemVersion}),f=nativeFixture(g,[false]);
  if(kind==='accessor')Object.defineProperty(f.icons[0],'visible',{get:()=>false,set(){}});
  if(kind==='subclass')Object.setPrototypeOf(f.icons[0],Object.create(Sprite.prototype));
  if(kind==='system')g.game.system.version='8.5.2';
  installSundryPatch({g});g.Hooks.callAll('refreshToken',f.token);assert.equal(f.reads(),1,kind);
 }
});

test('audited versions read live system eligibility after installation and keep unknown versions on native logic',async()=>{
 const {g}=await environment(),f=nativeFixture(g,[false]);
 assert.equal(installSundryPatch({g}).status,'installed');
 for(const [version,id,expectedReads] of [['8.5.1','pf2e',0],['8.5.2','pf2e',1],['8.5.1','sf2e',2],['8.5.0','pf2e',2]]) {
  g.game.system={id,version};g.Hooks.callAll('refreshToken',f.token);
  assert.equal(f.reads(),expectedReads,id+'/'+version);
  assert.deepEqual(f.view(),{background:false,icons:[false]});
 }
});

test('empty-icon refresh preserves background behavior while avoiding appliedEffects construction', async () => {
  const {g} = await environment();
  const item = fixture();
  installSundryPatch({g});
  g.Hooks.callAll('refreshToken', item.token);
  assert.deepEqual(item.view(), {background: false, overlay: true, icons: []});
  assert.equal(item.reads(), 0);
  item.token.hover = true;
  g.Hooks.callAll('refreshToken', item.token);
  assert.equal(item.view().background, true);
  assert.equal(item.reads(), 0);
});

test('all three existing background-owning modules retain their background visibility', async () => {
  for (const id of ['pf2e-dorako-ux', 'pathfinder-ui', 'pf2e-effects-halo']) {
    const {g} = await environment({modules: [id]});
    const item = fixture();
    installSundryPatch({g});
    g.Hooks.callAll('refreshToken', item.token);
    assert.equal(item.view().background, true);
    assert.equal(item.reads(), 0);
  }
});

test('nonempty icons retain every original mode, background and hover result', async () => {
  const info = [{type: 'base', statuses: new Set(['off-guard'])}, {slug: 'custom', duration: {secondsRemaining: 45}}, {slug: 'long', duration: {secondsRemaining: 4000}}];
  for (const mode of ['none', 'relevant', 'relevant-under-1-hour', 'under-1-hour', 'relevant-under-10-min', 'under-10-min', 'relevant-under-1-min', 'under-1-min']) {
    for (const hover of [false, true]) for (const modules of [[], ['pf2e-effects-halo']]) {
      const before = await environment({mode, modules}), after = await environment({mode, modules});
      const left = fixture(4, info), right = fixture(4, info);
      left.token.hover = right.token.hover = hover;
      installSundryPatch({g: after.g});
      before.g.Hooks.callAll('refreshToken', left.token);
      after.g.Hooks.callAll('refreshToken', right.token);
      assert.deepEqual(right.view(), left.view(), JSON.stringify({mode, hover, modules}));
      assert.equal(right.reads(), 1);
    }
  }
});

test('effect duration is read anew, and character-type settings remain live', async () => {
  const {g, settings} = await environment({mode: 'under-1-hour', types: 'pcs'});
  const info = [{slug: 'custom', duration: {secondsRemaining: 4000}}];
  const item = fixture(1, info);
  installSundryPatch({g});
  g.Hooks.callAll('refreshToken', item.token);
  assert.equal(item.view().icons[0], false);
  info[0].duration.secondsRemaining = 45;
  item.token.effects.children[1].visible = true;
  g.Hooks.callAll('refreshToken', item.token);
  assert.equal(item.view().icons[0], true);
  assert.equal(item.reads(), 2);
  settings.set('hide.effects.token.enabled-for', 'npcs');
  const empty = fixture();
  g.Hooks.callAll('refreshToken', empty.token);
  assert.equal(empty.view().background, true);
  assert.equal(empty.reads(), 0);
});

test('highlight uses current canvas state and preserves mixed-token behavior without reading empty actors', async () => {
  const {g} = await environment();
  const empty = fixture(), populated = fixture(1, [{slug: 'custom'}]);
  g.canvas.tokens.placeables = [empty.token, populated.token];
  g.canvas.tokens.highlightObjects = true;
  installSundryPatch({g});
  g.Hooks.callAll('highlightObjects', false);
  assert.equal(empty.view().background, true);
  assert.equal(populated.view().icons[0], true);
  assert.equal(empty.reads(), 0);
  assert.equal(populated.reads(), 1);
});

test('the two exact callbacks keep their live records, slots, IDs and descriptors; unrelated hooks continue', async () => {
  const {g} = await environment();
  const originals = [g.Hooks.events.refreshToken[0], g.Hooks.events.highlightObjects[0]];
  const descriptors=originals.map(entry=>Object.getOwnPropertyDescriptors(entry));
  const arrays=[g.Hooks.events.refreshToken,g.Hooks.events.highlightObjects];
  let otherCalls = 0;
  const unrelated = () => {otherCalls++;};
  g.Hooks.on('refreshToken', unrelated);
  const result = installSundryPatch({g});
  assert.equal(result.status, 'installed');
  for(let i=0;i<originals.length;i++) {
    assert.equal(arrays[i],g.Hooks.events[originals[i].hook]);
    assert.equal(arrays[i][0],originals[i]);
    const actual=Object.getOwnPropertyDescriptors(originals[i]);
    assert.notEqual(actual.fn.value,descriptors[i].fn.value);
    actual.fn.value=descriptors[i].fn.value;
    assert.deepEqual(actual,descriptors[i]);
  }
  assert.equal(g.Hooks.events.refreshToken.some(entry => entry.fn === unrelated), true);
  g.Hooks.callAll('refreshToken', fixture().token);
  assert.equal(otherCalls, 1);
});

test('version mismatch, inactive modules, missing callbacks and duplicate callbacks cause no mutations', async () => {
  for (const kind of ['version', 'inactive', 'missing', 'duplicate', 'modified']) {
    const {g} = await environment({version: kind === 'version' ? '1.10.4' : '1.10.2', active: kind !== 'inactive'});
    if (kind === 'missing') g.Hooks.off('highlightObjects', g.Hooks.events.highlightObjects[0].id);
    if (kind === 'duplicate') g.Hooks.on('refreshToken', g.Hooks.events.refreshToken[0].fn);
    if (kind === 'modified') {g.Hooks.off('refreshToken', g.Hooks.events.refreshToken[0].id); g.Hooks.on('refreshToken', token => token);}
    const before = {refresh: [...g.Hooks.events.refreshToken], highlight: [...g.Hooks.events.highlightObjects]};
    assert.equal(installSundryPatch({g}).status, 'skipped', kind);
    assert.deepEqual([...g.Hooks.events.refreshToken], before.refresh);
    assert.deepEqual([...g.Hooks.events.highlightObjects], before.highlight);
  }
});

test('repeated install does not add callbacks and restore reinstates originals without removing unrelated work', async () => {
  const {g} = await environment();
  const originalRefresh = g.Hooks.events.refreshToken[0].fn;
  const originalHighlight = g.Hooks.events.highlightObjects[0].fn;
  const installed = installSundryPatch({g});
  assert.equal(installSundryPatch({g}).status, 'installed');
  assert.equal(g.Hooks.events.refreshToken.length, 1);
  assert.equal(g.Hooks.events.highlightObjects.length, 1);
  const unrelated = () => {};
  g.Hooks.on('refreshToken', unrelated);
  installed.restore();
  installed.restore();
  assert.deepEqual(new Set(g.Hooks.events.refreshToken.map(entry => entry.fn)), new Set([originalRefresh, unrelated]));
  assert.equal(g.Hooks.events.highlightObjects[0].fn, originalHighlight);
});

test('unknown Hooks implementation leaves both original callbacks installed', async () => {
  const {g} = await environment();
  const originalRefresh = g.Hooks.events.refreshToken[0].fn;
  const originalHighlight = g.Hooks.events.highlightObjects[0].fn;
  const on = g.Hooks.on;
  g.Hooks.on = function(hook, ...args) {if (hook === 'highlightObjects') throw Error('registration failed'); return on.call(this, hook, ...args);};
  assert.equal(installSundryPatch({g}).status, 'skipped');
  assert.deepEqual(Array.from(g.Hooks.events.refreshToken,entry => entry.fn), [originalRefresh]);
  assert.deepEqual(Array.from(g.Hooks.events.highlightObjects,entry => entry.fn), [originalHighlight]);
});

test('a following listener observes the same Sundry state and keeps its final visibility override',async()=>{
  for(const hook of ['refreshToken','highlightObjects']) {
    const baseline=await environment(),patched=await environment();
    const run=({g},install)=>{
      const item=fixture(),seen=[];item.token.hover=true;item.token.effects.bg.visible=false;
      g.canvas.tokens.placeables=[item.token];
      g.Hooks.on(hook,()=>{seen.push(item.token.effects.bg.visible);item.token.effects.bg.visible=false;});
      if(install)assert.equal(installSundryPatch({g}).status,'installed');
      g.Hooks.callAll(hook,hook==='refreshToken'?item.token:false);
      return {seen,view:item.view()};
    };
    assert.deepEqual(run(patched,true),run(baseline,false),hook);
  }
});

test('original function, replacement function, and original ID all remove the same native record',async()=>{
  for(const selector of ['original','replacement','id']) {
    const {g}=await environment(),entry=g.Hooks.events.refreshToken[0],original=entry.fn;
    const installed=installSundryPatch({g});
    g.Hooks.off('refreshToken',selector==='id'?entry.id:selector==='original'?original:entry.fn);
    assert.equal(g.Hooks.events.refreshToken.length,0,selector);
    installed.restore();
    assert.equal(g.Hooks.events.refreshToken.length,0,'restore must not resurrect removed listener');
    g.Hooks.off('ignored-for-ID',entry.id);
    assert.equal(g.Hooks.events.refreshToken.length,0);
  }
});

test('original-function off keeps first-match order when the function is later registered again',async()=>{
  const {g}=await environment(),entry=g.Hooks.events.refreshToken[0],original=entry.fn;
  installSundryPatch({g});
  const added=g.Hooks.on('refreshToken',original);
  g.Hooks.off('refreshToken',original);
  assert.deepEqual(Array.from(g.Hooks.events.refreshToken,e=>e.id),[added]);
  g.Hooks.off('refreshToken',original);
  assert.equal(g.Hooks.events.refreshToken.length,0);
});

test('once, call short-circuit, callback context and snapshots keep core semantics',async()=>{
  const {g}=await environment(),entry=g.Hooks.events.refreshToken[0];
  entry.context={marker:'untouched'};
  installSundryPatch({g});
  let receiver,late=0,once=0;
  g.Hooks.once('refreshToken',function(){receiver=this;once++;return false;});
  const onceEntry=g.Hooks.events.refreshToken[1];
  g.Hooks.on('refreshToken',()=>late++);
  assert.equal(g.Hooks.call('refreshToken',fixture().token),false);
  assert.equal(receiver,onceEntry);assert.equal(once,1);assert.equal(late,0);
  g.Hooks.callAll('refreshToken',fixture().token);
  assert.equal(once,1);assert.equal(late,1);assert.equal(entry.context.marker,'untouched');
});

test('restore keeps original slot and descriptors and does not overwrite subsequent edits',async()=>{
  const {g}=await environment(),entries=[...g.Hooks.events.refreshToken,...g.Hooks.events.highlightObjects];
  const descriptors=entries.map(e=>Object.getOwnPropertyDescriptor(e,'fn'));
  const nativeOff=Object.getOwnPropertyDescriptor(g.Hooks,'off');
  const installed=installSundryPatch({g}),other=()=>{};
  g.Hooks.events.highlightObjects[0].fn=other;
  installed.restore();
  assert.equal(g.Hooks.events.refreshToken[0],entries[0]);
  assert.deepEqual(Object.getOwnPropertyDescriptor(entries[0],'fn'),descriptors[0]);
  assert.equal(entries[1].fn,other);
  assert.deepEqual(Object.getOwnPropertyDescriptor(g.Hooks,'off'),nativeOff);
});

test('unwritable or accessor callback fields, nonnative events and unsupported core skip without mutation',async()=>{
  for(const kind of ['frozen','accessor','events','core','off']) {
    const {g}=await environment(),entry=g.Hooks.events.refreshToken[0],original=entry.fn;
    const highlightEntry=g.Hooks.events.highlightObjects[0],highlight=highlightEntry.fn;
    if(kind==='frozen')Object.freeze(entry);
    if(kind==='accessor')Object.defineProperty(entry,'fn',{get:()=>original,configurable:true});
    if(kind==='events'){const events=g.Hooks.events;Object.defineProperty(g.Hooks,'events',{get:()=>({...events,refreshToken:[entry]}),configurable:true});}
    if(kind==='core')g.game.version='15.0';
    if(kind==='off')Object.defineProperty(g.Hooks,'off',{writable:false});
    assert.equal(installSundryPatch({g}).status,'skipped',kind);
    assert.equal(entry.fn,original);assert.equal(highlightEntry.fn,highlight);
  }
});

test('native callAll snapshots retain live record references when installation happens earlier in the dispatch',async()=>{
  let g,result;
  ({g}=await environment({beforeHooks:Hooks=>Hooks.once('refreshToken',()=>{result=installSundryPatch({g});})}));
  const item=fixture();
  g.Hooks.callAll('refreshToken',item.token);
  assert.equal(result.status,'installed');assert.equal(item.reads(),0);
  assert.equal(item.view().background,false);
});

test('a removed record in an in-flight native snapshot is restored without being reinserted',async()=>{
  let g,installed,original;
  ({g}=await environment({beforeHooks:Hooks=>Hooks.once('refreshToken',()=>{
    Hooks.off('refreshToken',original);installed.restore();
  })}));
  original=g.Hooks.events.refreshToken[1].fn;
  installed=installSundryPatch({g});
  const item=fixture();g.Hooks.callAll('refreshToken',item.token);
  assert.equal(item.reads(),1,'native snapshot still invokes its original record once');
  assert.equal(g.Hooks.events.refreshToken.length,0);
});

test('frozen or once duplicates of the callback still cause a conservative skip',async()=>{
  for(const once of [false,true]) {
    const {g}=await environment(),entry=g.Hooks.events.refreshToken[0],fn=entry.fn;
    g.Hooks.on('refreshToken',fn,{once});
    if(!once)Object.freeze(entry);
    assert.equal(installSundryPatch({g}).status,'skipped');
    assert.equal(entry.fn,fn);
  }
});

test('restoring leaves a subsequent Hooks.off wrapper in place and its native removal still works',async()=>{
  const {g}=await environment(),entry=g.Hooks.events.refreshToken[0],original=entry.fn;
  const installed=installSundryPatch({g}),adapter=g.Hooks.off;
  let calls=0;
  const later=g.Hooks.off=function(...args){calls++;return Reflect.apply(adapter,this,args);};
  installed.restore();assert.equal(g.Hooks.off,later);
  g.Hooks.off('refreshToken',original);
  assert.equal(calls,1);assert.equal(g.Hooks.events.refreshToken.length,0);
});
