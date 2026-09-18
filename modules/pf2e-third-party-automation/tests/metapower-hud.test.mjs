import test from 'node:test';
import assert from 'node:assert/strict';
import {createMetapowerProvider} from '../scripts/metapower/provider.mjs';
import {METAPOWER_SOURCES} from '../scripts/metapower/rules.mjs';

// The reviewed HUD 2.55.2 bundle preserves these class names and method bytes.
// Keep the real digest gate in these tests; a permissive fake would hide drift.
function actionControllers(item,{native=()=>{throw Error('unexpected unobserved action')},explore=()=>{}}={}){
 const al=native,hb=explore;
 class ActionsSidebarAction{use(e){let n=this.item;return n?.isOfType("feat","action")&&al(e,n,this.virtualData)}}
 class ActionShortcut{use(e){this.item&&(this.isExploration?hb(this.actor,this.item.id):al(e,this.item))}}
 return {
  sidebar:Object.assign(new ActionsSidebarAction(),{item}),
  persistent:Object.assign(new ActionShortcut(),{item,actor:item.actor,type:'action',isExploration:false})
 };
}

function setup(t,native=async()=>null){
 const hooks=new Map(),errors=[],requests=[],digests=[];
 const digest=crypto.subtle.digest.bind(crypto.subtle);
 t.mock.method(crypto.subtle,'digest',(...args)=>{const pending=digest(...args);digests.push(pending);return pending});
 const priorConfig=globalThis.CONFIG;
 globalThis.CONFIG={Actor:{sheetClasses:{character:{}}},Dice:{rolls:[]}};
 t.after(()=>{if(priorConfig===undefined)delete globalThis.CONFIG;else globalThis.CONFIG=priorConfig});
 const gm={id:'gm'},game={user:gm,users:{activeGM:gm},actors:new Map(),scenes:new Map(),pf2e:{actions:new Map()},
  modules:new Map([['pf2e-hud',{version:'2.55.2'}],['pf2e-toolbelt',{version:'3.56.2'}]]),
  toolbelt:{api:{actionable:Object.freeze({useAction:native})}}};
 const actor={uuid:'Actor.a',type:'character',items:new Map([['w',{sourceId:METAPOWER_SOURCES.widen}]])};
 const item={id:'i',uuid:'Actor.a.Item.i',actor,isOfType:(...types)=>types.includes('action')};
 const provider=createMetapowerProvider({game,fromUuid:async()=>null,onError:error=>errors.push(error)});
 provider.register({Hooks:{on:(name,fn)=>hooks.set(name,fn)},libWrapper:{register(){}},socket:{register(){},async executeAsUser(name,gmId,payload){
  assert.equal(gmId,'gm');requests.push({name,payload});return {ok:true,value:{status:'reserved',nonce:payload.nonce}};
 }}});
 return {actor,item,errors,requests,game,async render(kind,controllers){
  const event=kind==='sidebar'?'renderActionsSidebarPF2eHUD':'renderPersistentShortcutsPF2eHUD';
  hooks.get(event)({[kind==='sidebar'?'sidebarItems':'shortcuts']:new Map(controllers.map((controller,i)=>[i,controller]))});
  await Promise.all(digests.splice(0));await Promise.resolve();
 }};
}

for(const kind of ['sidebar','persistent'])test(`mixed ${kind} HUD controllers preserve unrelated controls and await one original action use`,async t=>{
 let finishNative,nativeCalls=0,seenEvent,seenItem;
 const nativeResult=new Promise(resolve=>{finishNative=resolve});
 const f=setup(t,(event,item)=>{nativeCalls++;seenEvent=event;seenItem=item;return nativeResult});
 const action=actionControllers(f.item)[kind];
 class ActionsStance{toggle(){return 'stance'}}
 class ActionsSidebarStrike{attack(){return 'strike'}}
 class ActionsSidebarBlast{attack(){return 'blast'}}
 class SpellShortcut{use(){return 'spell'}}
 class ConsumableShortcut{use(){return 'consumable'}}
 class StanceShortcut{use(){return 'stance-shortcut'}}
 const unrelated=(kind==='sidebar'?[new ActionsStance(),new ActionsSidebarStrike(),new ActionsSidebarBlast()]:[new SpellShortcut(),new ConsumableShortcut(),new StanceShortcut()])
  .map(controller=>Object.assign(controller,{item:f.item,type:controller instanceof SpellShortcut?'spell':'stance'}));
 const originals=unrelated.map(controller=>Object.getOwnPropertyDescriptors(Object.getPrototypeOf(controller)));
 await f.render(kind,[action,...unrelated]);
 await f.render(kind,[action,...unrelated]);
 assert.deepEqual(f.errors,[],'normal mixed HUD contents must not trigger action-compatibility warnings');
 unrelated.forEach((controller,i)=>assert.deepEqual(Object.getOwnPropertyDescriptors(Object.getPrototypeOf(controller)),originals[i]));
 const event={type:'click'},pending=action.use(event);let settled=false;pending.then(()=>{settled=true});
 // begin/start are awaited before the captured Toolbelt helper is entered.
 for(let i=0;i<10&&nativeCalls===0;i++)await Promise.resolve();
 assert.equal(nativeCalls,1);assert.equal(seenEvent,event);assert.equal(seenItem,f.item);assert.equal(settled,false);
 assert.deepEqual(f.requests.map(r=>r.name),['metapower:begin','metapower:start']);
 const result={id:'native-card'};finishNative(result);assert.equal(await pending,result);
 assert.equal(nativeCalls,1);assert.deepEqual(f.requests.map(r=>r.name),['metapower:begin','metapower:start','metapower:finish']);
});

for(const kind of ['sidebar','persistent'])test(`changed reviewed ${kind} action method still fails its exact fingerprint gate`,async t=>{
 const f=setup(t),action=actionControllers(f.item)[kind],prototype=Object.getPrototypeOf(action);
 const modified=function(){return 'changed native action'};prototype.use=modified;
 await f.render(kind,[action]);
 assert.equal(f.errors.length,1);assert.match(f.errors[0].message,/HUD use entry differs from the reviewed version/);
 assert.equal(action.use,modified);assert.deepEqual(f.requests,[]);
});

test('reviewed persistent exploration and ineligible actor uses retain their native routes',async t=>{
 let explores=0,nativeCalls=0;const f=setup(t);
 const actions=actionControllers(f.item,{native:()=>{nativeCalls++;return 'native'},explore:(actor,id)=>{assert.equal(actor,f.actor);assert.equal(id,'i');explores++}});
 await f.render('persistent',[actions.persistent]);actions.persistent.isExploration=true;
 assert.equal(actions.persistent.use({}),undefined);assert.equal(explores,1);
 await f.render('sidebar',[actions.sidebar]);f.actor.items.clear();
 assert.equal(actions.sidebar.use({}),'native');assert.equal(nativeCalls,1);assert.deepEqual(f.requests,[]);assert.deepEqual(f.errors,[]);
});
