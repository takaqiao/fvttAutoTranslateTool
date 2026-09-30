import test from 'node:test';
import assert from 'node:assert/strict';
import {createPartyAutomation,guardianActive} from '../scripts/party-automation.mjs';
import {createAvAutomation,AV_SOURCES} from '../scripts/av-automation.mjs';
import {createDirtyMaintenance} from '../scripts/maintenance-events.mjs';
const ID='pf2e-third-party-automation';
const flush=()=>new Promise(resolve=>setImmediate(resolve));
test('dirty maintenance merges selected actors and lets a global request cover them',async()=>{
 const scopes=[],maintenance=createDirtyMaintenance({enabled:()=>true,run:scope=>scopes.push(scope),onError:assert.fail});
 await Promise.all([maintenance.request('first'),maintenance.request('second')]);assert.deepEqual([...scopes[0]],['first','second']);
 await Promise.all([maintenance.request('first'),maintenance.request(),maintenance.request('second')]);assert.equal(scopes[1],null);
 maintenance.dispose();
});
function fixture(t,kind){
 const previous=globalThis.canvas;t.after(()=>{globalThis.canvas=previous});globalThis.canvas={scene:{tokens:new Map()}};
 const callbacks=new Map(),Hooks={on(name,fn){const list=callbacks.get(name)??[];list.push(fn);callbacks.set(name,list);return fn},off(name,fn){callbacks.set(name,(callbacks.get(name)??[]).filter(x=>x!==fn))}};
 const emit=(name,...args)=>Promise.all((callbacks.get(name)??[]).map(fn=>fn(...args)));
 const gm={id:'gm'},users=Object.assign(new Map([['gm',gm]]),{activeGM:gm});let reads=0;
 const items=new Map([['ordinary',{id:'ordinary',type:'feat',sourceId:'custom',flags:{},system:{rules:[]}}]]),nativeValues=items.values.bind(items);items.values=()=>{reads++;return nativeValues()};
 const actor={id:'actor',uuid:'Actor.actor',type:'character',items,flags:{},getActiveTokens:()=>[]};
 const game={user:gm,users,actors:new Map([[actor.id,actor]]),scenes:new Map(),time:{worldTime:100},messages:new Map(),combat:null};
 const castEvents={addMatcher(){},addCapture(){},register(){return ()=>{}},addActorUpdateMiddleware(){return ()=>{}}};
 const provider=(kind==='party'?createPartyAutomation:createAvAutomation)({game,castEvents,onError:assert.fail});
 const cleanup=provider.register({Hooks});t.after(cleanup);
 return {game,actor,items,emit,cleanup,callbacks,reads:()=>reads};
}
test('party never registers movement monitoring',t=>{
 const f=fixture(t,'party');assert.equal(f.callbacks.has('updateToken'),false);
});
test('guardian validity uses live shield state without canvas geometry',()=>{
 const source={actor:{attributes:{shield:{raised:true,broken:false,destroyed:false}}}},target={actor:{uuid:'Actor.ally'}};
 assert.equal(guardianActive(source,target),true);
 source.actor.attributes.shield.raised=false;assert.equal(guardianActive(source,target),false);
});
test('AV restore settles a recorded ally without canvas geometry',async t=>{
 const f=fixture(t,'av'),actor=f.actor;actor.level=1;actor.testUserPermission=()=>true;actor.isAllyOf=()=>true;
 actor.flags[ID]={av:{}};actor.update=async change=>{if(change[`flags.${ID}.av.receipts`])actor.flags[ID].av.receipts=change[`flags.${ID}.av.receipts`]};
 const restore={id:'restore',uuid:'Actor.actor.Item.restore',type:'action',sourceId:AV_SOURCES.restore,actor,system:{rules:[]}};
 f.items.set('psyche',{id:'psyche',type:'effect',slug:'unleash-psyche',system:{rules:[]}});f.items.set(restore.id,restore);
 let healed=0;const target={uuid:'Actor.ally',items:new Map(),async createEmbeddedDocuments(_type,data){return data.map((d,i)=>({...d,id:String(i)}))},async applyDamage({damage}){healed-=damage}};
 const token={uuid:'Scene.scene.Token.ally',actor:target},provider=createAvAutomation({game:f.game,castEvents:{addMatcher(){},addCapture(){}},fromUuid:async()=>token,choose:async()=> 'healing'});
 await provider.executeUsage({actor,item:restore,message:{id:'restore-message',flags:{[ID]:{usageInput:{targetUuids:[token.uuid]}}}},user:f.game.user,action:'av:restore'});
 assert.equal(healed,4);
});
for(const kind of ['party','av']){
 test(`${kind} maintenance coalesces a synchronous burst`,async t=>{
  const f=fixture(t,kind);await Promise.all(Array.from({length:8},()=>f.emit('updateWorldTime')));await flush();assert.equal(f.reads(),1);
 });
 test(`${kind} maintenance repeats once for changes arriving during a document write`,async t=>{
  const f=fixture(t,kind);let release,reached;const atWrite=new Promise(resolve=>reached=resolve),blocked=new Promise(resolve=>release=resolve);
  const state={kind:'anoint',expiresAt:1},effect={id:'effect',type:'effect',sourceId:'custom',system:{rules:[]},flags:{[ID]:kind==='party'?{party:state}:{av:state}}};
  effect.actor=f.actor;effect.delete=async()=>{reached();await blocked;f.items.delete(effect.id)};f.items.set(effect.id,effect);
  f.actor.deleteEmbeddedDocuments=async(_type,ids)=>{reached();await blocked;for(const id of ids)f.items.delete(id)};
  const first=f.emit('updateWorldTime');await atWrite;const more=Array.from({length:5},()=>f.emit('updateWorldTime'));release();await Promise.all([first,...more]);await flush();assert.equal(f.reads(),2);
 });
 test(`${kind} cosmetic combat updates skip inventory maintenance`,async t=>{
  const f=fixture(t,kind);await f.emit('updateCombat',{}, {name:'renamed',_stats:{modifiedTime:5}});await flush();assert.equal(f.reads(),0);
  await f.emit('updateCombat',{}, {round:2});await flush();assert.equal(f.reads(),1);
 });
 test(`${kind} queued maintenance stops on authority loss or disposal`,async t=>{
  const f=fixture(t,kind),pending=f.emit('updateWorldTime');f.game.users.activeGM={id:'other'};await pending;await flush();assert.equal(f.reads(),0);
  f.game.users.activeGM=f.game.user;const disposed=f.emit('updateWorldTime');f.cleanup();await disposed;await flush();assert.equal(f.reads(),0);
 });
}
