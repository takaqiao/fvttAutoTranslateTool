import test from 'node:test';
import assert from 'node:assert/strict';
import {createCampaignFeats} from '../scripts/campaign-feats.mjs';

const ID='pf2e-third-party-automation';
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function deferred(){let resolve;const promise=new Promise(done=>resolve=done);return {promise,resolve}}
function collection(rows=[]){
 const map=new Map(rows.map(row=>[row.id,row])),values=map.values.bind(map);
 let reads=0;map.values=()=>{reads++;return values()};
 return {map,reads:()=>reads};
}
function effect(id,kind='campaign-shatter',timing){return {id,type:'effect',flags:{[ID]:{kind,...timing?{timing}:{}}},system:{rules:[]}}}
function actor(id,rows=[],uuid=`Actor.${id}`){
 const items=collection(rows),deletes=[],conditions=[];
 const document={id,uuid,type:'character',items:items.map,flags:{},frightened:false,
  hasCondition(name){conditions.push(name);return name==='frightened'&&this.frightened},
  async deleteEmbeddedDocuments(type,ids){
   assert.equal(type,'Item');deletes.push([...ids]);
   await this.beforeDelete?.(ids);
   for(const id of ids)this.items.delete(id);
   return [];
  }
 };
 return {document,items,deletes,conditions};
}
function fixture(t,actors=[],scenes=[]){
 const callbacks=new Map(),errors=[],gm={id:'gm',isGM:true,active:true};
 const worldActors=collection(actors.map(a=>a.document)),worldScenes=collection(scenes);
 const game={user:gm,users:{activeGM:gm},actors:worldActors.map,scenes:worldScenes.map,combat:null};
 const Hooks={on(name,callback){callbacks.set(name,callback);return callback},off(name){callbacks.delete(name)}};
 const provider=createCampaignFeats({game,onError:error=>errors.push(error)}),cleanup=provider.register({Hooks});
 t.after(()=>{cleanup();assert.deepEqual(errors,[])});
 const emit=async(name,...args)=>{callbacks.get(name)(...args);await flush()};
 return {game,gm,provider,emit,worldActors,worldScenes};
}

for(const hook of ['updateCombat','deleteCombat']){
 for(const isGM of [false,true])test(`${hook} skips actor and scene enumeration for ${isGM?'a secondary GM':'a player'}`,async t=>{
  const target=actor('target',[effect('shatter')]),tokens=collection([{id:'token',actor:target.document}]);
  const f=fixture(t,[target],[{id:'scene',tokens:tokens.map}]);
  f.game.user={id:'other',isGM,active:true};
  await f.emit(hook,{},{});
  assert.deepEqual([f.worldActors.reads(),f.worldScenes.reads(),tokens.reads(),target.items.reads()],[0,0,0,0]);
  assert.deepEqual(target.deletes,[]);
 });

 test(`${hook} resumes cleanup on the next event after a GM handoff`,async t=>{
  const target=actor('target',[effect('shatter')]),f=fixture(t,[target]);
  const next={id:'next',isGM:true,active:true};f.game.users.activeGM=next;
  await f.emit(hook,{},{});
  assert.equal(f.worldActors.reads(),0);assert.deepEqual(target.deletes,[]);
  f.game.user=next;
  await f.emit(hook,{},{});
  assert.deepEqual(target.deletes,[['shatter']]);assert.equal(target.items.map.size,0);
  assert.deepEqual([f.worldActors.reads(),f.worldScenes.reads()],[1,1]);
 });
}

for(const shatters of [[],[effect('first'),effect('second')]])test(`cleanup scans inventory once with ${shatters.length} shatter effects and no expired effects`,async t=>{
 const ordinary={id:'ordinary',type:'feat',flags:{},system:{rules:[]}};
 const target=actor('target',[ordinary,...shatters]),f=fixture(t,[target]);
 await f.provider.cleanEffects(target.document);
 assert.deepEqual(target.deletes,shatters.length?[['first','second']]:[]);
 assert.deepEqual([...target.items.map.keys()],['ordinary']);
 assert.deepEqual(target.conditions,['frightened']);assert.equal(target.items.reads(),1);
});

test('cleanup keeps shatter while frightened without a second inventory scan',async t=>{
 const target=actor('target',[effect('shatter')]),f=fixture(t,[target]);target.document.frightened=true;
 await f.provider.cleanEffects(target.document);
 assert.deepEqual(target.deletes,[]);assert.equal(target.items.reads(),1);
 assert.deepEqual([...target.items.map.keys()],['shatter']);
});

const expiredTiming={combatId:'combat',combatantId:'turn',endRound:1,expiry:'turn-start'};
const liveTiming={...expiredTiming,endRound:3};
for(const scenario of [
 {name:'unchanged shatter',initial:true,after:['shatter'],deletes:[['expired'],['shatter']],remaining:['live']},
 {name:'new shatter',initial:false,after:['new'],deletes:[['expired'],['new']],remaining:['live']},
 {name:'replaced shatter',initial:true,after:['new'],replace:true,deletes:[['expired'],['new']],remaining:['live','shatter']},
 {name:'removed shatter',initial:true,after:[],deletes:[['expired']],remaining:['live']},
 {name:'frightened ending',initial:true,frightened:true,after:['shatter'],deletes:[['expired'],['shatter']],remaining:['live']},
 {name:'frightened starting',initial:true,after:['shatter'],afterFrightened:true,deletes:[['expired']],remaining:['live','shatter']},
])test(`cleanup rereads ${scenario.name} after deleting expired effects`,async t=>{
 const target=actor('target',[effect('expired','campaign-frightened-floor',expiredTiming),effect('live','campaign-cutting',liveTiming),...scenario.initial?[effect('shatter')]:[]]);
 const f=fixture(t,[target]),reached=deferred(),release=deferred();
 f.game.combat={id:'combat',started:true,round:2,turn:0,turns:[{id:'turn'}]};
 target.document.frightened=scenario.frightened??false;
 target.document.beforeDelete=async ids=>{if(ids.includes('expired')){reached.resolve();await release.promise}};
 const pending=f.provider.cleanEffects(target.document);
 await reached.promise;
 assert.equal(target.items.reads(),1);assert.deepEqual(target.conditions,[]);
 assert.deepEqual(target.deletes,[['expired']]);
 target.items.map.delete('shatter');
 if(scenario.replace)target.items.map.set('shatter',effect('shatter','unrelated-effect'));
 for(const id of scenario.after)target.items.map.set(id,effect(id));
 target.document.frightened=scenario.afterFrightened??false;
 release.resolve();await pending;
 assert.deepEqual(target.deletes,scenario.deletes);
 assert.deepEqual([...target.items.map.keys()],scenario.remaining);
 assert.deepEqual(target.conditions,['frightened']);
 assert.equal(target.items.reads(),scenario.afterFrightened?1:2);
});

test('consecutive combat events queue cleanup and read inventory when each event runs',async t=>{
 const target=actor('target',[effect('first')]),f=fixture(t,[target]),reached=deferred(),release=deferred(),trace=[];
 target.document.beforeDelete=async ids=>{
  trace.push(`start:${ids.join(',')}`);
  if(ids.includes('first')){reached.resolve();await release.promise}
  trace.push(`end:${ids.join(',')}`);
 };
 const first=f.emit('updateCombat',{},{});await reached.promise;
 target.items.map.set('second',effect('second'));
 await f.emit('deleteCombat',{});
 assert.deepEqual(trace,['start:first']);assert.equal(target.items.reads(),1);
 release.resolve();await first;await flush();
 assert.deepEqual(trace,['start:first','end:first','start:second','end:second']);
 assert.deepEqual(target.deletes,[['first'],['second']]);assert.equal(target.items.map.size,0);
 assert.equal(target.items.reads(),2);
});

test('combat cleanup waits for one actor before scanning the next actor',async t=>{
 const first=actor('first',[effect('first-effect')]),second=actor('second',[effect('second-effect')]);
 const f=fixture(t,[first,second]),reached=deferred(),release=deferred();
 first.document.beforeDelete=async()=>{reached.resolve();await release.promise};
 const pending=f.emit('updateCombat',{},{});await reached.promise;
 assert.equal(second.items.reads(),0);assert.deepEqual(second.deletes,[]);
 release.resolve();await pending;await flush();
 assert.deepEqual(first.deletes,[['first-effect']]);assert.deepEqual(second.deletes,[['second-effect']]);
});

for(const hook of ['updateCombat','deleteCombat'])test(`${hook} visits world actors and linked and unlinked actors across scenes once per UUID`,async t=>{
 const world=actor('world',[effect('world-effect')]),base=actor('base',[effect('base-effect')]);
 const first=actor('base',[effect('first-effect')],'Scene.first.Token.unlinked.Actor.base');
 const second=actor('base',[effect('second-effect')],'Scene.second.Token.unlinked.Actor.base');
 const scenes=[
  {id:'first',tokens:new Map([['linked',{actor:world.document,actorLink:true}],['unlinked',{actor:first.document,actorLink:false}]])},
  {id:'second',tokens:new Map([['linked',{actor:world.document,actorLink:true}],['unlinked',{actor:second.document,actorLink:false}],['missing',{actor:null}]])},
 ];
 const f=fixture(t,[world,base],scenes);
 await f.emit(hook,{},{});
 assert.deepEqual([world.deletes,base.deletes,first.deletes,second.deletes],[[['world-effect']],[['base-effect']],[['first-effect']],[['second-effect']]]);
 assert.deepEqual([world.items.reads(),base.items.reads(),first.items.reads(),second.items.reads()],[1,1,1,1]);
});

for(const changes of [{},{name:'renamed'},{unknown:true}])test(`combat updates retain cleanup checkpoints for ${JSON.stringify(changes)}`,async t=>{
 const target=actor('target',[effect('shatter')]),f=fixture(t,[target]);
 await f.emit('updateCombat',{},changes);
 assert.deepEqual(target.deletes,[['shatter']]);assert.equal(target.items.map.size,0);
});
