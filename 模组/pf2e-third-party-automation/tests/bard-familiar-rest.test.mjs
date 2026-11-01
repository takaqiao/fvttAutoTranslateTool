import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createBardFamiliarProvider,BARD_FAMILIAR_SOURCES as S} from '../scripts/bard-familiar.mjs';
import {getNativeCastEvents} from '../scripts/amp-cast-events.mjs';

function fixture({asGM=false}={}){
 const player={id:'player'},gm={id:'gm',isGM:true},user=asGM?gm:player;
 const actor=(id,type)=>({id,uuid:`Actor.${id}`,type,system:{},items:new Map(),testUserPermission:u=>u===player||u===gm});
 const master=actor('master','character'),familiar=actor('bell','familiar'),other=actor('other','character');
 master.system.resources={focus:{value:0,max:2}};familiar.system.master={id:master.id};
 const game={user,users:{activeGM:gm},world:{id:'bob-2'},system:{id:'pf2e'},actors:new Map([master,familiar,other].map(a=>[a.id,a]))};
 Object.defineProperty(familiar,'master',{get:()=>game.actors.get(familiar.system.master.id)});
 const focus={id:'focus',type:'action',sourceId:S.focus,actor:familiar,system:{frequency:{value:0,max:1,per:'day'}},get frequency(){return this.system.frequency;}};
 familiar.items.set(focus.id,focus);
 const recharges=[],writes=[],errors=[],callbacks=new Map();
 familiar.recharge=async options=>{
  recharges.push(options);
  if(options.duration!=='day')return;
  for(const item of familiar.items.values())if(item.system.frequency?.per==='day'&&item.system.frequency.value<item.system.frequency.max){
   writes.push({id:item.id,value:item.system.frequency.max});item.system.frequency.value=item.system.frequency.max;
  }
 };
 const Hooks={on(name,fn){const list=callbacks.get(name)??[];list.push(fn);callbacks.set(name,list);return fn;},off(name,fn){callbacks.set(name,(callbacks.get(name)??[]).filter(value=>value!==fn));}};
 const options={game,onError:error=>errors.push(error)},provider=createBardFamiliarProvider(options),unregister=provider.register({Hooks});
 const rest=(target=master)=>Promise.all((callbacks.get('pf2e.restForTheNight')??[]).map(fn=>fn(target)));
 return {game,master,familiar,other,focus,recharges,writes,errors,Hooks,options,rest,unregister};
}

for(const asGM of [false,true])test(`the local ${asGM?'GM':'player'} rest restores Familiar Focus without changing the master`,async()=>{
 const f=fixture({asGM}),masterBefore=structuredClone(f.master.system),familiarBefore=[...f.familiar.items.keys()];
 await f.rest();
 assert.equal(f.focus.system.frequency.value,1);assert.deepEqual(f.recharges,[{duration:'day'}]);
 assert.deepEqual(f.master.system,masterBefore);assert.deepEqual([...f.familiar.items.keys()],familiarBefore);assert.equal(f.familiar.master,f.master);assert.deepEqual(f.errors,[]);
});

test('repeated rest hooks and providers do not recharge an already restored familiar',async()=>{
 const f=fixture();createBardFamiliarProvider(f.options).register({Hooks:f.Hooks});
 await Promise.all([f.rest(),f.rest()]);await f.rest();
 assert.equal(f.focus.system.frequency.value,1);assert.equal(f.recharges.length,1);assert.deepEqual(f.writes,[{id:'focus',value:1}]);
});

for(const[reason,change]of[
 ['another world',f=>f.game.world.id='avecnu'],
 ['another system',f=>f.game.system.id='sf2e'],
 ['an unowned master',f=>f.master.testUserPermission=()=>false],
 ['an unowned familiar',f=>f.familiar.testUserPermission=()=>false],
 ['another master',f=>f.familiar.system.master.id=f.other.id],
 ['a missing master',f=>f.game.actors.delete(f.master.id)],
 ['a removed familiar',f=>f.game.actors.delete(f.familiar.id)],
 ['a familiar rest',f=>{f.rest=(()=>{const rest=f.rest;return()=>rest(f.familiar);})();}],
 ['a stale master document',f=>{f.rest=(()=>{const rest=f.rest;return()=>rest({...f.master});})();}],
 ['a wrong ability source',f=>f.focus.sourceId=S.accompanist],
 ['a wrong ability type',f=>f.focus.type='feat'],
 ['no Focus ability',f=>f.familiar.items.delete(f.focus.id)],
 ['a non-daily frequency',f=>f.focus.system.frequency.per='PT1H'],
 ['a full daily frequency',f=>f.focus.system.frequency.value=1],
 ['no frequency',f=>delete f.focus.system.frequency],
])test(`rest ignores ${reason}`,async()=>{
 const f=fixture();change(f);await f.rest();assert.deepEqual(f.recharges,[]);assert.deepEqual(f.writes,[]);assert.deepEqual(f.errors,[]);
});

test('rest waits for the master resource lock before recharging the familiar',async()=>{
 const f=fixture(),casts=getNativeCastEvents({game:f.game});let release;
 const held=casts.withActorResourceLock(f.master,()=>new Promise(resolve=>release=resolve));
 const resting=f.rest();await new Promise(resolve=>setImmediate(resolve));
 assert.equal(f.focus.system.frequency.value,0);assert.equal(f.recharges.length,0);
 release();await held;await resting;assert.equal(f.focus.system.frequency.value,1);
});

test('a familiar rebound while rest waits is not recharged by the previous master',async()=>{
 const f=fixture(),casts=getNativeCastEvents({game:f.game});let release;
 const held=casts.withActorResourceLock(f.master,()=>new Promise(resolve=>release=resolve));
 const resting=f.rest();await new Promise(resolve=>setImmediate(resolve));
 f.familiar.system.master.id=f.other.id;release();await held;await resting;
 assert.equal(f.focus.system.frequency.value,0);assert.equal(f.recharges.length,0);
 await f.rest(f.other);assert.equal(f.focus.system.frequency.value,1);
});

test('rest reports a native recharge failure and can retry the still-used frequency',async()=>{
 const f=fixture(),recharge=f.familiar.recharge;f.familiar.recharge=async()=>{throw Error('recharge failed');};
 await f.rest();assert.equal(f.focus.system.frequency.value,0);assert.equal(f.errors.length,1);assert.match(f.errors[0].message,/recharge failed/);
 f.familiar.recharge=recharge;await f.rest();assert.equal(f.focus.system.frequency.value,1);
});

test('unregister removes the rest listener',async()=>{
 const f=fixture();f.unregister();await f.rest();assert.equal(f.focus.system.frequency.value,0);assert.equal(f.recharges.length,0);
});

const bundle=process.env.PF2E_NATIVE_861_BUNDLE??process.env.FVTT_PF2E_861_BUNDLE??'';
test('rest delegates daily frequency updates to the real PF2e 8.6.1 recharge method',{skip:!fs.existsSync(bundle)},async()=>{
 const f=fixture(),source=fs.readFileSync(bundle,'utf8'),start=source.indexOf('\tasync recharge(e) {'),end=source.indexOf('\n\tstatic createDialog(',start);
 assert(start>=0&&end>start,'native ActorPF2e.recharge exists');
 const method=source.slice(start,end),createStart=source.indexOf('function createActorGroupUpdate('),createEnd=source.indexOf('\nasync function applyActorGroupUpdate(',createStart);
 assert(createStart>=0&&createEnd>createStart,'native group update builder exists');
 const updates=[],apply=async(actor,changes)=>{
  assert.equal(actor,f.familiar);assert.deepEqual(changes.actorUpdates,{});assert.deepEqual(changes.itemCreates,[]);assert.deepEqual(changes.itemDeletes,[]);
  updates.push(structuredClone(changes.itemUpdates));
  for(const update of changes.itemUpdates)Object.assign(actor.items.get(update._id).system.frequency,update.system.frequency);
 };
 f.familiar.itemTypes={action:[f.focus],feat:[]};f.familiar.synthetics={resources:{}};
 f.familiar.recharge=new Function('applyActorGroupUpdate',`${source.slice(createStart,createEnd)}\nreturn ({${method}}).recharge;`)(apply);
 await f.rest();await f.rest();
 assert.equal(f.focus.system.frequency.value,1);assert.deepEqual(updates,[[{_id:'focus',system:{frequency:{value:1}}}]]);assert.equal(f.master.system.resources.focus.value,0);assert.deepEqual(f.errors,[]);
});
