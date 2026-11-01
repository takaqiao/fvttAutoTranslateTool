import test from 'node:test';
import assert from 'node:assert/strict';
import {createEldamonElectricityProvider} from '../scripts/eldamon-electricity-provider.mjs';
import * as electricity from '../scripts/eldamon-electricity.mjs';
import {fixture} from './eldamon-electricity-fixture.mjs';

const ID='pf2e-third-party-automation',S=electricity.ELECTRICITY_SOURCES;
function instrument(actor){
 const stats={scans:0,visits:0},values=actor.items.values.bind(actor.items);
 actor.items.values=function*(){stats.scans++;for(const item of values()){stats.visits++;yield item}};
 return {stats,reset(){stats.scans=stats.visits=0}};
}
function install(f){
 const callbacks=new Map(),errors=[];
 const Hooks={on(name,fn){callbacks.set(name,[...callbacks.get(name)??[],fn])}};
 const provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),onError:error=>errors.push(error)});
 provider.register({Hooks});
 return {provider,errors,async fire(name,...args){for(const fn of callbacks.get(name)??[])await fn(...args)}};
}

for(const role of ['gm','owner'])for(const event of ['updateActor','createItem','updateItem','deleteItem'])test(`${event} inspects an ordinary inventory at most once per electricity index on ${role}`,async()=>{
 const f=fixture();f.game.user=f[role];
 for(let n=0;n<100;n++)f.item(f.target,`ordinary-${n}`,'ordinary',{type:'weapon'});
 const meter=instrument(f.target),runtime=install(f);meter.reset();
 await runtime.fire(event,event==='updateActor'?f.target:{actor:f.target},event==='updateActor'?{'system.attributes.hp.value':15}:{'system.quantity':2},{},f.game.user.id);
 assert.ok(meter.stats.scans<=2,JSON.stringify(meter.stats));
 assert.ok(meter.stats.visits<=200,JSON.stringify(meter.stats));
 assert.deepEqual(f.target.flags,{});assert.deepEqual(runtime.errors,[]);
});

test('ordinary turn checks and pure token movement do not revisit inactive inventories',async()=>{
 const f=fixture(),meter=instrument(f.target),runtime=install(f);meter.reset();
 await runtime.fire('updateToken',f.tokens[1],{x:100,y:200});
 await runtime.fire('pf2e.endTurn',f.combat.turns[0],f.combat);
 assert.deepEqual(meter.stats,{scans:0,visits:0});assert.deepEqual(f.target.flags,{});assert.deepEqual(runtime.errors,[]);
});

test('membership uses pending work, charge presence and only timed shock effects',()=>{
 const cases=[
  {name:'ordinary',flags:{},items:[],want:false},
  {name:'pending shock',flags:{[ID]:{electricity:{pendingShocks:{p:{}}}}},items:[],want:true},
  {name:'started operation',flags:{[ID]:{electricity:{operations:{p:{status:'started'}}}}},items:[],want:true},
  {name:'completed operation',flags:{[ID]:{electricity:{operations:{p:{status:'done'}}}}},items:[],want:false},
  {name:'zero charge still active',flags:{},items:[{sourceId:S.charged,system:{badge:{value:0}}}],want:true},
  {name:'legacy charged source',flags:{},items:[{_stats:{compendiumSource:S.charged.replace('.Item.','.')}}],want:true},
  {name:'untimed shock',flags:{},items:[{sourceId:S.shocked}],want:false},
  {name:'empty expiry still truthy',flags:{},items:[{sourceId:S.shocked,flags:{[ID]:{electricityShock:{expires:{}}}}}],want:true},
  {name:'false expiry',flags:{},items:[{sourceId:S.shocked,flags:{[ID]:{electricityShock:{expires:0}}}}],want:false},
  {name:'core source fallback',flags:{},items:[{flags:{core:{sourceId:S.shocked.replace('.Item.','.')},[ID]:{electricityShock:{expires:{combatId:'c'}}}}}],want:true},
  {name:'power alone is not active',flags:{},items:[{sourceId:S.surge}],want:false}
 ];
 for(const {name,flags,items,want} of cases){
  const actor={flags,items},before=structuredClone(actor);
  assert.equal(electricity.hasActiveElectricity(actor),want,name);assert.deepEqual(actor,before,name);
 }
 for(const state of [{pendingShocks:{p:{}}},{operations:{p:{status:'started'}}}]){
  assert.equal(electricity.hasActiveElectricity({flags:{[ID]:{electricity:state}},get items(){assert.fail('pending work must not read inventory')}}),true);
 }
});

test('new timed shock is cleaned once by the active GM and its deletion removes membership',async()=>{
 const f=fixture(),runtime=install(f),meter=instrument(f.target);
 const effect=f.item(f.target,'timed-shock',S.shocked.replace('.Item.','.'),{flags:{[ID]:{electricityShock:{expires:{combatId:'c',combatantId:'casterturn',round:1,phase:'end'}}}}});
 await runtime.fire('createItem',effect);
 let deletions=0;const original=f.target.deleteEmbeddedDocuments.bind(f.target);
 const update=f.target.update.bind(f.target);f.target.update=async changes=>{await update(changes);await runtime.fire('updateActor',f.target,changes)};
 f.target.deleteEmbeddedDocuments=async(type,ids)=>{deletions++;await original(type,ids);for(const id of ids)await runtime.fire('deleteItem',{id,actor:f.target})};
 f.game.user=f.owner;await runtime.fire('pf2e.endTurn',f.combat.turns[0],f.combat);assert.ok(f.target.items.has(effect.id));assert.equal(deletions,0);
 f.game.user=f.gm;await runtime.fire('pf2e.endTurn',f.combat.turns[0],f.combat);assert.equal(f.target.items.has(effect.id),false);assert.equal(deletions,1);
 meter.reset();await runtime.fire('pf2e.endTurn',f.combat.turns[0],f.combat);
 assert.equal(deletions,1);assert.deepEqual(meter.stats,{scans:0,visits:0});assert.deepEqual(runtime.errors,[]);
});

test('base actor updates refresh timed effects on synthetic dependents',async()=>{
 const f=fixture(),synthetic=f.target;synthetic.isToken=true;
 f.game.actors.delete(synthetic.id);synthetic.uuid='Scene.s.Token.target.Actor.target';
 f.game.scenes=new Map([['s',f.scene]]);f.caster.getDependentTokens=()=>[f.tokens[0],f.tokens[1]];
 const effect=f.item(synthetic,'timed-shock',S.shocked,{flags:{[ID]:{electricityShock:{}}}}),runtime=install(f);
 effect.flags[ID].electricityShock.expires={combatId:'c',combatantId:'casterturn',round:1,phase:'end'};
 await runtime.fire('updateActor',f.caster,{});await runtime.fire('pf2e.endTurn',f.combat.turns[0],f.combat);
 assert.equal(synthetic.items.has(effect.id),false);assert.deepEqual(runtime.errors,[]);
});
