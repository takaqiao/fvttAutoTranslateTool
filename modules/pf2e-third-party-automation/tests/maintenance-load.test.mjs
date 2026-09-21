import test from 'node:test';
import assert from 'node:assert/strict';
import {createMaintenance,repairCandidates} from '../scripts/maintenance.mjs';
import {createPartyAutomation,PARTY_SOURCES} from '../scripts/party-automation.mjs';
const ID='pf2e-third-party-automation';
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function bus(){const hooks=new Map();return {on(n,f){const list=hooks.get(n)??[];list.push(f);hooks.set(n,list);return f},off(n,f){hooks.set(n,hooks.get(n).filter(x=>x!==f))},async emit(n,...args){await Promise.all((hooks.get(n)??[]).map(f=>f(...args)));await flush()}}}
function legacy(){
 let reads=0,repairs=0;const gm={id:'gm'},members=Array.from({length:5},(_,i)=>({uuid:`Actor.${i}`,type:'character',flags:{},data:{type:'character',items:[]},toObject(){reads++;return this.data}}));
 const game={user:gm,users:{activeGM:gm},actors:{party:{members}}},run=createMaintenance({game,repair:async()=>{repairs++}});
 return {game,members,run,counts:()=>({reads,repairs})};
}
test('single actor maintenance serializes only that eligible party member',async()=>{const f=legacy();for(const a of f.members)await f.run(a);assert.deepEqual(f.counts(),{reads:5,repairs:0})});
test('excluded or absent actors do not serialize the party',async()=>{const f=legacy();f.members[0].flags[ID]={autoRepairDisabled:true};await f.run(f.members[0]);await f.run({uuid:'Actor.absent'});assert.deepEqual(f.counts(),{reads:0,repairs:0})});
test('whole party repair plans are built once and repairs remain deduplicated',async()=>{
 const f=legacy();for(const a of f.members)a.data.items=[{_id:'weapon',type:'weapon',_stats:{compendiumSource:'Compendium.pf2e-team-plus-magic.items.Item.xG5u93m95hjqPR4b'},system:{description:{value:'((@item.system.runes.striking + 2) * 2)'},rules:[]}}];
 await f.run();assert.deepEqual(f.counts(),{reads:5,repairs:5});assert.equal(repairCandidates(f.game).length,5);
});
function party(t){
 const old=globalThis.canvas;const hooks=bus(),gm={id:'gm'},actors=new Map(),docs=new Map(),errors=[];let rules=0,deletes=0,lookups=0,distance=5;
 const game={user:gm,users:{activeGM:gm},actors,time:{worldTime:0}};const scene={id:'s',tokens:new Map()};globalThis.canvas={scene};t.after(()=>{globalThis.canvas=old});
 const make=(id,effects=[])=>{const ordinary=Array.from({length:100},()=>({type:'feat',system:{get rules(){rules++;return []}},flags:{}}));const a={id,uuid:`Actor.${id}`,items:[...ordinary,...effects],itemTypes:{effect:effects},flags:{},attributes:{shield:{raised:true,itemId:'shield'}}};actors.set(id,a);return a};
 const source=make('source'),target=make('target');source.items.push({sourceId:PARTY_SOURCES.guardian});
 const token=(id,actor)=>{const x={id,uuid:`Scene.s.Token.${id}`,parent:scene,actor,object:{distanceTo:()=>distance}};scene.tokens.set(id,x);docs.set(x.uuid,x);return x};const s=token('s',source),d=token('t',target),other=token('other',make('other'));
 const effect={id:'effect',type:'effect',system:{rules:[]},flags:{[ID]:{party:{kind:'guardian',sourceToken:s.uuid,targetToken:d.uuid,shieldId:'shield'}}},async delete(){deletes++;target.items=target.items.filter(x=>x!==this);target.itemTypes.effect=target.itemTypes.effect.filter(x=>x!==this)}};
 const add=()=>{target.items.push(effect);target.itemTypes.effect.push(effect);effect.actor=target};
 const provider=createPartyAutomation({game,castEvents:{addMatcher(){}},fromUuid:async uuid=>{lookups++;return docs.get(uuid)},onError:e=>errors.push(e)}),cleanup=provider.register({Hooks:hooks});t.after(cleanup);
 return {hooks,game,source,target,s,d,other,effect,add,provider,docs,errors,setDistance:n=>distance=n,counts:()=>({rules,deletes,lookups})};
}
test('ordinary movement does not run legacy item repairs or resolve unrelated guardians',async t=>{
 const f=party(t);f.add();for(let i=0;i<3;i++)await f.hooks.emit('updateToken',f.other,{x:i});assert.deepEqual(f.counts(),{rules:0,deletes:0,lookups:0});
});
for(const who of ['s','d'])test(`moving guardian ${who} still ends protection outside adjacency`,async t=>{const f=party(t);f.add();f.setDistance(10);await f.hooks.emit('updateToken',f[who],{x:1});assert.equal(f.counts().deletes,1);assert.equal(f.counts().rules,0);assert.deepEqual(f.errors,[])});
test('native effect bucket replacement and newly created effects are read on the next movement',async t=>{const f=party(t);await f.hooks.emit('updateToken',f.s,{x:1});f.add();f.target.itemTypes={effect:[f.effect]};f.setDistance(10);await f.hooks.emit('updateToken',f.s,{elevation:5});assert.equal(f.counts().deletes,1)});
test('time maintenance still expires anoint and restores clue frequency',async t=>{
 const f=party(t);f.effect.flags[ID].party={kind:'anoint',expiresAt:10};f.add();let restored=0,cleared=0;f.source.flags[ID]={party:{clueUntil:10}};f.source.items.push({sourceId:PARTY_SOURCES.clue,system:{frequency:{max:1}},update:async c=>{restored=c['system.frequency.value']}});f.source.update=async()=>{cleared++};f.game.time.worldTime=11;await f.hooks.emit('updateWorldTime');assert.equal(f.counts().deletes,1);assert.equal(restored,1);assert.equal(cleared,1);
});
test('GM handoff checks existing guardians; former GM movement does no work',async t=>{
 const f=party(t);f.add();f.setDistance(10);f.game.users.activeGM={id:'new'};await f.hooks.emit('updateToken',f.s,{x:2});assert.deepEqual(f.counts(),{rules:0,deletes:0,lookups:0});f.game.user=f.game.users.activeGM;await f.hooks.emit('userConnected',f.game.user,true);assert.equal(f.counts().deletes,1);
});
test('token relinking retains a conservative maintenance checkpoint',async t=>{const f=party(t);f.add();f.docs.delete(f.s.uuid);await f.hooks.emit('updateToken',f.s,{actorId:'replacement',x:3});assert.equal(f.counts().deletes,1)});
test('queued work cannot write after losing active GM authority',async t=>{
 const f=party(t);f.add();f.setDistance(10);const pending=f.provider.maintain(f.target);f.game.users.activeGM={id:'other'};await pending;assert.equal(f.counts().deletes,0);
});
test('GM handoff cannot reuse a Raise Shield action followed by movement while inactive',async t=>{
 const f=party(t),gm=f.game.user,guardian={id:'guardian',sourceId:PARTY_SOURCES.guardian},shield={id:'shield'};f.source.items=new Map([[guardian.id,guardian],[shield.id,shield]]);f.source.type='character';f.source.testUserPermission=()=>true;f.s.documentName=f.d.documentName='Token';let created=0;f.target.createEmbeddedDocuments=async()=>{created++;return []};
 await f.hooks.emit('createChatMessage',{actor:f.source,item:{slug:'raise-a-shield'}});
 f.game.users.activeGM={id:'other'};await f.hooks.emit('userConnected',f.game.users.activeGM,true);await f.hooks.emit('updateToken',f.s,{x:100});f.game.users.activeGM=gm;await f.hooks.emit('userConnected',gm,true);
 await assert.rejects(f.provider.executeUsage({actor:f.source,item:guardian,user:gm,action:'party:guardian',message:{speaker:{scene:'s',token:'s'},flags:{pf2e:{context:{target:{token:f.d.uuid}}}}}}),/上一个动作是举盾/);assert.equal(created,0);
});
