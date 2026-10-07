import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {runInNewContext} from 'node:vm';
import {createKnowledgeAutomation,KNOWLEDGE_SOURCES} from '../scripts/knowledge-automation.mjs';
import {MODULE_ID as M} from '../scripts/rules.mjs';

function hooks(){
 const events=new Map();let id=0;
 return {on(name,fn){const rows=events.get(name)??new Map();events.set(name,rows);rows.set(++id,fn);return id},off(name,key){events.get(name)?.delete(key)},async emit(name,...args){await Promise.all([...events.get(name)?.values()??[]].map(fn=>fn(...args)))},count(){return [...events.values()].reduce((n,rows)=>n+rows.size,0)}};
}
function connection(){
 const events=new Map();
 return {on(name,fn){const callbacks=events.get(name)??new Set();events.set(name,callbacks);callbacks.add(fn)},off(name,fn){events.get(name)?.delete(fn)},emit(name){for(const fn of [...events.get(name)??[]])fn()},count(name){return events.get(name)?.size??0}};
}
function fixture(count=10,{fromUuid,socket}={}){
 const Hooks=hooks(),gm={id:'gm',isGM:true},game={user:gm,users:{activeGM:gm,get:id=>id==='gm'?gm:null},actors:new Map(),scenes:new Map(),messages:new Map(),modules:new Map(),time:{worldTime:100},socket};
 const combat={id:'combat',started:true,round:2,turn:0,turns:[{id:'first'},{id:'second'}]};game.combat=combat;
 const reads=new Map(),itemVisits=new Map(),deleted=[],updates=[],errors=[];
 function actor(id){
  const doc={id,uuid:`Actor.${id}`,type:'character',flags:{},items:new Map(),testUserPermission:()=>true,
   async deleteEmbeddedDocuments(type,ids){assert.equal(type,'Item');deleted.push([this.uuid,...ids]);for(const id of ids)this.items.delete(id);return ids},
   async update(changes){updates.push([this.uuid,changes]);for(const[path,value]of Object.entries(changes)){const keys=path.split('.');let at=this;for(const key of keys.slice(0,-1))at=at[key]??={};at[keys.at(-1)]=structuredClone(value)}await Hooks.emit('updateActor',this,changes);return this}};
  const original=doc.items.values.bind(doc.items);doc.items.values=()=>{reads.set(doc.uuid,(reads.get(doc.uuid)??0)+1);return function*(){for(const item of original()){itemVisits.set(doc.uuid,(itemVisits.get(doc.uuid)??0)+1);yield item}}()};return doc;
 }
 for(let i=0;i<count;i++){const doc=actor(`a${i}`);doc.items.set('ordinary',{id:'ordinary',type:'weapon'});game.actors.set(doc.id,doc)}
 function effect(doc,id='expiry'){
  const item={id,uuid:`${doc.uuid}.Item.${id}`,actor:doc,type:'effect',flags:{[M]:{knowledge:{timing:{combatId:combat.id,combatantId:'first',round:2}}}}};doc.items.set(id,item);return item;
 }
 let scans=0;for(const collection of [game.actors,game.scenes]){const original=collection.values.bind(collection);collection.values=()=>{scans++;return original()}}
 const provider=createKnowledgeAutomation({game,fromUuid,onError:error=>errors.push(error)});
 return {game,Hooks,gm,combat,provider,actor,effect,reads,itemVisits,deleted,updates,errors,reset(){scans=0;reads.clear();itemVisits.clear()},get scans(){return scans}};
}

function nativeHooks(){
 const file='C:/Program Files/Foundry Virtual Tabletop/resources/app/client/helpers/hooks.mjs';
 const digest=()=>createHash('sha256').update(readFileSync(file)).digest('hex');
 const expected='d026527620f2b03aa1924948129629e43e2626ed6a7c591816bf64013f928904';
 assert.equal(digest(),expected);
 const source=readFileSync(file,'utf8').replace('export default class Hooks','class Hooks')+'\nHooks;';
 const Native=runInNewContext(source,{CONFIG:{debug:{hooks:false}},CONST:{vtt:'Foundry'},Error,console:{warn(){},error(){}}});
 const pending=[],errors=[];
 const api={
  on(name,fn){
   const id=Native.on(name,(...args)=>{const result=fn(...args);if(result&&typeof result.then==='function')pending.push(result);return result});
   const rows=Native.events[name];
   if(!Object.hasOwn(rows,'findSplice'))Object.defineProperty(rows,'findSplice',{value:function(predicate){const at=this.findIndex(predicate);return at<0?undefined:this.splice(at,1)[0]}});
   return id;
  },
  off:(...args)=>Native.off(...args),
 };
 api.on('error',(_where,error)=>errors.push(error));
 return {api,errors,pending,callAll:(...args)=>Native.callAll(...args),count:()=>Object.getOwnPropertyNames(Native.events).reduce((sum,name)=>sum+Native.events[name].length,0),verify:()=>assert.equal(digest(),expected),async drain(){while(pending.length)await Promise.all(pending.splice(0))}};
}
function attachNative(f,t){
 const native=nativeHooks();t.after(()=>native.verify());
 // Document updates dispatch synchronously; only the test waits for business callbacks.
 f.Hooks.emit=async(name,...args)=>{native.callAll(name,...args)};
 return native;
}
test('cosmetic combat updates do not inspect actors or expire knowledge effects',async()=>{
 const f=fixture(),target=f.game.actors.get('a0');f.effect(target);const release=f.provider.register({Hooks:f.Hooks});f.reset();
 await f.Hooks.emit('updateCombat',f.combat,{name:'Encounter',_stats:{modifiedTime:101}});
 assert.equal(f.scans,0);assert.equal(f.reads.size,0);assert.deepEqual(f.deleted,[]);release();assert.equal(f.Hooks.count(),0);
});
test('registration defers world recovery until relevant maintenance on GM and player clients',()=>{
 for(const isGM of [true,false]){
  const f=fixture(100);if(!isGM)f.game.user={id:'player',isGM:false};f.reset();const release=f.provider.register({Hooks:f.Hooks});
  try{assert.equal(f.scans,0);assert.equal(f.reads.size,0);assert.equal(f.Hooks.count(),20)}finally{release()}
  assert.equal(f.Hooks.count(),0);
 }
});

for(const change of ['source replaced','synthetic rebound','recipient replaced','unregister','register again','GM changed'])test(`maintenance stops after recipient lookup when ${change}`,async()=>{
 let resume,entered;const waiting=new Promise(resolve=>resume=resolve),started=new Promise(resolve=>entered=resolve);
 const f=fixture(2,{fromUuid:async uuid=>{assert.equal(uuid,'Actor.a1');entered();return waiting}}),source=f.game.actors.get('a0'),recipient=f.game.actors.get('a1');
 if(change==='synthetic rebound'){
  source.isToken=true;source.uuid='Scene.scene.Token.token.Actor.a0';const scene={id:'scene',tokens:new Map()},token={id:'token',uuid:'Scene.scene.Token.token',actor:source,parent:scene};source.token=token;scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);f.game.actors.delete(source.id);
 }
 source.flags[M]={knowledge:{strategistStates:[{id:'state',status:'consumed',combatId:f.combat.id,settledClaim:{id:'claim',actorUuid:recipient.uuid},cleanupDone:false}]}};
 recipient.items.set('claim',{id:'claim',type:'effect',flags:{[M]:{knowledge:{kind:'strategist-claim',claim:'claim'}}}});
 let release=f.provider.register({Hooks:f.Hooks});const pending=f.Hooks.emit('updateCombat',f.combat,{turn:0});await started;
 if(change==='source replaced')f.game.actors.set(source.id,f.actor(source.id));
 if(change==='synthetic rebound'){const replacement=f.actor(source.id);replacement.isToken=true;replacement.uuid=source.uuid;replacement.token=source.token;source.token.actor=replacement;}
 if(change==='recipient replaced')f.game.actors.set(recipient.id,f.actor(recipient.id));
 if(change==='unregister')release();
 if(change==='register again'){release();release=f.provider.register({Hooks:f.Hooks});}
 if(change==='GM changed')f.game.users.activeGM={id:'new-GM',isGM:true};
 resume(recipient);await pending;assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.equal(source.flags[M].knowledge.strategistStates[0].cleanupDone,false);assert.deepEqual(f.errors,[]);release();
});
test('later turn maintenance visits the full cohort and expires current effects',async()=>{
 const f=fixture(100),target=f.game.actors.get('a0');f.effect(target);f.combat.round=1;const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.combat.round=2;f.reset();
 await f.Hooks.emit('updateCombat',f.combat,{turn:0});
 assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);assert.equal(f.scans,2);
 assert.equal([...f.reads.keys()].filter(uuid=>uuid!==target.uuid).length,99);assert.equal(f.reads.size,100);for(const reads of f.reads.values())assert.equal(reads,1);assert.deepEqual(f.errors,[]);release();
});
test('base actor changes discover new knowledge state on existing unlinked tokens',async()=>{
 const f=fixture(),base=f.game.actors.get('a0'),synthetic=f.actor('synthetic');synthetic.isToken=true;synthetic.uuid='Scene.scene.Token.token.Actor.a0';
 const token={id:'token',uuid:'Scene.scene.Token.token',actor:synthetic},scene={id:'scene',tokens:new Map([['token',token]])};token.parent=scene;synthetic.token=token;f.game.scenes.set(scene.id,scene);let dependents=0;base.getDependentTokens=()=>{dependents++;return [token]};
 const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.effect(synthetic);f.reset();await f.Hooks.emit('updateActor',base,{items:[]});
 assert.equal(f.reads.size,0);assert.equal(dependents,0);assert.equal(f.scans,0);f.reset();
 await f.Hooks.emit('updateCombat',f.combat,{round:2});assert.deepEqual(f.deleted,[[synthetic.uuid,'expiry']]);assert.equal(f.scans,2);release();
});
test('unknown combat fields remain conservative and preserve unresolved attack claims',async()=>{
 const f=fixture(),source=f.game.actors.get('a0');source.flags[M]={knowledge:{strategistStates:[{id:'state',status:'claimed',claim:'unresolved',combatId:f.combat.id,targetUuid:'Scene.scene.Token.enemy'}]}};
 source.items.set('stance-feat',{sourceId:KNOWLEDGE_SOURCES.stance});source.items.set('stance-effect',{sourceId:KNOWLEDGE_SOURCES.stanceEffect});
 const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.reset();await f.Hooks.emit('updateCombat',f.combat,{futureField:true});
 assert.equal(source.flags[M].knowledge.strategistStates[0].claim,'unresolved');assert.deepEqual(f.deleted,[]);assert.equal(f.scans,2);assert.deepEqual(f.errors,[]);release();
});
test('unknown claims survive combat deletion without a stance or native result',async()=>{
 const f=fixture(1),source=f.game.actors.get('a0'),state={id:'state',status:'claimed',claim:'unknown',combatId:f.combat.id};source.flags[M]={knowledge:{strategistStates:[state]}};
 const release=f.provider.register({Hooks:f.Hooks});f.game.combat=null;await f.Hooks.emit('deleteCombat',f.combat);
 assert.deepEqual(source.flags[M].knowledge.strategistStates,[state]);assert.deepEqual(f.updates,[]);release();
});
test('scheduled maintenance settles a genuine saved native attack proof once',async()=>{
 const f=fixture(2),source=f.game.actors.get('a0'),recipient=f.game.actors.get('a1'),claim='saved-claim',target='Scene.scene.Token.enemy';
 source.flags[M]={knowledge:{strategistStates:[{id:'state',status:'claimed',claim,actorUuid:recipient.uuid,userId:f.gm.id,targetUuid:target,combatId:f.combat.id}]}};
 source.items.set('stance-feat',{sourceId:KNOWLEDGE_SOURCES.stance});source.items.set('stance-effect',{sourceId:KNOWLEDGE_SOURCES.stanceEffect});
 recipient.items.set('claim',{id:'claim',flags:{[M]:{knowledge:{kind:'strategist-claim',claim}}}});
 const proof={id:'native-proof',isCheckRoll:true,actor:recipient,author:f.gm,rolls:[{total:14,options:{degreeOfSuccess:2}}],flags:{pf2e:{context:{type:'attack-roll',options:[`${M}:knowledge:claim:${claim}`],target:{token:target}}}}};f.game.messages.set(proof.id,proof);
 const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{turn:0});
 const settled=source.flags[M].knowledge.strategistStates[0];assert.equal(settled.status,'consumed');assert.equal(settled.cleanupDone,true);assert.equal(settled.attackMessageId,proof.id);assert.deepEqual(f.deleted,[[recipient.uuid,'claim']]);
 await f.Hooks.emit('updateCombat',f.combat,{round:2});assert.deepEqual(f.deleted,[[recipient.uuid,'claim']]);assert.deepEqual(f.errors,[]);release();
});
test('unregister cancels maintenance waiting behind an actual claim request in the serial queue',async()=>{
 let resume,entered;const waiting=new Promise(resolve=>resume=resolve),started=new Promise(resolve=>entered=resolve);let recipient;
 const f=fixture(2,{fromUuid:async uuid=>{if(uuid==='Actor.a1')return recipient;if(uuid==='Scene.scene.Token.attacker'){entered();return waiting}return null}}),source=f.game.actors.get('a0');recipient=f.game.actors.get('a1');
 source.flags[M]={knowledge:{strategistStates:[{id:'state',status:'consumed',combatId:f.combat.id,settledClaim:{id:'claim',actorUuid:recipient.uuid},cleanupDone:false}]}};
 recipient.items.set('claim',{id:'claim',flags:{[M]:{knowledge:{kind:'strategist-claim',claim:'claim'}}}});
 const release=f.provider.register({Hooks:f.Hooks}),attack=f.provider.claimAttack({actorUuid:recipient.uuid,sourceTokenUuid:'Scene.scene.Token.attacker',targetUuid:'Scene.scene.Token.enemy'},f.gm);await started;
 const maintenance=f.Hooks.emit('updateCombat',f.combat,{turn:0});await new Promise(resolve=>setImmediate(resolve));assert(f.reads.has(source.uuid));release();resume(null);await Promise.all([attack,maintenance]);
 assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.equal(source.flags[M].knowledge.strategistStates[0].cleanupDone,false);assert.deepEqual(f.errors,[]);
});
test('timing-only expiry does not wait behind an unrelated native attack claim',async()=>{
 let resume,entered;const waiting=new Promise(resolve=>resume=resolve),started=new Promise(resolve=>entered=resolve);let recipient;
 const f=fixture(2,{fromUuid:async uuid=>{if(uuid==='Actor.a1')return recipient;if(uuid==='Scene.scene.Token.attacker'){entered();return waiting}return null}}),source=f.game.actors.get('a0');recipient=f.game.actors.get('a1');f.effect(source);
 const release=f.provider.register({Hooks:f.Hooks}),attack=f.provider.claimAttack({actorUuid:recipient.uuid,sourceTokenUuid:'Scene.scene.Token.attacker',targetUuid:'Scene.scene.Token.enemy'},f.gm);await started;
 let done=false;const maintenance=f.Hooks.emit('updateCombat',f.combat,{turn:0}).then(()=>done=true);await new Promise(resolve=>setImmediate(resolve));const completed=done;
 resume(null);await Promise.all([attack,maintenance]);release();assert.equal(completed,true);assert.deepEqual(f.deleted,[[source.uuid,'expiry']]);assert.deepEqual(f.errors,[]);
});
test('unregister cancels maintenance already queued by a combat hook',async()=>{
 const f=fixture(),target=f.game.actors.get('a0');f.effect(target);const release=f.provider.register({Hooks:f.Hooks});
 const pending=f.Hooks.emit('updateCombat',f.combat,{turn:0});release();await pending;
 assert.deepEqual(f.deleted,[]);assert.equal(f.Hooks.count(),0);
});
test('a client that lost GM authority performs no world traversal or expiry',async()=>{
 const f=fixture(),target=f.game.actors.get('a0');f.effect(target);const release=f.provider.register({Hooks:f.Hooks});f.game.users.activeGM={id:'replacement',isGM:true};f.reset();
 await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,0);assert.deepEqual(f.deleted,[]);release();
});
test('a caller supplied callback cannot authorize non-GM maintenance',async()=>{
 const f=fixture(1),target=f.game.actors.get('a0');f.effect(target);f.game.user={id:'player',isGM:false};
 await f.provider.maintain(target,()=>true);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);
});
test('a world actor replaced before a turn cannot expire items on the old document',async()=>{
 const f=fixture(),old=f.game.actors.get('a0');f.effect(old);const release=f.provider.register({Hooks:f.Hooks});f.game.actors.set(old.id,f.actor(old.id));
 await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.deepEqual(f.deleted,[]);release();
});
test('scene token imports and delta changes are discovered on the next maintenance batch',async()=>{
 const f=fixture(),scene={id:'scene',tokens:new Map()};f.game.scenes.set(scene.id,scene);const release=f.provider.register({Hooks:f.Hooks});
 const synthetic=f.actor('synthetic');synthetic.isToken=true;synthetic.uuid='Scene.scene.Token.token.Actor.a0';const token={id:'token',uuid:'Scene.scene.Token.token',actor:synthetic,parent:scene};synthetic.token=token;scene.tokens.set(token.id,token);f.effect(synthetic);
 await f.Hooks.emit('updateScene',scene,{tokens:[{_id:token.id}]});await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.deepEqual(f.deleted,[[synthetic.uuid,'expiry']]);
 const replacement=f.actor('synthetic');replacement.isToken=true;replacement.uuid=synthetic.uuid;replacement.token=token;token.actor=replacement;f.effect(replacement,'later');
 await f.Hooks.emit('updateToken',token,{delta:{items:[]}});await f.Hooks.emit('updateCombat',f.combat,{round:2});assert.deepEqual(f.deleted,[[synthetic.uuid,'expiry'],[replacement.uuid,'later']]);release();
});

for(const position of [0,49])test(`ordinary actor notification does no inventory work for timing at position ${position}`,async()=>{
 const f=fixture(1),target=f.game.actors.get('a0');target.items.clear();
 for(let index=0;index<50;index++)if(index===position)f.effect(target);else target.items.set(`item${index}`,{id:`item${index}`,type:'weapon'});
 f.combat.round=1;const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.reset();await f.Hooks.emit('updateActor',target,{});
 assert.equal(f.itemVisits.size,0);assert.equal(f.reads.size,0);assert.equal(f.scans,0);
 f.combat.round=2;f.reset();
 await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,2);assert.equal(f.reads.get(target.uuid),1);assert.equal(f.itemVisits.get(target.uuid),50);assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);assert.deepEqual(f.errors,[]);release();
});

test('ordinary actor notification preserves an unknown claim without doing maintenance',async()=>{
 const f=fixture(1),target=f.game.actors.get('a0');target.flags[M]={knowledge:{strategistStates:[{id:'state',status:'claimed',claim:'unknown',combatId:f.combat.id}]}};
 f.combat.round=1;const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.reset();await f.Hooks.emit('updateActor',target,{});
 assert.equal(f.reads.size,0);assert.equal(target.flags[M].knowledge.strategistStates[0].claim,'unknown');
 f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,2);assert.equal(f.reads.get(target.uuid),1);assert.equal(target.flags[M].knowledge.strategistStates[0].claim,'unknown');assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);release();
});

for(const client of ['player','non-active-GM'])test(`${client} ordinary notifications do not inspect actors or dependent tokens`,async()=>{
 const f=fixture(1),base=f.game.actors.get('a0'),synthetic=f.actor('synthetic'),item=f.effect(base);synthetic.isToken=true;synthetic.uuid='Scene.scene.Token.token.Actor.a0';f.effect(synthetic);
 const scene={id:'scene',tokens:new Map()},token={id:'token',uuid:'Scene.scene.Token.token',actor:synthetic,parent:scene};synthetic.token=token;scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);
 let dependents=0;base.getDependentTokens=()=>{dependents++;return [token]};f.game.user={id:client,isGM:client!=='player'};
 const release=f.provider.register({Hooks:f.Hooks});f.reset();
 const events=[['createActor',base],['updateActor',base,{futureField:true}],['deleteActor',base],['createItem',item],['updateItem',item,{'-=sourceId':null}],['deleteItem',item],['createToken',token],['updateToken',token,{actorLink:false}],['updateToken',token,{delta:{items:[]}}],['deleteToken',token],['createScene',scene],['updateScene',scene,{tokens:[{_id:token.id}]}],['deleteScene',scene]];
 for(const event of events){await f.Hooks.emit(...event);assert.equal(f.reads.size,0,event[0]);assert.equal(dependents,0,event[0]);assert.equal(f.scans,0,event[0]);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[])}
 assert.deepEqual(f.errors,[]);release();assert.equal(f.Hooks.count(),0);
});

for(const event of ['userConnected','updateUser'])test(`${event} loss and immediate regain defer one fresh recovery until maintenance`,async()=>{
 const f=fixture(1),target=f.game.actors.get('a0'),otherGM={id:'other',isGM:true,active:true};f.effect(target);f.combat.round=1;
 const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.reset();
 f.game.users.activeGM=otherGM;const lost=f.Hooks.emit(event,otherGM,event==='userConnected'?true:{role:4});f.game.users.activeGM=f.gm;const regained=f.Hooks.emit(event,otherGM,event==='userConnected'?false:{role:1});await Promise.all([lost,regained]);
 assert.equal(f.scans,0);assert.equal(f.reads.size,0);assert.deepEqual(f.deleted,[]);
 f.combat.round=2;await f.Hooks.emit('updateCombat',f.combat,{round:2});assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);
 f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);assert.deepEqual(f.errors,[]);release();
});

test('user notifications with unchanged GM authority leave maintenance to the next full cohort',async()=>{
 const f=fixture(1),target=f.game.actors.get('a0');f.effect(target);f.combat.round=1;const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.reset();
 await f.Hooks.emit('updateUser',{id:'player',isGM:false},{name:'Renamed'});await f.Hooks.emit('userConnected',{id:'player',isGM:false},true);assert.equal(f.scans,0);assert.equal(f.reads.size,0);
 f.combat.round=2;await f.Hooks.emit('updateCombat',f.combat,{round:2});assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);assert.deepEqual(f.errors,[]);release();
});

test('a disabled combat request waits for a fresh cohort after GM authority returns',async()=>{
 const f=fixture(1),target=f.game.actors.get('a0');f.effect(target);f.combat.round=1;const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.reset();
 f.game.users.activeGM={id:'other',isGM:true};await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,0);assert.equal(f.reads.size,0);assert.deepEqual(f.deleted,[]);
 f.game.users.activeGM=f.gm;f.combat.round=2;await f.Hooks.emit('updateCombat',f.combat,{round:2});assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);release();
});

test('recovery includes new timing and source loss while leaving deleted sources and unknown claims alone',async()=>{
 const f=fixture(3),removed=f.game.actors.get('a0'),created=f.game.actors.get('a1'),retained=f.game.actors.get('a2');f.effect(removed,'old');const item=f.effect(retained,'retained');item.sourceId=KNOWLEDGE_SOURCES.knownEffect;
 const state={id:'state',status:'claimed',claim:'unknown',combatId:f.combat.id};retained.flags[M]={knowledge:{strategistStates:[state]}};f.combat.round=1;
 const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.game.users.activeGM={id:'other',isGM:true};await f.Hooks.emit('userConnected',f.game.users.activeGM,true);f.reset();
 const timing=f.effect(created,'new');delete item.sourceId;f.game.actors.delete(removed.id);
 await f.Hooks.emit('createItem',timing);await f.Hooks.emit('updateItem',item,{'-=sourceId':null});await f.Hooks.emit('deleteActor',removed);
 assert.equal(f.reads.size,0);assert.equal(f.scans,0);assert.deepEqual(f.deleted,[]);assert.deepEqual(retained.flags[M].knowledge.strategistStates,[state]);
 f.game.users.activeGM=f.gm;await f.Hooks.emit('userConnected',f.gm,true);f.combat.round=2;await f.Hooks.emit('updateCombat',f.combat,{round:2});
 assert.deepEqual(f.deleted,[[created.uuid,'new'],[retained.uuid,'retained']]);assert(removed.items.has('old'));assert.deepEqual(retained.flags[M].knowledge.strategistStates,[state]);assert.deepEqual(f.updates,[]);assert.equal(f.scans,2);assert.deepEqual(f.errors,[]);release();
});

test('recovery observes scene imports and current synthetic replacements skipped during lost authority',async()=>{
 const f=fixture(1),base=f.game.actors.get('a0'),old=f.actor('old'),scene={id:'scene',tokens:new Map()};old.isToken=true;old.uuid='Scene.scene.Token.token.Actor.a0';
 const token={id:'token',uuid:'Scene.scene.Token.token',actor:old,parent:scene};old.token=token;scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);f.effect(old,'old');f.combat.round=1;
 const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.game.users.activeGM={id:'other',isGM:true};await f.Hooks.emit('updateUser',f.game.users.activeGM,{role:4});f.reset();
 const current=f.actor('current');current.isToken=true;current.uuid=old.uuid;current.token=token;token.actor=current;f.effect(current,'current');
 const imported=f.actor('imported');imported.isToken=true;imported.uuid='Scene.scene.Token.imported.Actor.a0';const importedToken={id:'imported',uuid:'Scene.scene.Token.imported',actor:imported,parent:scene};imported.token=importedToken;scene.tokens.set(importedToken.id,importedToken);f.effect(imported,'imported');
 let dependents=0;base.getDependentTokens=()=>{dependents++;return [token,importedToken]};
 await f.Hooks.emit('updateActor',base,{});await f.Hooks.emit('updateToken',token,{delta:{items:[]}});await f.Hooks.emit('updateScene',scene,{tokens:[{_id:importedToken.id}]});assert.equal(f.reads.size,0);assert.equal(dependents,0);
 f.game.users.activeGM=f.gm;await f.Hooks.emit('updateUser',f.gm,{role:4});f.combat.round=2;await f.Hooks.emit('updateCombat',f.combat,{round:2});
 assert.deepEqual(f.deleted,[[current.uuid,'current'],[imported.uuid,'imported']]);assert(old.items.has('old'));assert.equal(f.scans,2);assert.deepEqual(f.errors,[]);release();
});

for(const changes of [{futureField:true},{}])test(`active GM unknown actor changes discover new timing: ${JSON.stringify(changes)}`,async()=>{
 const f=fixture(1),target=f.game.actors.get('a0'),release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{turn:0});f.effect(target);await f.Hooks.emit('updateActor',target,changes);f.reset();
 await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);assert.equal(f.scans,2);assert.deepEqual(f.errors,[]);release();
});

test('socket disconnect cancels its cohort and waits for a normal combat event',async()=>{
 const socket=connection(),f=fixture(1,{socket}),target=f.game.actors.get('a0');f.effect(target);f.combat.round=1;const release=f.provider.register({Hooks:f.Hooks});await f.Hooks.emit('updateCombat',f.combat,{round:1});f.reset();
 socket.emit('disconnect');socket.emit('connect');assert.equal(f.scans,0);assert.equal(f.reads.size,0);assert.deepEqual(f.deleted,[]);
 f.combat.round=2;await f.Hooks.emit('updateCombat',f.combat,{round:2});assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);assert.equal(socket.count('disconnect'),1);
 f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);release();assert.equal(socket.count('disconnect'),0);assert.equal(f.Hooks.count(),0);
});

test('a repeated old release preserves new hooks and the new socket connection',async()=>{
 const oldSocket=connection(),newSocket=connection(),f=fixture(1,{socket:oldSocket}),target=f.game.actors.get('a0');f.effect(target);
 const oldRelease=f.provider.register({Hooks:f.Hooks});f.game.socket=newSocket;oldRelease();const release=f.provider.register({Hooks:f.Hooks});oldRelease();
 await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.deepEqual(f.deleted,[[target.uuid,'expiry']]);assert.equal(oldSocket.count('disconnect'),0);assert.equal(newSocket.count('disconnect'),1);assert.deepEqual(f.errors,[]);
 release();assert.equal(newSocket.count('disconnect'),0);assert.equal(f.Hooks.count(),0);
});

test('cold maintenance reads later timing changed by a microtask after the first actor inventory read',async()=>{
 const f=fixture(2),first=f.game.actors.get('a0'),later=f.game.actors.get('a1');f.combat.round=1;
 const firstEffect=f.effect(first),laterEffect=f.effect(later),values=first.items.values.bind(first.items);let scheduled=false,mutationApplied=false;
 first.items.values=()=>{if(!scheduled){scheduled=true;queueMicrotask(()=>{laterEffect.flags[M].knowledge.timing.round=1;mutationApplied=true})}return values()};
 const release=f.provider.register({Hooks:f.Hooks});f.reset();
 try{
  await f.Hooks.emit('updateCombat',f.combat,{round:1});
  assert.equal(mutationApplied,true);assert.deepEqual(f.errors,[]);assert.deepEqual(f.updates,[]);
  assert.equal(first.items.get('expiry'),firstEffect);assert.equal(laterEffect.flags[M].knowledge.timing.round,1);
  assert.deepEqual(f.deleted,[[later.uuid,'expiry']]);assert.equal(later.items.has('expiry'),false);
 }finally{release()}
});

test('maintenance reads truthy timing again before deciding expiry',async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),item=f.effect(actor),knowledge=item.flags[M].knowledge,timing=knowledge.timing;let reads=0;
 Object.defineProperty(knowledge,'timing',{get(){return ++reads===1?{...timing,round:3}:timing}});
 const release=f.provider.register({Hooks:f.Hooks});
 try{
  await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(reads,2);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.equal(actor.items.has(item.id),false);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
 }finally{release()}
});

for(const mode of ['none','one','all'])test(`each scheduled cohort maintains every inventory once: ${mode}`,async()=>{
 const f=fixture(10),timed=[],original=[];f.combat.round=1;
 for(const [actorIndex,actor]of [...f.game.actors.values()].entries()){
  actor.items.clear();for(let itemIndex=0;itemIndex<50;itemIndex++){
   if(itemIndex===49&&(mode==='all'||mode==='one'&&actorIndex===0)){timed.push(actor);f.effect(actor)}
   else actor.items.set(`item${itemIndex}`,{id:`item${itemIndex}`,type:'weapon'});
  }
  original.push([actor,[...actor.items]]);
 }
 const release=f.provider.register({Hooks:f.Hooks});f.reset();await f.Hooks.emit('updateCombat',f.combat,{round:1});
 assert.equal([...f.itemVisits.values()].reduce((sum,visits)=>sum+visits,0),500);assert.equal(f.scans,2);
 for(const [actor,items]of original){assert.equal(f.reads.get(actor.uuid),1);assert.equal(f.itemVisits.get(actor.uuid),50);assert.deepEqual([...actor.items],items);for(const [id,item]of items)assert.equal(actor.items.get(id),item)}
 assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
 f.combat.round=2;f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});
 assert.equal(f.scans,2);assert.deepEqual(f.deleted,timed.map(actor=>[actor.uuid,'expiry']));
 assert.deepEqual([...f.reads.keys()],original.map(([actor])=>actor.uuid));assert.equal([...f.itemVisits.values()].reduce((sum,visits)=>sum+visits,0),500);
 for(const [actor]of original){assert.equal(f.reads.get(actor.uuid),1);assert.equal(f.itemVisits.get(actor.uuid),50)}
 f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,2);assert.equal([...f.itemVisits.values()].reduce((sum,visits)=>sum+visits,0),{none:500,one:499,all:490}[mode]);
 for(const [actor]of original)assert.equal(f.reads.get(actor.uuid),1);assert.deepEqual(f.deleted,timed.map(actor=>[actor.uuid,'expiry']));assert.deepEqual(f.errors,[]);release();
});

test('each cohort maintains a linked world actor inventory once',async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),scene={id:'scene',tokens:new Map()};f.combat.round=1;actor.items.clear();const effect=f.effect(actor);
 for(let index=1;index<50;index++)actor.items.set(`item${index}`,{id:`item${index}`,type:'weapon'});
 for(const id of ['first','second'])scene.tokens.set(id,{id,uuid:`Scene.scene.Token.${id}`,actor,parent:scene});f.game.scenes.set(scene.id,scene);
 const release=f.provider.register({Hooks:f.Hooks});f.reset();await f.Hooks.emit('updateCombat',f.combat,{round:1});
 assert.equal(f.itemVisits.get(actor.uuid),50);assert.equal(f.reads.get(actor.uuid),1);assert.equal(f.scans,2);assert.equal(actor.items.get('expiry'),effect);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);
 f.combat.round=2;f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});
 assert.equal(f.scans,2);assert.equal(f.reads.get(actor.uuid),1);assert.equal(f.itemVisits.get(actor.uuid),50);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);
 f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,2);assert.equal(f.reads.get(actor.uuid),1);assert.equal(f.itemVisits.get(actor.uuid),49);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.deepEqual(f.errors,[]);release();
});

test('each cohort maintains only the last actor object with a shared UUID',async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),replacement=f.actor('replacement'),scene={id:'scene',tokens:new Map()};f.combat.round=1;actor.items.clear();f.effect(actor);
 replacement.uuid=actor.uuid;replacement.isToken=true;
 for(let index=1;index<50;index++)actor.items.set(`item${index}`,{id:`item${index}`,type:'weapon'});
 for(let index=0;index<50;index++)replacement.items.set(`item${index}`,{id:`item${index}`,type:'weapon'});
 const token={id:'replacement',uuid:'Scene.scene.Token.replacement',actor:replacement,parent:scene};replacement.token=token;
 scene.tokens.set('linked',{actor,parent:scene});scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);
 let worldReads=0,syntheticReads=0;const worldValues=actor.items.values.bind(actor.items),syntheticValues=replacement.items.values.bind(replacement.items);
 actor.items.values=()=>{worldReads++;return worldValues()};replacement.items.values=()=>{syntheticReads++;return syntheticValues()};
 const release=f.provider.register({Hooks:f.Hooks});f.reset();await f.Hooks.emit('updateCombat',f.combat,{round:1});
 assert.equal(f.itemVisits.get(actor.uuid),50);assert.equal(worldReads,0);assert.equal(syntheticReads,1);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);
 f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.reads.get(actor.uuid),1);assert.equal(f.itemVisits.get(actor.uuid),50);assert.equal(f.scans,2);assert.equal(worldReads,0);assert.equal(syntheticReads,2);
 scene.tokens.delete(token.id);await f.Hooks.emit('deleteToken',token);const linked={id:'new-linked',uuid:'Scene.scene.Token.new-linked',actor,parent:scene};scene.tokens.set(linked.id,linked);await f.Hooks.emit('createToken',linked);f.combat.round=2;f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});
 assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.equal(replacement.items.size,50);assert.deepEqual(f.errors,[]);release();
});

test('identity recovery fails before maintenance and retries only on a later request',async()=>{
 const f=fixture(2),first=f.game.actors.get('a0'),later=f.game.actors.get('a1'),error=Error('scene enumeration failed');
 f.effect(first);const original=f.game.scenes.values;let attempts=0;
 f.game.scenes.values=()=>{attempts++;throw error};const release=f.provider.register({Hooks:f.Hooks});
 try{
  await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[error]);assert.equal(f.errors[0],error);assert.equal(attempts,1);
  f.game.scenes.values=original;f.effect(later,'added');f.reset();
  await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[first.uuid,'expiry'],[later.uuid,'added']]);assert.equal(attempts,1);assert.deepEqual(f.errors,[error]);assert.equal(f.errors[0],error);
 }finally{f.game.scenes.values=original;release()}
});

test('a maintenance timing read failure reports once without retrying writes',async()=>{
 const f=fixture(2),first=f.game.actors.get('a0'),failing=f.game.actors.get('a1'),error=Error('timing maintenance failed');f.combat.round=1;
 const firstEffect=f.effect(first),failedEffect=f.effect(failing),timing=failedEffect.flags[M].knowledge.timing;let attempts=0;
 Object.defineProperty(timing,'combatId',{configurable:true,get(){attempts++;throw error}});
 const release=f.provider.register({Hooks:f.Hooks});f.reset();await f.Hooks.emit('updateCombat',f.combat,{round:1});
 assert.equal(f.scans,2);assert.equal(attempts,1);assert.equal(f.errors.length,1);assert.equal(f.errors[0],error);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);
 Object.defineProperty(timing,'combatId',{configurable:true,writable:true,value:f.combat.id});timing.round=3;
 f.reset();await f.Hooks.emit('updateCombat',f.combat,{round:1});
 assert.equal(f.scans,2);assert.equal(attempts,1);assert.equal(f.errors.length,1);assert.equal(f.errors[0],error);
 assert.equal(first.items.get('expiry'),firstEffect);assert.equal(failing.items.get('expiry'),failedEffect);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);release();
});

test('fallback maintenance reads later actor timing after an earlier actor deletion awaits',async()=>{
 let resume,entered;const waiting=new Promise(resolve=>resume=resolve),started=new Promise(resolve=>entered=resolve);
 const f=fixture(2),first=f.game.actors.get('a0'),later=f.game.actors.get('a1');f.effect(first);const effect=f.effect(later);effect.flags[M].knowledge.timing.round=3;
 const remove=first.deleteEmbeddedDocuments.bind(first);first.deleteEmbeddedDocuments=async(type,ids)=>{entered();await waiting;return remove(type,ids)};
 const release=f.provider.register({Hooks:f.Hooks}),pending=f.Hooks.emit('updateCombat',f.combat,{turn:0});await started;
 effect.flags[M].knowledge.timing.round=2;resume();await pending;
 assert.deepEqual(f.deleted,[[first.uuid,'expiry'],[later.uuid,'expiry']]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);release();
});

test('a maintenance write failure does not retry the business write',async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),effect=f.effect(actor),error=Error('expiry delete failed');let attempts=0;
 actor.deleteEmbeddedDocuments=async()=>{attempts++;throw error};const release=f.provider.register({Hooks:f.Hooks});
 await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(attempts,1);assert.deepEqual(f.errors,[error]);assert.deepEqual(f.deleted,[]);
 effect.flags[M].knowledge.timing.round=3;f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});
 assert.equal(f.scans,2);assert.equal(attempts,1);assert.deepEqual(f.errors,[error]);assert.deepEqual(f.updates,[]);release();
});

test('registration installs only two authority listeners on the first relevant active GM request',async()=>{
 const socket=connection(),f=fixture(1,{socket}),release=f.provider.register({Hooks:f.Hooks});
 try{
  assert.equal(f.Hooks.count(),20);assert.equal(socket.count('disconnect'),1);
  await f.Hooks.emit('updateCombat',f.combat,{name:'Renamed'});assert.equal(f.Hooks.count(),20);
  f.game.user={id:'player',isGM:false};await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.Hooks.count(),20);
  f.game.user=f.gm;f.game.users.activeGM={id:'other',isGM:true};await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.Hooks.count(),20);
  f.game.users.activeGM=f.gm;await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.Hooks.count(),22);
  f.provider.register({Hooks:f.Hooks})();assert.equal(f.Hooks.count(),22);
  f.game.users.activeGM={id:'other',isGM:true};await f.Hooks.emit('updateUser',f.game.users.activeGM,{role:4});assert.equal(f.Hooks.count(),22);
  f.game.users.activeGM=f.gm;await f.Hooks.emit('userConnected',f.gm,true);await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.Hooks.count(),22);
  assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
 }finally{release()}
 assert.equal(f.Hooks.count(),0);assert.equal(socket.count('disconnect'),0);
});

test('direct item deletion and public maintenance do not activate scheduled maintenance',async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),item={id:'removed',actor,type:'weapon'},release=f.provider.register({Hooks:f.Hooks});
 try{
  await f.provider.maintain(actor);await f.Hooks.emit('deleteItem',item);
  assert.equal(f.Hooks.count(),20);assert.equal(f.scans,0);assert.deepEqual(f.errors,[]);
 }finally{release()}
});

for(const expiredFirst of [true,false])test(`cold per-actor read failure preserves completed work: ${expiredFirst}`,async()=>{
 const f=fixture(3),first=f.game.actors.get('a0'),later=f.game.actors.get('a1'),unvisited=f.game.actors.get('a2');
 f.combat.round=1;const firstEffect=f.effect(first),laterEffect=f.effect(later),error=Error('later timing failed');
 firstEffect.flags[M].knowledge.timing.round=expiredFirst?1:2;
 const knowledge=laterEffect.flags[M].knowledge,timing=knowledge.timing;let attempts=0;
 Object.defineProperty(knowledge,'timing',{configurable:true,get(){attempts++;throw error}});
 const release=f.provider.register({Hooks:f.Hooks});
 try{
  await f.Hooks.emit('updateCombat',f.combat,{round:1});
  assert.deepEqual(f.errors,[error]);assert.equal(f.errors[0],error);assert.equal(attempts,1);
  assert.deepEqual(f.deleted,expiredFirst?[[first.uuid,'expiry']]:[]);assert.deepEqual(f.updates,[]);
  assert.equal(later.items.get('expiry'),laterEffect);assert.equal(first.items.has('expiry'),!expiredFirst);
  Object.defineProperty(knowledge,'timing',{configurable:true,writable:true,value:timing});
  timing.round=3;f.effect(unvisited,'added').flags[M].knowledge.timing.round=1;f.reset();
  await f.Hooks.emit('updateCombat',f.combat,{round:1});
  assert.equal(f.scans,2);assert.deepEqual(f.errors,[error]);assert.equal(f.errors[0],error);assert.equal(attempts,1);
  assert.deepEqual(f.deleted,expiredFirst?[[first.uuid,'expiry'],[unvisited.uuid,'added']]:[[unvisited.uuid,'added']]);
  assert.equal(later.items.get('expiry'),laterEffect);assert.equal(unvisited.items.has('added'),false);assert.deepEqual(f.updates,[]);
 }finally{release()}
});

test('ordinary Doc changes during discovery do not rerun the captured cohort',async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0');f.effect(actor);
 const values=f.game.actors.values.bind(f.game.actors);let notified=false;
 f.game.actors.values=()=>{const cohort=[...values()];if(!notified){notified=true;void f.Hooks.emit('updateActor',actor,{})}return cohort.values()};
 const release=f.provider.register({Hooks:f.Hooks});
 try{
  await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(notified,true);assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.deepEqual(f.errors,[]);
  f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);
  f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,2);assert.deepEqual(f.errors,[]);assert.deepEqual(f.updates,[]);
 }finally{release()}
});

for(const phase of ['world enumeration','UUID deduplication'])for(const change of ['disconnect','loss and immediate regain'])test(`cold ${phase} ${change} cannot restore an old cohort`,async()=>{
 const socket=connection(),f=fixture(2,{socket}),first=f.game.actors.get('a0'),later=f.game.actors.get('a1');f.effect(first,'old');f.effect(later,'later');
 let changed=false,current;
 const invalidate=()=>{
  if(changed)return;changed=true;
  if(change==='disconnect')socket.emit('disconnect');
  else{f.game.users.activeGM={id:'other',isGM:true};void f.Hooks.emit('updateUser',f.game.users.activeGM,{role:4});f.game.users.activeGM=f.gm;void f.Hooks.emit('userConnected',f.gm,true)}
  current=f.actor(first.id);f.game.actors.set(first.id,current);f.effect(current,'current');
 };
 if(phase==='world enumeration'){
  const values=f.game.actors.values.bind(f.game.actors);f.game.actors.values=()=>{const cohort=[...values()];invalidate();return cohort.values()};
 }else{
  const uuid=first.uuid;let reads=0;Object.defineProperty(first,'uuid',{get(){if(++reads===1)invalidate();return uuid}});
 }
 const release=f.provider.register({Hooks:f.Hooks});
 try{
  await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(changed,true);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);assert.equal(first.items.has('old'),true);
  f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[current.uuid,'current'],[later.uuid,'later']]);assert.equal(first.items.has('old'),true);
  f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.equal(f.scans,2);assert.deepEqual(f.errors,[]);assert.equal(socket.count('disconnect'),1);
 }finally{release()}
 assert.equal(f.Hooks.count(),0);assert.equal(socket.count('disconnect'),0);
});

test('a synchronous state notification remains visible to the next fresh cohort',async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),state={id:'state',status:'claimed',claim:'unknown',combatId:f.combat.id};
 const knowledge={};actor.flags[M]={knowledge};let notified=false;
 Object.defineProperty(knowledge,'strategistStates',{configurable:true,get(){
  if(!notified){notified=true;Object.defineProperty(knowledge,'strategistStates',{configurable:true,writable:true,value:[state]});void f.Hooks.emit('updateActor',actor,{});return []}
  return [state];
 }});
 const release=f.provider.register({Hooks:f.Hooks});
 try{
  await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(notified,true);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
  f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(f.scans,2);assert.equal(f.reads.get(actor.uuid),1);assert.deepEqual(knowledge.strategistStates,[state]);assert.deepEqual(f.errors,[]);
 }finally{release()}
});

for(const inventory of ['missing','contents fallback'])test(`a new cohort maintains timing after restoring ${inventory} inventory`,async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),items=actor.items;actor.items=inventory==='missing'?null:{contents:[...items.values()]};
 const release=f.provider.register({Hooks:f.Hooks});
 try{
  await f.Hooks.emit('updateCombat',f.combat,{turn:0});assert.deepEqual(f.deleted,[]);assert.deepEqual(f.errors,[]);
  actor.items=items;f.effect(actor);f.reset();await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.equal(f.reads.get(actor.uuid),1);assert.deepEqual(f.errors,[]);assert.deepEqual(f.updates,[]);
 }finally{actor.items=items;release()}
});

test('native registration counts only provider listeners through lazy activation and release',async t=>{
 const f=fixture(1),native=attachNative(f,t),external=native.api.on('updateActor',()=>{}),baseline=native.count(),release=f.provider.register({Hooks:native.api});
 try{
  assert.equal(native.count()-baseline,20);
  native.callAll('updateCombat',f.combat,{name:'Renamed'});await native.drain();assert.equal(native.count()-baseline,20);
  native.callAll('updateCombat',f.combat,{turn:0});await native.drain();assert.equal(native.count()-baseline,22);
  native.callAll('updateCombat',f.combat,{turn:0});await native.drain();assert.equal(native.count()-baseline,22);assert.deepEqual(native.errors,[]);assert.deepEqual(f.errors,[]);
 }finally{release()}
 assert.equal(native.count()-baseline,0);native.api.off('updateActor',external);assert.equal(native.count(),1);
});

for(const placement of ['before','after'])test(`native world replacement ${placement} activation is discovered by the next cohort`,async t=>{
 const f=fixture(1),old=f.game.actors.get('a0'),native=attachNative(f,t);f.effect(old,'old');f.combat.round=1;
 const release=f.provider.register({Hooks:native.api});let current,continued=0;
 const external=()=>{native.api.on('updateActor',actor=>{continued++;if(actor===old&&!current){current=f.actor(old.id);f.game.actors.set(old.id,current);f.effect(current,'current')}})};
 try{
  if(placement==='before')external();native.callAll('updateCombat',f.combat,{round:1});await native.drain();if(placement==='after')external();
  native.callAll('updateActor',old,{});assert.equal(f.game.actors.get(old.id),current);assert.equal(continued,1);await native.drain();
  f.combat.round=2;f.reset();native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[current.uuid,'current']]);assert.equal(old.items.has('old'),true);assert.deepEqual(f.updates,[]);
  native.callAll('updateActor',current,{});await native.drain();native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.deepEqual(f.deleted,[[current.uuid,'current']]);assert.equal(old.items.has('old'),true);assert.deepEqual(native.errors,[]);assert.deepEqual(f.errors,[]);
 }finally{release()}
 assert.equal(native.count(),2);
});

for(const placement of ['before','after'])test(`native authority loss ${placement} activation waits for a current GM request`,async t=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),native=attachNative(f,t);f.effect(actor);f.combat.round=1;
 const release=f.provider.register({Hooks:native.api});let continued=0;
 const external=()=>native.api.on('updateActor',()=>{continued++;f.game.users.activeGM={id:'other',isGM:true}});
 try{
  if(placement==='before')external();native.callAll('updateCombat',f.combat,{round:1});await native.drain();if(placement==='after')external();
  native.callAll('updateActor',actor,{});assert.equal(continued,1);assert.equal(f.game.users.activeGM.id,'other');await native.drain();
  f.combat.round=2;f.reset();native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.equal(f.scans,0);assert.equal(f.reads.size,0);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);
  f.game.users.activeGM=f.gm;native.callAll('userConnected',f.gm,true);native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.deepEqual(native.errors,[]);assert.deepEqual(f.errors,[]);
 }finally{release()}
 assert.equal(native.count(),2);
});

for(const placement of ['before','after'])for(const change of ['replacement','authority loss'])test(`native deleteItem ${change} ${placement} activation precedes queued maintenance`,async t=>{
 const f=fixture(1),old=f.game.actors.get('a0'),native=attachNative(f,t);f.effect(old,'old');f.combat.round=1;
 const release=f.provider.register({Hooks:native.api});let continued=0,current=old;
 const external=()=>native.api.on('deleteItem',()=>{
  continued++;
  if(change==='replacement'){current=f.actor(old.id);f.game.actors.set(old.id,current);f.effect(current,'current')}
  else f.game.users.activeGM={id:'other',isGM:true};
 });
 try{
  if(placement==='before')external();native.callAll('updateCombat',f.combat,{round:1});await native.drain();if(placement==='after')external();
  f.combat.round=2;f.reset();native.callAll('deleteItem',{id:'removed',type:'weapon',actor:old});assert.equal(continued,1);assert.equal(f.reads.size,0);assert.deepEqual(f.deleted,[]);await native.drain();
  assert.deepEqual(f.deleted,[]);assert.equal(old.items.has('old'),true);assert.deepEqual(f.updates,[]);assert.equal(f.scans,0);
  if(change==='replacement')native.callAll('updateActor',current,{});
  else{f.game.users.activeGM=f.gm;native.callAll('userConnected',f.gm,true)}
  native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.deepEqual(f.deleted,[[current.uuid,change==='replacement'?'current':'old']]);assert.deepEqual(native.errors,[]);assert.deepEqual(f.errors,[]);
 }finally{release()}
 assert.equal(native.count(),2);
});

for(const placement of ['before','after'])test(`native combined token notification does no inventory work with an external ${placement}`,async t=>{
 const f=fixture(1),scene={id:'scene',tokens:new Map()},old=f.actor('old'),current=f.actor('current'),native=attachNative(f,t);
 old.isToken=current.isToken=true;old.uuid=current.uuid='Scene.scene.Token.token.Actor.a0';
 const token={id:'token',uuid:'Scene.scene.Token.token',actor:old,parent:scene};old.token=current.token=token;scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);f.effect(old,'old');f.effect(current,'current');f.combat.round=1;
 const release=f.provider.register({Hooks:native.api});let readsAtExternal,continued=0;
 const external=()=>native.api.on('updateToken',()=>{continued++;readsAtExternal=f.reads.get(current.uuid)??0});
 try{
  if(placement==='before')external();native.callAll('updateCombat',f.combat,{round:1});await native.drain();if(placement==='after')external();
  token.actor=current;f.reset();native.callAll('updateToken',token,{actorId:'current',actorLink:false,delta:{items:[]}});
  assert.equal(continued,1);assert.equal(readsAtExternal,0);assert.equal(f.reads.size,0);assert.equal(f.scans,0);await native.drain();
  f.combat.round=2;f.reset();native.callAll('updateCombat',f.combat,{turn:0});await native.drain();assert.equal(f.scans,2);assert.equal(f.reads.get(current.uuid),1);
  assert.deepEqual(f.deleted,[[current.uuid,'current']]);assert.equal(old.items.has('old'),true);assert.deepEqual(f.updates,[]);assert.deepEqual(native.errors,[]);assert.deepEqual(f.errors,[]);
 }finally{release()}
 assert.equal(native.count(),2);
});

test('native external timing error is isolated and the next cohort discovers the current effect',async t=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),native=attachNative(f,t),release=f.provider.register({Hooks:native.api});
 try{
  native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  const item=f.effect(actor),knowledge=item.flags[M].knowledge,timing=knowledge.timing,error=Error('external timing failed');let attempts=0,continued=0;
  Object.defineProperty(knowledge,'timing',{configurable:true,get(){attempts++;throw error}});native.api.on('updateItem',()=>{void knowledge.timing});native.api.on('updateItem',()=>{continued++});
  native.callAll('updateItem',item,{});assert.equal(continued,1);assert.equal(attempts,1);assert.equal(native.errors.length,1);assert.equal(native.errors[0].cause,error);await native.drain();assert.deepEqual(f.errors,[]);
  Object.defineProperty(knowledge,'timing',{configurable:true,writable:true,value:timing});f.reset();native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.equal(attempts,1);assert.equal(native.errors.length,1);assert.deepEqual(f.errors,[]);
  f.reset();native.callAll('updateCombat',f.combat,{turn:0});await native.drain();assert.equal(f.scans,2);assert.equal(f.reads.get(actor.uuid),1);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.deepEqual(f.updates,[]);
 }finally{release()}
 assert.equal(native.count(),3);
});

test('native combined token external error cannot suppress later listeners or current-cohort maintenance',async t=>{
 const f=fixture(1),scene={id:'scene',tokens:new Map()},old=f.actor('old'),current=f.actor('current'),native=attachNative(f,t);
 old.isToken=current.isToken=true;old.uuid=current.uuid='Scene.scene.Token.token.Actor.a0';
 const token={id:'token',uuid:'Scene.scene.Token.token',actor:old,parent:scene};old.token=current.token=token;scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);f.effect(old,'old');f.effect(current,'current');f.combat.round=1;
 const release=f.provider.register({Hooks:native.api});
 try{
  native.callAll('updateCombat',f.combat,{round:1});await native.drain();token.actor=current;
  const values=current.items.values.bind(current.items),error=Error('first external token callback failed');let attempts=0,continued=0;
  current.items.values=()=>{if(++attempts===1)throw error;return values()};native.api.on('updateToken',()=>{[...current.items.values()]});native.api.on('updateToken',()=>{continued++;[...current.items.values()]});
  f.reset();native.callAll('updateToken',token,{actorId:'current',actorLink:false,delta:{items:[]}});
  assert.equal(attempts,2);assert.equal(continued,1);assert.equal(native.errors.length,1);assert.equal(native.errors[0].cause,error);assert.deepEqual(f.errors,[]);await native.drain();
  current.items.values=values;f.combat.round=2;f.reset();native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[current.uuid,'current']]);assert.equal(old.items.has('old'),true);assert.equal(native.errors.length,1);assert.deepEqual(f.errors,[]);
  f.reset();native.callAll('updateCombat',f.combat,{turn:0});await native.drain();assert.equal(f.scans,2);assert.deepEqual(f.deleted,[[current.uuid,'current']]);assert.deepEqual(f.updates,[]);
 }finally{release()}
 assert.equal(native.count(),3);
});

test('native partial authority installation removes only its new listeners and waits for an independent request',async t=>{
 const socket=connection(),f=fixture(1,{socket}),actor=f.game.actors.get('a0'),native=attachNative(f,t);f.effect(actor);
 const baseline=native.count(),release=f.provider.register({Hooks:native.api}),on=native.api.on,error=Error('second authority subscription failed');let attempts=0;const attemptedHookNames=[];
 // Arm after base registration, before the second authority subscription reaches native on.
 native.api.on=(...args)=>{attemptedHookNames.push(args[0]);if(++attempts===2)throw error;return on(...args)};
 try{
  native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.deepEqual(f.errors,[error]);assert.equal(f.errors[0],error);assert.equal(attempts,2);assert.deepEqual(attemptedHookNames,['updateUser','userConnected']);assert.equal(native.count()-baseline,20);assert.equal(socket.count('disconnect'),1);
  assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(native.errors,[]);
  native.api.on=on;native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.equal(native.count()-baseline,22);assert.equal(attempts,2);assert.deepEqual(f.errors,[error]);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.deepEqual(native.errors,[]);
 }finally{native.api.on=on;release()}
 assert.equal(native.count()-baseline,0);assert.equal(socket.count('disconnect'),0);
});

for(const change of ['loss and immediate regain','disconnect','token replacement','release and reregister'])test(`native pending expiry stops later claim writes after ${change}`,async t=>{
 let resume,entered;const waiting=new Promise(resolve=>resume=resolve),started=new Promise(resolve=>entered=resolve);let recipient;
 const socket=connection(),f=fixture(2,{socket,fromUuid:async uuid=>{assert.equal(uuid,'Actor.a1');return recipient}}),source=f.game.actors.get('a0');recipient=f.game.actors.get('a1');
 if(change==='token replacement'){
  source.isToken=true;source.uuid='Scene.scene.Token.token.Actor.a0';const scene={id:'scene',tokens:new Map()},token={id:'token',uuid:'Scene.scene.Token.token',actor:source,parent:scene};source.token=token;scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);f.game.actors.delete(source.id);
 }
 f.effect(source);f.combat.round=1;const native=attachNative(f,t);let release=f.provider.register({Hooks:native.api});
 try{
  native.callAll('updateCombat',f.combat,{round:1});await native.drain();
  source.flags[M]={knowledge:{strategistStates:[{id:'state',status:'consumed',combatId:f.combat.id,settledClaim:{id:'claim',actorUuid:recipient.uuid},cleanupDone:false}]}};
  recipient.items.set('claim',{id:'claim',type:'effect',flags:{[M]:{knowledge:{kind:'strategist-claim',claim:'claim'}}}});
  const remove=source.deleteEmbeddedDocuments.bind(source);source.deleteEmbeddedDocuments=async(type,ids)=>{entered();await waiting;return remove(type,ids)};
  f.combat.round=2;native.callAll('updateCombat',f.combat,{round:2});await started;
  if(change==='loss and immediate regain'){f.game.users.activeGM={id:'other',isGM:true};native.callAll('updateUser',f.game.users.activeGM,{role:4});f.game.users.activeGM=f.gm;native.callAll('userConnected',f.gm,true)}
  if(change==='disconnect')socket.emit('disconnect');
  if(change==='token replacement'){const current=f.actor(source.id);current.isToken=true;current.uuid=source.uuid;current.token=source.token;source.token.actor=current;native.callAll('updateToken',source.token,{actorId:current.id,delta:{items:[]}})}
  if(change==='release and reregister'){const oldRelease=release;oldRelease();release=f.provider.register({Hooks:native.api});oldRelease()}
  resume();await native.drain();
  assert.deepEqual(f.deleted,[[source.uuid,'expiry']]);assert.equal(source.items.has('expiry'),false);assert.equal(recipient.items.has('claim'),true);
  assert.equal(source.flags[M].knowledge.strategistStates[0].cleanupDone,false);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);assert.deepEqual(native.errors,[]);
 }finally{resume();await native.drain();release()}
 assert.equal(native.count(),1);assert.equal(socket.count('disconnect'),0);
});

function changeDuringExpiryRead(f,actor,change){
 const item=f.effect(actor),knowledge=item.flags[M].knowledge,timing=knowledge.timing;let changed=false;
 Object.defineProperty(knowledge,'timing',{configurable:true,get(){if(!changed){changed=true;change()}return timing}});
 return item;
}

for(const entry of ['scheduled','deleteItem','public'])test(`expiry deletion stops when GM changes during ${entry} timing read`,async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),item=changeDuringExpiryRead(f,actor,()=>{f.game.users.activeGM={id:'other',isGM:true}});
 const release=entry==='public'?()=>{}:f.provider.register({Hooks:f.Hooks});
 const request=()=>entry==='scheduled'?f.Hooks.emit('updateCombat',f.combat,{turn:0}):entry==='deleteItem'?f.Hooks.emit('deleteItem',{actor}):f.provider.maintain(actor);
 try{
  await request();
  assert.equal(actor.items.get(item.id),item);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
  f.game.users.activeGM=f.gm;await request();
  assert.equal(actor.items.has(item.id),false);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
 }finally{f.game.users.activeGM=f.gm;release()}
 assert.equal(f.Hooks.count(),0);
});

test('expiry deletion stops after native GM loss and synchronous regain',async t=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),native=attachNative(f,t),baseline=native.count();
 const item=changeDuringExpiryRead(f,actor,()=>{
  f.game.users.activeGM={id:'other',isGM:true};native.callAll('updateUser',f.game.users.activeGM,{role:4});
  f.game.users.activeGM=f.gm;native.callAll('userConnected',f.gm,true);
 });
 const release=f.provider.register({Hooks:native.api});
 try{
  native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.equal(actor.items.get(item.id),item);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);assert.deepEqual(native.errors,[]);
  native.callAll('updateCombat',f.combat,{turn:0});await native.drain();
  assert.equal(actor.items.has(item.id),false);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);assert.deepEqual(native.errors,[]);
 }finally{f.game.users.activeGM=f.gm;release()}
 assert.equal(native.count(),baseline);native.verify();
});

for(const entry of ['scheduled','deleteItem'])for(const kind of ['world','synthetic'])test(`expiry deletion stops when ${kind} actor is replaced during ${entry} timing read`,async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0');let replacement,replacementItem;
 if(kind==='synthetic'){
  actor.isToken=true;actor.uuid='Scene.scene.Token.token.Actor.a0';
  const scene={id:'scene',tokens:new Map()},token={id:'token',uuid:'Scene.scene.Token.token',actor,parent:scene};
  actor.token=token;scene.tokens.set(token.id,token);f.game.scenes.set(scene.id,scene);f.game.actors.delete(actor.id);
 }
 const item=changeDuringExpiryRead(f,actor,()=>{
  replacement=f.actor(actor.id);replacement.uuid=actor.uuid;
  if(kind==='synthetic'){replacement.isToken=true;replacement.token=actor.token;actor.token.actor=replacement}
  else f.game.actors.set(actor.id,replacement);
  replacementItem=f.effect(replacement);
 });
 const release=f.provider.register({Hooks:f.Hooks});
 try{
  if(entry==='scheduled')await f.Hooks.emit('updateCombat',f.combat,{turn:0});else await f.Hooks.emit('deleteItem',{actor});
  assert.equal(actor.items.get(item.id),item);assert.equal(replacement.items.get(replacementItem.id),replacementItem);
  assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
  if(entry==='scheduled')await f.Hooks.emit('updateCombat',f.combat,{turn:0});else await f.Hooks.emit('deleteItem',{actor:replacement});
  assert.equal(actor.items.get(item.id),item);assert.equal(replacement.items.has(replacementItem.id),false);
  assert.deepEqual(f.deleted,[[replacement.uuid,'expiry']]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
 }finally{release()}
 assert.equal(f.Hooks.count(),0);
});

for(const lookup of ['GM','current'])test(`expiry deletion reports scheduled ${lookup} failure before writing and waits for a new request`,async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),error=Error(`scheduled ${lookup} lookup failed after expiry scan`);
 const descriptor=Object.getOwnPropertyDescriptor(f.game.users,'activeGM'),get=f.game.actors.get;let attempts=0;
 const item=changeDuringExpiryRead(f,actor,()=>{
  if(lookup==='GM')Object.defineProperty(f.game.users,'activeGM',{configurable:true,get(){attempts++;throw error}});
  else f.game.actors.get=function(id){attempts++;throw error};
 });
 const restore=()=>{Object.defineProperty(f.game.users,'activeGM',descriptor);f.game.actors.get=get};
 const release=f.provider.register({Hooks:f.Hooks});
 try{
  await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(actor.items.get(item.id),item);assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[error]);assert.equal(f.errors[0],error);assert.equal(attempts,1);
  restore();await f.Hooks.emit('updateCombat',f.combat,{turn:0});
  assert.equal(actor.items.has(item.id),false);assert.deepEqual(f.deleted,[[actor.uuid,'expiry']]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[error]);assert.equal(attempts,1);
 }finally{restore();release()}
 assert.equal(f.Hooks.count(),0);
});

test('deleteItem authority loss in the first identity lookup stops inventory traversal',async()=>{
 const f=fixture(1),actor=f.game.actors.get('a0'),item=f.effect(actor),get=f.game.actors.get;let changed=false;
 f.game.actors.get=function(id){const current=get.call(this,id);if(id===actor.id&&!changed){changed=true;f.game.users.activeGM={id:'other',isGM:true}}return current};
 const release=f.provider.register({Hooks:f.Hooks});f.reset();
 try{
  await f.Hooks.emit('deleteItem',{actor});
  assert.equal(changed,true);assert.equal(f.reads.size,0);assert.equal(actor.items.get(item.id),item);
  assert.deepEqual(f.deleted,[]);assert.deepEqual(f.updates,[]);assert.deepEqual(f.errors,[]);
 }finally{f.game.actors.get=get;f.game.users.activeGM=f.gm;release()}
 assert.equal(f.Hooks.count(),0);
});
