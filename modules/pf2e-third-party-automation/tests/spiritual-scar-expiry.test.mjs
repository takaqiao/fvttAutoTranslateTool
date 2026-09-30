import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './glimpse-fixture.mjs';
import {MODULE_ID as M} from '../scripts/rules.mjs';
let api={};try{api=await import('../scripts/spiritual-scar-expiry.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const condition='Compendium.pf2e.conditionitems.Item.xYTAsEpcJE1Ccni3';
function patch(doc,data){for(const [path,value]of Object.entries(data)){const parts=path.split('.');let at=doc;for(const k of parts.slice(0,-1))at=at[k]??={};at[parts.at(-1)]=structuredClone(value)}}
function setup(){
 assert.equal(typeof api.spiritualScarExpiryFor,'function');const f=fixture(),warnings=[],deleted=[];f.game.time={worldTime:100};f.combat.turns.forEach((c,i)=>{c.initiative=20-i;c.flags.pf2e={roundOfLastTurn:i===0?2:1}});
 const expiry=api.spiritualScarExpiryFor(f.allyToken,f.game),nonce='original-scar',record={...expiry,nonce,targetActorUuid:f.enemy.uuid,itemUuid:f.ally.uuid+'.Item.scar',damageMessageId:f.message.id,status:'armed'};
 const effect={id:'effect',uuid:f.enemy.uuid+'.Item.effect',type:'effect',actor:f.enemy,flags:{[M]:{spiritualScarExpiry:record}},system:{slug:'tpa-spiritual-scar-original-scar',context:{origin:{actor:f.ally.uuid,token:f.allyToken.uuid,item:record.itemUuid}},duration:{unit:'unlimited',value:-1,expiry:null},rules:[{key:'GrantItem',uuid:condition,inMemoryOnly:true,allowDuplicate:true,alterations:[{mode:'override',property:'badge-value',value:1}]}]},async update(data){patch(this,data);return this}};
 f.enemy.items.set(effect.id,effect);f.enemy.items.set('other-slowed',{id:'other-slowed',type:'condition',system:{slug:'slowed',value:{value:2}}});
 f.enemy.deleteEmbeddedDocuments=async(_type,ids)=>{for(const id of ids){deleted.push(id);f.enemy.items.delete(id)}return ids};
 const lifecycle=api.createSpiritualScarExpiry({game:f.game,onError:e=>warnings.push(e)});return {...f,expiry,record,effect,lifecycle,warnings,deleted};
}
test('one round ends at the creator next start, even before the next numerical round',()=>{const f=setup();assert.equal(f.expiry.endRound,2);assert.equal(f.expiry.combatantId,'ally');f.combat.turn=2;assert.equal(api.spiritualScarExpiryFor(f.allyToken,f.game).endRound,3);assert.notEqual(f.game.combat,f.combat)});
test('only the creator actual start receipt expires this source; fiend turn/end and time do not',async()=>{const f=setup();f.combat.turns[0].flags.pf2e={roundOfLastTurn:2,roundOfLastTurnEnd:2};f.game.time.worldTime=106;await f.lifecycle.reconcile();assert.deepEqual(f.deleted,[]);f.combat.turn=2;f.combat.turns[2].flags.pf2e.roundOfLastTurn=2;await f.lifecycle.reconcile();assert.deepEqual(f.deleted,['effect']);assert.equal(f.enemy.items.get('other-slowed').system.value.value,2)});
test('reload or GM handoff can expire the persisted source exactly once',async()=>{const f=setup();f.combat.turn=2;f.combat.turns[2].flags.pf2e.roundOfLastTurn=2;const recovered=api.createSpiritualScarExpiry({game:f.game});await Promise.all([recovered.reconcile(),recovered.reconcile()]);assert.deepEqual(f.deleted,['effect'])});
test('reverting or removing the exact damage receipt removes only this source',async()=>{for(const mutate of [f=>f.message.flags.pf2e.appliedDamage={isReverted:true},f=>f.game.messages.delete(f.message.id)]){const f=setup();mutate(f);await f.lifecycle.reconcile();assert.deepEqual(f.deleted,['effect']);assert.ok(f.enemy.items.has('other-slowed'))}});
test('ending or deleting the encounter removes its one-round source',async()=>{for(const mutate of [f=>f.combat.started=false,f=>f.game.combats.delete(f.combat.id)]){const f=setup();mutate(f);await f.lifecycle.reconcile();assert.deepEqual(f.deleted,['effect'])}});
for(const [name,mutate]of [['creator removed',f=>f.scene.tokens.delete(f.allyToken.id)],['round rollback',f=>f.combat.round=1],['turn reorder',f=>f.combat.turns.reverse()],['initiative change',f=>f.combat.turns[2].initiative=99],['skipped creator',f=>{f.combat.round=3;f.combat.turn=0;f.combat.turns[0].flags.pf2e.roundOfLastTurn=3}]])test(`${name} leaves a finite original-time fallback and warns once`,async()=>{const f=setup();f.game.time.worldTime=104;mutate(f);await f.lifecycle.reconcile();await f.lifecycle.reconcile();assert.equal(f.effect.system.duration.unit,'rounds');assert.equal(f.effect.system.duration.expiry,'turn-start');assert.equal(f.effect.system.start.value,100);assert.equal(f.warnings.length,1);assert.equal(f.record.status,'armed');assert.deepEqual(f.deleted,[])});
test('a re-ordered creator that already started cannot reuse an old start as a new turn',()=>{const f=setup();f.combat.turns[2].flags.pf2e.roundOfLastTurn=2;assert.throws(()=>api.spiritualScarExpiryFor(f.allyToken,f.game))});
test('foreign rules or changed effect origins do not authorize source deletion',async()=>{for(const mutate of [f=>f.effect.system.rules[0].uuid='foreign',f=>f.effect.system.context.origin.item='foreign']){const f=setup();mutate(f);f.message.flags.pf2e.appliedDamage={isReverted:true};await f.lifecycle.reconcile();assert.deepEqual(f.deleted,[])}});
test('non-GM clients do not alter duration or conditions',async()=>{const f=setup();f.game.user={id:'player'};f.combat.started=false;await f.lifecycle.reconcile();assert.deepEqual(f.deleted,[]);assert.equal(f.effect.system.duration.unit,'unlimited')});
test('scene-only synthetic fiend sources are included in reload recovery',async()=>{const f=setup();f.game.actors.delete(f.enemy.id);f.message.flags.pf2e.appliedDamage={isReverted:true};await f.lifecycle.reconcile();assert.deepEqual(f.deleted,['effect'])});

function hooks(){const callbacks=new Map();return {on(name,fn){const list=callbacks.get(name)??[];list.push(fn);callbacks.set(name,list);return fn},off(name,fn){callbacks.set(name,(callbacks.get(name)??[]).filter(x=>x!==fn))},emit(name,...args){return Promise.all((callbacks.get(name)??[]).map(fn=>fn(...args)))}}}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
test('unrelated chat does not revisit inventories after scar recovery',async()=>{
 const f=setup(),Hooks=hooks(),native=f.enemy.items.values.bind(f.enemy.items);let reads=0;f.enemy.items.values=()=>{reads++;return native()};
 f.lifecycle.register({Hooks});await flush();reads=0;await Hooks.emit('updateChatMessage',{id:'ordinary',flags:{}});await flush();assert.equal(reads,0);assert.deepEqual(f.deleted,[]);f.lifecycle.unregister();
});
test('the indexed damage receipt still removes its exact scar without scanning scenes',async()=>{
 const f=setup(),Hooks=hooks();f.lifecycle.register({Hooks});await flush();f.game.scenes.values=()=>{throw Error('receipt update must use the effect index')};
 f.message.flags.pf2e.appliedDamage={isReverted:true};await Hooks.emit('updateChatMessage',f.message);await flush();assert.deepEqual(f.deleted,['effect']);assert.ok(f.enemy.items.has('other-slowed'));f.lifecycle.unregister();
});
test('newly created scar effects join the receipt index after initial empty recovery',async()=>{
 const f=setup(),Hooks=hooks();f.enemy.items.delete(f.effect.id);f.lifecycle.register({Hooks});await flush();f.enemy.items.set(f.effect.id,f.effect);
 await Hooks.emit('createItem',f.effect);f.game.messages.delete(f.message.id);await Hooks.emit('deleteChatMessage',f.message);await flush();assert.deepEqual(f.deleted,['effect']);f.lifecycle.unregister();
});
test('actor imports replace stale indexed effect documents',async()=>{
 const f=setup(),Hooks=hooks();f.lifecycle.register({Hooks});await flush();const replacement={...f.effect,flags:structuredClone(f.effect.flags)};f.enemy.items.set(replacement.id,replacement);
 await Hooks.emit('updateActor',f.enemy,{items:[replacement]});f.game.messages.delete(f.message.id);await Hooks.emit('deleteChatMessage',f.message);await flush();assert.deepEqual(f.deleted,['effect']);f.lifecycle.unregister();
});
test('base actor reset refreshes rebuilt synthetic scar documents',async()=>{
 const f=setup(),Hooks=hooks(),base={uuid:'Actor.base',items:new Map(),getDependentTokens:()=>[{actor:f.enemy}]};f.enemy.isToken=true;f.game.actors.set('base',base);f.game.actors.delete(f.enemy.id);
 f.lifecycle.register({Hooks});await flush();const replacement={...f.effect,flags:structuredClone(f.effect.flags)};f.enemy.items.set(replacement.id,replacement);
 await Hooks.emit('updateActor',base,{'system.attributes.hp.value':1});f.game.messages.delete(f.message.id);await Hooks.emit('deleteChatMessage',f.message);assert.deepEqual(f.deleted,['effect']);f.lifecycle.unregister();
});
test('scar maintenance coalesces encounter event bursts without inventory rescans',async()=>{
 const f=setup(),Hooks=hooks();f.lifecycle.register({Hooks});await flush();const get=f.game.messages.get.bind(f.game.messages);let settlements=0;f.game.messages.get=id=>{settlements++;return get(id)};
 await Promise.all(Array.from({length:8},()=>Hooks.emit('updateCombat',f.combat,{round:2})));assert.equal(settlements,1);f.lifecycle.unregister();
});
test('a stale awaited scar cannot remove a freshly indexed replacement',async()=>{
 const f=setup(),Hooks=hooks();f.lifecycle.register({Hooks});await flush();let release,reached;const blocked=new Promise(resolve=>release=resolve),atWrite=new Promise(resolve=>reached=resolve),nativeUpdate=f.effect.update;
 f.effect.update=async function(data){reached();await blocked;return nativeUpdate.call(this,data)};f.combat.turns.reverse();const pending=Hooks.emit('updateCombat',f.combat,{turn:0});await atWrite;
 const replacement={...f.effect,update:nativeUpdate,flags:structuredClone(f.effect.flags)};f.enemy.items.set(replacement.id,replacement);await Hooks.emit('updateActor',f.enemy,{items:[replacement]});release();await pending;
 f.game.messages.delete(f.message.id);await Hooks.emit('deleteChatMessage',f.message);assert.deepEqual(f.deleted,['effect']);f.lifecycle.unregister();
});
test('unregister cancels queued receipt settlement',async()=>{
 const f=setup(),Hooks=hooks();f.lifecycle.register({Hooks});await flush();f.message.flags.pf2e.appliedDamage={isReverted:true};const pending=Hooks.emit('updateChatMessage',f.message);f.lifecycle.unregister();await pending;assert.deepEqual(f.deleted,[]);
});
