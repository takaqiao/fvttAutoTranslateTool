import test from 'node:test';
import assert from 'node:assert/strict';
import {createConfigurationMaintenance} from '../scripts/config-maintenance.mjs';
import {MODULE_ID} from '../scripts/rules.mjs';
import {canSuppressGlimpseReminder,glimpseReactionSetting,GLIMPSE_REACTION_REASON} from '../scripts/glimpse-reaction-setting.mjs';
import * as events from '../scripts/glimpse-configuration-events.mjs';

function fixture(){
 const callbacks=new Map(),writes=[];
 const Hooks={on:(name,fn)=>{const list=callbacks.get(name)??[];list.push(fn);callbacks.set(name,list);return fn},off:(name,fn)=>callbacks.set(name,(callbacks.get(name)??[]).filter(f=>f!==fn))};
 const emit=async(name,...args)=>{await Promise.all((callbacks.get(name)??[]).map(fn=>fn(...args)))};
 const data=new Map([[`${MODULE_ID}.configurationBackups`,[]],['pf2e-reaction.builtinReactionsEnabled',['glimpse-of-redemption','shield-block']]]);
 const game={user:{id:'gm'},users:{activeGM:{id:'gm'}},modules:new Map([['pf2e-reaction',{active:true}]]),settings:{get:(m,k)=>data.get(`${m}.${k}`),set:async(m,k,v)=>{data.set(`${m}.${k}`,v);writes.push(`${m}.${k}`);void emit('updateSetting',{key:`${m}.${k}`});}}};
 return {Hooks,emit,game,data,writes};
}
test('dependency changes reconcile the setting through serialized real maintenance, without recursive writes',async()=>{
 const f=fixture();let available=true;
 const configuration=createConfigurationMaintenance({game:f.game,settings:[{module:'pf2e-reaction',key:'builtinReactionsEnabled',when:()=>true,reason:'test ownership',transform:value=>available?value.filter(v=>v!=='glimpse-of-redemption'):[...new Set([...value,'glimpse-of-redemption'])]}]});
 const observer=events.registerGlimpseConfigurationEvents({game:f.game,Hooks:f.Hooks,reconcile:configuration,onError:assert.fail});
 await observer.reconcileNow();assert.deepEqual(f.data.get('pf2e-reaction.builtinReactionsEnabled'),['shield-block']);
 available=false;await f.emit('updateSetting',{key:'trigger-engine.pf2e-trigger-triggers'});
 assert.deepEqual(f.data.get('pf2e-reaction.builtinReactionsEnabled'),['shield-block','glimpse-of-redemption']);
 assert.equal(f.writes.filter(k=>k==='pf2e-reaction.builtinReactionsEnabled').length,2);
 await f.emit('updateSetting',{key:'unrelated.preference'});assert.equal(f.writes.length,4);observer.dispose();
});
test('actor/item/token lifecycle reevaluates coverage, while a non-primary client cannot write',async()=>{
 const f=fixture();let calls=0;const observer=events.registerGlimpseConfigurationEvents({game:f.game,Hooks:f.Hooks,reconcile:async()=>{calls++},onError:assert.fail});
 for(const hook of ['createActor','updateActor','deleteActor','createItem','updateItem','deleteItem','createToken','updateToken','deleteToken','createScene','deleteScene','combatStart','updateCombat','createCombatant','updateCombatant','deleteCombatant'])await f.emit(hook,{});
 assert.equal(calls,16);
 f.game.user.id='player';await f.emit('updateActor',{});await observer.reconcileNow();assert.equal(calls,16);
 f.game.users.activeGM.id='player';await f.emit('updateUser',{});assert.equal(calls,17);
 observer.dispose();await f.emit('updateActor',{});assert.equal(calls,17);
});
test('a qualification change during an in-flight write schedules a fresh pass and errors are reported',async()=>{
 const f=fixture();let release,passes=0;const errors=[];
 const observer=events.registerGlimpseConfigurationEvents({game:f.game,Hooks:f.Hooks,onError:e=>errors.push(e.message),reconcile:async()=>{passes++;if(passes===1)await new Promise(resolve=>release=resolve);else if(passes===3)throw Error('write rejected')}});
 const first=observer.reconcileNow();await Promise.resolve();const second=f.emit('updateItem',{});release();await Promise.all([first,second]);assert.equal(passes,2);
 await observer.reconcileNow();assert.deepEqual(errors,['write rejected']);observer.dispose();
});

test('ordinary token movement never scans actor coverage or reads Patreon rules',async()=>{
 const f=fixture();let actorReads=0,patreonReads=0;
 const actor={items:[{type:'action',slug:'glimpse-of-redemption'}],covered:true};
 const token={get actor(){actorReads++;return actor}};
 f.game.modules.set('patreon-v3',{active:true});
 const get=f.game.settings.get;
 f.game.settings.get=(module,key)=>{if(module==='patreon-v3'&&key==='rulesV3')patreonReads++;return get(module,key)};
 const configuration=createConfigurationMaintenance({game:f.game,settings:[{module:'pf2e-reaction',key:'builtinReactionsEnabled',when:()=>true,reason:GLIMPSE_REACTION_REASON,transform:value=>glimpseReactionSetting(value,f.game,canSuppressGlimpseReminder([token.actor],{handlesActor:a=>a.covered}))}]});
 const observer=events.registerGlimpseConfigurationEvents({game:f.game,Hooks:f.Hooks,reconcile:configuration,onError:assert.fail});
 await observer.reconcileNow();assert.deepEqual(f.data.get('pf2e-reaction.builtinReactionsEnabled'),['shield-block']);
 actorReads=0;patreonReads=0;
 for(const change of [{x:100,y:200},{elevation:5},{rotation:90},{level:'next-level'},{movementAction:'fly'},{_movementHistory:[]},{_regions:['room']},{_id:'token',x:200,y:300,_movementHistory:[{x:100,y:200}],_regions:[],_stats:{modifiedTime:123}}])await f.emit('updateToken',token,change);
 assert.equal(actorReads,0,'movement must not read even one synthetic actor');
 assert.equal(patreonReads,0,'movement must not load unrelated Patreon rules');
 assert.deepEqual(f.data.get('pf2e-reaction.builtinReactionsEnabled'),['shield-block']);
 actor.covered=false;
 await f.emit('updateToken',token,{x:400,'delta.items':[{type:'action',slug:'glimpse-of-redemption'}]});
 assert.ok(actorReads>0,'mixed movement and actor delta changes must refresh coverage');
 assert.deepEqual(f.data.get('pf2e-reaction.builtinReactionsEnabled'),['glimpse-of-redemption','shield-block']);
 observer.dispose();
});

test('actor association, synthetic data, flags and unknown token changes still reevaluate coverage',async()=>{
 const f=fixture();let calls=0;
 const observer=events.registerGlimpseConfigurationEvents({game:f.game,Hooks:f.Hooks,reconcile:async()=>{calls++},onError:assert.fail});
 for(const change of [{actorId:'other'},{actorLink:false},{delta:{items:[]}},{'delta.items':[]},{flags:{'pf2e-third-party-automation':{enabled:false}}},{'flags.pf2e-third-party-automation.enabled':true},{x:100,actorLink:true},{futureField:'unknown'},{},undefined])await f.emit('updateToken',{},change);
 assert.equal(calls,10);observer.dispose();
});
