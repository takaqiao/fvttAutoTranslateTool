import test from 'node:test';
import assert from 'node:assert/strict';
import {runDamagePipeline} from '../scripts/native-context.mjs';
let api={};try{api=await import('../scripts/activity-result-lifecycle.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
const ID='pf2e-third-party-automation';
const set=(object,path,value)=>{const bits=path.split('.');let at=object;for(const bit of bits.slice(0,-1))at=at[bit]??={};at[bits.at(-1)]=value;};
function fixture({applied=false}={}){
 const gm={id:'gm'},messages=new Map(),game={user:gm,users:{activeGM:gm},messages},callbacks=new Map(),Hooks={on(name,fn){callbacks.set(name,fn);return name},off(name){callbacks.delete(name)}};
 const doc=data=>({...data,async update(changes){for(const[key,value]of Object.entries(changes))set(this,key,value);}});
 const activity=doc({id:'activity',author:{id:'player'},flags:{[ID]:{usage:{status:'done'},spellCombinationUse:{state:'done'}}}});
 const attack=doc({id:'first',author:{id:'player'},speaker:{actor:'pc'},isCheckRoll:true,rolls:[{_evaluated:true}],flags:{[ID]:{spellCombinationAttack:{activityMessageId:activity.id,index:0,kind:'strike'}},pf2e:{origin:{actor:'Actor.pc',uuid:'Actor.pc.Item.weapon'},context:{type:'attack-roll',outcome:'success',target:{token:'Scene.s.Token.t'},options:[`${ID}:spell-combination:${activity.id}`]}}}});
 const damage=doc({id:'damage',flags:{[ID]:{spellCombinationDamage:{activityMessageId:activity.id,attacks:[{messageId:attack.id}]}},pf2e:{appliedDamage:applied?{isReverted:false}:null,context:{type:'damage-roll',options:[]}}}});
 for(const message of [activity,attack,damage])messages.set(message.id,message);
 assert.equal(typeof api.createActivityResultLifecycle,'function');
 const provider=api.createActivityResultLifecycle({game,getRollContext:()=>({messageId:'damage'})});provider.register({Hooks});
 const reroll=doc({...attack,id:'kept',flags:{pf2e:{...attack.flags.pf2e,context:{...attack.flags.pf2e.context,isReroll:true,outcome:'criticalSuccess'}}}});
 return {game,callbacks,activity,attack,damage,reroll,provider};
}
for(const applied of [false,true])test(`kept native reroll invalidates old ${applied?'applied':'unapplied'} combination damage without charging or changing HP`,async()=>{
 const f=fixture({applied});f.game.messages.delete(f.attack.id);f.game.messages.set(f.reroll.id,f.reroll);
 await f.callbacks.get('createChatMessage')(f.reroll);
 assert.equal(f.damage.flags[ID].activityResult.status,applied?'undo-required':'superseded');
 assert.equal(f.reroll.flags[ID].spellCombinationAttack.activityMessageId,'activity');
 assert.equal(f.activity.flags[ID].usage.status,'waiting');
 assert.throws(()=>f.provider.beforeDamage({}, {damage:{}}),/重掷|撤销/);
});
test('ordinary copied attack context is not a kept native reroll',async()=>{
 const f=fixture();delete f.reroll.flags.pf2e.context.isReroll;await f.callbacks.get('createChatMessage')(f.reroll);
 assert.equal(f.damage.flags[ID].activityResult,undefined);assert.equal(f.activity.flags[ID].usage.status,'done');
});
test('a foreign roller cannot supersede the original activity',async()=>{
 const f=fixture();f.reroll.author={id:'other'};await f.callbacks.get('createChatMessage')(f.reroll);assert.equal(f.damage.flags[ID].activityResult,undefined);
});
test('native damage-taken receipt requires undo even when the original roll has no applied flag',async()=>{
 const f=fixture();
 f.game.messages.set('receipt',{id:'receipt',flags:{pf2e:{appliedDamage:{uuid:'Actor.npc',isReverted:false},context:{type:'damage-taken',options:[`${ID}:source:damage:0`]}}}});
 await f.callbacks.get('createChatMessage')(f.game.messages.get('receipt'));
 f.game.messages.delete(f.attack.id);f.game.messages.set(f.reroll.id,f.reroll);await f.callbacks.get('createChatMessage')(f.reroll);
 assert.equal(f.damage.flags[ID].activityResult.status,'undo-required');
});
test('a reverted native receipt no longer requires an additional undo',async()=>{
 const f=fixture();
 f.game.messages.set('receipt',{id:'receipt',flags:{pf2e:{appliedDamage:{uuid:'Actor.npc',isReverted:true},context:{type:'damage-taken',options:[`${ID}:source:damage:0`]}}}});
 await f.callbacks.get('createChatMessage')(f.game.messages.get('receipt'));
 f.game.messages.delete(f.attack.id);f.game.messages.set(f.reroll.id,f.reroll);await f.callbacks.get('createChatMessage')(f.reroll);
 assert.equal(f.damage.flags[ID].activityResult.status,'superseded');
});
test('the active GM can keep a native reroll for the original player',async()=>{
 const f=fixture();f.reroll.author=f.game.user;f.game.messages.delete(f.attack.id);f.game.messages.set(f.reroll.id,f.reroll);
 await f.callbacks.get('createChatMessage')(f.reroll);assert.equal(f.damage.flags[ID].activityResult.status,'superseded');
});
test('kept native activity rerolls suppress Workbench auto damage before creation',async()=>{
 const f=fixture();let changes;
 f.reroll.updateSource=data=>{changes=data;};
 await f.callbacks.get('preCreateChatMessage')?.(f.reroll);
 assert.equal(changes?.['flags.xdy-pf2e-workbench.noAutoDamageRoll'],true);
});
test('old damage is blocked synchronously while kept-check linkage is still saving',async()=>{
 const f=fixture();let release;const gate=new Promise(resolve=>{release=resolve;});
 const update=f.reroll.update;f.reroll.update=async changes=>{await gate;return update.call(f.reroll,changes);};
 f.game.messages.delete(f.attack.id);f.game.messages.set(f.reroll.id,f.reroll);
 const pending=f.callbacks.get('createChatMessage')(f.reroll);
 try{assert.throws(()=>f.provider.beforeDamage({}, {damage:{}}),/重掷|撤销/);}finally{release();await pending;}
});
test('a player blocks stale damage locally before the GM broadcasts persisted invalidation',async()=>{
 const f=fixture();f.game.user={id:'player'};
 f.game.messages.delete(f.attack.id);f.game.messages.set(f.reroll.id,f.reroll);
 const pending=f.callbacks.get('createChatMessage')(f.reroll);
 assert.throws(()=>f.provider.beforeDamage({}, {damage:{}}),/重掷|撤销/);await pending;
 assert.equal(f.damage.flags[ID].activityResult,undefined);
});
test('damage waiting for a reaction rechecks its source immediately before native HP application',async()=>{
 const f=fixture();let entered,release,applications=0;
 const waiting=new Promise(resolve=>{entered=resolve;}),gate=new Promise(resolve=>{release=resolve;});
 const operation=runDamagePipeline({actor:{},params:{damage:{}},providers:[f.provider,{beforeDamage:async()=>{entered();await gate;return null;}}],apply:params=>f.provider.applyNativeDamage({},params,()=>{applications++;}),onError:()=>{}});
 await waiting;f.game.messages.delete(f.attack.id);f.game.messages.set(f.reroll.id,f.reroll);await f.callbacks.get('createChatMessage')(f.reroll);
 release();await assert.rejects(operation,/重掷|撤销/);assert.equal(applications,0);
});
test('a deleted damage source releases only its obsolete local invalidation record',async()=>{
 const f=fixture();f.game.messages.delete(f.attack.id);f.game.messages.set(f.reroll.id,f.reroll);await f.callbacks.get('createChatMessage')(f.reroll);
 assert.throws(()=>f.provider.beforeDamage({}, {damage:{}}),/重掷/);
 f.game.messages.delete(f.damage.id);f.callbacks.get('deleteChatMessage')(f.damage);
 assert.doesNotThrow(()=>f.provider.beforeDamage({}, {damage:{}}));
});
test('a late native damage receipt still has an explicit native-undo instruction',async()=>{
 const f=fixture();f.game.messages.delete(f.attack.id);f.game.messages.set(f.reroll.id,f.reroll);await f.callbacks.get('createChatMessage')(f.reroll);
 assert.match(f.activity.flags[ID].usage.result,/如旧伤害已应用.*撤销/);
 let rendered='';const root={querySelector:()=>null,querySelectorAll:()=>[],insertAdjacentHTML:(_position,html)=>{rendered=html;}};
 f.callbacks.get('renderChatMessageHTML')(f.damage,root);assert.match(rendered,/如旧伤害已应用.*撤销/);
});
