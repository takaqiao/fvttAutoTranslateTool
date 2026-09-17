import test from 'node:test';
import assert from 'node:assert/strict';
import {createReactionBudget,genericReactionAvailable} from '../scripts/reaction-budget.mjs';
import {GLIMPSE_SOURCES} from '../scripts/glimpse-source.mjs';
import {MODULE_ID as M} from '../scripts/rules.mjs';
function fixture({managed=true,paid=true}={}){
 const user={id:'gm',isGM:true},hooks=new Map(),errors=[];
 const update=function(changes){for(const[path,value]of Object.entries(changes)){let p=this;const keys=path.split('.');for(const k of keys.slice(0,-1))p=p[k]??={};p[keys.at(-1)]=structuredClone(value)}return Promise.resolve(this)};
 const actor={id:'pc',uuid:'Actor.pc',items:new Map(),flags:{},system:{resources:{reactions:{max:1}}},testUserPermission:u=>u===user};
 const item={id:'glimpse',uuid:'Actor.pc.Item.glimpse',actor,sourceId:GLIMPSE_SOURCES.glimpse,system:{actionType:{value:'reaction'},slug:'glimpse-of-redemption'}};actor.items.set(item.id,item);
 const token={uuid:'Scene.scene.Token.pc',actor},claim={nonce:'nonce',claimKey:'glimpse:nonce',actorUuid:actor.uuid,itemUuid:item.uuid,tokenUuid:token.uuid,userId:user.id,combatId:'encounter',combatantId:'pc',epoch:'encounter:2',status:'paid',messageId:'card'};
 const combatant={id:'pc',actor,token,flags:{[M]:{glimpseClaims:paid?[claim]:[],reactionBudget:{epoch:'encounter:2',entries:paid?[{type:'reaction',cost:1,slug:'glimpse-of-redemption',claimKey:claim.claimKey}]:[]}}},update};
 const combat={id:'encounter',started:true,round:2,turn:0,turns:[combatant]},game={user,users:new Map([[user.id,user]]),modules:new Map(),combat,combats:new Map([[combat.id,combat]]),messages:new Map()};game.users.activeGM=user;
 const message={id:'card',actor,item,author:user,speaker:{actor:'pc',scene:'scene',token:'pc'},rolls:[],flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid}},[M]:paid?{glimpseUse:{nonce:claim.nonce,claimKey:claim.claimKey}}:{}},update};game.messages.set(message.id,message);
 const budget=createReactionBudget({game,handlesGlimpse:a=>managed&&a===actor,onError:e=>errors.push(e)});budget.register({Hooks:{on:(name,fn)=>{hooks.set(name,fn);return fn},off(){}}});
 return {game,actor,claim,combat,combatant,message,budget,hooks,errors,entries:()=>combatant.flags[M].reactionBudget.entries};
}
test('paid Glimpse card merges with its existing claim instead of charging twice',async()=>{
 const f=fixture();assert.equal(await f.budget.record(f.message,f.game.user.id),true);assert.equal(f.entries().length,1);assert.equal(f.entries()[0].msgId,'card');assert.equal(await f.budget.record(f.message,f.game.user.id),false);
});
test('managed Glimpse display waits for payment proof; unavailable provider retains manual accounting',async()=>{
 const managed=fixture({paid:false});assert.equal(await managed.budget.record(managed.message,managed.game.user.id),false);assert.equal(managed.entries().length,0);
 const fallback=fixture({managed:false,paid:false});assert.equal(await fallback.budget.record(fallback.message,fallback.game.user.id),true);assert.equal(fallback.entries().length,1);
});
test('late Glimpse card retains its paid epoch without consuming the new turn',async()=>{
 const f=fixture();f.combat.round=3;f.combatant.flags[M].reactionBudget={epoch:'encounter:3',entries:[]};f.message.flags[M].reactionBudget={epoch:'encounter:2'};
 await f.budget.record(f.message,f.game.user.id);assert.equal(f.entries().length,0);assert.equal(f.message.flags[M].reactionBudget.epoch,'encounter:2');assert.equal(genericReactionAvailable(f.actor,f.game),true);
});
test('durable paid Glimpse remains spent even if the ordinary ledger is rebuilt',()=>{
 const f=fixture();f.combatant.flags[M].reactionBudget.entries=[];assert.equal(genericReactionAvailable(f.actor,f.game),false);f.claim.status='refunded';assert.equal(genericReactionAvailable(f.actor,f.game),true);
});
test('Glimpse proof update triggers exact card accounting after creation',async()=>{
 const f=fixture();f.hooks.get('updateChatMessage')(f.message,{[`flags.${M}.glimpseUse`]:f.message.flags[M].glimpseUse},{},f.game.user.id);await new Promise(r=>setImmediate(r));assert.equal(f.entries().length,1);assert.equal(f.entries()[0].msgId,'card');assert.deepEqual(f.errors,[]);
});
