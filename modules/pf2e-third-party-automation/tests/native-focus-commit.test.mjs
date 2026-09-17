import test from 'node:test';
import assert from 'node:assert/strict';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

function fixture({deferred=false,lostReply=false}={}){
 const wrappers=new Map(),rpc=new Map(),writes=[],gm={id:'gm'},user={id:'owner'},docs=new Map();
 const game={user:gm,users:new Map([['gm',gm],['owner',user]]),time:{worldTime:1}};game.users.activeGM=gm;
 const actor={uuid:'Actor.pc',type:'character',flags:{},system:{resources:{focus:{value:2,max:3}}},testUserPermission:u=>u===user||u===gm,items:new Map()};
 const apply=async changes=>{writes.push(structuredClone(changes));for(const[p,v]of Object.entries(changes)){let obj=actor;const keys=p.split('.');for(const k of keys.slice(0,-1))obj=obj[k]??={};obj[keys.at(-1)]=structuredClone(v);}if(lostReply&&Object.hasOwn(changes,'system.resources.focus.value'))throw Error('reply lost');return actor;};
 actor.update=changes=>{const wrap=wrappers.get('CONFIG.Actor.documentClass.prototype.update');return wrap?wrap.call(actor,apply,changes,{}):apply(changes)};
 const entry={uuid:'Actor.pc.Item.entry',id:'entry',type:'spellcastingEntry',actor};
 const item={uuid:'Actor.pc.Item.spell',id:'spell',type:'spell',actor,sourceId:'Compendium.test.spell',rank:1,spellcasting:entry,system:{location:{value:'entry'},cast:{focusPoints:1}}};
 actor.items.set(entry.id,entry);actor.items.set(item.id,item);for(const doc of [actor,item,entry])docs.set(doc.uuid,doc);
 const native=async()=>{if(deferred)await Promise.resolve();await actor.update({'system.resources.focus.value':actor.system.resources.focus.value-1});return true};
 entry.consume=(spell,rank,slot,cap)=>wrappers.get('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.consume').call(entry,native,spell,rank,slot,cap);
 const casts=createNativeCastEvents({game,fromUuid:async uuid=>docs.get(uuid)});casts.addActorMatcher(a=>a===actor);casts.register({libWrapper:{register:(_id,p,fn)=>wrappers.set(p,fn)},socket:{register:(p,fn)=>rpc.set(p,fn)}});
 const request={id:'cast-one',actorUuid:actor.uuid,itemUuid:item.uuid,sourceId:item.sourceId,entryUuid:entry.uuid,rank:1,slotId:null,focusPoints:1,overlayIds:[]};
 const pay=()=>rpc.get('native-cast-pay').call({socketdata:{userId:user.id}},request);
 return {casts,actor,item,entry,game,gm,user,writes,pay,request};
}
test('native focus debit and provider proof are submitted in the same original actor update',async()=>{
 const f=fixture();f.casts.addConsumePolicy(async(c,next)=>{assert.equal(typeof c.expectFocusCommit,'function');c.expectFocusCommit({before:2,cost:1,changes:proof=>({[`flags.${ID}.focusProof`]:proof})});return next()});
 const result=await f.pay();assert.equal(result.ok,true);assert.equal(f.actor.system.resources.focus.value,1);
 const debit=f.writes.filter(w=>Object.hasOwn(w,'system.resources.focus.value'));assert.equal(debit.length,1);assert.equal(debit[0][`flags.${ID}.focusProof`].castNonce,'cast-one');assert.equal(debit[0][`flags.${ID}.focusProof`].after,1);
 await f.pay();assert.equal(f.writes.filter(w=>Object.hasOwn(w,'system.resources.focus.value')).length,1);
});
test('a deferred native debit is not attributed by timing or matching amount',async()=>{
 const f=fixture({deferred:true});f.casts.addConsumePolicy(async(c,next)=>{c.expectFocusCommit({before:2,cost:1,changes:p=>({[`flags.${ID}.focusProof`]:p})});return next()});
 const result=await f.pay();assert.equal(result.ok,false);assert.match(result.error,/原生聚能/);assert.equal(f.actor.flags[ID]?.focusProof,undefined);assert.equal(f.actor.flags[ID].nativeCasts[0].state,'uncertain');assert.equal(f.actor.system.resources.focus.value,1);
});
test('lost reply leaves the atomic provider proof and uncertain native receipt without another debit',async()=>{
 const f=fixture({lostReply:true});f.casts.addConsumePolicy(async(c,next)=>{c.expectFocusCommit({before:2,cost:1,changes:p=>({[`flags.${ID}.focusProof`]:p})});return next()});
 assert.equal((await f.pay()).ok,false);assert.equal(f.actor.flags[ID].focusProof.after,1);assert.equal(f.actor.flags[ID].nativeCasts[0].state,'uncertain');assert.equal((await f.pay()).ok,false);assert.equal(f.actor.system.resources.focus.value,1);
});
test('focus extension cannot change native amount or other document fields',async()=>{
 const f=fixture();f.casts.addConsumePolicy(async(c,next)=>{c.expectFocusCommit({before:2,cost:1,changes:()=>({'system.resources.focus.value':0})});return next()});
 assert.equal((await f.pay()).ok,false);assert.equal(f.actor.system.resources.focus.value,2);
});
test('external resource operations share the native payment actor queue',async()=>{
 const f=fixture();assert.equal(typeof f.casts.withActorResourceLock,'function');let release;const gate=new Promise(r=>release=r);const order=[];
 const lock=f.casts.withActorResourceLock(f.actor,async()=>{order.push('grant');await gate;});const pay=f.pay();await Promise.resolve();assert.equal(f.actor.system.resources.focus.value,2);release();await lock;assert.equal((await pay).ok,true);order.push('paid');assert.deepEqual(order,['grant','paid']);
});
