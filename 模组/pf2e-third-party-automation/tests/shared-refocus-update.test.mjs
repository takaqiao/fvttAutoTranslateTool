import test from 'node:test';
import assert from 'node:assert/strict';
import {createNativeCastEvents} from '../scripts/amp-cast-events.mjs';
import {registerAvRefocusEvents} from '../scripts/av-refocus-events.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

test('Refocus and native casts share one Actor.update registration and retain the exact Refocus receipt',async()=>{
 const wrappers=new Map(),hooks=new Map(),writes=[];
 const gm={id:'gm'},game={user:gm,users:new Map([['gm',gm]]),time:{worldTime:1},PF2eWorkbench:{refocus(){}}};game.users.activeGM=gm;
 const actor={uuid:'Actor.pc',type:'character',items:new Map(),flags:{},system:{resources:{focus:{value:0,max:2}}},testUserPermission:()=>true};
 const token={actor,document:{uuid:'Scene.scene.Token.pc'}},canvas={tokens:{controlled:[token]}};
 const libWrapper={register(_id,path,fn){assert(!wrappers.has(path),`duplicate ${path}`);wrappers.set(path,fn)},unregister(_id,path){wrappers.delete(path)}};
 const Hooks={on(name,fn){hooks.set(name,fn);return fn},off(name){hooks.delete(name)}};
 const casts=createNativeCastEvents({game});casts.register({libWrapper});
 actor.update=(changes,options={})=>wrappers.get('CONFIG.Actor.documentClass.prototype.update').call(actor,(next,opts)=>{writes.push({changes:next,options:opts});if(Object.hasOwn(next,'system.resources.focus.value'))actor.system.resources.focus.value=next['system.resources.focus.value'];return Promise.resolve(actor)},changes,options);
 const off=registerAvRefocusEvents({game,Hooks,libWrapper,canvas,actorMatchers:[a=>a===actor],registerActorUpdate:fn=>casts.addActorUpdateMiddleware(fn)});
 await wrappers.get('game.PF2eWorkbench.refocus').call(game.PF2eWorkbench,()=>actor.update({'system.resources.focus.value':2}),[actor]);
 assert.equal(writes.length,1);const proof=writes[0].options[ID].refocusReceipt;
 assert.equal(proof.before,0);assert.equal(proof.after,2);assert.equal(proof.actorUuid,actor.uuid);assert.equal(proof.tokenUuid,token.document.uuid);
 assert.deepEqual(writes[0].changes[`flags.${ID}.avRefocusIntent`],proof);
 await actor.update({'system.resources.focus.value':1});assert.equal(writes[1].options[ID],undefined);
 off();assert(wrappers.has('CONFIG.Actor.documentClass.prototype.update'));assert(!wrappers.has('game.PF2eWorkbench.refocus'));
 await actor.update({'system.resources.focus.value':0});assert.equal(writes[2].options[ID],undefined);
});
