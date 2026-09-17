import test from 'node:test';
import assert from 'node:assert/strict';
import {createMedicNative} from '../scripts/medic-native.mjs';
const M='pf2e-third-party-automation';
test('native Poison and First Aid receive exact actor/target, continuation and player variant',async()=>{
 const calls=[],actor={uuid:'Actor.healer'},patient={uuid:'Actor.patient'},target={uuid:'Scene.s.Token.t',actor:patient},healer={object:{id:'h'}},user={id:'owner'},continuation={actorUuid:actor.uuid,cardId:'card',nonce:'nonce'};
 const game={pf2e:{actions:new Map(['treat-poison','administer-first-aid'].map(key=>[key,{use:async args=>{calls.push([key,args]);return [{actor}];}}]))}};
 const delegate=createMedicNative({game,choose:async()=> 'stabilize'});
 for(const branch of ['treat-poison','administer-first-aid'])await delegate({actor,target,healer,user,branch,continuation,validate:()=>{}});
 assert.equal(calls.length,2);for(const [,args]of calls){assert.deepEqual(args.actors,[actor]);assert.equal(args.target,patient);assert.deepEqual(args[M].metapowerContinuation,continuation);}
 assert.equal(calls[1][1].variant,'stabilize');
});
test('Workbench macro gets fixed lexical selections and cannot use the executing GM selection',async()=>{
 const actor={uuid:'Actor.healer'},healer={object:{id:'h',actor}},target={object:{id:'t',actor:{uuid:'Actor.patient'}}},gm={id:'gm',targets:new Set([{id:'unrelated'}])},game={user:gm,modules:new Map([['xdy-pf2e-workbench',{active:true}]]),packs:new Map()};
 const canvas={tokens:{controlled:[{id:'unrelated'}]}},calls=[];
 const macro={execute:async scope=>{calls.push(scope);const d=new scope.Dialog({buttons:{yes:{callback:async()=>{assert.equal(scope.game.user.targets.has(target.object),true);assert.equal(scope.canvas.tokens.controlled[0],healer.object);}}},render:()=>{}});await d.options.buttons.yes.callback({find:()=>({val:()=>1,prop(){return this},trigger(){return this}})});}};
 game.packs.set('xdy-pf2e-workbench.asymonous-benefactor-macros-internal',{getDocuments:async()=>[macro]});
 class Dialog{constructor(options){this.options=options;}}
 const delegate=createMedicNative({game,canvas,Dialog,ChatMessage:{getSpeaker:({actor})=>({actor:actor.uuid})}});
 await delegate({actor,healer,target,user:{id:'owner'},branch:'battle-medicine',validate:()=>{}});
 assert.equal(calls.length,1);assert.equal(game.user,gm);assert.equal([...gm.targets][0].id,'unrelated');assert.equal(canvas.tokens.controlled[0].id,'unrelated');assert.equal(calls[0].ChatMessage.getSpeaker().actor,actor.uuid);
});
test('Workbench validates immediately before native submission and propagates cancellation',async()=>{
 const actor={},healer={object:{actor}},target={object:{actor:{}}},game={user:{targets:new Set()},modules:new Map([['xdy-pf2e-workbench',{active:true}]]),packs:new Map()},calls=[];
 class Dialog{constructor(options){this.options=options;}}
 game.packs.set('xdy-pf2e-workbench.asymonous-benefactor-macros-internal',{getDocuments:async()=>[{execute:async scope=>{const d=new scope.Dialog({buttons:{yes:{callback:()=>calls.push('applied')},no:{}},render:()=>{}});await d.options.buttons.no.callback();}}]});
 const result=await createMedicNative({game,canvas:{tokens:{}},Dialog})({actor,healer,target,branch:'battle-medicine',validate:()=>{}});
 assert.equal(result.status,'cancelled');assert.deepEqual(calls,[]);
});
