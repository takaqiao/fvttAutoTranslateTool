import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createReactionChecks,REACTION_CHECK_SOURCES} from '../scripts/reaction-checks.mjs';

function fixture({coexisting=false,middleware=false}={}){
 const actor={id:'roller',uuid:'Actor.roller',type:'character',items:new Map(),flags:{}};
 if(coexisting)actor.items.set('clock',{id:'clock',type:'feat',sourceId:REACTION_CHECK_SOURCES.clock});
 const gm={id:'gm',isGM:true},game={user:gm,users:{activeGM:gm},actors:new Map([[actor.id,actor]]),scenes:[],messages:new Map(),time:{worldTime:0}};
 const registrations=[],calls=[],order=[];let hookId=0;
 const luck={handlesActor:a=>a===actor,interceptCheck:async(native,...args)=>{calls.push(args);order.push('luck');return native(...args)}};
 const provider=createReactionChecks({game,halflingLuck:luck,nativeCheckMiddleware:middleware?(native,check,...args)=>{order.push('outer');return native({...check,outerBonus:true},...args)}:undefined});
 const cleanup=provider.register({Hooks:{on:()=>++hookId,off:()=>{}},libWrapper:{register:(id,path,callback,type)=>registrations.push({id,path,callback,type}),unregister:()=>{}}});
 assert.equal(registrations.length,1);assert.equal(registrations[0].path,'game.pf2e.Check.roll');
 const check={slug:'arcana',modifiers:[]},event={type:'native-event'},callback=()=>{};
 const nativeCalls=[];const native=async(...args)=>{order.push('native');nativeCalls.push(args);return 'native-result'};
 const run=(extra={})=>registrations[0].callback.call({},native,check,{actor,type:'skill-check',options:new Set(),...extra},event,callback);
 return {actor,luck,run,calls,check,event,callback,nativeCalls,order,cleanup};
}

test('the existing single Check wrapper routes Luck draft callers without replacing their event or callback',async()=>{
 const f=fixture();assert.equal(await f.run({createMessage:false}),'native-result');
 assert.equal(f.calls.length,1);assert.equal(f.nativeCalls.length,1);assert.equal(f.calls[0][0],f.check);
 assert.equal(f.calls[0][1].createMessage,false);assert.equal(f.calls[0][2],f.event);assert.equal(f.calls[0][3],f.callback);f.cleanup();
});

test('existing outer native middleware runs once before Luck and preserves its prepared check',async()=>{
 const f=fixture({middleware:true});await f.run();assert.deepEqual(f.order,['outer','luck','native']);assert.equal(f.calls[0][0].outerBonus,true);f.cleanup();
});

test('rerolls and other check types bypass the Luck branch',async()=>{
 const f=fixture();await f.run({isReroll:true});await f.run({type:'attack-roll'});await f.run({type:'initiative'});
 assert.equal(f.calls.length,0);assert.equal(f.nativeCalls.length,3);f.cleanup();
});

test('a coexisting Clock holder keeps the existing provider rather than entering competing Luck pipelines',async()=>{
 const f=fixture({coexisting:true});await f.run({createMessage:false});assert.equal(f.calls.length,0);assert.equal(f.nativeCalls.length,1);f.cleanup();
});
