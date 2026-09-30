import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createManualEvents} from '../../scripts/exploration/manual-events.mjs';
test('a player native treatment is tagged locally and recorded once on the GM',async()=>{
 const handlers=new Map();const Hooks={on:(name,fn)=>{handlers.set(name,fn);return name},off(){}},records=[],message={id:'C',speaker:{actor:'H'},flags:{pf2e:{context:{actor:'H',origin:{actor:'Actor.H'},options:[]}}}};
 let middleware;const recorder=createManualEvents({game:{time:{worldTime:0},messages:new Map([['C',message]])},Hooks,ledger:{insertActivity:async a=>records.push(a)},nativeActions:{addMiddleware:fn=>{middleware=fn;return ()=>{}}},isAuthority:()=>false,sessionId:()=>null});recorder.start();
 const scope={slug:'treat-wounds',params:{target:{uuid:'Actor.P'}},tagRollOption:tag=>message.flags.pf2e.context.options.push(tag)};
 await middleware(scope,async()=>{handlers.get('preCreateChatMessage')?.(message,message);return [{actor:{uuid:'Actor.H',id:'H',items:[]},message}]});
 assert.equal(message.flags['pf2e-third-party-automation']?.explorationManualNative?.patientUUID,'Actor.P');assert.equal(records.length,0);
 const gmHandlers=new Map(),gm=createManualEvents({game:{time:{worldTime:0},messages:new Map([['C',message]])},Hooks:{on:(n,f)=>{gmHandlers.set(n,f);return n},off(){}},ledger:{insertActivity:async a=>records.push(a)},isAuthority:()=>true,sessionId:()=> 'S'});gm.start();gmHandlers.get('createChatMessage')(message);gmHandlers.get('createChatMessage')(message);await new Promise(r=>setImmediate(r));assert.equal(records.length,1);
});
test('async session validation cannot race the same manual source twice',async()=>{const records=[],r=createManualEvents({game:{time:{worldTime:0}},ledger:{getSession:async()=>({status:'recording'}),insertActivity:async a=>records.push(a)},isAuthority:()=>true,sessionId:()=> 'S'});await Promise.all([r.observe({id:'C',actorUUID:'H'}),r.observe({id:'C',actorUUID:'H'})]);assert.equal(records.length,1)});
