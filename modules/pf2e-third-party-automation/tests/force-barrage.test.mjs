import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createForceBarrageBridge} from '../scripts/force-barrage.mjs';
const ID='pf2e-third-party-automation';
function fixture(){
 const user={id:'gm',active:true,isGM:true},actor={id:'a',uuid:'Actor.a',type:'character',canAct:true,isDead:false,items:new Map(),testUserPermission:u=>u===user};
 const entry={id:'e',uuid:'Actor.a.Item.e',actor,system:{slots:{slot3:{value:2}}}},item={id:'s',uuid:'Actor.a.Item.s',actor,system:{},flags:{}};actor.items.set('s',item);actor.items.set('e',entry);
 const users=new Map([['gm',user]]);users.activeGM=user;const game={user,users,actors:new Map([['a',actor]]),messages:new Map(),settings:{get:()=> 'public'}};
 const scene={id:'sc',tokens:new Map()},token={id:'src',uuid:'Scene.sc.Token.src',actor,parent:scene},targets=[1,2].map(i=>({id:`t${i}`,uuid:`Scene.sc.Token.t${i}`,actor:{type:'npc'},parent:scene}));
 const docs=new Map([actor,item,entry,token,...targets].map(d=>[d.uuid,d]));const calls=[],rpcs=new Map(),adapters=new Map(),hooks=new Map();let record,seq=0;
 const ledger={current:()=>record,assertCastIntent:()=>true};
 for(const method of ['claim','startCast','bindCast','startTarget','recordRoll','beginPublication','finishPublication','uncertain','finishWithoutDamage'])ledger[method]=async p=>{calls.push([method,p]);if(method==='claim')record={...p,nonce:'bridge',status:'claimed'};if(method==='bindCast')record={...record,castNonce:p.outcome.castNonce,originalMessageUuid:p.outcome.message.uuid};if(method==='finishPublication'){record.targets??=[];record.targets.push({targetUuid:p.targetUuid,status:'published',messageUuid:p.message.uuid});record.status=record.targets.length===2?'delivered':'producing';}return record;};
 const nativeCasts={addInvocationAdapter:(k,a)=>adapters.set(k,a),withActorResourceLock:(_a,fn)=>fn()};
 const Hooks={on:(n,f)=>{hooks.set(n,f);return f},off:()=>{}};
 let answer={actions:3,visibilityConfirmed:true,allocations:targets.map((t,i)=>({targetUuid:t.uuid,count:i?2:4}))};
 let rolling=0,publishing=0,native=0;const rolls=[];
 const adapter={getMissileCount:()=>6,run:async p=>{calls.push(['adapter',p]);await p.bridge.payAndBindOriginalCast();for(const a of p.allocations){if(!a.count)continue;const roll={async evaluate(){rolling++;this.total=a.count*2;return this},toJSON(){return {class:'DamageRoll',formula:`${a.count}d4+${a.count}`,total:this.total,terms:[]}},async toMessage(data,opts){publishing++;const m={id:`d${publishing}`,uuid:`ChatMessage.d${publishing}`,...data};game.messages.set(m.id,m);calls.push(['publish',opts]);return m;}};rolls.push(roll);await p.bridge.publishTarget({roll,messageData:{flags:{'pf2e-toolbelt.targetHelper.targets':[a.targetUuid]},flavor:'native',speaker:{actor:'a',token:'src',scene:'sc'}},targetUuid:a.targetUuid});}return {status:'completed'};}};
 const outcome={status:'completed',castNonce:'cast',message:{id:'c',uuid:'ChatMessage.c'},receipt:{state:'used'}};
 const next=async()=>{native++;return 'original'};next.withOutcome=async p=>{native++;calls.push(['cast',p]);return outcome};
 docs.set(outcome.message.uuid,outcome.message);
 const config={game,nativeCasts,ledger,fromUuid:async u=>docs.get(u)??[...game.messages.values()].find(m=>m.uuid===u),choose:async()=>answer,loadWorkbench:async()=>adapter,assess:()=>({handled:true,eligible:true,rank:3,base:item}),validateTargets:()=>targets,getContext:()=>({token,targets}),randomId:()=>`inv-${++seq}`,onError:()=>{}};
 const bridge=createForceBarrageBridge(config);bridge.register({Hooks,socket:{register:(k,f)=>rpcs.set(k,f)}});
 return {game,user,actor,item,entry,token,targets,calls,ledger,adapter,adapters,rpcs,hooks,bridge,config,next,outcome,rolls,set answer(v){answer=v},get counts(){return {rolling,publishing,native}},run:()=>bridge.interceptCast({item,entry,options:{rank:3}},next)};
}
test('original Cast pays once then evaluates and publishes each positive target once with exact provenance',async()=>{
 const f=fixture();await f.run();assert.deepEqual(f.counts,{native:1,rolling:2,publishing:2});
 const steps=f.calls.map(c=>c[0]);assert.ok(steps.indexOf('cast')<steps.indexOf('startTarget'));assert.equal(steps.at(-1),'finishPublication');
 for(const [i,m]of [...f.game.messages.values()].entries()){
  assert.deepEqual(m.flags['pf2e-toolbelt'].targetHelper.targets,[f.targets[i].uuid]);
  assert.equal(m.flags[ID].forceBarrage.castNonce,'cast');assert.equal(m.flags.pf2e.origin.uuid,f.item.uuid);assert.equal(m.flags.pf2e.origin.castRank,3);assert.deepEqual(m.whisper,[]);assert.equal(m.blind,false);
 }
 assert.ok(f.calls.filter(c=>c[0]==='publish').every(c=>c[1].messageMode==='public'));
});
test('cancel and malformed allocation create no claim, native Cast, die, or card',async()=>{
 const f=fixture();f.answer=null;await f.run();assert.deepEqual(f.counts,{native:0,rolling:0,publishing:0});assert.equal(f.calls.length,0);
 f.answer={actions:3,visibilityConfirmed:true,allocations:f.targets.map(t=>({targetUuid:t.uuid,count:NaN}))};await assert.rejects(f.run());assert.equal(f.calls.length,0);
});
test('unrelated casts preserve original continuation without loading macro',async()=>{
 const f=fixture();const bridge=createForceBarrageBridge({...f.config,assess:()=>({handled:false}),loadWorkbench:()=>{throw Error('must not load')}});assert.equal(await bridge.interceptCast({item:f.item,entry:f.entry,options:{}},f.next),'original');assert.equal(f.counts.native,1);
});
test('local simultaneous double Cast is rejected before a second dialog or payment',async()=>{
 const f=fixture();let release;const pending=new Promise(r=>release=r);const bridge=createForceBarrageBridge({...f.config,choose:()=>pending});bridge.register({Hooks:{on(){},off(){}},socket:{register(){}}});
 const one=bridge.interceptCast({item:f.item,entry:f.entry,options:{rank:3}},f.next);await assert.rejects(bridge.interceptCast({item:f.item,entry:f.entry,options:{rank:3}},f.next));release(null);await one;assert.equal(f.counts.native,0);
});
test('failed or disrupted native Cast never evaluates damage or retries payment',async()=>{
 for(const status of ['disrupted','throw']){const f=fixture();f.next.withOutcome=async()=>{f.calls.push(['cast']);if(status==='throw')throw Error('lost response');return {...f.outcome,status};};await assert.rejects(f.run());assert.equal(f.counts.rolling,0);assert.equal(f.calls.filter(c=>c[0]==='cast').length,1);assert.equal(f.calls.some(c=>c[0]===(status==='disrupted'?'finishWithoutDamage':'uncertain')),true);}
});
test('second target publication failure preserves first card and never retries rolls or payment',async()=>{
 const f=fixture();f.ledger.finishPublication=async p=>{f.calls.push(['finishPublication',p]);if(p.targetUuid===f.targets[1].uuid)throw Error('lost message response');return {nonce:'bridge'};};await assert.rejects(f.run(),/lost message/);assert.deepEqual(f.counts,{native:1,rolling:2,publishing:2});assert.equal(f.calls.filter(c=>c[0]==='uncertain').length,1);assert.equal(f.calls.some(c=>c[0]==='finish'),false);
});
test('GM handoff before confirmation rejects prior to claim and original payment',async()=>{
 const f=fixture();const bridge=createForceBarrageBridge({...f.config,choose:async()=>{f.game.users.activeGM={id:'other',active:true};return {actions:3,visibilityConfirmed:true,allocations:f.targets.map((t,i)=>({targetUuid:t.uuid,count:i?2:4}))}}});await assert.rejects(bridge.interceptCast({item:f.item,entry:f.entry,options:{rank:3}},f.next));assert.equal(f.counts.native,0);
});
