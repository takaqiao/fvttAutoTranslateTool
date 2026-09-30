import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createEldamonElectricityProvider} from '../scripts/eldamon-electricity-provider.mjs';
import {createElectricityLedger,electricityState} from '../scripts/eldamon-electricity.mjs';
import {fixture} from './eldamon-electricity-fixture.mjs';
const ID='pf2e-third-party-automation',APPLY=`${ID}:electricity-apply:`;
function registered(f){
 const hooks=new Map(),routes=new Map(),provider=createEldamonElectricityProvider({game:f.game,fromUuid:async uuid=>f.docs.get(uuid)});
 const Hooks={on(name,fn){const list=hooks.get(name)??[];list.push(fn);hooks.set(name,list)}};
 provider.register({Hooks,socket:{register:(name,fn)=>routes.set(name,fn)}});
 const fire=(name,message)=>{for(const fn of hooks.get(name)??[])fn(message,{},f.gm.id)};
 const finish=(applied,user=f.gm)=>routes.get('electricity:finishDamage').call({socketdata:{userId:user.id}},{actorUuid:applied.payload.actorUuid,nonce:applied.payload.nonce,receiptUuid:applied.receipt.uuid});
 return {provider,fire,finish};
}
test('receipt settlement and startup recovery reuse one nonce index without iterating all chat history',async()=>{
 const f=fixture(),first=await f.damage(),second=await f.damage();for(let i=0;i<500;i++)f.game.messages.set('ordinary-'+i,{id:'ordinary-'+i,flags:{pf2e:{context:{type:'skill-check'}}}});
 const values=f.game.messages.values.bind(f.game.messages);let iterations=0;f.game.messages.values=()=>{iterations++;return values()};
 const live=registered(f),seedIterations=iterations;f.game.messages.values=()=>{throw Error('hot receipt path iterated chat history')};
 assert.equal((await live.finish(first)).ok,true);await live.provider.maintain(f.target);
 assert.equal(electricityState(f.target).damage[second.payload.nonce].status,'confirmed');assert.equal(seedIterations,1);assert.equal(iterations,1);
});
test('a duplicate receipt added after seed blocks settlement until its exact nonce entry is deleted',async()=>{
 const f=fixture(),applied=await f.damage(),live=registered(f),copy={...applied.receipt,id:'duplicate',uuid:'ChatMessage.duplicate',flags:structuredClone(applied.receipt.flags)};
 f.game.messages.set(copy.id,copy);f.docs.set(copy.uuid,copy);live.fire('createChatMessage',copy);f.game.messages.values=()=>{throw Error('duplicate lookup scanned chat')};
 assert.equal((await live.finish(applied)).ok,false);assert.equal(electricityState(f.target).damage[applied.payload.nonce].status,'pending');
 f.game.messages.delete(copy.id);f.docs.delete(copy.uuid);live.fire('deleteChatMessage',copy);assert.equal((await live.finish(applied)).ok,true);
});
test('updated duplicate nonce moves its index entry and cannot leave a stale duplicate',async()=>{
 const f=fixture(),applied=await f.damage(),live=registered(f),copy={...applied.receipt,id:'updated',uuid:'ChatMessage.updated',flags:structuredClone(applied.receipt.flags)};
 f.game.messages.set(copy.id,copy);f.docs.set(copy.uuid,copy);live.fire('createChatMessage',copy);assert.equal((await live.finish(applied)).ok,false);
 copy.flags.pf2e.context.options=[APPLY+'differentnonce'];copy.flags[ID].electricityApplied.nonce='differentnonce';live.fire('updateChatMessage',copy);
 f.game.messages.values=()=>{throw Error('update lookup scanned chat')};assert.equal((await live.finish(applied)).ok,true);
});
for(const [name,mutate]of [
 ['reverted receipt',m=>{m.flags.pf2e.appliedDamage.isReverted=true}],['wrong author',m=>{m.author={id:'foreign'}}],['wrong actor',m=>{m.speaker.actor='foreign'}],['wrong token',m=>{m.speaker.token='foreign'}],['wrong origin',m=>{m.flags.pf2e.origin.uuid='Actor.foreign.Item.source'}],['multiple APPLY tags',m=>{m.flags.pf2e.context.options.push(APPLY+'anothernonce')}],
])test(`indexed settlement still rejects ${name} after an update hook`,async()=>{
 const f=fixture(),applied=await f.damage(),live=registered(f);mutate(applied.receipt);live.fire('updateChatMessage',applied.receipt);f.game.messages.values=()=>{throw Error('invalid receipt lookup scanned chat')};
 assert.equal((await live.finish(applied)).ok,false);assert.equal(electricityState(f.target).damage[applied.payload.nonce].status,'pending');
});
test('deleted receipt and a stale replaced document identity both fail closed',async()=>{
 for(const replace of [false,true]){
  const f=fixture(),applied=await f.damage(),live=registered(f),original=applied.receipt;
  if(replace){const current={...original,flags:structuredClone(original.flags)};f.game.messages.set(current.id,current);live.fire('updateChatMessage',current);}
  else{f.game.messages.delete(original.id);f.docs.delete(original.uuid);live.fire('deleteChatMessage',original);}
  f.game.messages.values=()=>{throw Error('current-document lookup scanned chat')};assert.equal((await live.finish(applied)).ok,false);assert.equal(electricityState(f.target).damage[applied.payload.nonce].status,'pending');
 }
});
test('the pure ledger refuses an absent projection or one whose sole candidate is not the original receipt',async()=>{
 for(const missing of [true,false]){
  const f=fixture(),applied=await f.damage(),copy={...applied.receipt,id:'other-indexed',uuid:'ChatMessage.other-indexed',flags:structuredClone(applied.receipt.flags)};
  f.game.messages.set(copy.id,copy);f.docs.set(copy.uuid,copy);
  const ledger=createElectricityLedger({game:f.game,fromUuid:async uuid=>f.docs.get(uuid),...(!missing?{receiptMessages:()=>[copy]}:{})});
  f.game.messages.values=()=>{throw Error('ledger fallback scanned history')};await assert.rejects(ledger.finishDamage({actorUuid:applied.payload.actorUuid,nonce:applied.payload.nonce,receiptUuid:applied.receipt.uuid},f.gm),/唯一|回执/);
  assert.equal(electricityState(f.target).damage[applied.payload.nonce].status,'pending');
 }
});
test('a fresh provider seeds existing pending receipts once and a later active-GM handoff does not rescan history',async()=>{
 const f=fixture(),applied=await f.damage(),old=registered(f);old.fire('createChatMessage',applied.receipt);
 const nativeValues=f.game.messages.values.bind(f.game.messages);let scans=0;f.game.messages.values=()=>{scans++;return nativeValues()};const recovered=registered(f);assert.equal(scans,1);
 const nextGM={id:'nextGM',isGM:true};f.game.users.set(nextGM.id,nextGM);f.game.users.activeGM=nextGM;f.game.user=nextGM;
 f.game.messages.values=()=>{throw Error('GM handoff rescanned chat')};assert.equal((await recovered.finish(applied)).ok,true);assert.equal(electricityState(f.target).damage[applied.payload.nonce].status,'confirmed');
});
