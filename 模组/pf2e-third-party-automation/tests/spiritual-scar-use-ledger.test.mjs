import test from 'node:test';
import assert from 'node:assert/strict';
import {SPIRITUAL_SCAR_SOURCE} from '../scripts/spiritual-scar-native.mjs';
let api={};try{api=await import('../scripts/spiritual-scar-use-ledger.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const ID='pf2e-third-party-automation',PATH=`flags.${ID}.spiritualScarUse`,copy=v=>structuredClone(v);
const apply=(object,changes)=>{for(const[path,value]of Object.entries(changes)){const parts=path.split('.');let at=object;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=copy(value)}};
function fixture(){
 assert.equal(typeof api.createSpiritualScarUseLedger,'function');
 const gm={id:'gm',isGM:true,active:true},user={id:'owner',isGM:false,active:true},users=new Map([[gm.id,gm],[user.id,user]]);users.activeGM=gm;
 const actor={id:'actor',uuid:'Actor.actor',type:'character',items:new Map(),canAct:true,isDead:false,flags:{'pf2e-reaction':{state:true}},testUserPermission:u=>u===gm||u===user};
 const updates=[],item={id:'scar',uuid:actor.uuid+'.Item.scar',actor,type:'action',sourceId:SPIRITUAL_SCAR_SOURCE,flags:{unrelated:{keep:true}},system:{actionType:{value:'reaction'},frequency:{max:1,per:'day',value:1}},_source:{system:{frequency:{max:1,per:'day'}}},
  async update(changes,options={}){updates.push({changes:copy(changes),options:copy(options)});if(item.mode==='veto')return;if(item.mode==='noop')return item;apply(item,changes);if(item.mode==='lost')throw Error('reply lost');return item;}};
 actor.items.set(item.id,item);const actors=new Map([[actor.id,actor]]),messages=new Map(),docs=new Map([[actor.uuid,actor],[item.uuid,item]]),game={user:gm,users,actors,messages},clientGame={...game,user};let sequence=0;
 const options={fromUuid:async uuid=>docs.get(uuid),randomId:()=>`nonce-${++sequence}`},make=(g=game)=>api.createSpiritualScarUseLedger({game:g,...options}),ledger=make(),client=make(clientGame);
 const scope={actor,item,user,invocationId:'damage-1',fingerprint:'a'.repeat(64),privacy:{blind:true,whisper:['gm']}},current=()=>ledger.current(item);
 const claim=()=>ledger.claim(scope);
 async function pay(nonce,{observe=true}={}){
  await ledger.beginPayment({...scope,nonce});
  client.authorizePayment(item,nonce);const changes={'system.frequency.value':0},options={};assert.equal(client.preparePayment(item,changes,options,user.id),true);
  const frequencyReceipt={id:`receipt-${++sequence}`,itemUuid:item.uuid,userId:user.id,before:1,after:0,createdAt:100};options[ID]={...options[ID],frequencyReceipt};
  const result=await item.update(changes,options);
  if(observe&&result===item&&item.system.frequency.value===0){ledger.observePayment(item,changes,options,user.id);client.observePayment(item,changes,options,user.id)}
  return {changes,options,frequencyReceipt};
 }
 function card(nonce,frequencyReceipt){const id=`card-${++sequence}`,message={id,uuid:'ChatMessage.'+id,author:user,speaker:{actor:actor.id},blind:true,whisper:['gm'],rolls:[],flags:{pf2e:{origin:{actor:actor.uuid,uuid:item.uuid,type:'action'}},[ID]:{usageInput:{actualUse:true,frequencyReceiptId:frequencyReceipt.id},spiritualScarInput:{nonce,paymentNonce:current().paymentNonce}}}};messages.set(id,message);docs.set(message.uuid,message);return message;}
 async function ready(){const r=await claim(),p=await pay(r.nonce),message=card(r.nonce,p.frequencyReceipt),args={...scope,nonce:r.nonce,message,frequencyReceipt:p.frequencyReceipt};await ledger.bindUsage(args);return args;}
 return {gm,user,users,actor,item,updates,game,clientGame,docs,messages,make,ledger,client,scope,current,claim,pay,card,ready};
}
test('Scar claim binds one invocation and privacy without spending daily use or a reaction',async()=>{
 const f=fixture(),before=copy(f.actor.flags),r=await f.claim();assert.equal(r.status,'claimed');assert.deepEqual(r.privacy,f.scope.privacy);assert.equal(r.fingerprint,f.scope.fingerprint);assert.deepEqual(await f.claim(),r);assert.equal(f.updates.length,1);assert.equal(f.item.system.frequency.value,1);assert.equal(f.item._source.system.frequency.value,undefined);assert.deepEqual(f.actor.flags,before);assert.deepEqual(f.item.flags.unrelated,{keep:true});r.privacy.whisper.push('other');assert.deepEqual(f.current().privacy.whisper,['gm']);
});
test('concurrent source invocations and changed idempotency evidence cannot share one daily use',async()=>{
 const f=fixture(),results=await Promise.allSettled([f.claim(),f.make().claim({...f.scope,invocationId:'damage-2'})]);assert.equal(results.filter(r=>r.status==='fulfilled').length,1);assert.equal(f.updates.length,1);await assert.rejects(f.ledger.claim({...f.scope,privacy:{blind:false,whisper:[]}}));await assert.rejects(f.ledger.claim({...f.scope,fingerprint:'b'.repeat(64)}));
});
for(const[name,change]of[
 ['foreign owner',f=>f.scope.user={id:'outsider'}],['wrong source',f=>f.item.sourceId='wrong'],['not a reaction',f=>f.item.system.actionType.value='free'],['daily use exhausted',f=>f.item.system.frequency.value=0],['missing prepared frequency',f=>delete f.item.system.frequency.value],['deleted actor',f=>f.game.actors.delete(f.actor.id)],['deleted item',f=>f.actor.items.clear()],['unable to act',f=>f.actor.canAct=false],['inactive owner',f=>f.user.active=false],['malformed privacy',f=>f.scope.privacy.whisper=[{}]],['not elected GM',f=>f.game.user=f.user]
])test(`Scar claim rejects ${name} before writing`,async()=>{const f=fixture();change(f);await assert.rejects(f.claim());assert.equal(f.updates.length,0)});
test('unattempted cancellation spends nothing, and cancelled invocation cannot be resurrected',async()=>{
 const f=fixture(),r=await f.claim(),scope={...f.scope,nonce:r.nonce};await f.ledger.cancelClaim(scope);assert.equal(f.current().status,'cancelled');assert.equal(f.item.system.frequency.value,1);await assert.rejects(f.claim());const next=await f.ledger.claim({...f.scope,invocationId:'damage-2'});assert.notEqual(next.nonce,r.nonce);
});
test('original native 1-to-0 debit is the only payment; manual debit remains untouched',async()=>{
 const f=fixture(),changes={'system.frequency.value':0},options={};assert.equal(f.client.preparePayment(f.item,changes,options,f.user.id),undefined);assert.deepEqual(options,{});const r=await f.claim(),p=await f.pay(r.nonce);assert.equal(f.current().status,'paid');assert.equal(p.options[ID].spiritualScarPayment.nonce,r.nonce);assert.equal(p.changes[PATH].operations[r.nonce].paymentNonce,f.current().paymentNonce);assert.equal(f.item.system.frequency.value,0);assert.equal(f.actor.flags['pf2e-reaction'].state,true);await assert.rejects(f.ledger.cancelClaim({...f.scope,nonce:r.nonce}));
});
test('only the performing owner may authorize one debit and a changed claim cancels authorization',async()=>{
 const f=fixture(),r=await f.claim();await f.ledger.beginPayment({...f.scope,nonce:r.nonce});assert.throws(()=>f.ledger.authorizePayment(f.item,r.nonce));f.client.authorizePayment(f.item,r.nonce);assert.equal(f.client.preparePayment(f.item,{'system.frequency.value':0},{},f.user.id),true);assert.equal(f.client.preparePayment(f.item,{'system.frequency.value':0},{},f.user.id),false);
 const g=fixture(),q=await g.claim();await g.ledger.beginPayment({...g.scope,nonce:q.nonce});g.client.authorizePayment(g.item,q.nonce);g.users.activeGM={id:'othergm',isGM:true};assert.equal(g.client.preparePayment(g.item,{'system.frequency.value':0},{},g.user.id),false);
});
test('vetoed or no-op daily updates cannot authorize a displayed card',async()=>{
 for(const mode of ['veto','noop']){const f=fixture(),r=await f.claim();await f.ledger.beginPayment({...f.scope,nonce:r.nonce});f.client.authorizePayment(f.item,r.nonce);const changes={'system.frequency.value':0},options={};assert.equal(f.client.preparePayment(f.item,changes,options,f.user.id),true);const frequencyReceipt={id:'unused-receipt',itemUuid:f.item.uuid,userId:f.user.id,before:1,after:0,createdAt:100};options[ID]={...options[ID],frequencyReceipt};f.item.mode=mode;await f.item.update(changes,options);f.item.mode=null;assert.equal(f.current().status,'paying');assert.equal(f.ledger.observePayment(f.item,changes,options,f.user.id),null);const message=f.card(r.nonce,frequencyReceipt);await assert.rejects(f.ledger.bindUsage({...f.scope,nonce:r.nonce,message,frequencyReceipt}));assert.equal(f.item.system.frequency.value,1);}
});
test('the live original Use card binds exactly once to witnessed payment and matching privacy',async()=>{
 const f=fixture(),scope=await f.ready(),before=f.updates.length;assert.equal(f.current().status,'ready');assert.equal(f.current().messageUuid,scope.message.uuid);await f.ledger.bindUsage(scope);assert.equal(f.updates.length,before);await assert.rejects(f.ledger.bindUsage({...scope,message:f.card(scope.nonce,scope.frequencyReceipt)}));
});
for(const[name,change]of[
 ['unobserved debit',f=>f.ledger=f.make()],['display-only card',(_f,m)=>m.flags[ID].usageInput.actualUse=false],['wrong original item',(_f,m)=>m.flags.pf2e.origin.uuid='other'],['wrong author',(_f,m)=>m.author={id:'other'}],['public replacement',(_f,m)=>{m.blind=false;m.whisper=[]}],['wrong recipients',(_f,m)=>m.whisper=['owner']],['forged payment',(_f,m)=>m.flags[ID].spiritualScarInput.paymentNonce='forged'],['check card',(_f,m)=>m.rolls=[{}]],['foreign receipt',(_f,_m,p)=>p.frequencyReceipt={...p.frequencyReceipt,id:'forged'}]
])test(`Scar Use binding rejects ${name} without a second charge`,async()=>{const f=fixture(),r=await f.claim(),p=await f.pay(r.nonce),message=f.card(r.nonce,p.frequencyReceipt);change(f,message,p);const before=f.updates.length;await assert.rejects(f.ledger.bindUsage({...f.scope,nonce:r.nonce,message,frequencyReceipt:p.frequencyReceipt}));assert.equal(f.updates.length,before);assert.equal(f.item.system.frequency.value,0)});
test('confirmed payment can be consumed once; reloaded client flags and a recharge cannot replay it',async()=>{
 const f=fixture(),scope=await f.ready();assert.equal((await f.ledger.consume(scope)).status,'consumed');await assert.rejects(f.ledger.consume(scope));await assert.rejects(f.make().consume(scope));
 const g=fixture(),args=await g.ready();g.item.system.frequency.value=1;g.ledger.observePayment(g.item,{'system.frequency.value':1},{},g.user.id);g.item.system.frequency.value=0;await assert.rejects(g.ledger.consume(args));assert.equal(g.current().status,'ready');
});
test('uncertain attempted native use is terminal and is never silently retried or refunded',async()=>{
 for(const stage of ['claimed','paid','ready']){const f=fixture(),r=await f.claim(),scope={...f.scope,nonce:r.nonce};if(stage!=='claimed'){const p=await f.pay(r.nonce);if(stage==='ready')await f.ledger.bindUsage({...scope,message:f.card(r.nonce,p.frequencyReceipt),frequencyReceipt:p.frequencyReceipt});}await f.ledger.uncertain({...scope,reason:'native reply missing'});assert.equal(f.current().status,'uncertain');assert.equal(f.item.system.frequency.value,stage==='claimed'?1:0);await assert.rejects(f.ledger.consume(scope));await assert.rejects(f.ledger.cancelClaim(scope));await assert.rejects(f.claim());}
});
test('a lost claim-update reply is not permission and writes retain the recoverable claim',async()=>{
 const f=fixture();f.item.mode='lost';await assert.rejects(f.claim(),/reply lost/);assert.equal(f.current().status,'claimed');assert.equal(f.item.system.frequency.value,1);
});
test('an unstarted claim cannot authorize a native debit; beginning Use closes cancellation on every client',async()=>{
 const f=fixture(),r=await f.claim(),scope={...f.scope,nonce:r.nonce};assert.throws(()=>f.client.authorizePayment(f.item,r.nonce));
 assert.equal((await f.ledger.beginPayment(scope)).status,'paying');await assert.rejects(f.ledger.cancelClaim(scope));await assert.rejects(f.ledger.claim({...f.scope,invocationId:'damage-2'}));
 f.client.authorizePayment(f.item,r.nonce);await f.ledger.uncertain({...scope,reason:'Use update did not return'});assert.equal(f.client.preparePayment(f.item,{'system.frequency.value':0},{},f.user.id),false);
});
