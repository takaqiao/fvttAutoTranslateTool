import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHalflingLuckLedger,HALFLING_LUCK_SOURCE} from '../scripts/halfling-luck-ledger.mjs';
const ID='pf2e-third-party-automation',PATH=`flags.${ID}.halflingLuck`;
const copy=value=>structuredClone(value);
function apply(target,changes){for(const [key,value]of Object.entries(changes)){const parts=key.split('.');let at=target;for(const part of parts.slice(0,-1))at=at[part]??={};at[parts.at(-1)]=copy(value)}}
function fixture(){
 const gm={id:'gm',isGM:true},user={id:'owner',isGM:false},other={id:'other'},users=new Map([[gm.id,gm],[user.id,user],[other.id,other]]);users.activeGM=gm;
 const actor={id:'actor',uuid:'Actor.actor',type:'character',canAct:true,isDead:false,flags:{'pf2e-reaction':{state:false,quickShieldBlock:2}},items:new Map(),testUserPermission:u=>u===gm||u===user};
 const updates=[],item={id:'luck',uuid:'Actor.actor.Item.luck',type:'feat',sourceId:HALFLING_LUCK_SOURCE,actor,flags:{unrelated:{keep:true}},_source:{system:{frequency:{max:1,per:'day'}}},system:{actionType:{value:'free'},frequency:{value:1,max:1,per:'day'}},
  async update(changes,options={}){updates.push({changes:copy(changes),options:copy(options)});if(item.writeMode==='veto')return undefined;if(item.writeMode==='noop')return item;apply(item,changes);if(item.writeMode==='lost')throw Error('reply lost');return item;}};
 actor.items.set(item.id,item);const messages=new Map(),actors=new Map([[actor.id,actor]]),docs=new Map([[actor.uuid,actor],[item.uuid,item]]);
 const game={user:gm,users,actors,messages,time:{worldTime:100}},clientGame={...game,user};let sequence=0;
 const options={fromUuid:async uuid=>docs.get(uuid),randomId:()=>`nonce-${++sequence}`};
 const make=()=>createHalflingLuckLedger({game,...options}),ledger=make(),client=createHalflingLuckLedger({game:clientGame,...options});
 const scope={actor,item,user,invocationId:'check-1',fingerprint:'original-native-check'};
 const record=nonce=>item.flags[ID]?.halflingLuck?.operations[nonce],current=()=>record(item.flags[ID]?.halflingLuck?.currentNonce);
 async function claim(extra={}){return ledger.claim({...scope,...extra})}
 async function pay(nonce,{observe=true}={}){
  client.authorizePayment(item,nonce,user);const changes={'system.frequency.value':0},options={};assert.equal(client.preparePayment(item,changes,options,user.id),true);
  const frequencyReceipt={id:`receipt-${++sequence}`,itemUuid:item.uuid,userId:user.id,before:1,after:0,createdAt:100};options[ID]={...options[ID],frequencyReceipt};
  const result=await item.update(changes,options);
  if(observe&&result===item&&item.system.frequency.value===0){ledger.observePayment(item,changes,options,user.id);client.observePayment(item,changes,options,user.id)}
  return {frequencyReceipt,changes,options};
 }
 function card(nonce,frequencyReceipt,id=`card-${++sequence}`){const r=record(nonce),message={id,uuid:`ChatMessage.${id}`,author:user,speaker:{actor:actor.id},flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid,type:'feat'}},[ID]:{usageInput:{actualUse:true,frequencyReceiptId:frequencyReceipt.id},halflingLuckInput:{nonce,paymentNonce:r.paymentNonce}}},rolls:[]};messages.set(id,message);docs.set(message.uuid,message);return message;}
 async function ready(){const r=await claim(),p=await pay(r.nonce),message=card(r.nonce,p.frequencyReceipt);await ledger.bindUsage({...scope,message,frequencyReceipt:p.frequencyReceipt});return {nonce:r.nonce,...p,message,...scope}}
 return {gm,user,other,users,game,clientGame,actor,item,updates,docs,messages,ledger,client,scope,record,current,claim,pay,card,ready,make};
}

test('claim validates prepared exact source, is idempotent for one invocation, and never spends frequency or reaction',async()=>{
 const f=fixture(),reaction=copy(f.actor.flags),r=await f.claim();assert.equal(r.status,'claimed');assert.equal(r.invocationId,'check-1');assert.equal(r.fingerprint,'original-native-check');assert.equal(f.item.system.frequency.value,1);assert.equal(f.item._source.system.frequency.value,undefined);assert.equal(f.item.flags[ID].halflingLuck.currentNonce,r.nonce);assert.deepEqual(await f.claim(),r);assert.equal(f.updates.length,1);assert.deepEqual(f.actor.flags,reaction);assert.deepEqual(f.item.flags.unrelated,{keep:true});
});

test('same-item concurrent claims serialize even across ledger instances',async()=>{
 const f=fixture(),second=f.make(),results=await Promise.allSettled([f.claim(),second.claim({...f.scope,invocationId:'check-2'})]);
 assert.equal(results.filter(r=>r.status==='fulfilled').length,1);assert.equal(results.filter(r=>r.status==='rejected').length,1);assert.equal(Object.keys(f.item.flags[ID].halflingLuck.operations).length,1);assert.equal(f.item.system.frequency.value,1);
});

for(const [name,change]of [
 ['inactive GM',f=>{f.game.user=f.user}],['foreign owner',f=>{f.scope.user=f.other}],['replaced actor',f=>{f.game.actors.set('actor',{...f.actor})}],['synthetic actor',f=>{f.actor.isToken=true}],['replaced item',f=>{f.actor.items.set('luck',{...f.item})}],['wrong source',f=>{f.item.sourceId='Other.source'}],['wrong type',f=>{f.item.type='action'}],['reaction cost',f=>{f.item.system.actionType.value='reaction'}],['wrong frequency',f=>{f.item.system.frequency.max=2}],['empty frequency',f=>{f.item.system.frequency.value=0}],['missing prepared value',f=>{delete f.item.system.frequency.value}],
])test(`claim rejects ${name} without writes`,async()=>{const f=fixture();change(f);await assert.rejects(f.claim());assert.equal(f.updates.length,0)});

test('cancellation only terminates the current unpaid claimed operation',async()=>{
 const f=fixture(),r=await f.claim(),ctx={...f.scope,nonce:r.nonce};await f.ledger.cancelClaim(ctx);assert.equal(f.record(r.nonce).status,'cancelled');assert.equal(f.item.system.frequency.value,1);
 await assert.rejects(f.ledger.cancelClaim(ctx));const next=await f.claim({invocationId:'check-2'});await f.pay(next.nonce);await assert.rejects(f.ledger.cancelClaim({...ctx,nonce:next.nonce}));assert.equal(f.item.system.frequency.value,0);
});

test('only authorized original 1-to-0 update gets atomic payment proof; ordinary manual Use stays unchanged',async()=>{
 const f=fixture(),manual={'system.frequency.value':0},opts={};assert.equal(f.client.preparePayment(f.item,manual,opts,f.user.id),undefined);assert.deepEqual(manual,{'system.frequency.value':0});assert.deepEqual(opts,{});
 const r=await f.claim(),before=f.updates.length,p=await f.pay(r.nonce);assert.equal(f.updates.length,before+1);assert.equal(f.record(r.nonce).status,'paid');assert.equal(f.item.system.frequency.value,0);assert.equal(p.options[ID].halflingLuckPayment.nonce,r.nonce);assert.equal(p.options[ID].halflingLuckPayment.paymentNonce,f.record(r.nonce).paymentNonce);
 assert.ok(Object.keys(p.changes).some(k=>k.startsWith(PATH)));assert.deepEqual(f.actor.flags,{'pf2e-reaction':{state:false,quickShieldBlock:2}});
});

test('automatic payment authorization cannot survive owner/source/current-claim changes or pay twice',async()=>{
 for(const change of [f=>{f.scope.user=f.other;f.clientGame.user=f.other},f=>{f.item.sourceId='wrong'},f=>{f.current().status='cancelled'},f=>{f.item.flags[ID].halflingLuck.currentNonce='newer'}]){
  const f=fixture(),r=await f.claim();f.client.authorizePayment(f.item,r.nonce,f.user);change(f);assert.equal(f.client.preparePayment(f.item,{'system.frequency.value':0},{},f.user.id),false);assert.equal(f.item.system.frequency.value,1);
 }
 const f=fixture(),r=await f.claim();f.client.authorizePayment(f.item,r.nonce,f.user);assert.equal(f.client.preparePayment(f.item,{'system.frequency.value':0},{},f.user.id),true);assert.equal(f.client.preparePayment(f.item,{'system.frequency.value':0},{},f.user.id),false);
});

test('payment veto/no-op supplies no observation or usable paid card',async()=>{
 for(const mode of ['veto','noop']){const f=fixture(),r=await f.claim();f.item.writeMode=mode;const p=await f.pay(r.nonce);assert.equal(f.item.system.frequency.value,1);assert.equal(f.record(r.nonce).status,'claimed');assert.equal(f.ledger.observePayment(f.item,p.changes,p.options,f.user.id),null);f.item.writeMode=null;const m=f.card(r.nonce,p.frequencyReceipt);await assert.rejects(f.ledger.bindUsage({...f.scope,message:m,frequencyReceipt:p.frequencyReceipt}));}
});

test('bind requires the actual observed update and live original Use card, then same-card delivery is idempotent',async()=>{
 const f=fixture(),r=await f.claim(),p=await f.pay(r.nonce),message=f.card(r.nonce,p.frequencyReceipt),ctx={...f.scope,message,frequencyReceipt:p.frequencyReceipt};
 const ready=await f.ledger.bindUsage(ctx);assert.equal(ready.status,'ready');assert.equal(ready.messageUuid,message.uuid);const writes=f.updates.length;assert.deepEqual(await f.ledger.bindUsage(ctx),ready);assert.equal(f.updates.length,writes);
 await assert.rejects(f.ledger.bindUsage({...ctx,message:f.card(r.nonce,p.frequencyReceipt)}));
 const reloaded=f.make();await assert.rejects(reloaded.bindUsage(ctx));await assert.rejects(reloaded.startRolling({...f.scope,nonce:r.nonce}));
});

for(const [name,change]of [
 ['missing observation',f=>{f.ledger=f.make()}],['wrong author',(_f,m)=>{m.author={id:'other'}}],['copied document',(f,m)=>{f.messages.set(m.id,{...m})}],['display card',(_f,m)=>{m.flags[ID].usageInput.actualUse=false}],['wrong source',(_f,m)=>{m.flags.pf2e.origin.uuid='Actor.actor.Item.other'}],['wrong payment',(_f,m)=>{m.flags[ID].halflingLuckInput.paymentNonce='forged'}],['wrong receipt',(_f,_m,p)=>{p.frequencyReceipt={...p.frequencyReceipt,id:'forged'}}],['a roll card',(_f,m)=>{m.rolls=[{}]}],
])test(`bind rejects ${name} without continuing or charging`,async()=>{const f=fixture(),r=await f.claim(),p=await f.pay(r.nonce),m=f.card(r.nonce,p.frequencyReceipt);change(f,m,p);const writes=f.updates.length;await assert.rejects(f.ledger.bindUsage({...f.scope,message:m,frequencyReceipt:p.frequencyReceipt}));assert.equal(f.record(r.nonce).status,'paid');assert.equal(f.updates.length,writes);assert.equal(f.item.system.frequency.value,0)});

test('rolling and delivery are one-shot persisted permissions; identical results/finished callbacks are idempotent',async()=>{
 const f=fixture(),ctx=await f.ready(),rollJSON={class:'CheckRoll',evaluated:true,total:8,options:{isReroll:true,degreeOfSuccess:1}},result={...ctx,rollJSON,outcome:'failure'};
 assert.equal((await f.ledger.startRolling(ctx)).status,'rolling');await assert.rejects(f.ledger.startRolling(ctx));assert.equal((await f.ledger.recordResult(result)).status,'result-ready');assert.deepEqual(f.record(ctx.nonce).rollJSON,rollJSON);assert.equal((await f.ledger.recordResult(result)).status,'result-ready');
 await assert.rejects(f.ledger.recordResult({...result,outcome:'success'}));assert.equal((await f.ledger.beginDelivery(ctx)).status,'delivering');await assert.rejects(f.ledger.beginDelivery(ctx));assert.equal((await f.ledger.finishDelivery(ctx)).status,'callback-returned');assert.equal((await f.ledger.finishDelivery(ctx)).status,'callback-returned');assert.equal(f.item.system.frequency.value,0);
});

test('uncertain states never authorize another roll or callback and never refund',async()=>{
 const f=fixture(),ctx=await f.ready();await f.ledger.startRolling(ctx);await f.ledger.uncertain({...ctx,reason:'native response lost'});assert.equal(f.record(ctx.nonce).status,'uncertain');assert.equal(f.record(ctx.nonce).reason,'native response lost');await assert.rejects(f.ledger.startRolling(ctx));await assert.rejects(f.ledger.beginDelivery(ctx));await assert.rejects(f.ledger.cancelClaim(ctx));assert.equal(f.item.system.frequency.value,0);
});

test('daily recharge allows a fresh claim but cannot resume old paid/rolling records or use an old card',async()=>{
 for(const rolling of [false,true]){
  const f=fixture(),ctx=await f.ready();if(rolling)await f.ledger.startRolling(ctx);f.item.system.frequency.value=1;
  const fresh=await f.claim({invocationId:'check-new-day'});assert.notEqual(fresh.nonce,ctx.nonce);assert.equal(f.record(ctx.nonce).status,rolling?'rolling':'ready');
  await assert.rejects(f.ledger.bindUsage(ctx));await assert.rejects(f.ledger.startRolling(ctx));assert.throws(()=>f.client.authorizePayment(f.item,ctx.nonce,f.user));await assert.rejects(f.claim({invocationId:ctx.invocationId}));
  const payment=await f.pay(fresh.nonce),newCard=f.card(fresh.nonce,payment.frequencyReceipt);await f.ledger.bindUsage({...f.scope,message:newCard,frequencyReceipt:payment.frequencyReceipt});await assert.rejects(f.ledger.bindUsage({...ctx,frequencyReceipt:payment.frequencyReceipt}));assert.equal(f.item.flags[ID].halflingLuck.currentNonce,fresh.nonce);
 }
});

test('each GM write rejects veto/no-op and does not treat a lost reply as permission to repeat',async()=>{
 for(const mode of ['veto','noop']){
  const f=fixture();f.item.writeMode=mode;await assert.rejects(f.claim());assert.equal(f.item.flags[ID]?.halflingLuck,undefined);
  f.item.writeMode=null;const ctx=await f.ready();f.item.writeMode=mode;await assert.rejects(f.ledger.startRolling(ctx));assert.equal(f.record(ctx.nonce).status,'ready');
 }
 const f=fixture(),ctx=await f.ready();f.item.writeMode='lost';await assert.rejects(f.ledger.startRolling(ctx),/reply lost/);f.item.writeMode=null;assert.equal(f.record(ctx.nonce).status,'rolling');await assert.rejects(f.ledger.startRolling(ctx));
});

test('payment witness rejects mismatched receipt, user and durable nonce',async()=>{
 for(const change of [p=>{p.options[ID].frequencyReceipt.after=1},p=>{p.options[ID].frequencyReceipt.userId='other'},p=>{p.options[ID].halflingLuckPayment.paymentNonce='wrong'}]){
  const f=fixture(),r=await f.claim(),p=await f.pay(r.nonce,{observe:false});change(p);assert.equal(f.ledger.observePayment(f.item,p.changes,p.options,f.user.id),null);await assert.rejects(f.ledger.bindUsage({...f.scope,message:f.card(r.nonce,p.frequencyReceipt),frequencyReceipt:p.frequencyReceipt}));
 }
});

test('current returns a safe clone and clearing authorization only removes the matching local attempt',async()=>{
 const f=fixture(),r=await f.claim(),snapshot=f.ledger.current(f.item);snapshot.status='paid';assert.equal(f.current().status,'claimed');
 f.client.authorizePayment(f.item,r.nonce,f.user);assert.equal(f.client.clearAuthorization(f.item,'other-nonce'),false);
 assert.equal(f.client.clearAuthorization(f.item,r.nonce),true);const writes=f.updates.length;
 assert.equal(f.client.preparePayment(f.item,{'system.frequency.value':0},{},f.user.id),undefined);assert.equal(f.updates.length,writes);assert.equal(f.current().status,'claimed');
});

test('claim-time GM identity cannot silently migrate after payment',async()=>{
 const f=fixture(),ctx=await f.ready(),nextGM={id:'next-gm',isGM:true};f.users.set(nextGM.id,nextGM);f.users.activeGM=nextGM;f.game.user=nextGM;
 const writes=f.updates.length;await assert.rejects(f.ledger.bindUsage(ctx));await assert.rejects(f.ledger.startRolling(ctx));assert.equal(f.updates.length,writes);assert.equal(f.current().gmId,'gm');
});

test('a native recharge or unrelated later debit invalidates old in-memory payment evidence',async()=>{
 for(const reset of [true,false]){
  const f=fixture(),ctx=await f.ready();if(reset){f.item.system.frequency.value=1;f.ledger.observePayment(f.item,{'system.frequency.value':1},{},f.gm.id);}
  f.item.system.frequency.value=0;f.ledger.observePayment(f.item,{'system.frequency.value':0},{},f.user.id);
  await assert.rejects(f.ledger.startRolling(ctx));assert.equal(f.current().status,'ready');
 }
});

test('nested native update data preserves the same atomic payment and observation contract',async()=>{
 const f=fixture(),r=await f.claim(),changes={system:{frequency:{value:0}}},options={};f.client.authorizePayment(f.item,r.nonce);
 assert.equal(f.client.preparePayment(f.item,changes,options,f.user.id),true);const receipt={id:'nested-receipt',itemUuid:f.item.uuid,userId:f.user.id,before:1,after:0,createdAt:100};options[ID].frequencyReceipt=receipt;
 // Foundry expands flattened flag fields before broadcasting updateItem.
 f.item.system.frequency.value=0;apply(f.item,{[PATH]:changes[PATH]});const expanded={system:changes.system,flags:{[ID]:{halflingLuck:changes[PATH]}}};
 assert.ok(f.ledger.observePayment(f.item,expanded,options,f.user.id));assert.equal((await f.ledger.bindUsage({...f.scope,message:f.card(r.nonce,receipt),frequencyReceipt:receipt})).status,'ready');
});

test('Foundry updateItem diff can omit unchanged currentNonce and immutable claim fields',async()=>{
 const f=fixture(),r=await f.claim(),p=await f.pay(r.nonce,{observe:false}),paymentNonce=f.record(r.nonce).paymentNonce;
 const nativeDiff={system:{frequency:{value:0}},flags:{[ID]:{halflingLuck:{operations:{[r.nonce]:{status:'paid',paymentNonce}}}}}};
 assert.ok(f.ledger.observePayment(f.item,nativeDiff,p.options,f.user.id));
 assert.equal((await f.ledger.bindUsage({...f.scope,message:f.card(r.nonce,p.frequencyReceipt),frequencyReceipt:p.frequencyReceipt})).status,'ready');
});

test('every persisted stage rejects veto and no-op rather than granting the next permission',async()=>{
 for(const mode of ['veto','noop'])for(const stage of ['bindUsage','recordResult','beginDelivery','finishDelivery','uncertain','cancelClaim']){
  const f=fixture();let ctx;
  if(stage==='cancelClaim'){const r=await f.claim();ctx={...f.scope,nonce:r.nonce};}
  else if(stage==='bindUsage'){const r=await f.claim(),p=await f.pay(r.nonce);ctx={...f.scope,nonce:r.nonce,...p,message:f.card(r.nonce,p.frequencyReceipt)};}
  else{ctx=await f.ready();await f.ledger.startRolling(ctx);if(['beginDelivery','finishDelivery'].includes(stage))await f.ledger.recordResult({...ctx,rollJSON:{total:10},outcome:'failure'});if(stage==='finishDelivery')await f.ledger.beginDelivery(ctx);}
  const before=copy(f.current());f.item.writeMode=mode;await assert.rejects(f.ledger[stage]({...ctx,rollJSON:{total:10},outcome:'failure',reason:'lost native result'}));assert.deepEqual(f.current(),before);
 }
});

test('an intervening frequency reset while a permission write is pending cannot grant rolling',async()=>{
 for(const laterManualDebit of [false,true]){
  const f=fixture(),ctx=await f.ready(),native=f.item.update;f.item.update=async function(...args){const result=await native(...args);f.item.system.frequency.value=1;f.ledger.observePayment(f.item,{'system.frequency.value':1},{},f.gm.id);if(laterManualDebit){f.item.system.frequency.value=0;f.ledger.observePayment(f.item,{'system.frequency.value':0},{},f.user.id);}return result;};
  await assert.rejects(f.ledger.startRolling(ctx));assert.equal(f.current().status,'rolling');
 }
});

test('local payment authorization does not veto unrelated native item updates',async()=>{
 const f=fixture(),r=await f.claim();f.client.authorizePayment(f.item,r.nonce,f.user);
 for(const changes of [{name:'Updated name'},{flags:{other:{value:1}}},{system:{description:{value:'New description'}}}]){
  const before=copy(changes),options={};assert.equal(f.client.preparePayment(f.item,changes,options,f.user.id),undefined);assert.deepEqual(changes,before);assert.deepEqual(options,{});
 }
 assert.equal(f.client.preparePayment(f.item,{'system.frequency.value':0},{},f.user.id),true);
 assert.equal(f.client.preparePayment(f.item,{name:'After preparation'},{},f.user.id),undefined);
});

for(const [name,disable]of [['cannot act',a=>{a.canAct=false}],['unknown ability to act',a=>{delete a.canAct}],['dead',a=>{a.isDead=true}]]){
 test(`${name} cannot claim, authorize, or prepare payment`,async()=>{
  const f=fixture();disable(f.actor);await assert.rejects(f.claim());assert.equal(f.updates.length,0);
  const g=fixture(),r=await g.claim();disable(g.actor);assert.throws(()=>g.client.authorizePayment(g.item,r.nonce,g.user));assert.equal(g.item.system.frequency.value,1);
  const h=fixture(),q=await h.claim();h.client.authorizePayment(h.item,q.nonce,h.user);disable(h.actor);assert.equal(h.client.preparePayment(h.item,{'system.frequency.value':0},{},h.user.id),false);assert.equal(h.item.system.frequency.value,1);
 });
}

test('ability to act is an admission gate, not a veto on settling an already observed payment',async()=>{
 const f=fixture(),r=await f.claim(),p=await f.pay(r.nonce),message=f.card(r.nonce,p.frequencyReceipt),ctx={...f.scope,nonce:r.nonce,message,frequencyReceipt:p.frequencyReceipt};
 f.actor.canAct=false;f.actor.isDead=true;
 assert.equal((await f.ledger.bindUsage(ctx)).status,'ready');await f.ledger.startRolling(ctx);await f.ledger.recordResult({...ctx,rollJSON:{total:8},outcome:'failure'});await f.ledger.beginDelivery(ctx);assert.equal((await f.ledger.finishDelivery(ctx)).status,'callback-returned');assert.equal(f.item.system.frequency.value,0);
});
