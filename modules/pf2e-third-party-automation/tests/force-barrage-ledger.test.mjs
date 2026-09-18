import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createForceBarrageLedger} from '../scripts/force-barrage-ledger.mjs';
import {SerialActions} from '../scripts/runtime.mjs';
const ID='pf2e-third-party-automation',SOURCE='Compendium.pf2e.spells-srd.Item.gKKqvLohtrSJj3BM',PATH=`flags.${ID}.forceBarrage`,copy=v=>structuredClone(v);
function fixture(){
 const gm={id:'gm',isGM:true,active:true},user={id:'player',active:true},users=new Map([[gm.id,gm],[user.id,user]]);users.activeGM=gm;
 const actor={id:'caster',uuid:'Actor.caster',type:'character',canAct:true,isDead:false,items:new Map(),flags:{[ID]:{keep:'unchanged'},'pf2e-reaction':{state:false}},testUserPermission:u=>u===gm||u===user};
 const entry={id:'entry',uuid:'Actor.caster.Item.entry',type:'spellcastingEntry',actor,flags:{},system:{prepared:{value:'spontaneous'},tradition:{value:'occult'},slots:{slot1:{value:3,max:3},slot2:{value:3,max:3},slot3:{value:2,max:2}}}};
 const item={id:'spell',uuid:'Actor.caster.Item.spell',type:'spell',actor,sourceId:SOURCE,system:{level:{value:1},location:{value:entry.id,signature:true},traits:{value:['concentrate','manipulate']}}};
 actor.items.set(entry.id,entry);actor.items.set(item.id,item);
 const scene={id:'scene',uuid:'Scene.scene',tokens:new Map()};
 const token={id:'source',uuid:'Scene.scene.Token.source',documentName:'Token',parent:scene,actor,actorLink:true};
 const targets=['one','two','zero'].map(id=>({id,uuid:`Scene.scene.Token.${id}`,documentName:'Token',parent:scene,actor:{uuid:`Actor.${id}`,type:'npc'}}));
 for(const t of [token,...targets])scene.tokens.set(t.id,t);
 const docs=new Map([actor,item,entry,scene,token,...targets].map(d=>[d.uuid,d])),messages=new Map();
 const game={world:{id:'ujx5r8oipw7ercdr'},system:{id:'pf2e',version:'8.5.1'},user:gm,users,actors:new Map([[actor.id,actor]]),scenes:new Map([[scene.id,scene]]),messages};
 const writes=[],queue=new SerialActions();let seq=0;
 actor.update=async changes=>{writes.push(copy(changes));assert.deepEqual(Object.keys(changes),[PATH]);if(actor.mode==='veto')return undefined;if(actor.mode==='noop')return actor;actor.flags[ID].forceBarrage=copy(changes[PATH]);if(actor.mode==='tamper')actor.flags[ID].forceBarrage.currentByItem={};if(actor.mode==='lost')throw Error('reply lost');return actor.mode==='wrong'?{...actor}:actor;};
 const withActorResourceLock=(a,fn)=>queue.run(a.uuid,fn),make=()=>createForceBarrageLedger({game,fromUuid:async uuid=>docs.get(uuid),withActorResourceLock,randomId:()=>`bridge-${++seq}`}),ledger=make();
 const allocation={rank:3,actions:3,sourceTokenUuid:token.uuid,targets:[{targetUuid:targets[0].uuid,count:4},{targetUuid:targets[1].uuid,count:2},{targetUuid:targets[2].uuid,count:0}]};
 const scope={actor,item,entry,user,invocationId:'invocation-one',fingerprint:'a'.repeat(64),allocation};
 const claim=extra=>ledger.claim({...scope,...extra});
 const current=()=>actor.flags[ID].forceBarrage?.operations[actor.flags[ID].forceBarrage.currentByItem[item.id]];
 async function casting(){const record=await claim();const ctx={...scope,nonce:record.nonce};await ledger.startCast(ctx);return ctx;}
 function outcome(ctx){
  const castNonce='native-'+ctx.nonce,input={actorUuid:actor.uuid,itemUuid:item.uuid,entryUuid:entry.uuid,sourceId:SOURCE,rank:3,slotId:null,focusPoints:0,overlayIds:[]};
  const message={id:'original-'+ctx.nonce,uuid:'ChatMessage.original-'+ctx.nonce,documentName:'ChatMessage',author:user,speaker:{actor:actor.id,scene:scene.id,token:token.id},rolls:[],blind:false,whisper:[],flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid,castRank:3}},[ID]:{nativeCast:{id:castNonce,actorUuid:actor.uuid,itemUuid:item.uuid,userId:user.id},nativeCastInput:copy(input)}}};
  const receipt={...input,id:castNonce,state:'used',messageId:message.id,userId:user.id,invocation:{kind:'force-barrage',data:{bridgeNonce:ctx.nonce,fingerprint:scope.fingerprint},gmId:gm.id},nativeCastScope:{castNonce,tokenUuid:token.uuid},slotCommit:{castNonce,itemUuid:item.uuid,entryUuid:entry.uuid,rank:3,before:2,after:1,cost:1,userId:user.id,gmId:gm.id}};
  actor.flags[ID].nativeCasts??=[];actor.flags[ID].nativeCasts.push(receipt);entry.system.slots.slot3.value=1;messages.set(message.id,message);docs.set(message.uuid,message);
  return {status:'completed',castNonce,input:copy(input),receipt:copy(receipt),message,nativeResult:undefined};
 }
 async function paid(){const ctx=await casting(),result=outcome(ctx);await ledger.bindCast({...ctx,outcome:result});return {...ctx,outcome:result};}
 return {game,gm,user,users,actor,item,entry,token,targets,scene,docs,messages,writes,scope,allocation,ledger,make,claim,current,casting,outcome,paid,withActorResourceLock};
}

test('claim fixes one allocation, returns an isolated snapshot and never spends resources',async()=>{
 const f=fixture(),slots=copy(f.entry.system.slots),r=await f.claim();assert.equal(r.status,'claimed');assert.equal(r.nonce,'bridge-1');assert.equal(r.fingerprint,'a'.repeat(64));assert.deepEqual(r.allocation,f.allocation);assert.deepEqual(await f.claim(),r);assert.equal(f.writes.length,1);
 r.allocation.targets[0].count=90;assert.equal(f.current().allocation.targets[0].count,4);assert.deepEqual(f.entry.system.slots,slots);assert.equal(f.actor.flags[ID].keep,'unchanged');assert.deepEqual(f.actor.flags['pf2e-reaction'],{state:false});
});
test('two same-item claims across ledger instances serialize and only one obtains permission',async()=>{
 const f=fixture(),second=f.make(),r=await Promise.allSettled([f.claim(),second.claim({...f.scope,invocationId:'invocation-two'})]);assert.equal(r.filter(v=>v.status==='fulfilled').length,1);assert.equal(f.writes.length,1);
});
for(const [name,change]of [
 ['wrong source',f=>{f.item.sourceId='wrong'}],['wrong rank',f=>{f.allocation.rank=4}],['wrong actions',f=>{f.allocation.actions=0}],['fractional allocation',f=>{f.allocation.targets[0].count=1.5}],['negative allocation',f=>{f.allocation.targets[0].count=-1}],['NaN allocation',f=>{f.allocation.targets[0].count=NaN}],['duplicate targets',f=>{f.allocation.targets[1].targetUuid=f.targets[0].uuid}],['all zero',f=>{for(const t of f.allocation.targets)t.count=0}],['bad fingerprint',f=>{f.scope.fingerprint='nothex'}],['missing prepared slot',f=>{delete f.entry.system.slots.slot3.value}],['no slots',f=>{f.entry.system.slots.slot3.value=0}],['not signature',f=>{f.item.system.location.signature=false}],['prepared entry',f=>{f.entry.system.prepared.value='prepared'}],['non-occult entry',f=>{f.entry.system.tradition.value='arcane'}],['dead',f=>{f.actor.isDead=true}],['missing canAct',f=>{delete f.actor.canAct}],['inactive author',f=>{f.user.active=false}],['not GM',f=>{f.game.user=f.user}],['wrong owner',f=>{f.actor.testUserPermission=()=>false}],['replaced actor',f=>{f.game.actors.set(f.actor.id,{...f.actor})}],['replaced item',f=>{f.actor.items.set(f.item.id,{...f.item})}],['wrong entry',f=>{f.item.system.location.value='other'}],['source token deleted',f=>{f.scene.tokens.delete(f.token.id)}],['target moved',f=>{f.targets[0].parent={id:'other'}}],
])test(`claim rejects ${name} before writing`,async()=>{const f=fixture();change(f);await assert.rejects(f.claim());assert.equal(f.writes.length,0)});

test('startCast persists one permission and does not hold the resource lock over native continuation',async()=>{
 const f=fixture(),r=await f.claim(),ctx={...f.scope,nonce:r.nonce};await f.ledger.startCast(ctx);assert.equal(f.current().status,'casting');await assert.rejects(f.ledger.startCast(ctx));assert.equal(f.ledger.assertCastIntent({...ctx,fingerprint:f.scope.fingerprint}),true);assert.equal(await f.withActorResourceLock(f.actor,async()=>f.ledger.assertCastIntent({...ctx,fingerprint:f.scope.fingerprint})),true);
});
test('startCast and core intent recheck canAct and initial GM',async()=>{
 for(const stage of ['startCast','assertCastIntent'])for(const fail of ['dead','cannotAct','gm']){
  const f=fixture(),r=await f.claim(),ctx={...f.scope,nonce:r.nonce};if(stage==='assertCastIntent')await f.ledger.startCast(ctx);if(fail==='dead')f.actor.isDead=true;else if(fail==='cannotAct')f.actor.canAct=false;else{const g={id:'newgm',isGM:true,active:true};f.users.set(g.id,g);f.users.activeGM=g;f.game.user=g;}
  if(stage==='startCast')await assert.rejects(f.ledger.startCast(ctx));else assert.throws(()=>f.ledger.assertCastIntent(ctx));
 }
});
test('bind only the exact completed payment and original live card; later slot marker replacement is harmless',async()=>{
 const f=fixture(),ctx=await f.casting(),outcome=f.outcome(ctx);f.entry.flags[ID]={nativeSlotCommit:{castNonce:'a-later-legitimate-cast'}};await f.ledger.bindCast({...ctx,outcome});assert.equal(f.current().status,'paid');assert.equal(f.current().castNonce,outcome.castNonce);assert.equal(f.current().originalMessageUuid,outcome.message.uuid);const count=f.writes.length;await f.ledger.bindCast({...ctx,outcome});assert.equal(f.writes.length,count);assert.throws(()=>f.ledger.assertCastIntent(ctx));
});
for(const [name,change]of [
 ['void outcome',(_f,o)=>{o.status=undefined}],['not completed',(_f,o)=>{o.status='disrupted'}],['missing receipt',(f)=>{f.actor.flags[ID].nativeCasts=[]}],['uncertain payment',(f)=>{f.actor.flags[ID].nativeCasts[0].state='uncertain'}],['wrong nonce',(f)=>{f.actor.flags[ID].nativeCasts[0].invocation.data.bridgeNonce='wrong'}],['wrong fingerprint',(f)=>{f.actor.flags[ID].nativeCasts[0].invocation.data.fingerprint='b'.repeat(64)}],['missing slot witness',(f)=>{delete f.actor.flags[ID].nativeCasts[0].slotCommit}],['wrong debit',(f)=>{f.actor.flags[ID].nativeCasts[0].slotCommit.after=0}],['wrong original card',(f,o)=>{f.actor.flags[ID].nativeCasts[0].messageId='another'}],['copied message',(f,o)=>{f.messages.set(o.message.id,{...o.message})}],['wrong author',(_f,o)=>{o.message.author={id:'other'}}],['wrong PF source',(_f,o)=>{o.message.flags.pf2e.origin.uuid='Other.item'}],['wrong proof',(_f,o)=>{o.message.flags[ID].nativeCast.id='old'}],['wrong input',(_f,o)=>{o.message.flags[ID].nativeCastInput.rank=1}],['roll instead of cast',(_f,o)=>{o.message.rolls=[{}]}],
])test(`bind rejects ${name} without dice permission`,async()=>{const f=fixture(),ctx=await f.casting(),outcome=f.outcome(ctx);change(f,outcome);const count=f.writes.length;await assert.rejects(f.ledger.bindCast({...ctx,outcome}));assert.equal(f.writes.length,count);assert.equal(f.current().status,'casting')});

test('only an unstarted claim can cancel; a fresh explicit invocation never resumes the old nonce',async()=>{
 const f=fixture(),r=await f.claim(),ctx={...f.scope,nonce:r.nonce};await f.ledger.cancelClaim(ctx);assert.equal(f.current().status,'cancelled');await assert.rejects(f.ledger.startCast(ctx));await assert.rejects(f.claim());const next=await f.claim({invocationId:'new-explicit-cast'});assert.notEqual(next.nonce,r.nonce);assert.equal(f.ledger.read(f.actor,r.nonce).status,'cancelled');await f.ledger.startCast({...ctx,nonce:next.nonce});await assert.rejects(f.ledger.cancelClaim({...ctx,nonce:next.nonce}));
});
test('each write rejects veto/no-op/tampered result and a lost reply cannot repeat a Cast permit',async()=>{
 for(const mode of ['veto','noop','tamper','wrong']){const f=fixture();f.actor.mode=mode;await assert.rejects(f.claim());}
 const f=fixture(),r=await f.claim(),ctx={...f.scope,nonce:r.nonce};f.actor.mode='lost';await assert.rejects(f.ledger.startCast(ctx),/reply lost/);f.actor.mode=null;assert.equal(f.current().status,'casting');await assert.rejects(f.ledger.startCast(ctx));await assert.rejects(f.make().claim({...f.scope,invocationId:'reload-fresh'}));
});

const damageJSON=()=>({class:'DamageRoll',formula:'{4d4 + 4[force]}',evaluated:true,total:14,options:{type:'damage-roll'},terms:[{class:'InstancePool',options:{},rolls:[{class:'DamageInstance',formula:'4d4 + 4',evaluated:true,total:14,options:{flavor:'force'},terms:[{class:'ArithmeticExpression',operator:'+',options:{},operands:[{class:'Die',number:4,faces:4,options:{flavor:'force'},modifiers:[],results:[{result:2,active:true},{result:2,active:true},{result:3,active:true},{result:3,active:true}],evaluated:true},{class:'NumericTerm',number:4,options:{},evaluated:true}]}]}],results:[{result:14,active:true}],evaluated:true}]});
function damageCard(f,ctx,targetUuid,rollJSON=damageJSON(),id='damage-'+targetUuid.split('.').at(-1)){
 const r=f.current(),t=r.targets.find(t=>t.targetUuid===targetUuid),message={id,uuid:`ChatMessage.${id}`,documentName:'ChatMessage',author:f.user,isDamageRoll:true,speaker:{actor:f.actor.id,scene:f.scene.id,token:f.token.id},blind:false,whisper:[],rolls:[{toJSON:()=>copy(rollJSON)}],flags:{pf2e:{origin:{uuid:f.item.uuid,actor:f.actor.uuid,castRank:3}},'pf2e-toolbelt':{targetHelper:{targets:[targetUuid]}},[ID]:{forceBarrage:{bridgeNonce:ctx.nonce,castNonce:r.castNonce,originalMessageUuid:r.originalMessageUuid,targetUuid,count:t.count,fingerprint:r.fingerprint}}}};
 f.messages.set(id,message);f.docs.set(message.uuid,message);return message;
}
async function publishing(f,ctx,targetUuid=f.targets[0].uuid){
 const scope={...ctx,targetUuid};await f.ledger.startTarget(scope);const rollJSON=damageJSON();await f.ledger.recordRoll({...scope,rollJSON});await f.ledger.beginPublication(scope);return {...scope,rollJSON};
}
test('each target has durable single roll and publication permissions, zero allocations have none',async()=>{
 const f=fixture(),ctx=await f.paid(),targetUuid=f.targets[0].uuid,scope={...ctx,targetUuid};await assert.rejects(f.ledger.startTarget({...ctx,targetUuid:f.targets[2].uuid}));
 await f.ledger.startTarget(scope);assert.equal(f.current().targets[0].status,'rolling');await assert.rejects(f.ledger.startTarget(scope));const rollJSON=damageJSON();await f.ledger.recordRoll({...scope,rollJSON});await f.ledger.recordRoll({...scope,rollJSON});await assert.rejects(f.ledger.recordRoll({...scope,rollJSON:{...rollJSON,total:15}}));
 await f.ledger.beginPublication(scope);assert.equal(f.current().targets[0].status,'publishing');await assert.rejects(f.ledger.beginPublication(scope));const message=damageCard(f,ctx,targetUuid,rollJSON);await f.ledger.finishPublication({...scope,rollJSON,message});assert.equal(f.current().targets[0].status,'published');const writes=f.writes.length;await f.ledger.finishPublication({...scope,rollJSON,message});assert.equal(f.writes.length,writes);assert.equal(f.current().status,'producing');
 const second=await publishing(f,ctx,f.targets[1].uuid);await f.ledger.finishPublication({...second,message:damageCard(f,ctx,second.targetUuid,second.rollJSON)});assert.equal(f.current().status,'delivered');await assert.rejects(f.ledger.startTarget(second));assert.equal(f.current().targets.length,2);
});
test('DSN term display fields are ignored but mechanical damage type is preserved',async()=>{
 const f=fixture(),ctx=await f.paid(),p=await publishing(f,ctx),display=copy(p.rollJSON),die=display.terms[0].rolls[0].terms[0].operands[0];die.options.type='d4';die.results[0].indexThrow=3;
 await f.ledger.finishPublication({...p,message:damageCard(f,ctx,p.targetUuid,display)});assert.equal(f.current().targets[0].status,'published');
 for(const change of [r=>{r.terms[0].rolls[0].options.flavor='fire'},r=>{r.options.type='healing'},r=>{r.terms[0].rolls[0].type='fire'},r=>{r.terms[0].rolls[0].terms[0].operands[0].results[0].result=4}]){
  const g=fixture(),s=await g.paid(),q=await publishing(g,s),wrong=copy(q.rollJSON);change(wrong);await assert.rejects(g.ledger.finishPublication({...q,message:damageCard(g,s,q.targetUuid,wrong)}));assert.equal(g.current().targets[0].status,'publishing');
 }
});
const nativeDie=roll=>roll.terms[0].rolls[0].terms[0].operands[0];
function withDsnRole(roll){const result=copy(roll),die=nativeDie(result);die.options.dsnRoleManaged=true;die.options.dsnRole='force';for(const r of die.results)r.indexThrow=0;return result;}
test('DSN 6 role decoration on the native Die permits publication without changing the original receipt',async()=>{
 const f=fixture(),ctx=await f.paid(),p=await publishing(f,ctx),original=copy(p.rollJSON),display=withDsnRole(p.rollJSON),displayBefore=copy(display);
 await f.ledger.finishPublication({...p,message:damageCard(f,ctx,p.targetUuid,display)});
 const target=f.current().targets[0];assert.equal(target.status,'published');assert.deepEqual(target.rollJSON,original);assert.deepEqual(p.rollJSON,original);assert.deepEqual(display,displayBefore);
 assert.equal(nativeDie(target.rollWitness).options.dsnRole,undefined);assert.equal(nativeDie(target.rollWitness).options.dsnRoleManaged,undefined);
});
test('recordRoll retains original DSN-decorated input while its comparison witness omits only Die presentation',async()=>{
 const f=fixture(),ctx=await f.paid(),scope={...ctx,targetUuid:f.targets[0].uuid},rollJSON=withDsnRole(damageJSON()),original=copy(rollJSON);
 await f.ledger.startTarget(scope);await f.ledger.recordRoll({...scope,rollJSON});const target=f.current().targets[0];
 assert.deepEqual(target.rollJSON,original);assert.deepEqual(rollJSON,original);assert.equal(nativeDie(target.rollWitness).options.dsnRole,undefined);assert.equal(nativeDie(target.rollWitness).options.dsnRoleManaged,undefined);
});
for(const [name,change]of [
 ['die faces',r=>{nativeDie(r).faces=6}],['die count',r=>{nativeDie(r).number=5}],['die result',r=>{nativeDie(r).results[0].result=1}],['die active state',r=>{nativeDie(r).results[0].active=false}],['die modifier',r=>{nativeDie(r).modifiers.push('kh3')}],['total',r=>{r.total++}],['formula',r=>{r.formula='{4d4 + 4[fire]}'}],['instance flavor',r=>{r.terms[0].rolls[0].options.flavor='fire'}],['die flavor',r=>{nativeDie(r).options.flavor='fire'}],['Roll type',r=>{r.options.type='healing'}],['Roll dsnRole',r=>{r.options.dsnRole='force'}],['Roll dsnRoleManaged',r=>{r.options.dsnRoleManaged=true}],['instance dsnRole',r=>{r.terms[0].rolls[0].options.dsnRole='force'}],['instance dsnRoleManaged',r=>{r.terms[0].rolls[0].options.dsnRoleManaged=true}],['non-Die term dsnRole',r=>{r.terms[0].rolls[0].terms[0].options.dsnRole='force'}],['unknown Die option',r=>{nativeDie(r).options.dsnUnverified=true}],
])test(`DSN-decorated publication still rejects changed ${name}`,async()=>{
 const f=fixture(),ctx=await f.paid(),p=await publishing(f,ctx),display=withDsnRole(p.rollJSON);change(display);const before=copy(f.current()),writes=f.writes.length;
 await assert.rejects(f.ledger.finishPublication({...p,message:damageCard(f,ctx,p.targetUuid,display)}));assert.equal(f.writes.length,writes);assert.deepEqual(f.current(),before);
});
for(const [name,change]of [
 ['copied message',(f,m)=>{f.messages.set(m.id,{...m})}],['wrong author',(_f,m)=>{m.author={id:'other'}}],['wrong source',(_f,m)=>{m.flags.pf2e.origin.uuid='Other.item'}],['wrong rank',(_f,m)=>{m.flags.pf2e.origin.castRank=1}],['wrong speaker',(_f,m)=>{m.speaker.token='other'}],['wrong target',(_f,m)=>{m.flags['pf2e-toolbelt'].targetHelper.targets=['Scene.scene.Token.two']}],['extra target',(_f,m)=>{m.flags['pf2e-toolbelt'].targetHelper.targets.push('Scene.scene.Token.two')}],['wrong nonce',(_f,m)=>{m.flags[ID].forceBarrage.bridgeNonce='old'}],['wrong count',(_f,m)=>{m.flags[ID].forceBarrage.count=2}],['wrong fingerprint',(_f,m)=>{m.flags[ID].forceBarrage.fingerprint='b'.repeat(64)}],['blind',(_f,m)=>{m.blind=true}],['whisper',(_f,m)=>{m.whisper=['gm']}],['non damage card',(_f,m)=>{m.isDamageRoll=false}],
])test(`publication rejects ${name} without marking success`,async()=>{const f=fixture(),ctx=await f.paid(),p=await publishing(f,ctx),message=damageCard(f,ctx,p.targetUuid,p.rollJSON);change(f,message);const writes=f.writes.length;await assert.rejects(f.ledger.finishPublication({...p,message}));assert.equal(f.writes.length,writes)});
test('already paid results settle after inability to act, while actor ownership and original GM remain required',async()=>{
 const f=fixture(),ctx=await f.casting(),outcome=f.outcome(ctx);f.actor.canAct=false;f.actor.isDead=true;await f.ledger.bindCast({...ctx,outcome});const p=await publishing(f,ctx);await f.ledger.finishPublication({...p,message:damageCard(f,ctx,p.targetUuid,p.rollJSON)});assert.equal(f.current().targets[0].status,'published');
 f.actor.testUserPermission=()=>false;await assert.rejects(f.ledger.startTarget({...ctx,targetUuid:f.targets[1].uuid}));
});
test('partial publication followed by uncertainty preserves facts and never grants old permissions again',async()=>{
 const f=fixture(),ctx=await f.paid(),p=await publishing(f,ctx),message=damageCard(f,ctx,p.targetUuid,p.rollJSON);await f.ledger.finishPublication({...p,message});const other={...ctx,targetUuid:f.targets[1].uuid};await f.ledger.startTarget(other);await f.ledger.uncertain({...ctx,reason:'second result lost'});const old=f.ledger.read(f.actor,ctx.nonce);assert.equal(old.status,'uncertain');assert.equal(old.targets[0].messageUuid,message.uuid);assert.equal(old.targets[1].status,'rolling');
 await assert.rejects(f.make().startTarget(other));await assert.rejects(f.ledger.recordRoll({...other,rollJSON:damageJSON()}));const fresh=await f.claim({invocationId:'new-explicit-cast'});assert.notEqual(fresh.nonce,ctx.nonce);await assert.rejects(f.ledger.finishPublication({...p,message}));assert.deepEqual(f.ledger.read(f.actor,ctx.nonce),old);
});
test('reload never reissues rolling or publishing permits',async()=>{
 for(const status of ['rolling','publishing']){const f=fixture(),ctx=await f.paid(),scope={...ctx,targetUuid:f.targets[0].uuid};if(status==='publishing')await publishing(f,ctx);else await f.ledger.startTarget(scope);const reload=f.make();await assert.rejects(reload.startTarget(scope));await assert.rejects(reload.beginPublication(scope));await assert.rejects(reload.claim({...f.scope,invocationId:'reload'}));}
});
test('disrupted requires the exact core paid receipt and never gives a target permission or refunds',async()=>{
 const f=fixture(),ctx=await f.casting(),o=f.outcome(ctx),p=f.actor.flags[ID].nativeCasts[0];delete p.messageId;p.state='disrupted';o.status='disrupted';o.receipt=copy(p);o.message=null;await f.ledger.finishWithoutDamage({...ctx,outcome:o});assert.equal(f.current().status,'disrupted');assert.equal(f.entry.system.slots.slot3.value,1);await assert.rejects(f.ledger.startTarget({...ctx,targetUuid:f.targets[0].uuid}));
 const g=fixture(),s=await g.casting(),wrong=g.outcome(s);wrong.status='disrupted';wrong.message=null;await assert.rejects(g.ledger.finishWithoutDamage({...s,outcome:wrong}));
});
test('admission does not return a permit if canAct is lost during its durable write',async()=>{
 for(const stage of ['claim','startCast']){
  const f=fixture();let scope=f.scope;if(stage==='startCast'){const r=await f.claim();scope={...scope,nonce:r.nonce};}
  const update=f.actor.update;f.actor.update=async changes=>{const result=await update(changes);f.actor.canAct=false;return result;};await assert.rejects(f.ledger[stage](scope));
 }
});
test('a disrupted bridge nonce cannot also belong to another native receipt',async()=>{
 const f=fixture(),ctx=await f.casting(),o=f.outcome(ctx),p=f.actor.flags[ID].nativeCasts[0];delete p.messageId;p.state='disrupted';o.status='disrupted';o.receipt=copy(p);o.message=null;f.actor.flags[ID].nativeCasts.push({...copy(p),id:'other-cast'});await assert.rejects(f.ledger.finishWithoutDamage({...ctx,outcome:o}));
});
test('target stages reject each failed write without acquiring the next permission',async()=>{
 for(const mode of ['veto','noop'])for(const stage of ['startTarget','recordRoll','beginPublication','finishPublication','uncertain']){
  const f=fixture(),ctx=await f.paid(),scope={...ctx,targetUuid:f.targets[0].uuid,rollJSON:damageJSON()};
  if(['recordRoll','beginPublication','finishPublication'].includes(stage))await f.ledger.startTarget(scope);
  if(['beginPublication','finishPublication'].includes(stage))await f.ledger.recordRoll(scope);
  if(stage==='finishPublication'){await f.ledger.beginPublication(scope);scope.message=damageCard(f,ctx,scope.targetUuid,scope.rollJSON);}
  const before=copy(f.current());f.actor.mode=mode;await assert.rejects(f.ledger[stage](scope));assert.deepEqual(f.current(),before);
 }
});
