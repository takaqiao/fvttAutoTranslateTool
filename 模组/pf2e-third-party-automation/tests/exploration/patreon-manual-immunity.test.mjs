import {test} from 'node:test';
import assert from 'node:assert/strict';
import {getNativeActionEvents} from '../../scripts/native-action-events.mjs';
import {createPatreonManualImmunity} from '../../scripts/exploration/patreon-manual-immunity.mjs';
import {manualEvidenceFixture,flush,M} from './manual-evidence-fixture.mjs';

const sourceSHA='89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9';
const nativeTag='exploration-manual:U';

function nativeFixture({outcome='success'}={}){
 const f=manualEvidenceFixture();f.messages.clear();f.game.time.worldTime=100;
 f.game.system={version:'8.5.1'};f.game.release={generation:14};
 f.healer.type='character';f.patient.type='character';
 f.descriptor={version:1,providerId:'patreon-v3',providerVersion:'3.2.29',baseSourceSHA256:sourceSHA,
  pf2eSourceSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157'};
 f.binding={invocationId:'INV',messageId:'C',useId:'U',tag:nativeTag,actorUUID:'Actor.H',patientUUID:'Actor.P',sourceUserId:'HUSER',startedAt:100};
 const observers=new Set();
 f.provider={descriptor:f.descriptor,subscribe:fn=>{observers.add(fn);return ()=>observers.delete(fn)}};
 f.game.modules=new Map([['patreon-v3',{active:true,version:'3.2.29',api:{explorationManualImmunity:f.provider}}]]);
 f.healer.items.some=predicate=>[...f.healer.items.values()].some(predicate);
 f.options.hpPools={discover:actor=>({poolUUID:actor.uuid,ready:true})};
 f.check={id:'C',isCheckRoll:true,isReroll:false,author:f.users.get('HUSER'),speaker:{actor:'H'},
  rolls:[{_evaluated:true,formula:'1d20 + 9',total:19}],
  flags:{[M]:{explorationManualNative:{tag:nativeTag,useId:'U',patientUUID:'Actor.P',continualRecovery:false,startedAt:100,riskySurgery:false,patreonImmunity:{...f.binding}}},
   pf2e:{modifiers:[],context:{type:'skill-check',actor:'H',origin:{actor:'Actor.H'},target:{actor:'Actor.P'},options:['action:treat-wounds',nativeTag],outcome}}},
  updateSource(changes){this.flags=changes.flags}};
 f.child={id:'D',isCheckRoll:false,author:f.users.get('HUSER'),speaker:{actor:'H'},rolls:[{_evaluated:true,formula:'{2d8[healing]}'}],
  flags:{[M]:{explorationManualNative:{...f.check.flags[M].explorationManualNative}},
   pf2e:{origin:{messageId:'C'},context:{type:'damage-roll',origin:{actor:'Actor.H'},target:{actor:'Actor.P'},options:[nativeTag]}}},
  async update(changes){this.flags.pf2e.context.options=changes['flags.pf2e.context.options'];return this}};
 f.receipt=(id='R',author='PUSER')=>({id,author:f.users.get(author),speaker:{actor:'P'},
  flags:{pf2e:{context:{type:'damage-taken',options:[`${M}:source:D:0`]},appliedDamage:{uuid:'Actor.P',isHealing:true,isReverted:false}}}});
 f.options.fromUuid=async uuid=>f.actors.get(uuid)??[...f.patient.items.values()].find(item=>item.uuid===uuid);
 const createRecorder=f.createRecorder;
 f.createRecorder=()=>{
  f.options.patreonImmunity=createPatreonManualImmunity({game:f.game,fromUuid:f.options.fromUuid});
  return createRecorder();
 };
 f.item=(id='I')=>({id,uuid:`Actor.P.Item.${id}`,actor:f.patient,parent:f.patient,type:'effect',sourceId:'Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5',
  system:{slug:'treat-wounds-immunity',start:{value:100},duration:{value:60,unit:'minutes',expiry:'turn-start',sustained:false}},
  flags:{[M]:{explorationManualPatreonImmunity:{...f.binding,creatorId:'G'}}}});
 f.terminal=()=>({descriptor:f.descriptor,binding:{...f.binding},creatorId:'G',itemUUID:'Actor.P.Item.I',start:100,
  duration:{value:60,unit:'minutes',expiry:'turn-start',sustained:false},expiresAt:3700});
 f.emit=async(terminalPromise,binding=f.binding,descriptor=f.descriptor)=>{
  for(const observer of observers)observer({descriptor,binding,terminalPromise});await flush();
 };
 f.saveItem=async(item=f.item())=>{f.patient.items.set(item.id,item);f.handlers.get('createItem')?.(item,{},'G');await flush();return item};
 f.activity=()=>f.ledger.getActivity('manual:C');
 return f;
}

async function recordNative(f,{child=true,receipt=true}={}){
 const recorder=f.createRecorder();recorder.start();await f.fire(f.check);
 if(child)await f.fire(f.child);if(receipt)await f.fire(f.receipt());
 assert.equal((await f.activity())?.state,'awaiting-evidence');return recorder;
}

async function finishImmunity(f){await f.saveItem();await f.emit(Promise.resolve(f.terminal()))}

function defaultTargetFixture(){
 const f=nativeFixture(),resolveUUID=f.options.fromUuid;
 f.binding.targetSnapshot={type:'patreon-single-target',actorUUID:'Actor.P',tokenUUID:'Scene.S.Token.T'};
 f.check.flags.pf2e.context.target=null;f.child.flags.pf2e.context.target=null;
 f.check.flags[M].explorationManualNative.patientUUID=null;f.child.flags[M].explorationManualNative.patientUUID=null;
 delete f.check.flags[M].explorationManualNative.patreonImmunity;delete f.child.flags[M].explorationManualNative.patreonImmunity;
 f.scene={id:'S',uuid:'Scene.S',tokens:new Map()};f.token={id:'T',uuid:'Scene.S.Token.T',actor:f.patient,parent:f.scene};
 f.scene.tokens.set('T',f.token);f.game.scenes.set('S',f.scene);f.game.scenes.active=f.scene;
 f.game.user.targets=new Set([{uuid:'Scene.DECOY.Token.D',actor:{uuid:'Actor.DECOY'}}]);
 f.options.fromUuid=async uuid=>uuid==='Scene.S.Token.T'?f.scene.tokens.get('T'):resolveUUID(uuid);
 f.markPatient=()=>{
  f.check.flags[M].explorationManualNative.patientUUID='Actor.P';
  f.check.flags[M].explorationManualNative.patreonImmunity=structuredClone(f.binding);
 };
 return f;
}

async function useUnboundNative(f){
 const variant={async use(params){
  f.check.flags.pf2e.context.options=['action:treat-wounds',...params.rollOptions];
  f.handlers.get('preCreateChatMessage')(f.check,{flags:f.check.flags});
  const native=f.check.flags[M].explorationManualNative;f.binding.useId=native.useId;f.binding.tag=native.tag;
  f.child.flags[M].explorationManualNative={...native};f.child.flags.pf2e.context.options=[native.tag];
  await f.fire(f.check);await f.fire(f.child);await f.fire(f.receipt());
  return [{actor:f.healer,message:f.check,outcome:'success'}];
 }};
 const action={slug:'treat-wounds',variants:new Map([['default',variant]]),toActionVariant:()=>variant};
 f.game.pf2e={actions:new Map([['treat-wounds',action]])};
 const nativeActions=getNativeActionEvents({game:f.game});f.options.nativeActions=nativeActions;
 const recorder=f.createRecorder();recorder.start();nativeActions.register();
 try{
  await variant.use({actors:[f.healer]});await flush();
  assert.equal((await f.ledger.snapshot('S')).activities.length,0);
  assert.equal(f.check.flags[M].explorationManualNative.patientUUID,null);
  assert.equal(f.child.flags[M].explorationManualNative.patientUUID,null);
 }catch(error){recorder.stop();nativeActions.cleanup();throw error}
 return ()=>{recorder.stop();nativeActions.cleanup()};
}

async function assertNoDefaultConfirmation(f){
 const rows=(await f.ledger.snapshot('S')).activities;
 assert.ok(rows.every(activity=>activity.state==='awaiting-evidence'));
 for(const activity of rows)assert.ok(activity.options.missing.includes('native-immunity-receipt'));
}

test('an omitted native action target is recovered from its persisted check target',async()=>{
 const f=nativeFixture();delete f.check.flags[M];
 f.game.user.targets=new Set([{actor:{uuid:'Actor.DECOY'}}]);
 const variant={async use(params){
  f.check.flags.pf2e.context.options=[...params.rollOptions];
  f.handlers.get('preCreateChatMessage')(f.check,{flags:f.check.flags});
  await f.fire(f.check);return [{actor:f.healer,message:f.check,outcome:'success'}];
 }};
 const action={slug:'treat-wounds',variants:new Map([['default',variant]]),toActionVariant:()=>variant};
 f.game.pf2e={actions:new Map([['treat-wounds',action]])};
 const nativeActions=getNativeActionEvents({game:f.game});f.options.nativeActions=nativeActions;
 const recorder=f.createRecorder();recorder.start();nativeActions.register();
 try{
  await variant.use({actors:[f.healer]});await flush();
  const rows=(await f.ledger.snapshot('S')).activities;
  assert.equal(rows.length,1);assert.deepEqual(rows[0].patientUUIDs,['Actor.P']);
  assert.equal(f.check.flags[M].explorationManualNative.patientUUID,'Actor.P');
  assert.equal(f.check.flags[M].explorationManualNative.riskySurgery,false);
 }finally{recorder.stop();nativeActions.cleanup()}
});

test('a native player treatment confirms only after the original GM immunity promise fulfills',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);let resolve;
 const pending=new Promise(done=>{resolve=done});
 try{
  await f.saveItem();
  assert.deepEqual((await f.activity()).options.missing,['native-immunity-receipt']);
  await f.emit(pending);assert.equal((await f.activity()).state,'awaiting-evidence');
  resolve(f.terminal());await flush();const activity=await f.activity();
  assert.equal(activity.state,'confirmed');assert.deepEqual(activity.options.missing,[]);
  assert.deepEqual(activity.proof.checkIds,['C']);assert.deepEqual(activity.proof.resultIds,['D']);
  assert.deepEqual(activity.proof.receiptIds,['R']);assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);
  assert.equal(activity.proof.nativeImmunity.itemUUID,'Actor.P.Item.I');
  assert.equal(activity.startedAt,100);assert.equal(activity.endsAt,700);assert.equal(f.game.time.worldTime,100);
  assert.deepEqual(f.errors,[]);
 }finally{recorder.stop()}
});

test('a native child without context type closes through its real check origin and HP receipt',async()=>{
 const f=nativeFixture();delete f.child.flags.pf2e.context.type;
 const recorder=await recordNative(f);
 try{
  await finishImmunity(f);const activity=await f.activity();
  assert.equal(activity.state,'confirmed');assert.deepEqual(activity.options.missing,[]);
  assert.deepEqual(activity.proof.checkIds,['C']);assert.deepEqual(activity.proof.resultIds,['D']);
  assert.deepEqual(activity.proof.receiptIds,['R']);assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);
  assert.ok(f.child.flags.pf2e.context.options.includes(`${M}:source:D:0`));assert.deepEqual(f.errors,[]);
 }finally{recorder.stop()}
});

test('recorder reconstruction revalidates a saved native immunity seal and existing HP receipt',async()=>{
 const f=nativeFixture(),old=await recordNative(f,{receipt:false});await finishImmunity(f);
 assert.equal((await f.activity()).state,'awaiting-evidence');
 assert.deepEqual((await f.activity()).options.missing,['native-application-receipt']);
 assert.equal((await f.activity()).proof.nativeImmunity.itemUUID,'Actor.P.Item.I');old.stop();
 const receipt=f.receipt();f.messages.set(receipt.id,receipt);
 const fresh=f.createRecorder();fresh.start();await flush();
 try{
  const activity=await f.activity();assert.equal(activity.state,'confirmed');
  assert.deepEqual(activity.proof.receiptIds,['R']);assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);
  assert.equal(f.patient.items.size,1);assert.equal(f.game.time.worldTime,100);assert.deepEqual(f.errors,[]);
 }finally{fresh.stop()}
 const again=f.createRecorder();again.start();await flush();
 assert.deepEqual((await f.activity()).proof.immunityIds,['Actor.P.Item.I']);assert.equal(f.patient.items.size,1);again.stop();
});

test('a pinned ordinary native failure with no output stages needs immunity and no HP receipt',async()=>{
 const f=nativeFixture({outcome:'failure'}),recorder=await recordNative(f,{child:false,receipt:false});
 try{
  await finishImmunity(f);const activity=await f.activity();
  assert.equal(activity.state,'confirmed');assert.deepEqual(activity.options.missing,[]);
  assert.deepEqual(activity.proof.checkIds,['C']);assert.deepEqual(activity.proof.resultIds,[]);
  assert.deepEqual(activity.proof.receiptIds,[]);assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);
 }finally{recorder.stop()}
});

test('a native failure with a Risky Surgery stage still requires its native HP receipt',async()=>{
 const f=nativeFixture({outcome:'failure'});f.check.flags[M].explorationManualNative.riskySurgery=true;
 f.child.rolls[0].formula='{1d8[slashing]}';
 const recorder=await recordNative(f,{receipt:false});
 try{
  await finishImmunity(f);const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.ok(activity.options.missing.includes('native-application-receipt'));
  assert.deepEqual(activity.proof.resultIds,['D']);
 }finally{recorder.stop()}
});

for(const [name,invalidate,missing] of [
 ['missing HP receipt',()=>{},'native-application-receipt'],
 ['reverted HP receipt',f=>{f.savedReceipt.flags.pf2e.appliedDamage.isReverted=true},'native-application-receipt'],
 ['deleted HP receipt',f=>{f.messages.delete('R')},'native-application-receipt'],
 ['deleted native child card',f=>{f.messages.delete('D')},'native-application-receipt'],
 ['revoked HP author ownership',f=>{f.patient.testUserPermission=user=>user?.isGM===true},'native-application-receipt'],
 ['wrong HP patient',f=>{f.savedReceipt.flags.pf2e.appliedDamage.uuid='Actor.OTHER'},'native-application-receipt'],
 ['deleted native check',f=>{f.messages.delete('C')},'native-immunity-receipt'],
 ['revoked healer ownership',f=>{f.healer.testUserPermission=user=>user?.isGM===true},'native-immunity-receipt'],
 ['revoked immunity creator permission',f=>{f.users.get('G').isGM=false},'native-immunity-receipt'],
 ['changed check target',f=>{f.check.flags.pf2e.context.target.actor='Actor.OTHER'},'native-immunity-receipt'],
 ['changed check origin',f=>{f.check.flags.pf2e.context.origin.actor='Actor.OTHER'},'native-immunity-receipt'],
 ['changed check author',f=>{f.check.author=f.users.get('OUTSIDER')},'native-immunity-receipt'],
 ['unevaluated check',f=>{f.check.rolls[0]._evaluated=false},'native-immunity-receipt'],
 ['changed native use',f=>{f.check.flags[M].explorationManualNative.useId='OTHER'},'native-immunity-receipt'],
 ['external world time change',f=>{f.game.time.worldTime=101},'native-immunity-receipt'],
 ['shared HP pool',f=>{f.options.hpPools.discover=()=>({poolUUID:'Actor.Master',ready:true})},'shared-hp-completion-unavailable'],
])test(`an original immunity cannot complete a treatment with ${name}`,async()=>{
 const f=nativeFixture();
 if(name==='shared HP pool')invalidate(f);
 const recorder=await recordNative(f,{receipt:false});
 if(name!=='missing HP receipt'){f.savedReceipt=f.receipt();await f.fire(f.savedReceipt)}
 if(name!=='shared HP pool')invalidate(f);
 try{
  await finishImmunity(f);const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.ok(activity.options.missing.includes(missing),activity.options.missing.join(','));
 }finally{recorder.stop()}
});

test('two native immunity items for one invocation remain ambiguous',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);
 try{
  await f.saveItem();await f.saveItem(f.item('SECOND'));await f.emit(Promise.resolve(f.terminal()));
  const activity=await f.activity();assert.equal(activity.state,'awaiting-evidence');
  assert.ok(activity.options.missing.includes('native-immunity-receipt'));assert.deepEqual(activity.proof.immunityIds,[]);
 }finally{recorder.stop()}
});

test('a persisted Patreon marker without an original provider terminal cannot confirm immunity',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);
 try{
  await f.saveItem();const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.deepEqual(activity.options.missing,['native-immunity-receipt']);
  assert.deepEqual(activity.proof.immunityIds,[]);assert.equal(activity.proof.nativeImmunity,undefined);
 }finally{recorder.stop()}
});

for(const [name,change] of [
 ['unverified Patreon source',f=>{f.descriptor.baseSourceSHA256='unverified'}],
 ['unverified PF2e source',f=>{f.descriptor.pf2eSourceSHA256='unverified'}],
 ['unbound native use',f=>{f.binding.useId='FORGED'}],
 ['wrong invocation patient',f=>{f.binding.patientUUID='Actor.OTHER'}],
 ['unpersisted created Item',f=>{f.patient.items.clear()}],
 ['wrong native start time',f=>{f.patient.items.get('I').system.start.value=99}],
 ['wrong native duration',f=>{f.patient.items.get('I').system.duration.value=10}],
 ['wrong immunity template',f=>{f.patient.items.get('I').sourceId='Compendium.pf2e.feat-effects.Item.OTHER'}],
])test(`a terminal with ${name} cannot admit native immunity`,async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);await f.saveItem();change(f);
 try{
  await f.emit(Promise.resolve(f.terminal()));const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.ok(activity.options.missing.includes('native-immunity-receipt'));
  assert.deepEqual(activity.proof.immunityIds,[]);
 }finally{recorder.stop()}
});

test('an unsuccessful original create promise leaves native immunity unconfirmed',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);await f.saveItem();
 const rejected=Promise.reject(new Error('native-create-rejected'));rejected.catch(()=>{});
 try{
  await f.emit(rejected);assert.equal((await f.activity()).state,'awaiting-evidence');
  assert.ok((await f.activity()).options.missing.includes('native-immunity-receipt'));
 }finally{recorder.stop()}
});

for(const [name,change] of [
 ['an unknown PF2e callback version',f=>{f.game.system.version='8.5.2'}],
 ['a rerolled check',f=>{f.check.isReroll=true}],
 ['a missing Risky Surgery fact',f=>{delete f.check.flags[M].explorationManualNative.riskySurgery}],
 ['missing original check modifiers',f=>{delete f.check.flags.pf2e.modifiers}],
])test(`a failure with ${name} cannot prove an absent native HP stage`,async()=>{
 const f=nativeFixture({outcome:'failure'}),recorder=await recordNative(f,{child:false,receipt:false});change(f);
 try{
  await finishImmunity(f);const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.ok(activity.options.missing.includes('native-application-receipt'));
 }finally{recorder.stop()}
});

for(const [name,change] of [
 ['changed check target',f=>{f.check.flags.pf2e.context.target.actor='Actor.OTHER'}],
 ['revoked creator permission',f=>{f.users.get('G').isGM=false}],
 ['external time advancement',f=>{f.game.time.worldTime=101}],
])test(`an immunity promise cannot reuse authorization after ${name} while pending`,async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);await f.saveItem();let resolve;
 const pending=new Promise(done=>{resolve=done});await f.emit(pending);change(f);
 try{
  resolve(f.terminal());await flush();const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.ok(activity.options.missing.includes('native-immunity-receipt'));
  assert.deepEqual(activity.proof.immunityIds,[]);
 }finally{recorder.stop()}
});

test('repeated recorder starts and identical native terminals preserve one treatment and immunity',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);recorder.start();await f.saveItem();
 try{
  await f.emit(Promise.resolve(f.terminal()));await f.emit(Promise.resolve(f.terminal()));
  const rows=(await f.ledger.snapshot('S')).activities;
  assert.equal(rows.length,1);assert.equal(rows[0].state,'confirmed');
  assert.deepEqual(rows[0].proof.immunityIds,['Actor.P.Item.I']);assert.equal(f.patient.items.size,1);
 }finally{recorder.stop()}
});

test('a stopped recorder cannot admit its old unresolved terminal through a replacement subscription',async()=>{
 const f=nativeFixture(),old=await recordNative(f);await f.saveItem();let resolve;
 const pending=new Promise(done=>{resolve=done});await f.emit(pending);old.stop();
 const fresh=f.createRecorder();fresh.start();await flush();
 try{
  resolve(f.terminal());await flush();const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.ok(activity.options.missing.includes('native-immunity-receipt'));
  assert.deepEqual(activity.proof.immunityIds,[]);
 }finally{fresh.stop()}
});

test('a restored unsealed Item marker cannot substitute for the original subscribed terminal',async()=>{
 const f=nativeFixture(),old=await recordNative(f);await f.saveItem();old.stop();
 const fresh=f.createRecorder();fresh.start();await flush();
 try{
  const activity=await f.activity();assert.equal(activity.state,'awaiting-evidence');
  assert.ok(activity.options.missing.includes('native-immunity-receipt'));assert.deepEqual(activity.proof.immunityIds,[]);
 }finally{fresh.stop()}
});

for(const [name,invalidate] of [
 ['deleted immunity Item',f=>{f.patient.items.delete('I')}],
 ['revoked creator permission',f=>{f.users.get('G').isGM=false}],
 ['changed native patient',f=>{f.check.flags.pf2e.context.target.actor='Actor.OTHER'}],
 ['duplicate invocation Item',f=>{const item=f.item('SECOND');f.patient.items.set(item.id,item)}],
])test(`reconstruction cannot complete a saved native seal after ${name}`,async()=>{
 const f=nativeFixture(),old=await recordNative(f,{receipt:false});await finishImmunity(f);
 assert.ok((await f.activity()).proof.nativeImmunity,'the original terminal must have sealed immunity before reconstruction');old.stop();
 invalidate(f);const receipt=f.receipt();f.messages.set(receipt.id,receipt);
 const fresh=f.createRecorder();fresh.start();await flush();
 try{
  const activity=await f.activity();assert.equal(activity.state,'awaiting-evidence');
  assert.ok(activity.options.missing.includes('native-immunity-receipt'));
 }finally{fresh.stop()}
});

test('a copied child hook cannot borrow a persisted card with a different PF2e origin',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f,{child:false,receipt:false});
 const copy={...f.child,flags:structuredClone(f.child.flags)};
 f.child.flags.pf2e.origin.messageId='OTHER';f.messages.set('D',f.child);
 try{
  f.handlers.get('createChatMessage')(copy);await flush();await f.fire(f.receipt());await finishImmunity(f);
  const activity=await f.activity();assert.equal(activity.state,'awaiting-evidence');
  assert.deepEqual(activity.proof.resultIds,[]);assert.ok(activity.options.missing.includes('native-application-receipt'));
 }finally{recorder.stop()}
});

test('a same-shaped child without the real PF2e check origin cannot satisfy native HP evidence',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f,{child:false,receipt:false});
 f.child.flags.pf2e.origin.messageId='OTHER';
 try{
  await f.fire(f.child);await f.fire(f.receipt());await finishImmunity(f);const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.deepEqual(activity.proof.resultIds,[]);
  assert.ok(activity.options.missing.includes('native-application-receipt'));
 }finally{recorder.stop()}
});

test('an unmarked native immunity with the same template, start and healer makes the invocation ambiguous',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);
 try{
  const marked=await f.saveItem();marked.system.context={origin:{actor:'Actor.H'}};
  const duplicate=f.item('UNMARKED');duplicate.flags={};duplicate.system.context={origin:{actor:'Actor.H'}};
  await f.saveItem(duplicate);await f.emit(Promise.resolve(f.terminal()));
  const activity=await f.activity();assert.equal(activity.state,'awaiting-evidence');
  assert.ok(activity.options.missing.includes('native-immunity-receipt'));assert.deepEqual(activity.proof.immunityIds,[]);
 }finally{recorder.stop()}
});

test('a prior HP receipt is rechecked after the final asynchronous immunity Item lookup',async()=>{
 const f=nativeFixture(),resolveUUID=f.options.fromUuid;let armed=false,immunityReads=0,receipt;
 f.options.fromUuid=async uuid=>{
  const document=await resolveUUID(uuid);
  if(armed&&uuid==='Actor.P.Item.I'&&++immunityReads===2){await Promise.resolve();receipt.flags.pf2e.appliedDamage.isReverted=true}
  return document;
 };
 const recorder=await recordNative(f);receipt=f.messages.get('R');await f.saveItem();armed=true;
 try{
  await f.emit(Promise.resolve(f.terminal()));const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.deepEqual(activity.options.missing,['native-application-receipt']);
  assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);
 }finally{recorder.stop()}
});

for(const [name,invalidate] of [
 ['replacement card from another check',f=>{const copy={...f.child,flags:structuredClone(f.child.flags)};copy.flags.pf2e.origin.messageId='OTHER';f.messages.set('D',copy)}],
 ['child author without healer ownership',f=>{f.child.author=f.users.get('OUTSIDER')}],
 ['changed child check origin',f=>{f.child.flags.pf2e.origin.messageId='OTHER'}],
])test(`an asynchronous terminal cannot complete historical HP evidence after ${name}`,async()=>{
 const f=nativeFixture(),resolveUUID=f.options.fromUuid;let armed=false,changed=false;
 f.options.fromUuid=async uuid=>{
  const document=await resolveUUID(uuid);
  if(armed&&!changed&&uuid==='Actor.P.Item.I'){await Promise.resolve();invalidate(f);changed=true}
  return document;
 };
 const recorder=await recordNative(f);await f.saveItem();armed=true;
 try{
  await f.emit(Promise.resolve(f.terminal()));const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.ok(activity.options.missing.includes('native-application-receipt'));
  assert.deepEqual(activity.proof.resultIds,['D']);
 }finally{recorder.stop()}
});

test('a default native target closes from the original single-target snapshot after its earlier HP receipt',async()=>{
 const f=defaultTargetFixture(),stop=await useUnboundNative(f);
 try{
  f.markPatient();await f.saveItem();assert.equal((await f.ledger.snapshot('S')).activities.length,0);
  await f.emit(Promise.resolve(f.terminal()));const rows=(await f.ledger.snapshot('S')).activities;
  assert.equal(rows.length,1);const activity=rows[0];assert.equal(activity.state,'confirmed');
  assert.deepEqual(activity.patientUUIDs,['Actor.P']);assert.deepEqual(activity.proof.checkIds,['C']);
  assert.deepEqual(activity.proof.resultIds,['D']);assert.deepEqual(activity.proof.receiptIds,['R']);
  assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);assert.deepEqual(activity.options.missing,[]);
  assert.equal(f.check.flags.pf2e.context.target,null);assert.equal(f.child.flags[M].explorationManualNative.patientUUID,null);
 }finally{stop()}
});

test('a default native target without the original snapshot cannot be inferred from GM targets',async()=>{
 const f=defaultTargetFixture(),stop=await useUnboundNative(f);delete f.binding.targetSnapshot;
 try{f.markPatient();await finishImmunity(f);await assertNoDefaultConfirmation(f)}finally{stop()}
});

test('a default snapshot for a different patient cannot confirm the native treatment',async()=>{
 const f=defaultTargetFixture(),stop=await useUnboundNative(f);f.binding.targetSnapshot.actorUUID='Actor.OTHER';
 try{f.markPatient();await finishImmunity(f);await assertNoDefaultConfirmation(f)}finally{stop()}
});

test('a default snapshot cannot confirm after its actual target token is removed',async()=>{
 const f=defaultTargetFixture(),stop=await useUnboundNative(f);f.scene.tokens.delete('T');
 try{f.markPatient();await finishImmunity(f);await assertNoDefaultConfirmation(f)}finally{stop()}
});

test('a rewritten default snapshot cannot borrow the original provider terminal',async()=>{
 const f=defaultTargetFixture(),stop=await useUnboundNative(f);f.markPatient();
 f.check.flags[M].explorationManualNative.patreonImmunity.targetSnapshot.tokenUUID='Scene.S.Token.OTHER';
 try{await finishImmunity(f);await assertNoDefaultConfirmation(f)}finally{stop()}
});

test('a forged default-target check and Item marker without a terminal cannot complete the treatment',async()=>{
 const f=defaultTargetFixture(),stop=await useUnboundNative(f);
 try{f.markPatient();await f.saveItem();await assertNoDefaultConfirmation(f)}finally{stop()}
});

test('a denied observation metadata write preserves the original native action and rows',async()=>{
 const f=nativeFixture();let nativeCalls=0;
 f.check.updateSource=()=>{throw Error('metadata-denied')};
 const rows=[{actor:f.healer,message:f.check,outcome:'success'}];
 const variant={async use(params){
  nativeCalls++;f.check.flags.pf2e.context.options=['action:treat-wounds',...params.rollOptions];
  f.handlers.get('preCreateChatMessage')(f.check,{flags:f.check.flags});await f.fire(f.check);return rows;
 }};
 const action={slug:'treat-wounds',variants:new Map([['default',variant]]),toActionVariant:()=>variant};
 f.game.pf2e={actions:new Map([['treat-wounds',action]])};
 const nativeActions=getNativeActionEvents({game:f.game});f.options.nativeActions=nativeActions;
 const recorder=f.createRecorder();recorder.start();nativeActions.register();
 try{
  let result,error;try{result=await variant.use({actors:[f.healer]})}catch(caught){error=caught}
  assert.equal(error,undefined);assert.equal(result,rows);assert.equal(nativeCalls,1);
  await f.emit(Promise.resolve(f.terminal()));
  assert.ok((await f.ledger.snapshot('S')).activities.every(activity=>activity.state!=='confirmed'));
  assert.equal(f.patient.items.size,0);assert.equal(f.game.time.worldTime,100);
 }finally{recorder.stop();nativeActions.cleanup()}
});

test('two ordinary native child rolls with two authorized HP receipts cannot prove the fixed callback output',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);
 const second={...f.child,id:'E',flags:structuredClone(f.child.flags),rolls:[{_evaluated:true,formula:'{2d8[healing]}'}]};
 const secondReceipt=f.receipt('SECOND');secondReceipt.flags.pf2e.context.options=[`${M}:source:E:0`];
 try{
  await f.fire(second);await f.fire(secondReceipt);await finishImmunity(f);const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.ok(activity.options.missing.includes('native-application-receipt'));
  assert.deepEqual(activity.proof.resultIds,['D','E']);assert.deepEqual(activity.proof.receiptIds,['R','SECOND']);
 }finally{recorder.stop()}
});

test('an unregistered persistent child prevents a failure from proving that no HP stage exists',async()=>{
 const f=nativeFixture({outcome:'failure'}),recorder=await recordNative(f,{child:false,receipt:false});delete f.child.flags[M];
 try{
  await f.fire(f.child);assert.deepEqual((await f.activity()).proof.resultIds,[]);
  await finishImmunity(f);const activity=await f.activity();
  assert.equal(activity.state,'awaiting-evidence');assert.ok(activity.options.missing.includes('native-application-receipt'));
 }finally{recorder.stop()}
});

test('two distinct native HP receipts for one result remain ambiguous',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);
 try{
  await f.fire(f.receipt('R2'));await finishImmunity(f);const activity=await f.activity();
  assert.deepEqual(activity.proof.receiptIds,['R','R2']);assert.equal(activity.state,'awaiting-evidence');
  assert.ok(activity.options.missing.includes('native-application-receipt'));
 }finally{recorder.stop()}
});

test('repeated delivery of the same native HP receipt is one application',async()=>{
 const f=nativeFixture(),recorder=await recordNative(f);
 try{
  await f.fire(f.messages.get('R'));await finishImmunity(f);const activity=await f.activity();
  assert.deepEqual(activity.proof.receiptIds,['R']);assert.equal(activity.state,'confirmed');
 }finally{recorder.stop()}
});
