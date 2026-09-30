import {test} from 'node:test';import assert from 'node:assert/strict';
import {createRefocusProvider,createRefocusAdapter,focusFinishSatisfied,createFocusHealingProvider} from '../../scripts/exploration/refocus.mjs';
test('focus healing consumes native focus once and requires explicit healing overlay',async()=>{
  const actor={uuid:'Actor.H',flags:{},system:{resources:{focus:{value:1,max:1}}}},patient={uuid:'Actor.P',modeOfBeing:'living'};const ctx={validate(){}};const messages=new Map(),hooks=new Map();let consume,capture,seq=0,applies=0;
  const Hooks={on:(n,f)=>{hooks.set(++seq,{n,f});return seq},off:(n,id)=>hooks.delete(id)};const fire=(n,m)=>{for(const h of hooks.values())if(h.n===n)h.f(m,m)};
  const card={id:'CAST',flags:{pf2e:{origin:{uuid:'Actor.H.Item.L'}},'pf2e-third-party-automation':{}},rolls:[]};
  const roll={total:18,_evaluated:true,toJSON:()=>({total:18,formula:'{18[healing]}',evaluated:true})};
  const variant={uuid:'Actor.H.Item.L',actor,sourceId:'Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS',rank:3,system:{cast:{focusPoints:1}},damageKinds:new Set(['healing']),rollDamage:async()=>{const m={id:'D',flags:{pf2e:{origin:{uuid:variant.uuid},context:{options:[]}}},rolls:[roll]};fire('preCreateChatMessage',m);messages.set('D',m);fire('createChatMessage',m);return roll}};
  const item={uuid:variant.uuid,sourceId:variant.sourceId,type:'spell',actor,system:{overlays:{HEAL:{}}},loadVariant:({overlayIds})=>{assert.deepEqual(overlayIds,['HEAL']);return variant}};actor.items=new Map([['L',item]]);
  const entry={cast:async spell=>{const c={actor,item:spell,castNonce:'N',payload:{focusPoints:1},expectFocusCommit:({changes})=>{actor.flags['pf2e-third-party-automation']={explorationFocusCommits:{A:changes({castNonce:'N',before:1,after:0,cost:1})['flags.pf2e-third-party-automation.explorationFocusCommits.A']}}}};await consume(c,async()=>{actor.system.resources.focus.value=0;return true});card.flags['pf2e-third-party-automation']={...card.flags['pf2e-third-party-automation'],explorationFocus:capture(spell),nativeCast:{id:'N'}};messages.set(card.id,card)}};variant.spellcasting=entry;
  const game={user:{id:'G',settings:{showCheckDialogs:false}},time:{worldTime:6},messages};
  const provider=createFocusHealingProvider({game,Hooks,fromUuid:async uuid=>uuid===item.uuid?item:uuid===actor.uuid?actor:patient,ownerOperations:{isActivityContext:c=>c===ctx},castEvents:{addMatcher(){},addConsumePolicy:f=>{consume=f},addCapture:(id,f)=>{capture=f},ensurePaid:async()=>({id:'N',state:'used',messageId:'CAST'})},nativeTreatment:{applySavedResult:async()=>{applies++;return 'R'}}});provider.register();
  const activity={id:'A',actorUUID:actor.uuid,patientUUIDs:[patient.uuid],startedAt:0,endsAt:6,options:{itemUUID:item.uuid}};
  assert.equal((await provider.complete(activity,ctx)).status,'confirmed');assert.equal(actor.system.resources.focus.value,0);assert.equal(applies,1);
  await assert.rejects(provider.complete(activity,ctx),/already/);
});
import {fixture} from '../salubrious-kiss-fixture.mjs';
import {createSalubriousKiss} from '../../scripts/salubrious-kiss.mjs';
test('real Kiss subscriber settles bound original start once without reopening patient chooser',async()=>{
  const f=fixture();const ctx={validate:()=>{}};f.game.scenes.active=f.scene;let rolls=0;
  const kiss=createSalubriousKiss({...f,game:f.game,fromUuid:f.fromUuid,validateRefocusNote:()=>true,isExplorationContext:c=>c===ctx,choose:async()=>{throw Error('unexpected chooser')},executor:{roll:async()=>{rolls++;return {checkId:'C',damageId:'D',degree:2}},apply:async()=>({messageId:'R'})}});
  const a={id:'A1',actorUUID:f.actor.uuid,patientUUIDs:[f.patient.uuid],startedAt:100,endsAt:700,options:{rank:'trained'}};
  assert.equal((await kiss.claimActivity(a,ctx)).status,'started');
  f.game.time.worldTime=700;f.proof.nonce='A1';f.proof.userId=f.gm.id;f.proof.privacy={schema:1,userId:f.gm.id,mode:'public',whisper:[],blind:false};f.proof.noteId='F';f.actor.flags[f.M].avRefocusIntent={...f.proof};f.actor.flags[f.M].refocusEvents=[{...f.proof,state:'claimed'}];
  f.game.messages.set('F',{id:'F',author:f.gm,speaker:{actor:f.actor.id,scene:f.scene.id,token:f.token.id},blind:false,whisper:[],flags:{[f.M]:{avRefocusNote:{...f.proof,kind:'completion'}}}});
  await kiss.onRefocus({actor:f.actor,user:f.gm,proof:f.proof});assert.equal((await kiss.completeActivity(a,ctx)).status,'confirmed');
  await kiss.onRefocus({actor:f.actor,user:f.gm,proof:f.proof});assert.equal(rolls,1);assert.equal(f.effects[0].flags[f.M].salubriousKiss.expiresAt,3700);
});
test('completion and subscriber reuse one composite result, including full focus',async()=>{
  let treatments=0,focusCalls=0;const results=new Map(),ctx={};
  const kiss={claimActivity:async()=>({status:'started'}),completeActivity:async a=>{if(!results.has(a.id)){treatments++;results.set(a.id,{status:'confirmed',proof:{receiptIds:['H1']}})}return results.get(a.id)},getActivityResult:id=>results.get(id)};
  const provider=createRefocusProvider({game:{time:{worldTime:600}},ledger:{getActivity:async()=>({state:'completing'})},capabilities:{discover:async()=>({focus:{value:3,max:3},threePecks:true})},refocusEvents:{complete:async()=>{focusCalls++;return {id:'F1',focusBefore:3,focusAfter:3}}},salubriousKiss:kiss,ownerOperations:{isActivityContext:c=>c===ctx}});
  const a={id:'A',actorUUID:'Actor.H',startedAt:0,endsAt:600,options:{threePecks:true}};
  await provider.begin(a,ctx);await provider.complete(a,ctx);await provider.complete(a,ctx);
  assert.equal(treatments,1);assert.equal(focusCalls,1);
  assert.equal(focusFinishSatisfied({requireFullFocus:true},{focus:{value:3,max:3}}),true);
  assert.equal(focusFinishSatisfied({requireFullFocus:true},{focus:{value:1,max:3}}),false);
});
test('Workbench invocation requires the actual controlled actor and exact focus receipt',async()=>{
  const ctx={},actor={uuid:'Actor.H',system:{resources:{focus:{value:0,max:3}}}};
  const other={uuid:'Actor.Other'};let calls=0,adapter;
  const game={time:{worldTime:600},PF2eWorkbench:{refocus:async()=>{calls++;adapter.capture({actor,proof:{nonce:'A',before:0,after:1,startedAt:0}})}}};
  const canvas={tokens:{controlled:[{actor:other,document:{uuid:'Scene.S.Token.Other'}}]}};
  adapter=createRefocusAdapter({game,canvas,ownerOperations:{isActivityContext:c=>c===ctx},fromUuid:async()=>actor,timeoutMs:30});
  const activity={id:'A',actorUUID:actor.uuid,startedAt:0,endsAt:600};
  await assert.rejects(adapter.complete(activity,ctx),/controlled/);assert.equal(calls,0);
  canvas.tokens.controlled=[{actor,document:{uuid:'Scene.S.Token.H'}}];assert.equal((await adapter.complete(activity,ctx)).focusAfter,1);assert.equal(calls,1);
});
