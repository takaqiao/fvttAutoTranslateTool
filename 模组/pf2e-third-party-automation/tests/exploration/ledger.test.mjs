import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createActivity} from '../../scripts/exploration/schema.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createDocumentStore} from '../../scripts/exploration/document-store.mjs';

export const activity = (patch={}) => createActivity({
  id:'A1',sessionId:'S1',providerId:'treat-wounds',actorUUID:'Actor.H',
  patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],groupId:'A1',
  startedAt:-1200,endsAt:-600,kind:'treatment',mode:'automatic',
  source:{userId:'U1',origin:'coordinator'},options:{},state:'planned',...patch
});
export function memoryStore(authority=()=>true) {
  let state={sessions:{},activities:{},clocks:{}};
  return {read:async()=>structuredClone(state),write:async next=>{state=structuredClone(next)},isAuthority:authority};
}
const session={id:'S1',startedAt:-1200,budgetEndsAt:6000,goalsByPool:[],activityIds:[],status:'running',stopReason:null,assumptions:[]};
test('negative epochs, persisted unknown execution and exact state conflicts',async()=>{
  const store=memoryStore(),ledger=createLedger(store);
  await ledger.createSession(session); await ledger.insertActivity(activity());
  await ledger.transitionActivity('A1',{expected:['planned'],patch:{state:'started'}});
  await ledger.transitionActivity('A1',{expected:['started'],patch:{state:'uncertain',proof:{useId:'use1',checkIds:['c1'],resultIds:[],receiptIds:[],immunityIds:[]}}});
  await assert.rejects(ledger.transitionActivity('A1',{expected:['uncertain'],patch:{state:'planned'}}),/transition/);
  assert.equal((await createLedger(store).snapshot('S1')).activities[0].proof.useId,'use1');
  await assert.rejects(ledger.insertActivity(activity()),/duplicate/);
  assert.throws(()=>activity({endsAt:-1300}),/duration/);
});
test('claims serialize and authority is checked again after an awaited read',async()=>{
  let authority=true;
  const store=memoryStore(()=>authority),ledger=createLedger(store);
  await ledger.createSession(session); await ledger.insertActivity(activity());
  const claims=await Promise.allSettled([1,2].map(()=>ledger.transitionActivity('A1',{expected:['planned'],patch:{state:'started'}})));
  assert.equal(claims.filter(x=>x.status==='fulfilled').length,1);
  authority=false; await assert.rejects(ledger.updateSession('S1',{status:'stopped'}),/gm/);
  authority=true; const read=store.read;
  store.read=async()=>{const s=await read();authority=false;return s};
  await assert.rejects(createLedger(store).updateSession('S1',{status:'stopped'}),/gm/);
});
test('clock commits cannot retry uncertain or rewrite provenance',async()=>{
  const ledger=createLedger(memoryStore());await ledger.createSession(session);
  await ledger.upsertClockCommit({id:'C1',sessionId:'S1',from:-1200,to:-600,gmId:'U1',state:'started',evidence:[]});
  await ledger.transitionClockCommit('C1',{expected:['started'],patch:{state:'uncertain'}});
  await assert.rejects(ledger.upsertClockCommit({id:'C1',sessionId:'S1',from:-1200,to:-600,gmId:'U1',state:'started'}),/duplicate/);
  await assert.rejects(ledger.transitionClockCommit('C1',{expected:['uncertain'],patch:{state:'started'}}),/transition/);
  await assert.rejects(ledger.transitionActivity('missing',{expected:['planned'],patch:{state:'started'}}),/conflict/);
});
test('document storage stays readonly until a root is explicitly configured',async()=>{
  let created=0;
  const game={user:{id:'GM',isGM:true},users:{activeGM:{id:'GM'}},settings:{get:()=>''}};
  const store=createDocumentStore({game,JournalEntry:{create:async()=>{created++}}});
  assert.deepEqual(await store.read(),{sessions:{},activities:{},clocks:{}});
  assert.equal(store.write,undefined);
  await assert.rejects(store.transact(state=>{state.sessions.S1=session}),/root-not-configured/);
  assert.equal(created,0);
});
