import test from 'node:test';
import assert from 'node:assert/strict';
import {createRoaringSource,reduceRoaringSource,projectRoaringConditions} from '../scripts/roaring-lifecycle.mjs';

const sourceId='Compendium.pf2e.spells-srd.Item.czO0wbT1i320gcu9';
const clone=value=>structuredClone(value);
function turn(round=4,index=1,lastTurnEnd=round-1){return {combatId:'combat',combatantId:'caster',actorUuid:'Actor.caster',tokenUuid:'Scene.scene.Token.caster',started:true,round,turn:index,lastTurnEnd,latestTurnEndRound:lastTurnEnd,order:[{id:'before',initiative:20,overridePriority:null},{id:'caster',initiative:20,overridePriority:null},{id:'after',initiative:20,overridePriority:null}]}}
function envelope(value=100){return {start:{value,initiative:20},duration:{value:1,unit:'rounds',expiry:'turn-end',sustained:false}}}
function input(){return {sourceNonce:'source-one',castNonce:'cast-one',sourceId,itemUuid:'Actor.caster.Item.spell',entryUuid:'Actor.caster.Item.entry',rank:3,casterActorUuid:'Actor.caster',casterTokenUuid:'Scene.scene.Token.caster',targetActorUuid:'Actor.target',targetTokenUuid:'Scene.scene.Token.target',originalMessageUuid:'ChatMessage.cast',completedWorldTime:100,turn:turn(),finiteEnvelope:envelope()}}
function event(type,data={},worldTime=100,frame=turn()){return {type,sourceNonce:'source-one',observation:{worldTime,turn:frame},...data}}
const run=(s,e)=>reduceRoaringSource(s,e);
function active(outcome='criticalFailure'){return run(createRoaringSource(input()),event('save-confirmed',{revision:1,receiptId:'save-one',outcome})).source}
function use(s,nonce='use-one',worldTime=106){return run(s,event('sustain-use',{useNonce:nonce,userId:'owner',messageUuid:'ChatMessage.'+nonce,invocationId:nonce,turn:turn(5),finiteEnvelope:envelope(worldTime)},worldTime,turn(5))).source}
function verdict(s,nonce='use-one',terminal='completed',worldTime=106){return run(s,event('sustain-settled',{useNonce:nonce,verdictId:'verdict-'+nonce,gmId:'gm',terminal},worldTime,turn(5)))}
const kinds=result=>result.commands.map(c=>c.type);
function frozen(value){if(value&&typeof value==='object'){Object.freeze(value);for(const child of Object.values(value))frozen(child)}return value}

test('own-turn cast survives this end and next start, then ends once at actual next caster end',()=>{
  let s=active();assert.equal(s.timing.deadline.endRound,5);
  s=run(s,event('turn-end',{},100,turn(4,2,4))).source;assert.equal(s.status,'active');
  s=run(s,event('reconcile',{},106,turn(5,1,4))).source;assert.equal(s.status,'active');
  const end=run(s,event('turn-end',{},106,turn(5,2,5)));assert.equal(end.source.status,'ended');assert.deepEqual(kinds(end),['source-end']);
  assert.deepEqual(run(end.source,event('turn-end',{},106,turn(5,2,5))).commands,[]);
});

test('a pending native Sustain Use does not renew without a GM terminal fact',()=>{
  const s=use(active());assert.equal(s.timing.deadline.endRound,5);assert.equal(s.sustainUses['use-one'].status,'use-recorded');
  assert.equal(run(s,event('turn-end',{},106,turn(5,2,5))).source.status,'ended');
});

test('two completed Sustains in one turn choose the same next end and preserve the first finite envelope',()=>{
  let s=verdict(use(active())).source;assert.equal(s.timing.deadline.endRound,6);
  s=verdict(use(s,'use-two',107),'use-two','completed',107).source;
  assert.equal(s.timing.deadline.endRound,6);assert.equal(s.timing.finiteEnvelope.start.value,106);
  assert.equal(run(s,event('turn-end',{},107,turn(5,2,5))).source.status,'active');
  assert.equal(run(s,event('turn-end',{},112,turn(6,2,6))).source.status,'ended');
});

test('duplicate exact Use and GM verdict produce no commands and conflicting nonce payload is rejected',()=>{
  const first=use(active());const dupe=run(first,event('sustain-use',{useNonce:'use-one',userId:'owner',messageUuid:'ChatMessage.use-one',invocationId:'use-one',turn:turn(5),finiteEnvelope:envelope(106)},106,turn(5)));
  assert.deepEqual(dupe.commands,[]);assert.equal(dupe.decision,'duplicate');
  const done=verdict(first).source;assert.equal(verdict(done).decision,'duplicate');assert.deepEqual(verdict(done).commands,[]);
  const conflict=run(first,event('sustain-use',{useNonce:'use-one',userId:'owner',messageUuid:'ChatMessage.other',invocationId:'use-one',turn:turn(5),finiteEnvelope:envelope(106)},106,turn(5)));
  assert.equal(conflict.decision,'rejected');assert.equal(conflict.source.sustainUses['use-one'].messageUuid,'ChatMessage.use-one');
});

test('disrupted Sustain ends only this source with no payment or refund command',()=>{
  const result=verdict(use(active()),'use-one','disrupted');assert.equal(result.source.status,'ended');assert.deepEqual(kinds(result),['source-end']);
  assert.equal(result.commands[0].sourceNonce,'source-one');assert.equal(result.source.termination.reason,'sustain-disrupted');
});

test('GM conflict cannot replace the first terminal verdict or silently end a completed Use',()=>{
  const s=verdict(use(active())).source;const result=verdict(s,'use-one','disrupted');
  assert.equal(result.source.status,'active');assert.equal(result.source.sustainUses['use-one'].status,'completed');assert.equal(result.decision,'manual');
  assert.ok(kinds(result).includes('manual-review'));assert.equal(result.source.timing.deadline.endRound,6);
});

test('late GM completion and late save do not revive an expired source',()=>{
  let s=use(active());s=run(s,event('turn-end',{},106,turn(5,2,5))).source;
  assert.equal(verdict(s).source.status,'ended');assert.deepEqual(verdict(s).commands,[]);
  const waiting=run(createRoaringSource(input()),event('turn-end',{},106,turn(5,2,5))).source;
  const saved=run(waiting,event('save-confirmed',{revision:1,receiptId:'save-one',outcome:'failure'},106,turn(5,2,5)));
  assert.equal(saved.source.status,'ended');assert.deepEqual(saved.commands,[]);
});

test('a fresh observation detects the end before processing an otherwise valid GM verdict',()=>{
  const s=use(active());const r=run(s,event('sustain-settled',{useNonce:'use-one',verdictId:'v',gmId:'gm',terminal:'completed'},106,turn(5,2,5)));
  assert.equal(r.source.status,'ended');assert.equal(r.source.timing.deadline.endRound,5);assert.deepEqual(kinds(r),['source-end']);
});

test('equal initiatives and unrelated viewed-combat metadata never substitute for the caster end receipt',()=>{
  const r=run(active(),event('turn-end',{combatId:'viewed-other',combatantId:'before',endedRound:5},106,{...turn(5,1,4),viewedCombatId:'viewed-other',latestTurnEndRound:5}));
  assert.equal(r.source.status,'active');assert.deepEqual(r.commands,[]);
});

test('600 game seconds is an immutable upper bound even when a pending Use would renew',()=>{
  const s=use(active(),'use-one',699);assert.equal(s.hardStopAt,700);
  const r=verdict(s,'use-one','completed',700);assert.equal(r.source.status,'ended');assert.equal(r.source.termination.reason,'maximum-duration');
  assert.equal(r.source.hardStopAt,700);assert.equal(r.source.timing.deadline.endRound,5);
});

test('clock-only forward jumps expire once and never use real time',()=>{
  const s=active();const a=run(s,event('clock',{},100,undefined));assert.equal(a.source.status,'active');
  const r=run(s,{type:'clock',sourceNonce:s.sourceNonce,observation:{worldTime:701}});assert.equal(r.source.status,'ended');assert.deepEqual(kinds(r),['source-end']);
  assert.deepEqual(run(r.source,{type:'clock',sourceNonce:s.sourceNonce,observation:{worldTime:100}}).commands,[]);
});

test('clock rewind preserves cap and original finite fallback, forbids renewal and does not auto-rearm',()=>{
  const r=run(active(),event('clock',{},99));assert.equal(r.source.timing.mode,'manual-finite');assert.equal(r.source.hardStopAt,700);assert.equal(r.source.clockHighWater,100);
  assert.deepEqual(r.commands.find(c=>c.type==='restore-finite').finiteEnvelope,envelope());
  const again=run(r.source,event('reconcile',{},100));assert.equal(again.source.timing.mode,'manual-finite');assert.deepEqual(again.commands,[]);
  const refused=run(again.source,event('sustain-use',{useNonce:'later',userId:'owner',messageUuid:'ChatMessage.later',invocationId:'later',turn:turn(),finiteEnvelope:envelope()},100));
  assert.equal(refused.decision,'rejected');assert.equal(refused.source.timing.deadline.endRound,5);
});

for(const [name,change]of [
  ['order',f=>{f.order.reverse();return f}],
  ['initiative',f=>{f.order[1].initiative=25;return f}],
  ['priority',f=>{f.order[1].overridePriority=3;return f}],
  ['missing caster token/combatant',()=>null],
  ['ended encounter',f=>({...f,started:false})],
  ['wrong caster identity',f=>({...f,actorUuid:'Actor.other'})],
  ['round rollback',f=>({...f,round:3,lastTurnEnd:2})],
  ['old-ended record in the future',f=>({...f,lastTurnEnd:9})],
])test(`${name} restores the frozen finite envelope once without a new relative duration`,()=>{
  const frame=change(turn());const r=run(active(),event('reconcile',{},102,frame));
  assert.equal(r.source.timing.mode,'manual-finite');assert.deepEqual(r.source.timing.finiteEnvelope,envelope());
  assert.deepEqual(r.commands.find(c=>c.type==='restore-finite').finiteEnvelope,envelope());
  assert.deepEqual(run(r.source,event('reconcile',{},103,frame)).commands,[]);
});

test('fallback after an accepted Sustain uses that frozen envelope, not cast time or failure time',()=>{
  const s=verdict(use(active())).source;const r=run(s,event('reconcile',{},109,null));
  assert.equal(r.source.timing.finiteEnvelope.start.value,106);assert.equal(r.source.hardStopAt,700);
  assert.equal(r.commands.find(c=>c.type==='restore-finite').finiteEnvelope.start.value,106);
});

test('finite fallback already elapsed ends rather than creating a fresh round',()=>{
  const r=run(active(),event('reconcile',{},107,null));assert.equal(r.source.status,'ended');assert.deepEqual(kinds(r),['source-end']);
});

test('a later real end proves a skipped source end, but a UI round change alone does not',()=>{
  const noProof=run(active(),event('reconcile',{},106,turn(6,0,4)));assert.equal(noProof.source.status,'active');
  const proof=run(active(),event('reconcile',{},106,{...turn(6,0,4),latestTurnEndRound:6}));assert.equal(proof.source.timing.mode,'manual-finite');
});

test('turn rollback is measured from the last valid observation, not only original cast',()=>{
  const s=run(active(),event('reconcile',{},106,turn(5,2,4))).source;
  const r=run(s,event('reconcile',{},106,turn(5,1,4)));assert.equal(r.source.timing.mode,'manual-finite');
});

test('source death/incapacity does not clear existing effects or move their deadline',()=>{
  const r=run(active(),event('reconcile',{},102,{...turn(),isDead:true,canAct:false}));assert.equal(r.source.status,'active');assert.equal(r.source.timing.deadline.endRound,5);assert.deepEqual(r.commands,[]);
});

for(const [outcome,want]of [
  ['criticalSuccess',{noReactions:false,slowed:0,fascinated:false,clap:false}],
  ['success',{noReactions:true,slowed:0,fascinated:false,clap:false}],
  ['failure',{noReactions:true,slowed:1,fascinated:false,clap:true}],
  ['criticalFailure',{noReactions:true,slowed:1,fascinated:true,clap:true}],
])test(`native ${outcome} maps only this source's condition needs`,()=>{
  const s=active(outcome),p=projectRoaringConditions(s);for(const [key,value]of Object.entries(want))assert.equal(p[key],value,key);
  assert.equal(p.sourceNonce,'source-one');if(want.fascinated)assert.deepEqual(p.subject,{actorUuid:'Actor.caster',tokenUuid:'Scene.scene.Token.caster'});
});

test('duplicate save is a no-op and every later revision stays manual with original evidence',()=>{
  const s=active('failure');const duplicate=run(s,event('save-confirmed',{revision:1,receiptId:'save-one',outcome:'failure'}));assert.equal(duplicate.decision,'duplicate');assert.deepEqual(duplicate.commands,[]);
  const r=run(s,event('save-confirmed',{revision:2,receiptId:'reroll',outcome:'criticalSuccess'}));assert.equal(r.decision,'manual');assert.equal(r.source.result.outcome,'failure');assert.equal(projectRoaringConditions(r.source).manualReview,true);assert.equal(projectRoaringConditions(r.source).slowed,1);
});

test('unverified result before the first settlement prevents a late stale row from becoming active',()=>{
  const r=run(createRoaringSource(input()),event('save-unverified',{receiptId:'manual-edit',reason:'unverified-row'}));
  const late=run(r.source,event('save-confirmed',{revision:1,receiptId:'save-one',outcome:'failure'}));assert.equal(late.source.status,'awaiting-save');assert.equal(late.decision,'manual');assert.ok(!kinds(late).includes('condition-sync'));
});

test('manual own Fascinated deletion stays suppressed through Sustain, JSON reload, and later revision',()=>{
  let s=run(active(),event('own-child-deleted',{condition:'fascinated',itemUuid:'Actor.target.Item.fascinated',receiptId:'manual-delete'})).source;
  s=verdict(use(JSON.parse(JSON.stringify(s)))).source;s=run(s,event('save-confirmed',{revision:2,receiptId:'reroll',outcome:'criticalFailure'},106,turn(5))).source;
  const p=projectRoaringConditions(s);assert.equal(p.fascinated,false);assert.equal(p.slowed,1);assert.equal(p.noReactions,true);assert.equal(p.clap,true);
  assert.equal(s.tombstones.fascinated.receiptId,'manual-delete');
});

test('GM fascination end is durable and emits only source-specific removal once',()=>{
  const e=event('end-fascination',{receiptId:'hostile-fact',gmId:'gm'});const r=run(active(),e);assert.deepEqual(kinds(r),['end-fascination']);assert.equal(r.commands[0].sourceNonce,'source-one');
  assert.equal(projectRoaringConditions(r.source).fascinated,false);assert.deepEqual(run(r.source,e).commands,[]);
});

test('confirmed own Slowed removal is not repaired by Sustain',()=>{
  const s=run(active(),event('own-child-deleted',{condition:'slowed',itemUuid:'Actor.target.Item.slowed',receiptId:'slowed-delete'})).source;
  assert.equal(projectRoaringConditions(verdict(use(s)).source).slowed,0);
});

test('manual parent deletion terminates the source and no later action recreates it',()=>{
  const s=run(active(),event('own-parent-deleted',{itemUuid:'Actor.target.Item.parent',receiptId:'parent-delete'})).source;assert.equal(s.status,'ended');assert.equal(projectRoaringConditions(s).noReactions,false);
  assert.deepEqual(run(s,event('save-confirmed',{revision:1,receiptId:'save-one',outcome:'criticalFailure'})).commands,[]);
});

test('different Cast sources remain isolated and an event for another source is rejected',()=>{
  const a=active();const b=createRoaringSource({...input(),sourceNonce:'source-two',castNonce:'cast-two',originalMessageUuid:'ChatMessage.second'});
  assert.throws(()=>run(a,{...event('end-fascination',{receiptId:'x',gmId:'gm'}),sourceNonce:b.sourceNonce}),/source|来源/i);
  assert.equal(projectRoaringConditions(a).fascinated,true);assert.equal(b.status,'awaiting-save');
});

test('functions do not mutate input or retain caller-owned nested objects',()=>{
  const raw=input(),copy=clone(raw);const s=createRoaringSource(frozen(raw));assert.deepEqual(raw,copy);
  const before=clone(s),e=frozen(event('save-confirmed',{revision:1,receiptId:'save-one',outcome:'failure'}));const r=run(frozen(s),e);
  assert.deepEqual(s,before);r.source.timing.finiteEnvelope.start.value=999;assert.equal(s.timing.finiteEnvelope.start.value,100);
  const p=projectRoaringConditions(active());p.subject.actorUuid='changed';assert.equal(projectRoaringConditions(active()).subject.actorUuid,'Actor.caster');
});

test('unsupported entry scope, unsafe nonce and stale native end baseline are refused',()=>{
  for(const change of [
    {sourceNonce:'unsafe.key'},{sourceId:'other'},{rank:6},{turn:turn(4,0)},
    {turn:{...turn(),started:false}},{turn:{...turn(),lastTurnEnd:4}},
    {turn:{...turn(),tokenUuid:'Scene.other.Token.caster'}},{turn:{...turn(),actorUuid:'Actor.other'}},
    {completedWorldTime:NaN},{finiteEnvelope:envelope(101)},
    {finiteEnvelope:{start:{value:100,initiative:20},duration:{value:-1,unit:'unlimited',expiry:null}}},
  ])assert.throws(()=>createRoaringSource({...input(),...change}));
});

test('unproven or malformed event shape cannot turn into a completed Use',()=>{
  const s=active();assert.equal(run(s,event('sustain-settled',{useNonce:'missing',verdictId:'v',gmId:'gm',terminal:'completed'})).decision,'rejected');
  for(const e of [
    event('sustain-use',{useNonce:'u',userId:'owner',messageUuid:'',invocationId:'u',turn:turn(),finiteEnvelope:envelope()}),
    event('sustain-settled',{useNonce:'u',verdictId:'v',gmId:'',terminal:'completed'}),
    event('save-confirmed',{revision:1,receiptId:'save',outcome:'invalid'}),
  ])assert.throws(()=>run(s,e));
});

test('ending a combat with native round zero degrades instead of throwing and leaking an unlimited source',()=>{
  const r=run(active(),event('reconcile',{},102,{...turn(),started:false,round:0,turn:null,order:[]}));
  assert.equal(r.source.timing.mode,'manual-finite');assert.deepEqual(r.source.timing.finiteEnvelope,envelope());assert.ok(kinds(r).includes('restore-finite'));
});

test('a save cannot first materialize new conditions after timing became manual',()=>{
  const source=run(createRoaringSource(input()),event('reconcile',{},102,null)).source;
  const r=run(source,event('save-confirmed',{revision:1,receiptId:'save-one',outcome:'failure'},103,null));
  assert.equal(r.decision,'manual');assert.equal(r.source.status,'awaiting-save');assert.ok(!kinds(r).includes('condition-sync'));
});

test('an exact GM disruption may settle a recorded Use after its own turn while the source is still alive',()=>{
  let s=run(active(),event('sustain-use',{useNonce:'early',userId:'owner',messageUuid:'ChatMessage.early',invocationId:'early',turn:turn(),finiteEnvelope:envelope()})).source;
  const r=run(s,event('sustain-settled',{useNonce:'early',verdictId:'after-turn',gmId:'gm',terminal:'disrupted'},102,turn(4,2,4)));
  assert.equal(r.source.status,'ended');assert.equal(r.source.termination.reason,'sustain-disrupted');
});

test('delayed completed verdict never bases a new deadline on the later adjudication turn',()=>{
  let s=run(active(),event('sustain-use',{useNonce:'early',userId:'owner',messageUuid:'ChatMessage.early',invocationId:'early',turn:turn(),finiteEnvelope:envelope()})).source;
  const r=run(s,event('sustain-settled',{useNonce:'early',verdictId:'after-turn',gmId:'gm',terminal:'completed'},102,turn(4,2,4)));
  assert.equal(r.source.sustainUses.early.status,'completed');assert.equal(r.source.timing.deadline.endRound,5);assert.ok(!kinds(r).includes('renew-source'));
});

test('a nonfinite or numerically ineffective maximum duration cannot enter the source ledger',()=>{
  assert.throws(()=>createRoaringSource({...input(),completedWorldTime:Number.MAX_VALUE,finiteEnvelope:envelope(Number.MAX_VALUE)}));
});

test('manual timing remains explicitly labeled on the retained last-confirmed projection',()=>{
  const s=run(active(),event('reconcile',{},102,null)).source,p=projectRoaringConditions(s);
  assert.equal(p.manualReview,true);assert.equal(p.noReactions,true);assert.equal(p.slowed,1);
});

test('known disruption can end a pending exact Use even after source structure fell back to finite timing',()=>{
  const pending=run(active(),event('sustain-use',{useNonce:'early',userId:'owner',messageUuid:'ChatMessage.early',invocationId:'early',turn:turn(),finiteEnvelope:envelope()})).source;
  const degraded=run(pending,event('reconcile',{},102,null)).source;
  const r=run(degraded,event('sustain-settled',{useNonce:'early',verdictId:'disrupted',gmId:'gm',terminal:'disrupted'},102,null));
  assert.equal(r.source.status,'ended');assert.equal(r.source.termination.reason,'sustain-disrupted');assert.deepEqual(kinds(r),['source-end']);
});
