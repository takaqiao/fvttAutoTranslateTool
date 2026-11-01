import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createRecoverySimulation} from '../../scripts/exploration/computed-plan.mjs';
import {treatmentOutcomeRows} from '../../scripts/exploration/treatment-context.mjs';

function treatment({rank=1,modifier=10,assurance=false,riskySurgery=false,continualRecovery=true,capacity=1,medic=false,skill='medicine'}={}){
 return treatmentOutcomeRows({rank,modifier,assurance}).map((row,index)=>({ready:true,skill,rank:row.rank,dc:row.dc,modifier,assurance,riskySurgery,outcomes:row.cases.map(c=>riskySurgery&&c.outcome===2?3:c.outcome),bonus:[0,10,30,50][index]+(medic?[0,5,10,15][index]:0),medicBonus:medic?[0,5,10,15][index]:0,durationSeconds:600,cooldownSeconds:continualRecovery?600:3600,capacity}));
}
function actor(id,options={}){
 const actorUUID=`Actor.${id}`;
 return {actorUUID,name:id,modeOfBeing:'living',isDead:false,unconscious:false,canAct:true,wounded:false,slugs:[],hp:{value:40,max:40,temp:0},focus:{value:0,max:0},pool:{poolUUID:actorUUID,ready:true},cooldownExpiresAt:null,computed:{version:1,treatments:[],patient:{healingReady:true,damageReady:true,vitalityReady:true},focusHealing:{ready:false,reason:'focus-healing-unavailable'},refocus:{ready:true,recovery:1},threePecks:{ready:false}},...options};
}
function session(actors,options={}){
 return {id:'session',actorUUIDs:actors.map(a=>a.actorUUID),startedAt:0,budgetEndsAt:7200,recoveryStrategy:'success-first',goalsByPool:[...new Map(actors.map(a=>[a.pool.poolUUID,{poolUUID:a.pool.poolUUID,targetHP:a.hp.max}])).values()],recoveryGoals:{requireNoWounded:false,failureStop:{enabled:false,limit:3}},...options};
}
function healer(id='H',options={}){const h=actor(id);h.computed.treatments=treatment(options);return h}
function focusHealer(id='F',amount=12){const h=actor(id,{focus:{value:1,max:1}});h.computed.focusHealing={ready:true,itemUUID:`Actor.${id}.Item.L`,healing:{constant:amount,dice:[]},durationSeconds:6,focusCost:1};return h}
const dice=(...faces)=>{let i=0;return ()=>{assert.ok(i<faces.length,'unexpected random sample');const [face,sides]=faces[i++];return (face-0.5)/sides}};
const hp=(result,id)=>result.actors.find(a=>a.actorUUID===`Actor.${id}`).hp.value;

test('next reserves an event boundary without sampling; settle samples the d20 and healing dice once',()=>{
 const actors=[healer(),actor('P',{hp:{value:10,max:40,temp:0}})];let count=0;
 const simulation=createRecoverySimulation({actors,session:session(actors),random:()=>{count++;return 0.5}});
 const first=simulation.next();assert.equal(first.to,600);assert.equal(count,0);assert.deepEqual(simulation.next(),first);
 const batch=simulation.settle();assert.equal(count,3);assert.equal(batch.events[0].check.face,11);assert.equal(batch.events[0].healing,10);assert.equal(hp(simulation.snapshot(),'P'),20);
 assert.throws(()=>simulation.settle(),/no-pending/);assert.equal(count,3);
});

test('success-first selects sufficient Assurance but never treats insufficient Assurance as certain',()=>{
 for(const proficiency of [4,8]){
  const h=healer();h.computed.treatments.push(...treatment({assurance:true,modifier:proficiency}));const actors=[h,actor('P',{hp:{value:10,max:40,temp:0}})];
  const simulation=createRecoverySimulation({actors,session:session(actors),random:()=>0.5});const activity=simulation.next().activities[0];
  assert.equal(activity.options.assurance,proficiency>=5);simulation.settle();assert.equal(simulation.snapshot().history[0].outcome,2);
 }
});

test('the four outcomes retain natural 1 and 20 shifts and actual damage or healing',()=>{
 const expected=[[1,0,-8],[5,1,0],[10,2,16],[20,3,32]];
 for(const [face,outcome,delta] of expected){
  const actors=[healer('H',{modifier:5}),actor('P',{hp:{value:50,max:100,temp:0}})];
  const simulation=createRecoverySimulation({actors,session:session(actors),random:dice([face,20],...Array.from({length:4},()=>[8,8]))});
  simulation.next();const batch=simulation.settle();assert.equal(batch.events[0].outcome,outcome);assert.equal(hp(simulation.snapshot(),'P'),50+delta);
 }
});

test('Risky Surgery damage precedes healing and Medic adds the selected tier bonus',()=>{
 const actors=[healer('H',{rank:2,modifier:20,riskySurgery:true,medic:true}),actor('P',{hp:{value:80,max:100,temp:0}})];
 const simulation=createRecoverySimulation({actors,session:session(actors,{treatmentRank:'expert',riskySurgery:true}),random:dice([8,8],[10,20],[1,8],[1,8],[1,8],[1,8])});
 simulation.next();const {events}=simulation.settle();assert.equal(events[0].damage,8);assert.equal(events[0].healing,19);assert.equal(hp(simulation.snapshot(),'P'),91);assert.deepEqual(events[0].changes.map(c=>c.kind),['damage','healing']);
});

test('Ward Medic capacity uses separate checks, healing rolls and immunity per patient',()=>{
 const actors=[healer('H',{capacity:2}),...['A','B','C'].map(id=>actor(id,{hp:{value:10,max:40,temp:0}}))];
 const simulation=createRecoverySimulation({actors,session:session(actors),random:dice([10,20],[1,8],[2,8],[5,20],[3,8],[4,8])});
 assert.equal(simulation.next().activities[0].patientUUIDs.length,2);const batch=simulation.settle();assert.equal(batch.events.length,2);assert.deepEqual(batch.events.map(e=>e.healing),[3,7]);assert.deepEqual(batch.events.map(e=>e.cooldownExpiresAt),[600,600]);assert.equal(hp(simulation.snapshot(),'C'),10);
});

test('shared pools receive HP once while members can independently Refocus',()=>{
 const h=healer('H',{capacity:3}),a=actor('A',{hp:{value:10,max:40,temp:0},focus:{value:0,max:1}}),b=actor('B',{hp:{value:10,max:40,temp:0},focus:{value:0,max:1},pool:{poolUUID:'Actor.A',ready:true}}),actors=[h,a,b];
 const simulation=createRecoverySimulation({actors,session:session(actors,{requireFullFocus:true}),random:()=>0.5});const next=simulation.next();
 assert.equal(next.activities.filter(a=>a.providerId==='refocus').length,2);assert.equal(next.activities.find(a=>a.providerId==='treat-wounds').patientUUIDs.length,1);
 const batch=simulation.settle();assert.equal(batch.events.filter(e=>e.healing>0).length,1);assert.equal(hp(simulation.snapshot(),'A'),20);assert.equal(hp(simulation.snapshot(),'B'),20);assert.deepEqual(simulation.snapshot().actors.slice(1).map(a=>a.focus.value),[1,1]);
});

test('cooldown starts at treatment start, including the ordinary one-hour interval',()=>{
 for(const continualRecovery of [false,true]){
  const actors=[healer('H',{continualRecovery,assurance:true,modifier:8}),actor('P',{hp:{value:10,max:40,temp:0}})];
  const simulation=createRecoverySimulation({actors,session:session(actors),random:()=>0});simulation.next();const batch=simulation.settle();assert.equal(batch.events[0].cooldownExpiresAt,continualRecovery?600:3600);
  const next=simulation.next();assert.equal(next.to,continualRecovery?1200:3600);if(!continualRecovery){assert.deepEqual(next.activities,[]);simulation.settle();assert.equal(simulation.next().to,4200)}
 }
});

test('Focus is consumed before another cast and ordinary Refocus recovers one per ten minutes',()=>{
 const h=focusHealer();h.focus={value:1,max:3};const actors=[h,actor('P',{hp:{value:10,max:34,temp:0}})];
 const simulation=createRecoverySimulation({actors,session:session(actors,{requireFullFocus:true}),random:()=>{throw Error('fixed healing should not roll')}});const result=simulation.run();
 assert.equal(hp(result,'P'),34);assert.equal(result.actors[0].focus.value,3);assert.equal(result.cursorAt,2412);assert.deepEqual(result.history.filter(e=>e.providerId==='focus-healing').map(e=>[e.focusBefore,e.focusAfter]),[[1,0],[1,0]]);
});

test('successful Treat Wounds removes wounded and ten minutes at full HP can remove it without a healer',()=>{
 const actors=[actor('P',{wounded:true})];const simulation=createRecoverySimulation({actors,session:session(actors,{recoveryGoals:{requireNoWounded:true}})});
 assert.equal(simulation.next().activities[0].providerId,'rest');simulation.settle();assert.equal(simulation.snapshot().actors[0].wounded,false);assert.equal(simulation.next().reason,'goals-met');
 const h=healer('H',{assurance:true,modifier:8}),p=actor('P',{wounded:true,hp:{value:20,max:40,temp:0}});const recovery=createRecoverySimulation({actors:[h,p],session:session([h,p]),random:()=>0});recovery.next();recovery.settle();assert.equal(recovery.snapshot().actors[1].wounded,false);
});

test('budget, failure limit, already satisfied goals and unsupported patients stop explicitly',()=>{
 const h=healer('H',{modifier:0}),p=actor('P',{hp:{value:20,max:40,temp:0}}),actors=[h,p];
 assert.equal(createRecoverySimulation({actors,session:session(actors,{budgetEndsAt:599})}).next().reason,'budget');
 assert.equal(createRecoverySimulation({actors:[h],session:session([h])}).next().reason,'goals-met');
 const blocked=structuredClone(actors);blocked[1].computed.patient.healingReady=false;blocked[1].computed.patient.healingReason='healing-received-modifier-unverified';const sim=createRecoverySimulation({actors:blocked,session:session(blocked)});assert.equal(sim.next().reason,'unreachable');assert.ok(sim.snapshot().diagnostics.includes('healing-received-modifier-unverified'));
 const failure=createRecoverySimulation({actors,session:session(actors,{recoveryGoals:{failureStop:{enabled:true,limit:2}}}),random:()=>0.5});const result=failure.run();assert.equal(result.stopReason,'failure-stop');assert.equal(result.failures['Actor.P'],2);assert.equal(result.cursorAt,1200);
});

test('dropping to zero stops that pool with needs-review before any subsequent healing or activity',()=>{
 const actors=[healer('H',{riskySurgery:true}),actor('P',{hp:{value:3,max:40,temp:0}})];const simulation=createRecoverySimulation({actors,session:session(actors,{riskySurgery:true}),random:dice([8,8])});
 simulation.next();const batch=simulation.settle();assert.equal(batch.events[0].needsReview,true);assert.equal(batch.events[0].healing,0);assert.equal(hp(simulation.snapshot(),'P'),0);assert.equal(simulation.next().reason,'needs-review');
});

test('independent healers work in parallel and choose distinct HP pools',()=>{
 const actors=[healer('H1'),healer('H2'),actor('A',{hp:{value:10,max:40,temp:0}}),actor('B',{hp:{value:10,max:40,temp:0}})];const simulation=createRecoverySimulation({actors,session:session(actors),random:()=>0.5});
 const activities=simulation.next().activities;assert.equal(activities.length,2);assert.equal(new Set(activities.flatMap(a=>a.hpPoolUUIDs)).size,2);simulation.settle();assert.equal(hp(simulation.snapshot(),'A'),20);assert.equal(hp(simulation.snapshot(),'B'),20);
});

test('success-first and time-first differ when the slower certain check needs repeated cooldowns',()=>{
 const h=healer('H',{rank:2,modifier:14,continualRecovery:false});h.computed.treatments.push(...treatment({rank:2,modifier:8,assurance:true,continualRecovery:false}));const actors=[h,actor('P',{hp:{value:10,max:70,temp:0}})];
 const certain=createRecoverySimulation({actors,session:session(actors,{recoveryStrategy:'success-first',treatmentRank:'auto'})}).next().activities[0];
 const faster=createRecoverySimulation({actors,session:session(actors,{recoveryStrategy:'time-first',treatmentRank:'auto'})}).next().activities[0];
 assert.equal(certain.options.assurance,true);assert.equal(certain.options.rank,'trained');assert.equal(faster.options.assurance,false);assert.equal(faster.options.rank,'expert');
 const run=recoveryStrategy=>createRecoverySimulation({actors,session:session(actors,{recoveryStrategy,treatmentRank:'auto',budgetEndsAt:21600}),random:()=>0.5}).run();
 const sure=run('success-first'),fast=run('time-first');assert.equal(sure.stopReason,'goals-met');assert.equal(fast.stopReason,'goals-met');assert.equal(sure.cursorAt,18600);assert.equal(fast.cursorAt,7800);
});

test('settlement re-reads prepared actors after time effects and resumed pending state does not resample planning',()=>{
 const actors=[healer('H',{assurance:true,modifier:8}),actor('P',{hp:{value:10,max:40,temp:0}})];const simulation=createRecoverySimulation({actors,session:session(actors)});const next=simulation.next();
 const saved=JSON.parse(JSON.stringify(simulation.snapshot())),restored=createRecoverySimulation({actors,session:session(actors),state:saved,random:()=>0});assert.deepEqual(restored.next(),next);
 const fresh=structuredClone(actors);fresh[1].hp.value=30;const batch=restored.settle({actors:fresh});assert.equal(batch.beforeActors[1].hp.value,30);assert.equal(batch.afterActors[1].hp.value,32);assert.equal(restored.snapshot().initialActors[1].hp.value,10);
});

test('Three Pecks uses its own occultism model and restores Focus even when it began full',()=>{
 const h=actor('H',{focus:{value:1,max:1}});h.computed.threePecks={ready:true,treatments:treatment({skill:'occultism',assurance:true,modifier:8,continualRecovery:false})};const actors=[h,actor('P',{hp:{value:10,max:40,temp:0}})];const sim=createRecoverySimulation({actors,session:session(actors),random:()=>0.5});
 const next=sim.next();assert.equal(next.activities[0].providerId,'refocus');assert.equal(next.activities[0].options.threePecks,true);assert.equal(next.activities[0].options.skill,'occultism');const batch=sim.settle();assert.equal(batch.events.find(e=>e.patientUUID).healing,10);assert.equal(sim.snapshot().actors[0].focus.value,1);
});

test('an earlier effect boundary retains activities and rolls only at their actual end',()=>{
 const actors=[healer(),actor('P',{hp:{value:10,max:40,temp:0}})];let calls=0;
 const sim=createRecoverySimulation({actors,session:session(actors),random:()=>{calls++;return 0.5}});
 assert.equal(sim.next({until:60}).to,60);assert.equal(sim.settle().events.length,0);assert.equal(calls,0);assert.equal(sim.snapshot().active.length,1);
 assert.equal(sim.next().to,600);assert.equal(sim.next().activities.length,0);sim.settle();assert.equal(calls,3);
});

test('unknown immunity remains blocked through JSON snapshot round trips',()=>{
 const actors=[healer(),actor('P',{hp:{value:10,max:40,temp:0},cooldownExpiresAt:Infinity,hasActiveToken:undefined})];const sim=createRecoverySimulation({actors,session:session(actors)});
 assert.equal(sim.next().reason,'unreachable');const saved=JSON.parse(JSON.stringify(sim.snapshot()));assert.equal(saved.actors[1].cooldownUnknown,true);assert.equal(Object.hasOwn(saved.actors[1],'hasActiveToken'),false);
 const restored=createRecoverySimulation({actors,session:session(actors),state:saved});assert.equal(restored.next().reason,'unreachable');
});

test('resting must remain at full HP throughout intermediate effect boundaries',()=>{
 const actors=[actor('P',{wounded:true})],sim=createRecoverySimulation({actors,session:session(actors,{recoveryGoals:{requireNoWounded:true}})});
 sim.next({until:60});const fresh=structuredClone(actors);fresh[0].hp.value=39;sim.settle({actors:fresh});sim.next();const batch=sim.settle({actors});assert.equal(batch.afterActors[0].wounded,true);
});

test('a changed check model at the completion boundary supplies the actual outcome',()=>{
 const actors=[healer(),actor('P',{hp:{value:10,max:40,temp:0}})],sim=createRecoverySimulation({actors,session:session(actors),random:()=>0.5});sim.next();const fresh=structuredClone(actors);fresh[0].computed.treatments=treatment({modifier:0});const batch=sim.settle({actors:fresh});assert.equal(batch.events[0].outcome,1);assert.equal(batch.events[0].check.total,11);
});

test('Risky Surgery is unavailable unless the GM explicitly enables it',()=>{
 const h=healer();h.computed.treatments.push(...treatment({riskySurgery:true,modifier:12}));const actors=[h,actor('P',{hp:{value:10,max:40,temp:0}})];
 const sim=createRecoverySimulation({actors,session:session(actors,{recoveryStrategy:'time-first'})});assert.equal(sim.next().activities[0].options.riskySurgery,false);
});

test('activity limits do not overbook simultaneous work or abandon an already started activity',()=>{
 const actors=[healer('H1'),healer('H2'),actor('P',{hp:{value:10,max:40,temp:0}}),actor('Q',{hp:{value:10,max:40,temp:0}})];const sim=createRecoverySimulation({actors,session:session(actors,{maxActivities:1}),random:()=>0.5});
 assert.equal(sim.next({until:60}).activities.length,1);sim.settle();assert.equal(sim.next().to,600);sim.settle();assert.equal(sim.next().reason,'activity-limit');
});

test('Lay on Hands samples its prepared dice and grants its short AC effect only to another creature',()=>{
 for(const self of [false,true]){
  const h=focusHealer('H');h.computed.focusHealing.healing={constant:6,dice:[{number:1,faces:6}]};h.computed.focusHealing.effects=[{sourceUUID:'Compendium.pf2e.spell-effects.Item.lyLMiauxIVUM3oF1',durationSeconds:6,otherOnly:true}];
  const patient=self?h:actor('P');patient.hp={value:10,max:40,temp:0};const actors=self?[h]:[h,patient],sim=createRecoverySimulation({actors,session:session(actors),random:dice([2,6])});sim.next();const event=sim.settle().events[0];assert.equal(event.healing,8);assert.equal(event.effects?.length??0,self?0:1);if(!self)assert.equal(event.effects[0].expiresAt,12);
 }
});

test('an enabled treatment extension reuses the original healing after fifty more minutes without another roll or cooldown',()=>{
 const actors=[healer('H',{assurance:true,modifier:8,continualRecovery:false}),actor('P',{hp:{value:30,max:38,temp:0}})],sim=createRecoverySimulation({actors,session:session(actors,{extendTreatment:true}),random:dice([2,8],[2,8])});
 sim.next();const first=sim.settle().events[0];const next=sim.next();assert.equal(next.to,3600);assert.equal(next.activities[0].options.extensionOf,first.activityId);const second=sim.settle().events[0];assert.equal(second.healing,4);assert.equal(second.damage,0);assert.equal(second.check,undefined);assert.equal(second.cooldownExpiresAt,undefined);assert.equal(hp(sim.snapshot(),'P'),38);assert.equal(sim.next().reason,'goals-met');
});

test('resuming merges current prepared HP, Focus and eligibility while retaining prior cooldown and failure history',()=>{
 const actors=[healer('H',{assurance:true,modifier:8,continualRecovery:false}),actor('P',{hp:{value:10,max:40,temp:0}})],initial=createRecoverySimulation({actors,session:session(actors),random:()=>0});initial.next();initial.settle();
 const saved=initial.snapshot(),fresh=structuredClone(actors);fresh[1].hp.value=30;fresh[0].focus={value:0,max:1};fresh[0].computed.treatments=[];
 const resumed=createRecoverySimulation({actors:fresh,session:session(actors),state:saved});assert.equal(resumed.snapshot().actors[1].hp.value,30);assert.equal(resumed.snapshot().actors[0].focus.value,0);assert.equal(resumed.snapshot().actors[1].cooldownExpiresAt,3600);assert.deepEqual(resumed.snapshot().history,saved.history);assert.deepEqual(resumed.snapshot().failures,saved.failures);assert.equal(resumed.next().reason,'unreachable');
});

test('compact checkpoints omit capability copies and preserve only the immediate extension offer',()=>{
 const actors=[healer('H',{assurance:true,modifier:8,continualRecovery:false}),actor('P',{hp:{value:10,max:40,temp:0}})],sim=createRecoverySimulation({actors,session:session(actors,{extendTreatment:true}),random:()=>0});sim.next();sim.settle();const state=sim.checkpoint();
 assert.equal(state.history.length,1);assert.equal(state.initialActors,undefined);assert.ok(state.actors.every(a=>a.computed===undefined&&a.executionStamp===undefined));
 const resumed=createRecoverySimulation({actors:sim.snapshot().actors,session:session(actors,{extendTreatment:true}),state,random:()=>{throw Error('extension cannot roll')}});assert.equal(resumed.next().activities[0].options.extensionOf,state.history[0].activityId);resumed.settle();assert.equal(resumed.checkpoint().history.length,0);
});

test('a compact intermediate checkpoint resumes active work with newly prepared check models',()=>{
 const actors=[healer(),actor('P',{hp:{value:10,max:40,temp:0}})],sim=createRecoverySimulation({actors,session:session(actors)});sim.next({until:60});sim.settle();const state=sim.checkpoint();assert.equal(state.active[0].model.outcomes,undefined);
 const fresh=structuredClone(actors);fresh[0].computed.treatments=treatment({modifier:0});const resumed=createRecoverySimulation({actors:fresh,session:session(actors),state,random:()=>0.5});assert.equal(resumed.next().to,600);assert.equal(resumed.settle().events[0].outcome,1);
});

test('Robust Health adds its level to Treat Wounds and competes with the Medic circumstance bonus',()=>{
 for(const [rank,medic,level,expected] of [[1,false,5,10],[2,true,5,20],[2,true,8,23]]){
  const h=healer('H',{rank,medic,assurance:true,modifier:10}),p=actor('P',{hp:{value:10,max:60,temp:0}});p.computed.patient.robustHealth=level;const actors=[h,p],sim=createRecoverySimulation({actors,session:session(actors,{treatmentRank:rank===1?'trained':'expert'}),random:dice([2,8],[3,8])});sim.next();const event=sim.settle().events[0];assert.equal(event.healing,expected);assert.equal(event.hpStages[0].damageTaken,-expected);
 }
});

test('Robust Health never adds to Lay on Hands',()=>{
 const h=focusHealer(),p=actor('P',{hp:{value:10,max:40,temp:0}});p.computed.patient.robustHealth=5;const actors=[h,p],sim=createRecoverySimulation({actors,session:session(actors)});sim.next();const event=sim.settle().events[0];assert.equal(event.healing,12);assert.equal(event.hpStages[0].damageTaken,-12);
});

test('a treatment extension applies current Robust Health once to the original healing roll',()=>{
 const h=healer('H',{assurance:true,modifier:8,continualRecovery:false}),p=actor('P',{hp:{value:10,max:40,temp:0}});p.computed.patient.robustHealth=5;const actors=[h,p],sim=createRecoverySimulation({actors,session:session(actors,{extendTreatment:true}),random:dice([2,8],[2,8])});sim.next();assert.equal(sim.settle().events[0].healing,9);sim.next();const fresh=sim.snapshot().actors;fresh[1].computed.patient.robustHealth=6;const event=sim.settle({actors:fresh}).events[0];assert.equal(event.healing,10);
});

test('new finite or unknown immunity covering treatment start cancels settlement before sampling',()=>{
 for(const cooldownExpiresAt of [3600,Infinity]){
  const actors=[healer(),actor('P',{hp:{value:10,max:40,temp:0}})];let samples=0;const sim=createRecoverySimulation({actors,session:session(actors),random:()=>{samples++;return 0.5}});sim.next();const fresh=structuredClone(actors);fresh[1].cooldownExpiresAt=cooldownExpiresAt;
  const event=sim.settle({actors:fresh}).events[0];assert.equal(event.cancelled,true);assert.equal(samples,0);assert.equal(event.check,undefined);assert.equal(event.healing,0);assert.equal(hp(sim.snapshot(),'P'),10);
 }
});

test('fresh patient physiology and Stitch Flesh eligibility must still permit the selected treatment skill',()=>{
 for(const kind of ['became-undead','lost-stitch-flesh','nature','three-pecks']){
  const h=healer('H',{skill:kind==='nature'?'nature':'medicine'}),p=actor('P',{hp:{value:10,max:40,temp:0}});
  if(['lost-stitch-flesh','nature'].includes(kind))h.slugs=['stitch-flesh'];if(kind==='lost-stitch-flesh')p.modeOfBeing='undead';
  if(kind==='three-pecks'){h.computed.treatments=[];h.computed.threePecks={ready:true,treatments:treatment({skill:'occultism'})}}
  const actors=[h,p];let samples=0;const sim=createRecoverySimulation({actors,session:session(actors),random:()=>{samples++;return 0.5}});assert.equal(sim.next().activities.length,1);
  const fresh=structuredClone(actors);fresh[1].modeOfBeing='undead';if(kind==='lost-stitch-flesh')fresh[0].slugs=[];
  const event=sim.settle({actors:fresh}).events.find(event=>event.patientUUID);assert.equal(event.cancelled,true,kind);assert.equal(samples,0,kind);assert.equal(hp(sim.snapshot(),'P'),10);
 }
});

test('fresh Ward capacity and Three Pecks refocus qualification are checked before any patient roll',()=>{
 for(const kind of ['ward','three-pecks']){
  const h=healer('H',{capacity:2});if(kind==='three-pecks'){h.computed.treatments=[];h.computed.threePecks={ready:true,treatments:treatment({skill:'occultism'})};h.focus={value:0,max:1}}
  const actors=[h,actor('P',{hp:{value:10,max:40,temp:0}}),...kind==='ward'?[actor('Q',{hp:{value:10,max:40,temp:0}})]:[]];let samples=0;
  const sim=createRecoverySimulation({actors,session:session(actors),random:()=>{samples++;return 0.5}});sim.next();const fresh=structuredClone(actors);
  if(kind==='ward')fresh[0].computed.treatments[0].capacity=1;else fresh[0].computed.refocus={ready:false,reason:'refocus-recovery-unadapted'};
  const events=sim.settle({actors:fresh}).events;assert.ok(events.every(event=>event.cancelled));assert.equal(samples,0);assert.equal(hp(sim.snapshot(),'P'),10);
 }
});

test('losing the selected Continual Recovery option does not silently settle a different cooldown',()=>{
 const actors=[healer(),actor('P',{hp:{value:10,max:40,temp:0}})];let samples=0;const sim=createRecoverySimulation({actors,session:session(actors),random:()=>{samples++;return 0.5}});sim.next();const fresh=structuredClone(actors);fresh[0].computed.treatments[0].cooldownSeconds=3600;
 assert.equal(sim.settle({actors:fresh}).events[0].cancelled,true);assert.equal(samples,0);
});

test('extensions retain their own immunity exemption but recheck patient physiology',()=>{
 for(const changed of [false,true]){
  const actors=[healer('H',{assurance:true,modifier:8,continualRecovery:false}),actor('P',{hp:{value:10,max:40,temp:0}})],sim=createRecoverySimulation({actors,session:session(actors,{extendTreatment:true}),random:dice([2,8],[2,8])});sim.next();sim.settle();sim.next();const fresh=sim.snapshot().actors;
  assert.equal(fresh[1].cooldownExpiresAt,3600);if(changed)fresh[1].modeOfBeing='undead';const event=sim.settle({actors:fresh}).events[0];assert.equal(event.cancelled===true,changed);assert.equal(event.healing,changed?0:4);
 }
});

test('a newly undead Focus target cannot consume Focus or receive vitality healing at completion',()=>{
 const actors=[focusHealer(),actor('P',{hp:{value:10,max:40,temp:0}})],sim=createRecoverySimulation({actors,session:session(actors),random:()=>{throw Error('ineligible healing cannot sample')}});sim.next();const fresh=structuredClone(actors);fresh[1].modeOfBeing='undead';
 const batch=sim.settle({actors:fresh});assert.equal(batch.events[0].cancelled,true);assert.equal(batch.afterActors[0].focus.value,1);assert.equal(batch.afterActors[1].hp.value,10);
});
