import test from 'node:test';
import assert from 'node:assert/strict';
import {createSalubriousKiss} from '../scripts/salubrious-kiss.mjs';
import {captureSalubriousPrivacy} from '../scripts/salubrious-privacy.mjs';
import {fixture} from './salubrious-kiss-fixture.mjs';

function withPrivacy(f,mode){
 f.proof.privacy=captureSalubriousPrivacy({...f,user:f.user,requestedMode:mode});f.proof.noteId='native-refocus-note';
 f.actor.flags[f.M].avRefocusIntent=structuredClone(f.proof);f.actor.flags[f.M].refocusEvents=[{...structuredClone(f.proof),state:'claimed'}];
 f.game.messages.set(f.proof.noteId,{id:f.proof.noteId,author:f.user,speaker:{actor:f.actor.id,scene:f.scene.id,token:f.token.id},blind:f.proof.privacy.blind,whisper:[...f.proof.privacy.whisper],flags:{[f.M]:{avRefocusNote:{...structuredClone(f.proof),kind:'completion'}}}});
 return f;
}

function provider(f,{afterChoice=()=>{},afterRoll=()=>{}}={}){
 const choices=[],checks=[],applications=[];
 const api=createSalubriousKiss({game:f.game,fromUuid:f.fromUuid,validateRefocusNote:({proof})=>proof.noteId===f.proof.noteId&&f.game.messages.has(proof.noteId),choose:async question=>{
  choices.push(question);if(question.kind==='patient'){afterChoice();return f.target.uuid;}return '3';
 },executor:{async roll(claim){checks.push(structuredClone(claim));afterRoll();return {degree:2,checkId:'native-check',damageId:'native-healing'};},async apply(claim,result){applications.push({claim,result});return {messageId:'native-application',targetUuid:claim.targetUuid,kind:'healing'};}}});
 return {choices,checks,applications,invoke:()=>api.onRefocus({actor:f.actor,user:f.user,proof:f.proof})};
}

for(const distance of [60,undefined,5])test(`Salubrious Kiss completes its original patient and DC without measuring distance ${distance}`,async()=>{
 const f=fixture();let measurements=0;f.token.object.distanceTo=()=>{measurements++;return distance;};const p=provider(f);
 const result=await p.invoke();assert.equal(result.state,'done');assert.equal(measurements,0);
 assert.deepEqual(p.choices.map(q=>q.kind),['patient','tier']);assert(p.choices[0].choices.some(c=>c.value===f.target.uuid));
 assert.equal(p.checks[0].targetUuid,'Scene.s.Token.patient');assert.equal(p.checks[0].targetActorUuid,'Actor.patient');assert.equal(p.checks[0].dc,30);
 assert.equal(p.applications[0].claim.targetUuid,'Scene.s.Token.patient');assert.deepEqual(p.applications[0].result,{degree:2,checkId:'native-check',damageId:'native-healing'});
 assert.equal(f.actor.system.resources.focus.value,3);assert.equal(f.game.time.worldTime,100);assert.equal(f.effects.length,1);assert.equal(f.effects[0].flags[f.M].salubriousKiss.expiresAt,3700);
 assert.equal(f.patient.flags[f.M].salubriousKiss.pending,null);assert.deepEqual(f.patient.removed,{slug:'wounded',options:{forceRemove:true}});
 await p.invoke();assert.equal(p.checks.length,1);assert.equal(p.applications.length,1);assert.equal(f.effects.length,1);
});

test('Salubrious Kiss keeps the selected patient when tokens move during the normal patient choice',async()=>{
 const f=fixture(),p=provider(f,{afterChoice:()=>{f.token.object.distanceTo=()=>60;}});
 assert.equal((await p.invoke()).state,'done');assert.equal(p.checks.length,1);assert.equal(p.applications[0].claim.targetUuid,'Scene.s.Token.patient');
});

test('Salubrious Kiss settles the saved native result after the patient moves during the check',async()=>{
 const f=fixture(),p=provider(f,{afterRoll:()=>{f.token.object.distanceTo=()=>60;}});
 assert.equal((await p.invoke()).state,'done');assert.equal(p.applications.length,1);assert.equal(f.effects.length,1);
 await p.invoke();assert.equal(p.checks.length,1);assert.equal(p.applications.length,1);
});

test('Salubrious Kiss excludes initially hidden patients from the owner selector',async()=>{
 const f=fixture();f.target.hidden=true;const p=provider(f);assert.equal((await p.invoke()).state,'declined');
 assert.equal(p.choices[0].choices.some(c=>c.value===f.target.uuid),false);assert.equal(p.checks.length,0);assert.equal(f.effects.length,0);
});

for(const stage of ['afterChoice','afterRoll'])test(`Salubrious Kiss keeps its selected patient when hidden changes ${stage} under its existing blind audience`,async()=>{
 const f=withPrivacy(fixture(),'blind'),p=provider(f,{[stage]:()=>{f.target.hidden=true;}});
 assert.equal((await p.invoke()).state,'done');assert.equal(p.checks.length,1);assert.equal(p.applications[0].claim.targetUuid,'Scene.s.Token.patient');assert.equal(f.effects.length,1);
 assert.equal(p.applications[0].claim.privacy.mode,'blind');assert.deepEqual(p.applications[0].claim.privacy.whisper,['gm']);
});

test('Salubrious Kiss retains the existing audience guard when a public patient becomes private during the native roll',async()=>{
 const f=withPrivacy(fixture(),'public'),p=provider(f,{afterRoll:()=>{f.target.hidden=true;}});
 await assert.rejects(p.invoke,/更严格受众/);assert.equal(p.applications.length,0);assert.equal(f.effects.length,0);assert.equal(f.actor.flags[f.M].salubriousKiss.claims[0].state,'uncertain');
 await assert.rejects(p.invoke);assert.equal(p.checks.length,1);
});

for(const [name,change] of [
 ['stale patient token',f=>f.scene.tokens.delete(f.target.id)],
 ['changed patient actor',f=>{f.target.actor=f.actor;f.target.actorId=f.actor.id;}],
 ['lost source feat',f=>f.actor.items.delete(f.item.id)],
 ['offline original owner',f=>{f.user.active=false;}],
 ['no longer eligible DC tier',f=>{f.actor.skills.occultism.rank=1;}]
])test(`Salubrious Kiss still rejects ${name} before applying a native result`,async()=>{
 const f=fixture(),p=provider(f,{afterRoll:()=>change(f)});
 await assert.rejects(p.invoke);assert.equal(p.applications.length,0);assert.equal(f.effects.length,0);
 assert.equal(f.actor.flags[f.M].salubriousKiss.claims[0].state,'uncertain');await assert.rejects(p.invoke);assert.equal(p.checks.length,1);
});

test('Salubrious Kiss still declines a patient who gains treatment immunity during selection',async()=>{
 const f=fixture(),p=provider(f,{afterChoice:()=>f.patient.items.set('immune',{type:'effect',sourceId:f.pack.uuid,isExpired:false})});
 assert.equal((await p.invoke()).state,'declined');assert.equal(p.checks.length,0);assert.equal(f.effects.length,0);assert.equal(f.actor.system.resources.focus.value,3);
});
