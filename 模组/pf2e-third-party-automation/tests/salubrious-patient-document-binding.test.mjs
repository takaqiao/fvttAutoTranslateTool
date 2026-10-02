import test from 'node:test';
import assert from 'node:assert/strict';
import {createSalubriousKiss} from '../scripts/salubrious-kiss.mjs';
import {fixture as base} from './salubrious-kiss-fixture.mjs';
function fixture({choice=()=>{},queued,selection,template=()=>{},effect=()=>{},applied=()=>{},degree=1}={}){
 const f=base(),replacement={...f.patient,id:'replacement',uuid:'Actor.replacement',items:new Map(),flags:{}};f.game.actors.set(replacement.id,replacement);let checks=0,applications=0;
 const fromUuid=async uuid=>{const value=await f.fromUuid(uuid);if(uuid===f.pack.uuid)template(f,replacement);return value;};
 const create=f.patient.createEmbeddedDocuments;f.patient.createEmbeddedDocuments=async function(...args){const result=await create.apply(this,args);effect(f,replacement);return result;};
 const api=createSalubriousKiss({game:f.game,fromUuid,runExclusive:queued?async(_key,fn)=>{queued(f);return fn();}:undefined,choose:async question=>{if(question.kind==='patient'){choice(f,replacement);return selection??f.target.uuid;}return '3';},executor:{async roll(){checks++;return {degree,checkId:'original-check',damageId:degree===1?null:'original-damage'};},async apply(){applications++;applied(f,replacement);return {messageId:'original-application',targetUuid:f.target.uuid,kind:'healing'};}}});
 const invoke=()=>api.onRefocus({actor:f.actor,user:f.user,proof:f.proof});return {...f,replacement,api,invoke,get checks(){return checks;},get applications(){return applications;}};
}
test('a lawful native failure immunizes only its original patient and finishes once',async()=>{const f=fixture();assert.equal((await f.invoke()).state,'done');assert.deepEqual(f.effects.map(e=>e.actor.uuid),[f.patient.uuid]);assert.equal(f.patient.flags[f.M].salubriousKiss.pending,null);await f.invoke();assert.equal(f.checks,1);assert.equal(f.applications,0);assert.equal(f.effects.length,1);});
for(const [name,change] of [
 ['relink', (f,next)=>{f.target.actor=next;f.target.actorId=next.id;}],
 ['Token deletion',f=>f.scene.tokens.delete(f.target.id)],
 ['Token replacement',f=>f.scene.tokens.set(f.target.id,{...f.target})],
 ['source feat deletion',f=>f.actor.items.delete(f.item.id)],
 ['source owner loss',f=>{f.actor.testUserPermission=()=>false;}],
 ['original Refocus proof change',f=>{f.actor.flags[f.M].avRefocusIntent={...f.proof,nonce:'another-refocus'};}],
 ['current GM document replacement',f=>{f.game.users.set(f.gm.id,{...f.gm});}],
 ['GM handoff',f=>{f.game.users.activeGM={id:'next',active:true,isGM:true};}]
])test(`Salubrious ${name} during immunity template lookup cannot mutate a replacement patient`,async()=>{const f=fixture({template:change});await assert.rejects(f.invoke());assert.equal(f.effects.length,0);assert.equal(f.replacement.items.size,0);assert.equal(f.replacement.flags[f.M]?.salubriousKiss,undefined);assert.equal(f.checks,1);assert.equal(f.applications,0);await assert.rejects(f.invoke());assert.equal(f.checks,1);});
test('a relink during immunity creation preserves the completed original effect and cannot clear replacement pending',async()=>{const f=fixture({effect:(f,next)=>{next.flags[f.M]={salubriousKiss:{pending:{actorUuid:f.actor.uuid,nonce:f.proof.nonce}}};f.target.actor=next;f.target.actorId=next.id;}});await assert.rejects(f.invoke());assert.deepEqual(f.effects.map(e=>e.actor.uuid),[f.patient.uuid]);assert.deepEqual(f.replacement.flags[f.M].salubriousKiss.pending,{actorUuid:f.actor.uuid,nonce:f.proof.nonce});assert.equal(f.actor.flags[f.M].salubriousKiss.claims[0].state,'uncertain');await assert.rejects(f.invoke());assert.equal(f.effects.length,1);assert.equal(f.checks,1);});
test('a lawful success keeps original immunity, healing receipt and wounded removal',async()=>{const f=fixture({degree:2});assert.equal((await f.invoke()).state,'done');assert.equal(f.applications,1);assert.deepEqual(f.patient.removed,{slug:'wounded',options:{forceRemove:true}});assert.equal(f.patient.flags[f.M].salubriousKiss.pending,null);});
test('a patient relink after native healing cannot decrease replacement wounded or clear its reservation',async()=>{const f=fixture({degree:2,applied:(f,next)=>{next.flags[f.M]={salubriousKiss:{pending:{actorUuid:f.actor.uuid,nonce:f.proof.nonce}}};f.target.actor=next;f.target.actorId=next.id;}});await assert.rejects(f.invoke());assert.equal(f.applications,1);assert.equal(f.patient.removed,undefined);assert.equal(f.replacement.removed,undefined);assert.deepEqual(f.replacement.flags[f.M].salubriousKiss.pending,{actorUuid:f.actor.uuid,nonce:f.proof.nonce});await assert.rejects(f.invoke());assert.equal(f.checks,1);assert.equal(f.applications,1);});
test('a candidate relink while its patient choice is open cannot authorize its replacement',async()=>{const f=fixture({choice:(f,next)=>{f.target.actor=next;f.target.actorId=next.id;}});assert.equal((await f.invoke()).state,'declined');assert.equal(f.checks,0);assert.equal(f.effects.length,0);assert.equal(f.replacement.flags[f.M]?.salubriousKiss,undefined);});
test('known pre-roll refusal clears only the original exact reservation',async()=>{const f=fixture(),update=f.patient.update;f.patient.update=async function(changes){const result=await update.call(this,changes);if(changes[`flags.${f.M}.salubriousKiss.pending`]){f.replacement.flags[f.M]={salubriousKiss:{pending:{actorUuid:f.actor.uuid,nonce:f.proof.nonce}}};f.target.actor=f.replacement;f.target.actorId=f.replacement.id;}return result;};assert.equal((await f.invoke()).state,'declined');assert.equal(f.patient.flags[f.M].salubriousKiss.pending,null);assert.deepEqual(f.replacement.flags[f.M].salubriousKiss.pending,{actorUuid:f.actor.uuid,nonce:f.proof.nonce});assert.equal(f.checks,0);assert.equal(f.effects.length,0);});

for(const boundary of ['queued claim','canceled choice'])test(`a same-ID source Actor replacement at ${boundary} cannot write the obsolete source`,async()=>{
 let obsoleteWrites=0;
 const replace=f=>{const next={...f.actor,flags:structuredClone(f.actor.flags),items:new Map(f.actor.items)};f.game.actors.set(f.actor.id,next);};
 const f=fixture(boundary==='queued claim'?{queued:replace}:{choice:replace,selection:'only-refocus'}),update=f.actor.update;
 f.actor.update=async function(...args){if(f.game.actors.get(this.id)!==this)obsoleteWrites++;return update.apply(this,args);};
 await assert.rejects(f.invoke());assert.equal(obsoleteWrites,0);assert.equal(f.checks,0);assert.equal(f.effects.length,0);
});
