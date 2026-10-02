import test from 'node:test';
import assert from 'node:assert/strict';
import {createElementalMedicine} from '../scripts/elemental-medicine.mjs';
import {ELEMENTAL_MEDICINE_SOURCE, ELEMENTAL_MEDICINE_DAILY} from '../scripts/elemental-medicine-rules.mjs';
import {MODULE_ID as ID} from '../scripts/rules.mjs';

function update(document, changes) {
  for (const [path, value] of Object.entries(changes)) {
    const keys = path.split('.');
    let target = document;
    for (const key of keys.slice(0, -1)) target = target[key] ??= {};
    target[keys.at(-1)] = structuredClone(value);
  }
}
function fixture({linked = false, onNoticeClaim, resolvePatient} = {}) {
  const gm = {id: 'gm', isGM: true, active: true};
  const doctorOwner = {id: 'doctor-owner', active: true};
  const patientOwner = {id: 'patient-owner', active: true};
  const baseOwner = {id: 'base-owner', active: true};
  const inactiveOwner = {id: 'inactive-owner', active: false};
  const users = Object.assign(new Map([gm, doctorOwner, patientOwner, baseOwner, inactiveOwner].map(user => [user.id, user])), {activeGM: gm});
  const patientUuid = linked ? 'Actor.shared-base' : 'Scene.scene.Token.patient.Actor.shared-base';
  const patient = {id: 'shared-base', uuid: patientUuid, type: 'character', level: 1, items: new Map(), owners: new Set([patientOwner.id, inactiveOwner.id]), testUserPermission(user, permission) {return permission === 'LIMITED' || this.owners.has(user.id);}};
  const base = {id: patient.id, uuid: 'Actor.shared-base', testUserPermission: user => user === baseOwner};
  const doctor = {id: 'doctor', uuid: 'Actor.doctor', type: 'character', items: new Map(), flags: {[ID]: {elementalMedicineDaily: {requestUuid: 'Actor.doctor.Item.request'}}}, testUserPermission: user => user === doctorOwner, getStatistic: () => ({check: {roll: () => {throw Error('Saved diagnosis must never reroll');}}})};
  doctor.items.set('feat', {id: 'feat', actor: doctor, type: 'feat', sourceId: ELEMENTAL_MEDICINE_SOURCE});
  const row = {patientUuid, skill: 'medicine', state: 'checked', nonce: 'original', checkId: 'native-check', rollUserId: doctorOwner.id};
  const request = {id: 'request', uuid: 'Actor.doctor.Item.request', actor: doctor, flags: {'pf2e-dailies': {daily: `module.${ELEMENTAL_MEDICINE_DAILY}`}, [ID]: {elementalMedicine: {kind: 'preparation', actorUuid: doctor.uuid, userId: doctorOwner.id, status: 'diagnosing', factsId: 'facts', patients: [row]}}}, async update(changes) {update(this, changes); if (this.flags[ID].elementalMedicine.patients[0].noticeStarted && !this.claimObserved) {this.claimObserved = true; await onNoticeClaim?.({patient, request: this, patientOwner, baseOwner});}}};
  doctor.items.set(request.id, request);
  const fact = {patientUuid, skill: 'medicine', dc: 15, binding: {itemUuid: `${patientUuid}.Item.affliction`}, correctDiagnosis: '患者私密诊断'};
  const facts = {id: 'facts', author: gm, blind: true, whisper: [gm.id], flags: {[ID]: {elementalMedicineFacts: {requestUuid: request.uuid, facts: [fact]}}}};
  const check = {id: row.checkId, actor: doctor, author: doctorOwner, speaker: {actor: doctor.id}, blind: true, whisper: [gm.id], rolls: [{_evaluated: true, total: 10, options: {degreeOfSuccess: 1}}], flags: {pf2e: {context: {type: 'skill-check', dc: {value: 15}, options: [`${ID}:elemental-medicine:${row.nonce}`, 'check:statistic:medicine'], outcome: 'failure'}}, [ID]: {elementalMedicineCheck: {requestUuid: request.uuid, patientUuid, nonce: row.nonce, skill: row.skill}}}};
  let lookups = 0;
  const fromUuid = async uuid => {
    if (uuid === request.uuid) return request;
    if (uuid === patientUuid) {lookups++; return resolvePatient ? resolvePatient({patient, base, lookups, request}) : patient;}
    return null;
  };
  const game = {user: gm, users, actors: new Map([[doctor.id, doctor], [base.id, linked ? patient : base]]), messages: new Map([[facts.id, facts], [check.id, check]]), time: {worldTime: 100}};
  const notices = [];
  const provider = createElementalMedicine({game, fromUuid, publishDiagnosis: async context => {notices.push(context); return {id: 'notice'};}});
  return {provider, request, doctorOwner, patient, base, patientOwner, baseOwner, notices, row: () => request.flags[ID].elementalMedicine.patients[0]};
}

test('a synthetic patient notice reaches its owners without leaking to the base actor owner', async () => {
  const f = fixture();
  assert.equal((await f.provider.prepare(f.request.uuid, f.doctorOwner)).status, 'done');
  assert.deepEqual(f.notices[0].recipients.sort(), ['gm', 'doctor-owner', 'patient-owner'].sort());
  assert.equal(f.notices[0].patientUuid, f.patient.uuid);
  await f.provider.prepare(f.request.uuid, f.doctorOwner);
  assert.equal(f.notices.length, 1);
});
test('a world or linked-token patient retains its normal owner audience', async () => {
  const f = fixture({linked: true});
  await f.provider.prepare(f.request.uuid, f.doctorOwner);
  assert.deepEqual(f.notices[0].recipients.sort(), ['gm', 'doctor-owner', 'patient-owner'].sort());
});
for (const invalid of ['missing', 'base actor fallback', 'replacement document']) test(`a ${invalid} patient at notification time cannot publish the diagnosis`, async () => {
  const f = fixture({resolvePatient: ({patient, base, lookups}) => lookups === 1 ? patient : invalid === 'missing' ? null : invalid === 'base actor fallback' ? base : {...patient}});
  await assert.rejects(() => f.provider.prepare(f.request.uuid, f.doctorOwner), /患者/);
  assert.equal(f.notices.length, 0);
  assert.equal(f.request.flags[ID].elementalMedicine.status, 'uncertain');
});
test('patient ownership is read after the awaited notification claim', async () => {
  const f = fixture({onNoticeClaim: ({patient, baseOwner}) => {patient.owners = new Set([baseOwner.id]);}});
  await f.provider.prepare(f.request.uuid, f.doctorOwner);
  assert.deepEqual(f.notices[0].recipients.sort(), ['gm', 'doctor-owner', 'base-owner'].sort());
});
test('a patient disappearing during the awaited notification claim leaves the notice unissued', async () => {
  const f = fixture({resolvePatient: ({patient, request}) => request.claimObserved ? null : patient});
  await assert.rejects(() => f.provider.prepare(f.request.uuid, f.doctorOwner), /患者/);
  assert.equal(f.notices.length, 0);
  assert.equal(f.row().noticeStarted, true);
  assert.equal(f.row().noticeId, undefined);
});
