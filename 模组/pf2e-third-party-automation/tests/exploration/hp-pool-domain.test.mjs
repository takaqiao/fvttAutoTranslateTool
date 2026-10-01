import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHpPools} from '../../scripts/exploration/hp-pool.mjs';

function fixture(shared) {
  const patient = {id: 'Patient', uuid: 'Actor.Patient', system: {attributes: {hp: {value: 1, max: 20}}}};
  const first = {id: 'First', uuid: 'Actor.First', isOwner: true, system: {attributes: {hp: {value: 1, max: 20}}}};
  const second = {...first, id: 'Second', uuid: 'Actor.Second'};
  if (shared) patient.modules = {'pf2e-toolbelt': {shareData: {data: {health: true}}}};
  let master = first;
  let middleware;
  const game = {modules: new Map([['pf2e-toolbelt', {active: shared}]]), settings: {get: () => shared},
    toolbelt: {api: {shareData: {getMasterInMemory: () => master, getSlavesInMemory: () => [patient]}}}};
  const pools = createHpPools({game, actorUpdateEvents: {addActorUpdateMiddleware: fn => {middleware = fn; return () => {};}}});
  return {patient, first, second, pools, relink: () => {master = second},
    invoke: (actor, wrapped, fields) => middleware.call(actor, wrapped, fields, {})};
}

test('a native application cannot enter a different pool than its saved activity', async () => {
  const f = fixture(false);
  let applications = 0;
  await assert.rejects(f.pools.withNativeApplication({id: 'A', hpPoolUUIDs: ['Actor.Other']}, f.patient,
    async () => {applications++; return {}; }));
  assert.equal(applications, 0);
});

test('relink during native preparation cannot reach the patient update', async () => {
  const f = fixture(true);
  let updates = 0;
  await assert.rejects(f.pools.withNativeApplication({id: 'A', hpPoolUUIDs: [f.first.uuid]}, f.patient,
    async () => {
      await Promise.resolve();
      f.relink();
      return f.invoke(f.patient, async () => {updates++; return f.patient;}, {'system.attributes.hp.value': 10});
    }));
  assert.equal(updates, 0);
});

test('a synchronous Toolbelt forward cannot write a newly linked master', async () => {
  const f = fixture(true), fields = {'system.attributes.hp.value': 10};
  let updates = 0;
  await assert.rejects(f.pools.withNativeApplication({id: 'A', hpPoolUUIDs: [f.first.uuid]}, f.patient,
    async () => f.invoke(f.patient, async () => {
      f.relink();
      await f.invoke(f.second, async () => {updates++; return f.second;}, fields);
      return f.patient;
    }, fields)));
  assert.equal(updates, 0);
});

test('actual Toolbelt discovery rejects a relinked pool before native application', async () => {
  const f = fixture(true), expected = f.pools.discover(f.patient);
  assert.equal(expected.poolUUID, f.first.uuid);
  f.relink();
  assert.equal(f.pools.discover(f.patient).poolUUID, f.second.uuid);
  let applications = 0;
  await assert.rejects(f.pools.withNativeApplication({id: 'A', hpPoolUUIDs: [expected.poolUUID]}, f.patient,
    async () => {applications++; return {}; }));
  assert.equal(applications, 0);
});

for (const dataKey of ['data', '==data']) {
  test(`incoming Toolbelt ${dataKey} cannot select an unclaimed master before memory changes`, async () => {
    const f = fixture(true);
    let updates = 0;
    f.patient._preUpdate = async changes => {
      const incoming = changes.flags['pf2e-toolbelt'].shareData[dataKey];
      const target = incoming.master === f.second.id ? f.second : f.first;
      f.invoke(target, async () => {updates++; return target;}, {'system.attributes.hp.value': 10});
      return true;
    };
    const changes = {system: {attributes: {hp: {value: 10}}},
      flags: {'pf2e-toolbelt': {shareData: {[dataKey]: {master: f.second.id, health: true}}}}};
    await assert.rejects(f.pools.withNativeApplication({id: 'A', hpPoolUUIDs: [f.first.uuid]}, f.patient,
      async () => f.invoke(f.patient, async () => {
        await Promise.resolve();
        await f.patient._preUpdate(changes);
        return f.patient;
      }, changes)), /hp-pool-domain-changed/);
    assert.equal(updates, 0);
    assert.equal(f.pools.discover(f.patient).poolUUID, f.first.uuid);
  });
}
