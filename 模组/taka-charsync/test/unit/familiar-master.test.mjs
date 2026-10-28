import test from 'node:test';
import assert from 'node:assert/strict';
import {makeCampaignEnvironment, ruleSnapshot} from '../helpers/campaign-environment.mjs';

let generatedAt = 100;

async function environment() {
  const env = await makeCampaignEnvironment({isolated: false, setup: false});
  const {api, a, b, Character, Familiar, runSetup} = env;
  game.user = {id: 'player', isGM: false};
  a.isOwner = b.isOwner = true;
  game.folders.set('folderA', {_id: 'folderA', folder: null});
  const next = ruleSnapshot();
  next.generatedAt = ++generatedAt;
  next.folderWorld = {folderA: 'a'};
  api.updateRuleSnapshot(next);
  const pet = new Familiar('pet', null);
  pet.name = pet._source.name = 'Familiar';
  pet.documentName = 'Actor';
  pet._source.folder = 'folderA';
  pet._source.system.master.ability = 'cha';
  pet._source.system.attributes = {hp: {value: 0, temp: 0}};
  pet._source.system.details = {creature: {value: 'rat'}};
  game.actors.set(pet.id, pet);
  const other = (id, flags = {}, {isOwner = true, type = 'character'} = {}) => {
    const actor = new Character(id);
    actor.type = actor._source.type = type;
    actor.isOwner = isOwner;
    actor._source.folder = 'folderA';
    actor._source.flags['taka-charsync'] = flags;
    game.actors.set(id, actor);
    return actor;
  };
  other('copy', {copy: {targetWorld: 'a'}});
  other('backup', {backupOf: 'A'});
  other('retired', {retired: true});
  other('unowned', {}, {isOwner: false});
  other('npc', {}, {type: 'npc'});
  class FamiliarSheetPF2e {
    constructor(actor) { this.actor = actor; }
    _getSubmitData(data) { return structuredClone(data); }
    async getData() {
      await Promise.resolve();
      return {editable: true, masters: [...game.actors.values()].filter(actor => actor.type === 'character'
        && (actor.isOwner || actor.id === this.actor._source.system.master.id))};
    }
  }
  CONFIG.Actor.sheetClasses.familiar = {'pf2e.FamiliarSheetPF2e': {cls: FamiliarSheetPF2e}};
  await runSetup();
  const hooks = new Map();
  Hooks.on = (name, callback) => {
    const callbacks = hooks.get(name) ?? [];
    callbacks.push(callback);
    hooks.set(name, callbacks);
  };
  const warnings = [];
  globalThis.ui = {notifications: {warn: message => warnings.push(message)}};
  const session = await import('../../scripts/char/rules-session.mjs');
  session.installRuleEditGuards();
  const allowed = changes => !hooks.get('preUpdateActor').some(callback => callback(pet, changes, {}) === false);
  const form = masterId => ({_id: pet.id, name: pet.name, system: {
    master: {id: masterId, ability: 'cha'}, attributes: {hp: {value: 0, temp: 0}}, details: {creature: {value: 'rat'}},
  }});
  return {...env, pet, allowed, form, warnings, FamiliarSheetPF2e};
}

test('a familiar can select a master when Document.update includes its unchanged id', async () => {
  const {pet, allowed} = await environment();
  assert.equal(allowed({_id: pet.id, 'system.master.id': 'A'}), true);
});

test('an unlinked familiar accepts a native full form with only the master actually changed', async () => {
  const {allowed, form} = await environment();
  assert.equal(allowed(form('A')), true);
});

test('a new familiar can select a master without submitting its clamped display HP', async () => {
  const {pet, allowed, FamiliarSheetPF2e} = await environment();
  // PF2e 新魔宠的源 HP 是 5；没有主人时上限为 0，原生表单显示并提交 0。
  pet._source.system.attributes.hp.value = 5;
  pet.system.attributes = {hp: {value: 0, max: 0, temp: 0}};
  const submitted = new FamiliarSheetPF2e(pet)._getSubmitData({
    name: 'Familiar', 'system.master.id': 'A', 'system.attributes.hp.value': 0,
    'system.attributes.hp.temp': 0, 'system.details.creature.value': 'rat',
  });
  assert.equal(Object.hasOwn(submitted, 'system.attributes.hp.value'), false);
  assert.equal(allowed({...submitted, _id: pet.id}), true);
  assert.equal(pet._source.system.attributes.hp.value, 5);
});

test('master selection keeps an edited HP input for the normal guard to reject', async () => {
  const {pet, allowed, FamiliarSheetPF2e} = await environment();
  pet._source.system.attributes.hp.value = 5;
  pet.system.attributes = {hp: {value: 0, max: 0, temp: 0}};
  const submitted = new FamiliarSheetPF2e(pet)._getSubmitData({
    'system.master.id': 'A', 'system.attributes.hp.value': 99,
  });
  assert.equal(submitted['system.attributes.hp.value'], 99);
  // Creature._preUpdate 会先钳制 HP，再进入 preUpdateActor；守卫仍须拒绝这次附带修改。
  assert.equal(allowed({...submitted, 'system.attributes.hp.value': 0}), false);
});

test('the familiar adapter leaves HP-only form submissions unchanged', async () => {
  const {pet, allowed, FamiliarSheetPF2e} = await environment();
  pet._source.system.attributes.hp.value = 5;
  pet.system.attributes = {hp: {value: 0, max: 0, temp: 0}};
  const submitted = new FamiliarSheetPF2e(pet)._getSubmitData({
    'system.master.id': '', 'system.attributes.hp.value': 0,
  });
  assert.equal(submitted['system.attributes.hp.value'], 0);
  assert.equal(allowed(submitted), false);
});

test('native master forms preserve HP through rebinding, clearing and binding again', async () => {
  const {pet, Character, allowed, FamiliarSheetPF2e} = await environment();
  const another = new Character('another');
  another.isOwner = true;
  another._source.folder = 'folderA';
  game.actors.set(another.id, another);
  pet._source.system.master.id = 'A';
  pet._source.system.attributes.hp.value = 5;
  pet.system.attributes = {hp: {value: 5, max: 30, temp: 0}};
  const sheet = new FamiliarSheetPF2e(pet);
  for (const masterId of ['another', '']) {
    const submitted = sheet._getSubmitData({'system.master.id': masterId, 'system.attributes.hp.value': 5});
    assert.equal(submitted['system.attributes.hp.value'], 5);
    assert.equal(allowed(submitted), true);
    pet._source.system.master.id = masterId || null;
  }
  pet.system.attributes.hp.value = pet.system.attributes.hp.max = 0;
  const rebound = sheet._getSubmitData({'system.master.id': 'A', 'system.attributes.hp.value': 0});
  assert.equal(Object.hasOwn(rebound, 'system.attributes.hp.value'), false);
  assert.equal(allowed(rebound), true);
  assert.equal(pet._source.system.attributes.hp.value, 5);
});

test('the familiar adapter does not rewrite a frozen copy form', async () => {
  const {pet, FamiliarSheetPF2e} = await environment();
  pet._source.flags['taka-charsync'] = {copy: {targetWorld: 'a'}};
  pet._source.system.attributes.hp.value = 5;
  pet.system.attributes = {hp: {value: 0, max: 0, temp: 0}};
  const submitted = new FamiliarSheetPF2e(pet)._getSubmitData({
    'system.master.id': 'A', 'system.attributes.hp.value': 0,
  });
  assert.equal(submitted['system.attributes.hp.value'], 0);
});

test('a master repair cannot carry real changes to the rest of the familiar form', async () => {
  const {allowed, form} = await environment();
  const name = form('A'); name.name = 'Changed name';
  const hp = form('A'); hp.system.attributes.hp.value = 99;
  const creature = form('A'); creature.system.details.creature.value = 'owl';
  const items = form('A'); items.items = [{_id: 'newAbility', type: 'action'}];
  for (const changes of [name, hp, creature, items]) assert.equal(allowed(changes), false);
});

for (const master of ['B', 'copy', 'backup', 'retired', 'unowned', 'npc', 'missing']) {
  test(`a familiar with valid existing rules still rejects ${master} as its new master`, async () => {
    const {pet, allowed, form} = await environment();
    pet._source.system.master.id = 'A';
    assert.equal(allowed(form(master)), false);
  });
}

test('an existing link to a frozen copy can be repaired through the native form', async () => {
  const {pet, allowed, form} = await environment();
  pet._source.system.master.id = 'copy';
  assert.equal(allowed(form('A')), true);
});

test('clearing an invalid master leaves a repairable missing-master state', async () => {
  const {api, pet, allowed, form} = await environment();
  pet._source.system.master.id = 'missing';
  assert.equal(allowed(form(null)), true);
  pet._source.system.master.id = null;
  assert.equal(api.actorRuleState(pet).reason, 'missing-master');
  assert.equal(allowed(form('A')), true);
});

test('an unresolved familiar can clear its master while publication is unavailable but cannot bind another', async () => {
  const {api, pet, allowed, form} = await environment();
  pet._source.system.master.id = 'missing';
  api.setRuleRefreshError('offline');
  try {
    assert.equal(allowed(form(null)), true);
    assert.equal(allowed(form('A')), false);
  } finally { api.setRuleRefreshError(null); }
});

test('native familiar master choices include only owned same-campaign working characters', async () => {
  const {pet, FamiliarSheetPF2e} = await environment();
  pet._source.system.master.id = 'copy';
  const data = await new FamiliarSheetPF2e(pet).getData();
  assert.deepEqual(data.masters.map(actor => actor.id), ['A']);
  assert.equal(data.editable, true);
});

test('an unassigned familiar inherits the chosen master campaign instead of locking its selector', async () => {
  const {pet, allowed, form, FamiliarSheetPF2e} = await environment();
  delete pet._source.folder;
  const data = await new FamiliarSheetPF2e(pet).getData();
  assert.deepEqual(data.masters.map(actor => actor.id), ['A', 'B']);
  assert.equal(data.editable, true);
  assert.equal(allowed(form('B')), true);
});

test('frozen familiar copies retain their submitted master and campaign resolution', async () => {
  const {api, pet, allowed, form, FamiliarSheetPF2e} = await environment();
  pet._source.flags['taka-charsync'] = {copy: {targetWorld: 'a'}};
  pet._source.system.master.id = 'copy';
  const data = await new FamiliarSheetPF2e(pet).getData();
  assert.ok(data.masters.some(actor => actor.id === 'copy'));
  assert.equal(api.actorRuleState(pet).world, 'a');
  assert.equal(allowed(form('copy')), true);
});
