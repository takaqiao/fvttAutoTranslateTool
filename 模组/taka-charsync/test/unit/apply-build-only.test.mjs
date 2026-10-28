import test from 'node:test';
import assert from 'node:assert/strict';
import {applyCopy, applyForce, previewPending} from '../../scripts/main/apply.mjs';
import {actorDoc, applyEnvironment, makeCopy} from '../helpers/apply-environment.mjs';
import {freezeMediaFixtures} from '../helpers/media-fixtures.mjs';

const item = (_id, type, system = {}, flags = {}) => ({_id, type, name: _id, system, flags, effects: []});
const byId = (doc, id) => doc.items.find(entry => entry._id === id);

async function scenario(change, {mainChange = () => {}, kind = 'level'} = {}) {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  let base = actorDoc(actor.id);
  base.items = [
    item('sword', 'weapon', {quantity: 1, containerId: null, equipped: {carryType: 'held', handsHeld: 1}}),
    item('gold', 'treasure', {quantity: 42, price: {value: {gp: 1}}}),
    item('potion', 'consumable', {quantity: 2}),
    item('feat', 'feat', {frequency: {value: 0, max: 1}}),
    item('spell', 'spell', {location: {uses: {value: 0, max: 1}}}),
    item('casting', 'spellcastingEntry', {slots: {slot1: {value: 0, max: 2, prepared: [{id: 'spell', expended: true}]}}}),
    item('condition', 'condition', {value: {value: 2}}),
    item('effect', 'effect', {duration: {value: 2}}),
  ];
  base.effects = [{_id: 'active-effect', name: 'Main effect', disabled: false}];
  base.system.resources = {focus: {value: 0, max: 1}, heroPoints: {value: 1, max: 3}};
  base.prototypeToken = {name: 'Main token', texture: {src: 'old.webp'}, width: 1, height: 1, sight: {range: 60}, light: {dim: 0}};
  let main = structuredClone(base);
  mainChange(main);
  let char = structuredClone(base);
  change(char);
  [base, main, char] = await freezeMediaFixtures([base, main, char]);
  env.files.set('history/main-source-base.json', structuredClone(base));
  actor._source = main;
  actor.refresh();
  const copy = env.publishCopy(makeCopy('source', kind, {name: char.name}), char);
  return {...env, actor, base, main: structuredClone(main), char, copy};
}

test('ordinary apply preserves inventory, coins, equipment state, conditions and effects', async () => {
  const env = await scenario(char => {
    char.items = char.items.filter(entry => !['potion', 'condition', 'effect'].includes(entry._id));
    byId(char, 'sword').system = {quantity: 4, equipped: {carryType: 'worn'}, containerId: 'bag'};
    byId(char, 'gold').system.quantity = 999;
    char.items.push(item('new-equipment', 'equipment'), item('new-condition', 'condition'), item('kit', 'kit'));
    char.effects = [];
    byId(char, 'feat').name = 'Updated feat';
  });
  const html = await previewPending({actorId: env.actor.id, copy: env.copy});
  assert.match(html, /Updated feat/);
  assert.doesNotMatch(html, /new-equipment|new-condition|999|potion|sword|gold|Main effect/);
  await applyCopy(env.actor, env.copy, {manual: true});
  for (const id of ['sword', 'gold', 'potion', 'condition', 'effect']) {
    assert.deepEqual(byId(env.actor.toObject(), id), byId(env.main, id));
  }
  assert.equal(byId(env.actor.toObject(), 'feat').name, 'Updated feat');
  assert.equal(byId(env.actor.toObject(), 'new-equipment'), undefined);
  assert.equal(byId(env.actor.toObject(), 'kit'), undefined);
  assert.deepEqual(env.actor.toObject().effects, env.main.effects);
});

test('removing a granting feat preserves its granted physical inventory item', async () => {
  const env = await scenario(char => { char.items = char.items.filter(entry => entry._id !== 'feat'); }, {
    mainChange(main) {
      byId(main, 'feat').flags.pf2e = {itemGrants: {gift: {id: 'sword', onDelete: 'cascade'}}};
      byId(main, 'sword').flags.pf2e = {grantedBy: {id: 'feat', onDelete: 'cascade'}};
    },
  });
  await applyCopy(env.actor, env.copy, {manual: true});
  assert.equal(byId(env.actor.toObject(), 'feat'), undefined);
  assert.deepEqual(byId(env.actor.toObject(), 'sword'), byId(env.main, 'sword'));
});

test('changing a physical item type to feat cannot bypass ordinary inventory protection', async () => {
  const env = await scenario(char => { byId(char, 'sword').type = 'feat'; byId(char, 'sword').name = 'Replacement'; });
  await applyCopy(env.actor, env.copy, {manual: true});
  assert.deepEqual(byId(env.actor.toObject(), 'sword'), byId(env.main, 'sword'));
});

test('ordinary build updates preserve spent resources while allowing larger maxima and new slots', async () => {
  const env = await scenario(char => {
    byId(char, 'feat').system.frequency = {value: 2, max: 2};
    byId(char, 'spell').system.location.uses = {value: 3, max: 3};
    const slots = byId(char, 'casting').system.slots;
    slots.slot1 = {value: 3, max: 3, prepared: [{id: 'new-spell', expended: false}, {id: 'spell', expended: false}]};
    slots.slot2 = {value: 1, max: 1, prepared: [{id: 'new-spell', expended: false}]};
    char.items.push(item('new-spell', 'spell'));
    char.system.resources.focus = {value: 3, max: 3};
  });
  await applyCopy(env.actor, env.copy, {manual: true});
  const after = env.actor.toObject();
  assert.deepEqual(byId(after, 'feat').system.frequency, {value: 0, max: 2});
  assert.deepEqual(byId(after, 'spell').system.location.uses, {value: 0, max: 3});
  const slots = byId(after, 'casting').system.slots;
  assert.equal(slots.slot1.value, 0);
  assert.equal(slots.slot1.max, 3);
  assert.deepEqual(slots.slot1.prepared, [{id: 'new-spell', expended: true}, {id: 'spell', expended: false}]);
  assert.deepEqual(slots.slot2, {value: 1, max: 1, prepared: [{id: 'new-spell', expended: false}]});
  assert.deepEqual(after.system.resources.focus, {value: 0, max: 3});
});

test('ordinary appearance changes preserve token dimensions, sight, light and actor combat state', async () => {
  const env = await scenario(char => {
    char.name = 'New appearance'; char.img = 'portrait.webp';
    Object.assign(char.prototypeToken, {name: 'New token', texture: {src: 'token.webp'}, width: 4, sight: {range: 5}, light: {dim: 30}});
    char.system.attributes.hp.value = 20;
    char.system.resources.heroPoints.value = 3;
  });
  await applyCopy(env.actor, env.copy, {manual: true});
  assert.equal(env.actor.name, 'New appearance');
  assert.equal(env.actor.toObject().img, env.char.img);
  assert.equal(env.actor.toObject().prototypeToken.texture.src, env.char.prototypeToken.texture.src);
  for (const key of ['width', 'height', 'sight', 'light']) assert.deepEqual(env.actor.toObject().prototypeToken[key], env.main.prototypeToken[key]);
  assert.equal(env.actor.system.attributes.hp.value, 12);
  assert.equal(env.actor.system.resources.heroPoints.value, 1);
});

test('ordinary apply refuses a main card with invalid source inventory before creating undo points', async () => {
  const env = await scenario(char => { byId(char, 'feat').name = 'Updated feat'; });
  env.actor.items.delete('sword');
  env.actor.items.invalidDocumentIds.add('sword');
  const files = [...env.files.keys()];
  await assert.rejects(applyCopy(env.actor, env.copy, {manual: true}), /非法.*条目|条目.*校验/);
  assert.deepEqual(env.writes, []);
  assert.deepEqual([...env.files.keys()], files);
  assert.deepEqual(env.actor.toObject(), env.main);
});

test('force apply still replaces inventory and spent resource values', async () => {
  const env = await scenario(char => { byId(char, 'gold').system.quantity = 999; byId(char, 'feat').system.frequency.value = 1; }, {kind: 'force'});
  await applyForce(env.actor, env.copy);
  assert.equal(byId(env.actor.toObject(), 'gold').system.quantity, 999);
  assert.equal(byId(env.actor.toObject(), 'feat').system.frequency.value, 1);
});

test('ordinary automatic apply stops at appearance and ability conflicts until the GM applies them', async () => {
  const env = await scenario(char => {
    char.img = 'draft.webp'; char.system.abilities = {str: {mod: 4}};
  }, {mainChange(main) {
    main.img = 'main.webp'; main.system.abilities = {str: {mod: 2}};
  }});
  assert.equal(await applyCopy(env.actor, env.copy), 'conflict');
  assert.deepEqual(env.writes, []);
  assert.equal(env.actor._source.img, env.main.img);
  await applyCopy(env.actor, env.copy, {manual: true});
  assert.equal(env.actor._source.img, env.char.img);
  assert.equal(env.actor.system.abilities.str.mod, 4);
});
