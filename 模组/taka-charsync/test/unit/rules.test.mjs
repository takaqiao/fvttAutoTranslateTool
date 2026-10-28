import test from 'node:test';
import assert from 'node:assert/strict';
import { ownerOfActorPath, ownerOfItem, ownerOfItemPath } from '../../scripts/lib/rules.mjs';

test('actor paths: build belongs to the char server, state to the main server', () => {
  const cases = [
    ['system.details.level.value', 'char'],
    ['system.skills.acrobatics.rank', 'char'],
    ['system.skills.acrobatics.armor', 'main'],
    ['system.build.attributes.boosts.1', 'char'],
    ['system.attributes.hp.value', 'main'],
    ['system.details.xp.value', 'main'],
    ['flags.pf2e-skill-issue.x', 'char'],
    ['system.details.biography.appearance', 'char'],
    ['system.proficiencies.classDCs.wizard.rank', 'char'],
    ['system.details.biographyX', 'main'],
    ['name', 'main'],
  ];
  for (const [path, owner] of cases) assert.equal(ownerOfActorPath(path), owner, path);
});

test('item paths: build items belong to the char server except their combat state', () => {
  const cases = [
    ['feat', 'system.location', 'char'],
    ['feat', 'system.frequency.value', 'main'],
    ['spellcastingEntry', 'system.slots.slot1.max', 'char'],
    ['spellcastingEntry', 'system.slots.slot1.value', 'main'],
    ['spellcastingEntry', 'system.slots.slot1.prepared.0.id', 'char'],
    ['spellcastingEntry', 'system.slots.slot1.prepared.0.expended', 'main'],
    ['spell', 'system.location.uses.value', 'main'],
    ['spell', 'system.location.uses.max', 'char'],
    ['consumable', 'system.quantity', 'main'],
    ['affliction', 'system.x', 'main'],
    ['equipment', 'system.containerId', 'main'],
  ];
  for (const [type, path, owner] of cases) assert.equal(ownerOfItemPath(type, path), owner, `${type} ${path}`);
});

test('unlisted item types belong to the main server', () => {
  assert.equal(ownerOfItem('feat'), 'char');
  assert.equal(ownerOfItem('kit'), 'char');
  assert.equal(ownerOfItem('treasure'), 'main');
  assert.equal(ownerOfItem('affliction'), 'main');
});
