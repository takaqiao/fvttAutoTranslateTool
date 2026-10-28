import test from 'node:test';
import assert from 'node:assert/strict';
import { formatSummary, merge3, summarize } from '../../scripts/lib/merge.mjs';

function item(_id, type, name, system, flags = {}) {
  return {_id, type, name, system, flags, sort: 0, _stats: {compendiumSource: null, modifiedTime: 1}};
}

function baseActor() {
  return {
    _id: 'actor00000000001',
    name: 'Test Hero',
    type: 'character',
    folder: 'folderBase',
    sort: 0,
    ownership: {default: 0},
    _stats: {modifiedTime: 1},
    system: {
      details: {level: {value: 6}, xp: {value: 0, max: 1000}},
      attributes: {hp: {value: 40}},
      skills: {acrobatics: {rank: 1}},
    },
    flags: {},
    items: [
      item('F1', 'feat', 'Feat One', {location: 'class-2'}),
      item('F2', 'feat', 'Feat Two', {location: 'class-4'}, {pf2e: {itemGrants: {g1: {id: 'G1', onDelete: 'cascade'}}}}),
      item('G1', 'action', 'Granted One', {x: 1}, {pf2e: {grantedBy: {id: 'F2', onDelete: 'detach'}}}),
      item('C1', 'consumable', 'Potion One', {quantity: 3}),
      item('S1', 'spellcastingEntry', 'Spells', {
        slots: {slot1: {value: 2, max: 3, prepared: [{id: 'sp1', expended: false}, {id: 'sp2', expended: false}]}},
      }),
    ],
    effects: [{_id: 'E1', name: 'Effect One', disabled: false, sort: 0, _stats: {modifiedTime: 1}}],
  };
}

// 每个用例从同一份 base 派生两边，edit 回调就地改副本。
function sides(editMain = () => {}, editChar = () => {}) {
  const base = baseActor();
  const main = structuredClone(base);
  const char = structuredClone(base);
  editMain(main);
  editChar(char);
  return {base, main, char};
}

const find = (doc, id) => doc.items.find(entry => entry._id === id);
const removeItem = (doc, id) => { doc.items = doc.items.filter(entry => entry._id !== id); };

test('each side keeps its own change', () => {
  const {base, main, char} = sides(m => { m.system.attributes.hp.value = 35; }, c => { c.system.details.level.value = 7; });
  const {doc, conflicts, changed} = merge3({base, main, char, target: 'main'});
  assert.equal(doc.system.attributes.hp.value, 35);
  assert.equal(doc.system.details.level.value, 7);
  assert.deepEqual(conflicts, []);
  assert.equal(changed, true);
});

test('level conflict goes to the char server', () => {
  const {base, main, char} = sides(m => { m.system.details.level.value = 7; }, c => { c.system.details.level.value = 8; });
  const {doc, conflicts} = merge3({base, main, char, target: 'main'});
  assert.equal(doc.system.details.level.value, 8);
  assert.deepEqual(conflicts, [{scope: 'actor', path: 'system.details.level.value', winner: 'char'}]);
});

test('hp conflict goes to the main server', () => {
  const {base, main, char} = sides(m => { m.system.attributes.hp.value = 30; }, c => { c.system.attributes.hp.value = 20; });
  const {doc, conflicts} = merge3({base, main, char, target: 'char'});
  assert.equal(doc.system.attributes.hp.value, 30);
  assert.deepEqual(conflicts, [{scope: 'actor', path: 'system.attributes.hp.value', winner: 'main'}]);
});

test('loot added on the main server is copied with its _stats', () => {
  const loot = {...item('L1', 'treasure', 'Loot', {quantity: 1}), _stats: {compendiumSource: 'X', modifiedTime: 5}};
  const {base, main, char} = sides(m => { m.items.push(loot); });
  const {doc} = merge3({base, main, char, target: 'char'});
  assert.deepEqual(find(doc, 'L1')._stats, loot._stats);
});

test('main deletes a feat the char server edited: the edit survives', () => {
  const {base, main, char} = sides(m => removeItem(m, 'F1'), c => { find(c, 'F1').system.location = 'class-6'; });
  const {doc, conflicts} = merge3({base, main, char, target: 'main'});
  assert.deepEqual(find(doc, 'F1'), find(char, 'F1'));
  assert.deepEqual(conflicts, [{scope: 'item', id: 'F1', name: 'Feat One', path: '', winner: 'char'}]);
});

test('char deletes an untouched feat', () => {
  const {base, main, char} = sides(undefined, c => removeItem(c, 'F1'));
  const {doc, conflicts} = merge3({base, main, char, target: 'main'});
  assert.equal(find(doc, 'F1'), undefined);
  assert.deepEqual(conflicts, []);
});

test('char deletes a feat the main server edited: the owner deletion wins', () => {
  const {base, main, char} = sides(m => { find(m, 'F1').system.location = 'class-6'; }, c => removeItem(c, 'F1'));
  const {doc, conflicts} = merge3({base, main, char, target: 'main'});
  assert.equal(find(doc, 'F1'), undefined);
  assert.deepEqual(conflicts, [{scope: 'item', id: 'F1', name: 'Feat One', path: '', winner: 'char'}]);
});

test('main used up a potion the char server edited: the potion is gone', () => {
  const {base, main, char} = sides(m => removeItem(m, 'C1'), c => { find(c, 'C1').system.quantity = 5; });
  const {doc, conflicts} = merge3({base, main, char, target: 'char'});
  assert.equal(find(doc, 'C1'), undefined);
  assert.deepEqual(conflicts, [{scope: 'item', id: 'C1', name: 'Potion One', path: '', winner: 'main'}]);
});

test('potion quantity changed only on the main server', () => {
  const {base, main, char} = sides(m => { find(m, 'C1').system.quantity = 2; });
  const {doc} = merge3({base, main, char, target: 'char'});
  assert.equal(find(doc, 'C1').system.quantity, 2);
});

test('prepared slots: spell ids from the char server, expended from the main server', () => {
  const {base, main, char} = sides(
    m => { find(m, 'S1').system.slots.slot1.prepared[1].expended = true; },
    c => { find(c, 'S1').system.slots.slot1.prepared[0].id = 'sp9'; },
  );
  const {doc, conflicts} = merge3({base, main, char, target: 'main'});
  assert.deepEqual(find(doc, 'S1').system.slots.slot1.prepared, [{id: 'sp9', expended: false}, {id: 'sp2', expended: true}]);
  assert.deepEqual(conflicts, []);
});

test('prepared slots: main shortens the list while char swaps another spell', () => {
  const {base, main, char} = sides(
    m => { find(m, 'S1').system.slots.slot1.prepared = [{id: 'sp1', expended: true}]; },
    c => { find(c, 'S1').system.slots.slot1.prepared[0].id = 'sp9'; },
  );
  const {doc, conflicts} = merge3({base, main, char, target: 'main'});
  assert.deepEqual(find(doc, 'S1').system.slots.slot1.prepared, [{id: 'sp9', expended: false}]);
  assert.deepEqual(conflicts, []);
});

test('prepared slots: char shortens the list while main expends another spell', () => {
  const {base, main, char} = sides(
    m => { find(m, 'S1').system.slots.slot1.prepared[0].expended = true; },
    c => { find(c, 'S1').system.slots.slot1.prepared = [{id: 'sp1', expended: false}]; },
  );
  const {doc, conflicts} = merge3({base, main, char, target: 'char'});
  assert.deepEqual(find(doc, 'S1').system.slots.slot1.prepared, [{id: 'sp1', expended: true}]);
  assert.deepEqual(conflicts, []);
});

test('remaining spell slots belong to the main server', () => {
  const {base, main, char} = sides(m => { find(m, 'S1').system.slots.slot1.value = 0; }, c => { find(c, 'S1').system.slots.slot1.value = 1; });
  const {doc, conflicts} = merge3({base, main, char, target: 'char'});
  assert.equal(find(doc, 'S1').system.slots.slot1.value, 0);
  assert.deepEqual(conflicts, [{scope: 'item', id: 'S1', name: 'Spells', path: 'system.slots.slot1.value', winner: 'main'}]);
});

test('a granted item follows its root when the root is deleted', () => {
  const {base, main, char} = sides(m => { find(m, 'G1').system.x = 2; }, c => removeItem(c, 'F2'));
  const {doc, conflicts} = merge3({base, main, char, target: 'main'});
  assert.equal(find(doc, 'F2'), undefined);
  assert.equal(find(doc, 'G1'), undefined);
  assert.deepEqual(conflicts, [{scope: 'item', id: 'G1', name: 'Granted One', path: '', winner: 'char'}]);
});

test('a granted item follows its root when the root survives a deletion', () => {
  const {base, main, char} = sides(m => { removeItem(m, 'F2'); removeItem(m, 'G1'); }, c => { find(c, 'F2').system.location = 'class-6'; });
  const {doc, conflicts} = merge3({base, main, char, target: 'main'});
  assert.deepEqual(find(doc, 'F2'), find(char, 'F2'));
  assert.deepEqual(find(doc, 'G1'), find(char, 'G1'));
  assert.deepEqual(conflicts, [{scope: 'item', id: 'F2', name: 'Feat Two', path: '', winner: 'char'}]);
});

test('a granted item detached from a surviving root stays deleted', () => {
  const {base, main, char} = sides(undefined, c => {
    removeItem(c, 'G1');
    find(c, 'F2').flags.pf2e.itemGrants = {};
  });
  const {doc} = merge3({base, main, char, target: 'main'});
  assert.ok(find(doc, 'F2'));
  assert.equal(find(doc, 'G1'), undefined);
});

test('a new grant tree is added whole', () => {
  const {base, main, char} = sides(undefined, c => {
    c.items.push(item('F3', 'feat', 'Feat Three', {}, {pf2e: {itemGrants: {g3: {id: 'G3', onDelete: 'cascade'}}}}));
    c.items.push(item('G3', 'action', 'Granted Three', {}, {pf2e: {grantedBy: {id: 'F3', onDelete: 'detach'}}}));
  });
  const {doc} = merge3({base, main, char, target: 'main'});
  assert.ok(find(doc, 'F3'));
  assert.ok(find(doc, 'G3'));
});

test('actor-local fields come from the target', () => {
  const ownFlag = {base: {kind: 'main', hash: 'aaaa'}};
  const {base, main, char} = sides(m => {
    m.ownership = {default: 2};
    m.folder = 'folderMain';
    m._stats = {modifiedTime: 7};
    m.flags['taka-charsync'] = ownFlag;
  });
  for (const target of ['main', 'char']) {
    const source = target === 'main' ? main : char;
    const {doc, changed} = merge3({base, main, char, target});
    assert.deepEqual(doc.ownership, source.ownership, target);
    assert.equal(doc.folder, source.folder, target);
    assert.deepEqual(doc._stats, source._stats, target);
    assert.deepEqual(doc.flags['taka-charsync'], source.flags['taka-charsync'], target);
    assert.equal(changed, false, target);
  }
});

test('local fields of an existing item come from the target', () => {
  const {base, main, char} = sides(
    m => Object.assign(find(m, 'F1'), {sort: 1, _stats: {modifiedTime: 'A'}}),
    c => Object.assign(find(c, 'F1'), {sort: 9, _stats: {modifiedTime: 'B'}}),
  );
  const {doc} = merge3({base, main, char, target: 'char'});
  assert.equal(find(doc, 'F1').sort, 9);
  assert.deepEqual(find(doc, 'F1')._stats, {modifiedTime: 'B'});
});

test('nothing changed on either side', () => {
  const {base, main, char} = sides();
  const {doc, changed, conflicts} = merge3({base, main, char, target: 'main'});
  assert.equal(changed, false);
  assert.deepEqual(doc, main);
  assert.deepEqual(conflicts, []);
});

test('skill-issue flag conflict goes to the char server', () => {
  const {base, main, char} = sides(m => { m.flags['pf2e-skill-issue'] = {v: 1}; }, c => { c.flags['pf2e-skill-issue'] = {v: 2}; });
  const {doc} = merge3({base, main, char, target: 'main'});
  assert.equal(doc.flags['pf2e-skill-issue'].v, 2);
});

test('effect added on the main server', () => {
  const {base, main, char} = sides(m => { m.effects.push({_id: 'E2', name: 'Effect Two', disabled: false, sort: 0, _stats: {}}); });
  const {doc} = merge3({base, main, char, target: 'char'});
  assert.ok(doc.effects.find(effect => effect._id === 'E2'));
});

test('both sides add the same id: merged against an empty base', () => {
  const {base, main, char} = sides(
    m => { m.items.push(item('L9', 'treasure', 'Gem', {quantity: 1})); },
    c => { c.items.push(item('L9', 'treasure', 'Gem', {quantity: 2})); },
  );
  const {doc, conflicts} = merge3({base, main, char, target: 'char'});
  assert.equal(find(doc, 'L9').system.quantity, 1);
  assert.deepEqual(conflicts, [{scope: 'item', id: 'L9', name: 'Gem', path: 'system.quantity', winner: 'main'}]);
});

test('inputs are not mutated and the result is detached from them', () => {
  const {base, main, char} = sides(m => { m.system.attributes.hp.value = 35; }, c => { c.system.details.level.value = 7; });
  const snapshot = structuredClone({base, main, char});
  const {doc} = merge3({base, main, char, target: 'main'});
  doc.system.skills.acrobatics.rank = 4;
  doc.items[0].system.location = 'changed';
  assert.deepEqual({base, main, char}, snapshot);
});

test('summary of a level-up', () => {
  const {base, main, char} = sides(undefined, c => {
    c.system.details.level.value = 7;
    c.items.push(item('X1', 'feat', 'X', {}));
    removeItem(c, 'C1');
    c.system.skills.acrobatics.rank = 2;
  });
  const {doc} = merge3({base, main, char, target: 'main'});
  assert.deepEqual(summarize(base, doc), {
    level: [6, 7],
    added: [{type: 'feat', name: 'X'}],
    removed: [{type: 'consumable', name: 'Potion One'}],
    changed: [],
    skillRanks: [{skill: 'acrobatics', from: 1, to: 2}],
  });
});

test('formatSummary returns an empty string when nothing changed', () => {
  const base = baseActor();
  assert.equal(formatSummary(summarize(base, base), []), '');
});

test('formatSummary escapes item names', () => {
  const html = formatSummary({level: null, added: [{type: 'feat', name: '<b>X</b>'}], removed: [], changed: [], skillRanks: []}, []);
  assert.ok(html.startsWith('<ul>'));
  assert.ok(html.includes('&lt;b&gt;X&lt;/b&gt;'));
  assert.ok(!html.includes('<b>'));
});
