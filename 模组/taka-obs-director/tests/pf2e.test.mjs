import test from 'node:test';
import assert from 'node:assert/strict';

const adapter = await import('../scripts/pf2e.mjs').catch((error) => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});
function api(name) {
  assert.equal(typeof adapter[name], 'function', `${name} must be implemented`);
  return adapter[name];
}

function pc(overrides = {}) {
  return {
    id: 'pc', type: 'character', name: 'Zero HP hero', img: 'actor-avatar.webp',
    prototypeToken: { texture: { src: 'token-is-not-a-portrait.webp' } },
    attributes: { hp: { value: 0, max: 82, temp: 8 }, ac: { value: 27 }, shield: { itemId: null, hp: { value: 0, max: 0 }, raised: false } },
    system: { resources: { heroPoints: { value: 0, max: 3 }, mythicPoints: { value: 0, max: 0 }, focus: { value: 2, max: 3 }, investiture: { value: 2, max: 10 } } },
    movement: { speeds: { land: { type: 'land', label: '地面', value: 0 }, fly: { type: 'fly', label: '飞行', value: 30 }, travel: { type: 'travel', value: 60 }, swim: { type: 'swim', value: 0 } } },
    getStatistic: (slug) => ({ fortitude: { mod: 12 }, reflex: { mod: NaN }, will: { mod: 0 }, perception: { mod: -1 }, class: { label: '职业 DC', dc: { value: 26 } } })[slug],
    testUserPermission: (_viewer, level) => level === 'OBSERVER',
    spellcasting: { collections: new Map() }, conditions: { active: [] }, itemTypes: { effect: [] }, synthetics: { resources: {} },
    ...overrides,
  };
}
const options = () => ({ viewer: { id: 'obs', isGM: false }, allowedActorIds: new Set(['pc']), isStillAssigned: (actor) => actor.id === 'pc' });
function sheet(overrides = {}) {
  return { id: 'arcane', name: 'Arcane', sort: 0, category: 'prepared', statistic: { dc: { value: 25 } }, isPrepared: true, isFlexible: false, isSpontaneous: false, isInnate: false, isFocusPool: false, isRitual: false, isEphemeral: false, hasCollection: true, groups: [], ...overrides };
}
function spell(id, overrides = {}) {
  return { id, name: id, rank: 1, isCantrip: false, isRitual: false, system: { cast: { focusPoints: 0 }, location: {} }, ...overrides };
}
function collection(data) {
  return { entry: { getSheetData: async () => data } };
}

test('actor view preserves real HP0/temp8, prepared modifiers, land0, portrait and null shield', async () => {
  const view = await api('collectActorView')(pc(), options());
  assert.deepEqual(view.hp, { value: 0, max: 82, temp: 8 });
  assert.equal(view.portrait, 'actor-avatar.webp');
  assert.deepEqual(view.stats, [
    { label: 'AC', value: 27 }, { label: '强韧', value: 12 }, { label: '反射', value: null },
    { label: '意志', value: 0 }, { label: '察觉', value: -1 }, { label: '地面', value: 0, movementType: 'land' }, { label: '飞行', value: 30, movementType: 'fly' },
  ]);
  assert.equal(view.shield, null);
  assert.deepEqual(view.classDc, { label: '职业 DC', value: 26 });
  assert.deepEqual(view.spellcasting, []);
  assert.deepEqual(view.conditions, []);
  assert.equal(view.publicOnly, false);
  assert.equal((await api('collectActorView')(pc(), { ...options(), portraitOverride: 'confirmed-original.webp' })).portrait, 'confirmed-original.webp');
});

test('statistics project native DC-specific adjustments separately from modifiers with one lookup', async () => {
  const calls = [];
  const actor = pc({ getStatistic(slug) {
    calls.push(slug);
    return { fortitude: { mod: 12, dc: { value: 24 } }, reflex: { mod: 0, dc: { value: NaN } },
      will: { mod: -1 }, perception: { mod: null, dc: { value: 18 } }, class: { dc: { value: 26 } } }[slug];
  } });
  const view = await api('collectActorView')(actor, options());
  assert.deepEqual(view.stats.slice(0, 5), [{ label: 'AC', value: 27 }, { label: '强韧', value: 12, dc: 24 },
    { label: '反射', value: 0 }, { label: '意志', value: -1 }, { label: '察觉', value: null, dc: 18 }]);
  assert.equal(calls.filter((slug) => slug === 'fortitude').length, 1);
  assert.equal(calls.filter((slug) => slug === 'perception').length, 1);
  assert.deepEqual(view.classDc, { label: '职业 DC', value: 26 });
});

test('movement preserves native semantic types independently of localized speed labels', async () => {
  for (const label of ['Land Speed', '地面速度']) {
    const view = await api('collectActorView')(pc({ movement: { speeds: { land: { type: 'land', label, value: 25 } } } }), options());
    assert.deepEqual(view.stats.at(-1), { label, value: 25, movementType: 'land' });
    assert.equal(view.counters.find(c => c.id === 'hero-points').value, 0);
  }
});

test('binding context is mandatory and character/playerOwner never grant access', async () => {
  const collect = api('collectActorView');
  const actor = { id: 'pc', type: 'character', hasPlayerOwner: true, testUserPermission: () => true };
  for (const property of ['system', 'attributes', 'spellcasting', 'name', 'img']) {
    Object.defineProperty(actor, property, { get() { throw new Error(`private ${property}`); } });
  }
  assert.equal(await collect(actor, { viewer: {} }), null);
  assert.equal(await collect(actor, { ...options(), allowedActorIds: ['other'] }), null);
  assert.equal(await collect(actor, { ...options(), isStillAssigned: () => false }), null);
  assert.equal(await collect(actor, { ...options(), isStillAssigned: () => Promise.resolve(true) }), null);
  assert.equal(await collect(actor, { ...options(), isStillAssigned: () => { throw new Error('stale'); } }), null);
});

test('unobserved actor cannot invoke any detailed getters', async () => {
  const actor = { id: 'pc', testUserPermission: () => false };
  for (const property of ['type', 'system', 'attributes', 'name', 'img', 'spellcasting', 'getStatistic']) {
    Object.defineProperty(actor, property, { get() { throw new Error(`private ${property}`); } });
  }
  assert.equal(await api('collectActorView')(actor, options()), null);
  assert.equal(await api('collectActorView')(pc(), { ...options(), viewer: { isGM: true } }), null);
});

for (const change of ['observer', 'assignment', 'allowlist']) {
  test(`late sheet result is deleted when ${change} changes during await`, async () => {
    const collect = api('collectActorView');
    let finish;
    const pending = new Promise((resolve) => { finish = resolve; });
    let allowed = true;
    const actor = pc({ spellcasting: { collections: new Map([['arcane', { entry: { getSheetData: () => pending } }]]) } });
    const context = { ...options(), isStillAssigned: () => allowed };
    const result = collect(actor, context);
    if (change === 'observer') actor.testUserPermission = () => false;
    if (change === 'assignment') allowed = false;
    if (change === 'allowlist') context.allowedActorIds.delete('pc');
    finish(sheet());
    assert.equal(await result, null);
  });
}

test('prepared slots count usable spells, not holes, repeats or capacity', () => {
  const summary = api('summarizeSpellcasting')(sheet({ groups: [
    { id: 1, number: 1, maxRank: 1, label: '1环', uses: { value: undefined, max: 3 }, active: [{ spell: spell('A'), expended: false }, { spell: spell('A'), expended: true }, null] },
    { id: 'cantrips', maxRank: 0, label: '戏法', active: [{ spell: spell('Light', { isCantrip: true }), expended: true }] },
  ] }));
  assert.deepEqual(summary, { id: 'arcane', label: 'Arcane', dc: 25, kind: 'prepared', groups: [
    { rank: 1, label: '1环', value: 1, max: 3, unlimited: false },
    { rank: 0, label: '戏法', value: null, max: null, unlimited: true },
  ] });
});

for (const kind of ['spontaneous', 'flexible']) {
  test(`${kind} signature lists never inflate the real shared rank uses`, () => {
    const result = api('summarizeSpellcasting')(sheet({ category: kind, isPrepared: kind === 'flexible', isFlexible: kind === 'flexible', isSpontaneous: kind === 'spontaneous', flexibleAvailable: { value: 9, max: 10 }, groups: [
      { id: 2, number: 2, maxRank: 4, label: '2环', uses: { value: 1, max: 3 }, active: Array.from({ length: 9 }, () => ({ spell: spell('signature'), signature: true, virtual: true })) },
    ] }));
    assert.deepEqual(result.groups, [{ rank: 2, label: '2环', value: 1, max: 3, unlimited: false }]);
  });
}

test('focus is one counter across two sources and costly cantrips never claim unlimited', async () => {
  const focusA = sheet({ id: 'focus-a', name: 'Monk', category: 'focus', isPrepared: false, isFocusPool: true, groups: [{ id: 1, label: '专注', active: [{ spell: spell('Ki', { system: { cast: { focusPoints: 2 } } }) }] }] });
  const focusB = sheet({ id: 'focus-b', name: 'Blessing', category: 'focus', isPrepared: false, isFocusPool: true, groups: [{ id: 'cantrips', label: '戏法', active: [{ spell: spell('Chant', { isCantrip: true, system: { cast: { focusPoints: 2 } } }) }] }] });
  const actor = pc({ spellcasting: { collections: new Map([['a', collection(focusA)], ['b', collection(focusB)]]) } });
  const view = await api('collectActorView')(actor, options());
  assert.deepEqual(view.counters.filter((counter) => counter.id === 'focus'), [{ id: 'focus', label: '专注', value: 2, max: 3 }]);
  assert.equal(view.spellcasting.length, 2);
  for (const entry of view.spellcasting) {
    assert.equal(entry.kind, 'focus');
    for (const group of entry.groups) {
      assert.equal(group.value, null);
      assert.equal(group.max, null);
      assert.equal(group.unlimited, false);
      assert.match(group.label, /专注 2/);
    }
  }
});

test('innate spells have separate uses and ritual has no fabricated DC or slots', () => {
  const innate = api('summarizeSpellcasting')(sheet({ category: 'innate', isPrepared: false, isInnate: true, groups: [{ id: 1, label: '1环', active: [
    { spell: spell('First'), uses: { value: 0, max: 1 } },
    { spell: spell('Second', { system: { cast: { focusPoints: 0 }, location: { uses: { value: 2, max: 3 } } } }) },
  ] }] }));
  assert.deepEqual(innate.groups, [
    { rank: 1, label: 'First', value: 0, max: 1, unlimited: false },
    { rank: 1, label: 'Second', value: 2, max: 3, unlimited: false },
  ]);
  const ritual = api('summarizeSpellcasting')(sheet({ category: 'ritual', isPrepared: false, isRitual: true, statistic: null, groups: [{ id: 4, label: '仪式', active: [{ spell: spell('Consecrate', { isRitual: true }) }] }] }));
  assert.equal(ritual.dc, null);
  assert.deepEqual(ritual.groups, [{ rank: 4, label: '仪式', value: null, max: null, unlimited: true }]);
});

test('item casting uses genuine parent item even when entry ID impersonates another item', () => {
  const parentItem = { id: 'actual-parent-with-hyphens', name: 'Wand', category: 'wand', quantity: 4, uses: { value: 0, max: 1 }, isIdentified: true, isStowed: false };
  const scroll = { id: 'scroll', name: 'Scroll', category: 'scroll', quantity: 3, isIdentified: true, isStowed: false };
  const result = api('summarizeSpellcasting')(sheet({ id: 'fake-item-entry', category: 'items', isPrepared: false, isEphemeral: true, groups: [{ id: 1, label: '1环', active: [
    { spell: spell('Wand spell', { parentItem }) }, { spell: spell('Duplicate', { parentItem }) },
    { spell: spell('Scroll spell', { parentItem: scroll }) },
    { spell: spell('Hidden', { parentItem: { ...scroll, id: 'hidden', isIdentified: false } }) },
    { spell: spell('Stowed', { parentItem: { ...scroll, id: 'stowed', isStowed: true } }) },
  ] }] }));
  assert.equal(result.id, 'fake-item-entry');
  assert.deepEqual(result.groups, [
    { rank: 1, label: 'Wand · 库存', value: 4, max: null, unlimited: false },
    { rank: 1, label: 'Wand · 本件次数', value: 0, max: 1, unlimited: false },
    { rank: 1, label: 'Scroll · 库存', value: 3, max: null, unlimited: false },
  ]);
});

test('legacy infused reagents are actual core resources and do not need a synthetic rule', async () => {
  const actor = pc();
  actor.system.resources.crafting = { infusedReagents: { value: 0, max: 4 } };
  const view = await api('collectActorView')(actor, options());
  assert.deepEqual(view.counters.find((counter) => counter.id === 'infused-reagents'), { id: 'infused-reagents', label: '注入试剂', value: 0, max: 4 });
});

test('unknown casting capabilities have no invented unlimited or capacity pool', () => {
  const result = api('summarizeSpellcasting')(sheet({ category: 'charges', isPrepared: false, groups: [{ id: 1, label: 'Charge spell', active: [{ spell: spell('Unknown') }] }] }));
  assert.equal(result.kind, 'unknown');
  assert.deepEqual(result.groups, [{ rank: 1, label: 'Charge spell', value: null, max: null, unlimited: false }]);
});

test('each collection receives itself, failed source is isolated and returned Documents are projected', async () => {
  const collect = api('collectActorView');
  let good;
  good = { entry: { getSheetData: async (input) => {
    assert.equal(input.spells, good);
    assert.equal(input.prepList, false);
    return sheet({ groups: [{ id: 1, label: '1环', uses: { max: 1 }, active: [{ spell: spell('Safe', { description: 'DO NOT COPY', actor: { private: true } }) }] }] });
  } } };
  const actor = pc({ spellcasting: { collections: new Map([['bad', { entry: { getSheetData: async () => { throw new Error('one entry unavailable'); } } }], ['good', good]]) } });
  const view = await collect(actor, options());
  assert.equal(view.spellcasting.length, 1);
  assert.equal(JSON.stringify(view).includes('DO NOT COPY'), false);
  assert.equal(JSON.stringify(view).includes('private'), false);
});

test('mythic replaces hero and secret synthetic resource getter is never called', async () => {
  const actor = pc();
  actor.system.resources.mythicPoints = { value: 1, max: 3 };
  actor.synthetics.resources = {
    secretMana: { ignored: false, item: { isIdentified: false } },
    ignored: { ignored: true, item: { isIdentified: true } },
    publicMana: { label: 'RESOURCE.LABEL', ignored: false, item: { isIdentified: true } },
  };
  actor.getResource = (slug) => {
    if (slug === 'secret-mana' || slug === 'ignored') throw new Error('secret label leak');
    if (slug === 'public-mana') return { slug, label: 'RESOURCE.LABEL', value: 0, max: 2, description: 'private' };
    return null;
  };
  const view = await api('collectActorView')(actor, { ...options(), localize: (label) => label === 'RESOURCE.LABEL' ? '公开资源' : label });
  assert.deepEqual(view.counters, [
    { id: 'mythic-points', label: '神话点', value: 1, max: 3 }, { id: 'focus', label: '专注', value: 2, max: 3 },
    { id: 'investiture', label: '投资已占用', value: 2, max: 10 }, { id: 'public-mana', label: '公开资源', value: 0, max: 2 },
  ]);
});

test('only active identified unexpired conditions/effects are public, including for GM-independent projection', async () => {
  const privateEffect = { id: 'secret', isIdentified: false, system: { expired: false } };
  Object.defineProperty(privateEffect, 'name', { get() { throw new Error('secret effect name'); } });
  const actor = pc({
    conditions: { active: [{ id: 'frightened', name: 'Frightened 1', isIdentified: true }, { id: 'frightened', name: 'Duplicate', isIdentified: true }] },
    itemTypes: { condition: [{ id: 'inactive', name: 'Inactive', isIdentified: true }], effect: [
      privateEffect, { id: 'expired', name: 'Expired', isIdentified: true, system: { expired: true } },
      { id: 'public', name: 'Public blessing', isIdentified: true, system: { expired: false }, description: 'not public', origin: 'private origin' },
    ] },
  });
  const view = await api('collectActorView')(actor, options());
  assert.deepEqual(view.conditions, [{ id: 'frightened', label: 'Frightened 1' }, { id: 'public', label: 'Public blessing' }]);
  assert.equal(JSON.stringify(view).includes('origin'), false);
});

test('real shield uses prepared health/flags without adding AC and a changed actor has no previous resources', async () => {
  const actor = pc();
  actor.attributes.shield = { itemId: 'steel', hp: { value: 0, max: 20 }, hardness: 5, raised: true, broken: true };
  const view = await api('collectActorView')(actor, options());
  assert.deepEqual(view.shield, { hp: { value: 0, max: 20 }, hardness: 5, raised: true, broken: true });
  assert.equal(view.stats[0].value, 27);
  const empty = pc({ id: 'new', system: { resources: {} }, attributes: { hp: { value: 1, max: 2 } }, getStatistic: () => null });
  const next = await api('collectActorView')(empty, { ...options(), allowedActorIds: ['new'], isStillAssigned: () => true });
  assert.deepEqual(next.counters, []);
  assert.deepEqual(next.spellcasting, []);
  assert.equal(next.shield, null);
  assert.equal(next.classDc, null);
});
