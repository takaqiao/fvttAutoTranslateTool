import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  autoAllowed, copyIdFor, forceCopyIdFor, historyFile, isXpFull, normalizeCopy, planApply, planCatchUp,
  planForce, planNew, predOk, refKey, sameRef, syncAllowed, xpAfter,
} from '../../scripts/lib/chain.mjs';

const CARD = 'card000000000001';
const OTHER_CARD = 'card000000000002';

const mainRef = hash => ({kind: 'main', hash});
const copyRef = (copyId, rev = 1) => ({kind: 'copy', copyId, rev});
const refOf = copy => copyRef(copy.copyId, copy.rev);

const H5 = mainRef('h5');
const H6 = mainRef('h6');
const H7 = mainRef('h7');

function copyMeta(level, overrides = {}) {
  return {
    copyId: `copyL${level}`, rev: 1, cardId: CARD, level, kind: 'level', pred: null, targetWorld: 'w1',
    replaces: null, withdrawn: false, name: 'Test Hero', submittedBy: 'user0000000000001', companionOf: null,
    companions: [], ...overrides,
  };
}

// 夹具里的版本表：主服版本记等级和那一版的 applied，副本记等级和所属工作卡。
function history(entries) {
  const table = new Map(entries.map(([ref, info]) => [refKey(ref), info]));
  const info = ref => {
    const found = table.get(refKey(ref));
    assert.ok(found, `fixture has no ${refKey(ref)}`);
    return found;
  };
  return {
    levelOf: ref => info(ref).level,
    copyCard: ref => (ref.kind === 'main' ? null : ref.kind === 'char' ? ref.cardId : info(ref).card),
    appliedAt: ref => (ref.kind === 'main' ? info(ref).applied ?? null : ref),
  };
}

test('refKey covers all three ref kinds and null', () => {
  assert.equal(refKey(H6), 'main:h6');
  assert.equal(refKey(copyRef('copyA', 1700000000000)), 'copy:copyA:1700000000000');
  assert.equal(refKey({kind: 'char', cardId: 'C', hash: 'h'}), 'char:C:h');
  assert.equal(refKey(null), null);
});

test('sameRef compares by content and treats two nulls as equal', () => {
  assert.equal(sameRef(null, null), true);
  assert.equal(sameRef(mainRef('h6'), mainRef('h6')), true);
  assert.equal(sameRef(copyRef('copyA', 1), copyRef('copyA', 2)), false);
  assert.equal(sameRef(null, H6), false);
});

test('historyFile names one file per ref', () => {
  assert.equal(historyFile(H6), 'history/main-h6.json');
  assert.equal(historyFile(copyRef('copyA', 17)), 'history/copy-copyA-17.json');
  assert.equal(historyFile({kind: 'char', cardId: 'C', hash: 'h'}), 'history/char-h.json');
});

test('copyIdFor is stable, 16 base62 chars, and depends on the level', async () => {
  const id = await copyIdFor('AAAAAAAAAAAAAAAA', 7);
  assert.equal(await copyIdFor('AAAAAAAAAAAAAAAA', 7), id);
  assert.match(id, /^[0-9A-Za-z]{16}$/);
  assert.notEqual(await copyIdFor('AAAAAAAAAAAAAAAA', 8), id);
});

test('copyIdFor maps the first 16 SHA-256 bytes mod 62 onto 0-9A-Za-z', async () => {
  const alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
  const digest = createHash('sha256').update('taka-charsync|AAAAAAAAAAAAAAAA|7').digest();
  const expected = Array.from(digest.subarray(0, 16), byte => alphabet[byte % 62]).join('');
  assert.equal(await copyIdFor('AAAAAAAAAAAAAAAA', 7), expected);
});

test('forceCopyIdFor is stable and never collides with a level copy id of the same card', async () => {
  const id = await forceCopyIdFor('C');
  assert.equal(await forceCopyIdFor('C'), id);
  assert.match(id, /^[0-9A-Za-z]{16}$/);
  for (let level = 1; level <= 30; level++) assert.notEqual(await copyIdFor('C', level), id);
});

test('xp is full only when max is positive and value reaches it', () => {
  assert.equal(isXpFull({value: 1000, max: 1000}), true);
  assert.equal(isXpFull({value: 999, max: 1000}), false);
  assert.equal(isXpFull({value: 0, max: 0}), false);
});

test('xpAfter subtracts max per level and never goes below zero', () => {
  assert.equal(xpAfter({value: 1000, max: 1000}, 1), 0);
  assert.equal(xpAfter({value: 1200, max: 1000}, 1), 200);
  assert.equal(xpAfter({value: 1000, max: 1000}, 2), 0);
});

test('catch-up falls back to bootstrap and does nothing when main has not changed', () => {
  const plan = planCatchUp({
    cardId: CARD, mapped: true, base: undefined, bootstrap: H6,
    mainLatest: {hash: 'h6', level: 6, applied: null}, ...history([[H6, {level: 6}]]),
  });
  assert.deepEqual(plan, {action: 'none', mergeBase: null, newBase: H6});
});

test('catch-up merges from base when main changed without a level change', () => {
  const plan = planCatchUp({
    cardId: CARD, mapped: true, base: H6, bootstrap: undefined,
    mainLatest: {hash: 'h6b', level: 6, applied: null}, ...history([[H6, {level: 6}]]),
  });
  assert.deepEqual(plan, {action: 'merge', mergeBase: H6, newBase: mainRef('h6b')});
});

test('catch-up fast-forwards to this card\'s applied copy', () => {
  const c7 = copyRef('copyL7');
  const plan = planCatchUp({
    cardId: CARD, mapped: true, base: H6, bootstrap: undefined,
    mainLatest: {hash: 'h7', level: 7, applied: c7},
    ...history([[H6, {level: 6}], [c7, {level: 7, card: CARD}]]),
  });
  assert.deepEqual(plan, {action: 'merge', mergeBase: c7, newBase: H7});
});

test('catch-up fast-forwards to the later of two copies applied in a row', () => {
  const c8 = copyRef('copyL8');
  const plan = planCatchUp({
    cardId: CARD, mapped: true, base: H6, bootstrap: undefined,
    mainLatest: {hash: 'h8', level: 8, applied: c8},
    ...history([[H6, {level: 6}], [copyRef('copyL7'), {level: 7, card: CARD}], [c8, {level: 8, card: CARD}]]),
  });
  assert.deepEqual(plan, {action: 'merge', mergeBase: c8, newBase: mainRef('h8')});
});

test('catch-up does not fast-forward again once the base already covers the applied copy', () => {
  const c7 = copyRef('copyL7');
  const lookups = history([[H7, {level: 7, applied: c7}], [c7, {level: 7, card: CARD}]]);
  const unchanged = planCatchUp({
    cardId: CARD, mapped: true, base: H7, bootstrap: undefined,
    mainLatest: {hash: 'h7', level: 7, applied: c7}, ...lookups,
  });
  assert.deepEqual(unchanged, {action: 'none', mergeBase: null, newBase: H7});
  const changed = planCatchUp({
    cardId: CARD, mapped: true, base: H7, bootstrap: undefined,
    mainLatest: {hash: 'h7b', level: 7, applied: c7}, ...lookups,
  });
  assert.deepEqual(changed, {action: 'merge', mergeBase: H7, newBase: mainRef('h7b')});
});

test('catch-up resets after a level-up made directly on main', () => {
  const plan = planCatchUp({
    cardId: CARD, mapped: true, base: H6, bootstrap: undefined,
    mainLatest: {hash: 'h7', level: 7, applied: null}, ...history([[H6, {level: 6}]]),
  });
  assert.deepEqual(plan, {action: 'reset', mergeBase: null, newBase: H7});
});

test('catch-up resets when main levelled up but applied still names the copy the base absorbed', () => {
  const c6 = copyRef('copyL6');
  const plan = planCatchUp({
    cardId: CARD, mapped: true, base: H6, bootstrap: undefined,
    mainLatest: {hash: 'h7', level: 7, applied: c6},
    ...history([[H6, {level: 6, applied: c6}], [c6, {level: 6, card: CARD}]]),
  });
  assert.deepEqual(plan, {action: 'reset', mergeBase: null, newBase: H7});
});

test('catch-up ignores a copy of another card applied on main', () => {
  const foreign = copyRef('copyOther');
  const plan = planCatchUp({
    cardId: CARD, mapped: true, base: H6, bootstrap: undefined,
    mainLatest: {hash: 'h6b', level: 6, applied: foreign},
    ...history([[H6, {level: 6}], [foreign, {level: 6, card: OTHER_CARD}]]),
  });
  assert.deepEqual(plan, {action: 'merge', mergeBase: H6, newBase: mainRef('h6b')});
});

test('catch-up waits while there is no main version', () => {
  const plan = planCatchUp({
    cardId: CARD, mapped: true, base: H6, bootstrap: undefined, mainLatest: null, ...history([]),
  });
  assert.deepEqual(plan, {action: 'wait', mergeBase: null, newBase: null});
});

test('catch-up without any base reports unmapped or waits for a mapped card', () => {
  const args = {cardId: CARD, base: undefined, bootstrap: undefined,
                mainLatest: {hash: 'h6', level: 6, applied: null}, ...history([])};
  assert.deepEqual(planCatchUp({...args, mapped: false}), {action: 'unmapped', mergeBase: null, newBase: null});
  assert.deepEqual(planCatchUp({...args, mapped: true}), {action: 'wait', mergeBase: null, newBase: null});
});

test('catch-up fast-forwards to a whole-card overwrite of this card', () => {
  const charVersion = {kind: 'char', cardId: CARD, hash: 'hc'};
  const plan = planCatchUp({
    cardId: CARD, mapped: true, base: H6, bootstrap: undefined,
    mainLatest: {hash: 'h6c', level: 6, applied: charVersion},
    ...history([[H6, {level: 6}], [charVersion, {level: 6}]]),
  });
  assert.deepEqual(plan, {action: 'merge', mergeBase: charVersion, newBase: mainRef('h6c')});
});

test('catch-up without any base fast-forwards to this card\'s applied version, mapped or not', () => {
  const c6 = copyRef('copyL6');
  const charVersion = {kind: 'char', cardId: CARD, hash: 'hc'};
  const args = {cardId: CARD, base: null, bootstrap: null,
                ...history([[c6, {level: 6, card: CARD}], [charVersion, {level: 6}]])};
  for (const mapped of [true, false]) {
    for (const applied of [c6, charVersion]) {
      const plan = planCatchUp({...args, mapped, mainLatest: {hash: 'h6b', level: 6, applied}});
      assert.deepEqual(plan, {action: 'merge', mergeBase: applied, newBase: mainRef('h6b')});
    }
  }
});

test('predOk with a copy as predecessor requires main to have applied exactly that copy', () => {
  const {appliedAt} = history([]);
  const copy = copyMeta(7, {pred: copyRef('copyL6')});
  assert.equal(predOk(copy, copyRef('copyL6'), appliedAt), true);
  assert.equal(predOk(copy, copyRef('copyL5'), appliedAt), false);
});

test('predOk with a main version as predecessor requires main to have applied nothing since', () => {
  const {appliedAt} = history([[H6, {level: 6, applied: null}]]);
  const copy = copyMeta(7, {pred: H6});
  assert.equal(predOk(copy, null, appliedAt), true);
  assert.equal(predOk(copy, copyRef('copyL7'), appliedAt), false);
});

test('predOk with a char version as predecessor requires main to have applied exactly that version', () => {
  const {appliedAt} = history([]);
  const copy = copyMeta(7, {pred: {kind: 'char', cardId: CARD, hash: 'hc'}});
  assert.equal(predOk(copy, {kind: 'char', cardId: CARD, hash: 'hc'}, appliedAt), true);
  assert.equal(predOk(copy, {kind: 'char', cardId: CARD, hash: 'hd'}, appliedAt), false);
});

test('predOk without a predecessor holds only for a new card', () => {
  const {appliedAt} = history([]);
  assert.equal(predOk(copyMeta(1, {kind: 'new'}), null, appliedAt), true);
  assert.equal(predOk(copyMeta(7), null, appliedAt), false);
});

const C7 = copyMeta(7, {pred: H6});
const C8 = copyMeta(8, {pred: refOf(C7)});
const applyLookups = history([
  [H5, {level: 5}], [H6, {level: 6}], [refOf(C7), {level: 7, card: CARD}],
]);

function apply(copies, {main = {}, auto = true} = {}) {
  return planApply({
    main: {level: 6, xp: {value: 1000, max: 1000}, applied: null, undone: [], ...main},
    copies, appliedAt: applyLookups.appliedAt, levelOf: applyLookups.levelOf, autoAllowed: auto,
  });
}

test('apply picks the lowest pending level copy when everything is ready', () => {
  assert.deepEqual(apply([C8, C7]), {action: 'auto', copy: C7, reason: 'ready'});
});

test('apply skips copies below main level and the copy main already applied', () => {
  const old = copyMeta(5, {pred: H5});
  assert.deepEqual(apply([old, C7, C8], {main: {level: 7, applied: refOf(C7)}}),
                   {action: 'auto', copy: C8, reason: 'ready'});
});

test('apply asks the GM while a session is running', () => {
  assert.deepEqual(apply([C7, C8], {auto: false}), {action: 'gm', copy: C7, reason: 'session'});
});

test('apply waits until xp is full', () => {
  assert.deepEqual(apply([C7, C8], {main: {xp: {value: 999, max: 1000}}}), {action: 'wait', copy: C7, reason: 'xp'});
});

test('apply asks the GM when the copy skips a level', () => {
  const skip = copyMeta(8, {pred: H6});
  assert.deepEqual(apply([skip]), {action: 'gm', copy: skip, reason: 'gap'});
});

test('apply asks the GM for a same-level copy', () => {
  const same = copyMeta(6, {pred: H6});
  assert.deepEqual(apply([same]), {action: 'gm', copy: same, reason: 'same-level'});
});

test('apply skips an undone copy, leaving its successor without a predecessor', () => {
  const undone = [refKey(refOf(C7))];
  assert.deepEqual(apply([C7, C8], {main: {undone}}), {action: 'gm', copy: C8, reason: 'pred'});
  assert.deepEqual(apply([C7], {main: {undone}}), {action: 'none', copy: null, reason: 'none'});
});

test('apply skips a withdrawn copy, leaving its successor without a predecessor', () => {
  const withdrawn = {...C7, withdrawn: true};
  assert.deepEqual(apply([withdrawn, C8]), {action: 'gm', copy: C8, reason: 'pred'});
  assert.deepEqual(apply([withdrawn]), {action: 'none', copy: null, reason: 'none'});
});

test('apply asks the GM when the predecessor does not match what main applied', () => {
  const orphan = copyMeta(7, {pred: copyRef('copyL6')});
  assert.deepEqual(apply([orphan]), {action: 'gm', copy: orphan, reason: 'pred'});
});

test('apply flags a copy as stale after a level-up made directly on main', () => {
  assert.deepEqual(apply([C7], {main: {level: 7, applied: null}}), {action: 'gm', copy: C7, reason: 'stale'});
});

test('apply leaves new-card and whole-card copies to their own planners', () => {
  const force = copyMeta(6, {kind: 'force', copyId: 'copyForce', pred: null});
  const fresh = copyMeta(1, {kind: 'new', copyId: 'copyNew', pred: null});
  assert.deepEqual(apply([force]), {action: 'none', copy: null, reason: 'none'});
  assert.deepEqual(apply([force, fresh]), {action: 'none', copy: null, reason: 'none'});
});

const FROZEN_NAME = 'Test Pet〔已提交 7 级〕';

function frozenDoc(type, system, tokenName = 'Token Label') {
  return {
    _id: 'copyDoc000000001',
    name: FROZEN_NAME,
    type,
    prototypeToken: {name: tokenName},
    system,
    flags: {'taka-charsync': {copy: copyMeta(7, {name: 'Test Pet'})}, other: {kept: true}},
  };
}

test('normalizeCopy restores id, name and the familiar master without touching the input', () => {
  const doc = frozenDoc('familiar', {master: {id: 'masterCopy000001'}});
  const out = normalizeCopy(doc, {actorId: 'M1', masterId: 'P1'});
  assert.equal(out._id, 'M1');
  assert.equal(out.name, 'Test Pet');
  assert.equal(out.prototypeToken.name, 'Token Label');
  assert.equal(out.system.master.id, 'P1');
  assert.equal(out.flags['taka-charsync'].copy, undefined);
  assert.deepEqual(out.flags.other, {kept: true});
  assert.deepEqual(doc, frozenDoc('familiar', {master: {id: 'masterCopy000001'}}));
});

test('normalizeCopy restores the token name only when it matched the frozen actor name', () => {
  const followed = normalizeCopy(frozenDoc('character', {}, FROZEN_NAME), {actorId: 'A1', masterId: null});
  assert.equal(followed.prototypeToken.name, 'Test Pet');
  const tokenless = frozenDoc('character', {});
  delete tokenless.prototypeToken;
  const out = normalizeCopy(tokenless, {actorId: 'A1', masterId: null});
  assert.equal(out.name, 'Test Pet');
  assert.equal(out.prototypeToken, undefined);
});

test('normalizeCopy leaves system.master alone unless the doc is a familiar and a master is given', () => {
  const character = normalizeCopy(frozenDoc('character', {details: {level: {value: 7}}}), {actorId: 'A1', masterId: 'P1'});
  assert.deepEqual(character.system, {details: {level: {value: 7}}});
  const familiar = normalizeCopy(frozenDoc('familiar', {master: {id: 'masterCopy000001'}}), {actorId: 'M1', masterId: null});
  assert.equal(familiar.system.master.id, 'masterCopy000001');
});

test('planNew keeps only pending new-card copies aimed at this world for unmapped cards', () => {
  const wanted = copyMeta(1, {kind: 'new', copyId: 'newOk', cardId: 'cardNew000000001'});
  const copies = [
    wanted,
    copyMeta(1, {kind: 'new', copyId: 'newElsewhere', cardId: 'cardNew000000002', targetWorld: 'w2'}),
    copyMeta(1, {kind: 'new', copyId: 'newWithdrawn', cardId: 'cardNew000000003', withdrawn: true}),
    copyMeta(1, {kind: 'new', copyId: 'newUndone', cardId: 'cardNew000000004'}),
    copyMeta(1, {kind: 'new', copyId: 'newMapped', cardId: 'cardNew000000005'}),
    copyMeta(7, {copyId: 'levelCopy', cardId: 'cardNew000000006', pred: H6}),
  ];
  const plan = planNew({
    world: 'w1', copies, mappedCardIds: new Set(['cardNew000000005']), undone: new Set([refKey(copyRef('newUndone'))]),
  });
  assert.deepEqual(plan, [wanted]);
});

test('planForce returns the pending whole-card copy only', () => {
  const force = copyMeta(6, {kind: 'force', copyId: 'copyForce', pred: null});
  const level = copyMeta(7, {pred: H6});
  assert.equal(planForce({copies: [level, force], undone: new Set(), mainApplied: null}), force);
  assert.equal(planForce({copies: [{...force, withdrawn: true}], undone: new Set(), mainApplied: null}), null);
  assert.equal(planForce({copies: [force], undone: new Set([refKey(refOf(force))]), mainApplied: null}), null);
  assert.equal(planForce({copies: [force], undone: new Set(), mainApplied: refOf(force)}), null);
  assert.equal(planForce({copies: [level], undone: new Set(), mainApplied: null}), null);
});

test('autoAllowed only with no players online and no started combat', () => {
  assert.equal(autoAllowed({playersOnline: 0, combatStarted: false}), true);
  assert.equal(autoAllowed({playersOnline: 1, combatStarted: false}), false);
  assert.equal(autoAllowed({playersOnline: 0, combatStarted: true}), false);
});

test('syncAllowed requires both schemas present and equal', () => {
  const meta = {char: {schema: 's1'}, worlds: {w1: {schema: 's1'}, w2: {schema: 's0'}, w3: {}}};
  assert.equal(syncAllowed(meta, 'w1'), true);
  assert.equal(syncAllowed(meta, 'w2'), false);
  assert.equal(syncAllowed(meta, 'missing'), false);
  assert.equal(syncAllowed({char: {}, worlds: {w3: {}}}, 'w3'), false);
  assert.equal(syncAllowed({char: {schema: null}, worlds: {w1: {schema: null}}}, 'w1'), false);
  assert.equal(syncAllowed({worlds: {w1: {schema: 's1'}}}, 'w1'), false);
  assert.equal(syncAllowed({char: {schema: 's1'}}, 'w1'), false);
});
