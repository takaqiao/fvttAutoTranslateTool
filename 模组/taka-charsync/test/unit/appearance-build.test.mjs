import test from 'node:test';
import assert from 'node:assert/strict';
import {freezeMediaFixtures} from '../helpers/media-fixtures.mjs';
import {mergeBuildChanges, projectBuildChanges} from '../../scripts/lib/build-sync.mjs';
import {applyCopy} from '../../scripts/main/apply.mjs';
import {catchUpCard} from '../../scripts/char/catch-up.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {actorDoc, applyEnvironment, makeCopy} from '../helpers/apply-environment.mjs';

function setAvatar(doc, name) {
  doc.img = `${name}-avatar.webp`;
  Object.assign(doc.flags['tokenizer-2'], {
    avatarLayerStack: [{src: `${name}-portrait.webp`, masks: [{src: `${name}-mask.webp`}]}],
    avatarAspect: name === 'base' ? 'square' : 'portrait',
    avatarCanvasSize: {width: 512, height: name === 'base' ? 512 : 768},
  });
}

function setToken(doc, name, scale = 1) {
  Object.assign(doc.flags['tokenizer-2'], {
    layerStack: [{src: `${name}-body.webp`, paintMask: {src: `${name}-paint.webp`}}],
    isDynamicRingMode: true, isOversized: scale > 1, oversizedScale: scale, oversizedRingFill: false,
    modeIsOversized: scale > 1, modeOversizedScale: scale, modeOversizedRingFill: false,
  });
  Object.assign(doc.prototypeToken, {
    texture: {src: `${name}-token.webp`, scaleX: scale, scaleY: scale},
    ring: {enabled: true, subject: {texture: `${name}-subject.webp`, scale}, colors: {ring: '#ffffff'}},
  });
  doc.prototypeToken.flags['tokenizer-2'] = {oversizedScale: scale};
  doc.prototypeToken.flags.pf2e.autoscale = scale === 1;
}

function versions() {
  const base = actorDoc('main', {
    flags: {'tokenizer-2': {ornament: 'main-ornament', requests: ['main-request']}, other: {value: 'main'}},
    prototypeToken: {width: 0.5, height: 0.5, sight: {range: 60}, light: {dim: 0},
      flags: {pf2e: {autoscale: true, linkToActorSize: true}, other: {value: 'main'}}},
  });
  setAvatar(base, 'base');
  setToken(base, 'base');
  return {base, main: structuredClone(base), char: structuredClone(base)};
}

function avatar(doc) {
  const {avatarLayerStack, avatarAspect, avatarCanvasSize} = doc.flags?.['tokenizer-2'] ?? {};
  return {img: doc.img, avatarLayerStack, avatarAspect, avatarCanvasSize};
}

function token(doc) {
  const {avatarLayerStack, avatarAspect, avatarCanvasSize, ornament, requests, ...recipe} = doc.flags?.['tokenizer-2'] ?? {};
  return {recipe, texture: doc.prototypeToken.texture,
    ring: {enabled: doc.prototypeToken.ring.enabled, subject: doc.prototypeToken.ring.subject},
    tokenizer: doc.prototypeToken.flags?.['tokenizer-2'], autoscale: doc.prototypeToken.flags?.pf2e?.autoscale};
}

test('build projection carries both recipes and oversized settings without taking unrelated flags or dimensions', () => {
  const {main, char} = versions();
  setAvatar(char, 'draft');
  setToken(char, 'draft', 2);
  char.flags['tokenizer-2'].requests = ['draft-request'];
  char.flags['tokenizer-2'].ornament = 'draft-ornament';
  char.flags.other.value = 'draft';
  Object.assign(char.prototypeToken, {width: 4, height: 4, sight: {range: 5}, light: {dim: 30}});
  char.prototypeToken.flags.other.value = 'draft';
  char.prototypeToken.flags.pf2e.linkToActorSize = false;
  const before = structuredClone({main, char});
  const doc = projectBuildChanges(main, char);
  assert.deepEqual(avatar(doc), avatar(char));
  assert.deepEqual(token(doc), token(char));
  assert.equal(doc.flags['tokenizer-2'].ornament, 'main-ornament');
  assert.deepEqual(doc.flags['tokenizer-2'].requests, ['main-request']);
  assert.deepEqual(doc.flags.other, main.flags.other);
  assert.deepEqual(doc.prototypeToken.flags.other, main.prototypeToken.flags.other);
  assert.equal(doc.prototypeToken.flags.pf2e.linkToActorSize, true);
  for (const field of ['width', 'height', 'sight', 'light']) assert.deepEqual(doc.prototypeToken[field], main.prototypeToken[field]);
  assert.deepEqual({main, char}, before);
});

for (const target of ['main', 'char']) {
  test(`${target} keeps draft image URLs with their recipes and native oversized scale`, () => {
    const docs = versions();
    setAvatar(docs.char, 'draft');
    setToken(docs.char, 'draft', 2);
    docs.main.system.attributes.hp.value = 3;
    const before = structuredClone(docs);
    const result = mergeBuildChanges({...docs, target});
    assert.deepEqual(avatar(result.doc), avatar(docs.char));
    assert.deepEqual(token(result.doc), token(docs.char));
    assert.equal(result.doc.system.attributes.hp.value, 3);
    assert.equal(result.doc.prototypeToken.width, 0.5);
    assert.equal(result.doc.prototypeToken.height, 0.5);
    assert.deepEqual(result.conflicts, []);
    assert.deepEqual(docs, before);
  });

  test(`${target} resolves edits to different fields in the same appearance group as one conflict`, () => {
    const docs = versions();
    docs.main.img = 'main-avatar.webp';
    docs.main.prototypeToken.texture.src = 'main-token.webp';
    docs.char.flags['tokenizer-2'].avatarCanvasSize.height = 768;
    docs.char.flags['tokenizer-2'].oversizedScale = 2;
    const result = mergeBuildChanges({...docs, target});
    assert.deepEqual(avatar(result.doc), avatar(docs.char));
    assert.deepEqual(token(result.doc), token(docs.char));
    assert.deepEqual(result.conflicts, [
      {scope: 'actor', path: 'img', winner: 'char'},
      {scope: 'actor', path: 'prototypeToken.texture', winner: 'char'},
    ]);
  });

  test(`${target} merges independent avatar and token edits without a conflict`, () => {
    const docs = versions();
    setAvatar(docs.main, 'main');
    setToken(docs.char, 'draft', 2);
    const result = mergeBuildChanges({...docs, target});
    assert.deepEqual(avatar(result.doc), avatar(docs.main));
    assert.deepEqual(token(result.doc), token(docs.char));
    assert.deepEqual(result.conflicts, []);
  });

  test(`${target} preserves main ring colors and effects while the token image and recipe change`, () => {
    const docs = versions();
    docs.main.prototypeToken.ring.colors.ring = '#ff0000';
    docs.main.prototypeToken.ring.effects = 2;
    setToken(docs.char, 'draft', 2);
    const result = mergeBuildChanges({...docs, target});
    assert.deepEqual(token(result.doc), token(docs.char));
    assert.deepEqual(result.doc.prototypeToken.ring.colors, {ring: '#ff0000'});
    assert.equal(result.doc.prototypeToken.ring.effects, 2);
    assert.deepEqual(result.conflicts, []);
  });

  test(`${target} follows main appearance when the working card has no appearance draft`, () => {
    const docs = versions();
    setAvatar(docs.main, 'main');
    setToken(docs.main, 'main', 1.5);
    const result = mergeBuildChanges({...docs, target});
    assert.deepEqual(avatar(result.doc), avatar(docs.main));
    assert.deepEqual(token(result.doc), token(docs.main));
    assert.deepEqual(result.conflicts, []);
    assert.equal(result.changed, target === 'char');
  });

  test(`${target} deletes old recipes when the winning image source has no recipe`, () => {
    const docs = versions();
    setAvatar(docs.char, 'draft');
    setToken(docs.char, 'draft', 2);
    docs.char.flags['tokenizer-2'] = {ornament: 'draft-ornament'};
    delete docs.char.prototypeToken.flags['tokenizer-2'];
    const result = mergeBuildChanges({...docs, target});
    assert.deepEqual(avatar(result.doc), avatar(docs.char));
    assert.deepEqual(token(result.doc), token(docs.char));
    assert.deepEqual(result.doc.flags['tokenizer-2'], {ornament: 'main-ornament', requests: ['main-request']});
    assert.equal(Object.hasOwn(result.doc.prototypeToken.flags, 'tokenizer-2'), false);
  });

  test(`${target} carries an unchanged recipe when only its image URL changes`, () => {
    const docs = versions();
    docs.char.img = 'frozen-avatar.webp';
    docs.char.prototypeToken.ring.subject.texture = 'frozen-subject.webp';
    const result = mergeBuildChanges({...docs, target});
    assert.deepEqual(avatar(result.doc), avatar(docs.char));
    assert.deepEqual(token(result.doc), token(docs.char));
    assert.deepEqual(result.conflicts, []);
  });

  test(`${target} treats identical concurrent appearance updates as already merged`, () => {
    const docs = versions();
    setAvatar(docs.main, 'same');
    setToken(docs.main, 'same', 2);
    docs.char = structuredClone(docs.main);
    const result = mergeBuildChanges({...docs, target});
    assert.deepEqual(result.doc, docs[target]);
    assert.deepEqual(result.conflicts, []);
    assert.equal(result.changed, false);
  });
}

test('ordinary apply writes coherent appearance and rejects an atomic conflict before automatic apply', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  let {base, main, char} = versions();
  for (const doc of [base, main, char]) doc._id = actor.id;
  main.img = 'main-avatar.webp';
  setAvatar(char, 'draft');
  setToken(char, 'draft', 2);
  [base, main, char] = await freezeMediaFixtures([base, main, char]);
  actor._source = main;
  actor.refresh();
  env.files.set('history/main-source-base.json', base);
  const copy = env.publishCopy(makeCopy('source'), char);
  assert.equal(await applyCopy(actor, copy), 'conflict');
  assert.deepEqual(env.writes, []);
  await applyCopy(actor, copy, {manual: true});
  assert.deepEqual(avatar(actor._source), avatar(char));
  assert.deepEqual(token(actor._source), token(char));
});

test('catch-up retains an unsubmitted token recipe while adopting the main avatar', async () => {
  const env = applyEnvironment();
  let {base, main, char} = versions();
  setAvatar(main, 'main');
  setToken(char, 'draft', 2);
  [base, main, char] = await freezeMediaFixtures([base, main, char]);
  char._id = 'work';
  char.flags[OWN_FLAG] = {base: {kind: 'main', hash: 'base'}};
  env.meta.cards.work = {world: 'campaign', mainId: 'main', charOwner: 'char-player'};
  env.files.set('history/main-base.json', base);
  env.files.set('main/campaign/main.json', {hash: 'latest', readAt: 100, level: 2, applied: null, doc: main});
  const actor = env.addActor(char);
  assert.equal((await catchUpCard(actor, {meta: env.meta})).action, 'merge');
  assert.deepEqual(avatar(actor._source), avatar(main));
  assert.deepEqual(token(actor._source), token(char));
  assert.deepEqual(actor._source.flags[OWN_FLAG].base, {kind: 'main', hash: 'latest'});
  const writes = env.writes.length;
  assert.equal((await catchUpCard(actor, {meta: env.meta})).action, 'none');
  assert.equal(env.writes.length, writes);
});
