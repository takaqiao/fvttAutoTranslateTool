import test from 'node:test';
import assert from 'node:assert/strict';
import {applyCopy, previewPending} from '../../scripts/main/apply.mjs';
import {catchUpCard} from '../../scripts/char/catch-up.mjs';
import {snapshotActorMedia} from '../../scripts/foundry/media.mjs';
import {mediaComparisonDocs} from '../../scripts/lib/media.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {actorDoc, applyEnvironment, makeCopy} from '../helpers/apply-environment.mjs';

const SOURCE = 'tokenizer/subject.webp';
const bytes = text => new TextEncoder().encode(text);
const doc = id => actorDoc(id, {prototypeToken: {ring: {subject: {texture: SOURCE}}}});
const subject = doc => doc.prototypeToken.ring.subject.texture;

async function frozenDoc(source, text) {
  const files = new Map([[SOURCE, bytes(text)]]);
  const result = await snapshotActorMedia([source], {
    read: async path => files.get(path), store: async (path, data) => files.set(path, data),
  });
  return {doc: result.docs[0], files};
}

function mediaEnvironment(env, files, onRead = () => {}) {
  const fetchPublished = globalThis.fetch;
  const upload = foundry.applications.apps.FilePicker.implementation.upload;
  globalThis.fetch = async url => {
    if (url.startsWith('published/')) return fetchPublished(url);
    onRead(url);
    const data = files.get(decodeURIComponent(url));
    return {ok: Boolean(data), status: data ? 200 : 404, arrayBuffer: async () => data};
  };
  foundry.applications.apps.FilePicker.implementation.upload = async (storage, dir, file) => {
    if (dir.startsWith('published/')) return upload(storage, dir, file);
    files.set(`${dir}/${file.name}`, new Uint8Array(await file.arrayBuffer()));
    return {path: `${dir}/${file.name}`};
  };
  return env;
}

async function applyScenario({sameBytes = false, onRead} = {}) {
  const submitted = await frozenDoc(doc('hero'), 'char bytes');
  const env = applyEnvironment();
  const actor = env.mapped('hero');
  actor._source.prototypeToken = doc('hero').prototypeToken;
  actor.refresh();
  env.files.set('history/main-hero-base.json', actor.toObject());
  const copy = env.publishCopy(makeCopy('hero'), submitted.doc);
  mediaEnvironment(env, new Map([...submitted.files, [SOURCE, bytes(sameBytes ? 'char bytes' : 'main bytes')]]), onRead);
  return {...env, actor, copy, submitted: submitted.doc};
}

async function catchUpScenario({sameBytes = false, sameVersion = false, onRead} = {}) {
  const main = await frozenDoc(doc('main'), 'main bytes');
  const env = applyEnvironment();
  const working = doc('work');
  working.flags[OWN_FLAG] = {base: {kind: 'main', hash: sameVersion ? 'latest' : 'base'}};
  working.items.push({_id: 'draft', name: 'Draft feat', type: 'feat', system: {}, effects: []});
  const actor = env.addActor(working);
  env.meta.cards.work = {world: 'campaign', mainId: 'main', charOwner: 'char-player'};
  env.files.set('history/main-base.json', doc('main'));
  env.files.set('main/campaign/main.json', {hash: 'latest', level: 2, applied: null, doc: main.doc});
  mediaEnvironment(env, new Map([...main.files, [SOURCE, bytes(sameBytes ? 'main bytes' : 'char draft bytes')]]), onRead);
  return {...env, actor, main: main.doc};
}

test('media comparison does not assign a legacy URL the hash claimed by another document', async () => {
  const legacy = doc('hero');
  const frozen = await frozenDoc(legacy, 'one server');
  const [base, current] = mediaComparisonDocs([legacy, frozen.doc]);
  assert.equal(subject(base), SOURCE);
  assert.notEqual(subject(base), subject(current));
});

test('live apply freezes main bytes and blocks unknown-base media divergence before Actor writes', async () => {
  const env = await applyScenario();
  const before = env.actor.toObject();
  assert.equal(await applyCopy(env.actor, env.copy, {meta: env.meta}), 'conflict');
  assert.deepEqual(env.actor.toObject(), before);
  assert.equal(env.writes.length, 0);
});

test('live apply accepts independently frozen equal bytes despite an unknown legacy base', async () => {
  const env = await applyScenario({sameBytes: true});
  assert.equal(await applyCopy(env.actor, env.copy, {meta: env.meta}), 'applied');
  assert.equal(subject(env.actor._source), subject(env.submitted));
});

test('unknown legacy item bytes versus a main deletion block automatic apply and appear in the preview', async () => {
  const env = await applyScenario({sameBytes: true});
  const item = {_id: 'feat', name: 'Feat with unknown image', type: 'feat', img: SOURCE, system: {}, effects: []};
  env.files.get('history/main-hero-base.json').items.push(item);
  env.submitted.items.push({...item, img: subject(env.submitted)});
  env.publishCopy(env.copy, env.submitted);
  assert.equal(await applyCopy(env.actor, env.copy, {meta: env.meta}), 'conflict');
  assert.equal(env.writes.length, 0);
  assert.deepEqual(env.actor._source.items, []);
  const preview = await previewPending({actorId: env.actor.id, copy: env.copy});
  assert.match(preview, /Feat with unknown image/);
  assert.match(preview, /冲突/);
  assert.equal(env.writes.length, 0);
});

test('live catch-up freezes char bytes and preserves a divergent appearance and build draft', async () => {
  const env = await catchUpScenario();
  const result = await catchUpCard(env.actor, {meta: env.meta});
  assert.deepEqual(result.conflicts, [{scope: 'actor', path: 'prototypeToken.texture', winner: 'char'}]);
  assert.notEqual(subject(env.actor._source), subject(env.main));
  assert.match(subject(env.actor._source), /^assets\/taka-charsync-media\//);
  assert.ok(env.actor._source.items.some(item => item._id === 'draft'));
});

test('same-version catch-up writes raw-to-managed migration once while preserving a build draft', async () => {
  const env = await catchUpScenario({sameBytes: true, sameVersion: true});
  const result = await catchUpCard(env.actor, {meta: env.meta});
  assert.equal(result.changed, true);
  assert.equal(subject(env.actor._source), subject(env.main));
  assert.ok(env.actor._source.items.some(item => item._id === 'draft'));
  const writes = env.writes.length;
  assert.equal((await catchUpCard(env.actor, {meta: env.meta})).action, 'none');
  assert.equal(env.writes.length, writes);
});

for (const operation of ['apply', 'catch-up']) {
  test(`${operation} aborts if the Actor changes while local media is being read`, async () => {
    let env;
    let changed = false;
    const onRead = () => {
      if (!changed) { env.actor._source.name = 'Later draft'; changed = true; }
    };
    env = operation === 'apply' ? await applyScenario({sameBytes: true, onRead}) : await catchUpScenario({sameBytes: true, onRead});
    const run = () => operation === 'apply' ? applyCopy(env.actor, env.copy, {meta: env.meta}) : catchUpCard(env.actor, {meta: env.meta});
    await assert.rejects(run, /变化|修改/);
    assert.equal(env.actor._source.name, 'Later draft');
    assert.equal(env.writes.length, 0);
  });

  test(`${operation} leaves the Actor untouched when current local media is missing`, async () => {
    const env = operation === 'apply' ? await applyScenario({sameBytes: true}) : await catchUpScenario({sameBytes: true});
    const before = env.actor.toObject();
    mediaEnvironment(env, new Map());
    const run = () => operation === 'apply' ? applyCopy(env.actor, env.copy, {meta: env.meta}) : catchUpCard(env.actor, {meta: env.meta});
    await assert.rejects(run, /读不到媒体/);
    assert.deepEqual(env.actor.toObject(), before);
    assert.equal(env.writes.length, 0);
  });
}
