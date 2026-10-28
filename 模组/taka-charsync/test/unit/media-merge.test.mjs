import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {snapshotActorMedia} from '../../scripts/foundry/media.mjs';
import {merge3, summarize} from '../../scripts/lib/merge.mjs';
import {mergeBuildChanges, projectBuildChanges} from '../../scripts/lib/build-sync.mjs';
import {describeChanges} from '../../scripts/lib/change-summary.mjs';
import {actorDoc} from '../helpers/apply-environment.mjs';

const MEDIA_FLAG = 'taka-charsync-media';
const hash = text => createHash('sha256').update(text).digest('hex');
const frozen = text => `assets/taka-charsync-media/${hash(text)}.webp`;

async function freeze(doc, path, letter, source) {
  const actual = path[0] === 'items' ? doc.items.find(item => item._id === path[1]) : doc;
  const relative = path[0] === 'items' ? path.slice(2) : path;
  let parent = actual;
  for (const part of relative.slice(0, -1)) parent = parent[part];
  const key = relative.at(-1);
  source ??= parent[key];
  const files = new Map([[source.replace(/[?#].*$/s, ''), new TextEncoder().encode(letter)]]);
  const snapshot = await snapshotActorMedia([{img: source}], {
    read: async path => files.get(path), store: async (path, bytes) => files.set(path, bytes),
  });
  parent[key] = snapshot.docs[0].img;
  const metadata = doc.flags[MEDIA_FLAG] ??= {version: 1, references: []};
  metadata.references = metadata.references.filter(entry => JSON.stringify(entry.path) !== JSON.stringify(path));
  metadata.references.push({path, source, frozen: frozen(letter), sha256: hash(letter)});
}

function versions() {
  const base = actorDoc('hero', {
    img: 'art/avatar.webp?v=1',
    flags: {'tokenizer-2': {avatarLayerStack: [{src: 'art/source.webp'}]}},
    prototypeToken: {texture: {src: 'art/token.webp'}},
    items: [{_id: 'feat', name: 'Feat', type: 'feat', img: 'art/feat.webp', system: {}, effects: []}],
  });
  return {base, main: structuredClone(base), char: structuredClone(base)};
}

test('freezing an unchanged avatar does not override a simultaneous main avatar edit', async () => {
  const docs = versions();
  await freeze(docs.base, ['img'], 'a');
  await freeze(docs.base, ['flags', 'tokenizer-2', 'avatarLayerStack', '0', 'src'], 'b');
  docs.main.img = 'art/new-main.webp';
  docs.main.flags['tokenizer-2'].avatarLayerStack = [{src: 'art/new-main-source.webp'}];
  docs.char.prototypeToken.texture.src = 'art/draft-token.webp';
  await freeze(docs.char, ['img'], 'a');
  await freeze(docs.char, ['flags', 'tokenizer-2', 'avatarLayerStack', '0', 'src'], 'b');
  const result = mergeBuildChanges(docs);
  assert.equal(result.doc.img, docs.main.img);
  assert.deepEqual(result.doc.flags['tokenizer-2'].avatarLayerStack, docs.main.flags['tokenizer-2'].avatarLayerStack);
  assert.equal(result.doc.prototypeToken.texture.src, 'art/draft-token.webp');
  assert.deepEqual(result.conflicts, []);
  assert.equal(result.doc.flags[MEDIA_FLAG], undefined);
});

test('a main avatar with new bytes at the same original URL remains a real edit', async () => {
  const docs = versions();
  for (const doc of Object.values(docs)) await freeze(doc, ['img'], 'a');
  await freeze(docs.main, ['img'], 'b', 'art/avatar.webp?v=1');
  docs.char.prototypeToken.texture.src = 'art/draft-token.webp';
  const result = mergeBuildChanges(docs);
  assert.equal(result.doc.img, frozen('b'));
  assert.equal(result.doc.prototypeToken.texture.src, 'art/draft-token.webp');
  assert.deepEqual(result.conflicts, []);
  assert.deepEqual(result.doc.flags[MEDIA_FLAG].references, docs.main.flags[MEDIA_FLAG].references);
});

test('different known bytes for one original URL leave an unresolved legacy baseline as a conflict', async () => {
  const docs = versions();
  await freeze(docs.main, ['img'], 'a');
  await freeze(docs.char, ['img'], 'b');
  const result = mergeBuildChanges(docs);
  assert.equal(result.doc.img, frozen('b'));
  assert.deepEqual(result.conflicts, [{scope: 'actor', path: 'img', winner: 'char'}]);
});

for (const target of ['main', 'char']) {
  test(`${target} leaves a deletion versus unknown legacy item bytes as an explicit conflict`, async () => {
    const docs = versions();
    docs.main.items = [];
    await freeze(docs.char, ['items', 'feat', 'img'], 'a');
    for (const merge of [merge3, mergeBuildChanges]) {
      const result = merge({...docs, target});
      assert.deepEqual(result.conflicts, [{scope: 'item', id: 'feat', name: 'Feat', path: '', winner: 'char'}]);
      assert.equal(result.doc.items[0].img, frozen('a'));
      assert.equal(docs.base.items[0].img, 'art/feat.webp');
    }
  });

  test(`${target} does not resurrect a main-deleted feat when char only freezes its image`, async () => {
    const docs = versions();
    await freeze(docs.base, ['items', 'feat', 'img'], 'a');
    docs.main.items = [];
    await freeze(docs.char, ['items', 'feat', 'img'], 'a');
    for (const merge of [merge3, mergeBuildChanges]) {
      const result = merge({...docs, target});
      assert.deepEqual(result.doc.items, []);
      assert.deepEqual(result.conflicts, []);
      assert.equal(result.doc.flags[MEDIA_FLAG], undefined);
    }
  });

  test(`${target} retains a feat changed to genuinely new image bytes against a main deletion`, async () => {
    const docs = versions();
    for (const doc of Object.values(docs)) await freeze(doc, ['items', 'feat', 'img'], 'a');
    docs.main.items = [];
    await freeze(docs.char, ['items', 'feat', 'img'], 'b', 'art/feat.webp');
    const result = mergeBuildChanges({...docs, target});
    assert.equal(result.doc.items[0].img, frozen('b'));
    assert.deepEqual(result.conflicts, [{scope: 'item', id: 'feat', name: 'Feat', path: '', winner: 'char'}]);
    assert.deepEqual(result.doc.flags[MEDIA_FLAG].references, docs.char.flags[MEDIA_FLAG].references);
  });
}

test('recursive merging uses media equality while returning actual URLs and selected metadata', async () => {
  const docs = versions();
  docs.main.items[0].name = 'Main feat';
  docs.main.img = 'art/main.webp';
  await freeze(docs.base, ['items', 'feat', 'img'], 'a');
  await freeze(docs.main, ['items', 'feat', 'img'], 'a');
  docs.main.items[0].img += '?v=1';
  await freeze(docs.base, ['img'], 'b');
  await freeze(docs.char, ['items', 'feat', 'img'], 'a');
  await freeze(docs.char, ['img'], 'b');
  const result = merge3({...docs, target: 'main'});
  assert.deepEqual(result.doc.items, [{...docs.main.items[0], img: frozen('a')}]);
  assert.equal(result.doc.img, 'art/main.webp');
  assert.deepEqual(result.conflicts, []);
  assert.deepEqual(result.doc.flags[MEDIA_FLAG].references, [docs.char.flags[MEDIA_FLAG].references[0]]);
  assert.equal(result.changed, true);
  assert.equal(merge3({...docs, main: result.doc, target: 'main'}).changed, false);
});

test('build projection rebuilds media records by item id after preserving reordered inventory', async () => {
  const {main, char} = versions();
  main.items.unshift({_id: 'coin', name: 'Coin', type: 'treasure', img: 'art/coin.webp', system: {quantity: 10}});
  await freeze(main, ['items', 'coin', 'img'], 'a');
  await freeze(char, ['img'], 'b');
  await freeze(char, ['items', 'feat', 'img'], 'c');
  const projected = projectBuildChanges(main, char);
  assert.equal(projected.img, frozen('b'));
  assert.equal(projected.items[0].img, frozen('a'));
  assert.equal(projected.items[1].img, frozen('c'));
  const references = projected.flags[MEDIA_FLAG].references;
  for (const [path, hash] of [[['img'], 'b'], [['items', 'coin', 'img'], 'a'], [['items', 'feat', 'img'], 'c']]) {
    assert.equal(references.find(entry => JSON.stringify(entry.path) === JSON.stringify(path))?.sha256, createHash('sha256').update(hash).digest('hex'));
  }
  assert.equal(references.length, 3);
});

test('same-content URL freezing is absent from draft summaries and item changes', async () => {
  const {base, char} = versions();
  await freeze(base, ['img'], 'a');
  await freeze(base, ['items', 'feat', 'img'], 'b');
  await freeze(char, ['img'], 'a');
  await freeze(char, ['items', 'feat', 'img'], 'b');
  const description = describeChanges(base, char);
  assert.equal(description.changed, false);
  assert.equal(description.html, '');
  assert.deepEqual(summarize(base, char).changed, []);
});

test('a real changed media field is displayed with its URL and without metadata', async () => {
  const {base, char} = versions();
  await freeze(base, ['img'], 'a');
  await freeze(char, ['img'], 'b');
  const description = describeChanges(base, char);
  assert.equal(description.changed, true);
  assert.equal(description.counts.fields, 1);
  assert.ok(description.html.includes(frozen('a')));
  assert.ok(description.html.includes(frozen('b')));
  assert.equal(description.html.includes(`flags.${MEDIA_FLAG}`), false);
  assert.equal(description.html.includes(`${MEDIA_FLAG}:sha256:`), false);
});

test('writing a selected managed URL without its cache query remains a real update and a repeat is unchanged', async () => {
  const docs = versions();
  await freeze(docs.main, ['img'], 'a');
  docs.base.img = docs.char.img = `${frozen('a')}?v=1`;
  const first = mergeBuildChanges({...docs, target: 'char'});
  assert.equal(first.doc.img, frozen('a'));
  assert.equal(first.changed, true);
  const second = mergeBuildChanges({...docs, char: first.doc, target: 'char'});
  assert.equal(second.changed, false);
  assert.equal(describeChanges(docs.char, first.doc).changed, false);
});

test('equivalent appearance adopts a strict superset of managed references without splitting the group', async () => {
  const docs = versions();
  await freeze(docs.main, ['img'], 'a');
  await freeze(docs.char, ['img'], 'a');
  await freeze(docs.char, ['flags', 'tokenizer-2', 'avatarLayerStack', '0', 'src'], 'b');
  docs.base.img = `${frozen('a')}?v=1`;
  for (const doc of [docs.base, docs.main]) doc.flags['tokenizer-2'].avatarLayerStack[0].src = `${frozen('b')}?v=1`;
  const result = mergeBuildChanges(docs);
  assert.equal(result.doc.img, frozen('a'));
  assert.deepEqual(result.doc.flags['tokenizer-2'], docs.char.flags['tokenizer-2']);
  assert.equal(result.changed, true);
  assert.deepEqual(result.conflicts, []);
  assert.equal(mergeBuildChanges({...docs, main: result.doc}).changed, false);
});

for (const target of ['main', 'char']) {
  test(`${target} keeps its whole equivalent appearance when managed references are incomparable`, async () => {
    const docs = versions();
    for (const doc of Object.values(docs)) doc.flags['tokenizer-2'].avatarLayerStack[0].masks = [{src: 'art/mask.webp'}];
    await freeze(docs.main, ['img'], 'a');
    await freeze(docs.char, ['flags', 'tokenizer-2', 'avatarLayerStack', '0', 'src'], 'b');
    await freeze(docs.char, ['flags', 'tokenizer-2', 'avatarLayerStack', '0', 'masks', '0', 'src'], 'c');
    for (const doc of [docs.base, docs.char]) doc.img = `${frozen('a')}?v=1`;
    for (const doc of [docs.base, docs.main]) {
      doc.flags['tokenizer-2'].avatarLayerStack[0].src = `${frozen('b')}?v=1`;
      doc.flags['tokenizer-2'].avatarLayerStack[0].masks[0].src = `${frozen('c')}?v=1`;
    }
    const result = mergeBuildChanges({...docs, target});
    assert.equal(result.doc.img, docs[target].img);
    assert.deepEqual(result.doc.flags['tokenizer-2'], docs[target].flags['tokenizer-2']);
    assert.equal(result.changed, false);
    assert.deepEqual(result.conflicts, []);
    assert.equal(mergeBuildChanges({...docs, [target]: result.doc, target}).changed, false);
  });
}
