import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {collectMediaReferences, MANAGED_MEDIA_ROOT, managedMediaHash} from '../../scripts/lib/media.mjs';
import * as media from '../../scripts/lib/media.mjs';
import {snapshotActorMedia} from '../../scripts/foundry/media.mjs';

const bytes = value => new TextEncoder().encode(value);
const hash = value => createHash('sha256').update(value).digest('hex');
const managed = (value, ext = 'png') => `${MANAGED_MEDIA_ROOT}/${hash(value)}.${ext}`;

test('media collection includes both recipes and mask dependencies without traversing other flags', () => {
  const doc = {
    img: 'avatar.png', items: [{img: 'item.png'}],
    prototypeToken: {texture: {src: 'token.webp'}, ring: {subject: {texture: 'subject.webp'}},
      flags: {'tokenizer-2': {layerStack: [{src: 'prototype.png'}]}}},
    flags: {
      'tokenizer-2': {
        layerStack: [{src: 'body.png', masks: [{src: 'mask.png'}], paintMask: {src: 'paint.png'}},
          {ringConfig: {texture: 'ring.png'}}, {ringConfig: {texture: 'gold'}}],
        avatarLayerStack: [{src: 'portrait.png', masks: [{src: 'avatar-mask.png'}]}],
        unrelated: {src: 'ignored.png'},
      },
      other: {src: 'private.json'},
    },
  };
  const refs = collectMediaReferences(doc);
  assert.deepEqual(new Set(refs.map(ref => ref.src)), new Set([
    'avatar.png', 'item.png', 'token.webp', 'subject.webp', 'prototype.png',
    'body.png', 'mask.png', 'paint.png', 'ring.png', 'portrait.png', 'avatar-mask.png',
  ]));
  for (const {path, src} of refs) {
    assert.ok(path.every(key => typeof key === 'string'));
    assert.equal(path.reduce((value, key) => value[key], doc), src);
  }
});

test('managed paths expose hashes only for exact content-addressed media files', () => {
  const digest = 'a'.repeat(64);
  assert.equal(managedMediaHash(`${MANAGED_MEDIA_ROOT}/${digest}.webp`), digest);
  for (const path of [`${MANAGED_MEDIA_ROOT}/${digest}.json`, `${MANAGED_MEDIA_ROOT}/${digest}.png?v=1`,
    `${MANAGED_MEDIA_ROOT}/../${digest}.png`, `other/${digest}.png`, `${MANAGED_MEDIA_ROOT}/${digest.slice(1)}.png`]) {
    assert.equal(managedMediaHash(path), null, path);
  }
});

test('generated ring masks preserve layer links while all four recipe locations freeze media', async () => {
  const layers = [
    {id: 'body-layer', type: 'image', src: 'assets/body.webp', masks: [
      {id: 'ring-mask', type: 'ring-frame', src: 'assets/ring-id.png', inverted: true},
      {id: 'orphan-mask', type: 'ring-frame', src: 'missing-ring'},
      {id: 'frame-mask', type: 'frame-ray', src: 'assets/frame.png'},
      {id: 'custom-mask', type: 'custom', src: 'assets/custom.png', frameId: 'assets/ring-id.png'},
    ], paintMask: {src: 'assets/paint.png'}},
    {id: 'assets/ring-id.png', type: 'ring', ringConfig: {texture: 'assets/ring-texture.png'}},
  ];
  const recipe = {layerStack: layers, avatarLayerStack: structuredClone(layers)};
  const doc = {flags: {'tokenizer-2': recipe}, prototypeToken: {flags: {'tokenizer-2': structuredClone(recipe)}}};
  const before = structuredClone(doc);
  const {io} = memoryMedia({
    'assets/body.webp': 'body', 'assets/frame.png': 'frame', 'assets/custom.png': 'custom',
    'assets/paint.png': 'paint', 'assets/ring-texture.png': 'texture',
  });
  const result = await snapshotActorMedia([doc], io);
  assert.equal(result.manifest.assets.length, 5);
  assert.deepEqual(new Set(result.manifest.assets.map(asset => asset.src)), new Set([
    'assets/body.webp', 'assets/frame.png', 'assets/custom.png', 'assets/paint.png', 'assets/ring-texture.png',
  ]));
  for (const parent of [result.docs[0], result.docs[0].prototypeToken]) {
    for (const key of ['layerStack', 'avatarLayerStack']) {
      const [body, ring] = parent.flags['tokenizer-2'][key];
      assert.equal(body.id, 'body-layer');
      assert.equal(ring.id, 'assets/ring-id.png');
      assert.equal(body.src, managed('body', 'webp'));
      assert.equal(body.paintMask.src, managed('paint'));
      assert.equal(ring.ringConfig.texture, managed('texture'));
      assert.deepEqual(body.masks, [
        {id: 'ring-mask', type: 'ring-frame', src: ring.id, inverted: true},
        {id: 'orphan-mask', type: 'ring-frame', src: 'missing-ring'},
        {id: 'frame-mask', type: 'frame-ray', src: managed('frame')},
        {id: 'custom-mask', type: 'custom', src: managed('custom'), frameId: ring.id},
      ]);
    }
  }
  const refs = result.docs[0].flags['taka-charsync-media'].references;
  assert.equal(refs.length, 20);
  assert.ok(refs.every(ref => ref.source !== 'assets/ring-id.png' && ref.source !== 'missing-ring'));
  assert.deepEqual(doc, before);
});

test('ordinary masks still reject invalid or missing media alongside a generated ring mask', async () => {
  for (const type of [undefined, 'alpha', 'frame-ray', 'frame-marching', 'custom', 'unknown']) {
    for (const src of ['not-a-media-file', 'assets/missing.png']) {
      const doc = {flags: {'tokenizer-2': {layerStack: [{masks: [
        {type: 'ring-frame', src: 'ring-layer'}, {type, src},
      ]}]}}};
      await assert.rejects(snapshotActorMedia([doc], {
        read: async () => { throw new Error('missing image dependency'); }, store: async () => {},
      }), src === 'not-a-media-file' ? /无法冻结媒体路径：not-a-media-file/ : /missing image dependency/);
    }
  }
});

test('changing bytes under the same filename creates a new snapshot without altering the first', async () => {
  let source = bytes('abc');
  const stored = new Map();
  const io = {read: async () => source, store: async (path, data) => stored.set(path, data.slice())};
  const doc = {img: 'tokenizer/avatar.png?v=1'};
  const first = await snapshotActorMedia([doc], io);
  source = bytes('changed');
  const second = await snapshotActorMedia([doc], io);
  assert.equal(first.docs[0].img, `${MANAGED_MEDIA_ROOT}/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad.png`);
  assert.notEqual(first.docs[0].img, second.docs[0].img);
  assert.deepEqual(stored.get(first.docs[0].img), bytes('abc'));
  assert.deepEqual(doc, {img: 'tokenizer/avatar.png?v=1'});
  assert.equal(first.docs[0].flags?.['taka-charsync'], undefined);
});

test('one group snapshot rewrites item, avatar, token and mask paths after freezing their bytes', async () => {
  const reads = [];
  const stored = new Map();
  const doc = {img: 'assets/face%20one.png?v=2', items: [{img: 'icons/sword.svg'}],
    flags: {'tokenizer-2': {avatarLayerStack: [{src: 'assets/face%20one.png?v=3'}],
      layerStack: [{src: 'worlds/test/body.webp', masks: [{src: 'assets/mask.png'}], paintMask: {src: 'assets/paint.png'}}]}},
    prototypeToken: {texture: {src: 'modules/tokenizer-2/frame.webp'}, ring: {subject: {texture: 'https://example.org/remote.webp'}}}};
  const original = structuredClone(doc);
  const result = await snapshotActorMedia([doc, {items: [{img: 'assets/item.png'}]}], {
    read: async path => {
      if (stored.has(path)) return stored.get(path);
      reads.push(path);
      return bytes(path);
    },
    store: async (path, data) => stored.set(path, data),
  });
  assert.equal(reads.length, 5);
  assert.deepEqual(new Set(reads), new Set(['assets/face one.png', 'worlds/test/body.webp', 'assets/mask.png', 'assets/paint.png', 'assets/item.png']));
  assert.equal(result.docs[0].img, managed('assets/face one.png'));
  assert.equal(result.docs[0].flags['tokenizer-2'].layerStack[0].paintMask.src, managed('assets/paint.png'));
  assert.equal(result.docs[1].items[0].img, managed('assets/item.png'));
  assert.equal(result.docs[0].prototypeToken.texture.src, original.prototypeToken.texture.src);
  assert.equal(result.docs[0].prototypeToken.ring.subject.texture, original.prototypeToken.ring.subject.texture);
  assert.equal(result.manifest.assets.length, 5);
  assert.ok(result.manifest.external.includes('https://example.org/remote.webp'));
  assert.deepEqual(doc, original);
});

test('missing dependencies, corrupt managed media and failed stores reject the snapshot', async () => {
  const doc = {img: 'assets/face.png', flags: {'tokenizer-2': {layerStack: [{masks: [{src: 'assets/missing.png'}]}]}}};
  await assert.rejects(snapshotActorMedia([doc], {
    read: async path => { if (path.includes('missing')) throw new Error('missing mask'); return bytes('ok'); },
    store: async () => {},
  }), /missing mask/);
  await assert.rejects(snapshotActorMedia([{img: managed('abc')}], {
    read: async () => bytes('wrong'), store: async () => assert.fail('must not replace corrupt managed content'),
  }), /hash|哈希/i);
  await assert.rejects(snapshotActorMedia([{img: 'assets/face.png'}], {
    read: async () => bytes('ok'), store: async () => { throw new Error('upload failed'); },
  }), /upload failed/);
});

test('ephemeral and unsafe local sources cannot be published as frozen media', async () => {
  for (const src of ['blob:https://example.org/temp', '../secret.png', 'assets/%2e%2e/secret.png', 'assets/card.json', 'assets/token-*.webp']) {
    await assert.rejects(snapshotActorMedia([{img: src}], {read: async () => assert.fail(src), store: async () => {}}), /媒体|图片|路径/);
  }
});

test('browser storage verifies existing bytes and the upload readback instead of trusting an upload receipt', async () => {
  const files = new Map([['assets/original.png', bytes('abc')]]);
  let uploadCount = 0;
  let corruptUpload = false;
  globalThis.foundry = {utils: {getRoute: path => path}, applications: {apps: {FilePicker: {implementation: {
    async upload(source, folder, file) {
      assert.equal(source, 'data');
      uploadCount++;
      const path = `${folder}/${file.name}`;
      files.set(path, corruptUpload ? bytes('truncated') : new Uint8Array(await file.arrayBuffer()));
      return {path};
    },
  }}}}};
  globalThis.fetch = async (url, options) => {
    assert.equal(options.cache, 'no-store');
    const data = files.get(decodeURIComponent(url));
    return {ok: Boolean(data), status: data ? 200 : 404, arrayBuffer: async () => data.buffer};
  };
  const first = await snapshotActorMedia([{img: 'assets/original.png'}]);
  assert.equal(first.docs[0].img, managed('abc'));
  await snapshotActorMedia([{img: 'assets/original.png'}]);
  assert.equal(uploadCount, 1);
  files.set(managed('abc'), bytes('corrupt'));
  await assert.rejects(snapshotActorMedia([{img: 'assets/original.png'}]), /哈希/);
  assert.equal(uploadCount, 1);
  files.delete(managed('abc'));
  corruptUpload = true;
  await assert.rejects(snapshotActorMedia([{img: 'assets/original.png'}]), /上传后哈希/);
});

test('a player can upload without file browsing or directory creation privileges', async () => {
  const files = new Map([['assets/original.png', bytes('abc')]]);
  let createCalls = 0;
  let uploads = 0;
  globalThis.foundry = {utils: {getRoute: path => path}, applications: {apps: {FilePicker: {implementation: {
    async createDirectory() { createCalls++; throw new Error('You may not create directories in this location'); },
    async upload(source, target, file) {
      assert.equal(source, 'data');
      assert.equal(target, MANAGED_MEDIA_ROOT);
      uploads++;
      const path = `${target}/${file.name}`;
      files.set(path, new Uint8Array(await file.arrayBuffer()));
      return {path};
    },
  }}}}};
  globalThis.fetch = async url => {
    const data = files.get(decodeURIComponent(url));
    return {ok: Boolean(data), status: data ? 200 : 404, arrayBuffer: async () => data.buffer};
  };
  const result = await snapshotActorMedia([{img: 'assets/original.png'}]);
  assert.equal(result.docs[0].img, managed('abc'));
  assert.deepEqual(files.get(result.docs[0].img), bytes('abc'));
  assert.equal(uploads, 1);
  assert.equal(createCalls, 0);
});

test('an upload failure retains the server error and identifies directory preparation and upload permission', async () => {
  let uploads = 0;
  globalThis.foundry = {utils: {getRoute: path => path}, applications: {apps: {FilePicker: {implementation: {
    async createDirectory() { throw new Error('You may not create directories in this location'); },
    async upload() { uploads++; throw new Error('Target directory does not exist'); },
  }}}}};
  globalThis.fetch = async url => url === 'assets/original.png'
    ? {ok: true, status: 200, arrayBuffer: async () => bytes('abc').buffer}
    : {ok: false, status: 404};
  await assert.rejects(snapshotActorMedia([{img: 'assets/original.png'}]), error => {
    assert.match(error.message, /媒体保存失败.*同步引擎.*素材目录.*文件上传权限/);
    assert.match(error.message, /Target directory does not exist/);
    assert.equal(error.cause.message, 'Target directory does not exist');
    return true;
  });
  assert.equal(uploads, 1);
});

function memoryMedia(initial) {
  const files = new Map(Object.entries(initial).map(([path, value]) => [path, bytes(value)]));
  return {files, io: {
    read: async path => { assert.ok(files.has(path), `missing ${path}`); return files.get(path); },
    store: async (path, data) => files.set(path, data.slice()),
  }};
}

test('snapshots record exact source URLs and stable item ids independently of sync flags', async () => {
  const {io} = memoryMedia({'assets/face.png': 'abc', 'assets/item.png': 'item'});
  const original = {img: 'assets/face.png?v=1', items: [{_id: 'item-id', img: 'assets/item.png'}],
    flags: {'taka-charsync': {copy: {rev: 123}}}};
  const {docs} = await snapshotActorMedia([original], io);
  assert.deepEqual(docs[0].flags['taka-charsync-media'], {version: 1, references: [
    {path: ['img'], source: 'assets/face.png?v=1', frozen: managed('abc'), sha256: hash('abc')},
    {path: ['items', 'item-id', 'img'], source: 'assets/item.png', frozen: managed('item'), sha256: hash('item')},
  ]});
  assert.deepEqual(docs[0].flags['taka-charsync'], original.flags['taka-charsync']);
  const again = await snapshotActorMedia(docs, io);
  assert.deepEqual(again.docs[0].flags['taka-charsync-media'], docs[0].flags['taka-charsync-media']);
  assert.equal(original.flags['taka-charsync-media'], undefined);
});

test('media comparison equates independently frozen bytes without metadata appearing as a change', async () => {
  const {io} = memoryMedia({'assets/face.png': 'abc'});
  const old = {img: 'assets/face.png?v=1'};
  const {docs: [frozen]} = await snapshotActorMedia([old], io);
  const independent = (await snapshotActorMedia([old], memoryMedia({'assets/face.png': 'abc'}).io)).docs[0];
  const inputs = [independent, frozen];
  const before = structuredClone(inputs);
  const compared = media.mediaComparisonDocs(inputs);
  assert.deepEqual(compared[0], compared[1]);
  assert.equal(compared[1].flags?.['taka-charsync-media'], undefined);
  assert.deepEqual(inputs, before);
  const [otherQuery] = media.mediaComparisonDocs([{img: 'assets/face.png?v=2'}, frozen]);
  assert.equal(otherQuery.img, 'assets/face.png?v=2');
});

test('comparison preserves an empty flags object that did not contain media metadata', () => {
  const doc = {img: 'icons/avatar.svg', flags: {}};
  assert.deepEqual(media.mediaComparisonDocs([doc]), [doc]);
});

test('metadata-only flags compare consistently with both missing and empty original flags', async () => {
  const {io} = memoryMedia({'assets/face.png': 'abc'});
  const old = {img: 'assets/face.png', flags: {}};
  const frozen = (await snapshotActorMedia([old], io)).docs[0];
  const compared = media.mediaComparisonDocs([frozen, {img: frozen.img, flags: {}}, {img: frozen.img}]);
  assert.deepEqual(compared[0], compared[1]);
  assert.deepEqual(compared[0], compared[2]);
});

test('two known byte versions never compare equal just because their source URL was reused', async () => {
  const {io, files} = memoryMedia({'assets/face.png': 'abc'});
  const old = {img: 'assets/face.png'};
  const first = (await snapshotActorMedia([old], io)).docs[0];
  files.set('assets/face.png', bytes('changed'));
  const second = (await snapshotActorMedia([old], io)).docs[0];
  const compared = media.mediaComparisonDocs([old, first, second]);
  assert.equal(compared[0].img, old.img);
  assert.notEqual(compared[1].img, compared[2].img);
  assert.notEqual(compared[0].img, compared[1].img);
  assert.notEqual(compared[0].img, compared[2].img);
});

test('a stale or unsupported metadata entry cannot establish media equality', async () => {
  const {io} = memoryMedia({'assets/face.png': 'abc'});
  const frozen = (await snapshotActorMedia([{img: 'assets/face.png'}], io)).docs[0];
  frozen.img = 'assets/edited.png';
  frozen.system = {biography: managed('abc')};
  frozen.flags['taka-charsync-media'].references.push({
    path: ['system', 'biography'], source: 'assets/private.png', frozen: managed('abc'), sha256: hash('abc'),
  });
  const compared = media.mediaComparisonDocs([{img: 'assets/face.png'}, frozen, {img: 'assets/private.png'}]);
  assert.equal(compared[0].img, 'assets/face.png');
  assert.equal(compared[1].img, 'assets/edited.png');
  assert.equal(compared[2].img, 'assets/private.png');
  assert.equal(compared[1].system.biography, managed('abc'));
});

test('metadata follows item ids across reordering and only survives fields that still match', async () => {
  const {io} = memoryMedia({'assets/a.png': 'a', 'assets/b.png': 'b'});
  const original = {items: [{_id: 'A', img: 'assets/a.png'}, {_id: 'B', img: 'assets/b.png'}]};
  const frozen = (await snapshotActorMedia([original], io)).docs[0];
  const reordered = {items: [frozen.items[1], frozen.items[0]], flags: {other: {keep: true}}};
  media.combineMediaMetadata(reordered, [frozen]);
  assert.equal(reordered.flags['taka-charsync-media'].references.length, 2);
  const independent = (await snapshotActorMedia([original], memoryMedia({'assets/a.png': 'a', 'assets/b.png': 'b'}).io)).docs[0];
  const [old, compared] = media.mediaComparisonDocs([independent, reordered]);
  assert.equal(old.items[0].img, compared.items[1].img);
  assert.equal(old.items[1].img, compared.items[0].img);
  reordered.items = structuredClone(reordered.items);
  reordered.items[0].img = 'assets/new-b.png';
  media.combineMediaMetadata(reordered, [frozen]);
  assert.deepEqual(reordered.flags['taka-charsync-media'].references.map(ref => ref.path), [['items', 'A', 'img']]);
  reordered.items[1].img = 'assets/new-a.png';
  media.combineMediaMetadata(reordered, [frozen]);
  assert.deepEqual(reordered.flags, {other: {keep: true}});
});
