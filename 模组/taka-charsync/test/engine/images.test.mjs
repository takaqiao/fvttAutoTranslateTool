import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { collectImagePaths, mirrorImages, snapshotMainMedia } from '../../engine/images.mjs';

let workDir;
let fromData;
let toData;

function put(root, rel, content) {
  const file = path.join(root, rel);
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, content);
}

before(() => {
  workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'charsync-images-'));
  // Data 根各套一层，`..` 若被放行，会把 from/secret.txt 拷成 to/secret.txt
  fromData = path.join(workDir, 'from', 'Data');
  toData = path.join(workDir, 'to', 'Data');
  put(fromData, 'tokenizer/a b.webp', 'source a');
  put(fromData, 'uploads/deep/new.webp', 'source new');
  put(toData, 'tokenizer/a b.webp', 'target a');
  put(path.dirname(fromData), 'secret.txt', 'outside');
});

after(() => {
  if (workDir) fs.rmSync(workDir, {recursive: true, force: true});
});

test('collectImagePaths decodes, dedupes and leaves out paths every server already has', () => {
  const doc = {
    img: 'tokenizer/a%20b.webp',
    prototypeToken: {texture: {src: 'tokenizer/a b.webp'}},
    items: [
      {img: 'systems/pf2e/x.svg'},
      {img: 'icons/svg/mystery-man.svg'},
      {img: 'modules/some-module/art.webp'},
      {img: 'worlds/w/art.webp'},
      {img: 'https://example.com/art.webp'},
      {img: 'data:image/png;base64,AAAA'},
      {img: 'uploads/100%.webp'},
      {img: ''},
      {},
    ],
  };
  assert.deepEqual(collectImagePaths(doc), ['tokenizer/a b.webp', 'uploads/100%.webp']);
});

test('collectImagePaths copes with a doc that has no token or items', () => {
  assert.deepEqual(collectImagePaths({img: 'tokenizer/only.webp'}), ['tokenizer/only.webp']);
});

test('Tokenizer token and avatar recipes include layers, masks and paint masks', () => {
  const layer = name => ({src: `tokenizer/${name}.webp`, masks: [{src: `tokenizer/${name}-mask.webp`}],
    paintMask: {src: `tokenizer/${name}-paint.png`}});
  const doc = {flags: {'tokenizer-2': {layerStack: [layer('token')], avatarLayerStack: [layer('avatar')]}},
    prototypeToken: {flags: {'tokenizer-2': {layerStack: [layer('prototype')]}}}};
  assert.deepEqual(collectImagePaths(doc).sort(), ['token', 'avatar', 'prototype'].flatMap(name =>
    [`tokenizer/${name}.webp`, `tokenizer/${name}-mask.webp`, `tokenizer/${name}-paint.png`]).sort());
});

test('generated ring mask links never become files to mirror', () => {
  const doc = {flags: {'tokenizer-2': {layerStack: [
    {id: 'body', src: 'tokenizer/ring-body.webp', masks: [
      {type: 'ring-frame', src: 'ring-layer'},
      {type: 'frame-marching', src: 'tokenizer/ring-mask.png'},
    ]},
    {id: 'ring-layer', type: 'ring', ringConfig: {texture: 'gold'}},
  ]}}};
  put(fromData, 'tokenizer/ring-body.webp', 'ring body');
  put(fromData, 'tokenizer/ring-mask.png', 'image mask');
  const paths = collectImagePaths(doc);
  assert.deepEqual(paths, ['tokenizer/ring-body.webp', 'tokenizer/ring-mask.png']);
  assert.deepEqual(mirrorImages(paths, fromData, toData), {
    copied: paths, present: [], missing: [], refused: [], failed: [],
  });
  assert.equal(fs.readFileSync(path.join(toData, paths[0]), 'utf8'), 'ring body');
  assert.equal(fs.readFileSync(path.join(toData, paths[1]), 'utf8'), 'image mask');
});

test('managed images verify both source and target bytes and never replace a conflicting target', () => {
  const bytes = 'managed immutable artwork';
  const hash = crypto.createHash('sha256').update(bytes).digest('hex');
  const rel = `assets/taka-charsync-media/${hash}.webp`;
  put(fromData, rel, bytes);
  put(toData, rel, 'wrong target');
  const conflict = mirrorImages([rel], fromData, toData);
  assert.equal(conflict.present.length, 0);
  assert.match(conflict.failed[0]?.message ?? '', /SHA-256/);
  assert.equal(fs.readFileSync(path.join(toData, rel), 'utf8'), 'wrong target');

  fs.unlinkSync(path.join(toData, rel));
  put(fromData, rel, 'wrong source');
  const badSource = mirrorImages([rel], fromData, toData);
  assert.match(badSource.failed[0]?.message ?? '', /SHA-256/);
  assert.equal(fs.existsSync(path.join(toData, rel)), false);

  put(fromData, rel, bytes);
  assert.deepEqual(mirrorImages([rel], fromData, toData).copied, [rel]);
  put(fromData, rel, 'changed after mirroring');
  const changedSource = mirrorImages([rel], fromData, toData);
  assert.match(changedSource.failed[0]?.message ?? '', /SHA-256/);
  assert.equal(fs.readFileSync(path.join(toData, rel), 'utf8'), bytes);
  fs.unlinkSync(path.join(fromData, rel));
  assert.deepEqual(mirrorImages([rel], fromData, toData).missing, [rel], 'a valid target does not hide a missing source');
});

test('a separate dynamic-ring subject is mirrored and shares path normalization', () => {
  put(fromData, 'tokenizer/ring subject.webp', 'ring subject');
  const doc = {prototypeToken: {
    texture: {src: 'icons/svg/mystery-man.svg'},
    ring: {subject: {texture: 'tokenizer/ring%20subject.webp?v=2'}},
  }};
  const paths = collectImagePaths(doc);
  assert.deepEqual(paths, ['tokenizer/ring subject.webp']);
  assert.deepEqual(mirrorImages(paths, fromData, toData).copied, paths);
  assert.equal(fs.readFileSync(path.join(toData, paths[0]), 'utf8'), 'ring subject');
  doc.img = 'tokenizer/ring subject.webp';
  assert.deepEqual(collectImagePaths(doc), paths);
  doc.prototypeToken.ring.subject.texture = 'modules/rings/subject.webp';
  assert.deepEqual(collectImagePaths(doc), paths);
});

test('collectImagePaths normalizes paths before leaving out the shared directories', () => {
  const doc = {
    img: './tokenizer/b.webp',
    prototypeToken: {texture: {src: '/tokenizer/b.webp'}},
    items: [
      {img: './worlds/w/data/x.webp'},
      {img: '/modules/m/a.webp'},
      {img: '%2E/systems/pf2e/x.svg'},
      {img: 'tokenizer\\c.webp'},
    ],
  };
  assert.deepEqual(collectImagePaths(doc), ['tokenizer/b.webp', 'tokenizer/c.webp']);
});

test('collectImagePaths drops the query and hash before decoding, so cache-busted images still mirror', () => {
  put(fromData, 'tokenizer/q.webp', 'source q');
  const doc = {
    img: 'tokenizer/p.webp?1776446121605',
    prototypeToken: {texture: {src: 'tokenizer/p.webp'}},
    items: [
      {img: 'tokenizer/q.webp?v=9'},
      {img: 'tokenizer/r.webp#frag'},
      {img: 'tokenizer/x%3Fy.webp?v=1'},
      {img: '?only-a-query'},
    ],
  };
  const paths = collectImagePaths(doc);
  assert.deepEqual(paths, ['tokenizer/p.webp', 'tokenizer/q.webp', 'tokenizer/r.webp', 'tokenizer/x?y.webp']);

  const result = mirrorImages(['tokenizer/q.webp'], fromData, toData);
  assert.deepEqual(result.copied, ['tokenizer/q.webp']);
  assert.deepEqual(result.refused, []);
  assert.equal(fs.readFileSync(path.join(toData, 'tokenizer/q.webp'), 'utf8'), 'source q');
});

test('mirrorImages copies what the target lacks and never overwrites or escapes the data root', () => {
  const result = mirrorImages(
    ['tokenizer/a b.webp', 'uploads/deep/new.webp', '../secret.txt', 'tokenizer/../../secret.txt', 'tokenizer/none.webp'],
    fromData,
    toData,
  );
  assert.deepEqual(result, {
    copied: ['uploads/deep/new.webp'],
    present: ['tokenizer/a b.webp'],
    missing: ['tokenizer/none.webp'],
    refused: ['../secret.txt', 'tokenizer/../../secret.txt'],
    failed: [],
  });
  assert.equal(fs.readFileSync(path.join(toData, 'tokenizer/a b.webp'), 'utf8'), 'target a');
  assert.equal(fs.readFileSync(path.join(toData, 'uploads/deep/new.webp'), 'utf8'), 'source new');
  assert.deepEqual(fs.readdirSync(path.join(toData, 'uploads/deep')), ['new.webp']);
  assert.deepEqual(fs.readdirSync(path.dirname(toData)), ['Data']);
});

test('mirrorImages refuses paths that land in the shared directories after joining', () => {
  put(fromData, 'worlds/w/data/x.webp', 'world art');
  put(fromData, 'modules/m/a.webp', 'module art');
  put(fromData, 'systems/pf2e/x.svg', 'system art');
  const paths = ['./worlds/w/data/x.webp', '/modules/m/a.webp', './systems/pf2e/x.svg', '.'];
  assert.deepEqual(mirrorImages(paths, fromData, toData), {copied: [], present: [], missing: [], refused: paths, failed: []});
  for (const dir of ['worlds', 'modules', 'systems']) assert.equal(fs.existsSync(path.join(toData, dir)), false, dir);
});

test('mirrorImages copies only images and videos, whatever the case of the extension', () => {
  put(fromData, 'uploads/x.json', '{}');
  put(fromData, 'uploads/A.WEBP', 'upper');
  put(fromData, 'uploads/clip.Mp4', 'video');
  const result = mirrorImages(['uploads/x.json', 'uploads/A.WEBP', 'uploads/clip.Mp4'], fromData, toData);
  assert.deepEqual(result.refused, ['uploads/x.json']);
  assert.deepEqual(result.copied, ['uploads/A.WEBP', 'uploads/clip.Mp4']);
  assert.equal(fs.existsSync(path.join(toData, 'uploads/x.json')), false);
});

test('mirrorImages refuses paths inside the publish directory, in any letter case', () => {
  put(fromData, 'charsyncPublish01/history/char-a.webp', 'not art');
  const paths = ['charsyncPublish01/history/char-a.webp', 'CHARSYNCPUBLISH01/history/char-a.webp'];
  const result = mirrorImages(paths, fromData, toData, {publishDir: 'charsyncPublish01'});
  assert.deepEqual(result.refused, paths);
  assert.equal(fs.existsSync(path.join(toData, 'charsyncPublish01')), false);
});

test('a NUL in a decoded path is refused and the rest of the batch still copies', () => {
  put(fromData, 'tokenizer/after.webp', 'after');
  const [withNul] = collectImagePaths({img: 'tokenizer/foo%00.webp'});
  assert.equal(withNul, 'tokenizer/foo\0.webp');
  const result = mirrorImages([withNul, 'tokenizer/after.webp'], fromData, toData);
  assert.deepEqual(result.refused, [withNul]);
  assert.deepEqual(result.copied, ['tokenizer/after.webp']);
});

test('a path that fails to copy is reported and the rest of the batch still copies', () => {
  put(fromData, 'blocked/x.webp', 'x');
  put(fromData, 'tokenizer/later.webp', 'later');
  // 目标上该是目录的地方是个文件，建目录会失败
  put(toData, 'blocked', 'a file');
  const result = mirrorImages(['blocked/x.webp', 'tokenizer/later.webp'], fromData, toData);
  assert.deepEqual(result.failed.map(failure => failure.path), ['blocked/x.webp']);
  assert.deepEqual(result.copied, ['tokenizer/later.webp']);
});

test('mirrorImages refuses a source or target folder that links outside its Data root', t => {
  const outside = path.join(workDir, 'outside');
  put(outside, 'secret.webp', 'outside art');
  try {
    // 目录联接在 Windows 上不要管理员权限；Linux 上 type 被忽略，建的是普通符号链接
    fs.symlinkSync(outside, path.join(fromData, 'linked'), 'junction');
    fs.symlinkSync(outside, path.join(toData, 'escape'), 'junction');
  } catch (error) {
    t.skip(`cannot create links here: ${error.code}`);
    return;
  }
  put(fromData, 'escape/new.webp', 'would land outside');
  const result = mirrorImages(['linked/secret.webp', 'escape/new.webp'], fromData, toData);
  assert.deepEqual(result.refused, ['linked/secret.webp', 'escape/new.webp']);
  assert.equal(fs.existsSync(path.join(toData, 'linked')), false);
  assert.deepEqual(fs.readdirSync(outside), ['secret.webp']);
});

test('main freezing rejects missing files, traversal and publication paths without changing its source doc', async () => {
  for (const img of ['tokenizer/missing-main.webp', '../secret.webp', 'charsyncPublish01/history/char-a.webp']) {
    const doc = {img};
    await assert.rejects(snapshotMainMedia(doc, fromData, toData, {publishDir: 'charsyncPublish01'}));
    assert.deepEqual(doc, {img});
  }
});

test('main freezing verifies both managed source and output bytes without replacing corrupt files', async () => {
  const bytes = 'verified main artwork';
  const hash = crypto.createHash('sha256').update(bytes).digest('hex');
  const managed = `assets/taka-charsync-media/${hash}.webp`;
  put(fromData, managed, 'bad source');
  await assert.rejects(snapshotMainMedia({img: managed}, fromData, toData), /哈希不符/);
  assert.equal(fs.existsSync(path.join(toData, managed)), false);
  put(fromData, managed, bytes);
  put(toData, managed, 'bad output');
  await assert.rejects(snapshotMainMedia({img: managed}, fromData, toData), /SHA-256/);
  assert.equal(fs.readFileSync(path.join(toData, managed), 'utf8'), 'bad output');
});

test('main freezing rejects source and managed-output directories linked outside Data', async () => {
  const source = path.join(workDir, 'freeze-source');
  const target = path.join(workDir, 'freeze-output');
  const outside = path.join(workDir, 'freeze-outside');
  put(source, 'normal.webp', 'normal image');
  put(outside, 'outside.webp', 'outside image');
  fs.mkdirSync(path.join(target, 'assets'), {recursive: true});
  fs.symlinkSync(outside, path.join(source, 'linked'), 'junction');
  fs.symlinkSync(outside, path.join(target, 'assets/taka-charsync-media'), 'junction');
  await assert.rejects(snapshotMainMedia({img: 'linked/outside.webp'}, source, target), /越出 Data/);
  await assert.rejects(snapshotMainMedia({img: 'normal.webp'}, source, target), /越出 Data/);
  assert.deepEqual(fs.readdirSync(outside), ['outside.webp']);
});
