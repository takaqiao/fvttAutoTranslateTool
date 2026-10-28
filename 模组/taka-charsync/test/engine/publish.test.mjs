import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { gcHistory, mirrorCharHistory, publishAll, writeJsonAtomic, writeTextAtomic } from '../../engine/publish.mjs';
import { historyFile } from '../../scripts/lib/chain.mjs';

const PUBLISH_DIR = 'pubTest0000000001';
const DAY = 24 * 60 * 60 * 1000;
const NOW = Date.UTC(2026, 9, 16);

let workDir;

before(() => {
  workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'charsync-publish-'));
});

after(() => {
  if (workDir) fs.rmSync(workDir, {recursive: true, force: true});
});

const freshDir = name => {
  const dir = path.join(workDir, name);
  fs.mkdirSync(dir);
  return dir;
};

function listFiles(dir) {
  return fs.readdirSync(dir, {recursive: true, withFileTypes: true})
    .filter(entry => entry.isFile())
    .map(entry => path.relative(dir, path.join(entry.parentPath, entry.name)).split(path.sep).join('/'))
    .sort();
}

const readJson = file => JSON.parse(fs.readFileSync(file, 'utf8'));

function put(file, content, mtimeMs) {
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, content);
  if (mtimeMs !== undefined) fs.utimesSync(file, new Date(mtimeMs), new Date(mtimeMs));
}

function publishFixture() {
  const copy = {
    copyId: 'copy000000000001', rev: 1760000000000, cardId: 'card000000000001', level: 5, kind: 'level',
    pred: {kind: 'main', hash: 'a'.repeat(32)}, targetWorld: '-', replaces: null, withdrawn: false, name: 'Test Hero',
    submittedBy: 'user000000000001', companionOf: null, companions: [],
  };
  return {
    publishDir: PUBLISH_DIR,
    meta: {generatedAt: NOW, cards: {card000000000001: {world: '-', mainId: 'X'}}},
    mains: [{world: '-', mainId: 'X', entry: {hash: 'a'.repeat(32), readAt: NOW, level: 4, xp: {value: 0, max: 1000}, applied: null, doc: {_id: 'X'}}}],
    copies: {card000000000001: [copy]},
    working: {card000000000001: {hash: 'b'.repeat(32), readAt: NOW, level: 5, owner: 'user000000000001', name: 'Test Hero', doc: {_id: 'card000000000001'}}},
    history: [
      {ref: {kind: 'main', hash: 'a'.repeat(32)}, doc: {_id: 'X', name: 'main version'}},
      {ref: {kind: 'copy', copyId: copy.copyId, rev: copy.rev}, doc: {_id: copy.copyId, name: 'copy version'}},
    ],
  };
}

const PUBLISHED_FILES = [
  `copies/card000000000001.json`,
  `history/copy-copy000000000001-1760000000000.json`,
  `history/main-${'a'.repeat(32)}.json`,
  'main/-/X.json',
  'meta.json',
  'working/card000000000001.json',
];

test('writeTextAtomic creates parent directories and skips byte-identical content', () => {
  const file = path.join(freshDir('text'), 'nested', 'dir', 'rules.mjs');
  assert.equal(writeTextAtomic(file, 'export const A = 1;\n'), true);
  assert.equal(writeTextAtomic(file, 'export const A = 1;\n'), false);
  assert.equal(writeTextAtomic(file, 'export const A = 2;\n'), true);
  assert.equal(fs.readFileSync(file, 'utf8'), 'export const A = 2;\n');
  assert.deepEqual(fs.readdirSync(path.dirname(file)), ['rules.mjs']);
});

test('writeJsonAtomic writes JSON that parses back', () => {
  const file = path.join(freshDir('json'), 'value.json');
  assert.equal(writeJsonAtomic(file, {a: [1, 'x'], b: null}), true);
  assert.deepEqual(readJson(file), {a: [1, 'x'], b: null});
  assert.equal(writeJsonAtomic(file, {a: [1, 'x'], b: null}), false);
});

test('publishAll writes the same tree into every data dir, including a world named "-"', () => {
  const dataDirs = [freshDir('publish-main'), freshDir('publish-char')];
  const fixture = publishFixture();
  assert.deepEqual(publishAll({dataDirs, ...fixture}), {written: PUBLISHED_FILES.length * 2});
  for (const dataDir of dataDirs) {
    const root = path.join(dataDir, PUBLISH_DIR);
    assert.deepEqual(listFiles(root), PUBLISHED_FILES);
    assert.deepEqual(readJson(path.join(root, 'main', '-', 'X.json')), fixture.mains[0].entry);
    assert.deepEqual(readJson(path.join(root, 'working', 'card000000000001.json')), fixture.working.card000000000001);
    assert.deepEqual(readJson(path.join(root, 'copies', 'card000000000001.json')), fixture.copies.card000000000001);
    assert.deepEqual(readJson(path.join(root, 'meta.json')), fixture.meta);
    assert.deepEqual(readJson(path.join(root, `history/main-${'a'.repeat(32)}.json`)), fixture.history[0].doc);
  }
});

test('publishAll leaves unchanged files alone and rewrites only what changed', () => {
  const dataDirs = [freshDir('again-main'), freshDir('again-char')];
  publishAll({dataDirs, ...publishFixture()});
  const past = NOW - 3 * DAY;
  for (const dataDir of dataDirs) {
    for (const rel of PUBLISHED_FILES) fs.utimesSync(path.join(dataDir, PUBLISH_DIR, rel), new Date(past), new Date(past));
  }

  assert.deepEqual(publishAll({dataDirs, ...publishFixture()}), {written: 0});
  for (const dataDir of dataDirs) {
    for (const rel of PUBLISHED_FILES) assert.equal(fs.statSync(path.join(dataDir, PUBLISH_DIR, rel)).mtimeMs, past, rel);
  }

  const changed = publishFixture();
  changed.mains[0].entry.readAt = NOW + 1;
  assert.deepEqual(publishAll({dataDirs, ...changed}), {written: 2});
  for (const dataDir of dataDirs) {
    assert.deepEqual(listFiles(path.join(dataDir, PUBLISH_DIR)), PUBLISHED_FILES);
  }
});

test('mirrorCharHistory copies only the char versions the target lacks', () => {
  const fromData = freshDir('mirror-main');
  const toData = freshDir('mirror-char');
  const fromHistory = path.join(fromData, PUBLISH_DIR, 'history');
  const toHistory = path.join(toData, PUBLISH_DIR, 'history');
  put(path.join(fromHistory, 'char-a.json'), '{"v":"a"}');
  put(path.join(fromHistory, 'char-b.json'), '{"v":"b from main"}');
  put(path.join(fromHistory, 'main-x.json'), '{"v":"x"}');
  put(path.join(toHistory, 'char-b.json'), '{"v":"b already here"}');

  assert.deepEqual(mirrorCharHistory({fromData, toData, publishDir: PUBLISH_DIR}), {copied: ['char-a.json']});
  assert.deepEqual(listFiles(toHistory), ['char-a.json', 'char-b.json']);
  assert.equal(fs.readFileSync(path.join(toHistory, 'char-a.json'), 'utf8'), '{"v":"a"}');
  assert.equal(fs.readFileSync(path.join(toHistory, 'char-b.json'), 'utf8'), '{"v":"b already here"}');
});

test('mirrorCharHistory does nothing before the source has a history directory', () => {
  const fromData = freshDir('mirror-empty-main');
  const toData = freshDir('mirror-empty-char');
  assert.deepEqual(mirrorCharHistory({fromData, toData, publishDir: PUBLISH_DIR}), {copied: []});
  assert.deepEqual(fs.readdirSync(toData), []);
});

test('gcHistory deletes only old, unreferenced history files', () => {
  const dataDir = freshDir('gc');
  const root = path.join(dataDir, PUBLISH_DIR);
  const old = NOW - 61 * DAY;
  put(path.join(root, 'history', 'main-old.json'), '{}', old);
  put(path.join(root, 'history', 'copy-kept-1.json'), '{}', old);
  put(path.join(root, 'history', 'char-recent.json'), '{}', NOW - DAY);
  put(path.join(root, 'undo', '-', 'X-1.json'), '{}', old);

  const result = gcHistory({dataDir, publishDir: PUBLISH_DIR, keep: new Set(['copy-kept-1.json']), maxAgeDays: 60, now: NOW});
  assert.deepEqual(result, {deleted: ['main-old.json']});
  assert.deepEqual(listFiles(root), ['history/char-recent.json', 'history/copy-kept-1.json', 'undo/-/X-1.json']);
});

test('gcHistory also accepts keep entries exactly as historyFile returns them', () => {
  const dataDir = freshDir('gc-ref');
  const ref = {kind: 'copy', copyId: 'copy000000000002', rev: 1760000000000};
  const file = path.join(dataDir, PUBLISH_DIR, historyFile(ref));
  put(file, '{}', NOW - 61 * DAY);

  const result = gcHistory({dataDir, publishDir: PUBLISH_DIR, keep: new Set([historyFile(ref)]), maxAgeDays: 60, now: NOW});
  assert.deepEqual(result, {deleted: []});
  assert.equal(fs.existsSync(file), true);
});

test('gcHistory does nothing before the history directory exists', () => {
  const dataDir = freshDir('gc-empty');
  assert.deepEqual(gcHistory({dataDir, publishDir: PUBLISH_DIR, keep: new Set(), maxAgeDays: 60, now: NOW}), {deleted: []});
});
