import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { execFile, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { actorRows, foundryAppDir, makeDb, makeWorld } from './helpers.mjs';
import { loadConfig, loadSeed } from '../../engine/config.mjs';
import { loadClassicLevel } from '../../engine/leveldb.mjs';
import { createContext, tick } from '../../engine/main.mjs';
import { contentHash } from '../../scripts/lib/canon.mjs';

const ENGINE = fileURLToPath(new URL('../../engine/main.mjs', import.meta.url));
const WORLD = '-';
const CHAR_WORLD = 'pf2creation';
const PUBLISH_DIR = 'charsyncTestPublish01';
const SCHEMA = 0.959;
const SINCE = '2026-10-16T00:00:00.000Z';
const OLD = Date.parse(SINCE) - 24 * 60 * 60 * 1000;

const CARD = 'cardHero00000001';
const FAMILIAR = 'cardFami00000001';
const COPY_OK = 'copyOk0000000001';
const COPY_BAD = 'copyBad000000001';
const BACKUP = 'backup0000000001';
const OTHER = 'cardOther0000001';
const NPC = 'npcOnly000000001';
const P1 = 'userP10000000001';
const GM = 'userGM0000000001';
const MAIN_P1 = 'userP1main000001';
const LATE_ITEM = 'itemLate00000001';
const APPLIED = {kind: 'copy', copyId: 'copyOld000000001', rev: 1};

const item = id => ({_id: id, name: `Item ${id}`, type: 'feat', img: 'systems/pf2e/icons/feat.svg', system: {}, effects: [], _stats: {modifiedTime: OLD}});
const actor = (id, extra = {}) => ({
  _id: id, name: `Actor ${id}`, type: 'character', img: 'icons/svg/mystery-man.svg',
  system: {details: {level: {value: 3}, xp: {value: 0, max: 1000}}}, ownership: {default: 0}, flags: {},
  items: [], effects: [], _stats: {modifiedTime: OLD}, ...extra,
});
const copyMeta = (copyId, level, rev) => ({
  copyId, rev, cardId: CARD, level, kind: 'level', pred: {kind: 'main', hash: 'a'.repeat(32)}, targetWorld: WORLD,
  replaces: null, withdrawn: false, name: 'Hero', submittedBy: P1, companionOf: null, companions: [],
});
const frozen = (id, level, rev, items) => actor(id, {
  name: `Hero〔已提交 ${level} 级〕`, img: 'tokenizer/copy-ok.webp', ownership: {default: 0, [P1]: 3},
  flags: {'taka-charsync': {copy: copyMeta(id, level, rev)}}, items,
});

const BASE = {kind: 'main', hash: 'b'.repeat(32)};

const charActors = () => [
  actor(CARD, {name: 'Hero', ownership: {default: 0, [GM]: 3, [P1]: 3}, flags: {'taka-charsync': {base: BASE}}, items: [item('itemHero00000001')]}),
  actor(FAMILIAR, {name: 'Pip', type: 'familiar', system: {master: {id: CARD}, details: {level: {value: 3}}}, ownership: {[P1]: 3}}),
  frozen(COPY_OK, 4, 1760000000001, [item('itemCopy00000001')]),
  frozen(COPY_BAD, 5, 1760000000002, [item('itemBad000000001'), item(LATE_ITEM)]),
  actor(BACKUP, {name: 'Hero backup', ownership: {[P1]: 3}, flags: {'taka-charsync': {backupOf: CARD}}}),
  actor(OTHER, {name: 'Other', ownership: {[P1]: 3}, items: [item('itemOther0000001')]}),
  actor(NPC, {name: 'Npc', type: 'npc', ownership: {[GM]: 3}}),
];
const mainActors = () => [
  actor(CARD, {
    name: 'Hero', img: 'tokenizer/hero.webp', system: {details: {level: {value: 3}, xp: {value: 400, max: 1000}}},
    ownership: {[MAIN_P1]: 3}, flags: {'taka-charsync': {applied: {ref: APPLIED, at: 1, undoPath: 'undo/-/x.json', level: 3}}},
    items: [item('itemHero00000001')],
  }),
  actor(FAMILIAR, {name: 'Pip', type: 'familiar', system: {master: {id: CARD}}}),
];

let workDir;
let server;
let status = {active: false};
let statusUrl;

before(async () => {
  workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'charsync-main-'));
  server = http.createServer((request, response) => {
    response.writeHead(200, {'content-type': 'application/json'});
    response.end(JSON.stringify(status));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  statusUrl = `http://127.0.0.1:${server.address().port}/api/status`;
});

after(async () => {
  await new Promise(resolve => server.close(resolve));
  if (workDir) fs.rmSync(workDir, {recursive: true, force: true});
});

/** 两个假 Data 目录加配置和种子表；COPY_BAD 的行列了 LATE_ITEM，库里没有它的行。 */
async function buildFixture(name, configOverrides = {}) {
  const root = path.join(workDir, name);
  const mainData = path.join(root, 'main');
  const charData = path.join(root, 'char');
  await makeWorld(mainData, WORLD, {
    actors: mainActors(),
    settings: {'pf2e.worldSchemaVersion': SCHEMA, 'pf2e.freeArchetypeVariant': true},
    users: [{_id: MAIN_P1, name: 'P1', role: 1}],
  });
  await makeWorld(charData, CHAR_WORLD, {
    settings: {'pf2e.worldSchemaVersion': SCHEMA, 'pf2e.freeArchetypeVariant': false},
    users: [{_id: P1, name: 'P1', role: 1}, {_id: GM, name: 'Gamemaster', role: 4}],
  });
  const charActorsDir = path.join(charData, 'worlds', CHAR_WORLD, 'data', 'actors');
  await makeDb(charActorsDir, charActors().flatMap(actorRows).filter(([key]) => !key.endsWith(`.${LATE_ITEM}`)));
  for (const [dataDir, image] of [[mainData, 'tokenizer/hero.webp'], [charData, 'tokenizer/copy-ok.webp']]) {
    fs.mkdirSync(path.join(dataDir, 'tokenizer'), {recursive: true});
    fs.writeFileSync(path.join(dataDir, image), image);
  }
  fs.mkdirSync(path.join(root, 'tmp'));
  fs.writeFileSync(path.join(root, 'seed.json'), JSON.stringify({cards: [{charId: CARD, world: WORLD, mainId: CARD}], users: [], bootstrapSince: SINCE}));
  const configPath = path.join(root, 'config.json');
  fs.writeFileSync(configPath, JSON.stringify({
    foundryAppDir, publishDir: PUBLISH_DIR, worlds: [WORLD], titles: {[WORLD]: 'AV'}, intervalSeconds: 0,
    main: {dataDir: mainData, statusUrl}, char: {dataDir: charData, world: CHAR_WORLD, statusUrl},
    stateDir: path.join(root, 'state'), tmpRoot: path.join(root, 'tmp'), seedPath: path.join(root, 'seed.json'),
    ...configOverrides,
  }));
  return {root, mainData, charData, charActorsDir, configPath};
}

async function deleteRows(dir, keys) {
  const ClassicLevel = loadClassicLevel(foundryAppDir);
  const db = new ClassicLevel(dir, {keyEncoding: 'utf8', valueEncoding: 'utf8'});
  await db.open();
  await db.batch(keys.map(key => ({type: 'del', key})));
  await db.close();
}

function contextFor(fixture) {
  const config = loadConfig(fixture.configPath);
  return createContext(config, loadSeed(config.seedPath, config.worlds));
}

const published = (dataDir, rel) => path.join(dataDir, PUBLISH_DIR, rel);
const readPublished = (dataDir, rel) => JSON.parse(fs.readFileSync(published(dataDir, rel), 'utf8'));
const rulesPath = fixture => path.join(fixture.charData, 'modules', 'taka-charsync', 'generated', 'rules.mjs');
let rulesImport = 0;
const readRules = async fixture => (await import(`${pathToFileURL(rulesPath(fixture)).href}?read=${rulesImport++}`)).SNAPSHOT;

function managedImage(bytes) {
  return `assets/taka-charsync-media/${crypto.createHash('sha256').update(bytes).digest('hex')}.webp`;
}

function putImage(data, rel, bytes) {
  fs.mkdirSync(path.dirname(path.join(data, rel)), {recursive: true});
  fs.writeFileSync(path.join(data, rel), bytes);
}

async function completeCopies(fixture) {
  await makeDb(fixture.charActorsDir, [[`!actors.items!${COPY_BAD}.${LATE_ITEM}`, {...item(LATE_ITEM), effects: []}]]);
}

test('new main publication freezes its own legacy bytes and does not rewrite legacy history', async () => {
  const fixture = await buildFixture('main-media-identity');
  await completeCopies(fixture);
  putImage(fixture.charData, 'tokenizer/hero.webp', 'different char artwork');
  const old = mainActors()[0];
  const oldHash = await contentHash(old);
  const history = `history/main-${oldHash}.json`;
  putImage(fixture.mainData, `${PUBLISH_DIR}/${history}`, JSON.stringify(old));
  const context = contextFor(fixture);
  status = {active: false};
  assert.deepEqual((await tick(context)).errors, []);
  const main = readPublished(fixture.mainData, `main/${WORLD}/${CARD}.json`);
  assert.equal(main.doc.img, managedImage('tokenizer/hero.webp'));
  assert.equal(main.hash, await contentHash(main.doc));
  assert.notEqual(main.hash, oldHash);
  assert.deepEqual(readPublished(fixture.mainData, history), old);
  assert.equal(fs.readFileSync(path.join(fixture.charData, main.doc.img), 'utf8'), 'tokenizer/hero.webp');
  assert.equal(fs.readFileSync(path.join(fixture.charData, 'tokenizer/hero.webp'), 'utf8'), 'different char artwork');
});

test('legacy bytes changing without a DB change publish a new main version and unchanged bytes stay throttled', async () => {
  const fixture = await buildFixture('main-media-overwrite', {intervalSeconds: 5});
  await completeCopies(fixture);
  const context = contextFor(fixture);
  status = {active: false};
  const now = Date.now();
  await tick(context, now);
  const first = readPublished(fixture.mainData, `main/${WORLD}/${CARD}.json`);
  putImage(fixture.mainData, 'tokenizer/hero.webp', 'new main artwork');
  assert.deepEqual((await tick(context, now + 1000)).worldsRead, []);
  const next = await tick(context, now + 6000);
  assert.deepEqual(next.errors, []);
  const second = readPublished(fixture.mainData, `main/${WORLD}/${CARD}.json`);
  assert.notEqual(second.hash, first.hash);
  assert.equal(second.doc.img, managedImage('new main artwork'));
  assert.deepEqual(readPublished(fixture.mainData, `history/main-${first.hash}.json`), first.doc);
  assert.deepEqual((await tick(context, now + 12000)).worldsRead, []);
});

test('a removed legacy file cannot prevent reading a replacement media path from the database', async () => {
  const fixture = await buildFixture('main-media-removed');
  await completeCopies(fixture);
  const context = contextFor(fixture);
  status = {active: false};
  await tick(context);
  const main = mainActors()[0];
  main.img = 'tokenizer/replacement.webp';
  putImage(fixture.mainData, main.img, 'replacement bytes');
  await makeDb(path.join(fixture.mainData, 'worlds', WORLD, 'data', 'actors'), actorRows(main));
  fs.unlinkSync(path.join(fixture.mainData, 'tokenizer/hero.webp'));
  assert.deepEqual((await tick(context)).errors, []);
  assert.equal(readPublished(fixture.mainData, `main/${WORLD}/${CARD}.json`).doc.img, managedImage('replacement bytes'));
});

test('dry main publication freezes into dry Data and leaves source media and histories untouched', async () => {
  const fixture = await buildFixture('main-media-dry');
  await completeCopies(fixture);
  const config = loadConfig(fixture.configPath);
  config.dryOutDir = path.join(fixture.root, 'dry');
  const context = createContext(config, loadSeed(config.seedPath, config.worlds), {dry: true});
  status = {active: false};
  assert.deepEqual((await tick(context)).errors, []);
  const main = readPublished(context.out.main, `main/${WORLD}/${CARD}.json`);
  assert.equal(main.doc.img, managedImage('tokenizer/hero.webp'));
  for (const data of [context.out.main, context.out.char]) {
    assert.equal(fs.readFileSync(path.join(data, main.doc.img), 'utf8'), 'tokenizer/hero.webp');
  }
  for (const data of [fixture.mainData, fixture.charData]) {
    assert.equal(fs.existsSync(path.join(data, 'assets/taka-charsync-media')), false);
    assert.equal(fs.existsSync(path.join(data, PUBLISH_DIR)), false);
  }
});

test('the first tick prepares both managed upload directories even without managed references', async () => {
  const fixture = await buildFixture('media-directories');
  await completeCopies(fixture);
  const context = contextFor(fixture);
  status = {active: false};
  const report = await tick(context);
  assert.deepEqual(report.errors, []);
  for (const data of [fixture.mainData, fixture.charData]) {
    assert.ok(fs.statSync(path.join(data, 'assets/taka-charsync-media')).isDirectory());
  }
  const repeated = await tick(context);
  assert.deepEqual(repeated.errors, []);
});

test('managed upload directory initialization in dry mode never writes to the real Data roots', async () => {
  const fixture = await buildFixture('media-directories-dry');
  const config = loadConfig(fixture.configPath);
  config.dryOutDir = path.join(fixture.root, 'dry');
  const context = createContext(config, loadSeed(config.seedPath, config.worlds), {dry: true});
  status = {active: false};
  const report = await tick(context);
  assert.deepEqual(report.errors, []);
  for (const data of [context.out.main, context.out.char]) {
    assert.ok(fs.statSync(path.join(data, 'assets/taka-charsync-media')).isDirectory());
  }
  for (const data of [fixture.mainData, fixture.charData]) {
    assert.equal(fs.existsSync(path.join(data, 'assets')), false);
  }
});

test('managed upload directory initialization reports an existing file without replacing it', async () => {
  const fixture = await buildFixture('media-directory-file');
  putImage(fixture.mainData, 'assets/taka-charsync-media', 'existing file');
  status = {active: false};
  const report = await tick(contextFor(fixture));
  assert.ok(report.errors.some(error => error.scope === 'images' && error.message.includes('assets')));
  assert.equal(fs.readFileSync(path.join(fixture.mainData, 'assets/taka-charsync-media'), 'utf8'), 'existing file');
  assert.ok(fs.statSync(path.join(fixture.charData, 'assets/taka-charsync-media')).isDirectory());
});

test('managed upload directory initialization refuses a link outside Data', async t => {
  const fixture = await buildFixture('media-directory-link');
  const outside = path.join(fixture.root, 'outside');
  fs.mkdirSync(outside);
  fs.mkdirSync(path.join(fixture.charData, 'assets'));
  try {
    fs.symlinkSync(outside, path.join(fixture.charData, 'assets/taka-charsync-media'), 'junction');
  } catch (error) {
    t.skip(`cannot create links here: ${error.code}`);
    return;
  }
  status = {active: false};
  const report = await tick(contextFor(fixture));
  assert.ok(report.errors.some(error => error.scope === 'images' && error.message.includes('Data')));
  assert.deepEqual(fs.readdirSync(outside), []);
});

test('a missing recipe mask retains the previous submission and retries without a database change', async () => {
  const fixture = await buildFixture('media-retry');
  await completeCopies(fixture);
  const context = contextFor(fixture);
  status = {active: false};
  const now = Date.now();
  await tick(context, now);
  const oldRef = `copy:${COPY_OK}:1760000000001`;
  const oldHistory = readPublished(fixture.mainData, `history/copy-${COPY_OK}-1760000000001.json`);
  const newer = frozen(COPY_OK, 4, 1760000000010, []);
  const bytes = 'new frozen portrait';
  const maskBytes = 'frozen mask';
  newer.img = managedImage(bytes);
  newer.flags['tokenizer-2'] = {layerStack: [{src: newer.img, masks: [{src: managedImage(maskBytes)}]}]};
  putImage(fixture.charData, newer.img, bytes);
  await makeDb(fixture.charActorsDir, actorRows(newer));

  const blocked = await tick(context, now + 1000);
  const newRef = `copy:${COPY_OK}:1760000000010`;
  const newHistory = `history/copy-${COPY_OK}-1760000000010.json`;
  assert.ok(blocked.errors.some(error => error.scope === 'images' && error.message.includes(managedImage(maskBytes))));
  for (const data of [fixture.mainData, fixture.charData]) {
    const meta = readPublished(data, 'meta.json');
    assert.ok(meta.published.includes(oldRef));
    assert.ok(!meta.published.includes(newRef));
    assert.equal(fs.existsSync(published(data, newHistory)), false);
    assert.equal(readPublished(data, `copies/${CARD}.json`).find(copy => copy.copyId === COPY_OK).rev, 1760000000001);
    assert.deepEqual(readPublished(data, `history/copy-${COPY_OK}-1760000000001.json`), oldHistory);
  }
  await tick(context, now + 2000);
  putImage(fixture.charData, managedImage(maskBytes), maskBytes);
  const recovered = await tick(context, now + 3000);
  assert.deepEqual(recovered.errors, []);
  assert.ok(readPublished(fixture.mainData, 'meta.json').published.includes(newRef));
  assert.equal(fs.readFileSync(path.join(fixture.mainData, managedImage(maskBytes)), 'utf8'), maskBytes);
  assert.deepEqual(readPublished(fixture.mainData, `history/copy-${COPY_OK}-1760000000001.json`), oldHistory);
});

test('an existing copy history is rechecked for managed conflicts without rereading mutable copy rows', async () => {
  const fixture = await buildFixture('media-history-recheck');
  await completeCopies(fixture);
  const copy = frozen(COPY_OK, 4, 1760000000001, []);
  const bytes = 'submitted immutable image';
  copy.img = managedImage(bytes);
  putImage(fixture.charData, copy.img, bytes);
  await makeDb(fixture.charActorsDir, actorRows(copy));
  const context = contextFor(fixture);
  status = {active: false};
  const now = Date.now();
  await tick(context, now);
  const history = readPublished(fixture.mainData, `history/copy-${COPY_OK}-1760000000001.json`);
  putImage(fixture.mainData, copy.img, 'conflicting target');
  const blocked = await tick(context, now + 1000);
  assert.ok(blocked.errors.some(error => /SHA-256/.test(error.message)));
  assert.ok(!readPublished(fixture.mainData, 'meta.json').published.includes(`copy:${COPY_OK}:1760000000001`));
  assert.equal(fs.readFileSync(path.join(fixture.mainData, copy.img), 'utf8'), 'conflicting target');
  fs.unlinkSync(path.join(fixture.mainData, copy.img));
  const recovered = await tick(context, now + 2000);
  assert.deepEqual(recovered.errors, []);
  assert.ok(readPublished(fixture.mainData, 'meta.json').published.includes(`copy:${COPY_OK}:1760000000001`));
  assert.deepEqual(readPublished(fixture.mainData, `history/copy-${COPY_OK}-1760000000001.json`), history);
});

test('a missing history mirror is restored from the saved revision even if its database row was edited', async () => {
  const fixture = await buildFixture('media-restore-history');
  await completeCopies(fixture);
  const context = contextFor(fixture);
  status = {active: false};
  const now = Date.now();
  await tick(context, now);
  const historyPath = `history/copy-${COPY_OK}-1760000000001.json`;
  const saved = fs.readFileSync(published(fixture.mainData, historyPath), 'utf8');
  const changed = frozen(COPY_OK, 4, 1760000000001, []);
  changed.img = 'tokenizer/edited-after-publish.webp';
  putImage(fixture.charData, changed.img, 'edited image');
  await makeDb(fixture.charActorsDir, actorRows(changed));
  fs.unlinkSync(published(fixture.charData, historyPath));
  const report = await tick(context, now + 1000);
  assert.deepEqual(report.errors, []);
  assert.equal(fs.readFileSync(published(fixture.mainData, historyPath), 'utf8'), saved);
  assert.equal(fs.readFileSync(published(fixture.charData, historyPath), 'utf8'), saved);
  assert.equal(fs.existsSync(path.join(fixture.mainData, changed.img)), false);
});

test('a submitted owner and familiar publish together only after all recipe assets exist', async () => {
  const fixture = await buildFixture('media-group');
  await completeCopies(fixture);
  const owner = frozen(COPY_OK, 4, 1760000000010, []);
  const companionId = 'copyFami00000001';
  const companion = frozen(companionId, 4, 1760000000010, []);
  owner.flags['taka-charsync'].copy.companions = [{copyId: companionId, sourceId: FAMILIAR, rev: 1760000000010}];
  Object.assign(companion.flags['taka-charsync'].copy, {cardId: FAMILIAR, companionOf: COPY_OK});
  companion.img = managedImage('companion image');
  companion.flags['tokenizer-2'] = {avatarLayerStack: [{src: companion.img, paintMask: {src: managedImage('companion mask')}}]};
  putImage(fixture.charData, companion.img, 'companion image');
  await makeDb(fixture.charActorsDir, [...actorRows(owner), ...actorRows(companion)]);
  const context = contextFor(fixture);
  status = {active: false};
  const now = Date.now();
  await tick(context, now);
  const blocked = readPublished(fixture.mainData, 'meta.json');
  assert.ok(!blocked.published.includes(`copy:${COPY_OK}:1760000000010`));
  assert.ok(!blocked.published.includes(`copy:${companionId}:1760000000010`));
  assert.equal(fs.existsSync(published(fixture.mainData, `history/copy-${COPY_OK}-1760000000010.json`)), false);
  putImage(fixture.charData, managedImage('companion mask'), 'companion mask');
  const recovered = await tick(context, now + 1000);
  assert.deepEqual(recovered.errors, []);
  const ready = readPublished(fixture.mainData, 'meta.json');
  assert.ok(ready.published.includes(`copy:${COPY_OK}:1760000000010`));
  assert.ok(ready.published.includes(`copy:${companionId}:1760000000010`));
});

test('frozen new artwork cannot alter the main current portrait before approval', async () => {
  const fixture = await buildFixture('media-main-isolation');
  await completeCopies(fixture);
  putImage(fixture.charData, 'tokenizer/hero.webp', 'new mutable char artwork');
  const copy = frozen(COPY_OK, 4, 1760000000001, []);
  copy.img = managedImage('new mutable char artwork');
  putImage(fixture.charData, copy.img, 'new mutable char artwork');
  await makeDb(fixture.charActorsDir, actorRows(copy));
  status = {active: false};
  const report = await tick(contextFor(fixture));
  assert.deepEqual(report.errors, []);
  assert.equal(fs.readFileSync(path.join(fixture.mainData, 'tokenizer/hero.webp'), 'utf8'), 'tokenizer/hero.webp');
  assert.equal(readPublished(fixture.mainData, `main/${WORLD}/${CARD}.json`).doc.img, managedImage('tokenizer/hero.webp'));
  assert.equal(readPublished(fixture.mainData, `history/copy-${COPY_OK}-1760000000001.json`).img, copy.img);
  assert.equal(fs.readFileSync(path.join(fixture.mainData, copy.img), 'utf8'), 'new mutable char artwork');
});

test('a missing familiar row blocks its owner and a failed later group keeps the whole previous group', async () => {
  const fixture = await buildFixture('media-group-previous');
  await completeCopies(fixture);
  const owner = frozen(COPY_OK, 4, 1760000000010, []);
  const companionId = 'copyFami00000001';
  const companion = frozen(companionId, 4, 1760000000010, []);
  owner.flags['taka-charsync'].copy.companions = [{copyId: companionId, sourceId: FAMILIAR, rev: 1760000000010}];
  Object.assign(companion.flags['taka-charsync'].copy, {cardId: FAMILIAR, companionOf: COPY_OK});
  await makeDb(fixture.charActorsDir, actorRows(owner));
  const context = contextFor(fixture);
  status = {active: false};
  const now = Date.now();
  await tick(context, now);
  assert.ok(!readPublished(fixture.mainData, 'meta.json').published.includes(`copy:${COPY_OK}:1760000000010`));
  assert.equal(fs.existsSync(published(fixture.mainData, `history/copy-${COPY_OK}-1760000000010.json`)), false);
  await makeDb(fixture.charActorsDir, actorRows(companion));
  await tick(context, now + 1000);
  for (const id of [COPY_OK, companionId]) {
    assert.ok(readPublished(fixture.mainData, 'meta.json').published.includes(`copy:${id}:1760000000010`));
  }

  owner.flags['taka-charsync'].copy.rev = 1760000000020;
  owner.flags['taka-charsync'].copy.companions[0].rev = 1760000000020;
  companion.flags['taka-charsync'].copy.rev = 1760000000020;
  companion.img = managedImage('later familiar image');
  putImage(fixture.charData, companion.img, 'wrong later familiar bytes');
  await makeDb(fixture.charActorsDir, [...actorRows(owner), ...actorRows(companion)]);
  const rejected = await tick(context, now + 2000);
  assert.ok(rejected.errors.some(error => error.message.includes('SHA-256')));
  for (const id of [COPY_OK, companionId]) {
    const meta = readPublished(fixture.mainData, 'meta.json');
    assert.ok(meta.published.includes(`copy:${id}:1760000000010`));
    assert.ok(!meta.published.includes(`copy:${id}:1760000000020`));
    assert.equal(fs.existsSync(published(fixture.mainData, `history/copy-${id}-1760000000020.json`)), false);
  }
  putImage(fixture.charData, companion.img, 'later familiar image');
  const retried = await tick(context, now + 3000);
  assert.deepEqual(retried.errors, []);
  for (const id of [COPY_OK, companionId]) {
    assert.ok(readPublished(fixture.mainData, 'meta.json').published.includes(`copy:${id}:1760000000020`));
    assert.ok(fs.existsSync(published(fixture.mainData, `history/copy-${id}-1760000000010.json`)));
  }
});

for (const scenario of [
  {name: 'owner withdraws before its companion', target: 'owner'},
  {name: 'companion withdraws before its owner', target: 'companion'},
  {name: 'withdrawal survives a missing published image', target: 'owner', missing: true},
  {name: 'withdrawing an unpublished revision does not revive its previous revision', target: 'owner', newer: true, missing: true},
  {name: 'withdrawal survives an incomplete unpublished revision', target: 'owner', newer: true, incomplete: true},
]) {
  test(scenario.name, async () => {
    const fixture = await buildFixture(`withdrawal-${scenario.name.replaceAll(' ', '-')}`);
    await completeCopies(fixture);
    const owner = frozen(COPY_OK, 4, 1760000000010, []);
    const companionId = 'copyFami00000001';
    const companion = frozen(companionId, 4, 1760000000010, []);
    owner.flags['taka-charsync'].copy.companions = [{copyId: companionId, sourceId: FAMILIAR, rev: 1760000000010}];
    Object.assign(companion.flags['taka-charsync'].copy, {cardId: FAMILIAR, companionOf: COPY_OK});
    owner.img = managedImage('withdrawal owner portrait');
    putImage(fixture.charData, owner.img, 'withdrawal owner portrait');
    await makeDb(fixture.charActorsDir, [...actorRows(owner), ...actorRows(companion)]);
    const context = contextFor(fixture);
    status = {active: false};
    const now = Date.now();
    await tick(context, now);
    const oldHistory = `history/copy-${COPY_OK}-1760000000010.json`;
    const saved = fs.readFileSync(published(fixture.mainData, oldHistory), 'utf8');
    if (scenario.newer) {
      owner.flags['taka-charsync'].copy.rev = 1760000000020;
      owner.flags['taka-charsync'].copy.companions[0].rev = 1760000000020;
      companion.flags['taka-charsync'].copy.rev = 1760000000020;
      if (scenario.missing) owner.img = managedImage('missing new portrait');
      if (scenario.incomplete) owner.items = [item('itemAbsent000001')];
      await makeDb(fixture.charActorsDir, [actorRows(owner)[0], ...actorRows(companion)]);
      await tick(context, now + 1000);
      assert.ok(readPublished(fixture.mainData, 'meta.json').published.includes(`copy:${COPY_OK}:1760000000010`));
    } else if (scenario.missing) {
      fs.unlinkSync(path.join(fixture.charData, owner.img));
    }
    const target = scenario.target === 'owner' ? owner : companion;
    target.flags['taka-charsync'].copy.withdrawn = true;
    await makeDb(fixture.charActorsDir, [actorRows(target)[0]]);
    const report = await tick(context, now + 2000);
    if (!scenario.missing && !scenario.incomplete) assert.deepEqual(report.errors, []);
    for (const data of [fixture.mainData, fixture.charData]) {
      const ownerEntry = readPublished(data, `copies/${CARD}.json`).find(copy => copy.copyId === COPY_OK);
      const companionEntry = readPublished(data, `copies/${FAMILIAR}.json`).find(copy => copy.copyId === companionId);
      assert.equal(ownerEntry.withdrawn, scenario.target === 'owner');
      assert.equal(companionEntry.withdrawn, scenario.target === 'companion');
      const targetEntry = scenario.target === 'owner' ? ownerEntry : companionEntry;
      assert.equal(targetEntry.rev, target.flags['taka-charsync'].copy.rev);
      const refs = readPublished(data, 'meta.json').published;
      assert.ok(!refs.some(ref => ref.startsWith(`copy:${COPY_OK}:`) || ref.startsWith(`copy:${companionId}:`)));
      assert.equal(fs.readFileSync(published(data, oldHistory), 'utf8'), saved);
    }
  });
}

test('refused paths and failed file copies block submission history and retry after the file obstruction is removed', async () => {
  const fixture = await buildFixture('media-refused-failed');
  await completeCopies(fixture);
  const copy = frozen(COPY_OK, 4, 1760000000010, []);
  copy.flags['tokenizer-2'] = {layerStack: [{src: '../outside.webp'}]};
  await makeDb(fixture.charActorsDir, actorRows(copy));
  const context = contextFor(fixture);
  status = {active: false};
  const now = Date.now();
  const refused = await tick(context, now);
  assert.ok(refused.errors.some(error => error.scope === 'images' && error.message.includes('../outside.webp')));
  assert.equal(fs.existsSync(published(fixture.mainData, `history/copy-${COPY_OK}-1760000000010.json`)), false);

  copy.flags['tokenizer-2'].layerStack[0].src = 'blocked/mask.webp';
  putImage(fixture.charData, 'blocked/mask.webp', 'mask');
  putImage(fixture.mainData, 'blocked', 'file in place of directory');
  await makeDb(fixture.charActorsDir, actorRows(copy));
  const failed = await tick(context, now + 1000);
  assert.ok(failed.errors.some(error => error.scope === 'images' && error.message.includes('blocked/mask.webp')));
  assert.equal(fs.existsSync(published(fixture.mainData, `history/copy-${COPY_OK}-1760000000010.json`)), false);
  fs.unlinkSync(path.join(fixture.mainData, 'blocked'));
  const retried = await tick(context, now + 2000);
  assert.deepEqual(retried.errors, []);
  assert.ok(readPublished(fixture.mainData, 'meta.json').published.includes(`copy:${COPY_OK}:1760000000010`));
});

test('main media retries without a database change and dry output includes managed assets on both sides', async () => {
  const fixture = await buildFixture('media-main-dry');
  await completeCopies(fixture);
  const main = mainActors()[0];
  main.img = managedImage('main portrait');
  const copy = frozen(COPY_OK, 4, 1760000000010, []);
  copy.img = managedImage('char portrait');
  putImage(fixture.charData, copy.img, 'char portrait');
  await makeDb(path.join(fixture.mainData, 'worlds', WORLD, 'data', 'actors'), actorRows(main));
  await makeDb(fixture.charActorsDir, actorRows(copy));
  const config = loadConfig(fixture.configPath);
  config.dryOutDir = path.join(fixture.root, 'dry');
  const context = createContext(config, loadSeed(config.seedPath, config.worlds), {dry: true});
  status = {active: false};
  const now = Date.now();
  const blocked = await tick(context, now);
  assert.ok(blocked.errors.some(error => error.scope === 'images' && error.message.includes(main.img)));
  assert.equal(fs.existsSync(published(context.out.char, `main/${WORLD}/${CARD}.json`)), false);
  putImage(fixture.mainData, main.img, 'main portrait');
  const retried = await tick(context, now + 1000);
  assert.deepEqual(retried.errors, []);
  for (const data of [context.out.main, context.out.char]) {
    assert.equal(fs.readFileSync(path.join(data, main.img), 'utf8'), 'main portrait');
    assert.equal(fs.readFileSync(path.join(data, copy.img), 'utf8'), 'char portrait');
    assert.equal(readPublished(data, `main/${WORLD}/${CARD}.json`).doc.img, main.img);
  }
  assert.equal(fs.existsSync(path.join(fixture.charData, main.img)), false);
  assert.equal(fs.existsSync(path.join(fixture.mainData, copy.img)), false);
});

test('rule snapshots are identical in both meta files and the generated module, with a stable enforcement date', async () => {
  const fixture = await buildFixture('rule-snapshot', {campaignFolders: {folderEcnu000001: WORLD}});
  status = {active: false};
  const firstAt = Date.now();
  const report = await tick(contextFor(fixture), firstAt);
  assert.deepEqual(report.errors, []);
  const first = readPublished(fixture.mainData, 'meta.json').ruleSnapshot;
  assert.equal(first?.version, 2);
  assert.equal(first.enforcedSince, firstAt);
  assert.deepEqual(first.folderWorld, {folderEcnu000001: WORLD});
  assert.deepEqual(first, readPublished(fixture.charData, 'meta.json').ruleSnapshot);
  assert.deepEqual(first, await readRules(fixture));
  assert.deepEqual(first, JSON.parse(fs.readFileSync(path.join(fixture.root, 'state', 'rules.json'), 'utf8')));

  const charSettingsDir = path.join(fixture.charData, 'worlds', CHAR_WORLD, 'data', 'settings');
  await makeDb(charSettingsDir, [['!settings!setting000000001', {
    _id: 'setting000000001', key: 'pf2e.freeArchetypeVariant', value: 'true', user: null,
  }]]);
  await tick(contextFor(fixture), firstAt + 1000);
  const second = readPublished(fixture.charData, 'meta.json').ruleSnapshot;
  assert.deepEqual(second.hashes, first.hashes);
  assert.equal(second.enforcedSince, firstAt);
  assert.deepEqual(second, await readRules(fixture));
});

test('a legacy difference table cannot supply rules for an unread world', async () => {
  const fixture = await buildFixture('legacy-rules');
  fs.mkdirSync(path.join(fixture.root, 'state'));
  fs.writeFileSync(path.join(fixture.root, 'state', 'rules.json'), JSON.stringify({
    rules: {[WORLD]: {'pf2e.freeArchetypeVariant': true}}, cardWorld: {[CARD]: WORLD},
  }));
  status = {active: true, world: WORLD};
  await tick(contextFor(fixture));
  const snapshot = readPublished(fixture.charData, 'meta.json').ruleSnapshot;
  assert.equal(snapshot?.version, 2);
  assert.deepEqual(snapshot.rules, {});
  assert.deepEqual(snapshot.hashes, {});
  assert.equal(snapshot.cardWorld[CARD], WORLD);

  status = {active: false};
  await tick(contextFor(fixture));
  assert.equal(readPublished(fixture.charData, 'meta.json').ruleSnapshot.rules[WORLD]['pf2e.freeArchetypeVariant'], true);
});

test('new copies require the current target-world rules hash while rejected copies remain withdrawable', async () => {
  const fixture = await buildFixture('copy-rules');
  status = {active: false};
  const firstAt = Date.now();
  await tick(contextFor(fixture), firstAt);
  const snapshot = readPublished(fixture.charData, 'meta.json').ruleSnapshot;
  assert.ok(snapshot?.hashes?.[WORLD]);
  const newer = frozen(COPY_OK, 4, firstAt + 100, []);
  const copy = newer.flags['taka-charsync'].copy;
  await makeDb(fixture.charActorsDir, actorRows(newer));
  const missing = await tick(contextFor(fixture), firstAt + 200);
  assert.ok(missing.errors.some(error => error.scope === 'rules' && error.message.includes(COPY_OK)));
  assert.ok(!readPublished(fixture.charData, 'meta.json').published.includes(`copy:${COPY_OK}:${copy.rev}`));
  assert.ok(fs.existsSync(published(fixture.charData, `history/copy-${COPY_OK}-${copy.rev}.json`)));
  assert.equal(readPublished(fixture.charData, `copies/${CARD}.json`)[0].copyId, COPY_OK);

  copy.rev++;
  copy.rulesHash = snapshot.hashes[WORLD];
  copy.targetWorld = 'missing';
  await makeDb(fixture.charActorsDir, actorRows(newer));
  const wrongWorld = await tick(contextFor(fixture), firstAt + 300);
  assert.ok(wrongWorld.errors.some(error => error.scope === 'rules' && error.message.includes(COPY_OK)));
  assert.ok(!readPublished(fixture.charData, 'meta.json').published.includes(`copy:${COPY_OK}:${copy.rev}`));

  copy.rev++;
  copy.targetWorld = WORLD;
  await makeDb(fixture.charActorsDir, actorRows(newer));
  const matching = await tick(contextFor(fixture), firstAt + 400);
  assert.deepEqual(matching.errors, []);
  assert.ok(readPublished(fixture.charData, 'meta.json').published.includes(`copy:${COPY_OK}:${copy.rev}`));

  const settingsDir = path.join(fixture.mainData, 'worlds', WORLD, 'data', 'settings');
  await makeDb(settingsDir, [['!settings!setting000000001', {
    _id: 'setting000000001', key: 'pf2e.freeArchetypeVariant', value: 'false', user: null,
  }]]);
  const stale = await tick(contextFor(fixture), firstAt + 500);
  assert.ok(stale.errors.some(error => error.scope === 'rules' && error.message.includes(COPY_OK)));
  assert.ok(!readPublished(fixture.charData, 'meta.json').published.includes(`copy:${COPY_OK}:${copy.rev}`));
  assert.ok(fs.existsSync(published(fixture.charData, `history/copy-${COPY_OK}-${copy.rev}.json`)));
});

test('a partial publication invalidates rule snapshots on both Data directories and preserves enforcement across restart', async () => {
  const fixture = await buildFixture('rules-publish-failure');
  status = {active: false};
  const firstAt = Date.now();
  await tick(contextFor(fixture), firstAt);
  const blocked = published(fixture.charData, `working/${OTHER}.json`);
  fs.unlinkSync(blocked);
  fs.mkdirSync(blocked);
  await makeDb(fixture.charActorsDir, actorRows(actor(OTHER, {name: 'Changed', ownership: {[P1]: 3}})));
  const report = await tick(contextFor(fixture), firstAt + 1000);
  assert.ok(report.errors.some(error => error.scope === 'publish'));
  for (const data of [fixture.mainData, fixture.charData]) {
    const snapshot = readPublished(data, 'meta.json').ruleSnapshot;
    assert.equal(snapshot?.enforcedSince, firstAt);
    assert.deepEqual(snapshot.rules, {});
    assert.deepEqual(snapshot.hashes, {});
  }
  assert.deepEqual((await readRules(fixture)).rules, {});
  fs.rmdirSync(blocked);
  await tick(contextFor(fixture), firstAt + 2000);
  assert.equal(readPublished(fixture.charData, 'meta.json').ruleSnapshot.enforcedSince, firstAt);
});

test('a failed world read stays unavailable until that world can be read successfully again', async () => {
  const fixture = await buildFixture('rules-read-failure');
  const context = contextFor(fixture);
  status = {active: false};
  await tick(context);
  const settings = path.join(fixture.mainData, 'worlds', WORLD, 'data', 'settings');
  fs.renameSync(settings, `${settings}-unavailable`);
  const failed = await tick(context);
  assert.ok(failed.errors.some(error => error.scope === WORLD));
  assert.deepEqual(readPublished(fixture.charData, 'meta.json').ruleSnapshot.rules, {});

  status = {active: true, world: WORLD};
  await tick(context);
  assert.deepEqual(readPublished(fixture.charData, 'meta.json').ruleSnapshot.rules, {});

  fs.renameSync(`${settings}-unavailable`, settings);
  status = {active: false};
  await tick(context);
  assert.equal(readPublished(fixture.charData, 'meta.json').ruleSnapshot.rules[WORLD]['pf2e.freeArchetypeVariant'], true);
});

test('a failed first char read after restart revokes previously published rules', async () => {
  const fixture = await buildFixture('rules-char-read-failure');
  status = {active: false};
  await tick(contextFor(fixture));
  const charWorld = path.join(fixture.charData, 'worlds', CHAR_WORLD);
  fs.renameSync(charWorld, `${charWorld}-unavailable`);
  const failed = await tick(contextFor(fixture));
  assert.ok(failed.errors.some(error => error.scope === CHAR_WORLD));
  for (const data of [fixture.mainData, fixture.charData]) {
    assert.deepEqual(readPublished(data, 'meta.json').ruleSnapshot.rules, {});
  }
  assert.deepEqual((await readRules(fixture)).rules, {});
});

test('a first publication failure reports fallback errors without advertising unwritten bootstrap history', async () => {
  const fixture = await buildFixture('rules-first-publish-failure');
  status = {active: false};
  const firstAt = Date.now();
  fs.mkdirSync(rulesPath(fixture), {recursive: true});
  const failed = await tick(contextFor(fixture), firstAt);
  assert.ok(failed.errors.some(error => error.scope === 'rules' && error.message.includes('rules.mjs')));
  for (const data of [fixture.mainData, fixture.charData]) {
    const meta = readPublished(data, 'meta.json');
    assert.deepEqual(meta.ruleSnapshot.rules, {});
    assert.deepEqual(meta.bootstrap, {});
    assert.deepEqual(meta.working, {});
    assert.deepEqual(meta.published, []);
  }
  fs.rmdirSync(rulesPath(fixture));
  const recovered = await tick(contextFor(fixture), firstAt + 1000);
  assert.deepEqual(recovered.errors, []);
  assert.equal(readPublished(fixture.charData, 'meta.json').ruleSnapshot.enforcedSince, firstAt);
});

test('new submissions independently resolve working card campaign folders before they become applicable', async () => {
  const rootFolder = 'folderRoot000001';
  const childFolder = 'folderChild00001';
  const otherFolder = 'folderOther00001';
  const conflictFolder = 'folderMixed00001';
  const cycleA = 'folderCycle00001';
  const cycleB = 'folderCycle00002';
  const otherWorld = 'other';
  const fixture = await buildFixture('copy-folder-provenance', {
    worlds: [WORLD, otherWorld],
    campaignFolders: {[rootFolder]: WORLD, [otherFolder]: otherWorld, [conflictFolder]: otherWorld, folderGone000002: WORLD},
  });
  await makeWorld(fixture.mainData, otherWorld, {settings: {'pf2e.freeArchetypeVariant': false}});
  const folderRows = [
    {_id: rootFolder, folder: null}, {_id: childFolder, folder: rootFolder},
    {_id: otherFolder, folder: null}, {_id: conflictFolder, folder: rootFolder},
    {_id: cycleA, folder: cycleB}, {_id: cycleB, folder: cycleA},
  ];
  await makeDb(path.join(fixture.charData, 'worlds', CHAR_WORLD, 'data', 'folders'),
    folderRows.map(folder => [`!folders!${folder._id}`, {...folder, type: 'Actor'}]));
  status = {active: false};
  const context = contextFor(fixture);
  const firstAt = Date.now();
  await tick(context, firstAt);
  const snapshot = readPublished(fixture.charData, 'meta.json').ruleSnapshot;
  const directId = 'newCard000000001';
  const cases = [
    {card: actor(directId, {folder: rootFolder}), ok: true},
    {card: actor('newCard000000002', {folder: childFolder}), ok: true},
    {card: actor('newCard000000003'), ok: false},
    {card: actor('newCard000000004', {folder: 'folderGone000001'}), ok: false},
    {card: actor('newCard000000005', {folder: cycleA}), ok: false},
    {card: actor('newCard000000006', {folder: conflictFolder}), ok: false},
    {card: actor('newCard000000007', {folder: rootFolder}), targetWorld: otherWorld, ok: false},
    {cardId: 'missing000000001', ok: false},
    {cardId: COPY_OK, ok: false},
    {cardId: BACKUP, ok: false},
    {card: actor(CARD, {folder: otherFolder, ownership: {[P1]: 3}}), ok: false},
    {card: actor('newFami000000001', {type: 'familiar', system: {master: {id: directId}}}), ok: true},
    {card: actor('newCard000000008', {folder: 'folderGone000002'}), ok: false},
  ];
  for (const [index, entry] of cases.entries()) {
    const copyId = `newCopy${String(index).padStart(9, '0')}`;
    const targetWorld = entry.targetWorld ?? WORLD;
    entry.copy = {...copyMeta(copyId, 1, firstAt + 100 + index), cardId: entry.card?._id ?? entry.cardId,
      kind: 'new', pred: null, targetWorld, rulesHash: snapshot.hashes[targetWorld]};
    const rows = actorRows(actor(copyId, {flags: {'taka-charsync': {copy: entry.copy}}}));
    if (entry.card) rows.push(...actorRows(entry.card));
    await makeDb(fixture.charActorsDir, rows);
  }
  const report = await tick(context, firstAt + 1000);
  const meta = readPublished(fixture.charData, 'meta.json');
  for (const {copy, ok} of cases) {
    const key = `copy:${copy.copyId}:${copy.rev}`;
    assert.equal(meta.published.includes(key), ok, copy.copyId);
    assert.equal(report.errors.some(error => error.scope === 'rules' && error.message.includes(copy.copyId)), !ok, copy.copyId);
    assert.ok(fs.existsSync(published(fixture.charData, `history/copy-${copy.copyId}-${copy.rev}.json`)), copy.copyId);
    assert.ok(readPublished(fixture.charData, `copies/${copy.cardId}.json`).some(row => row.copyId === copy.copyId), copy.copyId);
  }
  assert.ok(meta.published.includes(`copy:${COPY_OK}:1760000000001`), 'legacy copies retain their existing history path');
  assert.equal(meta.ruleSnapshot.cardWorld[directId], undefined, 'folder provenance is not persisted as an explicit card mapping');
  await deleteRows(fixture.charActorsDir, [`!actors!${CARD}`]);
  await tick(context, firstAt + 2000);
  assert.ok(readPublished(fixture.charData, 'meta.json').published.includes(`copy:${COPY_OK}:1760000000001`),
    'a legacy copy does not lose its existing apply path when the old working card is gone');
});

test('a parent folder move alone rechecks an already published submission', async () => {
  const folder = 'folderChild00001';
  const ecnu = 'folderRoot000001';
  const sog = 'folderOther00001';
  const fixture = await buildFixture('copy-folder-move', {
    worlds: [WORLD, 'other'], campaignFolders: {[ecnu]: WORLD, [sog]: 'other'},
  });
  await makeWorld(fixture.mainData, 'other', {});
  const foldersDir = path.join(fixture.charData, 'worlds', CHAR_WORLD, 'data', 'folders');
  await makeDb(foldersDir, [
    [`!folders!${folder}`, {_id: folder, type: 'Actor', folder: ecnu}],
    [`!folders!${ecnu}`, {_id: ecnu, type: 'Actor', folder: null}],
    [`!folders!${sog}`, {_id: sog, type: 'Actor', folder: null}],
  ]);
  await makeDb(fixture.charActorsDir, actorRows(actor(OTHER, {folder, ownership: {[P1]: 3}})));
  // 补齐旧测试副本，保证无其他未完成文档触发每轮重读。
  await makeDb(fixture.charActorsDir, [[`!actors.items!${COPY_BAD}.${LATE_ITEM}`, {...item(LATE_ITEM), effects: []}]]);
  const context = contextFor(fixture);
  const firstAt = Date.now();
  status = {active: false};
  await tick(context, firstAt);
  const snapshot = readPublished(fixture.charData, 'meta.json').ruleSnapshot;
  const copy = {...copyMeta('newCopy000000001', 1, firstAt + 100), cardId: OTHER, kind: 'new',
    pred: null, rulesHash: snapshot.hashes[WORLD]};
  await makeDb(fixture.charActorsDir, actorRows(actor(copy.copyId, {flags: {'taka-charsync': {copy}}})));
  await tick(context, firstAt + 1000);
  assert.ok(readPublished(fixture.charData, 'meta.json').published.includes(`copy:${copy.copyId}:${copy.rev}`));

  await makeDb(foldersDir, [[`!folders!${folder}`, {_id: folder, type: 'Actor', folder: sog}]]);
  const changed = await tick(context, firstAt + 2000);
  assert.equal(changed.charRead, true);
  assert.ok(!readPublished(fixture.charData, 'meta.json').published.includes(`copy:${copy.copyId}:${copy.rev}`));
  assert.ok(changed.errors.some(error => error.scope === 'rules' && error.message.includes(copy.copyId)));
  fs.renameSync(foldersDir, `${foldersDir}-unavailable`);
  const unreadable = await tick(context, firstAt + 3000);
  assert.ok(unreadable.errors.some(error => error.scope === CHAR_WORLD));
  assert.deepEqual(readPublished(fixture.charData, 'meta.json').ruleSnapshot.rules, {});
});

let fx;
let ctx;

test('without a rule snapshot for the running world, copies retain their history but cannot be applied', async () => {
  fx = await buildFixture('run');
  ctx = contextFor(fx);
  status = {active: true, version: '14.369', world: WORLD};

  const report = await tick(ctx);

  assert.equal(report.mainActive, WORLD);
  assert.deepEqual(report.worldsRead, []);
  assert.equal(report.charRead, true);
  assert.equal(report.copiesPublished, 1);
  assert.deepEqual(report.copiesSkippedIncomplete, ['Hero〔已提交 5 级〕']);
  assert.deepEqual(report.errors.map(error => error.scope), ['rules']);

  for (const dataDir of [fx.mainData, fx.charData]) {
    assert.deepEqual(readPublished(dataDir, `copies/${CARD}.json`), [copyMeta(COPY_OK, 4, 1760000000001)]);
    assert.ok(fs.existsSync(published(dataDir, `history/copy-${COPY_OK}-1760000000001.json`)));
    assert.ok(!fs.existsSync(published(dataDir, `history/copy-${COPY_BAD}-1760000000002.json`)));
    assert.ok(!fs.existsSync(published(dataDir, `main/${WORLD}`)));
  }
  const meta = readPublished(fx.charData, 'meta.json');
  assert.deepEqual(meta.published, []);
  assert.deepEqual(meta.char, {schema: SCHEMA});
  assert.equal(meta.cards[CARD].world, WORLD);
  assert.ok(fs.existsSync(path.join(fx.mainData, 'tokenizer/copy-ok.webp')), 'copy images are mirrored to the main Data');
  assert.ok(fs.statSync(published(fx.mainData, `undo/${WORLD}`)).isDirectory(), 'the undo folder exists before any GM upload');
  assert.ok(!fs.existsSync(published(fx.charData, 'undo')));
});

test('player cards are published as working versions; copies, backups and GM-only actors are not', () => {
  const meta = readPublished(fx.charData, 'meta.json');
  assert.deepEqual(Object.keys(meta.working).sort(), [FAMILIAR, CARD, OTHER]);
  const entry = readPublished(fx.charData, `working/${OTHER}.json`);
  assert.equal(entry.owner, P1);
  assert.equal(entry.name, 'Other');
  assert.equal(entry.level, 3);
  assert.deepEqual(entry.doc.items.map(i => i._id), ['itemOther0000001']);
  assert.deepEqual(meta.working[OTHER], {name: 'Other', owner: P1, level: 3, readAt: entry.readAt, hash: entry.hash});
  for (const id of [COPY_OK, COPY_BAD, BACKUP, NPC]) assert.ok(!fs.existsSync(published(fx.charData, `working/${id}.json`)), id);
});

test('once the world is closed it is read: main versions, world meta and the rules table appear', async () => {
  status = {active: false, version: '14.369'};

  const report = await tick(ctx);

  assert.equal(report.mainActive, null);
  assert.deepEqual(report.worldsRead, [WORLD]);
  assert.deepEqual(report.errors, []);
  assert.deepEqual(report.copiesSkippedIncomplete, ['Hero〔已提交 5 级〕'], 'the incomplete copy is read again next round');

  const entry = readPublished(fx.charData, `main/${WORLD}/${CARD}.json`);
  assert.equal(entry.hash, await contentHash(entry.doc));
  assert.equal(entry.level, 3);
  assert.deepEqual(entry.xp, {value: 400, max: 1000});
  assert.deepEqual(entry.applied, APPLIED);
  assert.deepEqual(entry.doc.items.map(i => i._id), ['itemHero00000001']);
  assert.deepEqual(readPublished(fx.mainData, `main/${WORLD}/${CARD}.json`), entry);
  assert.ok(fs.existsSync(published(fx.charData, `history/main-${entry.hash}.json`)));
  assert.ok(fs.existsSync(published(fx.charData, `main/${WORLD}/${FAMILIAR}.json`)));

  const meta = readPublished(fx.mainData, 'meta.json');
  assert.equal(meta.worlds[WORLD].schema, SCHEMA);
  assert.equal(meta.worlds[WORLD].title, 'AV');
  assert.equal(meta.cards[FAMILIAR].companionOf, CARD);
  // 种子里没列这只魔宠，靠 system.master.id 认出来的同伴也要有起步基线
  assert.deepEqual(meta.bootstrap, {
    [CARD]: {kind: 'main', hash: entry.hash},
    [FAMILIAR]: {kind: 'main', hash: readPublished(fx.charData, `main/${WORLD}/${FAMILIAR}.json`).hash},
  });
  assert.deepEqual(meta.users, {[P1]: {[WORLD]: MAIN_P1}});
  assert.ok(fs.existsSync(path.join(fx.charData, entry.doc.img)), 'frozen main images are mirrored to the char Data');

  const rulesFile = path.join(fx.charData, 'modules', 'taka-charsync', 'generated', 'rules.mjs');
  const rules = await import(`${pathToFileURL(rulesFile).href}?t=${Date.now()}`);
  assert.equal(rules.RULES[WORLD]['pf2e.freeArchetypeVariant'], true);
  assert.equal(rules.CARD_WORLD[CARD], WORLD);
});

test('the incomplete copy is published once its missing row is there; unchanged rules are not rewritten', async () => {
  const rulesFile = path.join(fx.charData, 'modules', 'taka-charsync', 'generated', 'rules.mjs');
  const rulesBefore = fs.readFileSync(rulesFile, 'utf8');
  await makeDb(fx.charActorsDir, [[`!actors.items!${COPY_BAD}.${LATE_ITEM}`, {...item(LATE_ITEM), effects: []}]]);

  const report = await tick(ctx);

  assert.deepEqual(report.errors, []);
  assert.equal(report.copiesPublished, 1);
  assert.deepEqual(report.copiesSkippedIncomplete, []);
  assert.deepEqual(readPublished(fx.charData, `copies/${CARD}.json`).map(copy => copy.copyId).sort(), [COPY_BAD, COPY_OK]);
  const doc = readPublished(fx.charData, `history/copy-${COPY_BAD}-1760000000002.json`);
  assert.deepEqual(doc.items.map(i => i._id), ['itemBad000000001', LATE_ITEM]);
  assert.deepEqual(readPublished(fx.charData, 'meta.json').published, [`copy:${COPY_BAD}:1760000000002`, `copy:${COPY_OK}:1760000000001`]);
  assert.equal(fs.readFileSync(rulesFile, 'utf8'), rulesBefore);
});

test('withdrawing a published copy relists it without re-reading it; a missing working file is written again', async () => {
  const withdrawn = frozen(COPY_OK, 4, 1760000000001, [item('itemCopy00000001')]);
  withdrawn.flags['taka-charsync'].copy.withdrawn = true;
  await makeDb(fx.charActorsDir, [actorRows(withdrawn)[0]]);
  // 条目行没了：要是引擎重读这份副本，就会当成条目不全而沿用上次那份未撤回的
  await deleteRows(fx.charActorsDir, [`!actors.items!${COPY_OK}.itemCopy00000001`]);
  fs.rmSync(published(fx.charData, `working/${FAMILIAR}.json`));

  const report = await tick(ctx);

  assert.deepEqual(report.errors, []);
  assert.deepEqual(report.copiesSkippedIncomplete, []);
  assert.equal(readPublished(fx.charData, `copies/${CARD}.json`).find(copy => copy.copyId === COPY_OK).withdrawn, true);
  assert.ok(fs.existsSync(published(fx.charData, `working/${FAMILIAR}.json`)));
});

test('a player card caught mid-write keeps its last working file; char history uploads reach the char Data', async () => {
  const before = fs.readFileSync(published(fx.charData, `working/${OTHER}.json`), 'utf8');
  const metaBefore = readPublished(fx.charData, 'meta.json').working[OTHER];
  const torn = actor(OTHER, {name: 'Other renamed', ownership: {[P1]: 3}, items: [item('itemOther0000001'), item('itemTorn00000001')]});
  await makeDb(fx.charActorsDir, actorRows(torn).filter(([key]) => !key.endsWith('.itemTorn00000001')));
  fs.mkdirSync(published(fx.mainData, 'history'), {recursive: true});
  fs.writeFileSync(published(fx.mainData, 'history/char-x.json'), '{"_id":"x"}');

  const report = await tick(ctx);

  assert.deepEqual(report.errors, []);
  assert.deepEqual(report.copiesSkippedIncomplete, ['Other renamed']);
  assert.equal(fs.readFileSync(published(fx.charData, `working/${OTHER}.json`), 'utf8'), before);
  assert.deepEqual(readPublished(fx.charData, 'meta.json').working[OTHER], metaBefore);
  assert.equal(fs.readFileSync(published(fx.charData, 'history/char-x.json'), 'utf8'), '{"_id":"x"}');
});

test('after a restart while the world is running, its cards and world meta carry over from the last meta.json', async () => {
  const metaBefore = readPublished(fx.mainData, 'meta.json');
  status = {active: true, version: '14.369', world: WORLD};

  const report = await tick(contextFor(fx));

  assert.deepEqual(report.worldsRead, []);
  const meta = readPublished(fx.mainData, 'meta.json');
  assert.deepEqual(meta.worlds, metaBefore.worlds);
  assert.deepEqual(meta.cards, metaBefore.cards, 'the familiar found through the main world stays mapped');
  assert.deepEqual(meta.users, metaBefore.users);
});

test('a world that cannot be read does not stop the others', async () => {
  const fixture = await buildFixture('broken', {worlds: [WORLD, 'gone']});
  status = {active: false, version: '14.369'};

  const report = await tick(contextFor(fixture));

  assert.deepEqual(report.worldsRead, [WORLD]);
  assert.deepEqual(report.errors.map(error => error.scope), ['gone']);
  assert.ok(fs.existsSync(published(fixture.charData, `main/${WORLD}/${CARD}.json`)));
});

test('a charId flag that is not a Foundry id never becomes a publish path', async () => {
  const fixture = await buildFixture('evil');
  const mainActorsDir = path.join(fixture.mainData, 'worlds', WORLD, 'data', 'actors');
  await makeDb(mainActorsDir, actorRows(actor('evilActor0000001', {name: 'Evil', flags: {'taka-charsync': {charId: '../../evil'}}})));
  status = {active: false, version: '14.369'};

  const report = await tick(contextFor(fixture));

  assert.deepEqual(report.errors.map(error => error.scope), [WORLD]);
  assert.match(report.errors[0].message, /evilActor0000001/);
  for (const dataDir of [fixture.mainData, fixture.charData]) {
    assert.ok(!fs.existsSync(path.join(dataDir, 'evil.json')));
    assert.ok(fs.existsSync(published(dataDir, `main/${WORLD}/${CARD}.json`)), 'other cards are published as usual');
  }
  assert.ok(!('../../evil' in readPublished(fixture.mainData, 'meta.json').cards));

  // 世界正在开时映射沿用上次的 meta.json；混在里面的坏卡 id 在拼路径前被拦下
  const meta = readPublished(fixture.mainData, 'meta.json');
  meta.cards['../../evil2'] = {world: WORLD, mainId: 'evilActor0000001', charOwner: null, companions: [], companionOf: null};
  fs.writeFileSync(published(fixture.mainData, 'meta.json'), JSON.stringify(meta));
  status = {active: true, version: '14.369', world: WORLD};

  const carried = await tick(contextFor(fixture));

  assert.deepEqual(carried.errors.map(error => error.scope), ['publish']);
  for (const dataDir of [fixture.mainData, fixture.charData]) assert.ok(!fs.existsSync(path.join(dataDir, 'evil2.json')));
});

test('without any successful char read nothing is published and no camp world is read', async () => {
  const fixture = await buildFixture('no-char');
  const config = JSON.parse(fs.readFileSync(fixture.configPath, 'utf8'));
  fs.writeFileSync(fixture.configPath, JSON.stringify({...config, char: {...config.char, world: 'missing-char-world'}}));
  status = {active: false, version: '14.369'};

  const report = await tick(contextFor(fixture));

  assert.deepEqual(report.worldsRead, []);
  assert.deepEqual(report.errors.map(error => error.scope), ['missing-char-world']);
  assert.ok(!fs.existsSync(path.join(fixture.mainData, PUBLISH_DIR)));
});

test('history cleanup runs once a day and keeps every referenced version', async () => {
  const fixture = await buildFixture('gc');
  const gcCtx = contextFor(fixture);
  status = {active: false, version: '14.369'};
  await tick(gcCtx);
  const cardEntry = readPublished(fixture.mainData, `main/${WORLD}/${CARD}.json`);
  for (const dataDir of [fixture.mainData, fixture.charData]) {
    for (const name of [`main-${BASE.hash}`, `main-${'a'.repeat(32)}`, `main-${'c'.repeat(32)}`]) {
      fs.writeFileSync(published(dataDir, `history/${name}.json`), '{}');
    }
  }
  fs.writeFileSync(published(fixture.mainData, 'history/char-x.json'), '{}');

  const later = Date.now() + 100 * 24 * 60 * 60 * 1000;
  const report = await tick(gcCtx, later);

  assert.deepEqual(report.errors, []);
  for (const dataDir of [fixture.mainData, fixture.charData]) {
    const names = fs.readdirSync(published(dataDir, 'history')).sort();
    assert.deepEqual(names, [
      `copy-${COPY_OK}-1760000000001.json`,
      `main-${'a'.repeat(32)}.json`,
      `main-${BASE.hash}.json`,
      `main-${cardEntry.hash}.json`,
      `main-${readPublished(dataDir, `main/${WORLD}/${FAMILIAR}.json`).hash}.json`,
    ].sort(), 'kept: listed copy, copy pred, working-card base, latest main versions; dropped: unreferenced main version and char upload');
  }

  fs.writeFileSync(published(fixture.mainData, `history/main-${'d'.repeat(32)}.json`), '{}');
  await tick(gcCtx, later + 60 * 60 * 1000);
  assert.ok(fs.existsSync(published(fixture.mainData, `history/main-${'d'.repeat(32)}.json`)), 'no second cleanup within a day');
});

test('the CLI with --once --dry writes only under dryOutDir and prints the report as its last line', async () => {
  const fixture = await buildFixture('dry');
  const dryOutDir = path.join(fixture.root, 'dry-out');
  const config = JSON.parse(fs.readFileSync(fixture.configPath, 'utf8'));
  fs.writeFileSync(fixture.configPath, JSON.stringify({...config, dryOutDir}));
  status = {active: false, version: '14.369'};

  const {stdout} = await promisify(execFile)(process.execPath, [ENGINE, '--config', fixture.configPath, '--once', '--dry'], {timeout: 60_000});

  const lines = stdout.trim().split('\n');
  const report = JSON.parse(lines.at(-1));
  assert.deepEqual(report.worldsRead, [WORLD]);
  assert.equal(report.charRead, true);
  assert.deepEqual(report.errors, []);
  assert.match(lines.at(-2), /^\[charsync\] tick main=none read=2 copies=1 skipped=1 errors=0$/);
  assert.ok(fs.existsSync(path.join(dryOutDir, 'main', PUBLISH_DIR, `main/${WORLD}/${CARD}.json`)));
  assert.ok(fs.existsSync(path.join(dryOutDir, 'char', PUBLISH_DIR, 'meta.json')));
  assert.ok(fs.existsSync(path.join(dryOutDir, 'char', 'modules', 'taka-charsync', 'generated', 'rules.mjs')));
  assert.ok(fs.existsSync(path.join(dryOutDir, 'state', 'bootstrap.json')));
  assert.ok(!fs.existsSync(path.join(fixture.mainData, PUBLISH_DIR)));
  assert.ok(!fs.existsSync(path.join(fixture.charData, PUBLISH_DIR)));
  assert.ok(!fs.existsSync(path.join(fixture.charData, 'modules')));
  assert.ok(!fs.existsSync(path.join(fixture.root, 'state')));
  assert.ok(fs.existsSync(path.join(dryOutDir, 'char', managedImage('tokenizer/hero.webp'))), 'images are read from the real Data and written to the dry one');
  assert.ok(!fs.existsSync(path.join(fixture.charData, 'tokenizer/hero.webp')));
});

test('the CLI also starts when pm2 loads main.mjs through its container script, and a plain import does not start it', () => {
  // pm2 6 的 fork 模式：argv[1] 是 pm2 自己的容器脚本，要跑的脚本放在 pm_exec_path 里，经 import() 加载
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'charsync-pm2-'));
  try {
    const container = path.join(dir, 'ProcessContainerFork.mjs');
    fs.writeFileSync(container, "import { pathToFileURL } from 'node:url';\nawait import(pathToFileURL(process.env.LOAD).href);\n");
    const {pm_exec_path: _ignored, ...baseEnv} = process.env;
    const run = extra => spawnSync(process.execPath, [container], {env: {...baseEnv, LOAD: ENGINE, ...extra}, encoding: 'utf8', timeout: 60_000});

    // 没给 --config：CLI 真的跑起来就会报用法并以 1 退出
    const viaPm2 = run({pm_exec_path: ENGINE});
    assert.equal(viaPm2.status, 1, viaPm2.stderr);
    assert.match(viaPm2.stderr, /\[charsync\] .*用法/);

    const plainImport = run({});
    assert.equal(plainImport.status, 0, plainImport.stderr);
    assert.equal(plainImport.stdout + plainImport.stderr, '');

    const otherScript = run({pm_exec_path: container});
    assert.equal(otherScript.status, 0, otherScript.stderr);
    assert.equal(otherScript.stdout + otherScript.stderr, '');
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
});
