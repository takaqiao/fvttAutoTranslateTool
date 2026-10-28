import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { actorRows, foundryAppDir, makeDb, makeWorld } from './helpers.mjs';
import { bootstrapBases } from '../../engine/bootstrap.mjs';
import { loadConfig, loadSeed } from '../../engine/config.mjs';
import { createContext, tick } from '../../engine/main.mjs';
import { contentHash } from '../../scripts/lib/canon.mjs';

const HOUR = 60 * 60 * 1000;
const SINCE = Date.UTC(2026, 9, 16, 12);
const OLD = SINCE - 24 * HOUR;
const NEW = SINCE + HOUR;

let workDir;
let server;
let statusUrl;

before(async () => {
  workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'charsync-bootstrap-'));
  server = http.createServer((request, response) => {
    response.writeHead(200, {'content-type': 'application/json'});
    response.end(JSON.stringify({active: false, version: '14.369'}));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  statusUrl = `http://127.0.0.1:${server.address().port}/api/status`;
});

after(async () => {
  await new Promise(resolve => server.close(resolve));
  if (workDir) fs.rmSync(workDir, {recursive: true, force: true});
});

/** 一张卡：actor 本身、一个条目和条目上一个 effect 的修改时间分别给。 */
function card(id, {actor = OLD, item = OLD, effect = OLD, hp = 20} = {}) {
  return {
    _id: id, name: `Card ${id}`, type: 'character',
    system: {details: {level: {value: 5}}, attributes: {hp: {value: hp}}},
    _stats: {modifiedTime: actor},
    items: [{
      _id: 'itemFeat00000001', name: 'Feat', type: 'feat', system: {}, _stats: {modifiedTime: item},
      effects: [{_id: 'effect0000000001', name: 'Effect', _stats: {modifiedTime: effect}}],
    }],
    effects: [],
  };
}

const withHash = async doc => ({doc, hash: await contentHash(doc)});

test('bootstrapBases picks the base by which side changed since the sync', async () => {
  const mainA = card('cardA00000000001');
  const mainB = card('cardB00000000001', {effect: NEW});
  const charB = card('cardB00000000001', {hp: 15});
  const mainC = card('cardC00000000001', {item: NEW});
  const mainE = card('cardE00000000001', {item: NEW});
  const mainDocs = {
    cardA00000000001: await withHash(mainA),
    cardB00000000001: await withHash(mainB),
    cardC00000000001: await withHash(mainC),
    cardD00000000001: await withHash(card('cardD00000000001')),
    cardE00000000001: await withHash(mainE),
  };
  const charDocs = {
    cardA00000000001: card('cardA00000000001', {actor: NEW}),
    cardB00000000001: charB,
    cardC00000000001: card('cardC00000000001', {actor: NEW}),
    cardE00000000001: card('cardE00000000001', {actor: NEW}),
    cardM00000000001: card('cardM00000000001'),
  };

  const bases = await bootstrapBases({
    cards: ['cardA00000000001', 'cardB00000000001', 'cardC00000000001', 'cardD00000000001', 'cardE00000000001', 'cardM00000000001'],
    mainDocs, charDocs, since: SINCE, forced: {cardE00000000001: 'main'},
  });

  assert.deepEqual(bases.cardA00000000001, {case: 'main-unchanged', ref: {kind: 'main', hash: mainDocs.cardA00000000001.hash}});
  // 主服只有条目上的 effect 晚于同步时刻，也算动过
  assert.deepEqual(bases.cardB00000000001, {case: 'char-unchanged', ref: {kind: 'main', hash: await contentHash(charB)}, historyDoc: charB});
  assert.deepEqual(bases.cardC00000000001, {case: 'both-changed', ref: null});
  assert.deepEqual(bases.cardD00000000001, {case: 'missing', ref: null}, 'no char doc');
  assert.deepEqual(bases.cardE00000000001, {case: 'forced-main', ref: {kind: 'main', hash: mainDocs.cardE00000000001.hash}});
  assert.deepEqual(bases.cardM00000000001, {case: 'missing', ref: null}, 'no main doc');
});

const PUBLISH_DIR = 'bootstrapPublish01';

/** 一个团世界 `w1` 加建卡服的假 Data 目录，种子表只有 `id` 这一张卡。 */
async function engineFixture(id, {mainActors, charActors}) {
  const root = fs.mkdtempSync(path.join(workDir, 'engine-'));
  const mainData = path.join(root, 'main');
  const charData = path.join(root, 'char');
  await makeWorld(mainData, 'w1', {actors: mainActors, settings: {'pf2e.worldSchemaVersion': 0.959}});
  await makeWorld(charData, 'pf2creation', {actors: charActors, settings: {'pf2e.worldSchemaVersion': 0.959}});
  const seed = {cards: [{charId: id, world: 'w1', mainId: id}], users: [], bootstrapSince: new Date(SINCE).toISOString()};
  fs.writeFileSync(path.join(root, 'seed.json'), JSON.stringify(seed));
  fs.writeFileSync(path.join(root, 'config.json'), JSON.stringify({
    foundryAppDir, publishDir: PUBLISH_DIR, worlds: ['w1'], titles: {w1: 'World One'}, intervalSeconds: 0,
    main: {dataDir: mainData, statusUrl}, char: {dataDir: charData, world: 'pf2creation'},
    stateDir: path.join(root, 'state'), tmpRoot: path.join(root, 'tmp'), seedPath: path.join(root, 'seed.json'),
  }));
  fs.mkdirSync(path.join(root, 'tmp'));
  const config = loadConfig(path.join(root, 'config.json'));
  const readJson = file => (fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null);
  return {
    charData,
    mainActorsDir: path.join(mainData, 'worlds', 'w1', 'data', 'actors'),
    context: () => createContext(config, loadSeed(config.seedPath, config.worlds)),
    readMeta: () => readJson(path.join(charData, PUBLISH_DIR, 'meta.json')),
    readState: () => readJson(path.join(root, 'state', 'bootstrap.json')),
  };
}

test('tick computes the bootstrap once and keeps it after the main card changes', async () => {
  const id = 'cardHero00000001';
  // 同步之后 GM 在主服动过这张卡，玩家在建卡服没动：建卡服文档就是同步时的版本
  const charDoc = card(id, {hp: 20});
  const fixture = await engineFixture(id, {mainActors: [card(id, {item: NEW, hp: 12})], charActors: [charDoc]});
  const ctx = fixture.context();

  const first = await tick(ctx);
  assert.deepEqual(first.errors, []);
  const expected = {kind: 'main', hash: await contentHash(charDoc)};
  assert.deepEqual(fixture.readMeta().bootstrap, {[id]: expected});
  const history = path.join(fixture.charData, PUBLISH_DIR, 'history', `main-${expected.hash}.json`);
  assert.deepEqual(JSON.parse(fs.readFileSync(history, 'utf8')), charDoc, 'the char doc is published as that main version');
  assert.deepEqual(fixture.readState()[id], {case: 'char-unchanged', ref: expected});

  await makeDb(fixture.mainActorsDir, actorRows(card(id, {actor: NEW + HOUR, item: NEW, hp: 3})));
  const second = await tick(ctx);
  assert.deepEqual(second.errors, []);
  assert.deepEqual(second.worldsRead, ['w1'], 'the changed world was read again');
  assert.deepEqual(fixture.readMeta().bootstrap, {[id]: expected});

  // 引擎重启后照样只读 state，不重算
  await tick(fixture.context());
  assert.deepEqual(fixture.readMeta().bootstrap, {[id]: expected});
});

test('a seed card missing from the main world is not stored, and gets its base once it appears', async () => {
  const id = 'cardLate00000001';
  const charDoc = card(id);
  const fixture = await engineFixture(id, {mainActors: [], charActors: [charDoc]});
  const ctx = fixture.context();

  const first = await tick(ctx);
  assert.deepEqual(first.errors.map(error => error.scope), ['w1'], 'the mapped main card is reported as absent');
  assert.deepEqual(fixture.readMeta().bootstrap, {});
  assert.equal(fixture.readState()?.[id], undefined);

  await makeDb(fixture.mainActorsDir, actorRows(card(id, {item: NEW})));
  const second = await tick(ctx);
  assert.deepEqual(second.errors, []);
  const expected = {kind: 'main', hash: await contentHash(charDoc)};
  assert.deepEqual(fixture.readMeta().bootstrap, {[id]: expected});
  assert.deepEqual(fixture.readState()[id], {case: 'char-unchanged', ref: expected});
});
