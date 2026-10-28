import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { foundryAppDir, makeDb, tmpRoot } from './helpers.mjs';
import { loadClassicLevel, openSnapshot, readActorDoc, readSettings, readUsers, scanActors } from '../../engine/leveldb.mjs';

const ACTOR_A = 'actorA0000000001';
const ACTOR_B = 'actorB0000000001';
const ACTOR_C = 'actorC0000000001';
const USER = 'user000000000001';
const LOCK_MARKER = 'source-lock';

let workDir;
let actorsDir;
let settingsDir;
let usersDir;

async function withSnapshot(dbDir, fn) {
  const snap = await openSnapshot(dbDir, {tmpRoot, foundryAppDir});
  try {
    return await fn(snap);
  } finally {
    await snap.close();
  }
}

before(async () => {
  workDir = fs.mkdtempSync(path.join(tmpRoot, 'charsync-test-'));
  actorsDir = path.join(workDir, 'actors');
  settingsDir = path.join(workDir, 'settings');
  usersDir = path.join(workDir, 'users');

  await makeDb(actorsDir, [
    [`!actors!${ACTOR_A}`, {_id: ACTOR_A, name: 'A', type: 'character', items: ['i1', 'i2'], effects: ['e1']}],
    [`!actors.items!${ACTOR_A}.i1`, {_id: 'i1', name: 'I1', type: 'feat', effects: []}],
    [`!actors.items!${ACTOR_A}.i2`, {_id: 'i2', name: 'I2', type: 'equipment', effects: ['ie1']}],
    [`!actors.items.effects!${ACTOR_A}.i2.ie1`, {_id: 'ie1', name: 'IE1'}],
    [`!actors.effects!${ACTOR_A}.e1`, {_id: 'e1', name: 'E1'}],
    // B 的行列了 i9，库里没有 i9 的行（拷库时正赶上写入）
    [`!actors!${ACTOR_B}`, {_id: ACTOR_B, name: 'B', type: 'character', items: ['i8', 'i9'], effects: []}],
    [`!actors.items!${ACTOR_B}.i8`, {_id: 'i8', name: 'I8', type: 'feat', effects: []}],
    // C 的条目列了 effect ie9，库里没有它的行
    [`!actors!${ACTOR_C}`, {_id: ACTOR_C, name: 'C', type: 'character', items: ['i7'], effects: []}],
    [`!actors.items!${ACTOR_C}.i7`, {_id: 'i7', name: 'I7', type: 'equipment', effects: ['ie9']}],
  ]);
  fs.writeFileSync(path.join(actorsDir, 'LOCK'), LOCK_MARKER);

  await makeDb(settingsDir, [
    ['!settings!set0000000000001', {_id: 'set0000000000001', key: 'pf2e.freeArchetypeVariant', value: 'true'}],
    ['!settings!set0000000000002', {_id: 'set0000000000002', key: 'pf2e.campaignFeatSections', value: '[{"id":"x"}]'}],
    ['!settings!set0000000000003', {_id: 'set0000000000003', key: 'core.plainText', value: 'not json'}],
    // 按用户存的同名设置排在世界设置后面，不能把世界设置盖掉
    ['!settings!set0000000000004', {_id: 'set0000000000004', key: 'pf2e.freeArchetypeVariant', value: 'false', user: USER}],
  ]);

  await makeDb(usersDir, [
    [`!users!${USER}`, {_id: USER, name: 'Player One', role: 1, character: ACTOR_A, password: 'x'}],
    ['!users!user000000000002', {_id: 'user000000000002', name: 'Gamemaster', role: 4, character: null}],
  ]);
});

after(() => {
  fs.rmSync(workDir, {recursive: true, force: true});
});

test('scanActors returns every actor row, parsed, and nothing from embedded collections', async () => {
  const actors = await withSnapshot(actorsDir, scanActors);
  assert.deepEqual([...actors.keys()], [ACTOR_A, ACTOR_B, ACTOR_C]);
  assert.equal(actors.get(ACTOR_A).name, 'A');
});

test('readActorDoc assembles items, item effects and actor effects into documents', async () => {
  await withSnapshot(actorsDir, async snap => {
    const actors = await scanActors(snap);
    const {doc, complete} = await readActorDoc(snap, actors.get(ACTOR_A));
    assert.equal(complete, true);
    assert.deepEqual(doc.items.map(i => i._id), ['i1', 'i2']);
    assert.deepEqual(doc.items[0].effects, []);
    assert.equal(doc.items[1].effects[0]._id, 'ie1');
    assert.equal(doc.effects[0]._id, 'e1');
    assert.deepEqual(actors.get(ACTOR_A).items, ['i1', 'i2'], 'the scanned row is left as id lists');
  });
});

test('an item listed on the actor row without its own row makes the doc incomplete', async () => {
  await withSnapshot(actorsDir, async snap => {
    const actors = await scanActors(snap);
    const {doc, complete} = await readActorDoc(snap, actors.get(ACTOR_B));
    assert.equal(complete, false);
    assert.deepEqual(doc.items.map(i => i._id), ['i8']);
  });
});

test('an item effect listed on an item without its own row makes the doc incomplete', async () => {
  await withSnapshot(actorsDir, async snap => {
    const actors = await scanActors(snap);
    const {complete} = await readActorDoc(snap, actors.get(ACTOR_C));
    assert.equal(complete, false);
  });
});

test('readSettings parses world setting values and keeps unparseable ones as strings', async () => {
  const settings = await withSnapshot(settingsDir, readSettings);
  assert.equal(settings.get('pf2e.freeArchetypeVariant'), true);
  assert.deepEqual(settings.get('pf2e.campaignFeatSections'), [{id: 'x'}]);
  assert.equal(settings.get('core.plainText'), 'not json');
});

test('readUsers returns name, role and character by user id', async () => {
  const users = await withSnapshot(usersDir, readUsers);
  assert.deepEqual(users.get(USER), {name: 'Player One', role: 1, character: ACTOR_A});
  assert.deepEqual(users.get('user000000000002'), {name: 'Gamemaster', role: 4, character: null});
});

test('the snapshot copies everything but LOCK into tmpRoot and close() removes it', async () => {
  const snap = await openSnapshot(actorsDir, {tmpRoot, foundryAppDir});
  try {
    assert.equal(path.dirname(snap.dir), path.resolve(tmpRoot));
    assert.match(path.basename(snap.dir), new RegExp(`^charsync-${process.pid}-\\d+$`));
    // 打开副本时 LevelDB 自己会建一个空 LOCK；源目录的 LOCK 被拷过来的话大小是标记的长度。
    // 库开着时 Windows 上读不了 LOCK 的内容，只能看大小。
    const lockPath = path.join(snap.dir, 'LOCK');
    if (fs.existsSync(lockPath)) assert.notEqual(fs.statSync(lockPath).size, Buffer.byteLength(LOCK_MARKER));
  } finally {
    await snap.close();
  }
  assert.equal(fs.existsSync(snap.dir), false);
  assert.equal(fs.readFileSync(path.join(actorsDir, 'LOCK'), 'utf8'), LOCK_MARKER);
});

test('a snapshot can be taken while the source database is held open', async () => {
  const liveDir = path.join(workDir, 'live');
  await makeDb(liveDir, [[`!actors!${ACTOR_A}`, {_id: ACTOR_A, name: 'A', items: [], effects: []}]]);
  const ClassicLevel = loadClassicLevel(foundryAppDir);
  const source = new ClassicLevel(liveDir, {keyEncoding: 'utf8', valueEncoding: 'utf8'});
  await source.open();
  try {
    await source.put(`!actors!${ACTOR_B}`, JSON.stringify({_id: ACTOR_B, name: 'B', items: [], effects: []}));
    const actors = await withSnapshot(liveDir, scanActors);
    assert.deepEqual([...actors.keys()], [ACTOR_A, ACTOR_B]);
  } finally {
    await source.close();
  }
});
