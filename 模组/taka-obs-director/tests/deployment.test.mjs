import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import vm from 'node:vm';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { ClassicLevel } from 'classic-level';
import { parseHTML } from 'linkedom';

const load = async (file) => (await import(file).catch((error) => { if (error.code === 'ERR_MODULE_NOT_FOUND') return {}; throw error; }));
const deploy = (await load('../tools/deploy.cjs')).default ?? {};
const qa = (await load('../tools/prepare-qa.cjs')).default ?? {};
const delivery = await load('../tools/build-delivery.mjs');
const relay = (await load('../tools/qa-relay.cjs')).default ?? {};
const temp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'director-deployment-'));
const hash = (value) => createHash('sha256').update(value).digest('hex');
const setting = (id, key, value) => ({ key: `!settings!${id}`, value: Buffer.from(JSON.stringify({ _id: id, key, value: JSON.stringify(value), user: null, _stats: { retained: true } })) });
const world = { worldId: 'cotct', system: 'pf2e', skin: 'cotct', defaultSeatOrder: [{ userId: 'gm' }, { userId: 'pl' }], players: [{ userId: 'pl', actorId: 'actor', portrait: { originalVerified: true, defaultPath: 'assets/original.png' } }] };
const inputs = { recorderUserId: 'OBSRecorder00001', worlds: [world] };
async function writeDB(folder, rows) { const db = new ClassicLevel(folder, { valueEncoding: 'buffer' }); await db.open(); try { await db.batch(rows.map(({ key, value }) => ({ type: 'put', key, value }))); } finally { await db.close(); } }
async function readDB(folder) { const db = new ClassicLevel(folder, { valueEncoding: 'buffer', createIfMissing: false }); await db.open(); try { return new Map(await db.iterator().all()); } finally { await db.close(); } }
const decode = (map, key) => [...map.values()].map((v) => JSON.parse(v)).find((v) => v.key === key);
function files(folder) { return fs.readdirSync(folder).sort().flatMap((name) => { const full = path.join(folder, name); return fs.statSync(full).isDirectory() ? files(full).map(([f, h]) => [`${name}/${f}`, h]) : [[name, hash(fs.readFileSync(full))]]; }); }
async function fixture(root) {
  fs.mkdirSync(path.join(root, 'Config'), { recursive: true });
  fs.writeFileSync(path.join(root, 'Config/options.json'), JSON.stringify({ world: null }));
  const data = path.join(root, 'Data/worlds/cotct/data');
  await writeDB(path.join(data, 'settings'), [setting('modules', 'core.moduleConfiguration', { 'obs-utils': true, other: false }), setting('lk', 'avclient-livekit.liveKitConnectionSettings', { room: 'production-room', password: 'private-fixture' }), setting('clock', 'core.time', 123456), setting('extra', 'other.nested', { keep: [1, 2] })]);
  await writeDB(path.join(data, 'users'), [{ key: '!users!pl', value: Buffer.from(' {"_id":"pl","role":1,"password":"unchanged","character":"actor"} ') }]);
  await writeDB(path.join(data, 'actors'), [{ key: '!actors!actor', value: Buffer.from(' {"_id":"actor","ownership":{"OBSRecorder00001":2}} ') }]);
  return data;
}

test('deployment dry plan and exact apply preserve unrelated raw rows, users and actors with recoverable backup', async () => {
  assert.equal(typeof deploy.deploy, 'function', 'deploy must be implemented');
  const root = temp();
  try {
    const data = await fixture(root), before = await readDB(path.join(data, 'settings'));
    const users = files(path.join(data, 'users')), actors = files(path.join(data, 'actors'));
    const options = { dataRoot: root, inputs, DB: ClassicLevel, checkSetup: async () => {}, backupRoot: path.join(root, 'backups') };
    const plan = await deploy.deploy(options);
    assert.equal(plan.worlds[0].changes.length, 7);
    assert.deepEqual(files(path.join(data, 'users')), users);
    const result = await deploy.deploy({ ...options, applyPlan: plan });
    const after = await readDB(path.join(data, 'settings'));
    assert.equal(decode(after, 'taka-obs-director.resourceDetail').value, '"compact"');
    assert.equal(decode(after, 'taka-obs-director.seatOrder').value, '["gm","pl"]');
    assert.equal(decode(after, 'taka-obs-director.portraitOverrides').value, '{"actor":"assets/original.png"}');
    assert.deepEqual(JSON.parse(decode(after, 'core.moduleConfiguration').value), { 'obs-utils': true, other: false, 'taka-obs-director': true });
    for (const key of ['!settings!lk', '!settings!clock', '!settings!extra']) assert.deepEqual(after.get(key), before.get(key));
    assert.deepEqual(files(path.join(data, 'users')), users);
    assert.deepEqual(files(path.join(data, 'actors')), actors);
    const restored = await readDB(path.join(result.backupPath, 'cotct/settings'));
    assert.deepEqual(restored, before);
    const second = await deploy.deploy(options);
    assert.equal(second.worlds[0].changes.length, 0);
    const again = await deploy.deploy({ ...options, applyPlan: second });
    assert.equal(again.changed, 0);
    assert.deepEqual(await readDB(path.join(data, 'settings')), after);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('deployment stops before writes when Setup is active or the reviewed plan is stale', async () => {
  assert.equal(typeof deploy.deploy, 'function', 'deploy must be implemented');
  const root = temp();
  try {
    const data = await fixture(root), original = files(path.join(data, 'settings'));
    await assert.rejects(deploy.deploy({ dataRoot: root, inputs, DB: ClassicLevel, checkSetup: async () => { throw Error('active'); } }), /active/);
    assert.deepEqual(files(path.join(data, 'settings')), original);
    const options = { dataRoot: root, inputs, DB: ClassicLevel, checkSetup: async () => {}, backupRoot: path.join(root, 'backups') };
    const plan = await deploy.deploy(options);
    await writeDB(path.join(data, 'settings'), [setting('extra', 'other.nested', 'changed after review')]);
    const changed = await readDB(path.join(data, 'settings'));
    await assert.rejects(deploy.deploy({ ...options, applyPlan: plan }), /stale/i);
    assert.deepEqual(await readDB(path.join(data, 'settings')), changed);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('deployment preserves conflicting unrelated duplicate rows while refusing ambiguous target keys', async () => {
  const root = temp();
  try {
    const data = await fixture(root), folder = path.join(data, 'settings');
    const duplicates = [setting('extra-two', 'other.nested', { conflicting: true }), setting('clock-two', 'core.time', 999)];
    duplicates[0].value = Buffer.from(' ' + duplicates[0].value.toString() + '\n');
    await writeDB(folder, duplicates);
    const before = await readDB(folder), options = { dataRoot: root, inputs, DB: ClassicLevel, checkSetup: async () => {}, backupRoot: path.join(root, 'backups') };
    const plan = await deploy.deploy(options), result = await deploy.deploy({ ...options, applyPlan: plan }), after = await readDB(folder);
    assert.equal(result.readbackVerified, true);
    for (const [key, bytes] of before) if (!plan.worlds[0].changes.some(change => change.key === key)) assert.deepEqual(after.get(key), bytes);
    for (const name of ['core.moduleConfiguration', 'taka-obs-director.enabled']) {
      const conflict = new Map(before); const row = setting('target-conflict', name, 'conflicting'); conflict.set(row.key, row.value);
      if (name !== 'core.moduleConfiguration') { const existing = setting('target-original', name, true); conflict.set(existing.key, existing.value); }
      assert.throws(() => deploy.settingsChanges(conflict, deploy.desiredSettings(world, inputs.recorderUserId)), /Duplicate world setting/);
    }
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('Setup gate requires both null configured world and an inactive native join response', async () => {
  assert.equal(typeof deploy.assertSourceSetup, 'function', 'Setup gate must be implemented');
  const root = temp();
  try {
    fs.mkdirSync(path.join(root, 'Config'));
    fs.writeFileSync(path.join(root, 'Config/options.json'), '{"world":null}');
    await deploy.assertSourceSetup({ dataRoot: root, fetchJoin: async () => ({ status: 200, text: async () => 'There is currently no active game session.' }) });
    await assert.rejects(deploy.assertSourceSetup({ dataRoot: root, fetchJoin: async () => ({ status: 200, text: async () => 'Join active world' }) }), /Setup/);
    fs.writeFileSync(path.join(root, 'Config/options.json'), '{"world":"cotct"}');
    await assert.rejects(deploy.assertSourceSetup({ dataRoot: root, fetchJoin: async () => ({ status: 200, text: async () => 'There is currently no active game session.' }) }), /Setup/);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('fresh QA restores embedded Items and Alien effects, preserves core.time and isolates users and rooms', async () => {
  assert.equal(typeof qa.buildWorldRows, 'function', 'QA importer must be implemented');
  const snapshot = { sourceWorldId: 'lastdayhope', system: 'alienrpg', actorIds: ['actor'], actorRows: [{ key: '!actors!actor', value: { _id: 'actor', name: 'PC', type: 'character', img: 'icons/pc.svg', folder: 'old-folder', ownership: { old: 3 }, items: ['item'], effects: ['effect'], system: { clock: 99 }, flags: { X: 1, x: 2 } } }], embeddedRows: [{ key: '!actors.items!actor.item', value: { _id: 'item', name: 'Item', system: { uses: 2 } } }, { key: '!actors.effects!actor.effect', value: { _id: 'effect', duration: { startTime: 88 } } }], settingsRows: [setting('clock', 'core.time', 123456), setting('lk', 'avclient-livekit.liveKitConnectionSettings', { room: 'production-room', password: 'secret' })].map(({ key, value }) => ({ key, value: JSON.parse(value) })) };
  const a = qa.buildWorldRows(snapshot, { qaWorldId: 'obs-director-qa-alien', recorderId: 'OBSRecorder00001', room: 'obs-director-qa-alien-a' });
  const b = qa.buildWorldRows(snapshot, { qaWorldId: 'obs-director-qa-alien', recorderId: 'OBSRecorder00001', room: 'obs-director-qa-alien-b' });
  const actor = a.actors.find((r) => r.key === '!actors!actor').value;
  assert.equal(actor.folder, null);
  assert.deepEqual(actor.items, ['item']);
  assert.deepEqual(actor.effects, ['effect']);
  assert.deepEqual(actor.flags, { X: 1, x: 2 });
  assert.deepEqual(actor.ownership, { default: 0, OBSRecorder00001: 2, QAPL000000000001: 3 });
  assert.equal(a.actors.find((r) => r.key === '!actors.effects!actor.effect').value.duration.startTime, 88);
  assert.equal(a.settings.find((r) => r.value.key === 'core.time').value.value, '123456');
  assert.equal(a.settings.some((r) => r.value.key === 'core.worldTime'), false);
  assert.equal(JSON.parse(a.settings.find((r) => r.value.key === 'avclient-livekit.liveKitConnectionSettings').value.value).room, 'obs-director-qa-alien-a');
  assert.equal(JSON.parse(a.settings.find((r) => r.value.key === 'core.moduleConfiguration').value.value)['taka-obs-director'], false);
  for (const user of a.users) { assert.equal(user.value.permissions.BROADCAST_AUDIO, false); assert.equal(user.value.permissions.BROADCAST_VIDEO, false); assert.notEqual(user.value.password, b.users.find((r) => r.key === user.key).value.password); }
  assert.equal(a.users.length, 3);
  const root = temp();
  try {
    await writeDB(path.join(root, 'actors'), a.actors.map((row) => ({ key: row.key, value: Buffer.from(JSON.stringify(row.value)) })));
    const restored = await readDB(path.join(root, 'actors'));
    assert.equal(JSON.parse(restored.get('!actors.items!actor.item')).system.uses, 2);
    assert.equal(JSON.parse(restored.get('!actors.effects!actor.effect')).duration.startTime, 88);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
  assert.throws(() => qa.buildWorldRows(snapshot, { qaWorldId: 'cotct', room: 'production-room' }), /QA/);
  assert.throws(() => qa.buildWorldRows(snapshot, { qaWorldId: 'obs-director-qa-alien', room: 'production-room' }), /room/i);
});

test('QA status rejects wrong identities and strips secret objects and unconfigured speaking IDs', async () => {
  assert.equal(typeof relay.sanitizeStatus, 'function', 'QA status boundary must be implemented');
  const expected = { worldId: 'obs-director-qa-cotct', userId: 'OBSRecorder00001', userIds: ['gm', 'pl'] };
  const result = relay.sanitizeStatus({ worldId: expected.worldId, userId: expected.userId, guardPassed: true, roomConnected: true, localPublicationCount: 0, deviceAttempts: 0, room: 'secret-room', jwt: 'secret-jwt', participant: { secret: 'value' }, speakingUserIds: ['pl', 'unknown'], events: [{ userId: 'pl', type: 'isSpeakingChanged', speaking: true, at: 12, metadata: 'private' }] }, expected);
  assert.deepEqual(result, { worldId: expected.worldId, userId: expected.userId, guardPassed: true, roomConnected: true, localPublicationCount: 0, deviceAttempts: 0, speakingUserIds: ['pl'], events: [{ userId: 'pl', type: 'isSpeakingChanged', speaking: true, at: 12 }] });
  assert.throws(() => relay.sanitizeStatus({ worldId: 'cotct', userId: expected.userId, guardPassed: true }, expected), /refused/);
  assert.throws(() => relay.sanitizeStatus({ worldId: expected.worldId, userId: 'gm', guardPassed: true }, expected), /refused/);
  assert.throws(() => relay.sanitizeStatus({ worldId: expected.worldId, userId: expected.userId, guardPassed: false }, expected), /refused/);
});

test('QA telemetry retains primitive native renderer and subscription observations without SDK objects', () => {
  const expected = { worldId: 'obs-director-qa-cotct', userId: 'OBSRecorder00001', userIds: ['pl'] };
  const value = relay.sanitizeStatus({ worldId: expected.worldId, userId: expected.userId, guardPassed: true, mode: 'combat', focusName: 'PC', focusHP: '0 / 82', focusResourceText: '1 / 3', publicFocus: false, audioReceivedBytes: [{ userId: 'pl', bytes: 12345, track: { raw: 'private' } }], portraits: [{ userId: 'pl', scale: 1.06, outlineWidth: 3, outlineColor: '#fff', rawActor: 'private' }] }, expected);
  assert.equal(value.focusHP, '0 / 82'); assert.equal(value.focusResourceText, '1 / 3');
  assert.deepEqual(value.audioReceivedBytes, [{ userId: 'pl', bytes: 12345 }]);
  assert.deepEqual(value.portraits, [{ userId: 'pl', scale: 1.06, outlineWidth: 3, outlineColor: '#fff', filter: '' }]);
});

test('QA authoritative inventory fingerprint uses canonical parsed JSON and detects changed asset entries', () => {
  const assets = [{ relativePath: 'assets/pc.png', exists: true, sha256: 'abc' }];
  const expected = hash('[{"relativePath":"assets/pc.png","exists":true,"sha256":"abc"}]');
  assert.deepEqual(qa.verifyInventory(Buffer.from(JSON.stringify(assets, null, 2) + '\n'), expected), assets);
  assert.throws(() => qa.verifyInventory(Buffer.from('[{"relativePath":"assets/pc.png","exists":true,"sha256":"changed"}]'), expected), /fingerprint/);
});

test('QA canonical filesystem references retain literal percent and hash characters while portrait URLs decode once', () => {
  assert.equal(qa.assetFile('assets/portrait%20name#1.png'), 'assets/portrait%20name#1.png');
  assert.equal(qa.assetFile('assets/portrait%20name.png?v=1', true), 'assets/portrait name.png');
  assert.throws(() => qa.assetFile('../private/file'), /Unsafe/);
});

test('QA fidelity supplements restore only verified non-user settings without changing snapshot users or room', () => {
  assert.equal(typeof qa.mergeSupplementalSettings, 'function', 'verified supplemental merge must be implemented');
  const rows = [setting('clock', 'core.time', 123456), setting('lk', 'avclient-livekit.liveKitConnectionSettings', { room: 'source-room' })].map(({ key, value }) => ({ key, value: JSON.parse(value) }));
  const snapshot = { sourceWorldId: 'pnvfcgjbf2cjp7gz', system: 'pf2e', settingsRows: rows, actorIds: [], actorRows: [], embeddedRows: [] };
  const original = JSON.stringify(snapshot);
  const extra = (id, key, value) => ({ key: `!settings!${id}`, value: JSON.parse(setting(id, key, value).value) });
  const pf = { schemaVersion: 1, sourceWorldId: snapshot.sourceWorldId, system: 'pf2e', settingsRows: [extra('variant', 'pf2e.freeArchetypeVariant', true)] };
  const rule = { schemaVersion: 1, sourceWorldId: snapshot.sourceWorldId, settingsRows: [extra('cast', 'pf2e-toolbelt.actionable.cast', true)] };
  const input = (kind, value) => { const bytes = Buffer.from(JSON.stringify(value)); return { kind, bytes, sha256: hash(bytes) }; };
  const merged = qa.mergeSupplementalSettings(snapshot, [input('pf2e', pf), input('rules', rule)]);
  assert.equal(JSON.stringify(snapshot), original);
  assert.deepEqual(merged.settingsRows.slice(0, 2), rows);
  assert.equal(merged.settingsRows.find(r => r.value.key === 'pf2e.freeArchetypeVariant').value.value, 'true');
  assert.equal(merged.settingsRows.find(r => r.value.key === 'pf2e-toolbelt.actionable.cast').value.value, 'true');
  assert.throws(() => qa.mergeSupplementalSettings(snapshot, [{ ...input('pf2e', pf), sha256: 'stale' }]), /fingerprint/);
  assert.throws(() => qa.mergeSupplementalSettings(snapshot, [input('pf2e', { ...pf, sourceWorldId: 'cotct' })]), /world/i);
  assert.throws(() => qa.mergeSupplementalSettings(snapshot, [input('pf2e', { ...pf, settingsRows: [extra('lk', 'avclient-livekit.liveKitConnectionSettings', {})] })]), /scope/i);
  assert.throws(() => qa.mergeSupplementalSettings(snapshot, [input('pf2e', { ...pf, settingsRows: [{ ...pf.settingsRows[0], value: { ...pf.settingsRows[0].value, user: 'source-user' } }] })]), /user/i);
  assert.throws(() => qa.mergeSupplementalSettings(snapshot, [input('pf2e', { ...pf, settingsRows: [{ ...pf.settingsRows[0], key: '!settings!clock' }] })]), /row/i);
  assert.throws(() => qa.mergeSupplementalSettings({ ...snapshot, settingsRows: [...rows, extra('existing', 'pf2e.freeArchetypeVariant', false)] }, [input('pf2e', pf)]), /conflict/i);
  assert.throws(() => qa.mergeSupplementalSettings({ ...snapshot, sourceWorldId: 'sog' }, [input('rules', { ...rule, sourceWorldId: 'sog' })]), /scope/i);
});

test('QA fidelity enables captured providers and FOTRP rule dependencies while Alien remains minimal', () => {
  const expected = {
    cotct: ['battlezoo-eldamon-pf2e', 'battlezoo-eldamon-legends-pf2e', 'pf2e-vessel-cn'],
    sog: ['pf2e-item-activations', 'xdy-pf2e-workbench'],
    pnvfcgjbf2cjp7gz: ['pf2e-item-activations', 'pf2e-toolbelt', 'pf2e-third-party-automation', 'socketlib', 'pf2e-dailies'],
    '-': ['pf2e-team-plus-magic'],
    ujx5r8oipw7ercdr: ['battlezoo-eldamon-legends-pf2e', 'battlezoo-eldamon-pf2e', 'pf2e-item-activations']
  };
  for (const [sourceWorldId, additional] of Object.entries(expected)) {
    const snapshot = { sourceWorldId, system: 'pf2e', actorIds: [], actorRows: [], embeddedRows: [], settingsRows: [setting('clock', 'core.time', 123), setting('lk', 'avclient-livekit.liveKitConnectionSettings', { room: 'source-room' })].map(({ key, value }) => ({ key, value: JSON.parse(value) })) };
    const rows = qa.buildWorldRows(snapshot, { qaWorldId: 'obs-director-qa-cotct', room: 'obs-director-qa-cotct-fresh' });
    const active = JSON.parse(rows.settings.find(r => r.value.key === 'core.moduleConfiguration').value.value);
    assert.deepEqual(Object.keys(active).filter(k => active[k]), ['obs-utils', 'lib-wrapper', 'avclient-livekit', 'taka-obs-director', ...additional]);
  }
  const alien = { sourceWorldId: 'lastdayhope', system: 'alienrpg', actorIds: [], actorRows: [], embeddedRows: [], settingsRows: [setting('clock', 'core.time', 123), setting('lk', 'avclient-livekit.liveKitConnectionSettings', { room: 'source-room' })].map(({ key, value }) => ({ key, value: JSON.parse(value) })) };
  const rows = qa.buildWorldRows(alien, { qaWorldId: 'obs-director-qa-alien', room: 'obs-director-qa-alien-fresh' });
  assert.deepEqual(JSON.parse(rows.settings.find(r => r.value.key === 'core.moduleConfiguration').value.value), { 'obs-utils': true, 'lib-wrapper': true, 'avclient-livekit': true, 'taka-obs-director': false });
});

test('QA fidelity files reject tampered provider and setting bytes before a restored snapshot is used', () => {
  const root = temp();
  try {
    const snapshot = { sourceWorldId: 'cotct', system: 'pf2e', settingsRows: [] };
    const settingBytes = JSON.stringify({ schemaVersion: 1, sourceWorldId: 'cotct', system: 'pf2e', settingsRows: [{ key: '!settings!variant', value: JSON.parse(setting('variant', 'pf2e.staminaVariant', true).value) }] });
    const providerBytes = JSON.stringify({ schemaVersion: 1, modules: [], worlds: [] });
    fs.writeFileSync(path.join(root, 'cotct.pf2e-settings.rows.json'), settingBytes);
    fs.writeFileSync(path.join(root, 'source-homebrew-provider-manifest.json'), providerBytes);
    fs.writeFileSync(path.join(root, 'fidelity-manifest.json'), JSON.stringify({ schemaVersion: 1, providerManifest: { file: 'source-homebrew-provider-manifest.json', sha256: hash(providerBytes) }, worlds: [{ worldId: 'cotct', files: [{ kind: 'pf2e', file: 'cotct.pf2e-settings.rows.json', sha256: hash(settingBytes) }] }] }));
    const manifest = qa.readFidelityManifest(root), restored = qa.readSupplementalSettings(snapshot, root, manifest);
    assert.equal(restored.settingsRows[0].value.value, 'true'); assert.deepEqual(snapshot.settingsRows, []);
    fs.writeFileSync(path.join(root, 'cotct.pf2e-settings.rows.json'), settingBytes.replace('true', 'false'));
    assert.throws(() => qa.readSupplementalSettings(snapshot, root, manifest), /fingerprint/);
    assert.throws(() => qa.readSupplementalSettings(snapshot, root, { ...manifest, worlds: [{ worldId: 'cotct', files: [{ kind: 'pf2e', file: '../private.rows.json', sha256: 'abc' }] }] }), /path/);
    fs.writeFileSync(path.join(root, 'source-homebrew-provider-manifest.json'), providerBytes + ' ');
    assert.throws(() => qa.readFidelityManifest(root), /fingerprint/);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('QA package copies isolate pack writes and the FOTRP alias from immutable source code', async () => {
  assert.equal(typeof qa.clonePackage, 'function', 'isolated package clone must be implemented');
  assert.equal(typeof qa.patchRuneTransfer, 'function', 'owned QA rune alias patch must be implemented');
  const root = temp();
  try {
    const source = path.join(root, 'source'), destination = path.join(root, 'qa-package'), audit = path.join(root, 'private');
    fs.mkdirSync(path.join(source, 'scripts'), { recursive: true }); fs.mkdirSync(audit);
    fs.writeFileSync(path.join(source, 'module.json'), JSON.stringify({ id: 'pf2e-third-party-automation', packs: [{ path: 'packs/items' }] }));
    fs.writeFileSync(path.join(source, 'scripts/rune-transfer.mjs'), "const worlds=new Set(['-','sog','pnvfcgjbf2cjp7gz','team-automation-qa2']);\nexport const allows = id => worlds.has(id);\n");
    await writeDB(path.join(source, 'packs/items'), [setting('pack', 'unchanged', true)]);
    const before = files(source);
    assert.throws(() => qa.clonePackage(source, path.join(source, 'qa')), /isolated/);
    assert.deepEqual(files(source), before);
    qa.clonePackage(source, destination, { ownedDirectories: ['scripts'] });
    const patch = qa.patchRuneTransfer(source, destination, audit);
    assert.equal(fs.lstatSync(path.join(destination, 'scripts')).isSymbolicLink(), false);
    assert.equal(fs.lstatSync(path.join(destination, 'module.json')).isSymbolicLink(), true);
    const code = await import(pathToFileURL(path.join(destination, 'scripts/rune-transfer.mjs')));
    assert.equal(code.allows('pnvfcgjbf2cjp7gz'), true); assert.equal(code.allows('obs-director-qa-fotrp'), true); assert.equal(code.allows('obs-director-qa-cotct'), false);
    await writeDB(path.join(destination, 'packs/items'), [setting('pack', 'changed-only-in-QA', true)]);
    assert.deepEqual(files(source), before);
    assert.equal(patch.sourceSha256, hash(fs.readFileSync(path.join(source, 'scripts/rune-transfer.mjs'))));
    assert.equal(fs.existsSync(path.join(audit, 'rune-transfer-qa.diff')), true);
    const linked = path.join(root, 'linked-package'); qa.clonePackage(source, linked);
    assert.throws(() => qa.patchRuneTransfer(source, linked, audit), /owned/i);
    assert.deepEqual(files(source), before);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('delivery whitelist excludes private data, tests, tools, credentials and machine paths', async () => {
  assert.equal(typeof delivery.buildDelivery, 'function', 'delivery builder must be implemented');
  const root = temp();
  try {
    const source = fileURLToPath(new URL('..', import.meta.url));
    const dest = path.join(root, 'portable');
    const report = await delivery.buildDelivery({ moduleRoot: source, destination: dest });
    assert.equal(report.files.some((r) => /(?:node_modules|private|tools|tests|license|users)/i.test(r.path)), false);
    assert.equal(fs.existsSync(path.join(dest, 'module/main.mjs')), true);
    assert.equal(report.files.filter((r) => r.path.startsWith('module/assets/')).length, 13);
    assert.equal(fs.existsSync(path.join(dest, 'obs/FVTT-OBS-Director.scene-collection.json')), true);
    assert.equal(fs.existsSync(path.join(dest, 'macros/sync-recorder.js')), true);
    await assert.rejects(delivery.buildDelivery({ moduleRoot: source, destination: dest }), /exists/);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('OBS legacy CSS applies to Alien fallback and cannot match the activated director or remove native audio', async () => {
  const file = new URL('../obs/FVTT-OBS-Director.scene-collection.json', import.meta.url);
  assert.equal(fs.existsSync(file), true, 'OBS collection must be implemented');
  const collection = JSON.parse(fs.readFileSync(file));
  const browser = collection.sources.filter((s) => s.id === 'browser_source');
  assert.equal(browser.length, 1); assert.equal(collection.sources.length, 2);
  assert.equal(browser[0].settings.url, 'https://v14.taka.wang/game');
  assert.equal(browser[0].settings.reroute_audio, true); assert.equal(browser[0].mixers, 1); assert.equal(browser[0].monitoring_type, 0);
  const { document } = parseHTML('<html><body class="game"><div id="camera-views"><div class="camera-view speaking" data-user="OBSRecorder00001"><video class="user-camera"></video><audio></audio></div></div><div id="pause"></div></body></html>');
  const css = browser[0].settings.css.replace(/\/\*[\s\S]*?\*\//g, '');
  const selectors = [...css.matchAll(/([^{}]+)\{[^{}]*\}/g)].flatMap((m) => m[1].split(',').map((s) => s.trim()));
  assert.ok(selectors.length > 20);
  assert.ok(selectors.some((s) => document.querySelector(s)));
  assert.equal(selectors.some((s) => /audio/.test(s)), false);
  document.body.classList.add('taka-obs-director-active');
  for (const selector of selectors) assert.equal(document.querySelector(selector.replace(/::[a-z-]+$/, '')), null, selector);
});

test('QA bootstrap blocks and counts device requests before Foundry scripts load', async () => {
  assert.equal(typeof relay.bootstrapScript, 'function', 'QA bootstrap must be implemented');
  let calls = 0; const stored = new Map();
  const context = { location: { hostname: '127.0.0.1' }, navigator: { mediaDevices: { getUserMedia: async () => { calls++; } } }, localStorage: { setItem: (k, v) => stored.set(k, v) }, window: {}, DOMException };
  vm.runInNewContext(relay.bootstrapScript({ worldId: 'obs-director-qa-cotct', userId: 'OBSRecorder00001' }), context);
  const rtc = JSON.parse(stored.get('core.rtcClientSettings'));
  assert.equal(rtc.audioSrc, 'disabled'); assert.equal(rtc.videoSrc, 'disabled');
  await assert.rejects(context.navigator.mediaDevices.getUserMedia({ audio: true }), /disabled/);
  assert.equal(calls, 0); assert.equal(context.window.__directorQA.deviceAttempts, 1);
});

test('native QA playback errors count attached paused/error audio while ignoring idle native elements', async () => {
  let callback, observed;
  const room = 'obs-director-qa-cotct-fixture', client = { liveKitRoom: { state: 'connected', name: room, localParticipant: { trackPublications: new Map() } }, audioTrack: null, videoTrack: null };
  const attached = { getAudioTracks: () => [{}] };
  const context = { globalThis: null, game: { ready: true, world: { id: 'obs-director-qa-cotct' }, user: { id: 'OBSRecorder00001', isGM: false }, webrtc: { client: { _liveKitClient: client } }, settings: { get: () => ({ room }) } }, canvas: {}, window: { __directorQA: { post: async value => { observed = value; } }, addEventListener() {} }, document: { querySelectorAll: () => [{ srcObject: attached, paused: false }, { srcObject: attached, paused: true }, { srcObject: null, paused: true, error: true }], getElementById: () => null, body: { classList: { contains: () => true } } }, setInterval(fn) { callback = fn; }, clearInterval() {} };
  context.globalThis = context;
  vm.runInNewContext(relay.statusScript({ worldId: context.game.world.id, userId: context.game.user.id, room }), context);
  await callback(); assert.equal(observed.subscribedAudioCount, 2); assert.equal(observed.playbackErrors, 1);
});

test('QA device guard permits strict WebRTC shim assignment and binding while rejecting and counting every capture attempt', async () => {
  let nativeCalls = 0;
  const context = { location: { hostname: '127.0.0.1' }, navigator: { mediaDevices: { getUserMedia: async () => { nativeCalls++; } } }, localStorage: { setItem() {} }, window: {}, DOMException };
  vm.runInNewContext(relay.bootstrapScript({ worldId: 'obs-director-qa-cotct', userId: 'OBSRecorder00001' }), context);
  const guarded = context.navigator.mediaDevices.getUserMedia;
  assert.doesNotThrow(() => vm.runInNewContext(`'use strict'; const media = navigator.mediaDevices; const original = media.getUserMedia.bind(media); media.getUserMedia = function(constraints) { return original(constraints); }; window.shimCapture = media.getUserMedia.bind(media);`, context));
  assert.equal(context.navigator.mediaDevices.getUserMedia, guarded);
  await assert.rejects(context.window.shimCapture({ audio: true }), /disabled/);
  await assert.rejects(context.navigator.mediaDevices.getUserMedia({ video: true }), /disabled/);
  assert.equal(context.window.__directorQA.deviceAttempts, 2);
  assert.equal(nativeCalls, 0);
  assert.equal(Object.getOwnPropertyDescriptor(context.navigator.mediaDevices, 'getUserMedia').configurable, false);
});

test('fixed-user relay authenticates privately, injects bootstrap first and refuses a different user or production world', async () => {
  assert.equal(typeof relay.createRelay, 'function', 'QA relay must be implemented');
  let active = 'obs-director-qa-cotct'; const requests = [];
  const upstream = http.createServer(async (req, res) => {
    if (req.url === '/api/status') { res.setHeader('content-type', 'application/json'); return res.end(JSON.stringify({ active: true, world: active })); }
    if (req.url === '/join' && req.method === 'GET') return res.end(`<html><body data-world="${active}"></body></html>`);
    if (req.url === '/join') {
      res.setHeader('content-type', 'application/json');
      if (req.headers.origin !== `http://${req.headers.host}`) { res.writeHead(400); return res.end('{"error":"The request could not be processed."}'); }
      let body = ''; for await (const chunk of req) body += chunk;
      requests.push(JSON.parse(body)); res.setHeader('set-cookie', 'session=qa; HttpOnly'); return res.end('{"redirect":"/game"}');
    }
    res.setHeader('content-type', 'text/html'); res.end('<html><head><script src="foundry.js"></script></head><body></body></html>');
  });
  await new Promise((resolve) => upstream.listen(0, '127.0.0.1', resolve));
  const server = relay.createRelay({ upstream: `http://127.0.0.1:${upstream.address().port}`, worldId: active, user: { id: 'OBSRecorder00001', name: 'QA Recorder', password: 'private-test' } });
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${server.address().port}`;
    assert.equal((await fetch(base + '//localhost/escape', { redirect: 'manual' })).status, 403);
    assert.equal((await fetch(base + '/qa-login/other', { redirect: 'manual' })).status, 403);
    const login = await fetch(base + '/qa-login/OBSRecorder00001', { redirect: 'manual' });
    assert.equal(login.status, 302); assert.equal(login.headers.get('location'), '/game');
    assert.deepEqual(requests[0], { action: 'join', username: 'QA Recorder', userId: 'OBSRecorder00001', password: 'private-test' });
    assert.match(login.headers.get('set-cookie'), /session=qa/);
    const html = await (await fetch(base + '/game')).text();
    assert.ok(html.indexOf('__directorQA') < html.indexOf('foundry.js'));
    assert.equal(html.includes('private-test'), false);
    active = 'cotct';
    assert.equal((await fetch(base + '/qa-login/OBSRecorder00001', { redirect: 'manual' })).status, 409);
    assert.equal(requests.length, 1);
  } finally { await Promise.all([new Promise((r) => server.close(r)), new Promise((r) => upstream.close(r))]); }
});

test('QA relay blocks canonical join and controller routes before arbitrary user payloads reach upstream', async () => {
  const writes = [];
  const upstream = http.createServer(async (req, res) => {
    if (req.url === '/api/status') { res.setHeader('content-type', 'application/json'); return res.end('{"active":true,"world":"obs-director-qa-cotct"}'); }
    let body = ''; for await (const chunk of req) body += chunk;
    writes.push({ path: req.url, body }); res.end('upstream accepted');
  });
  await new Promise((resolve) => upstream.listen(0, '127.0.0.1', resolve));
  const server = relay.createRelay({ upstream: `http://127.0.0.1:${upstream.address().port}`, worldId: 'obs-director-qa-cotct', user: { id: 'OBSRecorder00001', name: 'QA Recorder', password: 'private-test' } });
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = server.address().port;
    for (const pathname of ['/join?probe=1', '/temporary/../join?probe=1', '/%6aoin?probe=1', '/%2e/join?probe=1', '/temporary%2f..%2fjoin?probe=1', '/join/?probe=1', '/JOIN?probe=1', '/temporary/../setup?probe=1', '/%61uth?probe=1', '/%71uit?probe=1']) {
      const response = await new Promise((resolve, reject) => {
        const req = http.request({ host: '127.0.0.1', port, method: 'POST', path: pathname, headers: { 'content-type': 'application/json' } }, (res) => { res.resume(); res.on('end', () => resolve(res.statusCode)); });
        req.on('error', reject); req.end('{"action":"join","userId":"different-user","password":"arbitrary"}');
      });
      assert.equal(response, 403, pathname);
      assert.equal(writes.length, 0, `${pathname} reached the upstream`);
    }
  } finally { await Promise.all([new Promise((r) => server.close(r)), new Promise((r) => upstream.close(r))]); }
});

test('GM sync touches only configured currently bound PCs and preserves other ownership and greater recorder permissions', async () => {
  const file = new URL('../macros/sync-recorder.js', import.meta.url);
  assert.equal(fs.existsSync(file), true, 'GM sync macro must be implemented');
  const writes = [], settingsWrites = [];
  function actor(id, ownership, type = 'character') { return { id, type, img: `assets/${id}.png`, ownership, update: async (value) => writes.push([id, value]) }; }
  const a = actor('new', { other: 3 }), b = actor('already-owner', { OBSRecorder00001: 3, other: 1 }), npc = actor('npc', {}, 'npc');
  const users = new Map([['gm', { id: 'gm', isGM: true }], ['OBSRecorder00001', { id: 'OBSRecorder00001', isGM: false }], ['pl', { id: 'pl', isGM: false, character: a }], ['pl2', { id: 'pl2', isGM: false, character: b }], ['pl3', { id: 'pl3', isGM: false, character: npc }], ['unconfigured', { id: 'unconfigured', isGM: false, character: actor('excluded', {}) }]]);
  const values = { recorderUserId: 'OBSRecorder00001', seatOrder: ['gm', 'pl', 'pl2', 'pl3'], portraitOverrides: { old: 'assets/old.png', 'already-owner': 'assets/original.png' } };
  const game = { user: { isGM: true }, system: { id: 'pf2e' }, users, settings: { get: (scope, key) => values[key], set: async (scope, key, value) => settingsWrites.push([scope, key, value]) }, scenes: [] };
  await vm.runInNewContext(fs.readFileSync(file, 'utf8'), { game, ui: { notifications: { info() {} } }, CONST: { DOCUMENT_OWNERSHIP_LEVELS: { OBSERVER: 2 } } });
  assert.equal(writes.length, 1); assert.equal(writes[0][0], 'new');
  assert.equal(writes[0][1]['ownership.OBSRecorder00001'], 2);
  assert.deepEqual(a.ownership, { other: 3 });
  assert.equal(settingsWrites.some(([scope]) => scope !== 'taka-obs-director' && scope !== 'obs-utils'), false);
  const portraitWrite = settingsWrites.find(([, key]) => key === 'portraitOverrides');
  assert.equal(portraitWrite[2].old, undefined); assert.equal(portraitWrite[2].new, undefined); assert.equal(portraitWrite[2]['already-owner'], 'assets/original.png');
  writes.length = 0; settingsWrites.length = 0;
  values.seatOrder = ['gm, pl, pl2, pl3'];
  await vm.runInNewContext(fs.readFileSync(file, 'utf8'), { game, ui: { notifications: { info() {} } }, CONST: { DOCUMENT_OWNERSHIP_LEVELS: { OBSERVER: 2 } } });
  assert.equal(writes.length, 1); assert.equal(writes[0][0], 'new');
  assert.equal(settingsWrites.find(([, key]) => key === 'portraitOverrides')[2]['already-owner'], 'assets/original.png');
  assert.equal(Array.from(settingsWrites.find(([, key]) => key === 'seatOrder')[2]).join(','), 'gm,pl,pl2,pl3');
  game.user.isGM = false;
  await assert.rejects(vm.runInNewContext(fs.readFileSync(file, 'utf8'), { game, ui: {}, CONST: { DOCUMENT_OWNERSHIP_LEVELS: { OBSERVER: 2 } } }), /GM/);
});
