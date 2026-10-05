const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { args, assertSourceSetup, hash } = require('./deploy.cjs');
const GM = 'QAGM000000000001';
const RECORDER = 'OBSRecorder00001';
const playerId = (seat) => `QAPL${String(seat).padStart(12, '0')}`;
const id = (key) => hash(key).slice(0, 16);
const SOURCE_MODULES = {
  cotct: ['battlezoo-eldamon-pf2e', 'battlezoo-eldamon-legends-pf2e', 'pf2e-vessel-cn'],
  sog: ['pf2e-item-activations', 'xdy-pf2e-workbench'],
  pnvfcgjbf2cjp7gz: ['pf2e-item-activations', 'pf2e-toolbelt', 'pf2e-third-party-automation', 'socketlib', 'pf2e-dailies'],
  '-': ['pf2e-team-plus-magic'],
  ujx5r8oipw7ercdr: ['battlezoo-eldamon-legends-pf2e', 'battlezoo-eldamon-pf2e', 'pf2e-item-activations']
};
function sourceModules(snapshot) {
  if (snapshot.system !== 'pf2e') return [];
  if (!Object.hasOwn(SOURCE_MODULES, snapshot.sourceWorldId)) throw Error('Unsupported PF2e QA source world');
  return SOURCE_MODULES[snapshot.sourceWorldId];
}
function mergeSupplementalSettings(snapshot, supplements) {
  const result = structuredClone(snapshot), keys = new Map(result.settingsRows.map(row => [row.value.key, row]));
  const rowIds = new Map(result.settingsRows.map(row => [row.key, row.value.key]));
  for (const supplement of supplements) {
    if (hash(supplement.bytes) !== supplement.sha256) throw Error('Supplemental setting fingerprint mismatch');
    const parsed = JSON.parse(supplement.bytes);
    if (parsed.schemaVersion !== 1 || parsed.sourceWorldId !== snapshot.sourceWorldId || snapshot.system !== 'pf2e') throw Error('Supplemental world identity mismatch');
    if (!Array.isArray(parsed.settingsRows) || (supplement.kind === 'pf2e' && parsed.system !== 'pf2e')) throw Error('Invalid supplemental setting scope');
    for (const row of parsed.settingsRows) {
      const value = row.value;
      if (!value || value.user != null) throw Error('Supplemental user setting refused');
      if (typeof value.key !== 'string' || !(supplement.kind === 'pf2e' ? value.key.startsWith('pf2e.') : supplement.kind === 'rules' && snapshot.sourceWorldId === 'pnvfcgjbf2cjp7gz' && value.key === 'pf2e-toolbelt.actionable.cast')) throw Error('Supplemental setting scope refused');
      if (typeof value._id !== 'string' || row.key !== `!settings!${value._id}` || typeof value.value !== 'string' || (rowIds.has(row.key) && rowIds.get(row.key) !== value.key)) throw Error('Invalid supplemental setting row');
      JSON.parse(value.value);
      const existing = keys.get(value.key);
      if (existing) {
        if (JSON.stringify(existing) !== JSON.stringify(row)) throw Error('Supplemental setting key conflict');
        continue;
      }
      result.settingsRows.push(row); keys.set(value.key, row); rowIds.set(row.key, value.key);
    }
  }
  return result;
}
function readFidelityManifest(root) {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'fidelity-manifest.json')));
  const entry = manifest.providerManifest;
  if (manifest.schemaVersion !== 1 || !Array.isArray(manifest.worlds) || entry?.file !== 'source-homebrew-provider-manifest.json') throw Error('Invalid fidelity manifest');
  const bytes = fs.readFileSync(path.join(root, entry.file));
  if (hash(bytes) !== entry.sha256) throw Error('Provider manifest fingerprint mismatch');
  return { ...manifest, providers: JSON.parse(bytes) };
}
function readSupplementalSettings(snapshot, root, manifest) {
  const world = manifest.worlds.find(w => w.worldId === snapshot.sourceWorldId);
  const kinds = snapshot.sourceWorldId === 'pnvfcgjbf2cjp7gz' ? ['pf2e', 'rules'] : ['pf2e'];
  if (!world || JSON.stringify(world.files.map(f => f.kind)) !== JSON.stringify(kinds)) throw Error('Missing audited supplemental settings');
  return mergeSupplementalSettings(snapshot, world.files.map(entry => {
    const expected = `${snapshot.sourceWorldId}.${entry.kind === 'pf2e' ? 'pf2e-settings' : 'rule-registry-settings'}.rows.json`;
    if (entry.file !== expected) throw Error('Invalid supplemental file path');
    return { ...entry, bytes: fs.readFileSync(path.join(root, entry.file)) };
  }));
}
function settingRow(key, value) { const _id = id(key); return { key: `!settings!${_id}`, value: { _id, key, value: JSON.stringify(value), user: null } }; }
function buildWorldRows(snapshot, { qaWorldId, room, recorderId = RECORDER, portraits = {}, skin = 'cotct' }) {
  if (!/^obs-director-qa-[a-z]+$/.test(qaWorldId) || !room?.startsWith(`${qaWorldId}-`)) throw Error('QA world and unique room are required');
  const settings = structuredClone(snapshot.settingsRows);
  const lookup = new Map(settings.map((row) => [row.value.key, row]));
  const connection = lookup.get('avclient-livekit.liveKitConnectionSettings');
  if (!connection) throw Error('Missing private LiveKit configuration');
  const value = JSON.parse(connection.value.value);
  if (value.room === room) throw Error('QA room matches source room');
  value.room = room;
  connection.value.value = JSON.stringify(value);
  const clock = lookup.get('core.time');
  if (!clock || !Number.isFinite(JSON.parse(clock.value.value))) throw Error('Missing canonical core.time');
  function set(key, value) {
    const existing = lookup.get(key);
    if (existing) existing.value.value = JSON.stringify(value);
    else { const row = settingRow(key, value); settings.push(row); lookup.set(key, row); }
  }
  const director = snapshot.system === 'pf2e';
  set('core.moduleConfiguration', { 'obs-utils': true, 'lib-wrapper': true, 'avclient-livekit': true, 'taka-obs-director': director, ...Object.fromEntries(sourceModules(snapshot).map(module => [module, true])) });
  set('avclient-livekit.debug', false); set('avclient-livekit.liveKitTrace', false); set('avclient-livekit.resetRoom', false); set('avclient-livekit.useExternalAV', false);
  set('obs-utils.obsModeUser', recorderId);
  set('taka-obs-director.enabled', director); set('taka-obs-director.recorderUserId', recorderId);
  set('taka-obs-director.seatOrder', [GM, ...snapshot.actorIds.map((_, i) => playerId(i + 1))]);
  set('taka-obs-director.portraitOverrides', portraits); set('taka-obs-director.skin', skin); set('taka-obs-director.resourceDetail', 'compact');
  const credentials = [];
  function user(_id, name, role, character = null) {
    const password = crypto.randomBytes(24).toString('base64url');
    const passwordSalt = crypto.randomBytes(32).toString('hex');
    credentials.push({ id: _id, name, password });
    return { key: `!users!${_id}`, value: { _id, name, role, character, passwordSalt, password: crypto.pbkdf2Sync(password, passwordSalt, 1000, 64, 'sha512').toString('hex'), color: '#68a6c2', avatar: 'icons/svg/mystery-man.svg', permissions: { BROADCAST_AUDIO: false, BROADCAST_VIDEO: false }, flags: {} } };
  }
  const users = [user(GM, 'QA GM', 4), user(recorderId, 'QA Recorder', 1), ...snapshot.actorIds.map((actorId, i) => user(playerId(i + 1), `QA PL ${i + 1}`, 1, actorId))];
  const actors = structuredClone(snapshot.actorRows);
  for (const row of actors) {
    const index = snapshot.actorIds.indexOf(row.value._id);
    if (index < 0) throw Error('Unexpected actor outside configured selection');
    row.value.folder = null;
    row.value.ownership = { default: 0, [recorderId]: 2, [playerId(index + 1)]: 3 };
  }
  const parentIds = new Set(snapshot.actorIds);
  for (const row of snapshot.embeddedRows) {
    const match = /^!actors\.(items|effects)!([^.]+)\.([^.]+)$/.exec(row.key);
    if (!match || !parentIds.has(match[2]) || row.value._id !== match[3]) throw Error('Invalid embedded actor row');
    const parent = actors.find((r) => r.value._id === match[2]).value;
    const references = parent[match[1]] ?? [];
    if (!references.includes(match[3])) throw Error('Embedded row has no parent reference');
    actors.push(structuredClone(row));
  }
  return { actors, settings, users, credentials };
}

async function writeRows(DB, folder, rows) {
  const db = new DB(folder, { valueEncoding: 'json' }); await db.open();
  try { await db.batch(rows.map(({ key, value }) => ({ type: 'put', key, value }))); } finally { await db.close(); }
}
function privateJSON(file, value) { fs.writeFileSync(file, JSON.stringify(value, null, 2), { flag: 'wx', mode: 0o600 }); }
function assetFile(relative, url = false) {
  const stripped = url ? decodeURIComponent(relative.split(/[?#]/)[0]) : relative;
  if (path.isAbsolute(stripped) || stripped.split(/[\\/]/).includes('..')) throw Error('Unsafe asset path');
  return stripped;
}
function verifyInventory(bytes, expectedHash) {
  const assets = JSON.parse(bytes);
  if (hash(JSON.stringify(assets)) !== expectedHash) throw Error('Private asset manifest fingerprint mismatch');
  return assets;
}
function linkFile(source, destination, expectedHash) {
  if (!fs.statSync(source).isFile() || (expectedHash && hash(fs.readFileSync(source)) !== expectedHash)) throw Error('Required QA asset changed');
  fs.mkdirSync(path.dirname(destination), { recursive: true, mode: 0o700 });
  if (fs.existsSync(destination)) { if (fs.realpathSync(destination) !== fs.realpathSync(source)) throw Error('QA asset collision'); return; }
  fs.symlinkSync(source, destination, 'file');
}
// Native pack databases may open writable: copy packs, link other package files.
function clonePackage(source, destination, { ownedDirectories = [] } = {}) {
  const resolvedSource = fs.realpathSync(source), resolvedDestination = path.resolve(destination);
  if (resolvedDestination === resolvedSource || resolvedDestination.startsWith(resolvedSource + path.sep) || resolvedSource.startsWith(resolvedDestination + path.sep) || fs.existsSync(destination)) throw Error('QA package destination must be new and isolated');
  fs.mkdirSync(destination, { recursive: true, mode: 0o700 });
  for (const name of fs.readdirSync(source)) {
    const from = path.join(source, name), to = path.join(destination, name);
    if (name === 'packs' || ownedDirectories.includes(name)) fs.cpSync(from, to, { recursive: true, dereference: true, errorOnExist: true, force: false });
    else fs.symlinkSync(from, to, fs.statSync(from).isDirectory() ? 'dir' : 'file');
  }
}
function patchRuneTransfer(source, destination, auditRoot) {
  const relative = 'scripts/rune-transfer.mjs', target = path.join(destination, relative);
  const ownedRoot = fs.realpathSync(destination), actual = fs.realpathSync(target);
  if (!actual.startsWith(ownedRoot + path.sep) || fs.realpathSync(source) === ownedRoot) throw Error('Rune patch requires an owned QA file');
  const original = fs.readFileSync(path.join(source, relative)), before = "const worlds=new Set(['-','sog','pnvfcgjbf2cjp7gz','team-automation-qa2']);";
  const after = "const worlds=new Set(['-','sog','pnvfcgjbf2cjp7gz','team-automation-qa2','obs-director-qa-fotrp']);";
  const text = original.toString('utf8');
  if (text.split(before).length !== 2 || hash(fs.readFileSync(target)) !== hash(original)) throw Error('Rune source changed before QA alias patch');
  const patched = text.replace(before, after);
  fs.writeFileSync(target, patched, { mode: 0o600 });
  if (hash(fs.readFileSync(path.join(source, relative))) !== hash(original)) throw Error('Rune source changed during QA alias patch');
  const result = { file: relative, sourcePath: path.join(source, relative), sourceSha256: hash(original), qaSha256: hash(patched), qaWorldId: 'obs-director-qa-fotrp' };
  privateJSON(path.join(auditRoot, 'rune-transfer-qa.json'), result);
  const line = text.slice(0, text.indexOf(before)).split('\n').length;
  fs.writeFileSync(path.join(auditRoot, 'rune-transfer-qa.diff'), `--- source/${relative}\n+++ qa/${relative}\n@@ -${line} +${line} @@\n-${before}\n+${after}\n`, { flag: 'wx', mode: 0o600 });
  return result;
}
function restrictPrivate(folder) {
  for (const name of fs.readdirSync(folder)) {
    const full = path.join(folder, name), stat = fs.lstatSync(full);
    if (stat.isSymbolicLink()) continue;
    if (stat.isDirectory()) { fs.chmodSync(full, 0o700); restrictPrivate(full); }
    else fs.chmodSync(full, 0o600);
  }
}
async function prepareQA({ destination, inputRoot, supplementalRoot, privateOutputRoot = inputRoot, sourceDataRoot, foundryRoot, moduleSource, inputs, DB, socketPath, checkSetup = () => assertSourceSetup({ dataRoot: sourceDataRoot }) }) {
  if (process.platform !== 'linux') throw Error('Native QA preparation requires the CN Linux host');
  if (!destination || !socketPath || !inputRoot || !moduleSource || !foundryRoot) throw Error('Missing QA preparation paths');
  if (destination !== '/root/obs-director-delivery-20260930/qa-foundry' || socketPath !== '/root/obs-director-delivery-20260930/qa.sock') throw Error('QA destination must be the dedicated task directory and socket');
  if (fs.existsSync(destination)) throw Error('QA destination already exists; never overwrite a native run');
  const manifest = JSON.parse(fs.readFileSync(path.join(inputRoot, 'resolved-manifest.json')));
  const selection = JSON.parse(fs.readFileSync(path.join(inputRoot, 'selection.json')));
  const fidelity = supplementalRoot ? readFidelityManifest(supplementalRoot) : null;
  if (selection.worlds.some(w => w.system === 'pf2e') && !fidelity) throw Error('PF2e QA requires audited fidelity supplements');
  if (fidelity && (path.resolve(privateOutputRoot) === path.resolve(inputRoot) || path.resolve(privateOutputRoot) === path.resolve(supplementalRoot) || fs.existsSync(privateOutputRoot))) throw Error('Fidelity preparation requires a new private output directory');
  const prepared = [], sourceRooms = new Set();
  for (const selected of selection.worlds) {
    const bytes = fs.readFileSync(path.join(inputRoot, `${selected.worldId}.rows.json`));
    const audit = manifest.worlds.find((w) => w.worldId === selected.worldId);
    if (!audit || hash(bytes) !== audit.privateRowFileSha256) throw Error('Private snapshot fingerprint mismatch');
    let snapshot = JSON.parse(bytes);
    if (JSON.stringify(snapshot.actorIds) !== JSON.stringify(selected.actorIds)) throw Error('Private actor selection mismatch');
    const assetBytes = fs.readFileSync(path.join(inputRoot, `${selected.worldId}.resolved-assets.json`));
    const assets = verifyInventory(assetBytes, audit.assetInventorySha256);
    if (snapshot.system === 'pf2e') snapshot = readSupplementalSettings(snapshot, supplementalRoot, fidelity);
    const connection = snapshot.settingsRows.find((row) => row.value.key === 'avclient-livekit.liveKitConnectionSettings');
    sourceRooms.add(JSON.parse(connection.value.value).room);
    const production = inputs.worlds.find((w) => w.worldId === selected.worldId);
    const skin = production?.skin ?? 'alien';
    const qaWorldId = `obs-director-qa-${skin}`;
    const room = `${qaWorldId}-${crypto.randomBytes(16).toString('hex')}`;
    const portraits = production ? Object.fromEntries(production.players.filter((p) => p.portrait.originalVerified).map((p) => [p.actorId, p.portrait.defaultPath])) : {};
    prepared.push({ snapshot, assets, qaWorldId, room, skin, rows: buildWorldRows(snapshot, { qaWorldId, room, portraits, skin: production?.skin ?? 'auto' }) });
  }
  if (new Set(prepared.map((p) => p.room)).size !== prepared.length || prepared.some((p) => sourceRooms.has(p.room))) throw Error('QA room isolation failed');
  const modules = [...new Set(['obs-utils', 'lib-wrapper', 'avclient-livekit', ...prepared.flatMap(p => sourceModules(p.snapshot))])];
  for (const preparedWorld of prepared.filter(p => p.snapshot.system === 'pf2e')) {
    const active = new Set(['obs-utils', 'lib-wrapper', 'avclient-livekit', ...sourceModules(preparedWorld.snapshot)]);
    for (const moduleId of sourceModules(preparedWorld.snapshot)) {
      const sourceManifest = JSON.parse(fs.readFileSync(path.join(sourceDataRoot, 'Data/modules', moduleId, 'module.json')));
      if (sourceManifest.id !== moduleId) throw Error('QA source package identity mismatch');
      for (const dependency of sourceManifest.relationships?.requires ?? []) if (dependency.type === 'module' && !active.has(dependency.id)) throw Error('Missing QA supporting module dependency');
      for (const pack of sourceManifest.packs ?? []) if (!pack.path?.startsWith('packs/') || pack.path.split(/[\\/]/).includes('..')) throw Error('Unsupported QA pack database path');
      const registered = sourceManifest.flags?.[moduleId]?.['pf2e-homebrew'];
      if (registered) {
        const provider = fidelity.providers.modules.find(m => m.id === moduleId);
        const sourceWorld = fidelity.providers.worlds.find(w => w.worldId === preparedWorld.snapshot.sourceWorldId);
        if (!provider || !sourceWorld?.activeHomebrewProviders.includes(moduleId) || provider.version !== sourceManifest.version || JSON.stringify(provider.flags) !== JSON.stringify(registered)) throw Error('QA homebrew provider fingerprint mismatch');
      }
    }
  }
  await checkSetup();
  fs.mkdirSync(privateOutputRoot, { recursive: true, mode: 0o700 });
  for (const dir of ['Config', 'Data/worlds', 'Data/modules', 'Data/systems', 'Logs']) fs.mkdirSync(path.join(destination, dir), { recursive: true, mode: 0o700 });
  fs.copyFileSync(path.join(inputRoot, 'license.json'), path.join(destination, 'Config/license.json'));
  fs.chmodSync(path.join(destination, 'Config/license.json'), 0o600);
  const adminPassword = crypto.randomBytes(24).toString('base64url');
  const passwordSalt = crypto.randomBytes(32).toString('hex');
  privateJSON(path.join(destination, 'Config/options.json'), { dataPath: destination, world: null, unixSocket: socketPath, port: 30991, upnp: false, telemetry: false, compressStatic: false, language: 'en.core', passwordSalt, adminPassword: crypto.pbkdf2Sync(adminPassword, passwordSalt, 1000, 64, 'sha512').toString('hex') });
  await checkSetup();
  for (const system of new Set(prepared.map((p) => p.snapshot.system))) { await checkSetup(); clonePackage(path.join(sourceDataRoot, 'Data/systems', system), path.join(destination, 'Data/systems', system)); }
  let runePatch = null;
  for (const mod of modules) {
    await checkSetup();
    const source = path.join(sourceDataRoot, 'Data/modules', mod), target = path.join(destination, 'Data/modules', mod);
    clonePackage(source, target, { ownedDirectories: mod === 'pf2e-third-party-automation' ? ['scripts'] : [] });
    if (mod === 'pf2e-third-party-automation') runePatch = patchRuneTransfer(source, target, privateOutputRoot);
  }
  const { runtimeFiles } = await import('./build-delivery.mjs');
  for (const file of runtimeFiles) {
    const target = path.join(destination, 'Data/modules/taka-obs-director', file);
    fs.mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 }); fs.copyFileSync(path.join(moduleSource, file), target);
  }
  const report = [], credentials = { adminPassword, worlds: [] };
  for (const preparedWorld of prepared) {
    await checkSetup();
    const { snapshot, qaWorldId, rows, assets, skin } = preparedWorld;
    const folder = path.join(destination, 'Data/worlds', qaWorldId);
    fs.mkdirSync(path.join(folder, 'data'), { recursive: true, mode: 0o700 });
    privateJSON(path.join(folder, 'world.json'), { id: qaWorldId, title: `OBS Director QA ${skin}`, system: snapshot.system, coreVersion: '14.368', systemVersion: snapshot.system === 'pf2e' ? '8.5.1' : '4.1.14', compatibility: { minimum: '14', verified: '14.368' }, packs: [] });
    for (const asset of assets.filter((a) => a.exists && a.routeRoot === 'Data')) {
      const relative = assetFile(asset.relativePath);
      // Package links above already resolve packaged art; other assets are single-file links only.
      const target = path.join(destination, 'Data', relative);
      if (fs.existsSync(target)) { if (hash(fs.readFileSync(target)) !== asset.sha256) throw Error('QA packaged asset fingerprint mismatch'); }
      else linkFile(path.join(sourceDataRoot, 'Data', relative), target, asset.sha256);
    }
    const production = inputs.worlds.find((w) => w.worldId === snapshot.sourceWorldId);
    for (const player of production?.players.filter((p) => p.portrait.originalVerified) ?? []) {
      const relative = assetFile(player.portrait.defaultPath, true), target = path.join(destination, 'Data', relative);
      if (fs.existsSync(target)) { if (hash(fs.readFileSync(target)) !== player.portrait.sha256) throw Error('QA original portrait fingerprint mismatch'); }
      else linkFile(path.join(sourceDataRoot, 'Data', relative), target, player.portrait.sha256);
    }
    const npcId = 'QAPublicNPC00001';
    rows.actors.push({ key: `!actors!${npcId}`, value: { _id: npcId, name: 'QA Public NPC', type: snapshot.system === 'pf2e' ? 'npc' : 'creature', img: 'icons/svg/mystery-man.svg', folder: null, ownership: { default: 0 }, system: {}, items: [], effects: [] } });
    const sceneId = 'QATestScene00001';
    const tokens = [...snapshot.actorIds, npcId].map((actorId, i) => ({ _id: `QAToken${String(i + 1).padStart(9, '0')}`, name: actorId === npcId ? 'QA Public NPC' : rows.actors.find((a) => a.value._id === actorId).value.name, actorId, actorLink: true, x: 200 + (i % 3) * 300, y: 200 + Math.floor(i / 3) * 300, width: 1, height: 1, texture: { src: 'icons/svg/mystery-man.svg' }, hidden: false, disposition: actorId === npcId ? -1 : 1 }));
    const scenes = [{ key: `!scenes!${sceneId}`, value: { _id: sceneId, name: 'QA Test Scene', active: true, navigation: true, width: 2400, height: 1600, padding: 0.1, grid: { type: 1, size: 100 }, backgroundColor: '#203344', background: { src: null }, tokenVision: false, globalLight: true, ownership: { default: 2 }, tokens: tokens.map((t) => t._id), walls: [], lights: [], sounds: [], drawings: [], notes: [], regions: [], tiles: [], flags: {} } }, ...tokens.map((value) => ({ key: `!scenes.tokens!${sceneId}.${value._id}`, value }))];
    await writeRows(DB, path.join(folder, 'data/actors'), rows.actors);
    await writeRows(DB, path.join(folder, 'data/users'), rows.users);
    await writeRows(DB, path.join(folder, 'data/settings'), rows.settings);
    await writeRows(DB, path.join(folder, 'data/scenes'), scenes);
    credentials.worlds.push({ worldId: qaWorldId, users: rows.credentials, room: preparedWorld.room });
    report.push({ worldId: qaWorldId, sourceWorldId: snapshot.sourceWorldId, pcCount: snapshot.actorIds.length, embeddedRows: snapshot.embeddedRows.length, missingAssets: assets.filter((a) => !a.exists).length, freshUsers: rows.users.length, freshRoom: true, canonicalClockRestored: true, directorEnabled: snapshot.system === 'pf2e', supportingModules: sourceModules(snapshot), supplementalSettingsRestored: snapshot.system === 'pf2e' });
  }
  privateJSON(path.join(privateOutputRoot, 'qa-credentials.json'), credentials);
  privateJSON(path.join(privateOutputRoot, 'qa-preparation-report.json'), { worlds: report, installedModules: [...modules, 'taka-obs-director'], sharedPackageCode: 'symlinked; rune scripts privately copied; no source update permitted', packDatabases: 'private copies', runePatch, socketOnly: true });
  restrictPrivate(destination);
  restrictPrivate(privateOutputRoot);
  return { worlds: report, socketOnly: true, packDatabases: 'private copies' };
}
module.exports = { buildWorldRows, prepareQA, mergeSupplementalSettings, readFidelityManifest, readSupplementalSettings, clonePackage, patchRuneTransfer, verifyInventory, assetFile, GM, RECORDER, playerId };
if (require.main === module) (async () => {
  const a = args(process.argv.slice(2)); const { ClassicLevel } = require(a['classic-level'] ?? '/root/foundryvtt/node_modules/classic-level');
  console.log(JSON.stringify(await prepareQA({ destination: a.destination, socketPath: a.socket, inputRoot: a['input-root'], supplementalRoot: a['supplemental-root'], privateOutputRoot: a['private-output-root'], sourceDataRoot: a['source-data-root'], foundryRoot: a['foundry-root'], moduleSource: a['module-source'], inputs: JSON.parse(fs.readFileSync(a.inputs)), DB: ClassicLevel })));
})().catch(() => { console.error('QA preparation failed; private inputs were not printed. Inspect the dedicated preparation directory.'); process.exitCode = 1; });
