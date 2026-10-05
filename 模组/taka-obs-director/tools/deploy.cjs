const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');

const MODULE = 'taka-obs-director';
const hash = (value) => crypto.createHash('sha256').update(value).digest('hex');
const idFor = (key) => hash(key).slice(0, 16);
function args(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!/^--[a-z-]+$/.test(argv[i]) || !argv[i + 1] || argv[i + 1].startsWith('--')) throw Error('Expected named option and value');
    result[argv[i].slice(2)] = argv[i + 1];
  }
  return result;
}
function safeChild(root, relative) {
  const target = path.resolve(root, relative);
  const rel = path.relative(path.resolve(root), target);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) throw Error('Path escapes expected directory');
  let current = path.resolve(root);
  for (const part of rel.split(path.sep)) { current = path.join(current, part); if (fs.existsSync(current) && fs.lstatSync(current).isSymbolicLink()) throw Error('Refusing symlink destination'); }
  return target;
}
async function assertSourceSetup({ dataRoot, joinUrl = 'http://127.0.0.1:30001/join', fetchJoin = () => fetch(joinUrl, { redirect: 'manual', signal: AbortSignal.timeout(5000) }) }) {
  const options = JSON.parse(fs.readFileSync(path.join(dataRoot, 'Config/options.json')));
  const response = await fetchJoin();
  if (options.world !== null || response.status !== 200 || !(await response.text()).includes('There is currently no active game session')) throw Error('Source must remain in Setup (null world and inactive /join)');
}
async function rawRows(DB, folder) {
  const db = new DB(folder, { valueEncoding: 'buffer', createIfMissing: false });
  await db.open();
  try { return new Map(await db.iterator().all()); } finally { await db.close(); }
}
async function snapshotRows(DB, folder, checkSetup) {
  await checkSetup();
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'director-settings-'));
  try {
    fs.cpSync(folder, path.join(tmp, 'db'), { recursive: true });
    return await rawRows(DB, path.join(tmp, 'db'));
  } finally { fs.rmSync(tmp, { recursive: true, force: true }); }
}
function desiredSettings(world, recorderId) {
  return {
    [`${MODULE}.enabled`]: true,
    [`${MODULE}.recorderUserId`]: recorderId,
    [`${MODULE}.seatOrder`]: world.defaultSeatOrder.map((seat) => seat.userId),
    [`${MODULE}.portraitOverrides`]: Object.fromEntries(world.players.filter((p) => p.portrait?.originalVerified === true).map((p) => [p.actorId, p.portrait.defaultPath])),
    [`${MODULE}.skin`]: world.skin,
    [`${MODULE}.resourceDetail`]: 'compact',
  };
}
function settingsChanges(rows, desired) {
  const records = new Map();
  const targetNames = new Set(['core.moduleConfiguration', ...Object.keys(desired)]);
  for (const [key, bytes] of rows) {
    const value = JSON.parse(bytes);
    if (!targetNames.has(value.key) || value.user) continue;
    if (records.has(value.key)) throw Error('Duplicate world setting');
    records.set(value.key, { key, value, bytes });
  }
  const config = records.get('core.moduleConfiguration');
  const configuration = config ? JSON.parse(config.value.value) : {};
  const values = { 'core.moduleConfiguration': { ...configuration, [MODULE]: true }, ...desired };
  const changes = [];
  for (const [name, value] of Object.entries(values)) {
    const current = records.get(name);
    if (current && JSON.stringify(JSON.parse(current.value.value)) === JSON.stringify(value)) continue;
    const id = current?.value._id ?? idFor(name);
    const key = current?.key ?? `!settings!${id}`;
    if (!current && rows.has(key)) throw Error('Setting record ID collision');
    const record = current ? { ...current.value, value: JSON.stringify(value) } : { _id: id, key: name, value: JSON.stringify(value), user: null };
    changes.push({ key, name, beforeHash: current ? hash(current.bytes) : null, bytes: Buffer.from(JSON.stringify(record)) });
  }
  return changes;
}
const rowsHash = (rows) => hash(Buffer.concat([...rows].sort(([a], [b]) => a.localeCompare(b)).flatMap(([key, value]) => [Buffer.from(key + '\0'), value, Buffer.from('\0')])));

async function deploy({ dataRoot, inputs, DB, checkSetup = () => assertSourceSetup({ dataRoot }), backupRoot, applyPlan, moduleSource }) {
  if (!dataRoot || !DB || inputs?.recorderUserId !== 'OBSRecorder00001' || !inputs.worlds?.length) throw Error('Invalid deployment inputs');
  const ids = new Set();
  for (const world of inputs.worlds) {
    if (world.system !== 'pf2e' || !['cotct', 'sog', 'fotrp', 'av', 'bob'].includes(world.skin) || !/^[\w-]+$/.test(world.worldId) || ids.has(world.worldId)) throw Error('Invalid production world selection');
    ids.add(world.worldId);
    if (!world.defaultSeatOrder?.length || !world.players?.length) throw Error('Missing configured roster');
  }
  await checkSetup();
  const worlds = [];
  const pending = [];
  for (const world of inputs.worlds) {
    const folder = safeChild(dataRoot, `Data/worlds/${world.worldId}/data/settings`);
    const rows = await snapshotRows(DB, folder, checkSetup);
    const changes = settingsChanges(rows, desiredSettings(world, inputs.recorderUserId));
    worlds.push({ worldId: world.worldId, beforeHash: rowsHash(rows), changes: changes.map((c) => ({ key: c.key, name: c.name, beforeHash: c.beforeHash, afterHash: hash(c.bytes) })) });
    pending.push({ folder, rows, changes, worldId: world.worldId });
  }
  let moduleFiles = [];
  if (moduleSource) {
    const { runtimeFiles } = await import('./build-delivery.mjs');
    moduleFiles = runtimeFiles.map((file) => ({ path: file, sha256: hash(fs.readFileSync(path.join(moduleSource, file))) }));
  }
  const plan = { schemaVersion: 1, moduleId: MODULE, worlds, moduleFiles };
  plan.fingerprint = hash(JSON.stringify(plan));
  if (!applyPlan) return plan;
  if (applyPlan.fingerprint !== plan.fingerprint || JSON.stringify(applyPlan) !== JSON.stringify(plan)) throw Error('Reviewed deployment plan is stale');
  const changed = pending.reduce((total, p) => total + p.changes.length, 0);
  if (!backupRoot) throw Error('Exact apply requires a private backup root');
  await checkSetup();
  fs.mkdirSync(backupRoot, { recursive: true, mode: 0o700 });
  const backupPath = fs.mkdtempSync(path.join(backupRoot, 'deploy-'));
  fs.chmodSync(backupPath, 0o700);
  for (const p of pending) {
    await checkSetup();
    fs.cpSync(p.folder, path.join(backupPath, p.worldId, 'settings'), { recursive: true });
  }
  fs.writeFileSync(path.join(backupPath, 'plan.json'), JSON.stringify(plan, null, 2), { mode: 0o600 });
  if (moduleSource) {
    await checkSetup();
    const target = safeChild(dataRoot, `Data/modules/${MODULE}`);
    if (fs.existsSync(target)) fs.cpSync(target, path.join(backupPath, 'module'), { recursive: true });
    for (const entry of moduleFiles) {
      const bytes = fs.readFileSync(path.join(moduleSource, entry.path));
      if (hash(bytes) !== entry.sha256) throw Error('Module source changed after plan');
      const destination = safeChild(target, entry.path);
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      fs.writeFileSync(destination, bytes);
      if (hash(fs.readFileSync(destination)) !== entry.sha256) throw Error('Module copy verification failed');
    }
  }
  for (const p of pending) {
    await checkSetup();
    const current = await snapshotRows(DB, p.folder, checkSetup);
    if (rowsHash(current) !== rowsHash(p.rows)) throw Error('Source settings changed immediately before write');
    if (!p.changes.length) continue;
    await checkSetup();
    const db = new DB(p.folder, { valueEncoding: 'buffer', createIfMissing: false });
    await db.open();
    try { await db.batch(p.changes.map((c) => ({ type: 'put', key: c.key, value: c.bytes }))); } finally { await db.close(); }
    const after = await snapshotRows(DB, p.folder, checkSetup);
    const expected = new Map(p.rows);
    for (const change of p.changes) expected.set(change.key, change.bytes);
    if (rowsHash(after) !== rowsHash(expected)) throw Error('Settings readback failed; retain backup for recovery');
  }
  return { changed, backupPath, readbackVerified: true, plan };
}

module.exports = { deploy, assertSourceSetup, desiredSettings, settingsChanges, safeChild, args, rawRows, hash };
if (require.main === module) (async () => {
  const a = args(process.argv.slice(2));
  const { ClassicLevel } = require(a['classic-level'] ?? '/root/foundryvtt/node_modules/classic-level');
  const result = await deploy({ dataRoot: a['data-root'], inputs: JSON.parse(fs.readFileSync(a.inputs)), DB: ClassicLevel, backupRoot: a['backup-root'], moduleSource: a['module-source'], applyPlan: a['apply-plan'] ? JSON.parse(fs.readFileSync(a['apply-plan'])) : undefined, checkSetup: () => assertSourceSetup({ dataRoot: a['data-root'], joinUrl: a['join-url'] }) });
  if (a['plan-out']) fs.writeFileSync(a['plan-out'], JSON.stringify(result, null, 2), { flag: 'wx', mode: 0o600 });
  console.log(JSON.stringify(a['apply-plan'] ? { changed: result.changed, readbackVerified: result.readbackVerified } : result));
})().catch(() => { console.error('Deployment failed; inspect the reviewed inputs and private backup.'); process.exitCode = 1; });
