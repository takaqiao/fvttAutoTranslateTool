import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash, randomUUID} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const run = promisify(execFile);
const targets = {
 pf2e: {file: 'Data/systems/pf2e/pf2e.mjs', manifest: 'Data/systems/pf2e/system.json', id: 'pf2e'},
 toolbelt: {file: 'Data/modules/pf2e-toolbelt/scripts/main.js', manifest: 'Data/modules/pf2e-toolbelt/module.json', id: 'pf2e-toolbelt'},
 patreon: {file: 'Data/modules/patreon-v3/src/index.js', manifest: 'Data/modules/patreon-v3/module.json', id: 'patreon-v3'}
};
const groups = {native: ['pf2e', 'toolbelt'], patreon: ['patreon']};
const sameIdentity = (a, b) => a.dev === b.dev && a.ino === b.ino && a.size === b.size && a.mtimeMs === b.mtimeMs;

async function readRegular(filename, io) {
 const before = await io.lstat(filename);
 assert(before.isFile() && !before.isSymbolicLink() && before.nlink === 1, `Unsafe regular file: ${filename}`);
 assert.equal(await io.realpath(filename), filename, `Noncanonical file: ${filename}`);
 const bytes = await io.readFile(filename), after = await io.lstat(filename);
 assert(sameIdentity(before, after), `File changed while reading: ${filename}`);
 return {filename, bytes, sha256: digest(bytes), stat: after};
}
async function unchanged(snapshot, io) {
 const now = await readRegular(snapshot.filename, io);
 assert(now.sha256 === snapshot.sha256 && sameIdentity(now.stat, snapshot.stat), `File changed during preparation: ${snapshot.filename}`);
}
async function syncDirectory(directory, io) {
 // Windows cannot open directories this way; the managed Linux startup fsyncs them.
 if (process.platform === 'win32') return;
 const handle = await io.open(directory, 'r');
 try { await handle.sync(); } finally { await handle.close(); }
}
async function saveExclusive(filename, bytes, mode, io) {
 const handle = await io.open(filename, 'wx', mode);
 try { await handle.writeFile(bytes); await handle.sync(); } finally { await handle.close(); }
 await syncDirectory(path.dirname(filename), io);
}
async function readPublished(filename, io) {
 const stat = await io.lstat(filename);
 if (stat.isFile() && !stat.isSymbolicLink() && stat.nlink === 2) {
  const temp = filename + '.publishing'; let temporary;
  try { temporary = await io.lstat(temp); } catch { throw Error(`Unknown publication links: ${filename}`); }
  assert(temporary.isFile() && !temporary.isSymbolicLink() && temporary.nlink === 2 && sameIdentity(stat, temporary), `Unknown publication links: ${filename}`);
  assert.equal(await io.realpath(filename), filename); assert.equal(await io.realpath(temp), temp);
  await io.unlink(temp); await syncDirectory(path.dirname(filename), io);
 }
 return readRegular(filename, io);
}
async function publishExclusive(filename, bytes, mode, io) {
 try { await readPublished(filename, io); throw Object.assign(Error(`Publication exists: ${filename}`), {code: 'EEXIST'}); }
 catch (error) { if (error.code !== 'ENOENT') throw error; }
 const temp = filename + '.publishing';
 try {
  // A final artifact is published only after this temporary file is complete.
  // A previous interrupted temporary write therefore cannot authorize source IO.
  await readRegular(temp, io); await io.unlink(temp);
 } catch (error) { if (error.code !== 'ENOENT') throw error; }
 let created = false;
 try {
  const handle = await io.open(temp, 'wx', mode); created = true;
  try { await handle.writeFile(bytes); await handle.sync(); } finally { await handle.close(); }
  // link is an atomic, exclusive publication: an existing backup or journal wins.
  await io.link(temp, filename);
  await io.unlink(temp); created = false; await syncDirectory(path.dirname(filename), io);
 } finally {
  if (created) { await io.rm(temp, {force: true}); await syncDirectory(path.dirname(filename), io); }
 }
}
async function temporary(filename, bytes, mode, io) {
 const name = `${filename}.third-party-${randomUUID()}.tmp.mjs`;
 await saveExclusive(name, bytes, mode, io);
 return name;
}
async function checkSyntax(filename) {
 try { await run(process.execPath, ['--check', filename], {timeout: 30000, maxBuffer: 1024 * 1024}); }
 catch (error) { throw Error(`Generated syntax failed: ${filename}: ${error.stderr || error.message}`); }
}
async function replace(filename, candidate, expected, io) {
 if (expected) await unchanged(expected, io);
 await io.rename(candidate, filename);
 await syncDirectory(path.dirname(filename), io);
}
async function ensureBackupDirectory(directory, io) {
 for (const current of [path.dirname(directory), directory]) {
  try { await io.mkdir(current, {mode: 0o700}); } catch (error) { if (error.code !== 'EEXIST') throw error; }
  const stat = await io.lstat(current);
  assert(stat.isDirectory() && !stat.isSymbolicLink(), 'Unsafe patch backup directory');
  assert.equal(await io.realpath(current), current, 'Patch backup directory changed');
  await syncDirectory(path.dirname(current), io);
 }
}
async function saveBackup(snapshot, role, directory, io) {
 const filename = path.join(directory, `${role}-${snapshot.sha256}.bin`);
 try { await publishExclusive(filename, snapshot.bytes, 0o600, io); }
 catch (error) { if (error.code !== 'EEXIST') throw error; }
 assert.equal((await readPublished(filename, io)).sha256, snapshot.sha256, `Original backup differs: ${role}`);
}

async function recoverPending({group, dataPath, directory, io, idle}) {
 const filename = path.join(directory, `${group}.pending.json`);
 let pending;
 try { pending = await readPublished(filename, io); } catch (error) { if (error.code === 'ENOENT') return; throw error; }
 assert(pending.bytes.length <= 16384, 'Oversized patch journal');
 const value = JSON.parse(pending.bytes);
 assert(value.schema === 1 && value.group === group && Array.isArray(value.entries), 'Invalid patch journal');
 assert(value.entries.length > 0 && value.entries.length <= groups[group].length, 'Invalid patch journal entries');
 const seen = new Set(), entries = [];
 for (const entry of value.entries) {
  assert(groups[group].includes(entry.role) && !seen.has(entry.role), 'Unknown or duplicate patch journal role'); seen.add(entry.role);
  assert(/^[a-f0-9]{64}$/.test(entry.before) && /^[a-f0-9]{64}$/.test(entry.after) && entry.before !== entry.after, 'Invalid patch journal hash');
  assert(Number.isInteger(entry.mode) && entry.mode >= 0 && entry.mode <= 0o777, 'Invalid patch journal mode');
  const current = await readRegular(path.join(dataPath, targets[entry.role].file), io);
  assert([entry.before, entry.after].includes(current.sha256), `Unknown later source edit prevents recovery: ${entry.role}`);
  const backup = await readPublished(path.join(directory, `${entry.role}-${entry.before}.bin`), io);
  assert.equal(backup.sha256, entry.before, `Original backup is corrupt: ${entry.role}`);
  entries.push({...entry, current, backup});
 }
 // Inspect every member and original before changing any member of a pending pair.
 await idle();
 await unchanged(pending, io);
 for (const entry of entries) await unchanged(entry.current, io);
 if (!entries.every(entry => entry.current.sha256 === entry.after)) {
  for (const entry of entries.filter(entry => entry.current.sha256 === entry.after)) {
   const candidate = await temporary(entry.current.filename, entry.backup.bytes, entry.mode, io);
   try { await replace(entry.current.filename, candidate, entry.current, io); }
   finally { await io.rm(candidate, {force: true}); }
  }
 }
 await io.unlink(filename); await syncDirectory(directory, io);
}

async function buildGroup(group, snapshots, builders) {
 const bytes = role => snapshots[role].source.bytes;
 const version = role => snapshots[role].version;
 if (group === 'native') {
  const bridge = await builders.buildNativeBridge({source: bytes('pf2e'), version: version('pf2e')});
  assert(Buffer.isBuffer(bridge.buffer), 'PF2e builder did not return source bytes');
  const pair = await builders.buildSharedManualPair({pf2eSource: bridge.buffer, toolbeltSource: bytes('toolbelt'), pf2eVersion: version('pf2e'), toolbeltVersion: version('toolbelt')});
  assert(Buffer.isBuffer(pair.pf2e) && Buffer.isBuffer(pair.toolbelt), 'Shared manual-pool builder did not return both source files');
  return {pf2e: pair.pf2e, toolbelt: pair.toolbelt};
 }
 const paid = await builders.buildPatreonSource({source: bytes('patreon'), version: version('patreon'), pf2eVersion: snapshots.pf2eVersion});
 assert(Buffer.isBuffer(paid.buffer), 'Patreon builder did not return source bytes');
 return {patreon: paid.buffer};
}
async function defaultBuilders() {
 const [native, patreon] = await Promise.all([import('./native.mjs'), import('./patreon.mjs')]);
 return {buildNativeBridge: native.buildNativeBridge, buildSharedManualPair: native.buildSharedManualPair, buildPatreonSource: patreon.buildPatreonSource};
}

export async function maintainSourcePatches({dataPath, assertIdle, builders, io = fs, syntaxCheck = checkSyntax, report = value => console.log(JSON.stringify(value))} = {}) {
 assert(typeof assertIdle === 'function', 'An idle service guard is required');
 assert(path.isAbsolute(dataPath), 'An absolute dataPath is required');
 assert.equal(await io.realpath(dataPath), dataPath, 'dataPath must be canonical');
 const idle = async () => { const result = await assertIdle(); assert(result?.idle === true, 'Source maintenance requires an idle server'); return result; };
 await idle();
 builders ??= await defaultBuilders();
 const directory = path.join(dataPath, 'Backups/pf2e-third-party-patches'), results = [];
 for (const [group, roles] of Object.entries(groups)) {
  let result, journalWritten = false, candidates = [];
  try {
   await recoverPending({group, dataPath, directory, io, idle});
   const snapshots = {};
   for (const role of roles) {
    const source = await readRegular(path.join(dataPath, targets[role].file), io);
    const manifest = await readRegular(path.join(dataPath, targets[role].manifest), io);
    const module = JSON.parse(manifest.bytes);
    assert.equal(module.id, targets[role].id, `Unexpected installed package: ${role}`);
    snapshots[role] = {source, manifest, version: String(module.version ?? '')};
   }
   if (group === 'patreon') {
    const manifest = await readRegular(path.join(dataPath, targets.pf2e.manifest), io);
    const system = JSON.parse(manifest.bytes); assert.equal(system.id, 'pf2e');
    snapshots.pf2eVersion = String(system.version ?? ''); snapshots.systemManifest = manifest;
   }
   let outputs;
   try { outputs = await buildGroup(group, snapshots, builders); }
   catch (error) { error.sourceSeamMismatch = true; throw error; }
   const changed = roles.filter(role => !outputs[role].equals(snapshots[role].source.bytes));
   if (!changed.length) result = {group, status: 'unchanged'};
   else {
    // The complete group is built before creating backups or replacement files.
    await ensureBackupDirectory(directory, io);
    for (const role of changed) {
     const source = snapshots[role].source;
     await saveBackup(source, role, directory, io);
     const temp = await temporary(source.filename, outputs[role], source.stat.mode & 0o777, io);
     candidates.push({role, temp});
     await syntaxCheck(temp);
    }
    await idle();
    for (const role of roles) { await unchanged(snapshots[role].source, io); await unchanged(snapshots[role].manifest, io); }
    if (snapshots.systemManifest) await unchanged(snapshots.systemManifest, io);
    const entries = changed.map(role => ({role, before: snapshots[role].source.sha256, after: digest(outputs[role]), mode: snapshots[role].source.stat.mode & 0o777}));
    await publishExclusive(path.join(directory, `${group}.pending.json`), Buffer.from(JSON.stringify({schema: 1, group, entries})), 0o600, io);
    journalWritten = true;
    for (const {role, temp} of candidates) {
     await replace(snapshots[role].source.filename, temp, snapshots[role].source, io);
     assert.equal((await readRegular(snapshots[role].source.filename, io)).sha256, digest(outputs[role]), `Source output differs: ${role}`);
    }
    await io.unlink(path.join(directory, `${group}.pending.json`)); await syncDirectory(directory, io); journalWritten = false;
    result = {group, status: 'patched', files: entries.map(entry => ({file: targets[entry.role].file, before: entry.before, after: entry.after}))};
   }
  } catch (error) {
   let recoveryError;
   if (journalWritten) {
    try { await recoverPending({group, dataPath, directory, io, idle}); }
    catch (recovery) { recoveryError = String(recovery.message ?? recovery); }
   }
   result = {group, status: error.sourceSeamMismatch ? 'needs-adaptation' : error.code === 'ENOENT' && !journalWritten ? 'unavailable' : 'failed', files: roles.map(role => targets[role].file), reason: String(error.message ?? error), ...(recoveryError ? {recoveryError} : {})};
  } finally {
   for (const {temp} of candidates) { try { await io.rm(temp, {force: true}); } catch {} }
  }
  results.push(result); report({component: 'pf2e-third-party-source-patches', ...result});
 }
 return {groups: results};
}
