import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {maintainSourcePatches} from '../tools/automatic-source-patches/maintain.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const files = {pf2e: 'Data/systems/pf2e/pf2e.mjs', toolbelt: 'Data/modules/pf2e-toolbelt/scripts/main.js', patreon: 'Data/modules/patreon-v3/src/index.js'};
const original = {pf2e: 'export const damage = 1;\n', toolbelt: 'export const heal = 1;\n', patreon: 'export const time = 1;\n'};
// patched.toolbelt is what an earlier release wrote there; no builder produces it now.
const patched = {pf2e: 'export const damage = 2;\n', toolbelt: 'export const heal = 2;\n', patreon: 'export const time = 2;\n'};
const builders = {
 buildNativeBridge({source}) { return {status: source.equals(Buffer.from(patched.pf2e)) ? 'unchanged' : 'patch', buffer: Buffer.from(patched.pf2e)}; },
 buildPatreonSource({source}) { return {status: source.equals(Buffer.from(patched.patreon)) ? 'unchanged' : 'patch', buffer: Buffer.from(patched.patreon)}; }
};
async function fixture(t) {
 const dataPath = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'source-maintenance-')));
 t.after(() => fs.rm(dataPath, {recursive: true, force: true}));
 for (const [role, relative] of Object.entries(files)) {
  await fs.mkdir(path.dirname(path.join(dataPath, relative)), {recursive: true});
  await fs.writeFile(path.join(dataPath, relative), original[role]);
 }
 for (const [relative, id] of [['Data/systems/pf2e/system.json', 'pf2e'], ['Data/modules/pf2e-toolbelt/module.json', 'pf2e-toolbelt'], ['Data/modules/patreon-v3/module.json', 'patreon-v3']]) {
  await fs.writeFile(path.join(dataPath, relative), JSON.stringify({id, version: 'unlisted-future-version'}));
 }
 const opts = {dataPath, builders, assertIdle: async () => ({idle: true}), report: () => {}};
 const file = role => path.join(dataPath, files[role]);
 const modified = async role => { const {ino, size, mode, mtimeMs, ctimeMs} = await fs.stat(file(role)); return {ino, size, mode, mtimeMs, ctimeMs}; };
 return {dataPath, opts, file, modified, backup: path.join(dataPath, 'Backups/pf2e-third-party-patches')};
}
const status = (result, name) => result.groups.find(group => group.group === name)?.status;

for (const [state, toolbelt] of [['an original', original.toolbelt], ['an already patched', patched.toolbelt]]) test(`automatically installs the PF2e bridge and independent Patreon without a version approval, leaving ${state} Toolbelt alone`, async t => {
 const f = await fixture(t), journals = [];
 await fs.writeFile(f.file('toolbelt'), toolbelt);
 const before = await f.modified('toolbelt');
 // A pending record is unlinked once its group is installed, so it is read as it is published.
 const io = new Proxy(fs, {get(target, key) {
  if (key !== 'link') return target[key];
  return async (from, to) => { if (to.endsWith('.pending.json')) journals.push(JSON.parse(await target.readFile(from, 'utf8'))); return target.link(from, to); };
 }});
 const result = await maintainSourcePatches({...f.opts, io});
 assert.equal(status(result, 'native'), 'patched'); assert.equal(status(result, 'patreon'), 'patched');
 for (const role of ['pf2e', 'patreon']) assert.equal(await fs.readFile(f.file(role), 'utf8'), patched[role]);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), toolbelt); assert.deepEqual(await f.modified('toolbelt'), before);
 assert.equal(await fs.readFile(path.join(f.backup, `pf2e-${sha(original.pf2e)}.bin`), 'utf8'), original.pf2e);
 assert.deepEqual((await fs.readdir(f.backup)).sort(), [`patreon-${sha(original.patreon)}.bin`, `pf2e-${sha(original.pf2e)}.bin`]);
 assert.deepEqual(journals.map(journal => [journal.group, journal.entries.map(entry => entry.role)]), [['native', ['pf2e']], ['patreon', ['patreon']]]);
 assert.deepEqual(result.groups[0].files.map(entry => entry.file), [files.pf2e]);
});
test('already patched groups perform no source, backup, or journal writes', async t => {
 const f = await fixture(t);
 for (const role of Object.keys(files)) await fs.writeFile(f.file(role), patched[role]);
 const before = await Promise.all(Object.keys(files).map(f.modified));
 const result = await maintainSourcePatches(f.opts);
 assert.equal(status(result, 'native'), 'unchanged'); assert.equal(status(result, 'patreon'), 'unchanged');
 assert.deepEqual(await Promise.all(Object.keys(files).map(f.modified)), before);
 await assert.rejects(fs.stat(f.backup), {code: 'ENOENT'});
});
test('a mismatched PF2e seam leaves both native files untouched while Patreon recovers', async t => {
 const f = await fixture(t);
 const result = await maintainSourcePatches({...f.opts, builders: {...builders, buildNativeBridge() { throw Error('pf2e pf2e.mjs: applyDamage seam is ambiguous'); }}});
 assert.equal(status(result, 'native'), 'needs-adaptation'); assert.equal(status(result, 'patreon'), 'patched');
 assert.match(result.groups[0].reason, /pf2e\.mjs.*applyDamage/);
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), original.pf2e);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), original.toolbelt);
});
test('a failed PF2e replacement leaves the original in place and no pending record', async t => {
 const f = await fixture(t); let failed = false;
 const io = new Proxy(fs, {get(target, key) {
  if (key !== 'rename') return target[key];
  return async (from, to) => { if (to === f.file('pf2e') && !failed) {failed = true; throw Error('disk write refused');} return target.rename(from, to); };
 }});
 const result = await maintainSourcePatches({...f.opts, io});
 assert.equal(status(result, 'native'), 'failed'); assert.equal(status(result, 'patreon'), 'patched');
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), original.pf2e);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), original.toolbelt);
 await assert.rejects(fs.stat(path.join(f.backup, 'native.pending.json')), {code: 'ENOENT'});
});
async function pending(f, states, unknown = false) {
 await fs.mkdir(f.backup, {recursive: true});
 const entries = [];
 for (const role of ['pf2e', 'toolbelt']) {
  await fs.writeFile(path.join(f.backup, `${role}-${sha(original[role])}.bin`), original[role]);
  entries.push({role, before: sha(original[role]), after: sha(patched[role]), mode: 0o644});
  await fs.writeFile(f.file(role), states[role] === 'after' ? patched[role] : original[role]);
 }
 if (unknown) await fs.writeFile(f.file('toolbelt'), 'export const aLaterUpdate = true;\n');
 await fs.writeFile(path.join(f.backup, 'native.pending.json'), JSON.stringify({schema: 1, group: 'native', entries}));
}
test('interrupted mixed pair restores originals before rebuilding at the next idle start', async t => {
 const f = await fixture(t); await pending(f, {pf2e: 'after', toolbelt: 'before'});
 let sawOriginal = false;
 const b = {...builders, buildNativeBridge(input) { sawOriginal = input.source.toString() === original.pf2e; return builders.buildNativeBridge(input); }};
 const result = await maintainSourcePatches({...f.opts, builders: b});
 assert.equal(sawOriginal, true); assert.equal(status(result, 'native'), 'patched');
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), patched.pf2e);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), original.toolbelt);
});
test('a completely installed interrupted pair is finalized without reapplying it', async t => {
 const f = await fixture(t); await pending(f, {pf2e: 'after', toolbelt: 'after'});
 const result = await maintainSourcePatches(f.opts);
 assert.equal(status(result, 'native'), 'unchanged');
 await assert.rejects(fs.stat(path.join(f.backup, 'native.pending.json')), {code: 'ENOENT'});
 for (const role of ['pf2e', 'toolbelt']) assert.equal(await fs.readFile(f.file(role), 'utf8'), patched[role]);
});
test('pending recovery never overwrites an unknown later edit or restores only half a pair', async t => {
 const f = await fixture(t); await pending(f, {pf2e: 'after', toolbelt: 'before'}, true);
 const result = await maintainSourcePatches(f.opts);
 assert.equal(status(result, 'native'), 'failed'); assert.match(result.groups[0].reason, /unknown.*toolbelt/i);
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), patched.pf2e);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), 'export const aLaterUpdate = true;\n');
 assert.equal(status(result, 'patreon'), 'patched');
});
test('a corrupted original backup prevents all pending restoration', async t => {
 const f = await fixture(t); await pending(f, {pf2e: 'after', toolbelt: 'before'});
 await fs.writeFile(path.join(f.backup, `toolbelt-${sha(original.toolbelt)}.bin`), 'corrupt');
 const result = await maintainSourcePatches(f.opts);
 assert.equal(status(result, 'native'), 'failed'); assert.match(result.groups[0].reason, /backup/i);
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), patched.pf2e);
});
test('generated syntax errors prevent any native replacement', async t => {
 const f = await fixture(t);
 const result = await maintainSourcePatches({...f.opts, builders: {...builders, buildNativeBridge() { return {status: 'patch', buffer: Buffer.from('export const = ;')}; }}});
 assert.equal(status(result, 'native'), 'failed'); assert.match(result.groups[0].reason, /syntax/i);
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), original.pf2e);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), original.toolbelt);
});
for (const [changed, kept] of [['pf2e', 'toolbelt'], ['toolbelt', 'pf2e']]) test(`a ${changed} source change between preparation and write is preserved`, async t => {
 const f = await fixture(t); let calls = 0;
 const assertIdle = async () => { if (++calls === 2) await fs.writeFile(f.file(changed), 'export const newer = true;\n'); return {idle: true}; };
 const result = await maintainSourcePatches({...f.opts, assertIdle});
 assert.equal(status(result, 'native'), 'failed'); assert.match(result.groups[0].reason, /changed/i);
 assert.equal(await fs.readFile(f.file(changed), 'utf8'), 'export const newer = true;\n');
 assert.equal(await fs.readFile(f.file(kept), 'utf8'), original[kept]);
});
test('a manifest update during preparation prevents the group write', async t => {
 const f = await fixture(t); let calls = 0;
 const assertIdle = async () => { if (++calls === 2) await fs.writeFile(path.join(f.dataPath, 'Data/systems/pf2e/system.json'), JSON.stringify({id: 'pf2e', version: 'newer'})); return {idle: true}; };
 const result = await maintainSourcePatches({...f.opts, assertIdle});
 assert.equal(status(result, 'native'), 'failed');
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), original.pf2e);
});
test('maintenance requires a real idle guard and refuses positive occupancy', async t => {
 const f = await fixture(t);
 await assert.rejects(maintainSourcePatches({...f.opts, assertIdle: undefined}), /idle.*guard/i);
 await assert.rejects(maintainSourcePatches({...f.opts, assertIdle: async () => ({idle: false})}), /busy|idle/i);
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), original.pf2e);
});
for (const role of ['pf2e', 'toolbelt']) test(`a hard-linked ${role} source leaves the native group unmodified`, async t => {
 const f = await fixture(t); await fs.link(f.file(role), path.join(f.dataPath, 'another-source.js'));
 const result = await maintainSourcePatches(f.opts);
 assert.equal(status(result, 'native'), 'failed');
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), original.pf2e);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), original.toolbelt);
});
test('pending journals cannot select a fourth source file', async t => {
 const f = await fixture(t); await pending(f, {pf2e: 'after', toolbelt: 'before'});
 const journal = path.join(f.backup, 'native.pending.json'), value = JSON.parse(await fs.readFile(journal));
 value.entries[1].role = '../outside'; await fs.writeFile(journal, JSON.stringify(value));
 const result = await maintainSourcePatches(f.opts);
 assert.equal(status(result, 'native'), 'failed'); assert.match(result.groups[0].reason, /role|journal/i);
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), patched.pf2e);
});
test('a redirected Backups directory cannot receive original source backups', async t => {
 const f = await fixture(t), outside = await fs.mkdtemp(path.join(os.tmpdir(), 'outside-backups-'));
 t.after(() => fs.rm(outside, {recursive: true, force: true}));
 await fs.symlink(outside, path.join(f.dataPath, 'Backups'), process.platform === 'win32' ? 'junction' : 'dir');
 const result = await maintainSourcePatches(f.opts);
 assert.equal(status(result, 'native'), 'failed');
 assert.deepEqual(await fs.readdir(outside), []);
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), original.pf2e);
});
for (const artifact of ['backup', 'journal']) test(`a partial ${artifact} write can recover on the next normal startup`, async t => {
 const f = await fixture(t); let failed = false;
 const io = new Proxy(fs, {get(target, key) {
  if (key !== 'open') return target[key];
  return async (filename, ...args) => {
   const handle = await target.open(filename, ...args);
   const selected = artifact === 'backup' ? /pf2e-[a-f0-9]+\.bin(?:\.publishing)?$/.test(String(filename)) : /native\.pending\.json(?:\.publishing)?$/.test(String(filename));
   if (!selected || failed) return handle;
   return new Proxy(handle, {get(object, property) {
    if (property === 'writeFile') return async bytes => {failed = true; await object.write(Buffer.from(bytes).subarray(0, 7)); throw Object.assign(Error('partial disk write'), {code: 'EIO'});};
    const value = object[property]; return typeof value === 'function' ? value.bind(object) : value;
   }});
  };
 }});
 assert.equal(status(await maintainSourcePatches({...f.opts, io}), 'native'), 'failed');
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), original.pf2e);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), original.toolbelt);
 assert.equal(status(await maintainSourcePatches(f.opts), 'native'), 'patched');
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), patched.pf2e);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), original.toolbelt);
});
test('interruption after atomic publication releases only the matching temporary hard links', async t => {
 const f = await fixture(t); await pending(f, {pf2e: 'after', toolbelt: 'before'});
 const journal = path.join(f.backup, 'native.pending.json'), backup = path.join(f.backup, `pf2e-${sha(original.pf2e)}.bin`);
 for (const filename of [journal, backup]) await fs.link(filename, filename + '.publishing');
 const result = await maintainSourcePatches(f.opts);
 assert.equal(status(result, 'native'), 'patched');
 for (const filename of [journal, backup]) await assert.rejects(fs.stat(filename + '.publishing'), {code: 'ENOENT'});
 assert.equal((await fs.stat(backup)).nlink, 1);
});
test('an unknown journal hard link is refused before creating another backup', async t => {
 const f = await fixture(t); await pending(f, {pf2e: 'after', toolbelt: 'before'});
 await fs.link(path.join(f.backup, 'native.pending.json'), path.join(f.dataPath, 'other-journal.json'));
 const nativeBackups = async () => (await fs.readdir(f.backup)).filter(name => /^(pf2e|toolbelt)-.*\.bin$/.test(name)).sort();
 const backups = await nativeBackups();
 const result = await maintainSourcePatches(f.opts);
 assert.equal(status(result, 'native'), 'failed'); assert.match(result.groups[0].reason, /Unknown publication links/);
 assert.deepEqual(await nativeBackups(), backups);
 assert.equal(status(result, 'patreon'), 'patched');
 assert.equal(await fs.readFile(f.file('pf2e'), 'utf8'), patched.pf2e);
 assert.equal(await fs.readFile(f.file('toolbelt'), 'utf8'), original.toolbelt);
});
