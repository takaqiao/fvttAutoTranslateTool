import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {parseArguments, runStartup} from '../tools/automatic-source-patches/startup.mjs';

async function fixture(t) {
 const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'auto-source-startup-')));
 t.after(() => fs.rm(root, {recursive: true, force: true}));
 const dataPath = path.join(root, 'data'); await fs.mkdir(path.join(dataPath, 'Config'), {recursive: true});
 const mainPath = path.join(root, 'main.mjs'), containerPath = path.join(root, 'pm2/lib/ProcessContainerFork.js');
 await fs.mkdir(path.dirname(containerPath), {recursive: true});
 await fs.writeFile(mainPath, 'export {};'); await fs.writeFile(containerPath, 'module.exports = {};');
 const config = {schema: 1, dataPath, lockPath: path.join(dataPath, 'Config/.pf2e-iwr-startup.lock'), nodePath: await fs.realpath(process.execPath), nodeArgs: ['--max-old-space-size=4096'], mainPath, cwd: root, containerPath, pm2Root: path.dirname(path.dirname(containerPath)), originalArgs: [`--dataPath=${dataPath}`, '--port=30001'], originalEnvironmentFields: Object.fromEntries(['pm_exec_path', 'exec_interpreter', 'args', 'node_args', 'stop_exit_codes'].map(key => [key, {present: false}]))};
 config.originalEnvironmentFields.pm_exec_path = {present: true, value: mainPath};
 const configPath = path.join(root, 'startup-config.json'); await fs.writeFile(configPath, JSON.stringify(config));
 const args = {configPath, configSHA256: '0'.repeat(64), originalArgs: config.originalArgs};
 const runtime = {platform: 'linux', execPath: process.execPath, execArgv: config.nodeArgs, getuid: () => 0};
 return {root, config, args, runtime};
}
test('the actual startup calls automatic maintenance before entering the unchanged PM2 bootstrap', async t => {
 const f = await fixture(t), events = [], logs = [];
 const result = await runStartup(f.args, {runtime: f.runtime, guardFactory: ({config}) => async context => { assert.equal(config.dataPath, f.config.dataPath); assert.equal(context.systemDir, path.join(f.config.dataPath, 'Data/systems/pf2e')); events.push('idle'); return {idle: true, lock: {pid: 123}}; }, maintain: async ({dataPath, assertIdle}) => { assert.equal(dataPath, f.config.dataPath); await assertIdle(); events.push('maintained'); return {groups: [{group: 'native', status: 'patched'}]}; }, bootstrap: config => { assert.deepEqual(config.originalArgs, f.config.originalArgs); events.push('boot'); }, report: value => logs.push(value)});
 assert.equal(result.started, true); assert.deepEqual(events, ['idle', 'maintained', 'idle', 'boot']);
 assert.equal(logs[0].configDigestChanged, true);
});
test('a source seam mismatch still starts Foundry and reports the specific maintenance result', async t => {
 const f = await fixture(t); let booted = false; const logs = [];
 await runStartup(f.args, {runtime: f.runtime, guardFactory: () => async () => ({idle: true}), maintain: async () => ({groups: [{group: 'patreon', status: 'needs-adaptation', reason: 'patreon-v3 src/index.js: create item seam changed'}]}), bootstrap: () => {booted = true;}, report: value => logs.push(value)});
 assert.equal(booted, true); assert.match(logs[0].maintenance.groups[0].reason, /patreon-v3.*create item/);
});
test('positive occupancy after maintenance prevents entry into Foundry', async t => {
 const f = await fixture(t); let booted = false;
 await assert.rejects(runStartup(f.args, {runtime: f.runtime, guardFactory: () => async () => ({idle: false}), maintain: async () => ({groups: []}), bootstrap: () => {booted = true;}, report: () => {}}), /busy|idle/i);
 assert.equal(booted, false);
});
test('arguments selecting another installation are rejected before maintenance', async t => {
 const f = await fixture(t); let maintained = false;
 await assert.rejects(runStartup({...f.args, originalArgs: ['--dataPath=/another-installation']}, {runtime: f.runtime, maintain: async () => {maintained = true;}}), /argument/i);
 assert.equal(maintained, false);
});
test('changed Node arguments are rejected before maintenance', async t => {
 const f = await fixture(t);
 await assert.rejects(runStartup(f.args, {runtime: {...f.runtime, execArgv: []}}), /Node arguments/);
});
test('malformed config arguments are rejected and the original port arguments are preserved', () => {
 assert.throws(() => parseArguments(['--config', '/a', '--config-sha', '0'.repeat(64), '--bad']), /separator/i);
 const value = parseArguments(['--config', path.resolve('config.json'), '--config-sha', 'f'.repeat(64), '--', '--dataPath=/root/data', '--port=30001']);
 assert.deepEqual(value.originalArgs, ['--dataPath=/root/data', '--port=30001']);
});
