import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {createServiceGuard, dataPathArgument} from './startup-guard.mjs';
import {maintainSourcePatches} from './maintain.mjs';

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
export function parseArguments(argv) {
 assert.equal(argv[0], '--config', 'Explicit --config required');
 assert(path.isAbsolute(argv[1] ?? ''), 'Absolute config path required');
 assert.equal(argv[2], '--config-sha', 'Legacy config diagnostic required');
 assert(/^[a-f0-9]{64}$/.test(argv[3] ?? ''), 'Invalid legacy config digest');
 assert.equal(argv[4], '--', 'Original argument separator required');
 return {configPath: argv[1], configSHA256: argv[3], originalArgs: argv.slice(5)};
}

export function enterPM2Bootstrap(config) {
 for (const key of ['pm_exec_path', 'exec_interpreter', 'args', 'node_args', 'stop_exit_codes']) {
  const field = config.originalEnvironmentFields?.[key];
  assert(field && typeof field.present === 'boolean', `Original PM2 metadata required: ${key}`);
  if (field.present) process.env[key] = String(field.value); else delete process.env[key];
 }
 assert.equal(process.env.pm_exec_path, config.mainPath);
 process.chdir(config.cwd); process.argv = [config.nodePath, config.containerPath, ...config.originalArgs];
 // Preserve the original CommonJS main and PM2 process metadata.
 createRequire(import.meta.url)('node:module')._load(config.containerPath, null, true);
}
async function regular(filename, io, runtime) {
 const stat = await io.lstat(filename);
 assert(stat.isFile() && !stat.isSymbolicLink() && stat.nlink === 1, `Unsafe startup file: ${filename}`);
 assert.equal(await io.realpath(filename), filename, `Startup path changed: ${filename}`);
 if (process.platform === 'linux') {
  assert.equal(stat.uid, runtime.getuid(), `Startup file owner changed: ${filename}`);
  assert.equal(stat.mode & 0o022, 0, `Startup file is writable by another user: ${filename}`);
 }
 return stat;
}
export async function runStartup(args, {io = fs, runtime = process, guardFactory = createServiceGuard, maintain = maintainSourcePatches, bootstrap = enterPM2Bootstrap, report = value => console.log(JSON.stringify(value))} = {}) {
 assert.equal(runtime.platform, 'linux', 'Managed startup requires Linux flock and /proc');
 await regular(args.configPath, io, runtime);
 const bytes = await io.readFile(args.configPath), config = JSON.parse(bytes);
 assert.equal(config.schema, 1, 'Unsupported startup configuration');
 for (const key of ['dataPath', 'lockPath', 'nodePath', 'mainPath', 'cwd', 'containerPath', 'pm2Root']) assert(path.isAbsolute(config[key] ?? ''), `Explicit ${key} required`);
 assert.equal(await io.realpath(config.dataPath), config.dataPath, 'dataPath changed');
 assert.equal(await io.realpath(config.cwd), config.cwd, 'Foundry working directory changed');
 assert.equal(config.lockPath, path.join(config.dataPath, 'Config/.pf2e-iwr-startup.lock'));
 assert.equal(await io.realpath(path.dirname(config.lockPath)), path.dirname(config.lockPath));
 assert.equal(await io.realpath(runtime.execPath), await io.realpath(config.nodePath));
 assert.deepEqual(args.originalArgs, config.originalArgs, 'Original Foundry arguments changed');
 assert.deepEqual(runtime.execArgv, config.nodeArgs, 'Node arguments changed');
 assert.equal(await io.realpath(dataPathArgument(config.originalArgs)), config.dataPath, 'Foundry argument points to another installation');
 assert.equal(config.containerPath, path.join(config.pm2Root, 'lib/ProcessContainerFork.js'));
 await regular(config.mainPath, io, runtime); await regular(config.containerPath, io, runtime);
 const context = {dataPath: config.dataPath, dataDirectory: path.join(config.dataPath, 'Data'), systemDir: path.join(config.dataPath, 'Data/systems/pf2e'), bundlePath: path.join(config.dataPath, 'Data/systems/pf2e/pf2e.mjs'), manifestPath: path.join(config.dataPath, 'Data/systems/pf2e/system.json')};
 const guard = guardFactory({config, io}), assertIdle = async () => { const result = await guard(context); assert(result?.idle === true, 'Foundry installation is busy'); return result; };
 let maintenance;
 try { maintenance = await maintain({dataPath: config.dataPath, assertIdle, io, report}); }
 catch (error) { maintenance = {groups: [], status: 'failed', reason: String(error.message ?? error)}; }
 const idle = await assertIdle();
 report({component: 'pf2e-third-party-startup', status: 'ready', dataPath: config.dataPath, configDigestChanged: digest(bytes) !== args.configSHA256, maintenance, lock: idle.lock});
 await bootstrap(config);
 return {started: true, maintenance};
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
 try { await runStartup(parseArguments(process.argv.slice(2))); }
 catch (error) { console.error(JSON.stringify({component: 'pf2e-third-party-startup', status: 'refused', reason: String(error.message ?? error), exitCode: 78})); process.exitCode = 78; }
}
