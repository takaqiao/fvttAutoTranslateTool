// enable-worlds.mjs 和 reset-xp.mjs 共用的离线写库流程。两个工具只管读输入、定要写的行和数据模型校验。
//
// MODE=dry    库目录（除 LOCK）拷到 TMP_ROOT 下，在副本上走和 apply 相同的备份、写入、读回；运行检查不通过只记警告。
//             不打开生产库，结束时删掉副本。
// MODE=apply  先在副本上定好要写什么，没有要写的就结束。再查运行状态：写团世界时主服状态接口要读得到、正开着的不是它；
//             写建卡服时 pm2 里核心和控制器都 stopped、两个端口都拒连。有一项不满足就退出，什么都不写。然后打开生产库
//             占住 LOCK，确认内容和副本相同，整份备份到 BACKUP（必填，已存在就拒绝）并核哈希，再查一次运行状态，
//             逐库写入，最后从新拷的副本读回核对。
// 报告是 stdout 上的一份 JSON，ok 为 false 时退出码 1。环境变量见 README.md「部署」。
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const env = process.env;
export const FOUNDRY_APP = env.FOUNDRY_APP || '/root/foundryvtt';
export const MAIN_DATA = env.MAIN_DATA || '/root/fvtt14-data/Data';
export const CHAR_DATA = env.CHAR_DATA || '/root/fvtt14-char-data/Data';
export const CHAR_WORLD = env.CHAR_WORLD || 'pf2creation';
const MAIN_STATUS_URL = env.MAIN_STATUS_URL || 'http://127.0.0.1:30001/api/status';
const CHAR_STATUS_URL = env.CHAR_STATUS_URL || 'http://127.0.0.1:30002/api/status';
const CONTROL_URL = env.CONTROL_URL || 'http://127.0.0.1:8793/control/status';
const PM2_BIN = env.PM2_BIN || 'pm2';
const CHAR_PM2_NAME = env.CHAR_PM2_NAME || 'fvtt-v14-char';
const CONTROL_PM2_NAME = env.CONTROL_PM2_NAME || 'fvtt-v14-char-control';
const TMP_ROOT = env.TMP_ROOT || os.tmpdir();

export const WORLD_ID = /^[A-Za-z0-9_-]+$/;

export const splitList = value => (value ?? '').split(',').map(entry => entry.trim()).filter(Boolean);
const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');
export const canonical = value => JSON.stringify(value, (key, inner) => (
  inner && typeof inner === 'object' && !Array.isArray(inner)
    ? Object.fromEntries(Object.keys(inner).sort().map(name => [name, inner[name]]))
    : inner
));
export const isPlainObject = value => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
export const worldDir = world => path.join(world === CHAR_WORLD ? CHAR_DATA : MAIN_DATA, 'worlds', world);

// ---- 库目录：拷贝、哈希、打开、读出

function copyDb(src, dst) {
  fs.mkdirSync(dst, {recursive: true});
  for (const entry of fs.readdirSync(src, {withFileTypes: true})) {
    if (entry.isFile() && entry.name !== 'LOCK') fs.copyFileSync(path.join(src, entry.name), path.join(dst, entry.name));
  }
}

function hashDb(dir) {
  const hashes = {};
  for (const entry of fs.readdirSync(dir, {withFileTypes: true}).sort((a, b) => a.name.localeCompare(b.name))) {
    if (entry.isFile() && entry.name !== 'LOCK') hashes[entry.name] = sha256(fs.readFileSync(path.join(dir, entry.name)));
  }
  return hashes;
}

let ClassicLevel;

/** 用 Foundry 自带的 classic-level 打开；库被别的进程占着（LOCK）时抛错。 */
async function openDb(dir) {
  ClassicLevel ??= createRequire(path.join(FOUNDRY_APP, 'package.json'))('classic-level').ClassicLevel;
  const db = new ClassicLevel(dir, {keyEncoding: 'utf8', valueEncoding: 'utf8', createIfMissing: false});
  await db.open();
  return db;
}

async function readAll(db) {
  const rows = new Map();
  for await (const [key, value] of db.iterator()) rows.set(key, value);
  return rows;
}

/** 把库拷到 `dst` 再读出全部行（键 → 原样的 JSON 字符串），副本留着。 */
export async function readCopy(src, dst) {
  copyDb(src, dst);
  const db = await openDb(dst);
  try {
    return await readAll(db);
  } finally {
    await db.close();
  }
}

const sameRows = (a, b) => a.size === b.size && [...a].every(([key, value]) => b.get(key) === value);

// ---- 运行状态

function probe(url) {
  return new Promise(resolve => {
    let request;
    try {
      request = http.get(url, {timeout: 5000}, response => {
        let body = '';
        response.setEncoding('utf8');
        response.on('data', chunk => { body += chunk; });
        response.on('end', () => resolve({status: response.statusCode, body}));
      });
    } catch (error) {
      resolve({error: error.code ?? error.message});
      return;
    }
    request.on('error', error => resolve({error: error.code ?? error.message}));
    request.on('timeout', () => {
      request.destroy();
      resolve({error: 'timeout'});
    });
  });
}

async function mainStatus() {
  const result = await probe(MAIN_STATUS_URL);
  if (result.error) return {error: result.error};
  if (result.status !== 200) return {error: `HTTP ${result.status}`};
  let status;
  try {
    status = JSON.parse(result.body);
  } catch {
    return {error: '返回的不是 JSON'};
  }
  // 形状不对就当读不到：当成「没开」会去写一个可能正开着的世界
  if (!isPlainObject(status) || typeof status.active !== 'boolean') return {error: '返回的 active 不是布尔值'};
  if (status.active && typeof status.world !== 'string') return {error: '报告世界已开，但没有给出世界 id'};
  return {active: status.active, world: status.active ? status.world : null};
}

function pm2Status() {
  // 本机测试在 Windows 上跑，pm2 在那里是 .cmd，要经 shell 才能执行
  const result = process.platform === 'win32'
    ? spawnSync(`"${PM2_BIN}" jlist`, {shell: true, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024})
    : spawnSync(PM2_BIN, ['jlist'], {encoding: 'utf8', maxBuffer: 64 * 1024 * 1024});
  if (result.error || result.status !== 0) return {error: `${PM2_BIN} jlist 失败：${result.error?.message ?? result.stderr.trim()}`};
  let list;
  try {
    list = JSON.parse(result.stdout);
  } catch {
    return {error: `${PM2_BIN} jlist 的输出不是 JSON`};
  }
  const statusOf = name => list.find(proc => proc.name === name)?.pm2_env?.status ?? 'missing';
  return {core: statusOf(CHAR_PM2_NAME), controller: statusOf(CONTROL_PM2_NAME)};
}

async function charStatus() {
  const [foundry, control] = await Promise.all([probe(CHAR_STATUS_URL), probe(CONTROL_URL)]);
  const summary = result => (result.error ? {error: result.error} : {status: result.status});
  return {pm2: pm2Status(), foundry: summary(foundry), control: summary(control)};
}

// 控制器也要停着：它停着就没有东西能在写库时把核心拉起来
const charStopped = state => state.pm2.core === 'stopped' && state.pm2.controller === 'stopped'
  && state.foundry.error === 'ECONNREFUSED' && state.control.error === 'ECONNREFUSED';

async function runtimeState(worlds) {
  const state = {};
  if (worlds.some(world => world !== CHAR_WORLD)) state.main = await mainStatus();
  if (worlds.includes(CHAR_WORLD)) state.char = await charStatus();
  return state;
}

function runtimeProblems(state, worlds) {
  const problems = [];
  if (state.main?.error) problems.push(`主服状态接口 ${MAIN_STATUS_URL} 读不到（${state.main.error}），判断不了团世界开没开`);
  for (const world of worlds) {
    if (world !== CHAR_WORLD && state.main?.world === world) problems.push(`主服正开着 ${world}`);
  }
  if (state.char && !charStopped(state.char)) problems.push(`建卡服没有完全停下（核心、控制器都要 stopped，两个端口都要拒连）：${JSON.stringify(state.char)}`);
  return problems;
}

// ---- 数据模型

/** Foundry 服务器端的文档类（`BaseSetting`、`BaseFolder`、`BaseActor` 等）。 */
export async function loadDocuments() {
  // common/server.mjs 在导入时就会读这两个全局对象
  globalThis.CONFIG = {User: {}, Actor: {}, Item: {}, Folder: {}, Token: {}, Setting: {}};
  globalThis.game = {};
  const {documents} = await import(pathToFileURL(path.join(FOUNDRY_APP, 'common', 'server.mjs')).href);
  globalThis.game.model = {Folder: Object.fromEntries(documents.BaseFolder.metadata.coreTypes.map(type => [type, {}]))};
  return documents;
}

// ---- 写入与读回

/** 在打开的库上写一批行。新建的键必须不存在，改的键必须仍是读副本时的值；不满足就一行都不写。 */
async function writeRows(db, before, puts) {
  const problems = [];
  for (const put of puts) {
    if (put.old === undefined ? before.has(put.key) : before.get(put.key) !== put.old) problems.push(`写之前 ${put.key} 已经变了`);
  }
  if (!problems.length) await db.batch(puts.map(put => ({type: 'put', key: put.key, value: put.value})), {sync: true});
  return problems;
}

async function verifyWritten(dir, before, puts, scratch) {
  const after = await readCopy(dir, scratch);
  const expected = new Map(before);
  for (const put of puts) expected.set(put.key, put.value);
  const putKeys = new Set(puts.map(put => put.key));
  const problems = [];
  for (const [key, value] of expected) {
    if (after.get(key) !== value) problems.push(putKeys.has(key) ? `写入的行没有落盘：${key}` : `没碰的行变了：${key}`);
  }
  for (const key of after.keys()) {
    if (!expected.has(key)) problems.push(`多出一行：${key}`);
  }
  const untouched = [...before.keys()].filter(key => !putKeys.has(key)).length;
  return {rowsBefore: before.size, rowsAfter: after.size, put: puts.length, untouchedVerified: untouched, problems};
}

/** 把各库的 `puts` 写下去。apply 写生产库，dry 写 `readDir` 里的副本，其余步骤相同。 */
async function writeJobs(report, jobs, work, opened) {
  const {problems, warnings, inputs} = report;
  const pending = jobs.filter(job => job.puts.length);
  if (!pending.length) {
    report.result = '全部已是目标状态，没有写入';
    return;
  }
  const pendingWorlds = [...new Set(pending.map(job => job.world))];

  report.runtime = {before: await runtimeState(pendingWorlds)};
  const refusals = runtimeProblems(report.runtime.before, pendingWorlds);
  if (inputs.mode === 'apply') {
    problems.push(...refusals);
    if (problems.length) return;
  } else {
    warnings.push(...refusals.map(message => `apply 现在会拒绝：${message}`));
  }

  // 打开即占住库的 LOCK：此后 Foundry 开不了这个世界，库也不会再变
  for (const job of pending) {
    job.target = inputs.mode === 'apply' ? job.srcDir : job.readDir;
    try {
      job.db = await openDb(job.target);
    } catch (error) {
      problems.push(`打不开 ${job.target}（多半有程序正开着这个世界）：${error.cause?.message ?? error.message}`);
      return;
    }
    opened.push(job.db);
    job.before = await readAll(job.db);
    if (!sameRows(job.before, job.rows)) {
      problems.push(`${job.target} 在读副本之后变了`);
      return;
    }
  }

  // 库已经被占住才拷：打开时 LevelDB 可能整理过日志文件，备份的文件和打开前不一定逐字节相同，但内容就是写前的
  const backupRoot = inputs.mode === 'apply' ? inputs.backup : path.join(work, 'backup');
  report.backup = {dir: backupRoot, dbs: {}};
  for (const job of pending) {
    const dst = path.join(backupRoot, job.world, job.name);
    copyDb(job.target, dst);
    const hashes = hashDb(job.target);
    if (canonical(hashes) !== canonical(hashDb(dst))) {
      problems.push(`备份 ${dst} 和原库的文件哈希对不上`);
      return;
    }
    report.backup.dbs[`${job.world}/${job.name}`] = Object.keys(hashes).length;
  }
  if (inputs.mode === 'apply') {
    const again = await runtimeState(pendingWorlds);
    problems.push(...runtimeProblems(again, pendingWorlds).map(message => `备份期间运行状态变了：${message}`));
    if (problems.length) return;
  }

  // 一个库一个 batch；后面的库出错时，报告里看得出前面哪些已经写了
  report.written = {};
  for (const job of pending) {
    const writeProblems = await writeRows(job.db, job.before, job.puts);
    problems.push(...writeProblems);
    if (writeProblems.length) return;
    (report.written[job.world] ??= {})[job.name] = {put: job.puts.length};
  }
  for (const job of pending) await job.db.close();
  for (const job of pending) {
    const result = await verifyWritten(job.target, job.before, job.puts, path.join(work, 'verify', job.world, job.name));
    Object.assign(report.written[job.world][job.name], result);
    problems.push(...result.problems.map(message => `${job.world}/${job.name} 读回：${message}`));
  }
  if (inputs.mode === 'apply') {
    report.runtime.after = await runtimeState(pendingWorlds);
    warnings.push(...runtimeProblems(report.runtime.after, pendingWorlds).map(message => `写完后：${message}`));
  }
}

// ---- 入口

/** 两个工具相同的输入：`MODE`，以及 apply 必填、不能已存在的 `BACKUP`。 */
export function readModeAndBackup(problems) {
  const mode = env.MODE;
  if (mode !== 'dry' && mode !== 'apply') problems.push('MODE 只能是 dry 或 apply');
  const backup = env.BACKUP ? path.resolve(env.BACKUP) : null;
  if (mode === 'apply' && !backup) problems.push('MODE=apply 要给 BACKUP');
  if (backup && fs.existsSync(backup)) problems.push(`BACKUP 已存在：${backup}`);
  return {mode, backup};
}

/**
 * 跑一个工具并在 stdout 打报告。`readInputs(problems)` 返回的输入里要有 `mode`、`backup`。
 * `plan(report, work)` 读 `work` 下的副本、做完校验，返回各库的 `{world, name, srcDir, readDir, rows, puts}`：
 * `rows` 是 `readCopy` 读出的副本内容，`puts` 每项 `{key, value, old}`，`old` 是改之前的原样字符串，新建的行不写。
 * `plan` 记了 problem 就一行都不写。
 */
export async function runTool({tool, readInputs, plan}) {
  const report = {tool, mode: env.MODE ?? null, ok: false, problems: [], warnings: []};
  const inputs = readInputs(report.problems);
  report.inputs = {...inputs, mainData: MAIN_DATA, charData: CHAR_DATA, charWorld: CHAR_WORLD};
  if (!report.problems.length) {
    const work = path.join(TMP_ROOT, `charsync-${inputs.mode}-${process.pid}`);
    fs.rmSync(work, {recursive: true, force: true});
    fs.mkdirSync(work, {recursive: true});
    const opened = [];
    try {
      const jobs = await plan(report, work);
      if (!report.problems.length) await writeJobs(report, jobs, work, opened);
    } catch (error) {
      report.problems.push(`意外错误：${error.stack ?? error}`);
    } finally {
      for (const db of opened) {
        try {
          await db.close();
        } catch (error) {
          report.problems.push(`关库失败：${error.message}`);
        }
      }
      fs.rmSync(work, {recursive: true, force: true});
    }
  }
  report.ok = report.problems.length === 0;
  console.log(JSON.stringify(report, null, 2));
  process.exitCode = report.ok ? 0 : 1;
}
