// 离线把指定角色卡的经验值归 0：只改 actor 行里的 `system.details.xp.value`，整行其余字节不动，`_stats` 也不动。
// 经验值已经是 0 的卡不写。改动行的 actor 顶层字段过 Foundry 自带数据模型的严格校验。
// dry / apply、运行检查、备份和读回见 offline-db.mjs。
import fs from 'node:fs';
import path from 'node:path';
import { WORLD_ID, isPlainObject, loadDocuments, readCopy, readModeAndBackup, runTool, splitList, worldDir } from './offline-db.mjs';

const FOUNDRY_ID = /^[A-Za-z0-9]{16}$/;
// 改动行要过严格校验的 actor 顶层字段；system 归 pf2e 的数据模型管，这里加载不到
const ACTOR_FIELDS = ['_id', 'name', 'img', 'folder', 'sort', 'ownership', '_stats'];

/** 改动行的 actor 顶层字段过严格校验。先拿一对样例确认校验真的在拦。 */
async function checkModels(jobs, problems) {
  const {BaseActor} = await loadDocuments();
  const {fields} = BaseActor.schema;
  const failure = (field, value) => {
    try {
      return field.validate(value, {strict: true}) ?? null;
    } catch (error) {
      return error;
    }
  };
  if (!failure(fields._id, 'short') || failure(fields._id, 'aaaaaaaaaaaaaaaa')) {
    problems.push('数据模型校验没有按预期工作：合法样例被拒或不合法样例被接受');
    return;
  }
  for (const job of jobs) {
    for (const put of job.puts) {
      for (const name of ACTOR_FIELDS) {
        const failed = failure(fields[name], put.doc[name]);
        if (failed) problems.push(`${job.world} ${put.key} 的 ${name} 过不了数据模型校验：${failed.asError?.().message ?? failed.message ?? String(failed)}`);
      }
    }
  }
}

/** 经验值不是 0 的卡改成 0；行要能原样重新序列化，改完把经验值放回去必须和原行逐字节相同。 */
function planActors(rows, actorIds, world, problems) {
  const puts = [];
  const actors = [];
  for (const id of actorIds) {
    const key = `!actors!${id}`;
    const raw = rows.get(key);
    if (raw === undefined) {
      problems.push(`${world} 里没有 actor ${id}`);
      continue;
    }
    const doc = JSON.parse(raw);
    const xp = doc.system?.details?.xp;
    if (!isPlainObject(xp) || typeof xp.value !== 'number') {
      problems.push(`${id}（${doc.name}）没有数字的 system.details.xp.value`);
      continue;
    }
    const entry = {id, name: doc.name, type: doc.type, level: doc.system.details.level?.value ?? null, xpBefore: xp.value, xpAfter: 0};
    actors.push(entry);
    if (xp.value === 0) {
      entry.status = 'already';
      continue;
    }
    const next = structuredClone(doc);
    next.system.details.xp.value = 0;
    const value = JSON.stringify(next);
    const restored = JSON.parse(value);
    restored.system.details.xp.value = xp.value;
    if (JSON.stringify(doc) !== raw || JSON.stringify(restored) !== raw) {
      problems.push(`${id}（${doc.name}）的行重新序列化后和原来不同，不改`);
      continue;
    }
    entry.status = 'reset';
    puts.push({key, old: raw, doc: next, value});
  }
  return {puts, actors};
}

function readInputs(problems) {
  const {mode, backup} = readModeAndBackup(problems);
  const world = process.env.WORLD ?? '';
  if (!WORLD_ID.test(world)) problems.push('WORLD 要是一个世界 id');
  const actors = splitList(process.env.ACTORS);
  if (!actors.length) problems.push('ACTORS 没给（逗号分隔的 actor id）');
  for (const id of actors) {
    if (!FOUNDRY_ID.test(id)) problems.push(`不是 Foundry 的文档 id：${id}`);
  }
  if (new Set(actors).size !== actors.length) problems.push('ACTORS 里有重复的 id');
  return {mode, world, actors, backup};
}

async function planJobs(report, work) {
  const {problems, inputs} = report;
  const dir = worldDir(inputs.world);
  if (!fs.existsSync(path.join(dir, 'world.json'))) {
    problems.push(`没有这个世界：${path.join(dir, 'world.json')}`);
    return [];
  }
  const srcDir = path.join(dir, 'data', 'actors');
  const readDir = path.join(work, 'read', inputs.world, 'actors');
  const rows = await readCopy(srcDir, readDir);
  const {puts, actors} = planActors(rows, inputs.actors, inputs.world, problems);
  report.actors = actors;
  return [{world: inputs.world, name: 'actors', srcDir, readDir, rows, puts}];
}

await runTool({
  tool: 'reset-xp',
  readInputs,
  async plan(report, work) {
    const jobs = await planJobs(report, work);
    if (!report.problems.length) await checkModels(jobs, report.problems);
    return jobs;
  },
});
