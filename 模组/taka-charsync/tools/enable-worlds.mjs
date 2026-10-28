// 离线给世界启用本模组：写 `core.moduleConfiguration`（加 `taka-charsync`，建卡服另加 CHAR_EXTRA_MODULES）
// 和世界设置 `taka-charsync.publishDir`；建卡服另建「已提交」「已撤回」「草稿备份」三个带 `role` 记号的角色文件夹。
// 新行和改动行先过 Foundry 自带数据模型的严格校验，写进库的是校验后的 toObject()。已是目标状态的项不写；
// publishDir 已有不同的值时报 problem，不覆盖。dry / apply、运行检查、备份和读回见 offline-db.mjs。
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {
  CHAR_DATA, CHAR_WORLD, MAIN_DATA, WORLD_ID, canonical, isPlainObject, loadDocuments, readCopy, readModeAndBackup, runTool,
  splitList, worldDir,
} from './offline-db.mjs';

const MODULE_ID = 'taka-charsync';
const MODULES_KEY = 'core.moduleConfiguration';
const PUBLISH_DIR_KEY = `${MODULE_ID}.publishDir`;
const ROLE_FOLDERS = [['submitted', '已提交'], ['withdrawn', '已撤回'], ['backup', '草稿备份']];
const FOLDER_SORT_STEP = 100000;
const GM_ROLE = 4;

const PUBLISH_DIR = /^[A-Za-z0-9_-]{12,}$/;
const ID_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

function newId(taken) {
  for (;;) {
    let id = '';
    for (let i = 0; i < 16; i++) id += ID_CHARS[crypto.randomInt(ID_CHARS.length)];
    if (!taken.has(id)) return id;
  }
}

// ---- 数据模型

const accepts = (Doc, data) => {
  try {
    new Doc(structuredClone(data), {strict: true});
    return true;
  } catch {
    return false;
  }
};

/**
 * 每行过严格校验，写进库的是校验后的 toObject()；`kept` 里的字段被数据模型改动也算失败。
 * 先拿一对只差一个字段的样例确认校验真的在拦。
 */
async function checkModels(jobs, problems) {
  const {BaseSetting, BaseFolder} = await loadDocuments();
  const stats = {
    coreVersion: '14', systemId: 'pf2e', systemVersion: '1', createdTime: 0, modifiedTime: 0, lastModifiedBy: null,
    compendiumSource: null, duplicateSource: null, exportSource: null,
  };
  const folder = {_id: 'aaaaaaaaaaaaaaaa', name: 'x', type: 'Actor', description: '', folder: null, sorting: 'a', sort: 0, color: null, flags: {}, _stats: stats};
  const setting = {_id: 'aaaaaaaaaaaaaaaa', key: 'a.b', value: '1', user: null, _stats: stats};
  if (!accepts(BaseFolder, folder) || accepts(BaseFolder, {...folder, type: 'Nope'})
    || !accepts(BaseSetting, setting) || accepts(BaseSetting, {...setting, key: 'nodot'})) {
    problems.push('数据模型校验没有按预期工作：合法样例被拒或不合法样例被接受');
    return;
  }
  const models = {settings: [BaseSetting, ['key', 'value', 'user']], folders: [BaseFolder, ['name', 'type', 'folder', 'flags']]};
  for (const job of jobs) {
    const [Doc, kept] = models[job.name];
    for (const put of job.puts) {
      try {
        const clean = new Doc(structuredClone(put.doc), {strict: true}).toObject();
        const changed = kept.filter(field => canonical(clean[field]) !== canonical(put.doc[field]));
        if (changed.length) problems.push(`${job.world} ${put.key}：数据模型改了 ${changed.join('、')}`);
        put.value = JSON.stringify(clean);
      } catch (error) {
        problems.push(`${job.world} ${put.key} 过不了数据模型校验：${error.message}`);
      }
    }
  }
}

// ---- 每个世界要写什么

function settingRows(rows, problems, world) {
  const parsed = [];
  for (const [key, raw] of rows) {
    if (!key.startsWith('!settings!')) continue;
    try {
      parsed.push({key, raw, doc: JSON.parse(raw)});
    } catch {
      problems.push(`${world} 的设置行不是 JSON：${key}`);
    }
  }
  return parsed;
}

/** 只认世界设置（`user` 为空）；同一个键有两行时报 problem。 */
function worldSetting(settings, settingKey, problems, world) {
  const found = settings.filter(row => row.doc.key === settingKey && !row.doc.user);
  if (found.length > 1) problems.push(`${world} 的 ${settingKey} 有 ${found.length} 行世界设置`);
  return found[0] ?? null;
}

function parseValue(row, problems, world) {
  try {
    return {value: JSON.parse(row.doc.value)};
  } catch {
    problems.push(`${world} 的 ${row.doc.key} 的值不是 JSON`);
    return null;
  }
}

/** role 4 的用户里 id 最小的那个；没有时为 null。 */
function gmUserOf(userRows) {
  const gms = [...userRows].filter(([key]) => key.startsWith('!users!'))
    .map(([, raw]) => JSON.parse(raw))
    .filter(user => user.role === GM_ROLE)
    .sort((a, b) => a._id.localeCompare(b._id));
  return gms.length ? {id: gms[0]._id, name: gms[0].name, count: gms.length} : null;
}

function planWorld({world, worldJson, rows, gm, inputs, now, problems, warnings}) {
  const isChar = world === CHAR_WORLD;
  const summary = {gmUser: gm.id};
  const settingPuts = [];
  const folderPuts = [];
  const created = {createdTime: now, modifiedTime: now, lastModifiedBy: gm.id};
  const newStats = {
    coreVersion: worldJson.coreVersion, systemId: worldJson.system, systemVersion: worldJson.systemVersion, ...created,
    compendiumSource: null, duplicateSource: null, exportSource: null,
  };
  const touch = doc => ({...doc, _stats: {...doc._stats, modifiedTime: now, lastModifiedBy: gm.id}});

  const settings = settingRows(rows.settings, problems, world);
  const settingIds = new Set(settings.map(row => row.doc._id));
  const createSetting = (key, value) => {
    const id = newId(settingIds);
    settingIds.add(id);
    settingPuts.push({key: `!settings!${id}`, doc: {_id: id, key, value: JSON.stringify(value), user: null, _stats: newStats}});
  };

  // 启用项
  const modules = isChar ? [MODULE_ID, ...inputs.extraModules] : [MODULE_ID];
  const modulesRow = worldSetting(settings, MODULES_KEY, problems, world);
  const current = modulesRow ? parseValue(modulesRow, problems, world)?.value : {};
  if (current !== undefined && !isPlainObject(current)) {
    problems.push(`${world} 的 ${MODULES_KEY} 不是对象`);
  } else if (current) {
    const enable = modules.filter(id => current[id] !== true);
    summary.moduleConfiguration = {status: enable.length ? (modulesRow ? 'update' : 'create') : 'already', enable, already: modules.filter(id => !enable.includes(id))};
    const next = {...current, ...Object.fromEntries(enable.map(id => [id, true]))};
    if (enable.length && modulesRow) settingPuts.push({key: modulesRow.key, old: modulesRow.raw, doc: touch({...modulesRow.doc, value: JSON.stringify(next)})});
    else if (enable.length) createSetting(MODULES_KEY, next);
  }

  // 发布目录名
  const dirRow = worldSetting(settings, PUBLISH_DIR_KEY, problems, world);
  const dirValue = dirRow ? parseValue(dirRow, problems, world) : {value: null};
  if (dirValue) {
    const existing = dirValue.value;
    if (existing === inputs.publishDir) {
      summary.publishDir = {status: 'already'};
    } else if (existing !== null && existing !== '') {
      problems.push(`${world} 的 ${PUBLISH_DIR_KEY} 已经是另一个值，不覆盖`);
    } else if (dirRow) {
      summary.publishDir = {status: 'update'};
      settingPuts.push({key: dirRow.key, old: dirRow.raw, doc: touch({...dirRow.doc, value: JSON.stringify(inputs.publishDir)})});
    } else {
      summary.publishDir = {status: 'create'};
      createSetting(PUBLISH_DIR_KEY, inputs.publishDir);
    }
  }

  // 建卡服的三个文件夹
  if (isChar) {
    const folders = [...rows.folders].filter(([key]) => key.startsWith('!folders!')).map(([, raw]) => JSON.parse(raw));
    const actorFolders = folders.filter(folder => folder.type === 'Actor');
    const folderIds = new Set(folders.map(folder => folder._id));
    let sort = Math.max(0, ...actorFolders.filter(folder => !folder.folder).map(folder => folder.sort ?? 0));
    summary.folders = {};
    for (const [role, name] of ROLE_FOLDERS) {
      const existing = actorFolders.filter(folder => folder.flags?.[MODULE_ID]?.role === role);
      if (existing.length > 1) warnings.push(`${world} 有 ${existing.length} 个 role 为 ${role} 的角色文件夹，模组只用第一个`);
      if (existing.length) {
        summary.folders[role] = {status: 'already', id: existing[0]._id, name: existing[0].name};
        continue;
      }
      const id = newId(folderIds);
      folderIds.add(id);
      sort += FOLDER_SORT_STEP;
      summary.folders[role] = {status: 'create', id, name};
      folderPuts.push({
        key: `!folders!${id}`,
        doc: {
          _id: id, name, type: 'Actor', description: '', folder: null, sorting: 'a', sort, color: null,
          flags: {[MODULE_ID]: {role}}, _stats: newStats,
        },
      });
    }
  }

  return {summary, puts: {settings: settingPuts, folders: folderPuts}};
}

// ---- 输入与计划

function readInputs(problems) {
  const {mode, backup} = readModeAndBackup(problems);
  const worlds = splitList(process.env.WORLDS);
  if (!worlds.length) problems.push('WORLDS 没给（逗号分隔的世界 id）');
  for (const world of worlds) {
    if (!WORLD_ID.test(world)) problems.push(`不是合法的世界 id：${world}`);
  }
  if (new Set(worlds).size !== worlds.length) problems.push('WORLDS 里有重复的世界');
  const publishDir = process.env.PUBLISH_DIR ?? '';
  if (!PUBLISH_DIR.test(publishDir)) problems.push('PUBLISH_DIR 要是至少 12 位的字母、数字、下划线或连字符');
  const extraModules = splitList(process.env.CHAR_EXTRA_MODULES);
  for (const id of extraModules) {
    if (!WORLD_ID.test(id)) problems.push(`不是合法的模组 id：${id}`);
  }
  return {mode, worlds, publishDir, extraModules, backup};
}

/** 读副本，定下每个世界每个库要写的行（`puts`），世界摘要记进 `report.worlds`。 */
async function planJobs(report, work) {
  const {problems, warnings, inputs} = report;
  const now = Date.now();
  const jobs = [];
  report.worlds = {};
  for (const world of inputs.worlds) {
    const dir = worldDir(world);
    const names = world === CHAR_WORLD ? ['settings', 'folders'] : ['settings'];
    const worldJsonFile = path.join(dir, 'world.json');
    if (!fs.existsSync(worldJsonFile)) {
      problems.push(`没有这个世界：${worldJsonFile}`);
      continue;
    }
    const worldJson = JSON.parse(fs.readFileSync(worldJsonFile, 'utf8'));
    for (const field of ['coreVersion', 'system', 'systemVersion']) {
      if (typeof worldJson[field] !== 'string') problems.push(`${world} 的 world.json 没有 ${field}`);
    }
    const rows = {};
    for (const name of [...names, 'users']) rows[name] = await readCopy(path.join(dir, 'data', name), path.join(work, 'read', world, name));
    const gm = gmUserOf(rows.users);
    if (!gm) {
      problems.push(`${world} 没有 role 为 ${GM_ROLE} 的用户，_stats.lastModifiedBy 没有着落`);
      continue;
    }
    if (gm.count > 1) warnings.push(`${world} 有 ${gm.count} 个 GM，_stats.lastModifiedBy 用 ${gm.id}`);
    const moduleDir = path.join(world === CHAR_WORLD ? CHAR_DATA : MAIN_DATA, 'modules');
    for (const id of world === CHAR_WORLD ? [MODULE_ID, ...inputs.extraModules] : [MODULE_ID]) {
      if (!fs.existsSync(path.join(moduleDir, id, 'module.json'))) warnings.push(`${world} 那台还没装 ${id}（${moduleDir}）`);
    }
    const plan = planWorld({world, worldJson, rows, gm, inputs, now, problems, warnings});
    report.worlds[world] = plan.summary;
    for (const name of names) {
      jobs.push({world, name, srcDir: path.join(dir, 'data', name), readDir: path.join(work, 'read', world, name), rows: rows[name], puts: plan.puts[name]});
    }
  }
  return jobs;
}

await runTool({
  tool: 'enable-worlds',
  readInputs,
  async plan(report, work) {
    const jobs = await planJobs(report, work);
    if (!report.problems.length) await checkModels(jobs, report.problems);
    return jobs;
  },
});
