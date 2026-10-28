// 引擎的配置与种子表，启动时读一次；改了要重启引擎。
import fs from 'node:fs';
import path from 'node:path';

// 世界 id 和发布目录名直接作路径段
const WORLD_ID = /^[A-Za-z0-9_-]+$/;
const PUBLISH_DIR = /^[A-Za-z0-9_-]{12,}$/;
const FOUNDRY_ID = /^[A-Za-z0-9]{16}$/;

/** Foundry 文档 id：16 位字母数字。 */
export const isFoundryId = id => typeof id === 'string' && FOUNDRY_ID.test(id);

const readJson = file => JSON.parse(fs.readFileSync(file, 'utf8'));

function required(value, name) {
  if (value === undefined || value === null || value === '') throw new Error(`配置缺少 ${name}`);
  return value;
}

function nonNegative(value, name, fallback) {
  const number = value ?? fallback;
  if (typeof number !== 'number' || !Number.isFinite(number) || number < 0) throw new Error(`配置 ${name} 必须是非负数：${value}`);
  return number;
}

function worldId(id, name) {
  if (typeof id !== 'string' || !WORLD_ID.test(id)) throw new Error(`${name} 不是合法的世界 id：${id}`);
  return id;
}

/**
 * 读 `config.json`。默认值：`intervalSeconds` 60、`historyDays` 60、`tmpRoot` `/tmp`、`titles` 和 `campaignFolders` `{}`。
 * 路径字段按配置文件所在目录解析成绝对路径。世界 id 不合 `^[A-Za-z0-9_-]+$`、`publishDir` 不合
 * `^[A-Za-z0-9_-]{12,}$`、缺必填字段时抛错。`char.statusUrl` 原样保留，引擎读建卡服不看它开没开。
 * `campaignFolders` 只接受 Foundry 文件夹 id 到已配置世界的显式映射。
 */
export function loadConfig(file) {
  const raw = readJson(file);
  const base = path.dirname(path.resolve(file));
  const resolvePath = (value, name) => path.resolve(base, required(value, name));

  if (!Array.isArray(raw.worlds)) throw new Error('配置 worlds 必须是世界 id 数组');
  const publishDir = required(raw.publishDir, 'publishDir');
  if (typeof publishDir !== 'string' || !PUBLISH_DIR.test(publishDir)) throw new Error(`publishDir 不合规则（字母数字下划线连字符，至少 12 位）：${publishDir}`);
  const worlds = raw.worlds.map(id => worldId(id, 'worlds 里'));
  const campaignFolders = raw.campaignFolders === undefined ? {} : raw.campaignFolders;
  if (!campaignFolders || typeof campaignFolders !== 'object' || Array.isArray(campaignFolders)) {
    throw new Error('配置 campaignFolders 必须是文件夹 id 到世界 id 的对象');
  }
  for (const [folder, world] of Object.entries(campaignFolders)) {
    if (!isFoundryId(folder)) throw new Error(`配置 campaignFolders 的文件夹 id 不合法：${folder}`);
    if (!worlds.includes(world)) throw new Error(`配置 campaignFolders 的世界不在 worlds 里：${world}`);
  }

  return {
    foundryAppDir: resolvePath(raw.foundryAppDir, 'foundryAppDir'),
    main: {
      dataDir: resolvePath(raw.main?.dataDir, 'main.dataDir'),
      statusUrl: required(raw.main?.statusUrl, 'main.statusUrl'),
    },
    char: {
      dataDir: resolvePath(raw.char?.dataDir, 'char.dataDir'),
      world: worldId(raw.char?.world, 'char.world'),
      statusUrl: raw.char?.statusUrl ?? null,
    },
    publishDir,
    worlds,
    campaignFolders,
    titles: raw.titles ?? {},
    intervalSeconds: nonNegative(raw.intervalSeconds, 'intervalSeconds', 60),
    historyDays: nonNegative(raw.historyDays, 'historyDays', 60),
    stateDir: resolvePath(raw.stateDir, 'stateDir'),
    tmpRoot: path.resolve(base, raw.tmpRoot ?? '/tmp'),
    seedPath: resolvePath(raw.seedPath, 'seedPath'),
    dryOutDir: raw.dryOutDir ? path.resolve(base, raw.dryOutDir) : null,
  };
}

/**
 * 读 `seed.json`：`{cards:[{charId, world, mainId, companions?, bootstrap?}], users:[{world, worldUser, charUser}], bootstrapSince}`。
 * 卡的世界必须在 `worlds` 里，卡 id 必须是 Foundry id，`bootstrap` 只能是 `auto`、`main` 或不写，
 * `bootstrapSince` 必须是能解析的时间，否则抛错。
 */
export function loadSeed(file, worlds) {
  const seed = readJson(file);
  const cards = seed.cards ?? [];
  for (const card of cards) {
    if (!worlds.includes(card.world)) throw new Error(`种子表里卡 ${card.charId} 的世界 ${card.world} 不在配置的 worlds 里`);
    for (const id of [card.charId, card.mainId, ...(card.companions ?? [])]) {
      if (!isFoundryId(id)) throw new Error(`种子表里卡 ${card.charId} 有不合法的 id：${id}`);
    }
    if (card.bootstrap !== undefined && card.bootstrap !== 'auto' && card.bootstrap !== 'main') {
      throw new Error(`种子表里卡 ${card.charId} 的 bootstrap 只能是 auto 或 main：${card.bootstrap}`);
    }
  }
  if (!Number.isFinite(Date.parse(seed.bootstrapSince))) throw new Error(`种子表的 bootstrapSince 不是有效时间：${seed.bootstrapSince}`);
  return {cards, users: seed.users ?? [], bootstrapSince: seed.bootstrapSince};
}
