// 发布目录的写入：所有文件先写同目录的临时文件再 rename，客户端不会读到写了一半的 JSON。
import fs from 'node:fs';
import path from 'node:path';
import { historyFile } from '../scripts/lib/chain.mjs';

const DAY_MS = 24 * 60 * 60 * 1000;

function replaceFile(file, writeTemp) {
  fs.mkdirSync(path.dirname(file), {recursive: true});
  const temp = `${file}.${process.pid}.tmp`;
  try {
    writeTemp(temp);
    fs.renameSync(temp, file);
  } catch (error) {
    fs.rmSync(temp, {force: true});
    throw error;
  }
}

function readIfExists(file) {
  try {
    return fs.readFileSync(file);
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

/** 目录里的普通文件名，按名字排序；目录不存在时为空。 */
function fileNames(dir) {
  try {
    return fs.readdirSync(dir, {withFileTypes: true}).filter(entry => entry.isFile()).map(entry => entry.name).sort();
  } catch (error) {
    if (error.code === 'ENOENT') return [];
    throw error;
  }
}

/** 原子写入文本，按需建父目录。内容与现有文件逐字节相同时不写（mtime 不变），返回 false。 */
export function writeTextAtomic(file, text) {
  const bytes = Buffer.from(text);
  if (readIfExists(file)?.equals(bytes)) return false;
  replaceFile(file, temp => fs.writeFileSync(temp, bytes));
  return true;
}

/** `writeTextAtomic(file, JSON.stringify(value))`。 */
export const writeJsonAtomic = (file, value) => writeTextAtomic(file, JSON.stringify(value));

/** 原子拷贝，按需建父目录；目标已存在会被替换，要保留已有文件的调用方先自己判断。 */
export function copyFileAtomic(source, target) {
  replaceFile(target, temp => fs.copyFileSync(source, temp));
}

/**
 * 把一轮的结果写进每个 `<dataDir>/<publishDir>/`，各目录内容相同。`written` 是实际写了的文件数，没变的不算。
 * 只增改，不删上一轮留下的文件。
 */
export function publishAll({dataDirs, publishDir, meta, mains, copies, working, history}) {
  // 历史文件先写、meta.json 最后写：客户端顺着 meta 和条目里的引用取文件，引用先到、文件没到就取不到
  const files = [
    ...history.map(({ref, doc}) => [historyFile(ref), doc]),
    ...mains.map(({world, mainId, entry}) => [`main/${world}/${mainId}.json`, entry]),
    ...Object.entries(copies).map(([cardId, list]) => [`copies/${cardId}.json`, list]),
    ...Object.entries(working).map(([cardId, entry]) => [`working/${cardId}.json`, entry]),
    ['meta.json', meta],
  ].map(([rel, value]) => [rel, JSON.stringify(value)]);

  let written = 0;
  for (const dataDir of dataDirs) {
    for (const [rel, text] of files) {
      if (writeTextAtomic(path.join(dataDir, publishDir, rel), text)) written++;
    }
  }
  return {written};
}

/**
 * 把 `fromData` 发布目录里 `history/char-*.json` 中 `toData` 还没有的拷过去，已有的不动。
 * `copied` 是 `history/` 下的文件名。
 */
export function mirrorCharHistory({fromData, toData, publishDir}) {
  const fromDir = path.join(fromData, publishDir, 'history');
  const toDir = path.join(toData, publishDir, 'history');
  const copied = [];
  for (const name of fileNames(fromDir)) {
    if (!/^char-.+\.json$/.test(name) || fs.existsSync(path.join(toDir, name))) continue;
    copyFileAtomic(path.join(fromDir, name), path.join(toDir, name));
    copied.push(name);
  }
  return {copied};
}

/**
 * 删掉 `history/` 里 mtime 早于 `maxAgeDays` 天、又不在 `keep` 里的文件；发布目录的其他子目录不看。
 * `keep` 里可以是 `historyFile(ref)` 的原值（`history/x.json`），也可以是 `history/` 下的文件名（`x.json`）；
 * 返回的 `deleted` 是文件名。
 */
export function gcHistory({dataDir, publishDir, keep, maxAgeDays = 60, now = Date.now()}) {
  const dir = path.join(dataDir, publishDir, 'history');
  const cutoff = now - maxAgeDays * DAY_MS;
  const kept = new Set([...keep].map(entry => entry.replace(/^history\//, '')));
  const deleted = [];
  for (const name of fileNames(dir)) {
    const file = path.join(dir, name);
    if (kept.has(name) || fs.statSync(file).mtimeMs >= cutoff) continue;
    fs.rmSync(file);
    deleted.push(name);
  }
  return {deleted};
}
