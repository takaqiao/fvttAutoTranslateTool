// 卡上引用的图片：一边的 Data 里有、另一边没有的拷过去，卡导入后头像和 token 才显示得出来。
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { collectMediaReferences, managedMediaHash, MANAGED_MEDIA_ROOT, MEDIA_EXTENSIONS } from '../scripts/lib/media.mjs';
import { snapshotActorMedia } from '../scripts/foundry/media.mjs';
import { copyFileAtomic } from './publish.mjs';

// 系统、模组、核心图标两台各有一份；世界目录不往对方建；网址和内嵌数据不是文件
const SKIPPED_PREFIXES = ['systems/', 'modules/', 'icons/', 'worlds/', 'http', 'data:'];

const isSkipped = rel => SKIPPED_PREFIXES.some(prefix => rel.startsWith(prefix));

function decodePath(src) {
  try {
    return decodeURIComponent(src);
  } catch {
    return src;
  }
}

// `./worlds/…`、`/modules/…` 这类写法 join 到 Data 根后照样落进共享目录，要先化成规范形式再判前缀
function normalizePath(src) {
  return path.posix.normalize(src.replaceAll('\\', '/')).replace(/^(?:\.?\/)+/, '');
}

/**
 * 卡的头像、默认 token、条目及 Tokenizer 配方里要镜像的路径（相对 Data 根，已规范化）。
 * 先去掉 `?`、`#` 及其后的部分，再解码、规范化、去重：同一张图带不带 query、编码和未编码各出现一次都只算一条；
 * 解码失败（如裸 `%`）用原文。含 `..` 的照样返回，由 `mirrorImages` 拒绝并记下。
 */
export function collectImagePaths(doc) {
  const paths = new Set();
  for (const {src} of collectMediaReferences(doc)) {
    // Tokenizer 这类模组在路径后加 `?时间戳` 防缓存，静态服务不看它，文件在去掉 query 的路径上；
    // 先剥再解码，文件名里编码过的 `%3F`、`%23` 解码后留作普通字符
    const file = src.replace(/[?#].*$/s, '');
    if (!file) continue;
    const rel = normalizePath(decodePath(file));
    if (!isSkipped(rel)) paths.add(rel);
  }
  return [...paths];
}

function verifyManagedBytes(bytes, expected, side) {
  const actual = crypto.createHash('sha256').update(bytes).digest('hex');
  if (actual !== expected) throw new Error(`${side} SHA-256 不符：期望 ${expected}，实际 ${actual}`);
}

/** 临时文件写完才公开；link 不替换现有目标，另一写入者抢先落盘时仍要核对哈希。 */
function writeManagedImage(bytes, target, hash) {
  fs.mkdirSync(path.dirname(target), {recursive: true});
  const temp = `${target}.${crypto.randomUUID()}.tmp`;
  try {
    fs.writeFileSync(temp, bytes, {flag: 'wx'});
    try {
      fs.linkSync(temp, target);
      return true;
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
      verifyManagedBytes(fs.readFileSync(target), hash, '目标');
      return false;
    }
  } finally {
    fs.rmSync(temp, {force: true});
  }
}

/** 路径是否在 `root` 之下（含 `root` 本身）。两个都要是同一种形式的绝对路径。 */
function contains(root, file) {
  const rel = path.relative(root, file);
  return !path.isAbsolute(rel) && rel.split(path.sep)[0] !== '..';
}

/** 解析符号链接后的真实路径；路径还不存在时，解析最近一级已存在的祖先再接上余下部分。 */
function realPathOf(file) {
  let existing = file;
  while (!fs.lstatSync(existing, {throwIfNoEntry: false})) existing = path.dirname(existing);
  return path.join(fs.realpathSync(existing), path.relative(existing, file));
}

/** 玩家只能向现有目录上传；由引擎预建受管目录，仍按真实路径限制在对应 Data 内。 */
export function ensureManagedMediaRoot(dataDir) {
  const target = path.join(dataDir, MANAGED_MEDIA_ROOT);
  if (!contains(realPathOf(dataDir), realPathOf(target))) throw new Error(`媒体目录越出 Data 根：${target}`);
  const existing = fs.statSync(target, {throwIfNoEntry: false});
  if (existing && !existing.isDirectory()) throw new Error(`媒体目录路径已被文件占用：${target}`);
  fs.mkdirSync(target, {recursive: true});
}

function mediaFile(dataDir, rel, publishDir) {
  const file = path.resolve(dataDir, rel);
  const publishRoot = publishDir?.toLowerCase();
  const inside = path.relative(dataDir, file).split(path.sep).join('/').toLowerCase();
  if (rel.includes('\0') || rel.split(/[\\/]/).includes('..')
      || !MEDIA_EXTENSIONS.has(path.extname(rel).slice(1).toLowerCase())
      || !contains(path.resolve(dataDir), file) || !contains(realPathOf(dataDir), realPathOf(file))
      || (publishRoot && (inside === publishRoot || inside.startsWith(`${publishRoot}/`)))) {
    throw new Error(`媒体路径越出 Data 或落入发布目录：${rel}`);
  }
  return file;
}

/** 只重查仍可覆写的来源文件；与数据库指纹一起使用现有读取节流。 */
export function sourceMediaHashes(paths, dataDir, {publishDir} = {}) {
  return Object.fromEntries(paths.map(rel => {
    try {
      return [rel, crypto.createHash('sha256').update(fs.readFileSync(mediaFile(dataDir, rel, publishDir))).digest('hex')];
    } catch (error) {
      // 旧文件可能随 Actor 换图被删除；让数据库重读确认当前引用，再报告真正缺少的媒体。
      return [rel, `error:${error.code ?? error.message}`];
    }
  }));
}

/** 新主服版本从自身 Data 取得字节；历史不回填，dry 模式的受管资产只写输出 Data。 */
export async function snapshotMainMedia(doc, fromData, toData, {publishDir} = {}) {
  const sources = new Map();
  const stored = new Set();
  const read = async rel => {
    if (stored.has(rel)) return fs.readFileSync(mediaFile(toData, rel, publishDir));
    if (!sources.has(rel)) {
      try {
        sources.set(rel, fs.readFileSync(mediaFile(fromData, rel, publishDir)));
      } catch (error) {
        throw new Error(`读不到媒体 ${rel}：${error.message}`, {cause: error});
      }
    }
    return sources.get(rel);
  };
  const store = async (rel, bytes) => {
    writeManagedImage(bytes, mediaFile(toData, rel, publishDir), managedMediaHash(rel));
    stored.add(rel);
  };
  const result = await snapshotActorMedia([doc], {read, store});
  for (const asset of result.manifest.assets) {
    if (!stored.has(asset.path)) await store(asset.path, sources.get(asset.src));
  }
  return {doc: result.docs[0], legacyHashes: Object.fromEntries(result.manifest.assets
    .filter(asset => !managedMediaHash(asset.src)).map(asset => [asset.src, asset.hash]))};
}

/** 按路径本身就能判定不拷的：含 NUL、含 `..` 段、不是图片或视频、join 后不在 Data 根之下、落进共享目录或发布目录。 */
function refusedByPath(rel, toData, publishDir) {
  if (rel.includes('\0') || rel.split(/[\\/]/).includes('..')) return true;
  if (!MEDIA_EXTENSIONS.has(path.posix.extname(rel).slice(1).toLowerCase())) return true;
  const inside = path.relative(toData, path.join(toData, rel)).split(path.sep).join('/');
  if (!inside || inside.startsWith('..') || path.isAbsolute(inside) || isSkipped(inside)) return true;
  // 发布目录里是整张卡的 JSON 和历史，图片路径不该指进去；大小写不敏感的文件系统上换个大小写也是同一个目录
  const publishRoot = publishDir?.toLowerCase();
  return Boolean(publishRoot) && (inside.toLowerCase() === publishRoot || inside.toLowerCase().startsWith(`${publishRoot}/`));
}

/**
 * 把 `paths`（相对 Data 根）从 `fromData` 拷到 `toData`。普通旧路径已有的不比内容、不覆盖。
 * 受管路径必须核对两边的 SHA-256，冲突记进 `failed`，不覆盖；源里没有的记进 `missing`。
 * 记进 `refused` 的不碰：含 NUL 或 `..` 段的、扩展名不是图片或视频的（大小写不敏感）、join 后不在 Data 根之下的、
 * 落进共享目录或 `publishDir` 的，以及解析符号链接后源文件不在 `fromData` 之下、或目标所在目录不在 `toData` 之下的。
 * 每条路径各自处理，读写出错的记进 `failed`（`{path, message}`），不影响后面的路径。
 */
export function mirrorImages(paths, fromData, toData, {publishDir} = {}) {
  const result = {copied: [], present: [], missing: [], refused: [], failed: []};
  const realFrom = realPathOf(fromData);
  const realTo = realPathOf(toData);
  for (const rel of paths) {
    if (refusedByPath(rel, toData, publishDir)) {
      result.refused.push(rel);
      continue;
    }
    try {
      const target = path.join(toData, rel);
      if (!contains(realTo, realPathOf(target))) {
        result.refused.push(rel);
        continue;
      }
      const hash = managedMediaHash(rel);
      if (!hash && fs.statSync(target, {throwIfNoEntry: false})?.isFile()) {
        result.present.push(rel);
        continue;
      }
      const source = path.join(fromData, rel);
      if (!fs.statSync(source, {throwIfNoEntry: false})?.isFile()) {
        result.missing.push(rel);
        continue;
      }
      if (!contains(realFrom, fs.realpathSync(source)) || !contains(realTo, realPathOf(path.dirname(target)))) {
        result.refused.push(rel);
        continue;
      }
      if (hash) {
        const bytes = fs.readFileSync(source);
        verifyManagedBytes(bytes, hash, '来源');
        if (fs.existsSync(target)) {
          verifyManagedBytes(fs.readFileSync(target), hash, '目标');
          result.present.push(rel);
        } else {
          result[writeManagedImage(bytes, target, hash) ? 'copied' : 'present'].push(rel);
        }
        continue;
      }
      // 拷到一半被打断也只留临时文件，不会留下一张以后永远算「已有」的残图
      copyFileAtomic(source, target);
      result.copied.push(rel);
    } catch (error) {
      result.failed.push({path: rel, message: error.message});
    }
  }
  return result;
}
