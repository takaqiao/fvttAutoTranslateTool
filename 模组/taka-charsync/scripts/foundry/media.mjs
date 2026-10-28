import {collectMediaReferences, combineMediaMetadata, MANAGED_MEDIA_ROOT, managedMediaHash,
  MEDIA_EXTENSIONS, MEDIA_FLAG, mediaMetadataPath} from '../lib/media.mjs';

const SHARED_MEDIA = /^(?:icons|systems|modules)\//;
const EXTERNAL_MEDIA = /^(?:https?:)?\/\//i;

function localPath(src) {
  let path = src.replace(/[?#].*$/s, '');
  try { path = decodeURIComponent(path); } catch { /* 文件名可以含未编码的百分号。 */ }
  path = path.replaceAll('\\', '/');
  if (path.includes('\0') || path.split('/').includes('..')) throw new Error(`媒体路径不能包含上级目录：${src}`);
  return path.split('/').filter(part => part && part !== '.').join('/');
}

async function sha256(bytes) {
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(hash)].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

async function readMedia(path, {allowMissing = false} = {}) {
  const url = foundry.utils.getRoute(path.split('/').map(encodeURIComponent).join('/'));
  const response = await fetch(url, {cache: 'no-store'});
  if (allowMissing && response.status === 404) return null;
  if (!response.ok) throw new Error(`读不到媒体 ${path}（HTTP ${response.status}）`);
  return new Uint8Array(await response.arrayBuffer());
}

function mediaStore() {
  return async (path, bytes) => {
    const expected = managedMediaHash(path);
    const current = await readMedia(path, {allowMissing: true});
    if (current) {
      if (await sha256(current) !== expected) throw new Error(`媒体哈希不符，未覆盖已有文件：${path}`);
      return;
    }
    const picker = foundry.applications.apps.FilePicker.implementation;
    const file = new File([bytes], path.slice(MANAGED_MEDIA_ROOT.length + 1));
    // 固定目录由同步引擎初始化；玩家只需上传权限，不尝试建目录或浏览文件。
    const failure = '媒体保存失败，请确认同步引擎已准备素材目录，且账号有文件上传权限';
    let uploaded;
    try {
      uploaded = await picker.upload('data', MANAGED_MEDIA_ROOT, file, {}, {notify: false});
    } catch (error) {
      throw new Error(`${failure}：${error?.message ?? error}`, {cause: error});
    }
    if (!uploaded?.path) throw new Error(`${failure}（未收到保存路径：${path}）`);
    if (localPath(uploaded.path) !== path) throw new Error(`媒体没有保存到预定路径：${path}`);
  };
}

/**
 * 冻结整组文档的本地媒体，成功后才返回改写过的副本；输入文档不变。
 * 核心/系统/模组资源保持安装路径，HTTP URL 保留并列在 manifest.external；不承诺冻结外站内容。
 * read(path) 返回 Uint8Array，store(path, bytes) 完成持久化或抛错，可替换文件 IO 做测试。
 */
export async function snapshotActorMedia(docs, {read = readMedia, store = mediaStore()} = {}) {
  const detached = structuredClone(docs);
  const previous = detached.map(doc => {
    combineMediaMetadata(doc, [doc]);
    return doc.flags?.[MEDIA_FLAG]?.references ?? [];
  });
  const sources = new Map();
  const references = detached.map(doc => collectMediaReferences(doc));
  const external = new Set();
  for (const refs of references) {
    for (const {src} of refs) {
      if (EXTERNAL_MEDIA.test(src)) { external.add(src); continue; }
      if (src.startsWith('data:')) continue;
      if (/^[a-z][a-z0-9+.-]*:/i.test(src)) throw new Error(`无法冻结临时或非本地媒体：${src}`);
      const source = localPath(src);
      if (SHARED_MEDIA.test(source)) continue;
      const extension = source.match(/\.([a-z0-9]+)$/i)?.[1].toLowerCase();
      if (!MEDIA_EXTENSIONS.has(extension) || /[*?]/.test(source)) throw new Error(`无法冻结媒体路径：${src}`);
      if (!sources.has(source)) {
        const bytes = await read(source);
        const hash = await sha256(bytes);
        const expected = managedMediaHash(source);
        if (expected && expected !== hash) throw new Error(`媒体哈希不符：${source}`);
        sources.set(source, {src: source, path: `${MANAGED_MEDIA_ROOT}/${hash}.${extension}`, hash, size: bytes.byteLength, bytes});
      }
    }
  }
  const stored = new Set();
  for (const asset of sources.values()) {
    if (!stored.has(asset.path) && asset.src !== asset.path) {
      await store(asset.path, asset.bytes);
      if (await sha256(await read(asset.path)) !== asset.hash) throw new Error(`媒体上传后哈希不符：${asset.path}`);
    }
    stored.add(asset.path);
  }
  for (const [index, refs] of references.entries()) {
    const metadata = [];
    for (const {path, src} of refs) {
      if (EXTERNAL_MEDIA.test(src) || src.startsWith('data:')) continue;
      const asset = sources.get(localPath(src));
      if (!asset) continue;
      const parent = path.slice(0, -1).reduce((value, key) => value[key], detached[index]);
      parent[path.at(-1)] = asset.path;
      const metadataPath = mediaMetadataPath(detached[index], path);
      if (!metadataPath) continue;
      if (managedMediaHash(localPath(src))) {
        metadata.push(...previous[index].filter(ref => JSON.stringify(ref.path) === JSON.stringify(metadataPath)));
      } else metadata.push({path: metadataPath, source: src, frozen: asset.path, sha256: asset.hash});
    }
    detached[index].flags ??= {};
    detached[index].flags[MEDIA_FLAG] = {version: 1, references: metadata};
    combineMediaMetadata(detached[index], [detached[index]]);
  }
  return {docs: detached, manifest: {
    assets: [...sources.values()].map(({bytes, ...asset}) => asset), external: [...external],
  }};
}
