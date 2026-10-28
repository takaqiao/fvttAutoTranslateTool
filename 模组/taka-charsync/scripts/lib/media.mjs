export const MANAGED_MEDIA_ROOT = 'assets/taka-charsync-media';
export const MEDIA_FLAG = 'taka-charsync-media';

export const MEDIA_EXTENSIONS = new Set([
  'webp', 'png', 'jpg', 'jpeg', 'gif', 'svg', 'avif', 'bmp', 'tif', 'tiff', 'apng', 'webm', 'mp4', 'm4v', 'ogv',
]);

const RING_TEXTURES = new Set(['gold', 'silver', 'bronze', 'stone', 'wood', 'chainmail']);

/** 图片字段的精确位置；只枚举已支持的 Tokenizer 配方，不遍历其他 flags。 */
export function collectMediaReferences(doc) {
  const references = [];
  const add = path => {
    const src = path.reduce((value, key) => value?.[key], doc);
    if (typeof src === 'string' && src) references.push({path, src});
  };
  add(['img']);
  add(['prototypeToken', 'texture', 'src']);
  add(['prototypeToken', 'ring', 'subject', 'texture']);
  for (const [index] of (doc.items ?? []).entries()) add(['items', String(index), 'img']);
  for (const prefix of [[], ['prototypeToken']]) {
    for (const key of ['layerStack', 'avatarLayerStack']) {
      const path = [...prefix, 'flags', 'tokenizer-2', key];
      const layers = path.reduce((value, key) => value?.[key], doc);
      if (!Array.isArray(layers)) continue;
      for (const [index, layer] of layers.entries()) {
        const base = [...path, String(index)];
        add([...base, 'src']);
        add([...base, 'paintMask', 'src']);
        for (const [mask, value] of (layer?.masks ?? []).entries()) {
          // 生成式圆环的 src 是同配方中的图层 ID，不是图片路径。
          if (value?.type !== 'ring-frame') add([...base, 'masks', String(mask), 'src']);
        }
        if (!RING_TEXTURES.has(layer?.ringConfig?.texture)) add([...base, 'ringConfig', 'texture']);
      }
    }
  }
  return references;
}

/** 仅接受完整、无 query 的内容寻址媒体路径；文件名本身就是引擎校验用的哈希。 */
export function managedMediaHash(path) {
  if (typeof path !== 'string') return null;
  const match = path.match(/^assets\/taka-charsync-media\/([a-f0-9]{64})\.([a-z0-9]+)$/);
  return match && MEDIA_EXTENSIONS.has(match[2]) ? match[1] : null;
}

/** 元数据按 Item id 定位；收集器仍返回可直接改写文档的数组下标。 */
export function mediaMetadataPath(doc, path) {
  if (path[0] !== 'items') return path;
  const id = doc.items?.[Number(path[1])]?._id;
  return typeof id === 'string' && id ? ['items', id, ...path.slice(2)] : null;
}

function currentMedia(doc) {
  return new Map(collectMediaReferences(doc).flatMap(({path, src}) => {
    const metadataPath = mediaMetadataPath(doc, path);
    return metadataPath ? [[JSON.stringify(metadataPath), src]] : [];
  }));
}

function validReferences(doc) {
  const metadata = doc.flags?.[MEDIA_FLAG];
  if (metadata?.version !== 1 || !Array.isArray(metadata.references)) return [];
  const current = currentMedia(doc);
  return metadata.references.filter(ref => Array.isArray(ref?.path) && ref.path.every(key => typeof key === 'string')
    && typeof ref.source === 'string' && ref.source
    && typeof ref.sha256 === 'string' && managedMediaHash(ref.frozen) === ref.sha256
    && current.get(JSON.stringify(ref.path)) === ref.frozen);
}

function removeMetadata(doc) {
  if (!doc.flags || !Object.hasOwn(doc.flags, MEDIA_FLAG)) return;
  delete doc.flags[MEDIA_FLAG];
  if (!Object.keys(doc.flags).length) delete doc.flags;
}

/** 从来源版本重新绑定当前字段仍在使用的媒体，已改图的旧记录不延续。原地更新 doc。 */
export function combineMediaMetadata(doc, sources) {
  const current = currentMedia(doc);
  const references = new Map();
  for (const source of sources) {
    for (const ref of validReferences(source)) {
      if (current.get(JSON.stringify(ref.path)) !== ref.frozen) continue;
      const entry = {path: [...ref.path], source: ref.source, frozen: ref.frozen, sha256: ref.sha256};
      // 同一字段、同一内容可保留不同原 URL；完全重复的记录只留一次。
      references.set(JSON.stringify(entry), entry);
    }
  }
  if (references.size) {
    doc.flags ??= {};
    doc.flags[MEDIA_FLAG] = {version: 1, references: [...references.values()]};
  } else removeMetadata(doc);
  return doc;
}

/** 仅供比较的副本：受管内容按哈希比较，旧 URL 不借用另一版本的媒体身份。 */
export function mediaComparisonDocs(docs) {
  const hasMetadata = docs.some(doc => doc.flags && Object.hasOwn(doc.flags, MEDIA_FLAG));
  return docs.map(doc => {
    const compared = structuredClone(doc);
    for (const {path, src} of collectMediaReferences(compared)) {
      const hash = managedMediaHash(src.replace(/[?#].*$/s, ''));
      if (!hash) continue;
      const parent = path.slice(0, -1).reduce((value, key) => value[key], compared);
      parent[path.at(-1)] = `${MEDIA_FLAG}:sha256:${hash}`;
    }
    removeMetadata(compared);
    if (hasMetadata) compared.flags ??= {};
    return compared;
  });
}
