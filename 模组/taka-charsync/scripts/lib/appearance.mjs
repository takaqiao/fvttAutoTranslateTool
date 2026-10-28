import {canonicalJson} from './canon.mjs';
import {collectMediaReferences, managedMediaHash, mediaComparisonDocs} from './media.mjs';

const GROUPS = [
  {path: 'img', fields: [
    'img', 'flags.tokenizer-2.avatarLayerStack', 'flags.tokenizer-2.avatarAspect',
    'flags.tokenizer-2.avatarCanvasSize',
  ]},
  {path: 'prototypeToken.texture', fields: [
    'prototypeToken.texture', 'prototypeToken.ring.enabled', 'prototypeToken.ring.subject',
    'prototypeToken.flags.pf2e.autoscale', 'prototypeToken.flags.tokenizer-2',
    'flags.tokenizer-2.layerStack', 'flags.tokenizer-2.isDynamicRingMode',
    'flags.tokenizer-2.isOversized', 'flags.tokenizer-2.oversizedScale',
    'flags.tokenizer-2.oversizedRingFill', 'flags.tokenizer-2.modeIsOversized',
    'flags.tokenizer-2.modeOversizedScale', 'flags.tokenizer-2.modeOversizedRingFill',
  ]},
].map(group => ({...group, paths: group.fields.map(field => field.split('.'))}));

const read = (doc, path) => path.reduce((value, key) => value?.[key], doc);
const comparable = (doc, group) => canonicalJson(Object.fromEntries(group.paths.map(path => [path.join('.'), read(doc, path)])));

export function isAppearanceGroupPath(path) {
  return GROUPS.some(group => group.fields.some(field => path === field || path.startsWith(`${field}.`)));
}

function copyGroup(out, source, group) {
  for (const path of group.paths) {
    const value = read(source, path);
    const key = path.at(-1);
    if (value === undefined) {
      const existing = read(out, path.slice(0, -1));
      if (existing != null) delete existing[key];
      continue;
    }
    let parent = out;
    for (const segment of path.slice(0, -1)) parent = parent[segment] ??= {};
    parent[key] = structuredClone(value);
  }
}

/** 只替换两组外观字段；来源缺少的配方字段一并删除，其他 flags 和 ring 颜色不动。 */
export function copyAppearanceGroups(out, source) {
  for (const group of GROUPS) copyGroup(out, source, group);
}

function managedPaths(source, group) {
  const doc = {};
  copyGroup(doc, source, group);
  return new Set(collectMediaReferences(doc).filter(({src}) => managedMediaHash(src)).map(({path}) => JSON.stringify(path)));
}

function equivalentSource(preferred, other, group) {
  const currentPaths = managedPaths(preferred, group);
  const otherPaths = managedPaths(other, group);
  // 只接受没有路径降级的整体升级；两边各有受管图片时保留目标侧，不能拼配方。
  return otherPaths.size > currentPaths.size && [...currentPaths].every(path => otherPaths.has(path)) ? other : preferred;
}

/** 图像、配方和缩放从同一版取；Avatar 与 Token 独立选版，两边都改时保留建卡草稿。 */
export function mergeAppearanceGroups({base, main, char, target = 'main',
  comparisonDocs = mediaComparisonDocs([base, main, char])}) {
  const doc = {}, conflicts = [];
  for (const group of GROUPS) {
    const [b, m, c] = comparisonDocs.map(source => comparable(source, group));
    let source;
    if (m === c) source = target === 'main' ? equivalentSource(main, char, group) : equivalentSource(char, main, group);
    else if (b === c) source = main;
    else {
      source = char;
      if (b !== m) conflicts.push({scope: 'actor', path: group.path, winner: 'char'});
    }
    copyGroup(doc, source, group);
  }
  return {doc, conflicts};
}
