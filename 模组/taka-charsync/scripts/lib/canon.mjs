// 内容哈希只看卡的内容：挪文件夹、改排序、改权限、Foundry 刷新时间戳都不算变化。
const VOLATILE_KEYS = new Set(['_stats', 'ownership', 'sort']);

function stripNested(value) {
  if (Array.isArray(value)) return value.map(stripNested);
  if (value === null || typeof value !== 'object') return value;
  const out = {};
  for (const [key, child] of Object.entries(value)) {
    if (!VOLATILE_KEYS.has(key)) out[key] = stripNested(child);
  }
  return out;
}

/** 去掉每一层的 `_stats`、`ownership`、`sort` 和顶层的 `folder`，返回新对象，不改入参。 */
export function stripVolatile(doc) {
  const stripped = stripNested(doc);
  delete stripped.folder;
  return stripped;
}

/**
 * 键按字典序排列的 JSON，同样内容不管键的插入顺序都得到同一个字符串。
 * 对象里值为 `undefined` 的键省略；数组里的 `undefined` 写成 `null`，与 `JSON.stringify` 一致。
 */
export function canonicalJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map(item => (item === undefined ? 'null' : canonicalJson(item))).join(',')}]`;
  }
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  const entries = Object.keys(value)
    .sort()
    .filter(key => value[key] !== undefined)
    .map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`);
  return `{${entries.join(',')}}`;
}

/** 去掉易变字段后的 SHA-256，取前 32 个小写十六进制字符。浏览器端要求安全上下文（HTTPS 或 localhost）。 */
export async function contentHash(doc) {
  const bytes = new TextEncoder().encode(canonicalJson(stripVolatile(doc)));
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest, 0, 16), byte => byte.toString(16).padStart(2, '0')).join('');
}
