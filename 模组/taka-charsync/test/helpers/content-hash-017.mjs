import {canonicalJson} from '../../scripts/lib/canon.mjs';

const VOLATILE_KEYS_017 = new Set(['_stats', 'ownership', 'sort']);

function strip(value) {
  if (Array.isArray(value)) return value.map(strip);
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).filter(([key]) => !VOLATILE_KEYS_017.has(key)).map(([key, child]) => [key, strip(child)]));
}

/** 0.1.7 的内容哈希：只去掉 `_stats`、`ownership`、`sort` 和顶层 `folder`，迁移记录还算在内容里。 */
export async function contentHash017(doc) {
  const stripped = strip(doc);
  delete stripped.folder;
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonicalJson(stripped)));
  return Array.from(new Uint8Array(digest, 0, 16), byte => byte.toString(16).padStart(2, '0')).join('');
}
