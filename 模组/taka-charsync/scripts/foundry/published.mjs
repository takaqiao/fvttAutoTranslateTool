// 读引擎发布到 Data 下的静态 JSON。Foundry 把整个 Data 目录静态下发，文件不存在时回 404。
import { historyFile } from '../lib/chain.mjs';

/** 世界设置里的发布目录名；没有设置时抛错。 */
export function getPublishDir() {
  const dir = game.settings.get('taka-charsync', 'publishDir');
  if (!dir) throw new Error('taka-charsync：世界设置 publishDir 没有填，找不到同步数据');
  return dir;
}

/** 取发布目录下 `rel` 处的 JSON；文件不存在时为 null，其余失败抛错。 */
export async function fetchPublished(rel) {
  const url = foundry.utils.getRoute(`${getPublishDir()}/${rel}`);
  const response = await fetch(url, {cache: 'no-store'});
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`taka-charsync：读取 ${rel} 失败（HTTP ${response.status}）`);
  return response.json();
}

export const getMeta = () => fetchPublished('meta.json');

/** 某团世界某张卡的最新主服版本 `{hash, readAt, level, xp, applied, doc}`。 */
export const getMainLatest = (world, mainId) => fetchPublished(`main/${world}/${mainId}.json`);

/** 某张工作卡当前的全部冻结副本（含已撤回的）；还没有副本时为空数组。 */
export const getCopies = async cardId => (await fetchPublished(`copies/${cardId}.json`)) ?? [];

/** 某个版本引用对应的原始文档。 */
export const getHistoryDoc = ref => fetchPublished(historyFile(ref));
