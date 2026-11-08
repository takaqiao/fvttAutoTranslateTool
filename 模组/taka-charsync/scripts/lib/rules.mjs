// 两边改了同一处且改得不同时听谁的：构筑归建卡服，开团状态归主服，与同步方向无关。
import {isAppearanceGroupPath} from './appearance.mjs';
import {MIGRATION_KEY} from './canon.mjs';

export const OWN_FLAG = 'taka-charsync';

/** 角色上不参与合并、直接取写入方那份的顶层键（另有 `flags[OWN_FLAG]`）。 */
export const LOCAL_ACTOR_KEYS = ['_id', 'ownership', 'folder', 'sort', '_stats'];

/** 已有内嵌条目上不参与合并的顶层键；新增条目整份照抄来源侧，不受此限。 */
export const LOCAL_ITEM_KEYS = ['_stats', 'ownership', 'sort'];

function setMigration(doc, current) {
  if (doc.system === null || typeof doc.system !== 'object') return;
  const own = current.system?.[MIGRATION_KEY];
  if (own === undefined) delete doc.system[MIGRATION_KEY];
  else doc.system[MIGRATION_KEY] = structuredClone(own);
}

/**
 * 角色和 `current` 里已有的条目、效果，`system._migration` 换回 `current` 那份（它没有就去掉）；新增的条目
 * 保留来源侧那份，它的数据就是按那一版迁好的。`current` 是写入方现在的文档。就地改 `doc`，返回 `doc`。
 *
 * PF2e 按 `_migration.version` 判断一份数据还要跑哪些迁移（新建条目时补跑、迁移报告列出没迁完的、重新迁移
 * 从最低的版本起），`previous` 只是记录。两边都是写入方那台服务器自己的账，不能把另一台的记到这台的文档上。
 */
export function keepLocalMigration(doc, current) {
  setMigration(doc, current);
  for (const key of ['items', 'effects']) {
    const own = new Map((current[key] ?? []).map(entry => [entry._id, entry]));
    for (const entry of doc[key] ?? []) {
      if (own.has(entry._id)) setMigration(entry, own.get(entry._id));
    }
  }
  return doc;
}

const BUILD_ITEM_TYPES = new Set([
  'feat', 'class', 'ancestry', 'heritage', 'background', 'spell', 'spellcastingEntry', 'lore', 'action', 'deity', 'kit',
]);

// 路径按前缀匹配，`*` 匹配一段。
const parsePatterns = patterns => patterns.map(pattern => pattern.split('.'));

const CHAR_ACTOR_PATHS = parsePatterns([
  'system.details.level',
  'system.build',
  'system.skills.*.rank',
  'system.details.keyability',
  'system.details.languages',
  'system.details.biography',
  'system.crafting.formulas',
  'system.proficiencies',
  'flags.pf2e-skill-issue',
]);

// 构筑条目里随开团变化的战斗状态。备法表的元素按下标写成 `prepared.<i>`；
// `system.location.uses.value` 是天生法术的剩余次数。
const MAIN_BUILD_ITEM_PATHS = parsePatterns([
  'system.frequency.value',
  'system.slots.*.value',
  'system.slots.*.prepared.*.expended',
  'system.location.uses.value',
  'system.containerId',
]);

function matchesAny(path, patterns) {
  const segments = path.split('.');
  return patterns.some(pattern =>
    pattern.length <= segments.length && pattern.every((segment, i) => segment === '*' || segment === segments[i]));
}

/** 角色本身（不含内嵌条目）某个路径的归属。 */
export function ownerOfActorPath(path) {
  return matchesAny(path, CHAR_ACTOR_PATHS) ? 'char' : 'main';
}

const APPEARANCE_PATHS = parsePatterns([
  'name', 'prototypeToken.name', 'prototypeToken.ring',
]);

/** 外观草稿可以提交；占格、视野和灯光仍是主服设置。 */
export function isAppearancePath(path) {
  return isAppearanceGroupPath(path) || matchesAny(path, APPEARANCE_PATHS);
}

/** 整个条目的归属，决定一方删了、另一方改了时的去留。未列出的类型归主服。 */
export function ownerOfItem(itemType) {
  return BUILD_ITEM_TYPES.has(itemType) ? 'char' : 'main';
}

/** 条目内部某个路径的归属。 */
export function ownerOfItemPath(itemType, path) {
  if (ownerOfItem(itemType) === 'main') return 'main';
  return matchesAny(path, MAIN_BUILD_ITEM_PATHS) ? 'main' : 'char';
}
