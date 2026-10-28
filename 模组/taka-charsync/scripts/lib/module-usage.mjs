import {OWN_FLAG} from './rules.mjs';

const EXCLUDED = new Set(['core', 'pf2e', 'world', OWN_FLAG]);
const ABC_TYPES = new Set(['ancestry', 'background', 'class']);
const compare = (a, b) => a < b ? -1 : a > b ? 1 : 0;

function moduleOf(uuid, itemOnly = false) {
  if (typeof uuid !== 'string') return null;
  const match = uuid.match(/^Compendium\.([A-Za-z0-9_-]+)\.[A-Za-z0-9_-]+\.(?:([A-Za-z]+)\.)?[A-Za-z0-9_-]+$/);
  if (!match || (itemOnly && match[2] && match[2] !== 'Item')) return null;
  return match[1];
}

function choiceValueType(value) {
  if (typeof value !== 'string') return 'nonItem';
  const path = value.startsWith('Compendium.') ? value.split('.').slice(3).join('.') : value;
  const type = path.match(/^(?:[A-Z][A-Za-z]*\.[A-Za-z0-9_-]+\.)*([A-Z][A-Za-z]*)\.[A-Za-z0-9_-]+$/)?.[1];
  if (type === 'Item') return 'Item';
  if (['Actor', 'JournalEntry', 'JournalEntryPage', 'ActiveEffect', 'Scene', 'Token', 'Macro', 'RollTable'].includes(type)) {
    return 'nonItem';
  }
  // 旧式 Compendium、相对或注入 UUID 缺少静态类型，不等同于普通选项。
  return value.includes('.') || value.includes('{') ? 'unknown' : 'nonItem';
}

/** modules 只提供模块命名空间和标题；证据始终从当前文档重新提取。 */
export function collectModuleUsage(doc, modules = []) {
  const titles = new Map([...modules].map(module => [module.id, module.title || module.id]));
  const usage = new Map();

  function visit(document, prefix) {
    if (!document || typeof document !== 'object') return;
    const pathOf = path => prefix ? `${prefix}.${path}` : path;
    const add = (id, kind, path) => {
      if (!id || EXCLUDED.has(id)) return;
      if (!usage.has(id)) usage.set(id, {id, title: titles.get(id) ?? id, evidence: []});
      usage.get(id).evidence.push({
        kind, documentId: document._id ?? '', documentName: document.name ?? '', path: pathOf(path),
      });
    };
    const reference = (uuid, path) => add(moduleOf(uuid, true), 'reference', path);

    add(moduleOf(document._stats?.compendiumSource), 'source', '_stats.compendiumSource');
    add(moduleOf(document.flags?.core?.sourceId), 'source', 'flags.core.sourceId');
    for (const id of Object.keys(document.flags ?? {})) {
      if (titles.has(id)) add(id, 'flag', `flags.${id}`);
    }

    // 这些字段在 PF2e 的规则准备、重新选择或授予流程中解引用；正文链接不算。
    const rules = document.system?.rules;
    for (const [index, rule] of (Array.isArray(rules) ? rules : []).entries()) {
      if (!rule) continue;
      const path = `system.rules.${index}`;
      switch (rule.key) {
        case 'GrantItem':
        case 'EphemeralEffect':
          reference(rule.uuid, `${path}.uuid`);
          break;
        case 'SpecialResource':
          reference(rule.itemUUID, `${path}.itemUUID`);
          break;
        case 'ChoiceSet':
          if (Array.isArray(rule.choices)) {
            const hasUnconditionalNonItem = rule.choices.some(choice => choiceValueType(choice?.value) === 'nonItem'
              && !(Array.isArray(choice?.predicate) && choice.predicate.length));
            // 全 Item 候选会批量读取；string rollOption 还可能读取选中项。predicate 不在这里求值。
            if (hasUnconditionalNonItem && typeof rule.rollOption !== 'string') break;
            for (const [choice, value] of rule.choices.entries()) {
              // 旧式 UUID 需要 pack 元数据才能判断文档类型，不能把普通选项猜成 Item。
              if (typeof value?.value === 'string' && value.value.split('.')[3] === 'Item') {
                reference(value.value, `${path}.choices.${choice}.value`);
              }
            }
          }
          break;
        case 'Aura':
          if (Array.isArray(rule.effects)) {
            for (const [effect, value] of rule.effects.entries()) reference(value?.uuid, `${path}.effects.${effect}.uuid`);
          }
          break;
      }
    }

    if (ABC_TYPES.has(document.type)) {
      for (const [key, entry] of Object.entries(document.system?.items ?? {})) {
        reference(entry?.uuid, `system.items.${key}.uuid`);
      }
    }

    for (const collection of ['items', 'effects']) {
      const children = document[collection];
      if (!Array.isArray(children)) continue;
      for (const [index, child] of children.entries()) visit(child, pathOf(`${collection}.${index}`));
    }
  }

  visit(doc, '');
  return [...usage.values()].sort((a, b) => compare(a.id, b.id)).map(entry => ({
    ...entry,
    evidence: entry.evidence.sort((a, b) => compare(a.kind, b.kind)
      || compare(a.documentId, b.documentId) || compare(a.path, b.path)),
  }));
}
