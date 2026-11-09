// pf2e-item-activations 14.3.3 生成的「启动项」的样子：flags 里存着父物品整份的 toObject()，
// 模组靠其中的 `_id` 认出某件物品的启动项（切换投资时改它们、删物品时一起删）。

export const ITEM_ACTIVATIONS = 'pf2e-item-activations';

export const charm = () => ({
  _id: 'charm', type: 'equipment', name: 'charm', effects: [], flags: {},
  system: {equipped: {carryType: 'worn', invested: true}, traits: {value: ['invested', 'magical']}},
});

/**
 * `parent` 的启动项。`enabled` 省略时是 createMissingActivations 新建的样子（没有 enabled、名字不带前缀）；
 * `type` 换成 spellcastingEntry、spell 就是法术物品生成的施法列表和法术。
 */
export function activation(_id, parent, {enabled, type = 'action'} = {}) {
  const flags = {grantedBy: structuredClone(parent)};
  if (enabled !== undefined) flags.enabled = enabled;
  return {
    _id, type, name: `${enabled === false ? '[X] ' : ''}启动：${parent.name}`, effects: [],
    system: {actionType: {value: 'action'}, actions: {value: 1}, traits: {value: ['activation', 'invested']}},
    flags: {[ITEM_ACTIVATIONS]: flags},
  };
}

/** 父物品投资状态切换后 item-activations 对启动项做的改动：名字加上或去掉「[X] 」，记下 enabled。 */
export function setEnabled(entry, enabled) {
  entry.name = `${enabled ? '' : '[X] '}${entry.name.replace('[X] ', '')}`;
  entry.flags[ITEM_ACTIVATIONS].enabled = enabled;
  return entry;
}

export const isActivation = entry => Boolean(entry.flags?.[ITEM_ACTIVATIONS]);
export const activationIds = doc => doc.items.filter(isActivation).map(entry => entry._id).sort();
