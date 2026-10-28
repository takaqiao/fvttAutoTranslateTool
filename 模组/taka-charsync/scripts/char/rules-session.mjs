import { getMeta } from '../foundry/published.mjs';
import { canonicalJson } from '../lib/canon.mjs';
import {
  actorRuleState, familiarMasterState, refreshActorRules, selfCheck, setRuleRefreshError, updateRuleSnapshot,
} from './rules-override.mjs';

export function ruleProblem(reason) {
  const messages = {
    'missing-assignment': '尚未指定所属团，请由 GM 将工作卡放入已配置的团文件夹',
    'assignment-conflict': '角色映射、主人或文件夹指向不同的团，请由 GM 核对归属',
    'assignment-cycle': '主人或备份关联出现循环，请由 GM 修复关联',
    'folder-cycle': '文件夹关联出现循环，请由 GM 修复文件夹',
    'missing-master': '魔宠尚未关联主人，请先关联本团的工作卡',
    'invalid-master': '魔宠主人必须是本团的正常角色工作卡，不能选择提交副本、备份或归档卡',
    'master-permission': '只能选择你有所有者权限的角色工作卡作为魔宠主人',
    'missing-actor': '找不到关联的主人或工作卡，请由 GM 核对关联',
    'missing-folder': '找不到所属文件夹，请由 GM 核对归属',
    'missing-copy-target': '提交副本缺少目标团，请从工作卡重新提交',
    'missing-profile': '所属团规则尚未发布，请等待同步引擎恢复',
  };
  if (messages[reason]) return messages[reason];
  if (reason?.startsWith('rule-refresh-failed:')) return '暂时读不到最新团规则，连接恢复后会自动重试';
  if (reason?.startsWith('adapter-failed:') || reason?.startsWith('unregistered-rule:')
    || reason === 'missing-pf2e-rule-cache') return '团规则未能正确加载，请联系 GM 检查模组兼容性';
  return '团规则资料不完整，请联系 GM 检查同步引擎';
}

/** 状态条、定时刷新和提交共用最新发布；读取失败时保留卡片，但不允许按旧资料继续构筑。 */
export async function loadRuleMeta() {
  try {
    const meta = await getMeta();
    if (meta?.ruleSnapshot?.version !== 2) throw new Error('团规则尚未发布，请联系 GM 检查同步引擎');
    setRuleRefreshError(null);
    updateRuleSnapshot(meta.ruleSnapshot);
    return meta;
  } catch (error) {
    setRuleRefreshError(error.message ?? String(error));
    throw error;
  }
}

export function requireActorRules(actor, targetWorld = null) {
  const state = actorRuleState(actor);
  if (!state.ok) throw new Error(`${actor.name}：${ruleProblem(state.reason)}`);
  if (targetWorld && state.world !== targetWorld) throw new Error(`${actor.name}所属团与提交目标不同，请先核对角色归属`);
  const errors = selfCheck(actor);
  if (errors.length) throw new Error(`${actor.name}的团规则未能套用：${errors.join('、')}`);
  return state;
}

export async function prepareSubmissionRules(actor, targetWorld, companions = []) {
  if (typeof targetWorld !== 'string' || !targetWorld.trim()) throw new Error('提交目标团缺失，请由 GM 核对角色映射');
  const before = actorRuleState(actor);
  const meta = await loadRuleMeta();
  const current = requireActorRules(actor, targetWorld);
  if (!before.ok || before.world !== current.world || before.hash !== current.hash) {
    throw new Error('所属团规则刚更新，请检查重新计算后的角色卡，再提交');
  }
  for (const companion of companions) requireActorRules(companion, targetWorld);
  return {meta, rulesHash: current.hash};
}

function repairedMasterState(actor, changes) {
  if (actor.type !== 'familiar') return null;
  const leaves = {};
  const flatten = (value, prefix = '') => {
    for (const [key, child] of Object.entries(value)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (child && typeof child === 'object' && !Array.isArray(child)) flatten(child, path);
      else leaves[path] = child;
    }
  };
  flatten(changes ?? {});
  // preUpdateActor 在 Foundry 求 diff 之前执行；原生选择器会带上整张表单和 _id。
  const changed = Object.keys(leaves).filter(path => canonicalJson(leaves[path])
    !== canonicalJson(path.split('.').reduce((value, key) => value?.[key], actor._source)));
  if (!changed.includes('system.master.id')) return null;
  const masterId = leaves['system.master.id'];
  const state = familiarMasterState(actor, masterId);
  if (!state) return null;
  return {state, clearing: masterId === null || masterId === '',
    onlyMaster: changed.every(path => ['system.master.id', 'system.master.ability'].includes(path))};
}

/** 只拦构筑数据；改文件夹仍可用来修复尚未归属的新卡。 */
export function installRuleEditGuards() {
  let lastNotice = 0;
  function guard(document, changes, options = {}) {
    if (options.takaCharsync) return;
    let actor = document.actor ?? document;
    while (actor && actor.documentName !== 'Actor') actor = actor.parent;
    if (!actor || !['character', 'familiar'].includes(actor.type)) return;
    try {
      const repair = document.documentName === 'Actor' ? repairedMasterState(actor, changes) : null;
      if (repair) {
        if (!repair.clearing && !repair.state.ok) throw new Error(`${actor.name}：${ruleProblem(repair.state.reason)}`);
        if (repair.onlyMaster) return;
      }
      if (document.documentName === 'Actor' && !Object.keys(changes ?? {}).some(key =>
        key === 'system' || key.startsWith('system.') || key === 'items' || key === 'effects'
        || key === 'flags' || key.startsWith('flags.'))) return;
      requireActorRules(actor);
    } catch (error) {
      if (Date.now() - lastNotice > 10_000) {
        lastNotice = Date.now();
        ui.notifications.warn(error.message);
      }
      return false;
    }
  }
  Hooks.on('preUpdateActor', guard);
  for (const type of ['Item', 'ActiveEffect']) {
    Hooks.on(`preCreate${type}`, guard);
    Hooks.on(`preUpdate${type}`, guard);
    Hooks.on(`preDelete${type}`, (document, options) => guard(document, null, options));
  }
  const provenanceChanged = changes => Object.keys(changes ?? {}).some(key =>
    key === 'folder' || key === 'flags' || key.startsWith('flags.taka-charsync')
    || key === 'system' || key === 'system.master' || key.startsWith('system.master.'));
  // 主人关联和副本标记会影响团规来源，文件夹也可被整体移入另一团。
  // 重算所有角色和魔宠以覆盖这些依赖；reset 不会产生文档更新。
  Hooks.on('updateActor', (_actor, changes) => {
    if (provenanceChanged(changes)) refreshActorRules();
  });
  Hooks.on('updateFolder', (_folder, changes) => {
    if (Object.hasOwn(changes, 'folder')) refreshActorRules();
  });
  for (const event of ['createActor', 'deleteActor', 'createFolder', 'deleteFolder']) {
    Hooks.on(event, () => refreshActorRules());
  }
}

export async function startRuleRefresh() {
  const refresh = () => loadRuleMeta().catch(error => console.warn('taka-charsync | 团规则读取失败', error));
  const initial = await refresh();
  // 用串行 timeout，慢请求不会叠出多个并发刷新。
  const repeat = () => setTimeout(async () => { await refresh(); repeat(); }, 30_000);
  repeat();
  return initial;
}
