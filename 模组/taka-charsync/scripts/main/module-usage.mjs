import {collectModuleUsage} from '../lib/module-usage.mjs';
import {OWN_FLAG} from '../lib/rules.mjs';

// 旧提交没有包信息时仍可识别字面量 UUID；保存的包信息只帮助认出命名空间，不直接充当依赖。
export function documentModuleUsage(doc, sourceUsage = []) {
  return collectModuleUsage(doc, [
    ...(doc.flags?.[OWN_FLAG]?.applied?.moduleUsage ?? []),
    ...sourceUsage,
    ...(game.modules?.values() ?? []),
  ]);
}

export function currentModuleUsage(actors) {
  return actors.flatMap(actor => documentModuleUsage(actor.toObject()));
}

const escape = value => foundry.utils.escapeHTML(String(value));

/** 只展示当前世界没有启用的包，每次渲染重新读取状态。来源和 flags 都不写成必需项。 */
export function moduleUsageHTML(usage) {
  const modules = new Map();
  for (const entry of usage) {
    if (game.modules?.get(entry.id)?.active) continue;
    const existing = modules.get(entry.id);
    if (existing) existing.evidence.push(...entry.evidence);
    else modules.set(entry.id, {...entry, evidence: [...entry.evidence]});
  }
  if (!modules.size) return '';
  const groups = [
    {kind: 'reference', title: '规则引用的模组', note: '以下模组尚未启用；相关规则、重新选择或授予条目时可能需要，请 GM 检查。'},
    {kind: 'flag', title: '模组数据，需核对', note: '角色保留了这些模组的数据；可能只是编辑记录，不代表必须开启。'},
    {kind: 'source', title: '内容来源（仅供参考）', note: '条目已复制到角色卡，来源模组通常不必为此开启。'},
  ];
  const parts = [];
  for (const {kind, title, note} of groups) {
    const entries = [...modules.values()].filter(entry => {
      const kinds = new Set(entry.evidence.map(evidence => evidence.kind));
      return kinds.has(kind) && (kind === 'reference' || !kinds.has('reference'))
        && (kind !== 'source' || !kinds.has('flag'));
    });
    if (!entries.length) continue;
    const lines = entries.sort((a, b) => a.id.localeCompare(b.id)).map(entry => {
      const installed = game.modules?.get(entry.id);
      const status = installed ? '未启用' : '本世界未找到（未安装或不兼容）';
      const names = [...new Set(entry.evidence.filter(evidence => evidence.kind === kind).map(evidence => evidence.documentName))];
      const detail = names.slice(0, 3).join('、') + (names.length > 3 ? ` 等 ${names.length} 个条目` : '');
      return `<li><strong>${escape(installed?.title ?? entry.title ?? entry.id)}</strong> (${escape(entry.id)})：${status}`
        + (detail ? `；${escape(detail)}` : '') + '</li>';
    });
    const content = `<p>${note}</p><ul>${lines.join('')}</ul>`;
    parts.push(kind === 'reference' ? `<div><strong>${title}</strong>${content}</div>`
      : `<details><summary>${title}</summary>${content}</details>`);
  }
  return parts.length ? `<div class="taka-charsync-module-usage">${parts.join('')}</div>` : '';
}

export function openModuleManagement() {
  if (!game.user?.isGM) return;
  return new foundry.applications.sidebar.apps.ModuleManagement().render({force: true});
}
