import { canonicalJson, stripVolatile } from './canon.mjs';
import { formatSummary, summarize } from './merge.mjs';
import { MEDIA_FLAG, mediaComparisonDocs } from './media.mjs';
import { LOCAL_ACTOR_KEYS, OWN_FLAG } from './rules.mjs';

const escape = value => String(value).replace(/[&<>"']/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[character]);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const same = (a, b) => canonicalJson(a) === canonicalJson(b);

const FIELD_NAMES = {
  name: '名称', img: '头像', type: '角色类型',
  'system.attributes.hp.value': '当前血量', 'system.attributes.hp.max': '最大血量',
  'system.attributes.hp.temp': '临时血量', 'system.details.xp.value': '经验',
  'system.details.keyability.value': '关键属性', 'system.master.id': '主人',
  'system.master.ability': '主人属性', 'prototypeToken.name': '棋子名称',
  'prototypeToken.texture.src': '棋子图像',
  'system.details.biography.appearance': '外貌',
  'system.details.biography.backstory': '人物经历',
  'system.details.biography.value': '人物背景',
};
const FIELD_GROUPS = {
  'system.details.languages': '语言', 'system.details.biography': '人物背景',
  'system.details.deity': '信仰', 'system.build': '构筑',
  'system.proficiencies': '熟练项', 'system.crafting.formulas': '制造配方',
  'system.resources': '资源', 'system.traits': '特征', 'prototypeToken': '棋子设置',
};

function fieldLabel(path) {
  if (FIELD_NAMES[path]) return FIELD_NAMES[path];
  const group = Object.keys(FIELD_GROUPS).find(prefix => path === prefix || path.startsWith(`${prefix}.`));
  return group ? `${FIELD_GROUPS[group]} · ${path.slice(group.length + 1) || '内容'}` : path;
}

function displayValue(value) {
  if (value === undefined) return '未设置';
  if (value === null) return '空';
  if (value === true) return '是';
  if (value === false) return '否';
  const text = typeof value === 'string' ? value : canonicalJson(value);
  return escape(text || '空');
}

function actorContent(doc) {
  const out = stripVolatile(doc);
  for (const key of [...LOCAL_ACTOR_KEYS, 'items', 'effects']) delete out[key];
  out.flags = {...out.flags};
  delete out.flags[OWN_FLAG];
  delete out.flags[MEDIA_FLAG];
  return out;
}

function fieldChanges(before, after, accepts, path = '', changes = [], compared = [before, after]) {
  if (same(...compared)) return changes;
  if ((object(before) || before === undefined) && (object(after) || after === undefined)) {
    const keys = new Set([...Object.keys(before ?? {}), ...Object.keys(after ?? {})]);
    if (keys.size) {
      for (const key of keys) fieldChanges(before?.[key], after?.[key], accepts, path ? `${path}.${key}` : key, changes,
        compared.map(value => value?.[key]));
      return changes;
    }
  }
  // 等级和技能由 summarize 生成中文说明，避免重复列出原始路径。
  if (path === 'system.details.level.value' || /^system\.skills\.[^.]+\.rank$/.test(path)) return changes;
  if (accepts(path)) changes.push({path, before, after});
  return changes;
}

function effectChanges(before, after) {
  const old = new Map((before ?? []).map(effect => [effect._id, effect]));
  const current = new Map((after ?? []).map(effect => [effect._id, effect]));
  const result = [];
  for (const [id, effect] of current) {
    if (!old.has(id)) result.push({kind: '新增', effect});
    else if (!same(stripVolatile(old.get(id)), stripVolatile(effect))) result.push({kind: '修改', effect});
  }
  for (const [id, effect] of old) if (!current.has(id)) result.push({kind: '删除', effect});
  return result;
}

/** 汇总内容差异；导入预览应传入实际计划写入的文档，不能直接比较来源卡。 */
export function describeChanges(before, after, {actorPathFilter = () => true} = {}) {
  const docs = [before, after].map(doc => ({...doc, items: doc.items ?? []}));
  const comparisonDocs = mediaComparisonDocs(docs);
  const summary = summarize(...docs, {comparisonDocs});
  if (!actorPathFilter('system.details.level.value')) summary.level = null;
  summary.skillRanks = summary.skillRanks.filter(({skill}) => actorPathFilter(`system.skills.${skill}.rank`));
  const fields = fieldChanges(actorContent(before), actorContent(after), actorPathFilter, '', [], comparisonDocs.map(actorContent));
  const effects = actorPathFilter('effects') ? effectChanges(before.effects, after.effects) : [];
  const counts = {
    added: summary.added.length, removed: summary.removed.length, changed: summary.changed.length,
    fields: fields.length + Number(Boolean(summary.level)) + summary.skillRanks.length, effects: effects.length,
  };
  const parts = [
    counts.added && `新增 ${counts.added}`, counts.removed && `删除 ${counts.removed}`,
    counts.changed && `修改 ${counts.changed}`, counts.fields && `角色字段 ${counts.fields}`,
    counts.effects && `效果 ${counts.effects}`,
  ].filter(Boolean);
  const rows = fields.map(change => `<li>${escape(fieldLabel(change.path))}：${displayValue(change.before)} → ${displayValue(change.after)}</li>`);
  rows.push(...effects.map(({kind, effect}) => `<li>${kind}效果「${escape(effect.name ?? effect._id)}」</li>`));
  return {
    html: formatSummary(summary, []) + (rows.length ? `<ul>${rows.join('')}</ul>` : ''),
    text: parts.join(' · '), changed: parts.length > 0, counts,
  };
}
