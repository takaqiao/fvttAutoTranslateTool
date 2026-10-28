// 各团的完整规则快照，客户端按 Actor 归属使用。
import { createHash } from 'node:crypto';
import { canonicalJson } from '../scripts/lib/canon.mjs';

/** 决定专长槽、属性提升和熟练加值的 PF2e 设置。 */
export const RULE_KEYS = [
  'pf2e.campaignFeats',
  'pf2e.campaignFeatSections',
  'pf2e.campaignType',
  'pf2e.mythic',
  'pf2e.automaticBonusVariant',
  'pf2e.freeArchetypeVariant',
  'pf2e.gradualBoostsVariant',
  'pf2e.proficiencyVariant',
  'pf2e.proficiencyUntrainedModifier',
  'pf2e.proficiencyTrainedModifier',
  'pf2e.proficiencyExpertModifier',
  'pf2e.proficiencyMasterModifier',
  'pf2e.proficiencyLegendaryModifier',
  'pf2e.staminaVariant',
];

/** 这两个模组的开关整组跟团走，它们才不会在玩家客户端里按建卡服的值去算专长分节。 */
export const RULE_PREFIXES = ['pf2e-team-plus-feats.', 'pf2e-sf2e-extra-feat-slots.'];

/** 团世界未保存的设置使用本版本注册默认值，不借用建卡服的保存值。 */
export const DEFAULT_SENTINEL = Object.freeze({$charsyncDefault: true});

const prefixedKeys = settings => [...settings.keys()].filter(key => RULE_PREFIXES.some(prefix => key.startsWith(prefix)));

/**
 * 每个团世界一张完整表：`RULE_KEYS` 加上建卡服、所有团世界出现过的前缀键。
 * 团世界未保存的键收 `DEFAULT_SENTINEL`，包括建卡服也没有保存的 PF2e 规则。
 * 两个参数都是 `readSettings` 的结果，`worldSettings` 按世界 id。
 */
export function buildRulesTable(charSettings, worldSettings) {
  const table = {};
  const keys = new Set([...RULE_KEYS, ...prefixedKeys(charSettings), ...Object.values(worldSettings).flatMap(prefixedKeys)]);
  for (const [world, settings] of Object.entries(worldSettings)) {
    const rules = {};
    for (const key of keys) rules[key] = settings.has(key) ? settings.get(key) : DEFAULT_SENTINEL;
    table[world] = rules;
  }
  return table;
}

export function buildRuleSnapshot({charSettings, worldSettings, cardWorld, folderWorld, generatedAt, enforcedSince}) {
  const version = 2;
  const rules = buildRulesTable(charSettings, worldSettings);
  const hashes = Object.fromEntries(Object.entries(rules).map(([world, profile]) => [
    world, createHash('sha256').update(canonicalJson({version, rules: profile})).digest('hex'),
  ]));
  return {version, generatedAt, enforcedSince, rules, hashes, cardWorld, folderWorld};
}

/** `generated/rules.mjs` 的全文，形状同仓库里的空表默认值。 */
export function renderRulesModule(snapshot) {
  return [
    '// 同步引擎生成，每次整份重写，不要手改。',
    `export const SNAPSHOT = ${JSON.stringify(snapshot, null, 2)};`,
    'export const RULES = SNAPSHOT.rules;',
    'export const CARD_WORLD = SNAPSHOT.cardWorld;',
    'export const GENERATED_AT = SNAPSHOT.generatedAt;',
    '',
  ].join('\n');
}
