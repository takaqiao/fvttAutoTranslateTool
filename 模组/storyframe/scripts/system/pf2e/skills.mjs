/**
 * PF2e Skills Configuration
 * Defines all skills with their actions
 */

export const PF2E_SKILLS = {
  per: {
    name: '察觉',
    actions: [
      { slug: 'seek', name: '搜索' },
      { slug: 'sense-direction', name: '辨别方向' },
      { slug: 'sense-motive', name: '察言观色' },
    ],
  },
  acr: {
    name: '特技',
    actions: [
      { slug: 'balance', name: '保持平衡' },
      { slug: 'tumble-through', name: '翻滚' },
      { slug: 'maneuver-in-flight', name: '空中机动' },
      { slug: 'squeeze', name: '挤入' },
    ],
  },
  arc: {
    name: '奥法',
    actions: [
      { slug: 'recall-knowledge', name: '回忆知识' },
      { slug: 'decipher-writing', name: '解译文书' },
      { slug: 'identify-magic', name: '辨识魔法' },
      { slug: 'learn-spell', name: '学习法术' },
    ],
  },
  ath: {
    name: '运动',
    actions: [
      { slug: 'climb', name: '攀爬' },
      { slug: 'force-open', name: '破拆' },
      { slug: 'grapple', name: '擒拿' },
      { slug: 'high-jump', name: '跳高' },
      { slug: 'long-jump', name: '跳远' },
      { slug: 'shove', name: '推撞' },
      { slug: 'swim', name: '游泳' },
      { slug: 'trip', name: '摔绊' },
      { slug: 'disarm', name: '卸武' },
    ],
  },
  cra: {
    name: '手艺',
    actions: [
      { slug: 'recall-knowledge', name: '回忆知识' },
      { slug: 'repair', name: '修理' },
      { slug: 'craft', name: '制造' },
      { slug: 'identify-alchemy', name: '辨识炼金术' },
    ],
  },
  dec: {
    name: '欺骗',
    actions: [
      { slug: 'create-a-diversion', name: '分神' },
      { slug: 'impersonate', name: '乔装' },
      { slug: 'lie', name: '说谎' },
      { slug: 'feint', name: '虚招' },
    ],
  },
  dip: {
    name: '交涉',
    actions: [
      { slug: 'gather-information', name: '搜集信息' },
      { slug: 'make-an-impression', name: '建立印象' },
      { slug: 'request', name: '请求' },
    ],
  },
  itm: {
    name: '威吓',
    actions: [
      { slug: 'coerce', name: '胁迫' },
      { slug: 'demoralize', name: '挫败士气' },
    ],
  },
  med: {
    name: '医疗',
    actions: [
      { slug: 'administer-first-aid', name: '急救' },
      { slug: 'recall-knowledge', name: '回忆知识' },
      { slug: 'treat-disease', name: '治疗疾病' },
      { slug: 'treat-poison', name: '治疗中毒' },
      { slug: 'treat-wounds', name: '治疗伤势' },
    ],
  },
  nat: {
    name: '自然',
    actions: [
      { slug: 'command-an-animal', name: '指挥动物' },
      { slug: 'recall-knowledge', name: '回忆知识' },
      { slug: 'identify-magic', name: '辨识魔法' },
      { slug: 'learn-spell', name: '学习法术' },
    ],
  },
  occ: {
    name: '神秘',
    actions: [
      { slug: 'recall-knowledge', name: '回忆知识' },
      { slug: 'decipher-writing', name: '解译文书' },
      { slug: 'identify-magic', name: '辨识魔法' },
      { slug: 'learn-spell', name: '学习法术' },
    ],
  },
  prf: {
    name: '表演',
    actions: [{ slug: 'perform', name: '表演' }],
  },
  rel: {
    name: '宗教',
    actions: [
      { slug: 'recall-knowledge', name: '回忆知识' },
      { slug: 'decipher-writing', name: '解译文书' },
      { slug: 'identify-magic', name: '辨识魔法' },
      { slug: 'learn-spell', name: '学习法术' },
    ],
  },
  soc: {
    name: '社群',
    actions: [
      { slug: 'recall-knowledge', name: '回忆知识' },
      { slug: 'create-forgery', name: '制作伪造品' },
      { slug: 'decipher-writing', name: '解译文书' },
      { slug: 'subsist', name: '求生' },
    ],
  },
  ste: {
    name: '隐秘',
    actions: [
      { slug: 'conceal-an-object', name: '隐藏物件' },
      { slug: 'hide', name: '躲藏' },
      { slug: 'sneak', name: '潜行' },
    ],
  },
  sur: {
    name: '生存',
    actions: [
      { slug: 'sense-direction', name: '辨别方向' },
      { slug: 'subsist', name: '求生' },
      { slug: 'track', name: '追踪' },
      { slug: 'cover-tracks', name: '掩盖行踪' },
    ],
  },
  thi: {
    name: '贼活',
    actions: [
      { slug: 'palm-an-object', name: '手上功夫' },
      { slug: 'steal', name: '盗窃' },
      { slug: 'pick-a-lock', name: '开锁' },
      { slug: 'disable-device', name: '解除装置' },
    ],
  },
};

/**
 * Additional skills from the sf2e-anachronism module (Starfinder 2e crossover)
 * Only used when game.modules.get('sf2e-anachronism')?.active is true
 */
export const SF2E_SKILLS = {
  com: {
    name: '电脑',
    actions: [
      { slug: 'access-infosphere', name: '接入信息网' },
      { slug: 'decipher-writing', name: '解读文字' },
      { slug: 'disable-device', name: '解除装置' },
      { slug: 'hack', name: '入侵' },
      { slug: 'operate-device', name: '操作装置' },
      { slug: 'recall-knowledge', name: '回忆知识' },
    ],
  },
  pil: {
    name: '驾驶',
    actions: [
      { slug: 'drive', name: '驾驶' },
      { slug: 'navigate', name: '导航' },
      { slug: 'plot-course', name: '规划航线' },
      { slug: 'recall-knowledge', name: '回忆知识' },
      { slug: 'run-over', name: '碾压' },
      { slug: 'stop', name: '停止' },
      { slug: 'stunt', name: '特技' },
      { slug: 'take-control', name: '夺取控制' },
    ],
  },
};

/**
 * Short name abbreviations for PF2e skills
 */
export const PF2E_SKILL_SHORT_NAMES = {
  per: '察觉',
  acr: '特技',
  arc: '奥法',
  ath: '运动',
  cra: '手艺',
  dec: '欺骗',
  dip: '交涉',
  itm: '威吓',
  med: '医疗',
  nat: '自然',
  occ: '神秘',
  prf: '表演',
  rel: '宗教',
  soc: '社群',
  ste: '隐秘',
  sur: '生存',
  thi: '贼活',
  com: '电脑',
  pil: '驾驶',
};

/**
 * Map short skill slugs to full PF2e skill slugs
 */
export const PF2E_SKILL_SLUG_MAP = {
  per: 'perception', // Special case - uses actor.perception not actor.skills
  acr: 'acrobatics',
  arc: 'arcana',
  ath: 'athletics',
  cra: 'crafting',
  dec: 'deception',
  dip: 'diplomacy',
  itm: 'intimidation',
  med: 'medicine',
  nat: 'nature',
  occ: 'occultism',
  prf: 'performance',
  rel: 'religion',
  soc: 'society',
  ste: 'stealth',
  sur: 'survival',
  thi: 'thievery',
  com: 'computers',
  pil: 'piloting',
};

/**
 * Map full skill names to skill slugs (for enricher pattern matching)
 */
export const PF2E_SKILL_NAME_MAP = {
  perception: 'per',
  察觉: 'per',
  acrobatics: 'acr',
  特技: 'acr',
  arcana: 'arc',
  奥法: 'arc',
  athletics: 'ath',
  运动: 'ath',
  crafting: 'cra',
  手艺: 'cra',
  deception: 'dec',
  欺骗: 'dec',
  diplomacy: 'dip',
  交涉: 'dip',
  intimidation: 'itm',
  威吓: 'itm',
  medicine: 'med',
  医疗: 'med',
  nature: 'nat',
  自然: 'nat',
  occultism: 'occ',
  神秘: 'occ',
  performance: 'prf',
  表演: 'prf',
  religion: 'rel',
  宗教: 'rel',
  society: 'soc',
  社群: 'soc',
  stealth: 'ste',
  隐秘: 'ste',
  survival: 'sur',
  生存: 'sur',
  thievery: 'thi',
  贼活: 'thi',
  computers: 'com',
  电脑: 'com',
  piloting: 'pil',
  驾驶: 'pil',
};
