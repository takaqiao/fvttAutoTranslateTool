# -*- coding: utf-8 -*-
"""Build the S3 edit list, assert every `old` is byte-unique, apply to temp
copies and syntax-check the result."""
import json, os, io, sys, shutil, subprocess, tempfile
sys.stdout.reconfigure(encoding='utf-8')

PROJ = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.dirname(os.path.abspath(__file__))

E = []
def edit(sig, file, old, new, why):
    E.append({"sig": sig, "file": file, "old": old, "new": new, "why": why})


# ------------------------------------------------------------------ #
# 1. mappings.mjs — tokenName converter                              #
# ------------------------------------------------------------------ #
edit(
 "api-semantics|Actor.tokenName-uses-babele-name-converter",
 r"3-常用脚本\extract\mappings.mjs",
 "  tokenName: { path: 'prototypeToken.name', converter: 'name' },\n",
 """  /**
   * ⚠ 这里**不能**用 Babele 内建的 `name` 转换器（`{path:'prototypeToken.name',
   * converter:'name'}` 正是 Babele 2.9.1 自己 Actor 默认里写的那一行，本块只是抄了它）。
   *
   * `name` ＝ `Converters.mappedField("name")`
   * （`babele/script/converter/converters.js:92-97`），实现是
   *   `(_value, _translation, data, tc) => tc.translateField("name", data)`
   * —— **第二个参数（也就是译文里的 `tokenName`）被下划线丢弃**，返回的是 `name`
   * 字段的译文。于是 `prototypeToken.name := entry.name`，`tokenName` 这个键
   * 从头到尾没被读过：本项目两仓 **575 条 `tokenName` 译文全是死的**。
   *
   * 后果不止观感。本库约定 `name` 双语并列、`tokenName` 裸中文，而其中 **94 处**
   * 英文基线里 `name` 与 `tokenName` 本来就是两个不同字符串（上游故意用 `Mage`
   * 之类的通名藏身份），套上 `name` 的译文＝把角色全名写到地图 token 上，直接剧透。
   *
   * `crucibleTokenName` 是内建行为的**超集**：有 `tokenName` 译文就用它，没有就退回
   * `translateField("name")`，与内建逐字一致。退路必须留着 —— `registerMapping`
   * 注册的是**全局**层，同世界里第三方合集的 Actor 也会走这一条。
   * （本项目自身缺口为 0：探针实测 en 侧 575 条 `tokenName` 在 cn 侧 575 条全有译文。）
   */
  tokenName: { path: 'prototypeToken.name', converter: 'crucibleTokenName' },
""",
 "babele 2.9.1 的内建 `name` 转换器丢弃译文参数、返回 `name` 字段的译文，"
 "所以 575 条 tokenName 译文全部无效、94 处还会把隐藏身份的 token 名换成角色全名。"
 "改用项目自有转换器（有 tokenName 用 tokenName，没有则退回内建语义）。"
 "抽取方向不变：extract_en.mjs 对 `name` 与 `crucibleTokenName` 读的是同一个 spec.path，英文基线一字不动。",
)

# ------------------------------------------------------------------ #
# 2. mappings.mjs — encounter token names field                      #
# ------------------------------------------------------------------ #
edit(
 "unmapped-field|JournalEntryPage.system.encounter.tokens[].actors[].tokenData.name",
 r"3-常用脚本\extract\mappings.mjs",
 "const P = (extra) => ({ ...PAGE_BASE, ...extra });\n",
 """/**
 * 遭遇模板里**预置 token 的覆盖名**：
 * `system.encounter.tokens[].actors[].tokenData.name`。
 *
 * 实测（probes/s3_encounter_shape.mjs，直接读 LevelDB）：`ember.crucible-adventure`
 * 与 `ember.adventure` **各 382 处 / 130 个唯一串**，合计 764 处；全部落在
 * `ember.questEvent`(229 页) 与 `ember.standaloneEvent`(18 页) 两个子类型上，
 * 别的子类型一个 `system.encounter` 都没有。
 *
 * 为什么此前完全不可见：`extract_en.mjs` 是**解释本文件**来生成英文基线的，
 * 映射没列的字段就不进基线，于是覆盖率 / 缺键 / 死键 / tokenName 四道闸的定义域里
 * 压根没有它 —— 与 `Scene.levels` 那次是同一类盲区。
 *
 * 为什么玩家一定会看见：`ember.mjs` 的 `_spawnEncounterTokens` 对每个
 * `{actor, tokenData}` 调 `actor.getTokenDocument(tokenData)`，`tokenData.name`
 * 会**覆盖**已被 Babele 译好的 `prototypeToken.name`；130 个名字里 128 个与所链
 * actor 的名字不同（`Friendly Ooze`↔`Oozeling`、`Beacon Brigade Patroller`↔
 * `Wandren Patroller`…），所以「actor 名译了 token 就跟着变」在这里必然不成立。
 *
 * 为什么要自定义转换器：`tokens[] -> actors[] -> tokenData.name` 是**两层嵌套数组**，
 * Babele 内建的 `structured`（只认一层 + 一个 key 字段）与 `nameCollection`
 * （只认 `collection[].name`）都表达不了。译文按**英文名本身**建键，与
 * `nameCollection` 同构，查不到就原样返回。
 *
 * 同一 blob 内的 `tokenData.delta.effects[].name`（Dead/Unconscious/Prone/
 * Restrained/Sleeping/Bloodied，各包 16 处 / 6 唯一）**有意不纳入**：它们是状态名，
 * 与 token 名共用一张扁平表会有撞键风险，且数量与可见度都低一个量级。
 */
const ENCOUNTER_TOKENS_FIELD = {
  path: 'system.encounter.tokens',
  converter: 'emberEncounterTokenNames',
};

const P = (extra) => ({ ...PAGE_BASE, ...extra });
""",
 "补上 `system.encounter.tokens[].actors[].tokenData.name` 的字段声明（764 处 / 130 唯一串从未进过管线）。"
 "只加常量定义，实际挂载在下一条编辑里。落地后主控需重跑 extract_en.mjs 生成英文基线，"
 "再翻这 130 个串（约定同 tokenName：**裸中文**，不带英文尾巴）。",
)

edit(
 "unmapped-field|JournalEntryPage.ember.questEvent system.encounter.tokens[].actors[].tokenData.name",
 r"3-常用脚本\extract\mappings.mjs",
 """  'JournalEntryPage.ember.questEvent': P({
    overview: 'system.overview',
    exposition: 'system.exposition',
    summary: 'system.summary',
    outcomes: OUTCOMES_FIELD,
  }),
  'JournalEntryPage.ember.standaloneEvent': P({
    overview: 'system.overview',
    exposition: 'system.exposition',
    summary: 'system.summary',
    outcomes: OUTCOMES_FIELD,
  }),
""",
 """  'JournalEntryPage.ember.questEvent': P({
    overview: 'system.overview',
    exposition: 'system.exposition',
    summary: 'system.summary',
    outcomes: OUTCOMES_FIELD,
    encounterTokens: ENCOUNTER_TOKENS_FIELD,
  }),
  'JournalEntryPage.ember.standaloneEvent': P({
    overview: 'system.overview',
    exposition: 'system.exposition',
    summary: 'system.summary',
    outcomes: OUTCOMES_FIELD,
    encounterTokens: ENCOUNTER_TOKENS_FIELD,
  }),
""",
 "把 ENCOUNTER_TOKENS_FIELD 挂到唯一两个带 `system.encounter` 的页面子类型上"
 "（探针实测 247 个带 encounter 的页面 = questEvent 229 + standaloneEvent 18，无第三种）。",
)

# ------------------------------------------------------------------ #
# 3. runtime-converters.js — crucibleDescription dnd5e shape         #
# ------------------------------------------------------------------ #
edit(
 "global-registry-write-scoped-by-surface|babele.registerMapping:Item.description+Actor.items",
 r"3-常用脚本\release\runtime-converters.js",
 """  if (isStr(translation)) {
    return foundry.utils.mergeObject(value, { public: translation }, { inplace: false });
  }
""",
 """  // ⚠ dnd5e / 通用形状 `{value, chat}` 必须单独处理，否则会**弄坏别人的翻译**。
  //
  // `babele.registerMapping()` 注册的是**全局**层（`core/babele.js:231` 只是
  // `registeredMappings.push(...)`，没有任何 module/pack 维度的作用域参数；
  // `mapping/document-mappings.js:267-287` 把 built-in→registered→loaded 合成一份
  // effectiveMappings，每个被 Babele 接管的合集都取这一份）。也就是说本项目的
  // `Item.description` 会**按键顶掉** Babele 内建的 `"system.description.value"`，
  // 对同世界里第三方汉化包（例如 dnd-simplified-chinese-babele-patch，它的
  // rules/mapping.json 没有 description 键、完全依赖内建默认）的 Item 一样生效。
  //
  // 少了这一支，`{value, chat}` 会掉进下面的 `{public: …}` 分支：`.value` 仍是英文，
  // 另外塞进一个 dnd5e schema 里根本没有的 `public` 键。这里按内建语义写回 `.value`。
  //
  // crucible 侧不会误入：它唯一的对象形态 description 是
  // `SchemaField{public, private}`（`module/models/item-physical.mjs:26-29`），
  // 没有 `value` 子字段。
  //
  // 只改 translate 方向，**extract 方向刻意保持原样** —— 让 extract 也吐
  // `{value, chat}` 会把 dnd5e 侧约 85 万字符拉进英文基线，那是项目所有者已定
  // 「先不管」的 Z1，属于另一件事。
  if (typeof value.value === 'string' && !('public' in value) && !('private' in value)) {
    const text = isStr(translation) ? translation : (isStr(translation.value) ? translation.value : null);
    return text === null ? value : foundry.utils.mergeObject(value, { value: text }, { inplace: false });
  }

  if (isStr(translation)) {
    return foundry.utils.mergeObject(value, { public: translation }, { inplace: false });
  }
""",
 "全局 mapping 层把 `Item.description` 改成 crucibleDescription，会让第三方 dnd5e 汉化包的"
 "`{value, chat}` 描述落空并被注入垃圾键 `public`。babele 不提供按包作用域的注册（已读 2.9.1 源码确认），"
 "所以按形状恢复内建语义是唯一不越界的修法。"
 "同 finding 的另一半 `Actor.items.fallbackPolicy` 未改，理由见 skipped。",
)

# ------------------------------------------------------------------ #
# 4. runtime-converters.js — two new converters                      #
# ------------------------------------------------------------------ #
edit(
 "api-semantics|Actor.tokenName-uses-babele-name-converter#converter",
 r"3-常用脚本\release\runtime-converters.js",
 """export const PROJECT_CONVERTERS = {
  crucibleDescription,
  crucibleNested,
  crucibleActions,
};
""",
 """/**
 * `prototypeToken.name` —— 地图 token 上显示的名字。
 *
 * Babele 内建 Actor 默认把 `tokenName` 写成
 * `{path:'prototypeToken.name', converter:'name'}`，而 `name` ＝
 * `Converters.mappedField("name")`：
 *   `(_value, _translation, data, tc) => tc.translateField("name", data)`
 * 译文参数被丢弃，返回的是 **`name` 字段**的译文 —— `tokenName` 键从来没被读过。
 *
 * 这里做成内建行为的**超集**：有 `tokenName` 译文就用它，没有就原样退回
 * `translateField("name")`。退路必须留着，因为 `registerMapping` 是全局层，
 * 第三方合集的 Actor 也会走这一条，不能让它们的 token 名退回英文。
 *
 * 参数顺序即 Babele 的函数式转换器签名（见文件头）；`contextCompendium` 的取法
 * 与内建 `mappedField` 逐字一致（`tc ?? runtime?.currentCompendium?.()`）。
 */
export function crucibleTokenName(value, translation, source, contextCompendium, _allTranslations, runtime = {}) {
  if (isStr(translation)) return translation;
  const pack = contextCompendium ?? runtime?.currentCompendium?.() ?? null;
  const fallback = pack?.translateField?.('name', source, runtime);
  return isStr(fallback) ? fallback : value;
}

crucibleTokenName.extract = (value) => (isStr(value) ? value : undefined);

/**
 * Ember 遭遇模板里预置 token 的覆盖名：
 * `system.encounter.tokens[].actors[].tokenData.name`（两层嵌套数组）。
 *
 * 译文按**英文名本身**建键（与内建 `nameCollection` 同构），查不到就原样返回。
 * 写回用 `mergeObject(actor, {tokenData: {name}})` —— `tokenData` 两边都是对象，
 * Foundry 的 mergeObject 会**递归**进去只覆盖 `name`，
 * `_id`/`texture`/`x`/`y`/`elevation`/`rotation`/`flags`/`delta`/`disposition`
 * 等 15 个兄弟键一个不动（已用真实 pack 数据逐条回归验证）。
 */
export function emberEncounterTokenNames(tokens, translation) {
  if (!Array.isArray(tokens) || !translation || typeof translation !== 'object') return tokens;

  let changed = false;
  const out = tokens.map((token) => {
    if (!token || typeof token !== 'object' || !Array.isArray(token.actors)) return token;

    let hit = false;
    const actors = token.actors.map((actor) => {
      const name = actor?.tokenData?.name;
      if (!isStr(name)) return actor;
      const cn = translation[name];
      if (!isStr(cn) || cn === name) return actor;
      hit = true;
      return foundry.utils.mergeObject(actor, { tokenData: { name: cn } }, { inplace: false });
    });

    if (!hit) return token;
    changed = true;
    return foundry.utils.mergeObject(token, { actors }, { inplace: false });
  });

  return changed ? out : tokens;
}

emberEncounterTokenNames.extract = (tokens) => {
  if (!Array.isArray(tokens)) return undefined;
  const out = {};
  for (const token of tokens) {
    for (const actor of (token?.actors ?? [])) {
      const name = actor?.tokenData?.name;
      if (isStr(name) && !(name in out)) out[name] = name;
    }
  }
  return Object.keys(out).length ? out : undefined;
};

export const PROJECT_CONVERTERS = {
  crucibleDescription,
  crucibleNested,
  crucibleActions,
  crucibleTokenName,
  emberEncounterTokenNames,
};
""",
 "新增两个转换器并登记进 PROJECT_CONVERTERS（`babele.registerConverters` 直接吃这个对象）。"
 "已用真实 LevelDB 数据跑过 extract→translate 回归："
 "两包各 382 处 tokenData.name 全部落地、源对象未被就地修改、tokenData 的兄弟键全部保留。",
)

# ------------------------------------------------------------------ #
# 5. extract_en.mjs —两个新 case                                      #
# ------------------------------------------------------------------ #
edit(
 "api-semantics|Actor.tokenName-uses-babele-name-converter#extract",
 r"3-常用脚本\extract\extract_en.mjs",
 """      case 'name': {
        if (isNonEmptyString(value)) out[field] = value;
        break;
      }
""",
 """      // `crucibleTokenName` 在**抽取方向**上与内建 `name` 完全一样：都读 spec.path
      // （`prototypeToken.name`）的原值当英文基线。差别只在运行时，见
      // mappings.mjs 里 `CRUCIBLE_ACTOR.tokenName` 的注释。所以两个 case 共用一段 ——
      // 换转换器**不会**让英文基线产生任何 diff。
      case 'name':
      case 'crucibleTokenName': {
        if (isNonEmptyString(value)) out[field] = value;
        break;
      }
""",
 "抽取器按转换器名分派；换名之后必须认新名，否则会掉进 default 分支（`value` 是字符串，"
 "结果碰巧一样，但那是巧合不是契约）。显式共用一段并写明「基线零 diff」。",
)

edit(
 "unmapped-field|JournalEntryPage.system.encounter.tokens[].actors[].tokenData.name#extract",
 r"3-常用脚本\extract\extract_en.mjs",
 "      case 'structured': {\n",
 """      case 'emberEncounterTokenNames': {
        // `system.encounter.tokens[].actors[].tokenData.name`：两层嵌套数组。
        // 键取英文名本身（与 nameCollection 同构）；两个冒险包各 382 处 / 130 唯一串。
        const map = {};
        for (const token of toArray(value)) {
          for (const actor of toArray(token?.actors)) {
            const n = actor?.tokenData?.name;
            if (isNonEmptyString(n)) map[n] ??= n;
          }
        }
        if (Object.keys(map).length) out[field] = map;
        break;
      }

      case 'structured': {
""",
 "抽取器要能产出 `encounterTokens` 的英文基线，否则新映射在运行时查一个永远不存在的键"
 "（与 AUDIT-2026-08-12 §2.2 的 ActiveEffect 死键同一个坑）。"
 "抽取逻辑与运行时 `emberEncounterTokenNames.extract` 逐字同构。",
)

# ------------------------------------------------------------------ #
# 6. babele-register.js — talent enricher prefix                     #
# ------------------------------------------------------------------ #
edit(
 "enricher-prefix-uncovered|crucible.enrichTalent",
 r"2-Crucible汉化插件\babele-register.js",
 """/**
 * Foundry's core `Sort` label collides with Crucible's own use of the key.
""",
 """/**
 * `[[/talent …]]` 增强器的 `Talent: ` 前缀 —— Crucible 自己漏掉的一处 i18n。
 *
 * `module/enrichers.mjs:739` 写的是
 *   tag.innerHTML = `Talent: ${talentIndex.name}`;
 * 而**紧邻的两个同类增强器**用的都是 i18n 键：
 *   :722 enrichKnowledge -> _loc("ACTOR.KnowledgeSpecific")   // 知识：{knowledge}
 *   :757 enrichLanguage  -> _loc("ACTOR.LanguageSpecific")    // 语言：{language}
 * 只有 talent 这一条把前缀写死了 —— 是上游自己的不一致，不是配置问题。
 *
 * 两条汉化通道都够不到：Babele 只管合集文档（`talentIndex.name` 确实已经是中文，
 * 所以玩家看到的是「Talent: 识别法术」这种半英半中）；i18n 那边**根本没有键可翻**。
 * 只能在运行时包一层增强器。
 *
 * 为什么放在 crucible-cn 而不是 ember 侧：这个串是 **Crucible 系统**产的，
 * 纯 Crucible 世界（没装 Ember）照样会露；放这里两种世界都覆盖到。
 * 计数：`ember.crucible-adventure` 68 处、dnd5e 孪生包 67 处。
 *
 * 做法刻意保守：
 *  - 按 `id === "crucibleTalent"` **精确定位**（上游给每个增强器都写了 id），
 *    不用正则去猜哪个增强器是谁的；
 *  - 只在结果确实以 `Talent: ` 开头时改，且只动前缀，`talentIndex.name` 原样保留；
 *  - `dataset.talentUuid` / class / `dataset.crucibleTooltip` 一概不碰 ——
 *    点击与悬浮逻辑全靠它们；
 *  - 包在 try/catch 里，失败只留一条警告，绝不影响开世界。
 *
 * 时机：`registerEnrichers()` 在 Crucible 自己的 `Hooks.once("init")` 里跑
 * （crucible-compiled.mjs:47439），所以补丁挂 `setup`（i18nInit 早于 init，太早）。
 * 返回值改成 async 是安全的：Foundry v14
 * `client/applications/ux/text-editor.mjs:267` 就是 `await enricher(match, options)`。
 */
const TALENT_PREFIX_EN = 'Talent: ';
const TALENT_PREFIX_CN = '天赋：';

function patchTalentEnricher() {
  const enrichers = CONFIG.TextEditor?.enrichers;
  if (!Array.isArray(enrichers)) return;

  const entry = enrichers.find((e) => e?.id === 'crucibleTalent');
  if (!entry || typeof entry.enricher !== 'function' || entry.__crucibleCnWrapped) return;

  const original = entry.enricher;
  entry.enricher = async function wrappedTalentEnricher(...args) {
    const result = await original.apply(this, args);
    try {
      const text = result instanceof HTMLElement ? result.textContent : null;
      if (typeof text === 'string' && text.startsWith(TALENT_PREFIX_EN)) {
        result.textContent = TALENT_PREFIX_CN + text.slice(TALENT_PREFIX_EN.length);
      }
    } catch (err) {
      console.warn('Crucible cn | 天赋增强器前缀改写失败：', err);
    }
    return result;
  };
  entry.__crucibleCnWrapped = true;
}

Hooks.once('setup', () => {
  try {
    patchTalentEnricher();
  } catch (err) {
    console.warn('Crucible cn | 天赋增强器补丁未生效：', err);
  }
});

/**
 * Foundry's core `Sort` label collides with Crucible's own use of the key.
""",
 "上游 enrichTalent 把 `Talent: ` 前缀写死（已读 crucible 0.10.1 源码确认，"
 "同文件相邻的 knowledge/language 两条都走 _loc）。babele 与 i18n 都够不到，"
 "只能运行时包增强器。按 id 精确定位、只改前缀、不碰 dataset/class。"
 "放 crucible-cn 而非 ember 插件，是因为串由系统产出，纯 Crucible 世界同样会露。",
)

# ------------------------------------------------------------------ #
# 7. babele-register.js — 7 keyless boon/bane labels                 #
# ------------------------------------------------------------------ #
edit(
 "i18n-keyless-literal|crucible boon/bane label slot",
 r"2-Crucible汉化插件\babele-register.js",
 """Hooks.once('i18nInit', () => {
  game.i18n.translations.Sort = 'Sort';
  game.i18n.translations.sort = '排序';
});
""",
 """/**
 * 恩惠 / 祸骰明细的 `label` 槽里有 **7 个裸英文串**走 `{{localize}}` 却没有 i18n 键。
 *
 * 写点（crucible 0.10.1，逐行核对过）：
 *   module/documents/combatant.mjs:27  {label: "Reserved Action"}
 *   module/documents/combatant.mjs:29  {label: "Slow Weaponry"}
 *   module/documents/combatant.mjs:30  {label: "Broken"}
 *   module/documents/combatant.mjs:31  {label: "Bulky Armor"}
 *   module/documents/combatant.mjs:36  {label: "Elite"}
 *   module/documents/combatant.mjs:37  {label: "Boss"}
 *   module/dice/standard-check-dialog.mjs:500 · module/dice/standard-check.mjs:189/:192 ·
 *   module/documents/actor.mjs:578/:579      {label: "Special"}   （共 5 个写点）
 * 渲染点：templates/dice/partials/standard-check-details.hbs:6/:19 的
 *   `{{localize boon.label}}` / `{{localize bane.label}}`
 * 该 partial 被掷骰对话框、检定聊天卡的骰子明细、动作使用页脚、危害对话框四处 include，
 * 所以**每一次检定**都会走到；combatant 那六条出现在先攻明细里，每场战斗开场就露。
 *
 * 上游同一个 `label` 槽的另外 ~37 个赋值点写的全是正规 i18n 键
 * （`ACTION.TAG.Difficult`…）或已本地化的文档名 —— 最直接的铁证是 **broken 被写了两遍**：
 * `actor.mjs:383` 用 `statuses.broken.name`（正确），`combatant.mjs:30` 却写死 `"Broken"`。
 *
 * 为什么本项目此前看不见：lang 的覆盖模型是「按 en.json 键清单逐键翻」，这 7 条
 * **不在 en.json 里**，于是「cn 键数 == en 键数」这个绿灯永远不会亮红。
 * 注意 crucible lang/en.json 里那个 `"Special"` 是**嵌套在分节对象里**的，
 * 而 `localize()` 走 `getProperty(translations, "Special")` 是顶层查找，查不到。
 *
 * 为什么写在这里而不是 `lang/cn.json`：这 7 个是**无点号的顶层键**，塞进 cn.json 会
 * 打破发版前 `flatten_lang.py` 的「拍平前 == 拍平后 == 英文键数」三数相等（现为 1842），
 * 而它们本来就不属于 en.json 的键空间。`Sort` 那两条也是同一理由写在这里的。
 *
 * 越界风险已实测：扫过 Foundry v14 本体 + 本机全部 systems/modules 的 101 个 lang 文件与
 * 3308 个 js/mjs/hbs/html，这 7 个串**既没有任何包把它们当顶层键定义过，也没有任何
 * `localize("…")` 字面调用** —— 冲突面为 0。即便如此仍加了「已存在就不覆盖」的守卫。
 *
 * 译名一律取本仓 lang/cn.json 里同概念的既有译法，不另起炉灶：
 *   Special ← ACTION.TAG_CATEGORIES.Special「特殊」
 *   Broken  ← ACTIVE_EFFECT.STATUSES.Broken「破碎」（必须与 actor.mjs:383 那条同名）
 *   Elite / Boss ← ACTOR.ADVERSARY.THREAT_RANKS.*「精英」「首领」
 *   Bulky Armor ← ARMOR.PROPERTIES.Bulky「笨重」＋ DEFENSES.Armor「护甲」
 *   Slow Weaponry：上游取的是 `weapons.slow`＝装备了 Oversized 武器（actor-base.mjs:736），
 *                  显示串是 "Slow Weaponry"，故译「迟缓武器」
 *   Reserved Action ← RESOURCES.Action「动作」，指回合结束时未花掉的动作点
 */
const BOON_BANE_LABELS = {
  'Special': '特殊',
  'Reserved Action': '保留动作',
  'Slow Weaponry': '迟缓武器',
  'Broken': '破碎',
  'Bulky Armor': '笨重护甲',
  'Elite': '精英',
  'Boss': '首领',
};

Hooks.once('i18nInit', () => {
  game.i18n.translations.Sort = 'Sort';
  game.i18n.translations.sort = '排序';

  for (const [key, value] of Object.entries(BOON_BANE_LABELS)) {
    // 顶层键是全局的：别人已经定义过就让给别人，宁可露英文也不顶掉。
    if (typeof game.i18n.translations[key] === 'string') continue;
    game.i18n.translations[key] = value;
  }
});
""",
 "7 个上游写死的裸串走 {{localize}} 但没有键，每张检定聊天卡与掷骰对话框都露英文。"
 "写进 i18nInit 而不是 lang/cn.json，是为了不破坏发版前 flatten_lang.py 的三数相等（1842）。"
 "已扫全机 101 个 lang 文件 + 3308 个代码文件确认零冲突，并仍加了不覆盖守卫。",
)

# ------------------------------------------------------------------ #
# 8. module.json — relationships.systems                             #
# ------------------------------------------------------------------ #
edit(
 "manifest_dep_unenforced|2-Crucible汉化插件/module.json:relationships.requires[crucible]",
 r"2-Crucible汉化插件\module.json",
 """  "relationships": {
    "requires": [
      {
        "id": "babele",
        "type": "module",
        "compatibility": {
          "minimum": "2.9.1"
        }
      },
      {
        "id": "crucible",
        "type": "system",
        "compatibility": {
          "minimum": "0.10.1",
          "verified": "0.10.1"
        }
      }
    ]
  },
""",
 """  "relationships": {
    "requires": [
      {
        "id": "babele",
        "type": "module",
        "compatibility": {
          "minimum": "2.9.1"
        }
      }
    ],
    "systems": [
      {
        "id": "crucible",
        "type": "system",
        "compatibility": {
          "minimum": "0.10.1",
          "verified": "0.10.1"
        }
      }
    ]
  },
""",
 "Foundry v14.365 读 relationships.requires 的四条路径全部第一句就跳过非 module 条目"
 "（common/packages/base-package.mjs:481、module-management.mjs:113 与 :490、"
 "client-package.mjs:168，均已回源逐行确认），所以原来那条 crucible 声明一处都不生效。"
 "真正校验系统版本的是 relationships.systems（base-package.mjs:517 _testSupportedSystems + "
 "module-management.mjs:159 #evaluateSystemCompatibility）。ember 0.6.0 与 5e_chn 用的都是这个字段。",
)

edit(
 "manifest_dep_unenforced|2-Crucible汉化插件/module.json:compatibility.minimum",
 r"2-Crucible汉化插件\module.json",
 """  "compatibility": {
    "minimum": "13",
    "verified": "14",
    "maximum": "14.999"
  },
""",
 """  "compatibility": {
    "minimum": "14",
    "verified": "14",
    "maximum": "14.999"
  },
""",
 "⚠ 用户可见的安装闸变更。原来 minimum 写 13、同一份 manifest 又声明依赖 crucible >= 0.10.1，"
 "而 crucible 0.10.1 的 system.json 写着 compatibility.minimum = 14.364 —— 这个组合在 v13 上不可能成立。"
 "上一条编辑把系统依赖改成真正生效的 relationships.systems 之后，v13 环境只会得到「已安装但不可启用」，"
 "继续广告 v13 是纯误导。对齐到 14（与姊妹仓 ember_cn 1.1.9 一致）。发布说明的同一处矛盾在后续编辑里一并改。",
)

# ------------------------------------------------------------------ #
# 9. release-body-template.md                                        #
# ------------------------------------------------------------------ #
edit(
 "doc_false_claim|2-Crucible汉化插件/.github/release-body-template.md:15-17",
 r"2-Crucible汉化插件\.github\release-body-template.md",
 """- Foundry VTT v13 ~ v14
- Crucible 系统 **v0.10.1+**
- [Babele](https://foundryvtt.com/packages/babele) **v2.9.1+**

> 这三项由 `module.json` 的 `relationships.requires` 强制校验：版本不满足时
> Foundry 会直接拒绝启用本模块，而不是只给个警告。请先升级系统与 Babele 再装本模块。
> Enforced by Foundry via `relationships.requires` — older versions are refused, not warned.
""",
 """- Foundry VTT **v14**（Crucible 0.10.1 自身要求核心 ≥ 14.364）
- Crucible 系统 **v0.10.1+**
- [Babele](https://foundryvtt.com/packages/babele) **v2.9.1+**

> 这三项都由 Foundry 强制校验，版本不满足时直接拒绝启用，而不是只给个警告：
> 核心版本看 `module.json` 的 `compatibility`，Babele 看 `relationships.requires`，
> Crucible 系统看 `relationships.systems`。请先升级系统与 Babele 再装本模块。
> Enforced by Foundry: core version via `compatibility`, Babele via
> `relationships.requires`, and the Crucible system via `relationships.systems`.
""",
 "原文用「这三项」把 Crucible 系统一并担保给 relationships.requires，而 Foundry 对非 module 条目"
 "一处都不校验 —— 这句话已随 0.9.1~0.9.6 六次发布印给用户。同段「v13 ~ v14」与「Crucible v0.10.1+」"
 "也不可能同时成立（crucible 0.10.1 要求核心 ≥ 14.364）。两处一并订正，并按字段分别说明由谁校验。",
)

# ------------------------------------------------------------------ #
# 10. README.md — 同一句假声明                                        #
# ------------------------------------------------------------------ #
edit(
 "doc_false_claim|2-Crucible汉化插件/README.md:35-45",
 r"2-Crucible汉化插件\README.md",
 """- Foundry VTT v13 ~ v14
- Crucible 系统 **v0.10.1+**
- [Babele](https://foundryvtt.com/packages/babele) **v2.9.1+**

> 这三项写在 `module.json` 的 `relationships.requires` 里，由 Foundry 强制校验：
> 版本低于上述要求时，Foundry 会**直接拒绝启用**本模块，而不是只给个警告。
""",
 """- Foundry VTT **v14**（Crucible 0.10.1 自身要求核心 ≥ 14.364）
- Crucible 系统 **v0.10.1+**
- [Babele](https://foundryvtt.com/packages/babele) **v2.9.1+**

> 三项都由 Foundry 强制校验（版本不满足时**直接拒绝启用**，而不是只给个警告），
> 但**写在不同字段里**：核心版本看 `module.json` 的 `compatibility`，
> Babele 看 `relationships.requires`，Crucible 系统看 `relationships.systems`。
> ⚠ 非 `type: "module"` 的条目放进 `relationships.requires` 是**完全不生效**的 ——
> Foundry v14 里读它的四条路径全部第一句就 `if (type !== "module") continue`。
""",
 "README 与发布模板印着同一句假声明，一起订正；并把「requires 只认 module」这条踩过的坑写在文档里，"
 "免得下次又把系统依赖放回 requires。",
)

edit(
 "unscoped-batch-rewrite|repair_bilingual_names.py:README-row",
 r"2-Crucible汉化插件\README.md",
 "| `repair_bilingual_names.py` | 修正中英混排的实体名 |\n",
 "| `repair_bilingual_names.py` | **只读报告**：列出疑似中英混排的实体名。"
 "判据是纯形状的，本库实测 652 条建议**没有一条可以直接采用**，"
 "所以它不写盘；确认要改的走 `3-常用脚本/qa/apply_translations.py` |\n",
 "README 把一个会静默毁数据的脚本推荐成「修正」工具。改成如实描述（只读报告 + 已知误报率）。",
)

# ------------------------------------------------------------------ #
# 11. repair_bilingual_names.py — report only                        #
# ------------------------------------------------------------------ #
edit(
 "unscoped-batch-rewrite|repair_bilingual_names.py:is_broken",
 r"2-Crucible汉化插件\scripts\repair_bilingual_names.py",
 '''"""For every dict node in CN packs that has a `name` field, look up the
parallel node in EN and ensure CN name follows `{CJK chunk} {EN name}`.
If current CN name is broken (any English token in it that isn't equal to
the EN trailing portion), rebuild it.
"""
''',
 '''"""⚠ 只读报告工具，**不写盘**。原本它是直接改 `compendium/cn` 的，已改掉，原因如下。

它做的事：遍历 CN 包里每一个带 `name` 的节点，跟 EN 侧同路径对照，
按「双语并列 `中文 英文`」的形状判断这个名字坏没坏，并给出一个重建建议。

为什么不再让它写盘（空跑实测，只读）
------------------------------------
判据 (`is_broken`) 与重建 (`rebuild`) 都是**纯形状**的 —— 只看当前字符串里有没有
拉丁字母、有没有空格、结不结尾于英文名，完全不看这个名字**是什么**。结果：

* 本仓 3 条建议里 1 条丢字：
  `试玩测试 1 - 英勇之戒 Playtest 1 - The Ring of Valor`
  -> `试玩测试 Playtest 1 - The Ring of Valor`（中文侧「1 - 英勇之戒」整段没了）。
* 把同一判据指向 ember_cn：**649 条建议里 403 条重建后比原来短**，即丢字。
  典型 `补丁 0.4.7 Patch 0.4.7` -> `补丁 Patch 0.4.7`（版本号被吃掉，同型 20 余条）；
  另有 6 条把英文名拼了第二遍，如 `V'玛尔 V'Mar` -> `V'玛尔 V'Mar V'Mar`
  （中文名以拉丁字母 V 开头，`first_cjk_chunk` 第一个字符就停）。
* 把 `rebuild` 换成「保留整段中文头」之后，剩下的 34 条建议**依然全错**：
  它们只是「中文没有以英文名结尾」，如 `1. 洞悉加值 Insight Bonus` 对英文
  `1. Insight Bonus`，建议是把整串英文再拼一遍。
  也就是说 **652 条建议（本仓 3 + ember 侧 649）里真阳性为 0**。

所以它现在只报不改。真要落地的改动一律走
`3-常用脚本/qa/apply_translations.py`（英文源漂移 / 无中文 / 标记破损三道闸），
这也是 PROJECT.md 的硬规矩：**任何情况下都不要直接改 `compendium/cn`**。
"""
''',
 "脚本无参数、无 dry-run，敲下去立即覆盖 compendium/cn，而 README 正在推荐它；"
 "空跑实测 652 条建议真阳性为 0（404 条重建后丢字、6 条把英文名拼第二遍；把重建换成「保留整段中文头」之后剩下的 34 条逐条看依然全错）。"
 "改成只读报告，并把测量结果写进文件头，免得下次有人再把写盘加回来。"
 "同时它直接改 compendium/cn 本身就违反 PROJECT.md 的硬规矩。",
)

edit(
 "unscoped-batch-rewrite|repair_bilingual_names.py:no-mutate",
 r"2-Crucible汉化插件\scripts\repair_bilingual_names.py",
 """            if is_broken(cn['name'], en['name']):
                new = rebuild(cn['name'], en['name'])
                fixed.append((cn['name'], new))
                cn['name'] = new
""",
 """            if is_broken(cn['name'], en['name']):
                # 只记录不改写：本脚本是只读报告，见文件头。
                fixed.append((cn['name'], rebuild(cn['name'], en['name'])))
""",
 "去掉就地改写，让「只读」在结构上成立，而不是靠调用方自觉。",
)

edit(
 "unscoped-batch-rewrite|repair_bilingual_names.py:no-write",
 r"2-Crucible汉化插件\scripts\repair_bilingual_names.py",
 """        fixed = []
        walk_pair(cn, en, fixed)
        if fixed:
            with open(cn_p, 'w', encoding='utf-8') as f:
                json.dump(cn, f, ensure_ascii=False, indent=2)
            print(f'== {fn}: {len(fixed)} fixed')
            for o, n in fixed[:20]:
                print(f'  {o!r} -> {n!r}')
            if len(fixed) > 20:
                print(f'  ... +{len(fixed)-20}')
            total += len(fixed)
    print(f'\\nTotal: {total}')
""",
 """        fixed = []
        walk_pair(cn, en, fixed)
        if fixed:
            print(f'== {fn}: {len(fixed)} suspicious')
            for o, n in fixed[:20]:
                flag = '   <== 建议值比原值短，会丢字' if len(n) < len(o) else ''
                print(f'  {o!r}\\n    建议 {n!r}{flag}')
            if len(fixed) > 20:
                print(f'  ... +{len(fixed)-20}')
            total += len(fixed)
    print(f'\\nTotal: {total} suspicious name(s).')
    print('本脚本不写盘。判据是纯形状的，误报率极高（见文件头的实测数据）——')
    print('逐条人看之后，要落地的走 3-常用脚本/qa/apply_translations.py。')
""",
 "删掉 json.dump 写盘路径，改为报告；对「建议值比原值短」的那一档单独标注，"
 "因为那正是会丢字的一档（ember 侧 403/649）。",
)


# ------------------------------------------------------------------ #
# validate                                                           #
# ------------------------------------------------------------------ #
tmp = os.path.join(OUT, 'apply')
if os.path.isdir(tmp):
    shutil.rmtree(tmp)
os.makedirs(tmp)

by_file = {}
for e in E:
    by_file.setdefault(e['file'], []).append(e)

ok = True
for rel, edits in by_file.items():
    src = os.path.join(PROJ, rel)
    raw = io.open(src, encoding='utf-8', newline='').read()
    crlf = '\r\n' in raw
    text = raw.replace('\r\n', '\n')  # universal-newline form, as a normal text read gives
    for e in edits:
        n = text.count(e['old'])
        print(f"[{n:>2}] {'CRLF' if crlf else 'LF  '} {rel} :: {e['sig']}")
        if n != 1:
            ok = False
            print('      !!! NOT UNIQUE / NOT FOUND')
            continue
        text = text.replace(e['old'], e['new'], 1)
    dst = os.path.join(tmp, os.path.basename(rel))
    io.open(dst, 'w', encoding='utf-8', newline='').write(text)

print('\n-- syntax checks --')
for rel in by_file:
    dst = os.path.join(tmp, os.path.basename(rel))
    if rel.endswith(('.js', '.mjs')):
        r = subprocess.run(['node', '--check', dst], capture_output=True, text=True)
        print(('OK   ' if r.returncode == 0 else 'FAIL ') + rel, r.stderr.strip()[:400])
        ok &= r.returncode == 0
    elif rel.endswith('.json'):
        try:
            json.load(io.open(dst, encoding='utf-8')); print('OK   ' + rel)
        except Exception as ex:
            print('FAIL ' + rel, ex); ok = False
    elif rel.endswith('.py'):
        r = subprocess.run([sys.executable, '-m', 'py_compile', dst], capture_output=True, text=True)
        print(('OK   ' if r.returncode == 0 else 'FAIL ') + rel, r.stderr.strip()[:400])
        ok &= r.returncode == 0
    else:
        print('---  ' + rel + ' (no syntax check)')

json.dump(E, io.open(os.path.join(OUT, 'edits.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nedits:', len(E), ' ALL OK' if ok else ' *** PROBLEMS ***')
