/**
 * Single source of truth for the Babele document-mapping layers used by this
 * project.
 *
 * The SAME data is consumed by two places, which is the whole point:
 *   - runtime  : each plugin's register file passes these to
 *                `babele.registerMapping()`
 *   - extract  : `extract_en.mjs` *interprets* these layers to decide which
 *                fields to pull out of the LevelDB packs
 *
 * Keeping one definition guarantees the extracted English baseline and the
 * translation files always have exactly the keys Babele will look for.
 *
 * Shape is identical to Babele's own `default-mappings.js`: an object keyed by
 * document type. Subtype keys (`JournalEntryPage.ember.location`) are natively
 * supported by Babele 2.9.1 and normalized internally into `_variants`.
 *
 * IMPORTANT: do not replace the `document` converter entries with hand-written
 * traversal converters. The built-in `document` converter is what gives us
 * source-pack fallback (`_stats.compendiumSource` -> original pack's
 * translation), which auto-translates ~82% of Ember's actor-embedded items.
 * See PROJECT.md section 3.2.
 */

/* ------------------------------------------------------------------ *
 * Crucible system schema
 * ------------------------------------------------------------------ */

/**
 * Crucible actions: `system.actions` is an array of objects with `id`, each
 * carrying name/description/condition plus an `effects` array whose entries
 * only need `name`.
 *
 * Babele's built-in `structured` converter cannot express the nested
 * `effects[].name` array, and the existing crucible-cn translations are already
 * in this shape, so this stays a named custom converter.
 *
 * ⚠ 只有 **Item** 与 **affix 子类型的 ActiveEffect** 真有这个字段。
 * `Actor.system.actions` **不是 schema 字段**：`CrucibleBaseActor.defineSchema()`
 * 只声明 abilities/defenses/resistances/resources/movement/currency/status/favorites，
 * 而 `actions` 是 `prepareBaseData` 期间挂上去的**派生属性**，且是
 * `Object<string, CrucibleAction>`（对象，不是数组），根本不进 `toObject()` / pack JSON。
 * 所以曾经写在 `CRUCIBLE_ACTOR` 里的那一行两个方向都恒空转 ——
 * 抽取侧 `toArray()` 拿到 undefined、运行时 `crucibleActions` 第一句就
 * `!Array.isArray(actions)` 返回原值。2026-08-14 已删除，实测复核：
 *   - 上游 LevelDB 全量（ember 10 包 + crucible 15 包）：`system.actions` 在
 *     actor 层 **0** 个文档、item 层 4615 个、ActiveEffect 层 34 个；
 *   - 两仓 compendium/{en,cn} 全量：文档层为 Actor 的 `actions` 键 **0** 处，
 *     Item 层 2466 处。
 * 别再往 `CRUCIBLE_ACTOR` 里加回去。
 */
const ACTIONS_FIELD = { path: 'system.actions', converter: 'crucibleActions' };

/**
 * Crucible `system.description` is POLYMORPHIC: a plain string on most item
 * types (talent, ancestry, archetype, background, taxonomy, spell) and a
 * `{public, private}` object on others (equipment, adversary gear).
 *
 * Discovered by cross-checking our extraction against Padhiver/Crucible-FR:
 * mapping it as `system.description.public` silently dropped ~80k characters
 * from `crucible.talent` alone. Keep the polymorphic converter.
 */
const DESCRIPTION_FIELD = { path: 'system.description', converter: 'crucibleDescription' };

/** `{name, description}` / `{public, private}` sub-objects translated in place. */
const nested = (path) => ({ path, converter: 'crucibleNested' });

const CRUCIBLE_ITEM = {
  name: 'name',
  description: DESCRIPTION_FIELD,
  // ⚠ 这里**不要**加 `adjective: 'system.adjective'`。它是 affix 子类型 ActiveEffect
  // 的字段（`CrucibleAffixActiveEffect` 的 schema，见下方 CRUCIBLE_ACTIVE_EFFECT），
  // Item 自己没有：上游 LevelDB 全量实测 item 层 `system.adjective` **0** 个文档
  // （ActiveEffect 层 172 个），两仓 compendium/{en,cn} 全量实测文档层为 Item 的
  // `adjective` 键 **0** 处（ActiveEffect 层 344 处）。2026-08-14 删除该死声明。
  actions: ACTIONS_FIELD,
  effects: {
    path: 'effects',
    converter: 'document',
    documentType: 'ActiveEffect',
    cardinality: 'many',
  },
};

const CRUCIBLE_ACTOR = {
  name: 'name',
  /**
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
  biography: nested('system.details.biography'),
  ancestry: nested('system.details.ancestry'),
  background: nested('system.details.background'),
  archetype: nested('system.details.archetype'),
  taxonomy: nested('system.details.taxonomy'),
  // ⚠ 这里**不要**加 `actions: ACTIONS_FIELD` —— Actor 没有 `system.actions` 这个
  // schema 字段，理由与实测数字见 ACTIONS_FIELD 的注释。2026-08-14 删除该死声明。
  // Left as Babele's built-in `document` converter ON PURPOSE: that is what
  // gives us source-pack fallback for embedded items (PROJECT.md §3.2).
  items: {
    path: 'items',
    converter: 'document',
    documentType: 'Item',
    cardinality: 'many',
    // Prefer the owning package's own translated packs before the generic
    // lookup, so Ember-sourced items resolve against Ember translations.
    fallbackPolicy: 'owner-package-before-generic',
  },
  effects: {
    path: 'effects',
    converter: 'document',
    documentType: 'ActiveEffect',
    cardinality: 'many',
  },
};

/**
 * Crucible's affix ActiveEffect subtype (`crucible.affixes`, plus the effects
 * embedded on equipment items) carries two translatable fields Babele's
 * built-in `ActiveEffect` mapping knows nothing about:
 *
 *   - `system.adjective` — composed into the enchanted item's displayed name.
 *     crucible `module/models/item-physical.mjs:324`:
 *       `const adj = affix.system.adjective || affix.name;`
 *     Without this key the player sees `Acid-Warding长剑`.
 *   - `system.actions`  — actions the affix grants to the item that bears it.
 *     Declared in `module/models/effect-affix.mjs:36`
 *     (`schema.actions = new fields.ArrayField(new CrucibleActionField())`),
 *     i.e. the SAME shape as `Item.system.actions`, so it reuses ACTIONS_FIELD
 *     rather than inventing a second converter.
 *
 * This block only has to name the two EXTRA fields. Babele merges registered
 * mapping layers into the built-in defaults **per document type, key by key** —
 * `script/mapping/document-mappings.js`: `#rebuild()` -> `#mergeLayer()` ->
 * `#mergedDefinition()` -> `foundry.utils.mergeObject(base, override)`. So this
 * ENRICHES `{name, description, changes}`; it does not replace them.
 *
 * Before this existed, `crucible.affixes` (a pack of documentType
 * `ActiveEffect`) fell through to that built-in default, `adjective`/`actions`
 * were never looked up, and 169 crucible-cn + 14 ember_cn translation leaves
 * were dead on arrival. AUDIT-2026-08-12-multiagent.md §2.2.
 */
const CRUCIBLE_ACTIVE_EFFECT = {
  adjective: 'system.adjective',
  actions: ACTIONS_FIELD,
};

export const CRUCIBLE_MAPPINGS = {
  Item: CRUCIBLE_ITEM,
  Actor: CRUCIBLE_ACTOR,
  ActiveEffect: CRUCIBLE_ACTIVE_EFFECT,
};

/* ------------------------------------------------------------------ *
 * Ember custom JournalEntryPage subtypes
 *
 * Ember's prose does NOT live in `text.content`. Each subtype puts it in
 * different `system.*` fields. Measured against ember 0.6.0; `system.overview`
 * and `text.content` differ in all 741 sampled pages, so both are translated.
 * ------------------------------------------------------------------ */

/** Fields every Ember page keeps from Babele's built-in JournalEntryPage. */
const PAGE_BASE = {
  name: 'name',
  text: 'text.content',
  caption: 'image.caption',
};

/** `system.outcomes` is an array of `{id, label, summary}` (quest events). */
const OUTCOMES_FIELD = {
  path: 'system.outcomes',
  converter: 'structured',
  cardinality: 'many',
  container: 'array',
  key: 'id',
  mapping: {
    label: 'label',
    summary: 'summary',
  },
};

/**
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

/*
 * NOT translatable, do not add back: `system.terrain` on `ember.location` /
 * `ember.biome`.
 *
 * It is an enum id, not prose. `EmberLocationPage` / `EmberBiomePage`
 * (ember.mjs:1898 / :570) build their schema from `EmberLocation` /
 * `EmberBiome`, whose terrain field is
 *   ember.mjs:418  `terrain: new fields.StringField({choices: terrainChoices,
 *                   initial: "normal", blank: false})`
 * with `terrainChoices = () => ember.scenes.region.terrain` (ember.mjs:409).
 * `initializePage()` copies the page's raw value into the region config, and
 * hex construction then does
 *   ember.mjs:846  `this.terrain = this.#region.terrain.get(terrain || ...)`
 *   ember.mjs:847  `if ( !this.terrain ) throw new Error("Terrain ... not
 *                   defined in Region configuration data.")`
 * A Chinese value is not in `choices`, so it either fails field validation and
 * falls back to `initial` ("normal" for biomes, "" for locations) or reaches
 * the registry lookup and throws — either way water/difficult/extreme terrain,
 * movement cost and impassability are lost.
 *
 * Translating it buys nothing anyway: the hex HUD renders the REGISTRY label,
 * not this raw id (ember.mjs:25512
 * `terrain: {label: terrain.label, page: terrainPage, ...}`). Terrain names are
 * localised through `lang/cn.json` instead.
 *
 * 104 already-written Chinese values (52 per adventure pack × 2) must be purged
 * from `compendium/cn`. AUDIT-2026-08-12-multiagent.md §1.3.
 */
export const EMBER_PAGE_MAPPINGS = {
  'JournalEntryPage.ember.location': P({
    overview: 'system.overview',
    exposition: 'system.exposition',
  }),
  'JournalEntryPage.ember.biome': P({
    overview: 'system.overview',
    exposition: 'system.exposition',
  }),
  'JournalEntryPage.ember.quest': P({
    overview: 'system.overview',
  }),
  'JournalEntryPage.ember.questEvent': P({
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
  'JournalEntryPage.ember.lore': P({
    contentOverview: 'system.content.overview',
    contentGamemaster: 'system.content.gamemaster',
    pronunciation: 'system.pronunciation',
    bannerCaption: 'system.banner.caption',
  }),
  'JournalEntryPage.ember.deity': P({
    contentOverview: 'system.content.overview',
    contentGamemaster: 'system.content.gamemaster',
    subtitle: 'system.subtitle',
    pronunciation: 'system.pronunciation',
    bannerCaption: 'system.banner.caption',
  }),
  'JournalEntryPage.ember.culture': P({
    contentOverview: 'system.content.overview',
    contentGamemaster: 'system.content.gamemaster',
    pronunciation: 'system.pronunciation',
    bannerCaption: 'system.banner.caption',
  }),
  'JournalEntryPage.ember.ancestry': P({
    contentOverview: 'system.content.overview',
    contentGamemaster: 'system.content.gamemaster',
    pronunciation: 'system.pronunciation',
    height: 'system.height',
    lifespan: 'system.lifespan',
    origin: 'system.origin',
  }),
  'JournalEntryPage.ember.cosmos': P({
    contentOverview: 'system.content.overview',
    contentGamemaster: 'system.content.gamemaster',
    subtitle: 'system.subtitle',
    pronunciation: 'system.pronunciation',
    bannerCaption: 'system.banner.caption',
  }),
  'JournalEntryPage.ember.organization': P({
    contentOverview: 'system.content.overview',
    contentGamemaster: 'system.content.gamemaster',
    pronunciation: 'system.pronunciation',
  }),
  'JournalEntryPage.ember.characterClass': P({
    contentOverview: 'system.content.overview',
  }),
  'JournalEntryPage.ember.questFlowchart': { name: 'name' },
};

/* ------------------------------------------------------------------ *
 * Ember custom RegionBehavior subtypes
 *
 * Babele 的 RegionBehavior 默认只覆盖**核心** Foundry 的两个子类型
 * （`displayScrollingText` / `teleportToken`，见 BABELE_DEFAULTS 里的 `_variants`）。
 * Ember 自己注册的两个子类型（`module.json:155-156` 的
 * `documentTypes.RegionBehavior = {trapTrigger:{}, areaEffect:{}}`）它当然不认识，
 * 于是这两档文本此前既不进英文基线、运行时也不会被查找 —— 映射里只有 `name`，
 * `system.*` 整档在管线外。
 *
 * 子类型键（`RegionBehavior.ember.trapTrigger`）是本文件已有的写法：Babele 2.9.1
 * 在 `mapping/document-mappings.js:309-345` 用 `#legacySubtypeKey()` 按**第一个点**
 * 切分（documentType=`RegionBehavior`，subtype=`ember.trapTrigger`），再归一成
 * `_variants` 里的 `{_when:{path:'type',equals:'ember.trapTrigger'}, …}`；
 * `mapping-block.js:126-140` 运行时把**基础字段与命中的变体字段取并集**，所以这里
 * 重复写 `name` 是刻意的 —— `extract_en.mjs` 的 `mappingFor()` 命中子类型键时是
 * **整块替换**而不是合并，漏写 `name` 会把行为名从英文基线里打掉。
 *
 * 实测（classic-level 直读上游 ember 包，2026-08-14）：
 *   - `ember.trapTrigger` 8 个行为，`system.message` 8 个非空（1 唯一：`Trap Triggered!`）；
 *   - `ember.areaEffect` 4 个行为，`system.description` 4 个非空（1 唯一，`<p>` 正文），
 *     `system.effects[]` 4 条（1 唯一名 `Bleeding`，形状 `{_id,name,statuses,duration}`，
 *     与 `nameCollection` 同构）。
 *   - 其余 9 个子类型（footstepSurface / defineSurface / modifyMovementCost /
 *     adjustDarknessLevel / suppressWeather / changeLevel / executeScript /
 *     teleportToken / displayScrollingText，合计 798 个行为）没有任何可译 `system.*` 文本，
 *     故意不列。
 *
 * ⚠ `system.effects[].name` 是**状态名**（`statuses:['bleeding']`）。译文必须与
 * `lang/cn.json` 里同一状态的用词一致，否则同一效果在两处显示不同名字。
 * ------------------------------------------------------------------ */
export const EMBER_REGION_BEHAVIOR_MAPPINGS = {
  'RegionBehavior.ember.trapTrigger': {
    name: 'name',
    message: 'system.message',
  },
  'RegionBehavior.ember.areaEffect': {
    name: 'name',
    description: 'system.description',
    effects: { path: 'system.effects', converter: 'nameCollection' },
  },
};

/* ------------------------------------------------------------------ *
 * Composed layers
 * ------------------------------------------------------------------ */

/**
 * Scene 层：**只补 `levels` 一个字段**，其余（name / drawings / notes / regions）
 * 保留 Babele 默认 —— 层的合并是**按字段**的（`#mergeLayer` -> `#mergedDefinition`），
 * 所以这里写一个键不会顶掉那四个。
 *
 * 为什么需要它：Ember 的 Vista 场景把每个构图/楼层做成一个 `level`，玩家在
 * Foundry 的层级选择器里直接看到 `level.name`；而 **Babele 2.9.1 的 Scene 默认
 * mapping 里没有 `levels`**，所以这一档此前既不会被抽取、也不会被翻译 ——
 * 实测 195 个场景、517 处、**255 个唯一层名全是英文**。
 *
 * 这个盲区是 2026-08-13 拿 crucible-fr 做**外部第二实现对照**才发现的：所有静态检查
 * 都以「英文基准里有这一条」为起点，而基准里压根没有它，于是不在任何判据的定义域内
 * （与 08-12 的「中文侧整条不存在」是同一类盲区的另一侧）。FR 为此写了自定义的
 * `ember_scene_levels_converter`（`levels[level._id] ?? levels[level.name]`），
 * 但本项目用内建 `nameCollection`（＝`fieldCollection("name")`）就够，且更安全：
 * 查不到译文原样返回，查到也只 `mergeObject` 掉 `name` 一个键，
 * `_id` / `bottom` / `top` / `flags` 全部保留，对文档数据是非破坏的。
 */
const SCENE_LEVELS = {
  Scene: {
    levels: { path: 'levels', converter: 'nameCollection' },

    // 同一个盲区的另外两块（2026-08-13 由 crucible-fr 对照查出，三者是同一个改造点）：
    //
    // `tokens[].name` —— **已经摆在场景上**的 token 的覆盖名。我们此前只翻
    // `prototypeToken.name`（`tokenName`），那只影响**新放置**的 token；冒险包里摆好的
    // token 存的是自己的 `name`。而且不能假设「actor 名翻了 token 就跟着变」——
    // 实测 ember 3 个 token 名里有 2 个与 actor 名**不是同一个字符串**
    // （token `Vivisector` 对 actor `Mutagist Vivisector`、`Kalasak` 对 `Kalasak the Cutter`）。
    // crucible.playtest 另有 8 处（预生角色的 token，`actorLink: true`）。
    tokens: { path: 'tokens', converter: 'nameCollection' },

    // `navName` —— 画布顶部**场景导航条**上的名字。ember 有 2 处 / 1 唯一
    // （场景 `Aedir Signalpost` 的 navName＝`Tower Overlook`），该字符串在中文包里
    // 一次都不出现 —— 不是挂错位置，是整条没有中文。
    navName: 'navName',

    // `sounds[].name` —— 场景里**环境音**（AmbientSound）的名字，GM 在音效层与
    // 场景搜索里直接看到。Babele 2.9.1 的 Scene 默认里同样没有 `sounds`
    // （`default-mappings.js:202-217` 只有 name/drawings/notes/regions），所以它和
    // `levels` 是同一类盲区：既不进英文基线，也不会被任何判据看见。
    //
    // 实测（classic-level 直读上游包）：`packs/adventure` 与 `packs/crucible-adventure`
    // 各 80 个 `sounds[].name` 叶，合计 **160 叶 / 40 唯一**，取值全英文
    // （Tar Pit Bubbles×50、Waterfall×12、Silver Beam Alarm×10…，另有上游作者备注
    // `Spider Scuttling (TO DO)`）。字段真实性：Foundry `common/documents/ambient-sound.mjs`
    // 的 `name: new fields.StringField({textSearch: true})`。
    //
    // 用内建 `nameCollection`（＝`fieldCollection("name")`），与同块 levels / tokens
    // 完全同型：按英文名建键、查不到原样返回、命中也只 `mergeObject` 掉 `name` 一个键，
    // `_id`/`path`/`radius`/`volume`/`flags` 全保留，对文档非破坏。
    sounds: { path: 'sounds', converter: 'nameCollection' },
  },
};

/** Layers for crucible-cn (Crucible system packs only). */
export const CRUCIBLE_LAYER = { ...CRUCIBLE_MAPPINGS, ...SCENE_LEVELS };

/** Layers for ember_cn_unofficial (Ember packs; Crucible + Ember schemas). */
export const EMBER_LAYER = {
  ...CRUCIBLE_MAPPINGS,
  ...EMBER_PAGE_MAPPINGS,
  ...EMBER_REGION_BEHAVIOR_MAPPINGS,
  ...SCENE_LEVELS,
};

/**
 * Babele's own defaults that the extractor needs in order to walk documents
 * this project does not override.
 *
 * **逐块对齐 `babele/script/mapping/default-mappings.js` (2.9.1)；最后核对
 * 2026-08-14。** 在此之前这里只写着「kept minimal and in sync」，而实际有 6 处出入
 * （RegionBehavior 的 `_variants` 整块、Adventure.cards、JournalEntry.description、
 * JournalEntryPage 的 src/width/height、Macro.command、TableResult.name 的
 * referencedDocumentField 形态），外加 Actor/Item/Cards/Card/Folder 五个基础层整块缺失。
 * 「以为镜像是忠实的」本身就是一类缺陷源：`effectiveMappings()` 是按 documentType
 * 浅合并，镜像缺一个键就意味着抽取端与运行时的键集不同。
 *
 * ⚠ **本表不进运行时**：`release/generate_runtime.mjs` 只序列化
 * `CRUCIBLE_LAYER` / `EMBER_LAYER`（它 import 的就是这两个），运行时的默认值由 Babele
 * 自己提供。所以这里的每一条只影响 `extract_en.mjs` 抽出来的英文基线。
 *
 * ⚠ **刻意偏离上游的三处**（不是遗漏，别「顺手补齐」）：
 *   1. `Macro.command: 'command'` —— 上游有，这里**故意不写**。宏正文是 JavaScript，
 *      不是散文。补上它会把 14 个宏、2433 字符的 JS 源码拉进英文基线当待译串，
 *      而 `extract_en.mjs` 目前没有「按字段名跳过」的机制（要做得改抽取器，
 *      不在本文件的职责内）。实测：ember+crucible 全量 14 个 Macro，14 个有 command。
 *   2. `Actor.description: 'system.details.biography.value'` —— 上游有，这里
 *      **暂不写**。CRUCIBLE_ACTOR 没有 `description` 键，补上它等于把 dnd5e 形状的
 *      传记（实测上游 255 个 actor 的 `system.details.biography.value` 非空，约 55.7 万字符）
 *      一次性拉进英文基线与待译量 —— 这正是 PROJECT.md 第 1 节挂着的「dnd5e 侧：
 *      补 / 不补 / 只补传记」待裁项，与 `crucibleNested` 白名单是否加 `value` 是同一个决定。
 *      裁决之前不动；裁决为「补」时，这里与 runtime-converters.js 的 `crucibleNested.extract`
 *      要同一次改。
 *   3. `Scene` 块里**没有** `levels` —— 上游确实没有，本项目的补充在 SCENE_LEVELS 层（见上）。
 */
export const BABELE_DEFAULTS = {
  Adventure: {
    name: 'name',
    description: 'description',
    caption: 'caption',
    folders: { path: 'folders', converter: 'nameCollection' },
    journals: { path: 'journal', converter: 'document', documentType: 'JournalEntry', cardinality: 'many' },
    scenes: { path: 'scenes', converter: 'document', documentType: 'Scene', cardinality: 'many' },
    macros: { path: 'macros', converter: 'document', documentType: 'Macro', cardinality: 'many' },
    playlists: { path: 'playlists', converter: 'document', documentType: 'Playlist', cardinality: 'many' },
    tables: { path: 'tables', converter: 'document', documentType: 'RollTable', cardinality: 'many' },
    items: { path: 'items', converter: 'document', documentType: 'Item', cardinality: 'many' },
    actors: { path: 'actors', converter: 'document', documentType: 'Actor', cardinality: 'many' },
    cards: { path: 'cards', converter: 'document', documentType: 'Cards', cardinality: 'many' },
  },
  /*
   * Actor / Item 基础层：本项目的 CRUCIBLE_MAPPINGS 会按键盖住其中大部分，写在这里
   * 只是为了让镜像成立、并让「哪些键是上游给的、哪些是本项目加的」一眼可读。
   * 实测：加上这两块后英文基线**零 diff**（唯一会产生 diff 的
   * `Actor.description` 见文件头刻意偏离第 2 条，故不写）。
   */
  Actor: {
    name: 'name',
    items: { path: 'items', converter: 'document', documentType: 'Item', cardinality: 'many' },
    effects: { path: 'effects', converter: 'document', documentType: 'ActiveEffect', cardinality: 'many' },
    // 上游是 `{path:'prototypeToken.name', converter:'name'}`；本项目在 CRUCIBLE_ACTOR
    // 里用 `crucibleTokenName` 覆盖它（内建行为的超集，理由见那里的长注释）。
    tokenName: { path: 'prototypeToken.name', converter: 'name' },
  },
  Item: {
    name: 'name',
    // 本项目用 DESCRIPTION_FIELD（`crucibleDescription`）按键覆盖它，见 CRUCIBLE_ITEM。
    description: 'system.description.value',
    effects: { path: 'effects', converter: 'document', documentType: 'ActiveEffect', cardinality: 'many' },
  },
  /*
   * Babele's REAL 2.9.1 default (verbatim from default-mappings.js lines 80-91).
   *
   * This used to be written as `{name, description, adjective, actions}`, which
   * this file claimed was "in sync with babele's defaults" but was not: Babele
   * never looks up `adjective`/`actions` on an ActiveEffect unless a registered
   * layer says so. The extractor believed the lie and emitted keys the runtime
   * could not resolve — root cause of AUDIT-2026-08-12-multiagent.md §2.2. Those
   * two fields now live where they belong, in CRUCIBLE_ACTIVE_EFFECT above.
   *
   * `changes` extracts to nothing here: the extractor's `structured` case only
   * implements the `mapping` form, not the `key`+`valuePath` form, and that is
   * the right outcome — AE change values are numbers and formulas, not prose.
   * It is listed anyway so this table stays a faithful mirror of Babele's.
   */
  ActiveEffect: {
    name: 'name',
    description: 'description',
    changes: {
      path: 'changes',
      converter: 'structured',
      cardinality: 'many',
      container: 'array',
      key: 'key',
      valuePath: 'value',
    },
  },
  /*
   * Cards / Card / Folder：上游有，这里此前整块缺。两个包里一张牌都没有
   * （实测 ember + crucible 全部 pack：`!cards!` 桶 0 个文档、Adventure.cards 0 条），
   * 所以补上它们对英文基线是零 diff，纯粹是让镜像成立、并让 Adventure.cards 的
   * `document` 转换器有 childMapping 可用。
   */
  Cards: {
    name: 'name',
    description: 'description',
    cards: { path: 'cards', converter: 'document', documentType: 'Card', cardinality: 'many' },
  },
  Card: {
    name: 'name',
    description: 'description',
    suit: 'suit',
    faces: {
      path: 'faces',
      converter: 'structured',
      cardinality: 'many',
      compact: true,
      mapping: { img: 'img', name: 'name', text: 'text' },
    },
    back: {
      path: 'back',
      converter: 'structured',
      cardinality: 'one',
      mapping: { img: 'img', name: 'name', text: 'text' },
    },
  },
  Folder: {},
  JournalEntry: {
    name: 'name',
    // 上游 `default-mappings.js:141`。v10 起 JournalEntry 已无 `content` 字段
    // （正文搬进 pages），实测两包 162 个 JournalEntry 里 `content` 非空 0 个 ⇒ 抽取零 diff。
    description: 'content',
    categories: { path: 'categories', converter: 'nameCollection' },
    pages: { path: 'pages', converter: 'document', documentType: 'JournalEntryPage', cardinality: 'many' },
  },
  JournalEntryPage: {
    name: 'name',
    caption: 'image.caption',
    // 上游 `default-mappings.js:156-159`：src / width / height 是给「本地化版
    // 图片或视频」用的。本项目不本地化素材，但镜像照抄 —— 实测两包 3208 个页面里
    // `src` 非空 0 个、`video.width` 非空 0 个，且抽取器只吐非空字符串（数字不吐），
    // 所以现状零 diff。若哪天上游加了视频页，`src` 会以资源路径的形态进基线，
    // 到时候按「不译资源路径」处理，别再把这三行删掉制造隐性偏离。
    src: 'src',
    text: 'text.content',
    width: 'video.width',
    height: 'video.height',
  },
  // 上游还有 `command: 'command'`，刻意不写，理由见文件头「刻意偏离」第 1 条。
  Macro: { name: 'name' },
  Playlist: {
    name: 'name',
    description: 'description',
    sounds: { path: 'sounds', converter: 'document', documentType: 'PlaylistSound', cardinality: 'many' },
  },
  PlaylistSound: { name: 'name', description: 'description' },
  RollTable: {
    name: 'name',
    description: 'description',
    results: { path: 'results', converter: 'document', documentType: 'TableResult', cardinality: 'many' },
  },
  TableResult: {
    _identity: { export: ['range', '_id'], match: ['_id', 'range'] },
    // 上游 `default-mappings.js:194-199` 是这个对象形态，不是裸 `'name'`：结果行指向
    // 别的文档时，名字要从**被引用文档的译文**里取。抽取方向零 diff —— `extract_en.mjs`
    // 对未知 converter 走 `default:` 分支，读 `spec.path ?? field` 也就是 `name` 的原值，
    // 与裸字符串写法逐字等价。
    name: {
      path: 'name',
      converter: 'referencedDocumentField',
      uuidPath: 'documentUuid',
      referencedField: 'name',
    },
    description: 'description',
  },
  Scene: {
    name: 'name',
    drawings: { path: 'drawings', converter: 'textCollection' },
    notes: { path: 'notes', converter: 'textCollection' },
    // 注意：Babele 2.9.1 的 Scene 默认里**没有 `levels`**，本块是它的忠实副本，所以这里也不能有。
    // 本项目对 `levels` 的补充放在 SCENE_LEVELS 那一层（见下），由 registerMapping 增补上去。
    regions: { path: 'regions', converter: 'document', documentType: 'Region', cardinality: 'many' },
  },
  Region: {
    name: 'name',
    behaviors: { path: 'behaviors', converter: 'document', documentType: 'RegionBehavior', cardinality: 'many' },
  },
  /*
   * 上游 `default-mappings.js:228-239` 的 `_variants` 整块此前是缺的。它管的是 Foundry
   * 核心的两个区域行为子类型：
   *   - `displayScrollingText` -> `system.text`（在画布上飘字给玩家看）
   *   - `teleportToken`        -> `system.dialog.revealed` / `.unrevealed`
   *     （`client/data/region-behaviors/teleport-token.mjs` 把它们当玩家可见的确认提示上屏）
   * 补上它是「镜像忠实」这一条本身的要求；**但对本项目的抽取器它是惰性的**——
   * `extract_en.mjs:176` 跳过一切 `_` 开头的键，`mappingFor()` 只实现了子类型键、
   * 没有实现 `_variants` 归一（第十四轮的建议稿说「已实现」，是错的，已实测证伪）。
   * 真正让这两档进英文基线的是下面的 EXTRACTOR_SUBTYPE_SHIMS。
   */
  RegionBehavior: {
    name: 'name',
    _variants: [
      {
        _when: { path: 'type', equals: 'displayScrollingText' },
        text: 'system.text',
      },
      {
        _when: { path: 'type', equals: 'teleportToken' },
        revealedDialog: 'system.dialog.revealed',
        unrevealedDialog: 'system.dialog.unrevealed',
      },
    ],
  },
};

/**
 * 只服务**抽取方向**的子类型垫片。
 *
 * 运行时不需要它们：Babele 自己的默认里就带着上面那两段 `_variants`，
 * `mapping-block.js:126-140` 会把基础字段与命中的变体字段取并集，所以运行时查的键就是
 * `{name, text}` / `{name, revealedDialog, unrevealedDialog}`。
 *
 * 抽取器需要它们：`extract_en.mjs` 的 `mappingFor()` 只认 `Type.subtype` 形式的键
 * （和 `EMBER_PAGE_MAPPINGS` 一样），不认 `_variants`。键名与上游变体逐字相同，
 * 所以抽出来的英文基线与运行时查找的键集仍然一致。
 * 注意 `mappingFor()` 命中子类型键时是**整块替换**，所以每块都要重复写 `name`，
 * 否则会把行为名从基线里打掉（`Scrolling Text` / `Teleport Token` 等 398 条 name 叶）。
 *
 * 实测（classic-level 直读上游 ember 包，2026-08-14）：
 *   - `displayScrollingText` 2 个行为，`system.text` 2 个非空、1 唯一：`Searing Light!`
 *     —— 这就是此前整条不在英文基线里的那一串（en/cn 四个文件 grep 计数皆 0）。
 *   - `teleportToken` 156 个行为，`system.dialog.revealed`/`unrevealed` **非空 0 个**
 *     （全是 `null`）⇒ 这一档当前无实害，补上是为了上游哪天填了内容时不再是盲区。
 */
export const EXTRACTOR_SUBTYPE_SHIMS = {
  'RegionBehavior.displayScrollingText': {
    name: 'name',
    text: 'system.text',
  },
  'RegionBehavior.teleportToken': {
    name: 'name',
    revealedDialog: 'system.dialog.revealed',
    unrevealedDialog: 'system.dialog.unrevealed',
  },
};

/**
 * Effective mapping used by the extractor: Babele defaults overlaid with this
 * project's layer.
 *
 * The overlay is PER DOCUMENT TYPE, field by field — not a whole-type replace.
 * That is what Babele itself does with a registered layer
 * (`script/mapping/document-mappings.js`: `#mergeLayer()` -> `#mergedDefinition()`
 * -> `foundry.utils.mergeObject(base, override)`), and the extractor has to
 * reproduce the runtime's own resolution or it extracts a different key set than
 * the runtime looks up — the exact failure mode PROJECT.md §8 records as
 * "校验必须复刻被验证系统的查找语义".
 *
 * It became load-bearing when `ActiveEffect` gained a project layer: a shallow
 * `{...BABELE_DEFAULTS, ...layer}` would have replaced `{name, description,
 * changes}` with `{adjective, actions}` and silently dropped every ActiveEffect
 * name and description from the English baseline.
 *
 * Subtype keys (`JournalEntryPage.ember.location`) have no counterpart in
 * BABELE_DEFAULTS, so they simply pass through; `mappingFor()` in
 * `extract_en.mjs` resolves them by looking up `` `${documentType}.${doc.type}` ``.
 *
 * ⚠ 一个已经骗过一轮 triage 的差别：`mappingFor()` 只实现了**子类型键**，
 * **没有**实现 `_variants` 归一，而且命中子类型键时是**整块替换**、不与基础块合并
 * （Babele 运行时则是取并集，`mapping-block.js:126-140`）。所以
 *   ① 写进 BABELE_DEFAULTS 的 `_variants` 对抽取器是惰性的（`extract_en.mjs:176`
 *      跳过一切 `_` 开头的键）—— 要让那一档进基线得走 EXTRACTOR_SUBTYPE_SHIMS；
 *   ② 每个子类型块都必须把基础块里仍需要的键（尤其 `name`）重复写一遍。
 *
 * @param {'crucible'|'ember'} target
 * @returns {object}
 */
export function effectiveMappings(target) {
  const layer = target === 'ember' ? EMBER_LAYER : CRUCIBLE_LAYER;
  // 垫片只补 `Type.subtype` 形式的新键，与 BABELE_DEFAULTS 的键不相交，浅并即可。
  const effective = { ...BABELE_DEFAULTS, ...EXTRACTOR_SUBTYPE_SHIMS };
  for (const [documentType, definition] of Object.entries(layer)) {
    effective[documentType] = { ...(effective[documentType] ?? {}), ...definition };
  }
  return effective;
}
