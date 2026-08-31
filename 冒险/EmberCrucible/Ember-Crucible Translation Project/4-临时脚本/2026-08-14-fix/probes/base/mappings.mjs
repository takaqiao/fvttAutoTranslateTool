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
  adjective: 'system.adjective',
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
  tokenName: { path: 'prototypeToken.name', converter: 'name' },
  biography: nested('system.details.biography'),
  ancestry: nested('system.details.ancestry'),
  background: nested('system.details.background'),
  archetype: nested('system.details.archetype'),
  taxonomy: nested('system.details.taxonomy'),
  actions: ACTIONS_FIELD,
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
  }),
  'JournalEntryPage.ember.standaloneEvent': P({
    overview: 'system.overview',
    exposition: 'system.exposition',
    summary: 'system.summary',
    outcomes: OUTCOMES_FIELD,
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
  },
};

/** Layers for crucible-cn (Crucible system packs only). */
export const CRUCIBLE_LAYER = { ...CRUCIBLE_MAPPINGS, ...SCENE_LEVELS };

/** Layers for ember_cn_unofficial (Ember packs; Crucible + Ember schemas). */
export const EMBER_LAYER = { ...CRUCIBLE_MAPPINGS, ...EMBER_PAGE_MAPPINGS, ...SCENE_LEVELS };

/**
 * Babele's own defaults that the extractor needs in order to walk documents
 * this project does not override. Kept minimal and in sync with
 * babele/script/mapping/default-mappings.js (2.9.1).
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
  JournalEntry: {
    name: 'name',
    categories: { path: 'categories', converter: 'nameCollection' },
    pages: { path: 'pages', converter: 'document', documentType: 'JournalEntryPage', cardinality: 'many' },
  },
  JournalEntryPage: {
    name: 'name',
    caption: 'image.caption',
    text: 'text.content',
  },
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
    name: 'name',
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
  RegionBehavior: { name: 'name' },
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
 * `extract_en.mjs` resolves them the way Babele's `_variants` normalization does.
 *
 * @param {'crucible'|'ember'} target
 * @returns {object}
 */
export function effectiveMappings(target) {
  const layer = target === 'ember' ? EMBER_LAYER : CRUCIBLE_LAYER;
  const effective = { ...BABELE_DEFAULTS };
  for (const [documentType, definition] of Object.entries(layer)) {
    effective[documentType] = { ...(effective[documentType] ?? {}), ...definition };
  }
  return effective;
}
