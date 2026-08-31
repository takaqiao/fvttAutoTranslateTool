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
 * Crucible actions: on the 13 Item subtypes and on the `affix` ActiveEffect
 * subtype, `system.actions` is an ArrayField of objects with `id`, each
 * carrying name/description/condition plus an `effects` array whose entries
 * only need `name`.
 *
 * NOT on Actor — `CrucibleBaseActor` has no such schema field; see the note in
 * CRUCIBLE_ACTOR below.
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
  // NO `adjective` here. In crucible 0.10.1 `system.adjective` exists on exactly
  // one document shape: the `affix` ActiveEffect subtype
  // (`CrucibleAffixActiveEffect`, crucible-compiled.mjs:40771) — declared where
  // it belongs in CRUCIBLE_ACTIVE_EFFECT below. None of the 13 Item subtypes has
  // the field, so this line was a silent no-op in both directions (Babele never
  // warns about a mapping path that does not exist). Verified against the data:
  // all 274 `adjective` keys in either repo's `compendium/{en,cn}` sit at effect
  // level, none at item level.
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
  /*
   * PLAIN PATH — deliberately NOT Babele's built-in `name` converter.
   *
   * Babele 2.9.1's own default is
   *   `tokenName: {path: 'prototypeToken.name', converter: 'name'}`
   * and `name` resolves to `Converters.mappedField("name")`
   * (`script/converter/converters.js:92-97`):
   *   (_value, _translation, data, tc) => tc.translateField("name", data)
   * The second parameter — the `tokenName` value out of our translation files —
   * is discarded, and what gets written to `prototypeToken.name` is the
   * translation of `name`. Mirroring that default made all 575 `tokenName`
   * leaves dead, and in the 94 places where upstream deliberately gives the
   * token a DIFFERENT string from the actor (`Mage` for `Corvana Vortest`,
   * `Pallid Drake` for `Pallid Ultra Drake`) it pushed the actor's full
   * bilingual name onto the canvas, spoiling hidden identities.
   *
   * As a plain path, `PrimitiveConverter.translate` returns
   * `translations.tokenName`, and `FieldMapping.map()` writes nothing at all
   * when that key is absent, so the English source survives untouched.
   * Coverage measured before the change: 575 English leaves / 575 Chinese / 0
   * missing, so nothing loses a translation by switching.
   */
  tokenName: 'prototypeToken.name',
  biography: nested('system.details.biography'),
  ancestry: nested('system.details.ancestry'),
  background: nested('system.details.background'),
  archetype: nested('system.details.archetype'),
  taxonomy: nested('system.details.taxonomy'),

  // NO `actions` here. Unlike Item and the affix ActiveEffect, no Crucible Actor
  // subtype has an `actions` schema field: `CrucibleBaseActor.defineSchema()`
  // (crucible-compiled.mjs:40948-41005) returns only abilities / defenses /
  // resistances / resources / movement / currency / status / favorites, and the
  // hero and adversary subtypes add only advancement / details. What a PREPARED
  // actor exposes as `actions` is the derived class field at :41014, typed
  // `Object<string, CrucibleAction>` — a keyed object, not the array
  // `crucibleActions` requires — and it is never persisted, so pack JSON has no
  // such key at all. Both directions were silent no-ops; actor action text is
  // covered by the `items` document converter below. Verified against the data:
  // 0 actor-level `actions` keys in either repo's `compendium/{en,cn}` (all
  // 1984 of them sit on items or effects).

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
 * `system.encounter.tokens` on quest / standalone event pages: an array of spawn
 * GROUPS, each with an `actors` array whose entries carry a `tokenData` override
 * blob. `tokenData.name` OVERRIDES the (already translated)
 * `prototypeToken.name` the moment a GM runs the encounter —
 * `EmberScene#_createActorToken` (ember.mjs:29345) hands `tokenData` straight to
 * `actor.getTokenDocument()` and then bakes the result into the token's actor
 * delta:
 *   `if ( !token.actorLink && (token.name !== actor.prototypeToken.name) )`
 *   `  token.delta.updateSource({name: token.name});`
 * so translating the actor is not enough: 128 of the 130 distinct override
 * strings differ from the actor they hang off (`Friendly Ooze` on `Oozeling`,
 * `Beacon Brigade Patroller` on `Wandren Patroller`, and every named NPC hung
 * off a generic `Arcturian` / `Ordani`).
 *
 * Two levels of UNKEYED arrays (groups -> actors), which Babele's built-in
 * `structured` converter cannot express — and index keys would silently
 * re-target on any upstream reorder. The 382 leaves per pack are only 130
 * distinct strings, so translations are keyed BY THE ENGLISH NAME, exactly like
 * the built-in `nameCollection`.
 */
const ENCOUNTER_TOKENS_FIELD = {
  path: 'system.encounter.tokens',
  converter: 'emberEncounterTokens',
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

    // 同一次改造漏下的第四块（2026-08-14 补）：`sounds[].name`。
    // **Foundry v14 起 AmbientSound 才有 `name` 字段**
    // （`common/documents/ambient-sound.mjs:37`，`textSearch: true`，会进场景搜索），
    // 而 Babele 2.9.1 的 Scene 默认表是 v13 时代的，没有它 —— 又一个「英文基线里
    // 压根没有这一条」的定义域外盲区。实测两个 adventure 包各 80 叶、共 160 处、
    // 40 个唯一值全英文（`Tar Pit Bubbles`×25 / `Waterfall`×6 / `Silver Beam Alarm`×5
    // …其中 `Spider Scuttling (TO DO)` 是上游作者拿它当备注用）。受众只有 GM
    // （音效层与场景搜索），所以价值低于 levels/tokens/navName，但同样一行就能补。
    // 与 `BABELE_DEFAULTS.Playlist.sounds`（PlaylistSound 文档）无关：层的合并是
    // 按文档类型的，两者不会互相影响。
    sounds: { path: 'sounds', converter: 'nameCollection' },
  },
};

/**
 * RegionBehavior 的 `system.*` 子字段。
 *
 * Babele 2.9.1 的默认表用 `_variants` 表达「按 `type` 分子类型」，而本项目的
 * BABELE_DEFAULTS 只抄了 `{name: 'name'}`（见下面那块的偏离清单第 1 条），于是
 * `displayScrollingText.system.text` 这类字段从来没进过英文基线 —— 定义域外盲区，
 * 任何判据都看不见（与 `Scene.levels` / `Scene.sounds` 同型）。
 *
 * 这里用**子类型键**而不是 `_variants`，因为两侧都认它：
 *   - 运行时：`DocumentMappings#normalizedLayer()` 把 `A.b` 归一成
 *     `_variants: [{_when: {path: 'type', equals: 'b'}, …}]`，并**追加**在 Babele
 *     自带的 variants 之后（`#mergedDefinition()` 把两个数组拼起来）；基础层的
 *     `name` 仍然生效，因为 `MappingBlock#activeFields()` 取的是「基础字段 ＋
 *     命中的 variant 字段」。
 *   - 抽取器：`extract_en.mjs` 的 `mappingFor()` 优先查 `${documentType}.${doc.type}`，
 *     但它**只取子类型块、不与基础块合并**，所以每一块都必须自带 `name: 'name'`，
 *     否则这些行为已译好的 796 个名字会从基线里掉出去。
 *
 * 实测取值（两个 adventure 包逐条对称，共 810 个 behavior）：
 *   displayScrollingText  各 1 个。`system.text` = "Searing Light!"，
 *                         events=[tokenTurnStart, tokenAnimateIn] —— 玩家回合开始
 *                         时直接飘在画布上。
 *   ember.trapTrigger     各 4 个。`system.message` = "Trap Triggered!"×3 +
 *                         "Pressure Plate Triggered!"×1；schema 是
 *                         `StringField({initial: "Trap Triggered!"})`
 *                         （ember.mjs:2566），hint 写明是陷阱触发时的滚动文字。
 *   ember.areaEffect      各 2 个。`system.description` 是整段陷阱散文（HTMLField，
 *                         进聊天卡），`system.effects[].name` = "Bleeding"。
 *                         `effects` 是 `ArrayField(ObjectField())`（ember.mjs:2702），
 *                         所以 `nameCollection` 往元素上写 `translated: true` 是
 *                         schema 安全的。
 *   teleportToken         各 78 个，但 `system.dialog.revealed/unrevealed`
 *                         **全是 null** —— 今天既抽不出也翻不出东西。列在这里是为了
 *                         让镜像与 Babele 真会查的键一致：上游哪天填了对话框文本，
 *                         它自动进基线，而不是再当一次盲区。
 */
const REGION_BEHAVIOR_VARIANTS = {
  'RegionBehavior.displayScrollingText': {
    name: 'name',
    text: 'system.text',
  },
  'RegionBehavior.teleportToken': {
    name: 'name',
    revealedDialog: 'system.dialog.revealed',
    unrevealedDialog: 'system.dialog.unrevealed',
  },
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

/**
 * Layers for crucible-cn (Crucible system packs only).
 *
 * REGION_BEHAVIOR_VARIANTS is shared even though two of its four subtype keys
 * name Ember behavior types: a subtype key can only ever match a document whose
 * `type` equals it, so `ember.trapTrigger` / `ember.areaEffect` are inert in
 * Crucible's own packs rather than wrong there.
 */
export const CRUCIBLE_LAYER = { ...CRUCIBLE_MAPPINGS, ...SCENE_LEVELS, ...REGION_BEHAVIOR_VARIANTS };

/** Layers for ember_cn_unofficial (Ember packs; Crucible + Ember schemas). */
export const EMBER_LAYER = {
  ...CRUCIBLE_MAPPINGS,
  ...EMBER_PAGE_MAPPINGS,
  ...SCENE_LEVELS,
  ...REGION_BEHAVIOR_VARIANTS,
};

/**
 * Babele's own defaults that the extractor needs in order to walk documents this
 * project does not override.
 *
 * ⚠ THIS IS A DELIBERATE SUBSET, NOT A VERBATIM MIRROR of
 * babele/script/mapping/default-mappings.js (2.9.1). The header used to claim it
 * was "kept in sync" with Babele's defaults. It was not — and taking that claim
 * at face value is exactly what let the ActiveEffect `adjective`/`actions` bug
 * through (AUDIT-2026-08-12-multiagent §2.2, "The extractor believed the lie")
 * and then hid `RegionBehavior.displayScrollingText.system.text` for another
 * eight rounds. So every divergence is listed below with what it costs today. If
 * you add a document type here, diff it against upstream and extend the list.
 *
 * Scope: this table is consumed ONLY by `effectiveMappings()` -> `extract_en.mjs`.
 * It is never handed to `registerMapping()`, so a converter name written here
 * has no runtime meaning and `usesConverters()` never sees it. Its one job is to
 * decide which fields land in `compendium/en`.
 *
 * Divergences from babele 2.9.1, all re-measured 2026-08-14:
 *
 *  1. `RegionBehavior` — upstream also has `_variants` for `displayScrollingText`
 *     (`text: 'system.text'`) and `teleportToken`
 *     (`revealedDialog`/`unrevealedDialog`). NOT copied here, because the
 *     extractor skips `_`-prefixed keys by design; the equivalent lives in
 *     REGION_BEHAVIOR_VARIANTS above, written as subtype keys so that BOTH the
 *     runtime and the extractor resolve them. This was the one divergence that
 *     cost something real: 2 leaves of player-facing canvas text.
 *  2. `TableResult.name` — upstream is `{path: 'name', converter:
 *     'referencedDocumentField', uuidPath: 'documentUuid', referencedField:
 *     'name'}`; kept here as the plain path `'name'`. Same output either way:
 *     that converter's extract is `return context.value`, and
 *     `ReferencedDocumentFieldConverter` puts an explicit local translation
 *     first anyway. Naming a converter the extractor does not implement would
 *     only make it fall through to the `default:` plain-path branch — the same
 *     result by a longer road. (229 of 703 TableResults carry `documentUuid`;
 *     zero loss measured.)
 *  3. `Macro` — upstream also maps `command: 'command'`. Omitted ON PURPOSE:
 *     that is executable JavaScript, and pulling it into the English baseline
 *     would invite someone to translate it. `crucible.macros` has 5 entries,
 *     `name` only.
 *  4. `JournalEntryPage` — upstream also maps `src`, `width: 'video.width'`,
 *     `height: 'video.height'`. Omitted: `src` is an asset path (translating it
 *     would repoint an image), and the two video dimensions are numbers, which
 *     the extractor filters out anyway. Measured: 0 pages in either package
 *     carry any of the three.
 *  5. `Adventure` — upstream also maps `cards` (documentType `Cards`). Omitted
 *     along with the `Cards` / `Card` / `Folder` blocks: 0 card documents in
 *     either package. Add all four together if that ever changes.
 *  6. `JournalEntry` — upstream also maps `description: 'content'`, a v10-era
 *     field that no longer exists on the document. Omitted.
 *  7. No `Actor` / `Item` base blocks at all: CRUCIBLE_MAPPINGS supplies both
 *     and `effectiveMappings()` merges per field, so nothing is lost — with one
 *     asymmetry worth knowing: Babele's own
 *     `Actor.description = 'system.details.biography.value'` stays alive at
 *     RUNTIME while the extractor never emits that key. Inert on the Crucible
 *     side (no such field) and, on the dnd5e side, one half of the two pipeline
 *     blockers the project owner ruled 「先不管」 (PROJECT.md §1).
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
  // 只有 `name`。Babele 真默认还带 `_variants`（displayScrollingText 的
  // `system.text`、teleportToken 的 `system.dialog.*`）—— 见上面偏离清单第 1 条：
  // 抽取器按设计跳过 `_` 开头的键，所以那两块写成子类型键放在
  // REGION_BEHAVIOR_VARIANTS 里，运行时与抽取器都认。动这里之前先读那一段。
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
