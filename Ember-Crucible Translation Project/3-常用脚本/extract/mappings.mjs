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

export const CRUCIBLE_MAPPINGS = {
  Item: CRUCIBLE_ITEM,
  Actor: CRUCIBLE_ACTOR,
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

export const EMBER_PAGE_MAPPINGS = {
  'JournalEntryPage.ember.location': P({
    overview: 'system.overview',
    exposition: 'system.exposition',
    terrain: 'system.terrain',
  }),
  'JournalEntryPage.ember.biome': P({
    overview: 'system.overview',
    exposition: 'system.exposition',
    terrain: 'system.terrain',
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

/** Layers for crucible-cn (Crucible system packs only). */
export const CRUCIBLE_LAYER = { ...CRUCIBLE_MAPPINGS };

/** Layers for ember_cn_unofficial (Ember packs; Crucible + Ember schemas). */
export const EMBER_LAYER = { ...CRUCIBLE_MAPPINGS, ...EMBER_PAGE_MAPPINGS };

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
  ActiveEffect: {
    name: 'name',
    description: 'description',
    adjective: 'system.adjective',
    actions: ACTIONS_FIELD,
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
 * @param {'crucible'|'ember'} target
 * @returns {object}
 */
export function effectiveMappings(target) {
  const layer = target === 'ember' ? EMBER_LAYER : CRUCIBLE_LAYER;
  return { ...BABELE_DEFAULTS, ...layer };
}
