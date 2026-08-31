/**
 * ALIEN RPG — THE single source of truth for Babele mapping data.
 *
 * Two consumers read this ONE definition, which is the whole point:
 *   - runtime  : `4-常用脚本/release/generate_runtime.mjs` serialises `ALIEN_LAYER`
 *                with `JSON.stringify` into each plugin's `babele-mappings.js`,
 *                where it is handed to `babele.registerMapping()`.
 *                => EVERYTHING IN `ALIEN_LAYER` MUST BE JSON-SERIALISABLE.
 *                   Converters are referenced BY NAME only; the functions live
 *                   in `4-常用脚本/release/runtime-converters.js`.
 *   - extract  : `4-常用脚本/extract/extract_en.mjs` *interprets* `effectiveMappings()`
 *                to decide which fields to pull out of the LevelDB packs.
 *
 * Extracting a different key set than the runtime looks up is the one defect
 * class this design exists to prevent. EC's hardest-won lesson, verbatim:
 * declaring a field that appears on 0 documents, or adding a field to ONE
 * direction only, silently zeroed 766K chars once.
 *
 * ---------------------------------------------------------------------------
 * CONTRACT WITH `extract_en.mjs` (its header, lines 41-53)
 * ---------------------------------------------------------------------------
 *   export function effectiveMappings(target) -> object keyed by documentType
 *   export const ALIEN_TARGETS = ['alienrpg', 'starterset', 'corerules']
 *   export const ALIEN_LAYER   = { ... }        // project layer alone, NO defaults
 *   export const EXTRACT_CONVERTERS = { ... }   // extract half of custom converters
 *
 * `extract_en.mjs` refuses a `mappings.mjs` that does not positively declare it
 * serves the target (it falls back to Babele's raw defaults and prints a
 * banner), precisely because the EC module it was ported from exports
 * `effectiveMappings` too and would have silently returned a Crucible layer.
 *
 * ---------------------------------------------------------------------------
 * MEASUREMENT PROVENANCE — what `// n=… chars=…` means
 * ---------------------------------------------------------------------------
 * Every mapped path below carries `// n=<documents with a non-empty string>
 * chars=<sum of their lengths>` and, where it differs, `present=<documents on
 * which the path resolves at all, including empty strings and objects>`.
 *
 * Measured 2026-08-29 by walking all three raw pack dumps
 *   6-工作区/raw-dumps/{system,starterset,corerules}.json
 * (one Adventure document each) through the same document tree the extractor
 * walks: Adventure -> folders / actors(->items,effects) / items(->effects) /
 * journal(->pages) / scenes(->regions->behaviors) / tables(->results) /
 * macros / playlists / cards.
 *
 * Corpus census reproduced from that walk (matches verify-inventory.json to the
 * document):
 *   Adventure 3 · Folder 49 · Actor 74 · Item 523 · JournalEntry 13 ·
 *   JournalEntryPage 151 · Scene 15 · Region 4 · RegionBehavior 4 ·
 *   RollTable 137 · TableResult 1530 · Macro 5 ·
 *   ActiveEffect / Playlist / PlaylistSound / Cards / Card: **0 documents**.
 *   Item subtypes  : spacecraftmods 124, item 93, talent 85, weapon 78,
 *                    planet-system 68, spacecraftweapons 20, specialty 19,
 *                    agenda 15, skill-stunts 12, armor 9;
 *                    colony-initiative / spacecraft-crit / critical-injury = 0.
 *   Actor subtypes : creature 30, character 28, spacecraft 7, vehicles 6,
 *                    synthetic 3; colony / planet / territory = 0.
 *   Per pack       : system {Item 26, RollTable 3, TableResult 31, Macro 4,
 *                            JournalEntryPage 1, Folder 7, Actor 0, Scene 0}
 *                    starterset {Actor 13, Item 65 (34 top + 31 embedded),
 *                            JournalEntryPage 38, Scene 7, Region 4,
 *                            RollTable 11, TableResult 124, Folder 17}
 *                    corerules {Actor 61, Item 432 (283 top + 149 embedded),
 *                            JournalEntryPage 112, Scene 8, RollTable 123,
 *                            TableResult 1375, Macro 1, Folder 25}.
 *
 * The machine-readable twin of those comments is `FIELD_CENSUS` at the bottom.
 * **Edit both or neither.** A QA script that re-measures the dumps and diffs
 * against `FIELD_CENSUS` is the guard that would have caught the 766K-char
 * regression; a comment alone is not.
 *
 * ---------------------------------------------------------------------------
 * PROVENANCE OF THE DESIGN
 * ---------------------------------------------------------------------------
 * Started from `7-其他内容/findings/2026-08-29-survey/babele-mapping.json`
 * `proposed_mapping_js`, with EVERY correction in the sibling
 * `verify-mapping.json` applied. Where the two disagree the verifier is right.
 * Corrections carried in, each marked (V) at its site:
 *   1. `crit-inj` -> `critical-injury` (the former is a FILENAME; the variant
 *      could never match). alienrpg.mjs:144, system.json documentTypes.Item.
 *   2. `alienTokenName` DROPPED entirely, and — a consequence the verifier did
 *      not have to reach but the extractor forces — `tokenName` is dropped from
 *      the extract direction too. See EXTRACTOR_DEVIATIONS.
 *   3. `exists` guards on EVERY variant, not only the `description` ones.
 *   4. `system.header.type.value` DO-NOT-TRANSLATE, correct reason recorded.
 *   5. `system.modifiers.*.label` / `.ability` DO-NOT-TRANSLATE on ONE leg only.
 *   6. sigItem / relOne / relTwo carry REAL content and are mapped.
 *   7. Forward-compat fields the proposal omitted, added.
 *   8. `territory` marked forward-compat-only for `system.notes`.
 *   9. Item `system.notes` is a LIVE HTMLField — mapped.
 *
 * FOUND AFTER THE VERIFIER, by running the extractor and diffing its output
 * against this file instead of re-reading this file (marked (V-2) at each site):
 *  10. `RegionBehavior.name` — 4 docs / 70 chars, live via Babele's default,
 *      emitted by the extractor, and MISSING from FIELD_CENSUS. Added.
 *  11. The FIELD_CENSUS total was wrong (2,700,536 vs the actual 2,705,563)
 *      because it was written by hand instead of reduced from the array.
 *  12. FIELD_CENSUS counts SOURCE DOCUMENTS; the baseline counts EMITTED KEYS.
 *      They differ by 299 chars for four nameable reasons. See the census
 *      header — a naive equality check would have failed on day one and been
 *      "fixed" by loosening the check, which is how a real hole gets in.
 * Every source citation below was re-derived on 2026-08-29 against
 * systems/alienrpg 4.1.13 and modules/babele 2.9.1; the survey's citations had
 * drifted 1-4 lines in several places, so none of them are quoted second-hand.
 *
 * ---------------------------------------------------------------------------
 * ⚠ THIS IS A **GLOBAL** LAYER
 * ---------------------------------------------------------------------------
 * `registerMapping` pushes one layer that Babele merges into the effective
 * mapping of EVERY Babele-managed pack in the world
 * (`mapping/document-mappings.js:267-281 #rebuild()` -> `:283-287 #mergeLayer()`
 * -> `:347-361 #mergedDefinition()`), key by key via `foundry.utils.mergeObject`,
 * with `_variants` CONCATENATED (built-ins first, ours appended).
 *
 * `character`, `creature`, `weapon`, `armor`, `item` and `talent` are among the
 * most common `type` keys across Foundry systems. So every variant here is
 * gated on `{all: [{path:'type', in:[…]}, {path:'<an alienrpg-only path>',
 * exists:true}]}`. `_when` semantics verified verbatim at
 * `mapping/mapping-block.js:176-206 #matches()`: `all` / `any` recurse,
 * `equals` is `===`, `in` is `Array.includes`, and `exists:true` is
 * `typeof value !== "undefined"` — note that an EMPTY STRING still counts as
 * existing, so the guard protects against foreign systems where the path is
 * ABSENT, not against alienrpg documents where it is blank. That is exactly the
 * protection needed here.
 *
 * Variant fields shadow base fields BY KEY, last definition winning
 * (`mapping-block.js:126-140 #activeFields()`: `effective.delete(key)` then
 * `effective.set(key, field)`), and `#field()` at :122-124 reads the LAST match.
 * Consequence: the `description` variants below have DISJOINT type sets on
 * purpose — all 13 Item types are partitioned, so no two `description`
 * definitions can ever both match one document.
 */

/* ================================================================== *
 * 0. Targets and shared type vocabularies
 * ================================================================== */

/** Declares to `extract_en.mjs` that this module serves the Alien project. */
export const ALIEN_TARGETS = ['alienrpg', 'starterset', 'corerules'];

/**
 * The 13 registered Item subtypes, byte-exact.
 *
 * (V) `critical-injury`, NOT `crit-inj`. `crit-inj` is the FILENAME
 * (`module/data/item-crit-inj.mjs`, `templates/item/item-crit-inj.hbs`); the
 * registered key is `critical-injury`:
 *   module/alienrpg.mjs:144   `"critical-injury": models.alienrpgCritInj,`
 *   module/alienrpg.mjs:215   `"critical-injury",`
 *   module/sheets/item-sheet.mjs:124  `case "critical-injury":`
 *   system.json documentTypes.Item keys (verified, all 13 listed here).
 * The proposal's `{_when:{path:'type',equals:'crit-inj'}}` could never match a
 * document — `equals` is strict `===` (mapping-block.js:196). Fixed, not deleted.
 */
const ITEM_TYPES = [
  'item', 'weapon', 'armor', 'talent', 'specialty', 'spacecraftweapons',
  'spacecraftmods', 'agenda', 'colony-initiative', 'planet-system',
  'spacecraft-crit', 'critical-injury', 'skill-stunts',
];

/**
 * The 8 registered Actor subtypes (system.json documentTypes.Actor).
 * Kept as a named constant so a future variant can be written against the
 * complete set rather than an ad-hoc list.
 */
export const ALIEN_ACTOR_TYPES = [
  'character', 'creature', 'synthetic', 'vehicles',
  'colony', 'spacecraft', 'planet', 'territory',
];

/**
 * Item types whose long body lives at `system.attributes.comment.value`
 * (prose-mirror + `TextEditor.enrichHTML`). Partition of ITEM_TYPES, part 1/7.
 */
const BODY_IN_ATTRIBUTES = ['item', 'weapon', 'armor', 'spacecraftmods', 'spacecraftweapons'];

/** Item types whose long body lives at `system.general.comment.value`. Part 2/7. */
const BODY_IN_GENERAL = ['agenda', 'talent', 'specialty'];

/**
 * Actor types that inherit base-actor's `notes` HTMLField as a STRING and have
 * a sheet that actually renders it.
 *   module/data/base-actor.mjs:10  `schema.notes = new fields.HTMLField()`
 * Enriched in exactly 6 sheets (re-derived 2026-08-29):
 *   character-sheet.mjs:217 · colony-sheet.mjs:142 · creature-sheet.mjs:150 ·
 *   planet-sheet.mjs:128 · spacecraft-sheet.mjs:180 · synthetic-sheet.mjs:214
 * (V) `vehicles` is absent because the path is an OBJECT there, and `territory`
 * is absent because nothing renders it — both handled in their own variants
 * below, each with its own reason. There is no `_enrichTextFields` function
 * anywhere in the system; the survey invented it.
 */
const NOTES_RENDERED_TYPES = ['character', 'synthetic', 'creature', 'spacecraft', 'colony', 'planet'];

/**
 * Build a variant guard.
 *
 * EVERY variant gets `{path:'type', in:[…]}` AND at least one
 * `{path:'<alienrpg-only path>', exists:true}` — (V) correction 3. The proposal
 * guarded only the `description` variants, which left `notes`, `weaponClass`,
 * `capacity`, `appearance`, `signatureItem`, `relationshipOne` / `Two`,
 * `special` and the whole spacecraft block firing unguarded on every dnd5e /
 * pf2e document whose `type` happened to collide. Translate time is fail-open
 * (`converter/primitive-converter.js` returns undefined on a missing path), but
 * every foreign pack's EXPORT would have gained alien-only keys.
 *
 * @param {string[]} types      subtype keys this variant applies to
 * @param {...string} guards    alienrpg-only paths that must exist
 */
const when = (types, ...guards) => ({
  all: [
    { path: 'type', in: types },
    ...guards.map((path) => ({ path, exists: true })),
  ],
});

/* ================================================================== *
 * 1. THE LAYER
 * ================================================================== */

/**
 * The project's own Babele layer — Babele's defaults are NOT copied in here.
 *
 * Writing the defaults into the registered layer would FREEZE upstream's
 * behaviour into our module: Babele supplies them itself at runtime, and a
 * frozen copy stops tracking a Babele upgrade. This block only ENRICHES.
 * Fields deliberately left to the built-ins are named in each block's comment.
 */
export const ALIENRPG_MAPPINGS = {

  /* ---------------------------------------------------------------- *
   * Adventure
   *
   * All three targets are Adventure-type packs holding exactly ONE Adventure
   * document, so this block is the root of the entire corpus:
   *   alienrpg.alien-rpg-system                            (system 4.1.13)
   *   alien-evolved-starterset.alien-evolved-starter-set   (module 1.0.2)
   *   alien-evolved-corerules.alien-evolved-core-rules     (module 1.0.2)
   *
   * Left to Babele's built-ins on purpose: `folders` (nameCollection),
   * `journals` (note the path is `journal`, singular — that is upstream's, and
   * it matches the dump key), `scenes`, `macros`, `playlists`, `actors`,
   * `cards`.
   * ---------------------------------------------------------------- */
  Adventure: {
    // Re-declared byte-identically to Babele's default rather than inherited,
    // because `description` alone is 3,572 chars of visible HTML blurb and a
    // field that big should be visible in the generated runtime file too. The
    // merge is per key with identical values, so this is a no-op at runtime.
    //
    // ⚠⚠ (V-3) FOUND 2026-08-29. `name` IS A HARD-CODED LOOKUP KEY for the
    // SYSTEM pack's Adventure, 'Alien RPG System' — T-FROZEN, byte-exact.
    //   module/apps/init.mjs:9    `export const adventurePackName = "Alien RPG System"`
    //   module/apps/init.mjs:77   `await pack.getName(adventurePackName).sheet._updateObject(...)`
    //                             ** NO NULL CHECK ** — a renamed Adventure makes
    //                             FirstTimeSetup() throw and the first-run import
    //                             never completes.
    //   module/apps/init.mjs:94   `pack.index.find((a) => a.name === adventurePackName)?._id`
    //   module/apps/init.mjs:102  `if (adventure.name === adventurePackName) {`  (import hook)
    //   module/alienrpg.mjs:583   same `pack.index.find` in the release-note updater
    // The other two Adventure names ('Alien Evolved Starter Set',
    // 'Alien Evolved Core Rules') have no reader and ARE translatable — but
    // this is ONE global layer, so the freeze has to be applied per entry in
    // the cn file, not by dropping the field.
    name: 'name',                 // n=3 chars=65
    description: 'description',   // n=3 chars=3572  (1782 system + 898 starter + 892 core)
    caption: 'caption',           // n=0 chars=0  present=3 (empty on all three) — forward compat

    /**
     * `expose: true` is a REAL Babele parameter, verified at
     *   converter/document-converter.js:152  `expose: context.params?.expose === true`
     *   converter/document-converter.js:60-62
     *     `if (descriptor.expose === true) { scope.publish(key, contextPack); }`
     * It publishes this field's EmbeddedCompendium into the per-document
     * `DocumentTranslationScope`, where `Actor.items` finds it:
     *   compendium/embedded-compendium.js:26
     *     `const scopedPack = activeRuntime.localMappedCompendiumFor(documentType);`
     *   compendium/embedded-compendium.js:29-32
     *     `{...new TranslationEntries(scopedPack?.translations).snapshot(),`
     *     ` ...new TranslationEntries(translations).snapshot()}`
     *   => the scope is the BASE layer and a field's own inline translations
     *      WIN, so the 25 uncovered embedded items can still be translated
     *      under their actor and will override.
     *
     * Why it is needed at all: `_stats.compendiumSource` re-sourcing saves
     * EXACTLY 0 chars here. Measured over all 180 actor-embedded items —
     * 144 are WORLD uuids (`Item.<id>`), and
     *   converter/document-converter.js:538-539
     *     `_sourceCollection(data) { return String(this._sourceUuid(data) ?? "")`
     *     `  .match(/^Compendium[.]([^.]+[.][^.]+)[.]/)?.[1] ?? null; }`
     *   converter/document-converter.js:550
     *     `return data?.flags?.core?.sourceId ?? data?._stats?.compendiumSource ?? null;`
     * skips anything without a `Compendium.` prefix outright. The other 36
     * point at packs that no longer exist in this install
     * (alienrpg-corerules.alienrpg-talents-career x11,
     *  alien-evolved-starterset.alienrpg-weapons x13,
     *  alienrpg-corerules.alienrpg-talents-general x6,
     *  alienrpg-starterset.alienrpg-weapons x3,
     *  alienrpg-starterset.alienrpg-starter-talentscareer x2,
     *  alienrpg-corerules.alienrpg-weapons x1).
     *
     * Measured coverage WITH expose (reproduced 2026-08-29): 155/180 embedded
     * items (86.1%) and 48,181 / 52,998 mapped chars (90.9%). Per pack:
     * core 144/149, starter 11/31, system 0/0.
     * ⚠ Read the char figures with their base stated: they EXCLUDE
     * `system.notes`, which is non-empty on 84 of the 180 embedded items and is
     * the literal '[object Object]' on every one of them (15 chars each — 1,260
     * total, 1,170 of it on covered items). Counting it gives 49,351 / 54,258
     * = 91.0%, the same story with 1.3k of upstream migration residue folded
     * in. The 155/180 document figure is unaffected either way. Whichever base
     * a QA script uses, it must use the same one on both sides.
     *
     * ⚠ This is a GLOBAL behaviour change: every Adventure pack in the world
     * now publishes its item translations into scope. Benign superset for third
     * parties, but it is a change, not a no-op.
     */
    items: {                                   // 343 adventure-level Item docs across 3 packs
      path: 'items',
      converter: 'document',
      documentType: 'Item',
      cardinality: 'many',
      expose: true,
    },

    /**
     * Same trick, and it is what makes `alienRollTableRef` possible: the
     * creature actors store a RollTable NAME in `system.rTables` / `cTables`,
     * and the converter resolves it out of THIS block via
     * `runtime.localMappedCompendiumFor('RollTable')`
     * (compendium/compendium-runtime.js:104).
     *
     * Without `expose` the two fields cannot see the table names and drift is
     * guaranteed. Do not remove it while `alienRollTableRef` is referenced.
     */
    tables: {                                  // n=137 RollTable docs (3 / 11 / 123)
      path: 'tables',
      converter: 'document',
      documentType: 'RollTable',
      cardinality: 'many',
      expose: true,
    },
  },

  /* ---------------------------------------------------------------- *
   * Item — 523 documents, 13 registered subtypes
   *
   * `name`, `effects` and Babele's built-in `description:
   * 'system.description.value'` are LEFT ALONE at base level:
   *   - `name`   : identical to what we would write. n=523 chars=7986.
   *                ⚠⚠ (V-3) FOUND 2026-08-29 — TWO ITEM NAMES ARE LOOKUP KEYS
   *                and must stay English (T-FROZEN), which this block did not
   *                say. Both exist in the corerules pack as `talent` items:
   *                  'Pack Mule'    -> `i.name.toUpperCase() === "PACK MULE"` at
   *                    module/sheets/character-sheet.mjs:491,
   *                    module/sheets/synthetic-sheet.mjs:478,
   *                    module/sheets/colony-sheet.mjs:339
   *                    (it doubles the carried-weight allowance; a translated
   *                     name fails the test silently and the sheet just shows a
   *                     wrong encumbrance)
   *                  'Take Control' -> `Attrib.name.toUpperCase() === "TAKE CONTROL"`
   *                    at module/actor/old-actor-sheet.js:526 (legacy sheet only)
   *                The comparison is `.toUpperCase()`, so case is free but the
   *                bytes are not: no Chinese, no bilingual tail, no rename.
   *                The other 521 names are ordinary prose.
   *                See also the `skill-stunts` leg below for the 12 T-EXACT
   *                names, which are a DIFFERENT and stricter rule.
   *   - `effects`: 0 ActiveEffects exist anywhere in the corpus, but the
   *                built-in keeps working for foreign packs.
   *   - `description` at BASE level: the path exists on NO alienrpg item type,
   *                so it fails open here, and keeping it is what stops a dnd5e
   *                item from losing its description to this global layer. Our
   *                own `description` definitions all live in type-gated
   *                variants that shadow it only for alienrpg documents.
   *
   * The 7 `description` variants partition all 13 subtypes with no overlap:
   *   BODY_IN_ATTRIBUTES(5) + BODY_IN_GENERAL(3) + skill-stunts + planet-system
   *   + critical-injury + spacecraft-crit + colony-initiative = 13.
   * ---------------------------------------------------------------- */
  Item: {
    _variants: [

      /**
       * (V) Item `system.notes` — the proposal called this DEAD DATA and
       * refused to map it. It is a LIVE, user-editable, rendered HTMLField:
       *   module/data/base-item.mjs:11   `schema.notes = new fields.HTMLField()`
       *     (declared on the BASE class => all 13 subtypes carry it; measured
       *      present on 523/523 documents)
       *   templates/item/item-notes.hbs:7
       *     `<prose-mirror ... name='system.notes'`
       *     ` data-document-uuid='{{item.uuid}}' value='{{system.notes}}'`
       *     ` collaborate='true' toggled='true'>`
       *   module/sheets/item-sheet.mjs:401
       *     `context.enrichedNotes = await foundry.applications.ux.TextEditor`
       *     `  .enrichHTML(this.item.system.notes, {`
       *
       * ⚠ TODAY'S VALUES ARE AN UPSTREAM MIGRATION BUG, NOT CONTENT. All 267
       * non-empty values are the literal string `[object Object]` — the residue
       * of an old migration that stringified the pre-4.x `{notes: HTMLField}`
       * SchemaField into the flat HTMLField. Do NOT translate them; leave the
       * key untranslated (Babele's FieldMapping fails open on a missing key) or
       * emit the empty string. The mapping exists so that (a) the field is
       * inside the definition domain of every QA gate instead of being an
       * invisible blind spot, and (b) the day upstream fixes the migration
       * there is already a hook.
       * The proposal mapped four ALWAYS-EMPTY fields "for forward compat" and
       * omitted this one, which is the only live editor of the five.
       */
      {
        _when: when(ITEM_TYPES, 'system.notes'),
        notes: 'system.notes',
      },
      // n=267 chars=4005  present=523 — every value is '[object Object]'

      /* --- description, leg 1/7: the main item body ------------------ */
      /**
       * prose-mirror + `TextEditor.enrichHTML`; re-derived enrich sites at
       * module/sheets/item-sheet.mjs:262 / :318 / :340 / :351, templates
       * item-sheet.hbs:91, item-weapon.hbs:65, item-armor.hbs:8,
       * item-spacecraftmods.hbs:13, item-spacecraftweapons.hbs:52.
       * Counts pool adventure-level + actor-embedded items.
       *
       * ⚠ The SAME path is PLAIN TEXT in a raw <textarea> on the `vehicles`
       * ACTOR (templates/actor/vehicle-general.hbs:9). The two must never share
       * a translation-processing pipeline — see Actor's vehicles variant.
       */
      {
        _when: when(BODY_IN_ATTRIBUTES, 'system.attributes.comment.value'),
        description: 'system.attributes.comment.value',
      },
      // n=307 chars=99697  present=324
      // (item 93 · spacecraftmods 124 · weapon 61 · spacecraftweapons 20 · armor 9)

      /* --- description, leg 2/7 -------------------------------------- */
      /**
       * prose-mirror + enrichHTML at item-sheet.mjs:285 / :307 / :329;
       * templates item-talent.hbs:10, item-agenda.hbs:8, item-specialty.hbs:6.
       * Also read VERBATIM into a chat card by `_talentBtn`
       * (module/sheets/character-sheet.mjs — the survey's :1164 is off; the
       * `<p>` shape test is at :1173, see the placeholder warning below).
       * All 19 `specialty` values are empty today; the type is in the guard
       * because the schema declares the field.
       */
      {
        _when: when(BODY_IN_GENERAL, 'system.general.comment.value'),
        description: 'system.general.comment.value',
      },
      // n=100 chars=19595  present=119  (talent 85 · agenda 15 · specialty 0)

      /* --- description, leg 3/7: skill-stunts ------------------------ */
      /**
       * `skill-stunts` is the ONLY type whose data model declares
       * `system.description` (module/data/item-stunts.mjs), and it is a flat
       * HTMLField — NOT the `{value}` SchemaField Babele's built-in expects.
       * prose-mirror at templates/item/item-skill-stunts.hbs:6, enriched at
       * item-sheet.mjs:274.
       *
       * ⚠ PLACEHOLDER STRINGS ARE CONTROL FLOW. module/sheets/character-sheet.mjs
       * tests `chatData.startsWith("<h2>No Stunts Entered</h2>")` at :1125 and
       * `temp3.startsWith("<p>")` at :1173 (the survey cited :1124 / :1164 —
       * both drifted). Translate the visible words but keep the `<h2>` / `<p>`
       * shape, or the ALIENRPG.* fallback branch stops firing.
       *
       * ⚠ AND: these 12 items' NAMES are a lookup key, not prose. See the
       * T-EXACT rule in PROJECT decision 4 — the 12 names must be byte-equal to
       * `lang/cn.json`'s `ALIENRPG.Skill<key>`, no English tail, because
       * templates/actor/character-skills.hbs:15 emits
       * `data-pmbut='{{skill.description}}'` (itself overwritten every
       * `prepareDerivedData` with `game.i18n.localize('ALIENRPG.Skill<key>')`,
       * actor-character.mjs:466-468) and character-sheet.mjs:1119 does
       * `game.items.getName(dataset.pmbut)`.
       */
      {
        _when: when(['skill-stunts'], 'system.description'),
        description: 'system.description',
      },
      // n=12 chars=1957  present=12 — all 12 live in the SYSTEM pack

      /* --- description, leg 4/7: planet-system gazetteer -------------- */
      /**
       * Two guards because this variant redefines `description` AND carries 11
       * further keys: `system.misc.description.value` is the description's own
       * path, `system.details.classification.value` is a second alienrpg-only
       * discriminator so the block cannot fire on a foreign `planet-system`.
       *
       * `system.misc.description.value` is a declared HTMLField
       * (module/data/item-planet-system.mjs), prose-mirror at
       * templates/item/item-planet-description.hbs:7, enriched at
       * item-sheet.mjs:373 — present on all 68 planet-system items and EMPTY on
       * every one of them today. Mapped both directions for forward compat.
       */
      {
        _when: when(
          ['planet-system'],
          'system.misc.description.value',
          'system.details.classification.value',
        ),
        description: 'system.misc.description.value',            // n=0   chars=0     present=68
        commonName: 'system.header.commonName.value',            // n=58  chars=2045  present=68
        starSystem: 'system.header.system.value',                // n=68  chars=902
        sector: 'system.header.sector.value',                    // n=68  chars=1170
        location: 'system.header.location.value',                // n=64  chars=1675  present=68
        affiliation: 'system.details.affiliation.value',         // n=67  chars=1796  present=68
        classification: 'system.details.classification.value',   // n=67  chars=1752  present=68
        climate: 'system.details.climate.value',                 // n=67  chars=3095  present=68
        meanTemperature: 'system.details.meanTemperature.value', // n=67  chars=516   present=68
        terrain: 'system.details.terrain.value',                 // n=66  chars=2715  present=68
        colonies: 'system.details.colonies.value',               // n=67  chars=3040  present=68
        keyResources: 'system.details.keyResources.value',       // n=66  chars=2310  present=68
      },
      // whole block: 68 planet-system items, ALL in the corerules pack.
      // Sample values: commonName 'Znoy Outpost' / 'TRAON'; starSystem
      // '17 Phei Phei' / '268G.CET SYSTEM'; location '-6.8 Rimward, 2.0
      // Spinward' ('Rimward' / 'Spinward' are words); climate 'Chilly and thin
      // but breathable'; meanTemperature is mostly '16°C' but 'at night to …
      // during the day' phrases occur — translate that one with care.

      /* --- description, leg 5/7: critical-injury (forward compat) ----- */
      /**
       * (V) THE `critical-injury` FIX. 0 documents in all three packs, but the
       * data model declares both fields, so this is mapped in BOTH directions:
       *   module/data/item-crit-inj.mjs:29     `effects: new fields.HTMLField(),`
       *   module/data/item-crit-inj.mjs:19-21  `healingtime: new fields.SchemaField({`
       *                                        `  value: new fields.StringField({...}) }),`
       * Rendered: templates/item/item-crit-inj.hbs:9-11 (prose-mirror on
       * `system.attributes.effects`) and templates/item/item-header.hbs:78
       * (`<input ... name='system.attributes.healingtime.value'>`).
       * These items are created at runtime by a Critical Injuries table draw,
       * so a supplement or a played world WILL have them.
       */
      {
        _when: when(['critical-injury'], 'system.attributes.effects'),
        description: 'system.attributes.effects',             // n=0 chars=0 present=0
        healingTime: 'system.attributes.healingtime.value',   // n=0 chars=0 present=0
      },

      /* --- description, leg 6/7: spacecraft-crit (forward compat) ----- */
      /**
       * 0 documents. Both mapped fields are real and reachable:
       *   module/data/item-spacecraft-crit.mjs:18  `effects: new fields.HTMLField(),`
       *     -> prose-mirror at templates/item/item-spacecraft-crit.hbs:6
       *   module/data/item-spacecraft-crit.mjs:17  `repairroll: new fields.HTMLField({ initial: "-" }),`
       *     -> `<input ... name='system.header.repairroll'>` at item-header.hbs:167
       * Both are WRITTEN at runtime by
       *   module/documents/actor.mjs:2139-2148 `await actor.createEmbeddedDocuments("Item", [{`
       *     `type: "spacecraft-crit", ... "system.header.effects": testArray[2],`
       *     ` "system.header.repairroll": testArray[5] }]);`
       * and echoed to chat at templates/chat/crit-roll-spacecraft.hbs:11-12.
       *
       * `system.header.damage.value` (declared at item-spacecraft-crit.mjs:14-16)
       * is DELIBERATELY NOT MAPPED: a grep of the whole system for
       * `header.damage` finds the schema declaration and NOTHING else — no
       * writer in module/, no reader in templates/. Mapping it would be a key
       * no code path can ever fill or display. Recorded rather than omitted so
       * the next reader does not "complete" the block.
       */
      {
        _when: when(['spacecraft-crit'], 'system.header.effects'),
        description: 'system.header.effects',                 // n=0 chars=0 present=0
        repairRoll: 'system.header.repairroll',               // n=0 chars=0 present=0
      },

      /* --- description, leg 7/7: colony-initiative (forward compat) --- */
      /**
       * 0 documents. `system.header.comment` is a genuine HTMLField body:
       *   module/data/item-colony-initiative.mjs:13
       *     `comment: new fields.HTMLField({ required: true, blank: true }),`
       *   templates/item/item-colony-initiative.hbs:9-11 (prose-mirror)
       *
       * ⚠⚠ NEW FINDING, 2026-08-29 — NOT in the survey and NOT in the verifier.
       * The proposal also mapped `initiativeType: 'system.header.type'` on the
       * strength of it being declared `new fields.HTMLField()`
       * (item-colony-initiative.mjs:14). **It is an ENUM CODE, not prose**, and
       * mapping it breaks the sheet. Two independent readers:
       *   templates/item/item-header.hbs:193-194
       *     `<select class='select-css' name='system.header.type' ...>`
       *     `{{selectOptions config.colony_policy_list selected=system.header.type`
       *     ` labelAttr='label' localize=true}}`
       *     — the visible label comes from CONFIG, the stored value is the key.
       *   module/sheets/item-sheet.mjs:234-248 `_prepareColonyInitiativeData(item)`
       *     `switch (item.system.header.type) { case "1": ... case "2": ... case "3": ... }`
       *     picking the item's icon — reached from :196, gated on the type.
       * So it is DO-NOT-TRANSLATE, and the HTMLField declaration is upstream
       * sloppiness, not evidence. Do not add it back.
       */
      {
        _when: when(['colony-initiative'], 'system.header.comment'),
        description: 'system.header.comment',                 // n=0 chars=0 present=0
      },

      /* --- non-description free text -------------------------------- */

      /**
       * weapon class free text.
       * Measured distribution over all 78 weapons (2026-08-29):
       *   'Close Combat' x18 · 'Pistol' x15 · 'Rifle' x14 · 'Vehicle Weapon' x13
       *   · 'Heavy' x12 · 'Melee' x3 · 'Ranged' x2 · 'Heavy Weapon' x1.
       *
       * ⚠⚠ (V-3) FOUND 2026-08-29 — NOT free text end to end. THIS FIELD IS
       * COMPARED TO A STRING LITERAL IN JS, the EC terrain-enum failure mode:
       *   module/sheets/character-sheet.mjs:428
       *     `if (i.system.attributes.class.value === "RPG" || i.name.includes(" RPG ")`
       *     ` || i.name.startsWith("RPG") || i.name.endsWith("RPG")) { ammoweight = 0.5; }`
       *   module/sheets/synthetic-sheet.mjs:415  — same test
       *   module/actor/old-actor-sheet.js:701    — legacy sheet, `==` form
       * It picks the per-round ammo weight inside the encumbrance total, so a
       * translated 'RPG' silently halves the carried weight of a rocket
       * launcher and nothing errors.
       *
       * Today the risk is LATENT, not live: ZERO of the 78 weapons carry the
       * class 'RPG' (see the distribution above), so all 78 current values are
       * genuinely free text and safe to translate. The rule is forward-looking:
       *   => 'RPG' is T-FROZEN as a class value. If a supplement ships a weapon
       *      whose `class.value` is 'RPG', leave that ONE value English.
       * ⚠ The three `i.name` fallbacks in the same condition are a naming
       * constraint too: under T-BILINGUAL (`中文 English`) `startsWith('RPG')`
       * stops matching, and only `includes(' RPG ')` survives — and only when
       * the English tail keeps a space on both sides of 'RPG'.
       */
      {
        _when: when(['weapon'], 'system.attributes.class.value'),
        weaponClass: 'system.attributes.class.value',         // n=78 chars=657 present=78
      },

      /** spacecraftmods capacity free text: 'Added Hardpoint, size I'. */
      {
        _when: when(['spacecraftmods'], 'system.attributes.capacity.value'),
        capacity: 'system.attributes.capacity.value',         // n=124 chars=2015 present=124
      },

      /**
       * (V) Item `system.attributes.notes.*` — the declared free-text fields
       * the proposal left off the map with no note, while mapping four
       * always-empty ones. They are TWO DIFFERENT SHAPES on two types:
       *   module/data/item-item.mjs:64-66
       *     `notes: new fields.SchemaField({`
       *     `  value: new fields.StringField({ required: true, blank: true }),`
       *     `}),`
       *     -> `system.attributes.notes.value`, present on all 93 `item` docs
       *   module/data/item-weapon.mjs:80-81
       *     `notes: new fields.SchemaField({`
       *     `  notes: new fields.StringField({ required: true, blank: true }),`
       *     -> `system.attributes.notes.notes`, present on all 78 `weapon` docs
       * Both empty everywhere, and a grep finds ZERO readers in templates/ or
       * module/ — they are declared-but-unwired. Mapped anyway, in BOTH
       * directions, under ONE translation key on disjoint type sets, so the
       * forward-compat policy is applied consistently instead of selectively.
       * If they are ever wired up they are already inside every QA gate.
       */
      {
        _when: when(['item'], 'system.attributes.notes.value'),
        attributeNotes: 'system.attributes.notes.value',      // n=0 chars=0 present=93
      },
      {
        _when: when(['weapon'], 'system.attributes.notes.notes'),
        attributeNotes: 'system.attributes.notes.notes',      // n=0 chars=0 present=78
      },
    ],
  },

  /* ---------------------------------------------------------------- *
   * Actor — 74 documents, 8 registered subtypes
   *
   * `name`, `items`, `effects` and `tokenName` are LEFT to Babele's built-ins.
   *
   * (V) `tokenName` in particular: the proposal's `alienTokenName` converter is
   * DROPPED ENTIRELY. Babele's built-in is
   *   `tokenName: {path:'prototypeToken.name', converter:'name'}`
   * and `name` is `Converters.mappedField('name')`:
   *   converter/converters.js:92-97
   *     `mappedField(field) {`
   *     `  return (_value, _translation, data, tc, _allTranslations, runtime = {}) => {`
   *     `    const contextCompendium = this.currentCompendium(tc, runtime);`
   *     `    return contextCompendium?.translateField(field, data, runtime); }; }`
   * — the `tokenName` translation (2nd arg) is DISCARDED and the translated
   * `name` is returned. That is already the correct behaviour here, because
   * `prototypeToken.name === name` for 74/74 actors (measured). The function
   * `mappedField` returns has NO `.extract` property, so
   *   converter/functional-converter.js:63-66
   *     `extract(context) {`
   *     `  if (typeof this.fn.extract !== "function") { return undefined; } ... }`
   * makes Babele's own export OMIT the key. `alienTokenName.extract` would have
   * re-introduced 74 redundant `tokenName` keys (~1,847 chars) that a
   * translator must fill in twice or delete.
   *
   * ⚠ The extract direction needs the same treatment for a different reason —
   * see EXTRACTOR_DEVIATIONS.Actor.tokenName.
   *
   * ⚠ Babele's built-in `Actor.description: 'system.details.biography.value'`
   * is left in place at base level and is DEAD here: alienrpg has no
   * `system.details` on actors at all (measured 0/74). It fails open, and
   * keeping it is what stops this global layer from breaking dnd5e actors.
   * ---------------------------------------------------------------- */
  Actor: {
    _variants: [

      /**
       * The base-actor HTMLField, on the 6 types whose sheet renders it.
       * Largest actor field after `special.value`.
       */
      {
        _when: when(NOTES_RENDERED_TYPES, 'system.notes'),
        notes: 'system.notes',
      },
      // n=45 chars=40265  present=68
      // (creature 30 · character 7 · spacecraft 7 · synthetic 1; 38 of the 45
      //  contain HTML tags). colony / planet have 0 documents.

      /**
       * (V) `territory` — FORWARD COMPAT ONLY, in its own variant so the reason
       * is not silently inherited from the block above.
       * `alienrpgTerritory` (module/data/actor-territiory.mjs — note upstream's
       * filename typo) extends `alienrpgActorBase` at :9, so it inherits
       * `schema.notes` (base-actor.mjs:10). But **territory-sheet.mjs contains
       * zero occurrences of the string `notes`** — no `enrichedNotes`, no
       * prose-mirror, no code path that renders it. The survey put territory in
       * the main notes variant on the strength of a `_enrichTextFields`
       * function that does not exist anywhere in the system. 0 territory
       * documents today.
       */
      {
        _when: when(['territory'], 'system.notes'),
        notes: 'system.notes',                               // n=0 chars=0 present=0
      },

      /**
       * ⚠ `vehicles` — notes is ONE LEVEL DEEPER. Separate variant, mandatory.
       *   module/data/actor-vehicle.mjs:126-128
       *     `schema.notes = new fields.SchemaField({`
       *     `  notes: new fields.StringField({ required: true, blank: true }),`
       *     `})`
       *   (the survey's :130 is `schema.crew`)
       * All 6 vehicle actors carry `system.notes = {notes: ""}` — an OBJECT.
       * `PrimitiveConverter._invalidStaticTranslation` only type-guards
       * PRIMITIVE originals, so mapping `system.notes` for vehicles would write
       * a string straight over the SchemaField and replace it with a scalar.
       * Nothing renders it either (vehicle-sheet.mjs has no `notes` hit), so
       * this is forward compat with a data-integrity motive.
       */
      {
        _when: when(['vehicles'], 'system.notes.notes'),
        notes: 'system.notes.notes',                         // n=0 chars=0 present=6
      },

      /**
       * creature block. `system.general.special.value` is the single largest
       * actor field in the corpus.
       *
       * ⚠ PLAIN TEXT in a raw <textarea>: templates/actor/creature-general.hbs:14
       * and templates/actor/crt/crtui-creature-general.hbs:11. Newlines are the
       * paragraph separator and HTML must NOT be introduced — `<p>` tags render
       * literally on the sheet.
       *
       * ⚠ `system.rTables` / `system.cTables` hold a RollTable **NAME**, not an
       * id, and are compared verbatim. Re-derived readers:
       *   module/helpers/rollTableData.mjs:6-21 `rTableget()`
       *     :7  `const folder = game.folders.contents.find((x) => x.name === "Alien Creature Tables")`
       *     :12 `lTables[0] = { key: "None", label: "None" }`
       *     :16-17 `key: aTables[index].name, label: aTables[index].name,`
       *   module/helpers/rollTableData.mjs:23-39 `cTableget()`
       *     :24 `const folder = game.folders.contents.find((x) => x.name === "Alien Mother Tables")`
       *     :27 `const aTables = folder.contents.filter((x) => x.name.startsWith("Critical Injuries"))`
       *   module/documents/actor.mjs:1839 `atable = game.tables.getName(dataset.atttype);`
       *   module/documents/actor.mjs:2497 `const table = game.tables.contents.find((b) => b.name === targetTable);`
       *
       * Translating them by hand drifts from the translated table names and
       * breaks BOTH the <select> selection (selectOptions compares against the
       * option key) and the creature attack / critical-injury roll (`getName`
       * returns null -> the `ALIENRPG.NoCharCrit` warning at :1841, or the
       * `find` at :2497 returns undefined and `table.roll` throws).
       * `alienRollTableRef` removes the drift by resolving out of the same
       * Adventure `tables` block, which is why that block carries `expose`.
       *
       * MEASURED VALUES (V) — these are NOT all 'None':
       *   rTables: 30 docs / 805 chars / 13 distinct — 'EV - Chestburster
       *     Attacks' x5, 'EV - Stalker, Scout and Drone Attacks' x5,
       *     'EV - Facehugger Attacks' x3, 'EV - Praetorian, Charger and Queen
       *     Attacks' x3, ... and 'None' x3. All 12 non-sentinel values exist as
       *     a RollTable in the packs (verified name by name).
       *   cTables: 30 docs / 147 chars — 'None' x29 and
       *     'Critical Injuries on Xenomorphs' x1.
       *
       * ⚠ 'None' IS A SENTINEL, compared verbatim at
       *   module/documents/actor.mjs:2493 `if (targetTable === "None") {`
       * and produced as a literal option key at rollTableData.mjs:12 / :30.
       * It must NEVER be translated. `EXTRACT_CONVERTERS.alienRollTableRef`
       * below keeps it out of the English baseline entirely (32 leaves) so a
       * translator is never shown it.
       */
      {
        _when: when(['creature'], 'system.general.special.value'),
        special: 'system.general.special.value',             // n=30 chars=81617 present=30
        rollTable: { path: 'system.rTables', converter: 'alienRollTableRef' }, // n=30 chars=805
        critTable: { path: 'system.cTables', converter: 'alienRollTableRef' }, // n=30 chars=147
      },

      /**
       * character / synthetic block.
       * Guarded on `system.general.appearance.value` — present on all 31
       * character + synthetic actors, absent in every other system.
       *
       * (V) The survey claimed sigItem / relOne / relTwo are all literally
       * 'None'. False for all three; re-measured over all 74 actors:
       *   sigItem: n=6 chars=58  — 'Toy dinosaur', 'Company ID badge',
       *            'Cross necklace', 'Lab coat', 'None' x2
       *   relOne : n=7 chars=41  — 'Hirsch' x2, 'MacWhirr', 'Sigg',
       *            'Singleton', 'None' x2
       *   relTwo : n=7 chars=40  — 'Hirsch' x2, 'MacWhirr' x2, 'Sigg',
       *            'None' x2
       * Only `agenda` is genuinely all-'None' (2 docs / 8 chars).
       *
       * ⚠⚠ CROSS-REFERENCE HAZARD — relOne / relTwo are character SURNAMES that
       * name OTHER pregen actors in the same pack:
       *   'Hirsch'    -> actor 'Morgan Hirsch (Evolved)'
       *   'MacWhirr'  -> actor 'Janice Macwhirr (Evolved)'   <- note the case
       *   'Sigg'      -> actor 'Sonny Sigg (Evolved)'
       *   'Singleton' -> actor 'Hannah Singleton (Evolved)'
       * They MUST track the translated Actor names or the pregen relationship
       * lines point at people who no longer exist by that name. They are NOT
       * lookup keys — every reader is a plain `<input type='text'>`
       * (templates/actor/character-general.hbs:231 / :240 / :245,
       *  character-enhanced-general.hbs:168 / :177 / :182,
       *  crt/crtui-character-general.hbs:244 / :252) and nothing compares them
       * — which is exactly why the drift would be SILENT. The 'MacWhirr' vs
       * 'Macwhirr' case mismatch proves upstream already treats them as free
       * text, so no automated sync is possible: this is a translator rule.
       * => QA: assert every non-'None' relOne / relTwo value occurs as a
       *    substring of some translated Actor name in the same pack file.
       *
       * ⚠ 'None' here is NOT the rollTable sentinel — nothing compares these
       * fields — but translating it while `agenda`'s 'None' stays English would
       * look broken. Treat all four consistently.
       */
      {
        _when: when(['character', 'synthetic'], 'system.general.appearance.value'),
        appearance: 'system.general.appearance.value',       // n=8 chars=1062 present=31
                                                             // plain multi-line ('Age: 37 \n Lin is a mess…')
        adhocItems: 'system.adhocitems',                     // n=8 chars=358 present=31 — HTML,
                                                             // prose-mirror + enrichHTML (character-sheet.mjs)
        signatureItem: 'system.general.sigItem.value',       // n=6 chars=58 present=31
        agenda: 'system.general.agenda.value',               // n=2 chars=8  present=31 (both 'None')
        relationshipOne: 'system.general.relOne.value',      // n=7 chars=41 present=31
        relationshipTwo: 'system.general.relTwo.value',      // n=7 chars=40 present=31
      },

      /**
       * vehicles block. `system.attributes.comment.value` is at the SAME PATH
       * that is rich prose-mirror HTML on the Item side — deliberately a
       * separate variant, because here it is PLAIN TEXT in a raw <textarea>:
       *   templates/actor/vehicle-general.hbs:9
       */
      {
        _when: when(['vehicles'], 'system.attributes.comment.value'),
        comment: 'system.attributes.comment.value',          // n=6 chars=6038 present=6
        misc: 'system.general.misc.value',                   // n=0 chars=0 present=6
      },

      /** spacecraft block. */
      {
        _when: when(['spacecraft'], 'system.attributes.manufacturer'),
        misc: 'system.general.misc.value',                   // n=0 chars=0 present=7
        manufacturer: 'system.attributes.manufacturer',      // n=7 chars=68  ('Weyland-Yutani', 'Lockmart')
        model: 'system.attributes.model',                    // n=7 chars=96  ('FRIGATE/CONESTOGA', 'BISON / M')
        ai: 'system.attributes.ai',                          // n=7 chars=73  ('MU/TH/UR 9000' — proper noun)
        modules: 'system.attributes.modules',                // n=7 chars=235 ('5 x size V 7 x size IV …')
        armaments: 'system.attributes.armaments',            // n=6 chars=44  present=7
      },

      /**
       * colony block — 0 documents in all three packs; mapped from the data
       * model so a colony supplement is covered.
       *
       * (V) The proposal stopped after location / mission / established /
       * sponsor / commander / economydirector. The SIX free-text fields it
       * omitted are declared in the same SchemaField and are the user-facing
       * LABELS of the colony development tracks — exactly what a supplement
       * ships in English:
       *   module/data/actor-colony.mjs:24  `cycles: new fields.StringField({ required: true, blank: true }),`
       *   module/data/actor-colony.mjs:31  `potentialname: ...`
       *   module/data/actor-colony.mjs:32  `productivityname: ...`
       *   module/data/actor-colony.mjs:33  `maintenancename: ...`
       *   module/data/actor-colony.mjs:34  `sciencename: ...`
       *   module/data/actor-colony.mjs:35  `spiritname: ...`
       * A partial forward-compat block is worse than none because it looks
       * complete. `system.stats.notes` (actor-colony.mjs:73) is included for
       * the same reason. `system.header.type.value` / `.label` (:13-16) are NOT
       * mapped — enum + derived label, same family as the Item case.
       */
      {
        _when: when(['colony'], 'system.attributes.mission'),
        misc: 'system.general.misc.value',
        colonyName: 'system.header.colonyname',                 // actor-colony.mjs:17
        location: 'system.attributes.location',                 // :21
        mission: 'system.attributes.mission',                   // :22
        established: 'system.attributes.established',           // :23
        cycles: 'system.attributes.cycles',                     // :24  (V) added
        sponsor: 'system.attributes.sponsor',                   // :25
        commander: 'system.attributes.commander',               // :29
        economyDirector: 'system.attributes.economydirector',   // :30
        potentialName: 'system.attributes.potentialname',       // :31  (V) added
        productivityName: 'system.attributes.productivityname', // :32  (V) added
        maintenanceName: 'system.attributes.maintenancename',   // :33  (V) added
        scienceName: 'system.attributes.sciencename',           // :34  (V) added
        spiritName: 'system.attributes.spiritname',             // :35  (V) added
        statsNotes: 'system.stats.notes',                       // :73
      },
      // whole block: n=0 chars=0 present=0 — 0 colony actors in any pack

      /**
       * planet block — 0 documents. All paths re-derived from
       * module/data/actor-planet.mjs (line numbers in trailing comments); every
       * one is `new fields.StringField({ required: true, blank: true })`.
       * `system.header.type.value` / `.label` (:13-16) excluded: enum + derived.
       */
      {
        _when: when(['planet'], 'system.attributes.parentstar'),
        misc: 'system.general.misc.value',                   // :68-69
        colonyName: 'system.header.colonyname',              // :17
        parentStar: 'system.attributes.parentstar',          // :21
        radiation: 'system.attributes.radiation',            // :38
        planetSize: 'system.attributes.planetsize',          // :39
        atmosphere: 'system.attributes.atmosphere',          // :40
        hydrosphere: 'system.attributes.hydrosphere',        // :41
        dayLength: 'system.attributes.daylength',            // :42
        axialTilt: 'system.attributes.axialtilt',            // :43
        gravity: 'system.attributes.gravity',                // :44
        climate: 'system.attributes.climate',                // :45
        globalFeature: 'system.attributes.globalfeature',    // :46
        orbitalPeriod: 'system.attributes.orbitalperiod',    // :47
        personality: 'system.attributes.personality',        // :48
        northPole: 'system.attributes.northpole',            // :49
        equator: 'system.attributes.equator',                // :51
        southPole: 'system.attributes.southpole',            // :53
        nhWest: 'system.attributes.nhwest',                  // :55
        nhSouth: 'system.attributes.nhsouth',                // :57
        nhEast: 'system.attributes.nheast',                  // :59
        shEast: 'system.attributes.sheast',                  // :61
        shWest: 'system.attributes.shwest',                  // :63
      },
      // whole block: n=0 chars=0 present=0 — 0 planet actors in any pack

      /**
       * territory block — 0 documents.
       *   module/data/actor-territiory.mjs:19-22  `schema.sectors = new fields.SchemaField({value, label})`
       *   module/data/actor-territiory.mjs:23-26  `schema.comment = new fields.SchemaField({value, label})`
       * The `.label` halves are NOT mapped (same family as every other
       * `.label`: never rendered, or overwritten from CONFIG).
       */
      {
        _when: when(['territory'], 'system.sectors.value'),
        sectors: 'system.sectors.value',                     // n=0 chars=0 present=0
        comment: 'system.comment.value',                     // n=0 chars=0 present=0
      },
    ],
  },

  /* ---------------------------------------------------------------- *
   * Scene — 15 documents
   *
   * `name`, `drawings` (textCollection), `notes` (textCollection) and `regions`
   * (document -> Region) keep Babele's built-ins.
   *
   * ⚑ `regions[].name` — listed as missing by the task. It is NOT missing:
   * Babele's default `Scene.regions` document converter plus the default
   * `Region: {name, behaviors}` block already carries it, and the extractor
   * reaches it because for an Adventure pack the regions are nested inside the
   * Adventure document (and `extract_en.mjs:554` re-attaches the
   * `!scenes.regions!` bucket for non-Adventure packs). Measured: 4 regions,
   * all in starterset, 69 chars — 'HH Level 2 Stairs', 'HH Level 1 stairs',
   * 'Tavern Ground Floor', 'Tavern Top Floor'. Re-declaring it here would be a
   * byte-identical no-op; it is recorded instead so nobody "fixes" it twice.
   *
   * ⚑ `notes` (map pins): 37 notes / 349 chars, ALL in starterset. (V) 25 of
   * them carry an `entryId` + `pageId` (the survey said 33), so for those 25
   * the pin text is an override LABEL over the journal page name and the two
   * must be translated consistently; the other 12 are standalone strings.
   *
   * ⚑ `drawings`: 0 in all 15 scenes. Mapping present, corpus empty.
   * ---------------------------------------------------------------- */
  Scene: {
    /**
     * The canvas navigation bar name. Genuinely absent from Babele 2.9.1's
     * Scene default (`mapping/default-mappings.js:202-218` has only
     * name / drawings / notes / regions). Empty on all 15 scenes, declared in
     * BOTH directions so a future named scene is not an invisible blind spot —
     * the same class of gap that hid `Scene.levels` from the EC project for
     * months.
     *
     * ⚠ (V-3) SCOPE NOTE, added 2026-08-29. `navName` and `tokens` are the ONLY
     * two entries in this whole layer declared at BASE level rather than inside
     * a `when(...)` variant, so they are the only two that reach EVERY Scene in
     * EVERY Babele-managed pack in the world, alienrpg or not. That is
     * defensible — both are core Foundry Scene fields, not alienrpg paths, so
     * this is a superset of Babele's default rather than a foreign-system leak
     * — but it IS the unguarded-global case the `when()` doctrine above
     * otherwise forbids, and it was verified empirically rather than assumed:
     * a scan of every dnd5e / pf2e / crucible pack on this install (2026-08-29)
     * finds ZERO documents matching any `_when` in this layer, so the variants
     * are clean and these two base keys are the entire global footprint.
     * Translate-time cost on a foreign Scene is nil (missing key -> undefined ->
     * `mapping/field-mapping.js:58-68 map()` skips the field); export-time cost
     * is two extra keys per foreign Scene.
     */
    navName: 'navName',                                      // n=0 chars=0 present=15

    /**
     * Placed-token override names. 0 tokens in all 15 scenes today.
     *
     * ⚠⚠ CONFLICT, and it is why this is declared with its eyes open rather
     * than quietly. Babele post-processes Adventure imports:
     *   babele.js:52-70
     *     `Hooks.on('importAdventure', (_adventure, _options, created = {}) => {`
     *     :53 `if (game.settings.get("babele","syncImportedAdventureTokenNames") === false) { return; }`
     *     :60 `if (!createdActors.has(token.actorId) || token.delta?.name) { continue; }`
     *     :66 `token.update({name: actor.prototypeToken.name});`
     *   foundry/settings.js:62-72 — scope 'world', **default true**.
     * The hook runs AFTER the documents are created, so for a token whose actor
     * came from the same adventure and which has no `delta.name`, it OVERWRITES
     * whatever this `nameCollection` translation wrote.
     *
     * Today that is harmless and in fact desirable: `prototypeToken.name ===
     * name` for 74/74 actors and we let Babele's built-in `tokenName` set
     * `prototypeToken.name` to the translated `name`, so the hook writes the
     * TRANSLATED actor name onto the token — the outcome we want anyway.
     * It only bites if upstream later ships a placed token whose name
     * deliberately differs from its actor's (EC measured 128 of 130 such names
     * in Ember). If that day comes the fix is one of: set
     * `syncImportedAdventureTokenNames` to false in the hub module, or give the
     * token a `delta.name`. Recorded so it is a decision, not a surprise.
     */
    tokens: { path: 'tokens', converter: 'nameCollection' }, // n=0 chars=0 (0 tokens in 15 scenes)
  },
};

/**
 * ⚠ THE THREE "LAYERS" ARE ONE OBJECT, AND THAT IS THE POINT.
 *
 * PROJECT decision 1: there is a SINGLE global `babele.registerMapping` layer
 * and it lives in the hub module `alienrpg-cn`; the two content modules ship
 * `compendium/cn` files only. `4-常用脚本/release/generate_runtime.mjs:35-49`
 * already encodes that — it reads exactly one export, `ALIEN_LAYER`, and emits
 * exactly one `babele-mappings.js`.
 *
 * So `STARTERSET_LAYER` and `CORERULES_LAYER` are the SAME object, not copies.
 * Three separate layer objects would be a fiction: `registerMapping` cannot
 * scope a layer to a pack, so a divergence between them could never take
 * effect — it would only mean the English baselines for the three packs were
 * extracted under three different rules while the runtime applied one. That is
 * the exact "extract a different key set than the runtime looks up" failure the
 * two-consumers-one-definition design exists to prevent.
 *
 * It also satisfies the folder risk directly: the four folder `_id`s
 * Btepu5tifRV0Pj7w / LrpFZIuICZfNAr2v / A0N3Ct1TQq9BuGk6 / w3xO69hCmTtYDF8d are
 * IDENTICAL in all three packs (plus 8 more shared across two — measured 12
 * shared-id groups over 49 folder docs / 31 distinct names / 755 chars), so
 * whichever adventure is imported last wins. One layer => one `folders` rule.
 *
 * The exports are deep-frozen so an accidental in-place edit of one name
 * cannot silently diverge the other two.
 */
function deepFreeze(o) {
  if (o && typeof o === 'object' && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o)) deepFreeze(v);
  }
  return o;
}
deepFreeze(ALIENRPG_MAPPINGS);

/** Contract name required by `release/generate_runtime.mjs:35`. */
export const ALIEN_LAYER = ALIENRPG_MAPPINGS;
/** Same object — see the note above. Not a copy, deliberately. */
export const STARTERSET_LAYER = ALIENRPG_MAPPINGS;
/** Same object — see the note above. Not a copy, deliberately. */
export const CORERULES_LAYER = ALIENRPG_MAPPINGS;

/* ================================================================== *
 * 2. Converter contract
 * ================================================================== */

/**
 * The named converters `ALIENRPG_MAPPINGS` references, and what the runtime
 * half must do. This is DATA, not an implementation: `ALIEN_LAYER` is
 * `JSON.stringify`-d by `generate_runtime.mjs`, so a function here could not
 * travel with it anyway, and a second copy of the converter would be a drift
 * hazard. The implementation belongs in
 * `4-常用脚本/release/runtime-converters.js`, which the generator inlines
 * verbatim into `babele-mappings.js`.
 *
 * ⚠⚠ AS OF 2026-08-29 `release/runtime-converters.js` IS STILL THE VERBATIM
 * EMBER/CRUCIBLE COPY (it exports crucibleDescription / crucibleNested /
 * crucibleActions / crucibleTokenName / emberEncounterTokenNames). It does NOT
 * export `alienRollTableRef`. Porting it is a separate task; until it lands,
 * `system.rTables` / `system.cTables` have no runtime half. The EXTRACT half is
 * supplied below and works today.
 *
 * A QA script should assert that
 *   Object.keys(PROJECT_CONVERTERS) is a superset of
 *   Object.keys(ALIENRPG_CONVERTER_CONTRACT)
 * so a mapping can never name a converter nobody registered.
 *
 * Registration is `babele.init`-only for both mappings and converters:
 *   babele.js:197-198  `registerConverters(converters) {`
 *                      `  this.#assertConfigurable("registerConverters"); ... }`
 *   babele.js:291-292  `registerMapping(mapping) {`
 *                      `  this.#assertConfigurable("registerMapping"); ... }`
 *   babele.js:984      `#assertConfigurable(operation)`
 * Registering from a `ready` hook throws.
 *
 * Functional-converter signature (converter/functional-converter.js:44-53):
 *   fn(value, translation, source, contextCompendium, allTranslations, runtime, params)
 */
export const ALIENRPG_CONVERTER_CONTRACT = {
  alienRollTableRef: {
    usedBy: ['Actor.rollTable (system.rTables)', 'Actor.critTable (system.cTables)'],
    translate:
      "Return `translation` when the actor's own entry supplies one. Otherwise resolve "
      + 'the ENGLISH table name against the RollTable EmbeddedCompendium published by '
      + 'Adventure.tables (expose:true): '
      + "runtime.localMappedCompendiumFor('RollTable')?.translateField('name', {name: value}, runtime). "
      + "Fall back to the untranslated value. NEVER translate the literal 'None' "
      + '(sentinel, module/documents/actor.mjs:2493).',
    extract:
      "Emit the raw value, except the literal 'None'. Implemented below in "
      + 'EXTRACT_CONVERTERS so the English baseline never shows a translator the sentinel.',
    dependsOn: 'Adventure.tables must keep `expose: true`.',
  },
};
deepFreeze(ALIENRPG_CONVERTER_CONTRACT);

const isStr = (v) => typeof v === 'string' && v.trim().length > 0;

/**
 * The extract-direction half of the custom converters, per the
 * `extract_en.mjs` contract (its header lines 48-53):
 *   { converterName: (value, ctx) => extracted | undefined }
 *   ctx = {spec, doc, documentType, ctx, extractDocument, toArray}
 *
 * ⚠ The property really is `ctx`, nested inside `ctx` — read off the CALL SITE,
 *   extract_en.mjs:353
 *     `const v = custom(value, { spec, doc, documentType, ctx, extractDocument, toArray });`
 *   not off that file's own header comment at :50, which says `mappings` and is
 *   the half that drifted. Nothing here reads the property, so the discrepancy
 *   is inert today; it is recorded so nobody "fixes" the correct line to match
 *   the wrong one. If a future converter needs the extractor context, verify
 *   against :353 again first.
 *
 * Without an entry here, `extract_en.mjs:479-488` degrades an unknown converter
 * to a plain path read AND counts it into the `unknownConverters` warning block
 * — never silently, but the raw read would put the 32 'None' sentinel leaves
 * (3 rTables + 29 cTables) into the English baseline, where a translator would
 * quite reasonably translate them and break the creature attack roll.
 *
 * ⚑ DELIBERATELY NOT SHIMMED: `referencedDocumentField`.
 * Babele's built-in `TableResult.name` converter lands in that same `default:`
 * branch, and its extract behaviour there (a plain read of `spec.path`, i.e.
 * `name`) is byte-identical to what a shim would do — the extractor's own
 * comment at :480-484 says so. Shimming it would only SUPPRESS a diagnostic the
 * extractor's author wants visible. Instead, assert the positive expectation:
 * `_source.json.unknownConverters` must contain EXACTLY ONE key,
 * `TableResult.name:referencedDocumentField`, with value
 *   alienrpg 31 · starterset 124 · corerules 1375   (one per TableResult doc)
 * Any other key means a converter was named and never wired up.
 */
export const EXTRACT_CONVERTERS = {
  alienRollTableRef: (value) => (isStr(value) && value !== 'None' ? value : undefined),
};

/* ================================================================== *
 * 3. Babele 2.9.1 defaults — VERBATIM SNAPSHOT
 * ================================================================== */

/**
 * A byte-faithful mirror of
 *   C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/babele/script/mapping/default-mappings.js
 * as shipped with Babele 2.9.1, transcribed 2026-08-29 with quoting normalised
 * to this file's style and NOTHING else changed.
 *
 * It is VERBATIM on purpose. EC's copy carried three silent deviations inside
 * the mirror itself, and "we thought the mirror was faithful" is its own defect
 * class: `effectiveMappings()` merges per documentType, so one missing key
 * means the extractor and the runtime work from different key sets. Keeping
 * this block literal makes the mirror trivially checkable — a QA script can
 * deep-equal it against the installed `default-mappings.js` and fail on a
 * Babele upgrade.
 *
 * Every deviation this project wants lives in `EXTRACTOR_DEVIATIONS` below,
 * where it is named, reasoned and measured.
 *
 * ⚠ THIS TABLE DOES NOT REACH THE RUNTIME. `release/generate_runtime.mjs`
 * serialises `ALIEN_LAYER` only; Babele supplies its own defaults at runtime.
 * Everything here affects ONLY the English baseline `extract_en.mjs` produces.
 *
 * ---------------------------------------------------------------------------
 * CROSS-ALIGNMENT, BLOCK BY BLOCK, AGAINST THIS PROJECT (measured, 3 packs)
 * ---------------------------------------------------------------------------
 *   Adventure   name 3/65 · description 3/3572 · caption 0/0 · folders 49/755 ·
 *               journals 13 · scenes 15 · macros 5 · tables 137 · items 343 ·
 *               actors 74 · cards 0 · playlists 0.
 *               Our layer adds `expose:true` to items and tables and
 *               re-declares name / description / caption byte-identically.
 *   Actor       name 74/1847 · description ('system.details.biography.value')
 *               **0 documents — dead here, kept for foreign packs** ·
 *               items / effects fine · tokenName -> EXTRACTOR_DEVIATIONS.
 *   ActiveEffect / Cards / Card / Playlist / PlaylistSound
 *               **0 documents anywhere**. Kept so the mirror is complete and so
 *               `Adventure.cards` / `playlists` have a childMapping.
 *   Folder      `{}` upstream. Folder names arrive through `Adventure.folders`
 *               (nameCollection): 49 docs / 31 distinct / 755 chars.
 *               ⚠ Do NOT add a `folders` block at the ROOT of the emitted
 *               translation files: `FolderTranslations`
 *               `.translateImportedCompendiumFolder` (hooked on `createFolder`,
 *               babele.js:72-74) renames imported world folders from the
 *               pack-level `folders` payload, independently of this mapping,
 *               and folder names would then be rewritten twice from two
 *               different sources. `extract_en.mjs:556-562` writes
 *               `folders: {}` for an Adventure pack because the `!folders!`
 *               bucket is empty there — an empty object is inert; anything
 *               inside it is a bug.
 *               ⚠⚠ HARD RULE, unrelated to Babele: 'Alien Creature Tables' and
 *               'Alien Mother Tables' MUST STAY ENGLISH.
 *               `module/helpers/rollTableData.mjs:7` and `:24` do
 *               `game.folders.contents.find((x) => x.name === "<English>")` and
 *               then read `folder.contents` with NO null check — a rename makes
 *               `folder` undefined and the creature sheet render throws. The
 *               two system-pack macros compare `t.folder.name` against the same
 *               two literals.
 *               ⚠⚠ (V-3) CORRECTED 2026-08-29. An earlier revision of this
 *               line said "'Alien Tables' / 'Alien Sub-Tables' are safe."
 *               THAT IS WRONG — ALL FOUR `Alien *Tables` folder names are
 *               hard-coded lookups and ALL FOUR are T-FROZEN:
 *                 module/apps/init.mjs:49
 *                   `if (!game.settings.get(moduleKey,"imported") && game.user.isGM`
 *                   `    && !game.folders.getName("Alien Tables")) { FirstTimeSetup() }`
 *                   — the folder-name probe is the second half of the
 *                   already-imported guard. Translate it and a world that was
 *                   imported manually (so the `imported` setting is still
 *                   false) re-runs FirstTimeSetup on every load.
 *                 module/apps/migratefolders.js:17-38 `acFolderIDs` lists
 *                   'Alien Creature Tables' / 'Alien Mother Tables' /
 *                   'Alien Sub-Tables' / 'Alien Tables' with their canonical
 *                   folder ids, and :52-59 matches on
 *                   `a.name === folderName && a.id !== folderID` to collect the
 *                   stale folder's contents for re-parenting (:64-80). A
 *                   translated name makes the match fail, the re-parent is
 *                   skipped, and the assets are orphaned in the old folder —
 *                   silently, since nothing throws.
 *               The other 27 folder names have no reader and are translatable.
 *   Item        name 523/7986 · description ('system.description.value')
 *               **path exists on no alienrpg item type — dead here, kept for
 *               foreign packs** · effects 0.
 *   JournalEntry name 13/305 · description ('content') **0 documents: `content`
 *               was removed in Foundry v10, all text lives in pages** ·
 *               categories empty on all 13 · pages 151.
 *               ⚠⚠ (V-3) FOUND 2026-08-29 — ONE OF THE 13 NAMES IS A HARD-CODED
 *               LOOKUP AND IS T-FROZEN: `'MU/TH/ER Instructions.'` (system
 *               pack, note the trailing period is part of the string).
 *                 module/apps/init.mjs:14
 *                   `export const welcomeJournalEntry = "MU/TH/ER Instructions."`
 *                 module/apps/init.mjs:81  `game.journal.getName(welcomeJournalEntry).show()`
 *                 module/apps/init.mjs:107 same call inside the importAdventure hook
 *                 module/apps/migratefolders.js:120
 *                   `game.journal.getName("MU/TH/ER Instructions.").show()`
 *                 module/alienrpg.mjs:571 `const releaseNoteName = "MU/TH/ER Instructions.";`
 *                   :574 `game.journal.getName(releaseNoteName)` (null-checked)
 *                   :592 `game.journal.getName(releaseNoteName).id`   ** NO NULL CHECK **
 *                   :613 `game.journal.getName(releaseNoteName)`
 *               Three of those six sites dereference the result with no null
 *               check, so a translated name is a TypeError thrown immediately
 *               after a successful import, and again on every release-note
 *               migration. The other 12 JournalEntry names have no reader.
 *               Give it the T-FROZEN treatment, not T-BILINGUAL: `getName` is
 *               an exact match, so an appended Chinese half breaks it too.
 *   JournalEntryPage name 151/2256 · text 59/2,230,280 (82.6% of the corpus) ·
 *               caption 0 (present on 9, all empty) · src -> EXTRACTOR_DEVIATIONS ·
 *               width / height (video.width / video.height) are inert and stay
 *               in the mirror. (V-3) CORRECTED 2026-08-29: an earlier revision
 *               said they are "`null` on all 151". They are **undefined** on
 *               all 151 — `video` exists on every page but holds only
 *               `{controls, volume}`, so the path does not resolve at all
 *               (present=0, not present=151). Same inert outcome, different
 *               reason, and the distinction matters because `present` in this
 *               file means "the path resolves, empty and null included".
 *   Macro       name 5/177 · command -> EXTRACTOR_DEVIATIONS.
 *   RollTable   name 137/3640 · description 16/888 · results 1530.
 *               ⚠⚠ 11 table names are hard-coded in system JS and must stay
 *               byte-identical: actor.mjs:554 'Panic Table', :845 'Stress
 *               Response Table', :1067 'Panic Response Table',
 *               :1812 `getName(localize("ALIENRPG.EVCriticalInjuries")) || getName("EV - Critical Injuries")`,
 *               :1815-1817 `getName(localize("ALIENRPG.CriticalInjuries")) || getName("Critical Injuries") || getName("Critical injuries")`,
 *               :1830 'Critical Injuries on Synthetics' (plus a lowercase
 *               retry), :1848 'Spaceship Minor Component Damage',
 *               :1855 'Spaceship Major Component Damage', plus
 *               'EV - 48a. LS - DANGER EVENT DETAIL' inside the corerules
 *               macro. Two have an i18n escape hatch via lang/cn.json;
 *               (V) `ALIENRPG.EVCriticalInjuries` is PRESENT in cn.json with a
 *               JSON `null` value (not missing) — Foundry's `localize` only
 *               accepts a string and falls through to English, so the practical
 *               constraint is unchanged, but do not "fix" the file by filling
 *               that key in.
 *               ⚠ `cTableget()`'s `startsWith("Critical Injuries")` filter
 *               (rollTableData.mjs:27) means those table names need a
 *               PREFIX-PRESERVING bilingual form, not a free translation.
 *   TableResult _identity {export:['range','_id'], match:['_id','range']} ·
 *               name 806/9925 — the `referencedDocumentField` fallback can
 *               NEVER fire here: all 45 `documentUuid` values are WORLD uuids
 *               (RollTable.<id> x44, Macro.<id> x1) and
 *               document-converter.js:539 requires a `^Compendium.` prefix, so
 *               these 806 names must be translated EXPLICITLY and kept
 *               consistent with the translated sub-table names ·
 *               description 1338/163191 (the real result text in v13+/v14;
 *               there is NO `text` field on any result in these packs).
 *               ⚠ 38 results share a `range` with a sibling (37 core /
 *               1 starter), so the key allocator falls back to the raw `_id`
 *               for the second one and the file mixes '1-6' keys with opaque
 *               16-char ids.
 *   Scene       name 15/291 · drawings 0 · notes 37/349 (starterset only) ·
 *               regions 4/69 (starterset only). Our layer adds navName + tokens.
 *   Region      name 4/69 · behaviors 4.
 *   RegionBehavior name 4/70 — 'Teleport Token to HH Level 1' x1 and
 *               'Teleport Token' x3, all starterset. ⚠ CORRECTED 2026-08-29:
 *               an earlier revision of this comment said "only `Region.name` is
 *               real" and FIELD_CENSUS had no RegionBehavior row, yet a live
 *               extract emits 4 `name` keys / 70 chars nested at
 *               scenes -> regions -> behaviors -> name. That is precisely the
 *               census-blind-spot class this table exists to close, found by
 *               diffing the emitted baselines against FIELD_CENSUS rather than
 *               by reading the mapping — do the diff, do not trust the prose.
 *               `_variants` for displayScrollingText / teleportToken: all 4
 *               behaviors in the corpus are `teleportToken` with system keys
 *               {choice, destination} — there is NO `dialog` sub-object in this
 *               Foundry version's data, so the variant matches and contributes
 *               nothing. Harmless; kept verbatim.
 *               (`extract_en.mjs` understands `_variants` natively —
 *               `normalizeDefinition` :225-233, `whenMatches` :274-288,
 *               `activeFields` :295-309 — so unlike the EC extractor it needs
 *               NO subtype shims for this, and this file ships none.)
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
  Actor: {
    name: 'name',
    description: 'system.details.biography.value',
    items: { path: 'items', converter: 'document', documentType: 'Item', cardinality: 'many' },
    effects: { path: 'effects', converter: 'document', documentType: 'ActiveEffect', cardinality: 'many' },
    tokenName: { path: 'prototypeToken.name', converter: 'name' },
  },
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
  Item: {
    name: 'name',
    description: 'system.description.value',
    effects: { path: 'effects', converter: 'document', documentType: 'ActiveEffect', cardinality: 'many' },
  },
  JournalEntry: {
    name: 'name',
    description: 'content',
    categories: { path: 'categories', converter: 'nameCollection' },
    pages: { path: 'pages', converter: 'document', documentType: 'JournalEntryPage', cardinality: 'many' },
  },
  JournalEntryPage: {
    name: 'name',
    caption: 'image.caption',
    src: 'src',
    text: 'text.content',
    width: 'video.width',
    height: 'video.height',
  },
  Macro: {
    name: 'name',
    command: 'command',
  },
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
    regions: { path: 'regions', converter: 'document', documentType: 'Region', cardinality: 'many' },
  },
  Region: {
    name: 'name',
    behaviors: { path: 'behaviors', converter: 'document', documentType: 'RegionBehavior', cardinality: 'many' },
  },
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
deepFreeze(BABELE_DEFAULTS);

/* ================================================================== *
 * 4. Deliberate deviations — EXTRACT DIRECTION ONLY
 * ================================================================== */

/**
 * Fields removed from the effective mapping the EXTRACTOR uses. `null` means
 * "drop this key".
 *
 * These are NOT runtime deviations and cannot be: `registerMapping` merges, it
 * cannot delete, so Babele keeps its built-in for all three at translate time.
 * The consequence is exactly what we want — the translation files simply never
 * contain the key, Babele's FieldMapping fails open on a missing key, and the
 * document field is left untouched. Both directions stay consistent, which is
 * the EC rule: never add (or remove) a field in one direction only.
 */
export const EXTRACTOR_DEVIATIONS = {
  Actor: {
    /**
     * (V) DROP `tokenName` from the English baseline.
     *
     * Babele's own export already omits it: the built-in converter is
     * `Converters.mappedField('name')` (converter/converters.js:92-97), a bare
     * arrow function with no `.extract` property, so
     * `FunctionalConverter.extract` returns undefined
     * (converter/functional-converter.js:63-66) and the key never appears.
     *
     * `extract_en.mjs` does NOT reproduce that: its `case 'name'` at :363-368
     * emits the raw `prototypeToken.name`. Left alone it would write 74
     * `tokenName` keys / 1,847 chars into the baseline that the runtime can
     * never consume — `mappedField` discards the `tokenName` translation and
     * returns the translated `name`. That is the same pure cost the verifier
     * charged against `alienTokenName`, just relocated from the converter to
     * the extractor; dropping the converter without dropping this would have
     * left the cost in place.
     *
     * Measured: `prototypeToken.name === name` for 74/74 actors, so nothing is
     * lost. If upstream ever ships a token name that differs from its actor's,
     * revisit BOTH this line and the `alienTokenName` decision together.
     */
    tokenName: null,   // would be n=74 chars=1847, 100% duplicates of `name`
  },

  Macro: {
    /**
     * DECISION: `Macro.command` is EXCLUDED. Reason in full, because the task
     * required an explicit ruling either way.
     *
     * The corpus: 5 macros (4 system, 1 corerules), 9,347 chars of JavaScript.
     * Every user-visible literal in all five, enumerated by scanning the raw
     * command bodies rather than trusting the survey:
     *   'Player - Roll Alien Dice.'  — `new Dialog({title: ...})`, system macro 1
     *   'Roll Alien Dice.'           — `new Dialog({title: ...})`, system macro 2
     *   'GM'                         — `let label = 'GM'`, system macro 2
     *   'for '                       — chat-label concatenation, system macro 1
     *   'OK' x3                      — `label: 'OK'` in the NOTABLES fallback dialogs
     * That is about 53 chars of visible text out of 9,347. Everything else is
     * already `game.i18n.localize('ALIENRPG.*')`, ids, or CONFIG lookups.
     *
     * 1. COST/BENEFIT: mapping it drags 9,347 chars of JS into the English
     *    baseline as translatable strings to reach 0.57% of it.
     * 2. BABELE REPLACES `command` WHOLESALE. A translated command freezes
     *    upstream's JS into the cn pack. When upstream edits a macro, the
     *    Chinese user gets OUR FROZEN OLD CODE. Stale prose is merely stale;
     *    stale code is a BROKEN MACRO — and both are gated by the same drift
     *    check, which only fires when we chase a version.
     * 3. FIVE LITERALS INSIDE MUST STAY BYTE-IDENTICAL and nothing validates
     *    them: 'Alien Mother Tables' and 'Alien Creature Tables'
     *    (`t.folder.name === ...` folder comparisons in the two V10 macros),
     *    'EV - 48a. LS - DANGER EVENT DETAIL'
     *    (`game.tables.getName("EV - 48a. LS - DANGER EVENT DETAIL")` in the
     *    corerules macro), and 'Black' / 'Stress' (dice-type args to
     *    `yze.yzeRoll`). A translator editing JS has no syntax check and no
     *    error surface until the macro is run.
     * 4. THERE IS A BETTER CHANNEL. All six visible strings are
     *    `new Dialog({title})` / `buttons.*.label` values produced at run time,
     *    so the hub module's runtime patcher (the alienrpg-cn counterpart of
     *    EC's `scripts/ember-hardcoded-cn.mjs`) can rewrite them at display
     *    time. That translates them WITHOUT freezing code and WITHOUT touching
     *    the English baseline.
     *
     * => EXCLUDED. `Macro.name` (5 / 177 chars) is still translated normally.
     * If this is ever revisited, revisit it together with the runtime-patcher
     * decision, not on its own.
     */
    command: null,   // would be n=5 chars=9347 of executable JavaScript
  },

  JournalEntryPage: {
    /**
     * DROP `src` from the English baseline.
     *
     * Babele's default maps it so a translation can point at a LOCALISED image
     * or video. This project does not localise assets. Measured: 92 image pages
     * carry a non-empty `src`, 5,879 chars of file paths, which the extractor's
     * plain-path read would put in front of a translator as 92 translatable
     * strings.
     *
     * ⚠ THIS IS A REAL DIFFERENCE FROM THE EC PRECEDENT, where `src` was
     * non-empty on 0 pages and keeping it was free. Here it is not free.
     *
     * Runtime is unaffected (Babele keeps its default; a missing key fails
     * open), so the field is simply never rewritten. Re-enable BOTH this line
     * and the asset pipeline together if the project ever ships localised maps.
     *
     * `width` / `height` (video.width / video.height) are NOT dropped: they are
     * `null` on all 151 pages and `extract_en.mjs` only emits non-empty
     * strings, so they are inert and the mirror stays closer to upstream.
     */
    src: null,       // would be n=92 chars=5879 of asset paths
  },
};
deepFreeze(EXTRACTOR_DEVIATIONS);

/* ================================================================== *
 * 5. effectiveMappings — what the extractor actually walks
 * ================================================================== */

/**
 * Babele defaults, minus the recorded deviations, overlaid with this project's
 * layer.
 *
 * The overlay is PER DOCUMENT TYPE, FIELD BY FIELD — not a whole-type replace —
 * and `_variants` are CONCATENATED with the base's first. That is exactly what
 * Babele itself does with a registered layer:
 *   mapping/document-mappings.js:267-281 `#rebuild()` built-ins, then each registered layer
 *   mapping/document-mappings.js:283-287 `#mergeLayer()`
 *     `target[key] = this.#mergedDefinition(target[key] ?? {}, value);`
 *   mapping/document-mappings.js:347-361 `#mergedDefinition()`
 *     `const merged = foundry.utils.mergeObject(baseMapping, overrideMapping, {inplace: false});`
 *     `const variants = [...(baseVariants ?? []), ...(overrideVariants ?? [])];`
 * Reproducing the runtime's own resolution is mandatory, not stylistic: a naive
 * `{...defaults, ...layer}` would replace `RegionBehavior._variants` with
 * nothing on any layer that touched that type, and would replace `Item`'s whole
 * definition with `{_variants: [...]}` — dropping `name`, `description` and
 * `effects` from the baseline for all 523 items.
 *
 * `target` is one of ALIEN_TARGETS. All three resolve to the same layer by
 * design (see the note on STARTERSET_LAYER); the parameter is honoured so the
 * signature matches what `extract_en.mjs:612` calls and so an unknown target is
 * rejected loudly instead of silently returning a plausible-looking mapping —
 * the exact trap the EC module set for this one.
 *
 * @param {'alienrpg'|'starterset'|'corerules'} target
 * @returns {object} keyed by documentType
 */
export function effectiveMappings(target) {
  if (!ALIEN_TARGETS.includes(target)) {
    throw new Error(
      `effectiveMappings: unknown target '${target}'.`
      + ` Expected one of ${ALIEN_TARGETS.join('|')}.`
      + ' Returning a default layer for an unknown target is how a wrong English'
      + ' baseline gets produced without anyone noticing.',
    );
  }

  const merge = (base = {}, override = {}) => {
    const { _variants: baseVariants, ...baseRest } = base;
    const { _variants: overrideVariants, ...overrideRest } = override;
    const out = { ...baseRest, ...overrideRest };
    const variants = [
      ...(Array.isArray(baseVariants) ? baseVariants : []),
      ...(Array.isArray(overrideVariants) ? overrideVariants : []),
    ];
    if (variants.length) out._variants = variants;
    return out;
  };

  const effective = {};
  for (const [documentType, definition] of Object.entries(BABELE_DEFAULTS)) {
    effective[documentType] = merge({}, definition);
  }

  // Deviations: `null` deletes a key from the extract direction.
  for (const [documentType, fields] of Object.entries(EXTRACTOR_DEVIATIONS)) {
    if (!effective[documentType]) continue;
    for (const [field, spec] of Object.entries(fields)) {
      if (spec === null) delete effective[documentType][field];
      else effective[documentType][field] = spec;
    }
  }

  // This project's layer, merged the way Babele merges a registered layer.
  const layer = target === 'starterset'
    ? STARTERSET_LAYER
    : (target === 'corerules' ? CORERULES_LAYER : ALIENRPG_MAPPINGS);
  for (const [documentType, definition] of Object.entries(layer)) {
    effective[documentType] = merge(effective[documentType] ?? {}, definition);
  }

  return effective;
}

/* ================================================================== *
 * 6. FIELD_CENSUS — machine-readable twin of the `// n=… chars=…` comments
 * ================================================================== */

/**
 * Every path this file maps, with what the three raw dumps actually contain,
 * measured 2026-08-29.
 *
 *   doc     document type
 *   types   subtype filter the variant applies (null = every document of `doc`)
 *   path    source path
 *   n       documents holding a NON-EMPTY string there
 *   chars   total length of those strings
 *   present documents on which the path resolves at all (empty strings and
 *           objects included) — omitted when equal to `n`
 *
 * Purpose: `qa/` re-measures the dumps and diffs against this. A path whose `n`
 * drops to 0 after an upstream bump, or a mapped path that was never present at
 * all, is then a FAILING CHECK rather than an invisible 766K-char hole.
 *
 * Sum of `chars` over this table: **2,705,563** across the three packs.
 * (An earlier revision claimed 2,700,536; that figure was arithmetically wrong
 * AND was computed from a table missing its RegionBehavior row. Re-derived by
 * summing the array itself — `FIELD_CENSUS.reduce((a,r)=>a+r.chars,0)` — not by
 * adding up the bullets below. If you edit a row, re-run the reduce.)
 * By documentType: JournalEntryPage 2,232,536 · TableResult 173,116 ·
 *   Item 156,928 · Actor 132,802 · RollTable 4,528 · Adventure 4,392 ·
 *   Scene 640 · JournalEntry 305 · Macro 177 · RegionBehavior 70 · Region 69.
 *
 * ---------------------------------------------------------------------------
 * ⚠ THIS TABLE COUNTS SOURCE DOCUMENTS, THE BASELINE COUNTS EMITTED KEYS
 * ---------------------------------------------------------------------------
 * A QA script that diffs FIELD_CENSUS straight against the files
 * `extract_en.mjs` writes will fire FOUR false positives on day one. The two
 * numbers differ by exactly 299 chars — a live 3-pack extract on 2026-08-29
 * emitted 2,705,264 — and every one of the 299 is accounted for:
 *
 *   -5   and -106  Item 'Lucky' (corerules). TWO adventure-level `talent`
 *                  documents share that name (_id JSAsfhzNgBJCFXU4 and
 *                  XZpzDVSfgaDxIoTg). The extractor keys entries BY NAME, so
 *                  the second collapses onto the first. Harmless HERE — their
 *                  `system` sub-objects are byte-identical, so one translation
 *                  correctly serves both — but it is only harmless by luck.
 *                  A QA gate should assert duplicate-named documents in one
 *                  collection are byte-identical, not merely count them.
 *   -39            Folder name 'Alien Evolved' x3 (starterset 1, corerules 2).
 *                  `nameCollection` emits a name->name map, so within one pack
 *                  duplicates collapse: 49 folder docs / 755 chars become
 *                  46 keys / 716 chars (system 7/99, starter 16/264, core 23/353).
 *   -21            Scene note text repeated inside one scene ('Elevator',
 *                  'Storage', 'Stairs' …). `textCollection` is a map too:
 *                  37 notes / 349 chars become 34 keys / 328 chars.
 *   -12 and -116   The 32 'None' sentinels `EXTRACT_CONVERTERS.alienRollTableRef`
 *                  deliberately withholds: rollTable 30/805 -> 27/793,
 *                  critTable 30/147 -> 1/31. This one is the design working.
 *
 * So the correct assertion is
 *   sum(FIELD_CENSUS.chars) - 299 === sum(chars in the three baselines)
 * with each of the four terms named, not a bare equality. Re-derive the 299
 * whenever the corpus changes; a silently drifting fudge factor is worse than
 * no check at all.
 */
export const FIELD_CENSUS = [
  { doc: 'Adventure', types: null, path: 'name', n: 3, chars: 65 },
  { doc: 'Adventure', types: null, path: 'description', n: 3, chars: 3572 },
  { doc: 'Adventure', types: null, path: 'caption', n: 0, chars: 0, present: 3 },
  { doc: 'Adventure', types: null, path: 'folders[].name', n: 49, chars: 755 },

  { doc: 'Item', types: null, path: 'name', n: 523, chars: 7986 },
  { doc: 'Item', types: 'ITEM_TYPES', path: 'system.notes', n: 267, chars: 4005, present: 523 },
  { doc: 'Item', types: 'BODY_IN_ATTRIBUTES', path: 'system.attributes.comment.value', n: 307, chars: 99697, present: 324 },
  { doc: 'Item', types: 'BODY_IN_GENERAL', path: 'system.general.comment.value', n: 100, chars: 19595, present: 119 },
  { doc: 'Item', types: 'skill-stunts', path: 'system.description', n: 12, chars: 1957 },
  { doc: 'Item', types: 'planet-system', path: 'system.misc.description.value', n: 0, chars: 0, present: 68 },
  { doc: 'Item', types: 'planet-system', path: 'system.header.commonName.value', n: 58, chars: 2045, present: 68 },
  { doc: 'Item', types: 'planet-system', path: 'system.header.system.value', n: 68, chars: 902 },
  { doc: 'Item', types: 'planet-system', path: 'system.header.sector.value', n: 68, chars: 1170 },
  { doc: 'Item', types: 'planet-system', path: 'system.header.location.value', n: 64, chars: 1675, present: 68 },
  { doc: 'Item', types: 'planet-system', path: 'system.details.affiliation.value', n: 67, chars: 1796, present: 68 },
  { doc: 'Item', types: 'planet-system', path: 'system.details.classification.value', n: 67, chars: 1752, present: 68 },
  { doc: 'Item', types: 'planet-system', path: 'system.details.climate.value', n: 67, chars: 3095, present: 68 },
  { doc: 'Item', types: 'planet-system', path: 'system.details.meanTemperature.value', n: 67, chars: 516, present: 68 },
  { doc: 'Item', types: 'planet-system', path: 'system.details.terrain.value', n: 66, chars: 2715, present: 68 },
  { doc: 'Item', types: 'planet-system', path: 'system.details.colonies.value', n: 67, chars: 3040, present: 68 },
  { doc: 'Item', types: 'planet-system', path: 'system.details.keyResources.value', n: 66, chars: 2310, present: 68 },
  { doc: 'Item', types: 'weapon', path: 'system.attributes.class.value', n: 78, chars: 657 },
  { doc: 'Item', types: 'spacecraftmods', path: 'system.attributes.capacity.value', n: 124, chars: 2015 },
  { doc: 'Item', types: 'item', path: 'system.attributes.notes.value', n: 0, chars: 0, present: 93 },
  { doc: 'Item', types: 'weapon', path: 'system.attributes.notes.notes', n: 0, chars: 0, present: 78 },
  { doc: 'Item', types: 'critical-injury', path: 'system.attributes.effects', n: 0, chars: 0, present: 0 },
  { doc: 'Item', types: 'critical-injury', path: 'system.attributes.healingtime.value', n: 0, chars: 0, present: 0 },
  { doc: 'Item', types: 'spacecraft-crit', path: 'system.header.effects', n: 0, chars: 0, present: 0 },
  { doc: 'Item', types: 'spacecraft-crit', path: 'system.header.repairroll', n: 0, chars: 0, present: 0 },
  { doc: 'Item', types: 'colony-initiative', path: 'system.header.comment', n: 0, chars: 0, present: 0 },

  { doc: 'Actor', types: null, path: 'name', n: 74, chars: 1847 },
  { doc: 'Actor', types: 'NOTES_RENDERED_TYPES', path: 'system.notes', n: 45, chars: 40265, present: 68 },
  { doc: 'Actor', types: 'territory', path: 'system.notes', n: 0, chars: 0, present: 0 },
  { doc: 'Actor', types: 'vehicles', path: 'system.notes.notes', n: 0, chars: 0, present: 6 },
  { doc: 'Actor', types: 'creature', path: 'system.general.special.value', n: 30, chars: 81617 },
  { doc: 'Actor', types: 'creature', path: 'system.rTables', n: 30, chars: 805 },
  { doc: 'Actor', types: 'creature', path: 'system.cTables', n: 30, chars: 147 },
  { doc: 'Actor', types: 'character,synthetic', path: 'system.general.appearance.value', n: 8, chars: 1062, present: 31 },
  { doc: 'Actor', types: 'character,synthetic', path: 'system.adhocitems', n: 8, chars: 358, present: 31 },
  { doc: 'Actor', types: 'character,synthetic', path: 'system.general.sigItem.value', n: 6, chars: 58, present: 31 },
  { doc: 'Actor', types: 'character,synthetic', path: 'system.general.agenda.value', n: 2, chars: 8, present: 31 },
  { doc: 'Actor', types: 'character,synthetic', path: 'system.general.relOne.value', n: 7, chars: 41, present: 31 },
  { doc: 'Actor', types: 'character,synthetic', path: 'system.general.relTwo.value', n: 7, chars: 40, present: 31 },
  { doc: 'Actor', types: 'vehicles', path: 'system.attributes.comment.value', n: 6, chars: 6038 },
  { doc: 'Actor', types: 'vehicles,spacecraft', path: 'system.general.misc.value', n: 0, chars: 0, present: 13 },
  { doc: 'Actor', types: 'spacecraft', path: 'system.attributes.manufacturer', n: 7, chars: 68 },
  { doc: 'Actor', types: 'spacecraft', path: 'system.attributes.model', n: 7, chars: 96 },
  { doc: 'Actor', types: 'spacecraft', path: 'system.attributes.ai', n: 7, chars: 73 },
  { doc: 'Actor', types: 'spacecraft', path: 'system.attributes.modules', n: 7, chars: 235 },
  { doc: 'Actor', types: 'spacecraft', path: 'system.attributes.armaments', n: 6, chars: 44, present: 7 },
  { doc: 'Actor', types: 'colony', path: 'system.header/attributes/stats (15 fields)', n: 0, chars: 0, present: 0 },
  { doc: 'Actor', types: 'planet', path: 'system.header/attributes/general (22 fields)', n: 0, chars: 0, present: 0 },
  { doc: 'Actor', types: 'territory', path: 'system.sectors.value + system.comment.value', n: 0, chars: 0, present: 0 },

  { doc: 'JournalEntry', types: null, path: 'name', n: 13, chars: 305 },
  { doc: 'JournalEntry', types: null, path: 'content', n: 0, chars: 0, present: 0 },
  { doc: 'JournalEntryPage', types: null, path: 'name', n: 151, chars: 2256 },
  { doc: 'JournalEntryPage', types: null, path: 'text.content', n: 59, chars: 2230280 },
  { doc: 'JournalEntryPage', types: null, path: 'image.caption', n: 0, chars: 0, present: 9 },

  { doc: 'RollTable', types: null, path: 'name', n: 137, chars: 3640 },
  { doc: 'RollTable', types: null, path: 'description', n: 16, chars: 888, present: 137 },
  { doc: 'TableResult', types: null, path: 'name', n: 806, chars: 9925, present: 1530 },
  { doc: 'TableResult', types: null, path: 'description', n: 1338, chars: 163191, present: 1530 },

  { doc: 'Macro', types: null, path: 'name', n: 5, chars: 177 },

  { doc: 'Scene', types: null, path: 'name', n: 15, chars: 291 },
  { doc: 'Scene', types: null, path: 'notes[].text', n: 37, chars: 349 },
  { doc: 'Scene', types: null, path: 'drawings[].text', n: 0, chars: 0 },
  { doc: 'Scene', types: null, path: 'navName', n: 0, chars: 0, present: 15 },
  { doc: 'Scene', types: null, path: 'tokens[].name', n: 0, chars: 0 },
  { doc: 'Region', types: null, path: 'name', n: 4, chars: 69 },
  // (V-2) Added 2026-08-29. Babele's default `RegionBehavior: {name:'name'}` is
  // live and the extractor emits it; the table had no row for it. 4 docs, all
  // starterset, all `teleportToken`.
  { doc: 'RegionBehavior', types: null, path: 'name', n: 4, chars: 70 },
];
deepFreeze(FIELD_CENSUS);

/**
 * Fields that LOOK translatable and are NOT. Recorded here because "it is not
 * in the mapping" is indistinguishable from "we forgot", and every one of these
 * has already been proposed once. About 32.5k chars of noise avoided.
 *
 * ITEM SIDE
 *   system.modifiers.skills.<12>.label / .ability,
 *   system.modifiers.attributes.<6>.label
 *     101 docs (92 item + 9 armor) / 19,493 chars over 30 paths.
 *     (V) DO-NOT-TRANSLATE ON **ONE** LEG, NOT TWO. The survey's first reason —
 *     "derived via prepareDerivedData on the `feature` type" — is FALSE.
 *     `base-item.mjs:37-46` does overwrite them from `game.i18n.localize`, but
 *     that method is UNREACHABLE: all 13 registered types override it with an
 *     empty or fully commented-out body (item-item.mjs:78, item-weapon.mjs:95,
 *     item-armor.mjs:57, item-talent.mjs:24, item-specialty.mjs:18,
 *     item-agenda.mjs:18, item-stunts.mjs:13, item-spacecraftmods.mjs:43,
 *     item-spacecraftweapons.mjs:61, item-planet-system.mjs:63,
 *     item-colony-initiative.mjs:63, item-crit-inj.mjs:34,
 *     item-spacecraft-crit.mjs:22), and `feature` is NOT a registered Item type
 *     at all — `module/data/item-feature.mjs` is 5 lines and absent from
 *     `CONFIG.Item.dataModels` (alienrpg.mjs:138-153).
 *     The surviving reason is the template one, and it holds:
 *     `templates/item/item-modifiers.hbs` contains ZERO reads of `.label` or
 *     `.ability` — it emits `{{localize 'ALIENRPG.*'}}` literals instead.
 *     => the verdict stands; the REASONING does not transfer, so if the
 *        template ever changes, re-derive rather than reusing the old note.
 *   system.header.type.value — 191 docs / 191 chars.
 *     (V) NOT a '0'/'1' enum. Measured distribution across the 3 packs:
 *     '1' x76, '2' x57, '3' x14, '4' x11, '5' x11, '6' x7, '7' x7, '9' x8 —
 *     on item / weapon / spacecraftweapons. The '0'/'1' switch the survey cited
 *     is guarded by `if (Attrib.type === "spacecraft-crit")` and there are ZERO
 *     spacecraft-crit items, so it explains 0/191 of the real data.
 *     The real readers: templates/item/item-header.hbs:13-14 / :33-34 /
 *     :124-131 / :163-164 (a `<select>` bound to a CONFIG list with
 *     `localize=true` — the LABEL is localised, the stored value is the key),
 *     templates/item/item-modifiers.hbs:2 / :50 / :61 / :81 / :92
 *     (`{{#if (ne system.header.type.value '9')}}`), and
 *     module/sheets/item-sheet.mjs:227 / :229 (the spacecraft-crit icon
 *     switch). Translating it breaks the sheet layout.
 *   system.header.type (colony-initiative) — see the leg-7 comment above.
 *   system.header.active — 180 docs / 899 chars, a StringField holding
 *     'false' / 'true'. Translating it breaks the equipped toggle.
 *   system.attributes.cost.value — 324 docs / 3,026 chars of currency.
 *   system.general.career.value (talent) — 85 docs, an index into
 *     CONFIG.ALIENRPG.career_list, rendered via item-header.hbs:58-59.
 *   system.attributes.size.value / hardpoint.value / details.population.value —
 *     212 docs / 895 chars of Roman numerals and raw numbers.
 *   system.header.damage.value (spacecraft-crit) — declared, zero writers,
 *     zero readers. See the leg-6 comment.
 *
 * ACTOR SIDE
 *   system.skills.<12>.label AND .description — 22 docs / 5,236 chars.
 *     Both overwritten every `prepareDerivedData` (actor-character.mjs:466-468,
 *     actor-synthetic.mjs:456-458), so translating them in the pack is a no-op
 *     — AND `.description` is a LOOKUP KEY (character-skills.hbs:15 ->
 *     character-sheet.mjs:1119 `game.items.getName(dataset.pmbut)`).
 *     Translating it here would also HIDE the real skill-stunts naming rule.
 *   system.attributes.<abl>.label, system.general.<x>.label — (V-3) RE-MEASURED
 *     2026-08-29: **46 docs with a non-empty value (54 where the path resolves)
 *     / 2,258 chars**, not the 74 docs / 2,358 chars claimed here before.
 *     Derived from CONFIG.ALIENRPG.
 *   system.header.*.label, system.general.special.label — (V-3) RE-MEASURED:
 *     **61 docs / 1,409 chars**, not 74 / 1,471. Dead: zero readers in
 *     templates/ or module/; values are a mix of i18n keys ('ALIENRPG.Health')
 *     and literals ('Health', 'Text', 'Number').
 *   system.modifiers.<skill>.label — (V-3) MISSING FROM THIS ROSTER ENTIRELY
 *     until 2026-08-29. 6 docs / 126 chars: `modifiers.rangedCbt.label`
 *     ('Ranged Combat', 78) and `modifiers.piloting.label` ('Piloting', 48),
 *     all on `vehicles` actors. Same family as every other `.label` — derived
 *     from CONFIG, no template reads it — so the verdict is DO-NOT-TRANSLATE,
 *     but a roster that silently omits a path is exactly the blind spot this
 *     block exists to close. Its sibling `.ability` WAS listed; the `.label`
 *     half was not.
 *   system.general.cash.value / career.value /
 *     mobility|observation|acidSplash.value / attributes.leasecost.value /
 *     attributes.cost.value / modifiers.*.ability — (V-3) RE-MEASURED:
 *     **61 docs / 378 chars**, not 74 / 421, of numbers, currency, list
 *     indexes and attribute codes.
 *   ⚠ The three corrected figures above were produced by walking the three raw
 *     dumps path by path, not by re-reading this comment. The doc counts that
 *     were here before appear to have been "all 74 actors" written as if it
 *     were a measurement; the char totals were 100 / 62 / 43 high. Nothing
 *     operational changes — every path in this roster stays untranslated — but
 *     a QA gate that diffs against these numbers now has correct ones.
 *
 * ALSO LEFT OUT DELIBERATELY, recorded so it is a decision:
 *   Scene.grid.units — 15 docs / 19 chars, 'm' x11 and 'ft' x4. Foundry
 *     renders it verbatim in the scene config and on measurement labels, so it
 *     is the one visible leftover in the whole corpus. Not mapped: Babele has
 *     no default for it, translating it would desync the unit from the numeric
 *     distances the journals quote in feet, and 19 chars is not worth a
 *     bespoke field. Revisit only alongside a units-conversion pass.
 *
 * NOT A FIELD, BUT PART OF THE SAME SYNC SURFACE — the RollTable-name problem
 * has a SECOND home the survey missed, inside the 2.23M-char journal blob a
 * translator actually edits:
 *   36 draw-enrichers, all in corerules: `@TEXTDRAW[RollTable.<id>]{label}` x33
 *   and `@DRAW[RollTable.<id>]{label}` x3. The `{label}` reproduces a RollTable
 *   NAME; the id keeps resolving, so a drifted label fails SILENTLY.
 * And the plain link inventory: 41 `@UUID[...]{label}` — `@UUID[Item.<id>]` x39
 *   (38 core + 1 starter) and `@UUID[JournalEntry.<id>]` x2.
 *   (V) THE SYNTAX IS `@UUID[Type.id]{label}`, the Foundry v10+ form. There are
 *   ZERO occurrences of the literal `@Item[` or `@JournalEntry[` anywhere in
 *   any pack (verified by direct count), so a QA regex written against the
 *   survey's stated shape matches nothing and would pass a file in which every
 *   link had been destroyed.
 */
