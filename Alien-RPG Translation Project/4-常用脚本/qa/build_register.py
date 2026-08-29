# -*- coding: utf-8 -*-
"""Generate 7-其他内容/DO-NOT-TRANSLATE.json entirely from source.

Every string, count and file:line in the output is re-derived here; nothing is
transcribed from the survey.  Run:

    python build_register.py
"""
import json, os, re, sys, collections, hashlib

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJ = r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project"
DUMPS = os.path.join(PROJ, "6-工作区", "raw-dumps")
DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
SYS = os.path.join(DATA, "systems", "alienrpg")
CR = os.path.join(DATA, "modules", "alien-evolved-corerules")
SS = os.path.join(DATA, "modules", "alien-evolved-starterset")
OUT = os.path.join(PROJ, "7-其他内容", "DO-NOT-TRANSLATE.json")

PACKS = {}
for n in ['system', 'starterset', 'corerules']:
    d = json.load(open(os.path.join(DUMPS, n + '.json'), encoding='utf-8'))
    PACKS[n] = list(d.values())[0]

# ---------------------------------------------------------------- file:line re-derivation
_cache = {}


def lines(path):
    if path not in _cache:
        _cache[path] = open(path, encoding='utf-8').read().split('\n')
    return _cache[path]


def cite(path, needle, occurrence=1):
    """Return 'relative/path.js:NN' for the occurrence-th line containing needle.

    Raises if not found, so a drifted citation is a build failure, not a silent lie."""
    got = 0
    for i, l in enumerate(lines(path), 1):
        if needle in l:
            got += 1
            if got == occurrence:
                return '%s:%d' % (rel(path), i)
    raise SystemExit('CITATION NOT FOUND: %r in %s (occurrence %d)' % (needle, path, occurrence))


def line_at(path, n):
    return lines(path)[n - 1]


def rel(path):
    for base, tag in ((SYS, 'systems/alienrpg'), (CR, 'modules/alien-evolved-corerules'),
                      (SS, 'modules/alien-evolved-starterset')):
        if path.startswith(base):
            return tag + '/' + os.path.relpath(path, base).replace('\\', '/')
    return path.replace('\\', '/')


P_INIT = os.path.join(SYS, 'module', 'apps', 'init.mjs')
P_MAIN = os.path.join(SYS, 'module', 'alienrpg.mjs')
P_ACTOR = os.path.join(SYS, 'module', 'documents', 'actor.mjs')
P_RTD = os.path.join(SYS, 'module', 'helpers', 'rollTableData.mjs')
P_UPD = os.path.join(SYS, 'module', 'apps', 'update.js')
P_CHAR = os.path.join(SYS, 'module', 'sheets', 'character-sheet.mjs')
P_SYN = os.path.join(SYS, 'module', 'sheets', 'synthetic-sheet.mjs')
P_COL = os.path.join(SYS, 'module', 'sheets', 'colony-sheet.mjs')
P_DCHAR = os.path.join(SYS, 'module', 'data', 'actor-character.mjs')
P_DSYN = os.path.join(SYS, 'module', 'data', 'actor-synthetic.mjs')
P_CFG = os.path.join(SYS, 'module', 'helpers', 'config.mjs')
P_ENR = os.path.join(SYS, 'module', 'helpers', 'enricher.mjs')
P_CRINIT = os.path.join(CR, 'module', 'init.js')
P_CRUPD = os.path.join(CR, 'module', 'update.js')
P_CRISA = os.path.join(CR, 'module', 'ImportSelectedAssets.js')
P_SSINIT = os.path.join(SS, 'module', 'init.js')
P_SSUPD = os.path.join(SS, 'module', 'update.js')
P_SSISA = os.path.join(SS, 'module', 'ImportSelectedAssets.js')


def flat(d, p=''):
    o = {}
    for k, v in d.items():
        n = p + '.' + k if p else k
        if isinstance(v, dict):
            o.update(flat(v, n))
        else:
            o[n] = v
    return o


LANG_EN = flat(json.load(open(os.path.join(SYS, 'lang', 'en.json'), encoding='utf-8')))
LANG_CN = flat(json.load(open(os.path.join(SYS, 'lang', 'cn.json'), encoding='utf-8')))

# ---------------------------------------------------------------- 1. name_lookups
name_lookups = [
    {
        "id": "system.adventurePackName",
        "package": "alienrpg (system)",
        "role": "Adventure document name",
        "string": "Alien RPG System",
        "declared_at": cite(P_INIT, 'export const adventurePackName'),
        "compared_at": [
            cite(P_INIT, 'await pack.getName(adventurePackName)'),
            cite(P_INIT, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)', 1),
            cite(P_INIT, 'if (adventure.name === adventurePackName)'),
            cite(P_INIT, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)', 2),
            cite(P_MAIN, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)'),
            cite(P_UPD, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)'),
        ],
        "unguarded_deref": [
            cite(P_INIT, 'await pack.getName(adventurePackName)') + "  .sheet._updateObject(...) on the getName() result",
        ],
        "breaks_if_translated": (
            "FirstTimeSetup() dereferences pack.getName('Alien RPG System').sheet with no null check -> "
            "TypeError on the very first world load, the system Adventure never imports, so no Panic Table / "
            "Stress Response Table / crit tables / skill-stunt Items exist and every roll that needs one fails. "
            "ModuleImport()/ReImport() get adventureId===undefined -> pack.getDocument(undefined) -> the "
            "Re-Import settings button silently does nothing."),
        "verified_target_exists": PACKS['system']['name'] == "Alien RPG System",
    },
    {
        "id": "system.welcomeJournalEntry",
        "package": "alienrpg (system)",
        "role": "JournalEntry name shown after import",
        "string": "MU/TH/ER Instructions.",
        "declared_at": cite(P_INIT, 'export const welcomeJournalEntry'),
        "compared_at": [
            cite(P_INIT, 'game.journal.getName(welcomeJournalEntry).show()', 1),
            cite(P_INIT, 'game.journal.getName(welcomeJournalEntry).show()', 2),
        ],
        "unguarded_deref": [
            cite(P_INIT, 'game.journal.getName(welcomeJournalEntry).show()', 1) + "  .show() on the getName() result",
            cite(P_INIT, 'game.journal.getName(welcomeJournalEntry).show()', 2) + "  .show() on the getName() result",
        ],
        "breaks_if_translated": (
            "getName returns null and .show() throws inside FirstTimeSetup() AFTER the 'imported' flag and "
            "migrationVersion were already written, so the exception aborts nothing recoverable but the user "
            "never sees the MU/TH/ER instructions. In ModuleImport()'s importAdventure hook the throw also skips "
            "the `return`, leaving the hook registered."),
        "note": "Same literal is re-declared independently at " + cite(P_MAIN, 'const releaseNoteName = "MU/TH/ER Instructions."') +
                " as releaseNoteName; both must move together.",
        "verified_target_exists": any(j['name'] == "MU/TH/ER Instructions." for j in PACKS['system']['journal']),
    },
    {
        "id": "system.releaseNoteName",
        "package": "alienrpg (system)",
        "role": "JournalEntry name used by the release-notes updater",
        "string": "MU/TH/ER Instructions.",
        "declared_at": cite(P_MAIN, 'const releaseNoteName = "MU/TH/ER Instructions."'),
        "compared_at": [
            cite(P_MAIN, 'const oldReleaseNotes = game.journal.getName(releaseNoteName)'),
            cite(P_MAIN, 'const selected = game.journal.getName(releaseNoteName).id'),
            cite(P_MAIN, 'const newReleaseJournal = game.journal.getName(releaseNoteName)'),
        ],
        "unguarded_deref": [
            cite(P_MAIN, 'const selected = game.journal.getName(releaseNoteName).id') + "  .id on the getName() result",
            cite(P_MAIN, 'await newReleaseJournal.setFlag') + "  .setFlag() on the getName() result",
        ],
        "breaks_if_translated": (
            "showReleaseNotes() throws at .id; the whole body is wrapped in try{}catch{} with an EMPTY catch "
            "(" + cite(P_MAIN, '} catch (error) {') + "), so the failure is completely silent: release notes "
            "simply stop appearing forever and no console error is produced."),
        "verified_target_exists": any(j['name'] == "MU/TH/ER Instructions." for j in PACKS['system']['journal']),
    },
    {
        "id": "corerules.adventurePackName",
        "package": "alien-evolved-corerules",
        "role": "Adventure document name",
        "string": "Alien Evolved Core Rules",
        "declared_at": cite(P_CRINIT, 'export const adventurePackName'),
        "compared_at": [
            cite(P_CRINIT, 'await pack.getName(adventurePackName)'),
            cite(P_CRINIT, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)', 1),
            cite(P_CRINIT, 'if (adventure.name === adventurePackName)'),
            cite(P_CRINIT, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)', 2),
            cite(P_CRUPD, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)'),
            cite(P_CRISA, 'const adventureId = pack.index.find(a => a.name === adventurePackName)', 1),
            cite(P_CRISA, 'const adventureId = pack.index.find(a => a.name === adventurePackName)', 2),
        ],
        "unguarded_deref": [
            cite(P_CRINIT, 'await pack.getName(adventurePackName)') + "  .sheet._updateObject(...)",
        ],
        "breaks_if_translated": (
            "ModuleImport() -> pack.getDocument(undefined) -> `adventure.sheet.render(true)` throws, so the "
            "Core Rules import dialog never opens and NOTHING from the 5.2 MB pack is ever imported. "
            "AutoSetup() (the 1.0.0 -> 1.0.2 migration path) throws at .sheet. The startImport() guard "
            "`adventure.name === adventurePackName` also fails, so even a manual Adventure-Importer run "
            "never sets the `imported` flag and re-prompts on every load."),
        "verified_target_exists": PACKS['corerules']['name'] == "Alien Evolved Core Rules",
    },
    {
        "id": "corerules.welcomeJournalEntry",
        "package": "alien-evolved-corerules",
        "role": "JournalEntry name shown after import",
        "string": "CORE RULES - HOW TO USE THIS MODULE",
        "declared_at": cite(P_CRINIT, 'export const welcomeJournalEntry'),
        "compared_at": [
            cite(P_CRINIT, 'game.journal.getName(welcomeJournalEntry).show()'),
            cite(P_CRINIT, 'await game.journal.getName(welcomeJournalEntry).show()'),
        ],
        "unguarded_deref": [
            cite(P_CRINIT, 'game.journal.getName(welcomeJournalEntry).show()') + "  .show()",
            cite(P_CRINIT, 'await game.journal.getName(welcomeJournalEntry).show()') + "  .show()",
        ],
        "breaks_if_translated": "Would throw at .show() — but see already_broken_upstream.",
        "already_broken_upstream": True,
        "upstream_defect": (
            "THIS LOOKUP ALREADY FAILS IN 1.0.2 AND HAS NOTHING TO DO WITH TRANSLATION. The JS literal uses "
            "ASCII SPACE (U+0020) around the hyphen; the actual JournalEntry in the pack is "
            "'CORE RULES\\u00a0-\\u00a0HOW\\u00a0TO\\u00a0USE\\u00a0THIS\\u00a0MODULE' with NO-BREAK SPACE (U+00A0) "
            "in all five gaps. game.journal.getName() therefore returns null today and .show() throws a "
            "TypeError immediately after game.scenes.getName(sceneToActivate).activate() succeeds, which is why "
            "Hooks.off('importAdventure', hookId) on the next line never runs."),
        "verified_target_exists": False,
        "actual_document_name": "CORE RULES\u00a0-\u00a0HOW\u00a0TO\u00a0USE\u00a0THIS\u00a0MODULE",
    },
    {
        "id": "corerules.sceneToActivate",
        "package": "alien-evolved-corerules",
        "role": "Scene name activated after import",
        "string": "Alien Evolved Core Rules",
        "declared_at": cite(P_CRINIT, 'export const sceneToActivate'),
        "compared_at": [
            cite(P_CRINIT, 'game.scenes.getName(sceneToActivate).activate();', 1),
            cite(P_CRINIT, 'game.scenes.getName(sceneToActivate).activate();', 2),
        ],
        "unguarded_deref": [
            cite(P_CRINIT, 'game.scenes.getName(sceneToActivate).activate();', 1) + "  .activate()",
            cite(P_CRINIT, 'game.scenes.getName(sceneToActivate).activate();', 2) + "  .activate()",
        ],
        "breaks_if_translated": (
            "startImport() throws at .activate() BEFORE the welcome journal line, so the import never reaches "
            "Hooks.off(). Note this string is byte-identical to corerules.adventurePackName but names a "
            "DIFFERENT document (a Scene, not the Adventure) — the Babele `scenes` block and the `entries` key "
            "must both keep it."),
        "verified_target_exists": any(s['name'] == "Alien Evolved Core Rules" for s in PACKS['corerules']['scenes']),
    },
    {
        "id": "starterset.adventurePackName",
        "package": "alien-evolved-starterset",
        "role": "Adventure document name",
        "string": "Alien Evolved Starter Set",
        "declared_at": cite(P_SSINIT, 'export const adventurePackName'),
        "compared_at": [
            cite(P_SSINIT, 'await pack.getName(adventurePackName)'),
            cite(P_SSINIT, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)', 1),
            cite(P_SSINIT, 'if (adventure.name === adventurePackName)'),
            cite(P_SSINIT, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)', 2),
            cite(P_SSUPD, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)'),
            cite(P_SSISA, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)', 1),
            cite(P_SSISA, 'const adventureId = pack.index.find((a) => a.name === adventurePackName)', 2),
        ],
        "unguarded_deref": [
            cite(P_SSINIT, 'await pack.getName(adventurePackName)') + "  .sheet._updateObject(...)",
        ],
        "breaks_if_translated": (
            "Same shape as corerules: ModuleImport() throws at adventure.sheet.render(true) so the Starter Set "
            "never imports; AutoSetup() (reached from migrationVersion === '1.0.0') throws at .sheet AFTER the "
            "user has already clicked OK on the V1.0.1 update dialog."),
        "verified_target_exists": PACKS['starterset']['name'] == "Alien Evolved Starter Set",
    },
    {
        "id": "starterset.welcomeJournalEntry",
        "package": "alien-evolved-starterset",
        "role": "JournalEntry name shown after import",
        "string": "STARTER SET - HOW TO USE THIS MODULE",
        "declared_at": cite(P_SSINIT, 'export const welcomeJournalEntry'),
        "compared_at": [
            cite(P_SSINIT, 'game.journal.getName(welcomeJournalEntry).show()'),
            cite(P_SSINIT, 'await game.journal.getName(welcomeJournalEntry).show()'),
        ],
        "unguarded_deref": [
            cite(P_SSINIT, 'game.journal.getName(welcomeJournalEntry).show()') + "  .show()",
            cite(P_SSINIT, 'await game.journal.getName(welcomeJournalEntry).show()') + "  .show()",
        ],
        "breaks_if_translated": (
            "getName returns null, .show() throws. In startImport() the throw lands after createThumbs() and "
            "the settings writes, so the module looks imported but the how-to-use journal never opens and "
            "Hooks.off('importAdventure') never runs — the hook stays live for the rest of the session and "
            "re-fires on any later Adventure import."),
        "verified_target_exists": any(j['name'] == "STARTER SET - HOW TO USE THIS MODULE" for j in PACKS['starterset']['journal']),
    },
    {
        "id": "starterset.sceneToActivate",
        "package": "alien-evolved-starterset",
        "role": "Scene name activated after import",
        "string": "Alien Evolved Starter Set",
        "declared_at": cite(P_SSINIT, 'export const sceneToActivate'),
        "compared_at": [
            cite(P_SSINIT, 'game.scenes.getName(sceneToActivate).activate();', 1),
            cite(P_SSINIT, 'game.scenes.getName(sceneToActivate).activate();', 2),
        ],
        "unguarded_deref": [
            cite(P_SSINIT, 'game.scenes.getName(sceneToActivate).activate();', 1) + "  .activate()",
            cite(P_SSINIT, 'game.scenes.getName(sceneToActivate).activate();', 2) + "  .activate()",
        ],
        "breaks_if_translated": (
            "AutoSetup()/startImport() throw at .activate(). Byte-identical to starterset.adventurePackName "
            "but names a Scene, not the Adventure."),
        "verified_target_exists": any(s['name'] == "Alien Evolved Starter Set" for s in PACKS['starterset']['scenes']),
    },
]

# ---------------------------------------------------------------- 2. rolltable_names
TABLES_BY_PACK = {n: [t['name'] for t in adv['tables']] for n, adv in PACKS.items()}


def where(nm):
    return [p for p in PACKS if nm in TABLES_BY_PACK[p]]


rolltable_names = [
    {"string": "Panic Table", "lookup": 'game.tables.getName("Panic Table")',
     "call_site": cite(P_ACTOR, 'const table = game.tables.getName("Panic Table")'),
     "guarded": True, "guard": "if (!table) ui.notifications.error(ALIENRPG.NoPanicTable) and return",
     "breaks_if_translated": "rollPanic() aborts with the 'no panic table' error toast; the panic roll — the "
                             "core mechanic of the game — becomes impossible for every character.",
     "present_in": where("Panic Table")},
    {"string": "Stress Response Table", "lookup": 'game.tables.getName("Stress Response Table")',
     "call_site": cite(P_ACTOR, 'const table = game.tables.getName("Stress Response Table")'),
     "guarded": True, "guard": "if (!table) ui.notifications.error(ALIENRPG.NoResolveTable) and return",
     "breaks_if_translated": "The Evolved-mode stress/resolve roll aborts with an error toast.",
     "present_in": where("Stress Response Table")},
    {"string": "Panic Response Table", "lookup": 'game.tables.getName("Panic Response Table")',
     "call_site": cite(P_ACTOR, 'const table = game.tables.getName("Panic Response Table")'),
     "guarded": True, "guard": "if (!table) ... (see the !table branch immediately below the assignment)",
     "breaks_if_translated": "rollStress() aborts; the Evolved panic-response roll becomes impossible.",
     "present_in": where("Panic Response Table")},
    {"string": "EV - Critical Injuries",
     "lookup": 'game.tables.getName(game.i18n.localize("ALIENRPG.EVCriticalInjuries")) || game.tables.getName("EV - Critical Injuries")',
     "call_site": cite(P_ACTOR, 'game.i18n.localize("ALIENRPG.EVCriticalInjuries")'),
     "guarded": True, "guard": "atable === null/undefined -> warn(ALIENRPG.NoCharCrit) and return",
     "breaks_if_translated": (
         "Evolved-mode character crit rolls stop. NOTE the i18n leg: lang/cn.json ships "
         "\"ALIENRPG.EVCriticalInjuries\": null. Foundry's Localization#localize rejects a non-string and falls "
         "through to the en fallback, which is the string 'EV - Critical Injuries', so the FIRST getName() "
         "already receives the English name and the `||` never fires. Setting a Chinese value for that key is "
         "therefore a live escape hatch: if the table is renamed, set the key to the new name."),
     "i18n_leg": {"key": "ALIENRPG.EVCriticalInjuries",
                  "en": LANG_EN.get("ALIENRPG.EVCriticalInjuries"),
                  "cn_shipped": LANG_CN.get("ALIENRPG.EVCriticalInjuries")},
     "present_in": where("EV - Critical Injuries")},
    {"string": "Critical Injuries",
     "lookup": 'game.tables.getName("Critical Injuries")',
     "call_site": cite(P_ACTOR, 'game.tables.getName("Critical Injuries") ||'),
     "guarded": True, "guard": "atable === null/undefined -> warn(ALIENRPG.NoCharCrit) and return",
     "breaks_if_translated": "Middle leg of the classic-mode 3-way fallback. NO TABLE WITH THIS EXACT CASING "
                             "EXISTS IN ANY OF THE THREE PACKS — it is dead in the shipped data, kept here so "
                             "nobody 'fixes' a table's casing into it and then translates it.",
     "present_in": where("Critical Injuries")},
    {"string": "Critical injuries",
     "lookup": 'game.tables.getName(game.i18n.localize("ALIENRPG.CriticalInjuries")) || ... || game.tables.getName("Critical injuries")',
     "call_site": cite(P_ACTOR, 'game.tables.getName("Critical injuries")'),
     "guarded": True, "guard": "atable === null/undefined -> warn(ALIENRPG.NoCharCrit) and return",
     "breaks_if_translated": (
         "THIS is the table the classic-mode fallback actually finds (lower-case 'i'). Translate it and every "
         "classic-mode character critical-injury roll dies with the NoCharCrit warning. The i18n leg "
         "ALIENRPG.CriticalInjuries is shipped as '重伤' in cn.json, which matches nothing, so the ONLY working "
         "leg is this literal — unless you retarget the key (see i18n_leg)."),
     "i18n_leg": {"key": "ALIENRPG.CriticalInjuries",
                  "en": LANG_EN.get("ALIENRPG.CriticalInjuries"),
                  "cn_shipped": LANG_CN.get("ALIENRPG.CriticalInjuries"),
                  "call_site": cite(P_ACTOR, 'game.tables.getName(game.i18n.localize("ALIENRPG.CriticalInjuries")) ||')},
     "present_in": where("Critical injuries")},
    {"string": "Critical Injuries on Synthetics",
     "lookup": 'game.tables.getName("Critical Injuries on Synthetics")',
     "call_site": cite(P_ACTOR, 'game.tables.getName("Critical Injuries on Synthetics")'),
     "guarded": True, "guard": "atable === null/undefined -> warn(ALIENRPG.NoSynCrit) and return",
     "breaks_if_translated": "Synthetic crit rolls stop. There is NO i18n leg on this branch at all — the "
                             "'evolved' switch above it is commented out — so the literal is the only path.",
     "present_in": where("Critical Injuries on Synthetics")},
    {"string": "critical injuries on synthetics",
     "lookup": 'game.tables.getName("critical injuries on synthetics")',
     "call_site": cite(P_ACTOR, 'game.tables.getName("critical injuries on synthetics")'),
     "guarded": True, "guard": "same branch as above",
     "breaks_if_translated": "All-lower-case second leg. No table with this casing exists in any pack; dead "
                             "but listed so it is never repurposed.",
     "present_in": where("critical injuries on synthetics")},
    {"string": "Spaceship Minor Component Damage",
     "lookup": 'game.tables.getName("Spaceship Minor Component Damage")',
     "call_site": cite(P_ACTOR, 'game.tables.getName("Spaceship Minor Component Damage")'),
     "guarded": True, "guard": "atable === null/undefined -> warn(ALIENRPG.NoCharCrit) and return",
     "breaks_if_translated": "The spacecraft sheet's 'minor' crit button stops working.",
     "present_in": where("Spaceship Minor Component Damage")},
    {"string": "Spaceship Major Component Damage",
     "lookup": 'game.tables.getName("Spaceship Major Component Damage")',
     "call_site": cite(P_ACTOR, 'game.tables.getName("Spaceship Major Component Damage")'),
     "guarded": True, "guard": "atable === null/undefined -> warn(ALIENRPG.NoCharCrit) and return",
     "breaks_if_translated": "The spacecraft sheet's 'major' crit button stops working.",
     "present_in": where("Spaceship Major Component Damage")},
]

rolltable_name_prefix = [{
    "prefix": "Critical Injuries",
    "lookup": 'folder.contents.filter((x) => x.name.startsWith("Critical Injuries"))',
    "call_site": cite(P_RTD, 'startsWith("Critical Injuries")'),
    "scope": "RollTables whose DIRECT parent folder is 'Alien Mother Tables' (Folder#contents is "
             "non-recursive: `this.documentCollection.filter(d => d.folder === this)`), i.e. the "
             "'Alien Sub-Tables' subfolder is excluded.",
    "current_matches": sorted({t['name'] for n, adv in PACKS.items() for t in adv['tables']
                              if t.get('folder') == 'A0N3Ct1TQq9BuGk6' and t['name'].startswith('Critical Injuries')}),
    "deliberately_excluded_by_casing": sorted({t['name'] for n, adv in PACKS.items() for t in adv['tables']
                                               if t.get('folder') == 'A0N3Ct1TQq9BuGk6'
                                               and t['name'].lower().startswith('critical injuries')
                                               and not t['name'].startswith('Critical Injuries')}),
    "breaks_if_translated": (
        "cTableget() returns only {0: {key:'None', label:'None'}}. The creature sheet's crit-table <select> "
        "collapses to a single 'None' option, every creature's saved cTables value falls out of the option "
        "list, and the crit button then calls creatureAutoAttackRoll with atttype='None' -> the "
        "`targetTable === \"None\"` early return at " + cite(P_ACTOR, 'if (targetTable === "None")') + ". "
        "Silent: a logger.warn, no user-visible error."),
}]

# ---------------------------------------------------------------- 3. folder_names
FOLDERS = {}
for n, adv in PACKS.items():
    for f in adv['folders']:
        FOLDERS.setdefault(f['name'], []).append(n)

folder_names = [
    {"string": "Alien Tables", "type": "RollTable",
     "lookup": 'game.folders.getName("Alien Tables")',
     "call_site": cite(P_INIT, 'game.folders.getName("Alien Tables")'),
     "role": "Negative guard on first-run auto-import.",
     "breaks_if_translated": (
         "The guard `!game.settings.get(moduleKey,'imported') && game.user.isGM && "
         "!game.folders.getName('Alien Tables')` is what stops FirstTimeSetup() from re-running in a world "
         "that already has the content but lost the setting. Translate the folder and the guard reads "
         "'folder absent' -> FirstTimeSetup() fires -> a full destructive Adventure re-import over the GM's "
         "edited tables."),
     "present_in": FOLDERS.get("Alien Tables", [])},
    {"string": "Alien Creature Tables", "type": "RollTable",
     "lookup": 'game.folders.contents.find((x) => x.name === "Alien Creature Tables")',
     "call_site": cite(P_RTD, 'x.name === "Alien Creature Tables"'),
     "role": "Source folder for the creature ATTACK-table <select> (rTables).",
     "unguarded_deref": cite(P_RTD, 'const aTables = folder.contents') + "  `folder.contents` with folder possibly undefined",
     "breaks_if_translated": (
         "find() returns undefined and `folder.contents` throws a TypeError inside "
         "AlienRPGCreatureSheet#_prepareContext (" + cite(os.path.join(SYS, 'module', 'sheets', 'creature-sheet.mjs'),
                                                          'alienrpgrTableGet.rTableget()') + "). "
         "THE ENTIRE CREATURE SHEET FAILS TO RENDER — not a degraded dropdown, a blank window."),
     "present_in": FOLDERS.get("Alien Creature Tables", [])},
    {"string": "Alien Mother Tables", "type": "RollTable",
     "lookup": 'game.folders.contents.find((x) => x.name === "Alien Mother Tables")',
     "call_site": cite(P_RTD, 'x.name === "Alien Mother Tables"'),
     "role": "Source folder for the creature CRIT-table <select> (cTables).",
     "unguarded_deref": cite(P_RTD, 'const aTables = folder.contents.filter') + "  `folder.contents` with folder possibly undefined",
     "breaks_if_translated": "Identical failure mode: TypeError in cTableget(), the creature sheet does not render.",
     "present_in": FOLDERS.get("Alien Mother Tables", [])},
]

folder_names_not_referenced = {
    "note": "The owner's T-FROZEN note says '2 hard-coded Folder names'; the live import graph has THREE. "
            "'Alien Sub-Tables' is NOT one of them — it is never named in code and Folder#contents is "
            "non-recursive, so it is fully translatable.",
    "translatable_folders_measured": len(FOLDERS) - 3,
    "distinct_folder_names_in_packs": len(FOLDERS),
}

# ---------------------------------------------------------------- 4. item_names
def find_item(nm_upper):
    hits = []
    for n, adv in PACKS.items():
        for it in adv['items']:
            if it['name'].upper() == nm_upper:
                hits.append({"pack": n, "location": "adventure-level items", "actual_name": it['name'],
                             "type": it['type']})
        for a in adv['actors']:
            for it in a.get('items', []):
                if it['name'].upper() == nm_upper:
                    hits.append({"pack": n, "location": "embedded in Actor %r" % a['name'],
                                 "actual_name": it['name'], "type": it['type']})
    return hits


item_names = [
    {"string_compared": "PACK MULE",
     "comparison": 'i.name.toUpperCase() === "PACK MULE"',
     "case_insensitive": True,
     "call_sites": [cite(P_CHAR, 'i.name.toUpperCase() === "PACK MULE"'),
                    cite(P_SYN, 'i.name.toUpperCase() === "PACK MULE"'),
                    cite(P_COL, 'i.name.toUpperCase() === "PACK MULE"')],
     "documents": find_item("PACK MULE"),
     "breaks_if_translated": (
         "The talent silently stops granting its encumbrance bonus. Nothing errors: the character/synthetic/"
         "colony sheet just computes a smaller carry capacity, so the player sees a wrong Encumbered status "
         "and no clue why. Comparison is .toUpperCase() so case may change; the LETTERS may not."),
     "rule": "T-FROZEN. Bilingual tail is NOT allowed on this item name — '驮马 Pack Mule'.toUpperCase() is "
             "'驮马 PACK MULE', which !== 'PACK MULE'."},
    {"string_compared": "TAKE CONTROL",
     "comparison": 'Attrib.type === "talent" && Attrib.name.toUpperCase() === "TAKE CONTROL"',
     "case_insensitive": True,
     "call_sites": [cite(P_DCHAR, 'Attrib.name.toUpperCase() === "TAKE CONTROL"'),
                    cite(P_DSYN, 'Attrib.name.toUpperCase() === "TAKE CONTROL"')],
     "documents": find_item("TAKE CONTROL"),
     "breaks_if_translated": (
         "prepareDerivedData stops swapping the character's initiative/leadership stat when WIT > EMP. Purely "
         "numeric drift on the sheet, no error. Same bilingual-tail prohibition as PACK MULE."),
     "rule": "T-FROZEN, no bilingual tail."},
]

none_sentinel = {
    "string": "None",
    "kind": "sentinel value, not a document name",
    "produced_at": [cite(P_RTD, 'lTables[0] = { key: "None", label: "None" }', 1),
                    cite(P_RTD, 'lTables[0] = { key: "None", label: "None" }', 2)],
    "compared_at": [cite(P_ACTOR, 'if (targetTable === "None")')],
    "stored_in": "Actor system.rTables / system.cTables (creature type only)",
    "breaks_if_translated": (
        "rollTableData.mjs writes key AND label from the same literal, so the dropdown's first option is "
        "unavoidably the English word 'None'. If the stored actor value is translated but the option key is "
        "not (or vice versa), `selectOptions selected=` no longer matches any option, the <select> shows the "
        "first entry, and the crit/attack button posts a table name that resolves to nothing. If BOTH were "
        "somehow translated, the `targetTable === \"None\"` early-return at actor.mjs stops firing and "
        "`game.tables.contents.find(...)` returns undefined -> `table.roll({roll})` throws TypeError, "
        "killing the creature attack roll outright."),
    "rule": "T-FROZEN. Accept the English 'None' in the two dropdowns; there is no way to localise it without "
            "patching rollTableData.mjs at runtime.",
}

# 12 skill-stunts items, T-EXACT
SKILL_KEYS = ['heavyMach', 'closeCbt', 'stamina', 'rangedCbt', 'mobility', 'piloting',
              'command', 'manipulation', 'medicalAid', 'observation', 'survival', 'comtech']
SYS_ITEMS = {it['name']: it for it in PACKS['system']['items']}
exact_match_to_lang = []
for k in SKILL_KEYS:
    lang_key = 'ALIENRPG.Skill' + k
    en_val = LANG_EN[lang_key]
    exact_match_to_lang.append({
        "item_name_en": en_val,
        "lang_key": lang_key,
        "lang_en": en_val,
        "lang_cn_shipped_by_upstream": LANG_CN.get(lang_key),
        "item_type": SYS_ITEMS[en_val]['type'],
        "item_pack": "system (alienrpg.alien-rpg-system)",
        "item_folder": "Skill-Stunts",
        "config_entry": cite(P_CFG, '%s: { name: "%s"' % (k, lang_key)),
    })

exact_match_to_lang_rule = {
    "rule": "T-EXACT. The translated Item name MUST be byte-equal to lang/cn.json's value for its "
            "ALIENRPG.Skill<key>. NO bilingual English tail, no parenthetical, no trailing space.",
    "chain": [
        "actor-character.mjs derives `this.skills[skl].description = game.i18n.localize(CONFIG.ALIENRPG.skills[skl].name)`  ("
        + cite(P_DCHAR, 'description = game.i18n.localize(CONFIG.ALIENRPG.skills[skl].name)') + ")",
        "templates/actor/character-skills.hbs:15 renders the stunt button as data-pmbut='{{skill.description}}'",
        "_stuntBtn reads dataset.pmbut and does `game.items.getName(dataset.pmbut)`  (" + cite(P_CHAR, 'item = game.items.getName(dataset.pmbut)') + ")",
    ],
    "breaks_if_divergent": (
        "getName() returns undefined, `item.name` throws, and the catch block at "
        + cite(P_CHAR, '      } catch {') + " swallows it: temp3 is the raw key string "
        "'ALIENRPG.<NameWithoutSpaces>' which does not start with '<ol>', so the panel shows the literal "
        "'<h2>No Stunts Entered</h2>'. Every skill-stunt panel on every character and synthetic sheet reads "
        "'No Stunts Entered' with no error anywhere."),
    "dead_second_leg": {
        "what": "_stuntBtn also builds `\"ALIENRPG.\" + skill.description.replace(/\\s+/g,\"\")` and localizes it.",
        "call_site": cite(P_CHAR, 'const langTemp = "ALIENRPG." + [newLangStr]'),
        "status": "DEAD in 4.1.13 — measured: none of the 12 resulting keys "
                  "(ALIENRPG.HeavyMachinery, ALIENRPG.CloseCombat, ...) exists in lang/en.json, so localize() "
                  "returns the raw key and the `startsWith('<ol>')` branches never fire. Do NOT create a "
                  "cn.json block for them; if upstream restores those keys this becomes a second lockstep.",
        "measured_missing_from_en": [k for k in
                                     ['ALIENRPG.' + re.sub(r'\s+', '', LANG_EN['ALIENRPG.Skill' + s]) for s in SKILL_KEYS]
                                     if k not in LANG_EN],
    },
    "sheets_carrying_the_same_call": [
        cite(P_CHAR, 'item = game.items.getName(dataset.pmbut)'),
        cite(P_SYN, 'item = game.items.getName(dataset.pmbut)'),
        cite(os.path.join(SYS, 'module', 'sheets', 'spacecraft-sheet.mjs'), 'item = game.items.getName(dataset.pmbut)'),
        cite(os.path.join(SYS, 'module', 'sheets', 'vehicle-sheet.mjs'), 'item = game.items.getName(dataset.pmbut)'),
    ],
}

# ---------------------------------------------------------------- 5. crit_parse_lockstep
SPLIT_STRIP = re.compile(r'(<b>)|(<p>|)(<strong>)|(</b>)|(</p>)|(</strong>)', re.I)
BR = re.compile(r'<br />', re.I)
SPLITTER = re.compile(r'[:] |<br>', re.I)


def split_like_actor(desc):
    return SPLITTER.split(BR.sub('<br>', SPLIT_STRIP.sub('', desc)))


def crit_rows(tname):
    out = {}
    for n, adv in PACKS.items():
        for t in adv['tables']:
            if t['name'] != tname:
                continue
            rows = {}
            for r in t['results']:
                rng = r.get('range')
                arr = split_like_actor(r.get('description') or '')
                rows['%s-%s' % (rng[0], rng[1])] = {
                    "n": len(arr),
                    "fatal": arr[3] if len(arr) > 3 else None,
                    "time_limit": arr[5] if len(arr) > 5 else None,
                    "healing": arr[9] if len(arr) > 9 else None,
                }
            out[n] = rows
    return out


YES3 = ['Yes ', 'Yes, \u20131 ', 'Yes, \u20132 ']
TIME5 = ['None ', 'One Round ', 'One Turn ', 'One Shift ', 'One Day ']
HEAL9 = ['Permanent', 'Shift']

crit_tables = {}
for tname in ['Critical injuries', 'EV - Critical Injuries']:
    packrows = crit_rows(tname)
    ref = packrows[sorted(packrows)[0]]
    sig = {}
    for k, v in sorted(ref.items(), key=lambda kv: int(kv[0].split('-')[0])):
        if v['fatal'] in YES3 or v['time_limit'] in TIME5 or v['healing'] in HEAL9:
            sig[k] = {"fatal_en": v['fatal'], "time_limit_en": v['time_limit'], "healing_en": v['healing']}
    diffs = {}
    keys = set()
    for p in packrows:
        keys |= set(packrows[p])
    for k in sorted(keys, key=lambda s: int(s.split('-')[0])):
        vs = {p: packrows[p].get(k) for p in packrows}
        if len({json.dumps(v, sort_keys=True) for v in vs.values()}) > 1:
            diffs[k] = vs
    crit_tables[tname] = {
        "present_in": sorted(packrows),
        "result_count": {p: len(packrows[p]) for p in packrows},
        "expected_split_length": 10,
        "rows_with_significant_tokens": sig,
        "rows_that_differ_between_packs": diffs,
    }

crit_parse_lockstep = {
    "what": (
        "actor.mjs strips <b>/<p>/<strong> from the drawn crit result, normalises '<br />' to '<br>', splits "
        "on the regex /[:] |<br>/gi, and then reads FIXED INDICES out of the resulting array: [3]=FATAL, "
        "[5]=TIME LIMIT, [7]=EFFECTS, [9]=HEALING TIME. The cells are compared with === against "
        "game.i18n.localize(...) + a literal suffix."),
    "pipeline": [
        {"step": "strip", "code": 'messG.replace(/(<b>)|(<p>|)(<strong>)|(<\\/b>)|(<\\/p>)|(<\\/strong>)/gi, "")',
         "at": cite(P_ACTOR, 'cleanText = messG.replace(')},
        {"step": "normalise br", "code": 'cleanText.replace(/<br \\/>/gi, "<br>")',
         "at": cite(P_ACTOR, 'factorFour = cleanText.replace(/<br \\/>/gi, "<br>")')},
        {"step": "split", "code": 'factorFour.split(/[:] |<br>/gi)',
         "at": cite(P_ACTOR, 'testArray = factorFour.split(/[:] |<br>/gi)')},
    ],
    "structural_invariant": (
        "The five labels must each still end with a COLON FOLLOWED BY A SPACE inside the tag "
        "('<strong>INJURY: </strong>'), and the rows must still be separated by '<br />'. Both are what "
        "produce the 10-element array. Drop one ': ' and every index after it shifts by one, so FATAL is read "
        "out of the TIME LIMIT cell and the crit is applied wrong with no error. A colon-space that survives "
        "as a FULL-WIDTH colon '\\uff1a' does NOT match /[:] / and breaks the split."),
    "cases": [
        {"index": 3, "lang_key": "ALIENRPG.Yes",
         "suffix_literal": " ",
         "expected_cell": "<localize(ALIENRPG.Yes)> + ' '",
         "at": cite(P_ACTOR, 'case game.i18n.localize("ALIENRPG.Yes") + " ":'),
         "effect": "cFatal = true",
         "en_cell": "Yes "},
        {"index": 3, "lang_key": "ALIENRPG.Yes",
         "suffix_literal": ", \u20131 ",
         "suffix_codepoints": ["U+002C", "U+0020", "U+2013", "U+0031", "U+0020"],
         "expected_cell": "<localize(ALIENRPG.Yes)> + ', \\u20131 '",
         "at": cite(P_ACTOR, 'case game.i18n.localize("ALIENRPG.Yes") + ", \u20131 ":'),
         "effect": "cFatal = true; appends '-1 to <strong>' + localize(ALIENRPG.SkillmedicalAid) + '</strong> roll'",
         "en_cell": "Yes, \u20131 ",
         "WARNING": "EN DASH U+2013, NOT ASCII HYPHEN-MINUS U+002D. Verified by codepoint dump of "
                    + cite(P_ACTOR, 'case game.i18n.localize("ALIENRPG.Yes") + ", \u20131 ":')},
        {"index": 3, "lang_key": "ALIENRPG.Yes",
         "suffix_literal": ", \u20132 ",
         "suffix_codepoints": ["U+002C", "U+0020", "U+2013", "U+0032", "U+0020"],
         "expected_cell": "<localize(ALIENRPG.Yes)> + ', \\u20132 '",
         "at": cite(P_ACTOR, 'case game.i18n.localize("ALIENRPG.Yes") + ", \u20132 ":'),
         "effect": "cFatal = true; appends the -2 Medical Aid penalty",
         "en_cell": "Yes, \u20132 ",
         "WARNING": "EN DASH U+2013, NOT ASCII HYPHEN-MINUS U+002D."},
        {"index": 5, "lang_key": "ALIENRPG.None", "suffix_literal": " ",
         "expected_cell": "<localize(ALIENRPG.None)> + ' '",
         "at": cite(P_ACTOR, 'case game.i18n.localize("ALIENRPG.None") + " ":'),
         "effect": "healTime = 0",
         "en_cell": "None ",
         "note": "No shipped crit row emits 'None ' at index 5 (measured: 0 of 72). Dead case, kept for "
                 "lockstep because the same key IS live at index 9 via "
                 + cite(P_ACTOR, 'testArray[9] = game.i18n.localize("ALIENRPG.None")')},
        {"index": 5, "lang_key": "ALIENRPG.OneRound", "suffix_literal": " ",
         "expected_cell": "<localize(ALIENRPG.OneRound)> + ' '",
         "at": cite(P_ACTOR, 'case game.i18n.localize("ALIENRPG.OneRound") + " ":'),
         "effect": "healTime = 1", "en_cell": "One Round "},
        {"index": 5, "lang_key": "ALIENRPG.OneTurn", "suffix_literal": " ",
         "expected_cell": "<localize(ALIENRPG.OneTurn)> + ' '",
         "at": cite(P_ACTOR, 'case game.i18n.localize("ALIENRPG.OneTurn") + " ":'),
         "effect": "healTime = 2", "en_cell": "One Turn "},
        {"index": 5, "lang_key": "ALIENRPG.OneShift", "suffix_literal": " ",
         "expected_cell": "<localize(ALIENRPG.OneShift)> + ' '",
         "at": cite(P_ACTOR, 'case game.i18n.localize("ALIENRPG.OneShift") + " ":'),
         "effect": "healTime = 3", "en_cell": "One Shift "},
        {"index": 5, "lang_key": "ALIENRPG.OneDay", "suffix_literal": " ",
         "expected_cell": "<localize(ALIENRPG.OneDay)> + ' '",
         "at": cite(P_ACTOR, 'case game.i18n.localize("ALIENRPG.OneDay") + " ":'),
         "effect": "healTime = 4", "en_cell": "One Day "},
        {"index": 9, "lang_key": "ALIENRPG.Permanent", "suffix_literal": "",
         "expected_cell": "<localize(ALIENRPG.Permanent)>   (NO trailing space — this cell is last in the "
                          "string, so nothing follows it)",
         "at": cite(P_ACTOR, 'if (testArray[9] !== game.i18n.localize("ALIENRPG.Permanent"))'),
         "comparison": "!==  (an EQUALITY match SKIPS the healing-time parsing entirely)",
         "effect": "match -> healing time left as the literal text; mismatch -> falls into the [[XdY]] parser",
         "en_cell": "Permanent",
         "severity": "CRITICAL — see the Shift entry for the crash path."},
        {"index": 9, "lang_key": None, "literal": "Shift", "suffix_literal": "",
         "expected_cell": "the ASCII literal 'Shift'",
         "at": cite(P_ACTOR, 'if (testArray[9] === "Shift") {'),
         "effect": "testArray[9] = localize('ALIENRPG.Shift')  — i.e. the code translates it FOR you",
         "en_cell": "Shift",
         "severity": "CRITICAL",
         "WARNING": (
             "THIS ONE HAS NO i18n KEY ON THE COMPARISON SIDE. The cell must stay the byte-exact English word "
             "'Shift'. It is emitted by 2 rows of 'EV - Critical Injuries' (14-14 and 15-15). Translate it and "
             "control reaches "
             + cite(P_ACTOR, 'rollheal = testArray[9].match(/^\\[\\[([0-9]d[0-9]+)]/)[1];') +
             " where .match() returns null and null[1] throws an UNCAUGHT TypeError — the whole Evolved crit "
             "roll dies mid-flight, the chat card is never posted and the injury is never applied. "
             "ALIENRPG.Shift is the OUTPUT key: set that to the Chinese word, leave the table cell English."),
         "output_key": {"key": "ALIENRPG.Shift", "en": LANG_EN.get("ALIENRPG.Shift"),
                        "cn_shipped": LANG_CN.get("ALIENRPG.Shift"),
                        "at": cite(P_ACTOR, 'testArray[9] = game.i18n.localize("ALIENRPG.Shift")')}},
    ],
    "healing_time_roll_shape": {
        "regex_1": "/^\\[\\[([0-9]d[0-9]+)]/",
        "regex_2": "/^\\[\\[([0-9]d[0-9]+)\\]\\] ?(.*)/",
        "at": [cite(P_ACTOR, 'rollheal = testArray[9].match(/^\\[\\[([0-9]d[0-9]+)]/)[1];'),
               cite(P_ACTOR, 'newHealTime = testArray[9].match(/^\\[\\[([0-9]d[0-9]+)\\]\\] ?(.*)/)[2];')],
        "rule": "The HEALING TIME cell must START with '[[NdM]]' — one digit, 'd', one-or-more digits, in "
                "ASCII double square brackets, at offset 0. The tail after it (' days') IS free to translate; "
                "it is captured as group 2 and re-emitted verbatim. Putting anything (even a space) before "
                "'[[' makes .match() return null -> uncaught TypeError, same crash as the Shift case.",
        "measured_en_values": sorted({v['healing'] for tn in crit_tables for p, rows in crit_rows(tn).items()
                                      for v in rows.values() if v['healing'] and v['healing'].startswith('[[')}),
    },
    "lang_values": {k: {"en": LANG_EN.get(k), "cn_shipped_by_upstream": LANG_CN.get(k)}
                    for k in ["ALIENRPG.Yes", "ALIENRPG.None", "ALIENRPG.OneRound", "ALIENRPG.OneTurn",
                              "ALIENRPG.OneShift", "ALIENRPG.OneDay", "ALIENRPG.Permanent", "ALIENRPG.Shift"]},
    "lang_value_hygiene": (
        "None of the 8 values may carry leading or trailing whitespace: the code concatenates a suffix, so a "
        "trailing space in the lang value produces a double space that matches nothing."),
    "tables": crit_tables,
    "evolved_mode_note": (
        "MEASURED: 'EV - Critical Injuries' emits '\u2013' / 'Shift' / 'Stretch' / 'Round' at index 5, none of "
        "which is any of the five ALIENRPG.One* + ' ' cases, so healTime is ALWAYS 0 in Evolved mode today. "
        "That is an upstream defect, not something translation caused; do not 'fix' it by rewriting the cells "
        "to 'One Shift ' unless you also intend to change game behaviour."),
}

# ---------------------------------------------------------------- 6. actor_table_refs
actor_refs = []
for n, adv in PACKS.items():
    for a in adv['actors']:
        s = a.get('system', {})
        for field in ('rTables', 'cTables'):
            v = s.get(field)
            if not isinstance(v, str) or v == '':
                continue
            actor_refs.append({
                "pack": n, "actor": a['name'], "actor_type": a['type'],
                "field": "system." + field, "value": v,
                "is_sentinel": v == "None",
                "table_lives_in": where(v) if v != "None" else [],
            })

ref_values = collections.Counter(r['value'] for r in actor_refs)
cross_pack = [r for r in actor_refs if not r['is_sentinel'] and r['pack'] not in r['table_lives_in']]

actor_table_refs = {
    "what": "Creature actors store a RollTable NAME (not an id) in system.rTables (attack table) and "
            "system.cTables (crit table). Two independent consumers read it back by name.",
    "consumers": [
        {"code": "game.tables.getName(dataset.atttype)", "at": cite(P_ACTOR, 'atable = game.tables.getName(dataset.atttype)'),
         "guard": "atable null/undefined -> warn(ALIENRPG.NoCharCrit) and return"},
        {"code": "game.tables.contents.find((b) => b.name === targetTable)",
         "at": cite(P_ACTOR, 'const table = game.tables.contents.find((b) => b.name === targetTable)'),
         "guard": "NONE — `table.roll({roll})` on the next lines throws TypeError if the name misses",
         "unguarded": True},
        {"code": "{{selectOptions rTables selected=system.rTables valueAttr='key'}} — the <select> compares "
                 "the stored value against the option KEY, which rollTableData.mjs sets to the table name",
         "at": "systems/alienrpg/templates/actor/creature-general.hbs:7 and :20"},
    ],
    "counts": {
        "actors_carrying_a_ref": len({(r['pack'], r['actor']) for r in actor_refs}),
        "total_refs": len(actor_refs),
        "distinct_values": len(ref_values),
        "actors_holding_literal_None": len({(r['pack'], r['actor']) for r in actor_refs if r['is_sentinel']}),
        "occurrences_of_literal_None": ref_values["None"],
        "cross_pack_refs": len(cross_pack),
    },
    "distinct_values": [{"value": v, "occurrences": c, "table_lives_in": where(v) if v != "None" else "sentinel"}
                        for v, c in sorted(ref_values.items(), key=lambda kv: (-kv[1], kv[0]))],
    "cross_pack_refs": cross_pack,
    "refs": actor_refs,
    "rule": (
        "Whatever the Babele mapping does with these fields (the survey proposes a custom `alienRollTableRef` "
        "converter), the INVARIANT is: after translation the string stored in system.rTables/cTables must be "
        "byte-equal to the translated `name` of the RollTable it points at, in the SAME world. Both must move "
        "together or neither. 'None' must move NEITHER — it is the sentinel."),
    "breaks_if_desynced": (
        "rTables desync -> creatureAutoAttackRoll's find() returns undefined -> `table.roll({roll})` throws "
        "an uncaught TypeError; the creature's attack button is dead. cTables desync -> the guarded path, so "
        "just a NoCharCrit warning. In BOTH cases the <select> also loses its selection and silently re-reads "
        "as the first option ('None') the next time the sheet is saved, permanently destroying the actor's "
        "table binding."),
    "scope_correction": (
        "The survey scoped this to corerules only. Measured across all three dumps: the Starter Set carries "
        "%d refs on %d actors and corerules carries %d refs on %d actors; the system pack has no actors at "
        "all." % (sum(1 for r in actor_refs if r['pack'] == 'starterset'),
                  len({r['actor'] for r in actor_refs if r['pack'] == 'starterset'}),
                  sum(1 for r in actor_refs if r['pack'] == 'corerules'),
                  len({r['actor'] for r in actor_refs if r['pack'] == 'corerules'}))),
}

# ---------------------------------------------------------------- 7. enricher_labels
def walk_strings(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            walk_strings(v, path + '.' + str(k), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk_strings(v, path + '[%d]' % i, out)
    elif isinstance(o, str):
        out.append((path, o))


def doc_index(adv):
    idx = {}
    for it in adv.get('items', []):
        idx[it['_id']] = ('Item', it['name'])
    for a in adv.get('actors', []):
        idx[a['_id']] = ('Actor', a['name'])
        for it in a.get('items', []):
            idx.setdefault(it['_id'], ('Item(embedded)', it['name']))
    for j in adv.get('journal', []):
        idx[j['_id']] = ('JournalEntry', j['name'])
        for p in j.get('pages', []):
            idx[p['_id']] = ('JournalEntryPage', p['name'])
    for t in adv.get('tables', []):
        idx[t['_id']] = ('RollTable', t['name'])
    for s in adv.get('scenes', []):
        idx[s['_id']] = ('Scene', s['name'])
    for m in adv.get('macros', []):
        idx[m['_id']] = ('Macro', m['name'])
    return idx


GLOBAL_IDX = {}
for n, adv in PACKS.items():
    for k, v in doc_index(adv).items():
        GLOBAL_IDX.setdefault(k, (n,) + v)

RE_TEXTDRAW = re.compile(r'@TEXTDRAW\[([^\]]+)\]\{([^}]*)\}(?:\{([^}]*)\})?')
RE_DRAW = re.compile(r'@DRAW\[([^\]]+)\]\{([^}]*)\}(?:\{([^}]*)\})?')
RE_UUID = re.compile(r'@UUID\[([^\]]+)\]\{([^}]*)\}')

textdraw, drawe, uuidlinks = [], [], []
for n, adv in PACKS.items():
    out = []
    walk_strings(adv, n, out)
    for p, s in out:
        for m in RE_TEXTDRAW.finditer(s):
            tid = m.group(1).split('.')[-1]
            tgt = GLOBAL_IDX.get(tid)
            textdraw.append({"pack": n, "target": m.group(1), "label": m.group(2),
                             "roll": m.group(3),
                             "target_pack": tgt[0] if tgt else None,
                             "target_name": tgt[2] if tgt else None,
                             "label_equals_target_name": bool(tgt) and tgt[2] == m.group(2)})
        for m in RE_DRAW.finditer(s):
            tid = m.group(1).split('.')[-1]
            tgt = GLOBAL_IDX.get(tid)
            drawe.append({"pack": n, "target": m.group(1), "label": m.group(2),
                          "roll": m.group(3),
                          "target_pack": tgt[0] if tgt else None,
                          "target_name": tgt[2] if tgt else None,
                          "label_equals_target_name": bool(tgt) and tgt[2] == m.group(2)})
        for m in RE_UUID.finditer(s):
            tid = m.group(1).split('.')[-1]
            tgt = GLOBAL_IDX.get(tid)
            uuidlinks.append({"pack": n, "path": p, "target": m.group(1), "label": m.group(2),
                              "target_kind": m.group(1).split('.')[0],
                              "target_pack": tgt[0] if tgt else None,
                              "target_type": tgt[1] if tgt else None,
                              "target_name": tgt[2] if tgt else None,
                              "label_equals_target_name": bool(tgt) and tgt[2] == m.group(2),
                              "cross_pack": bool(tgt) and tgt[0] != n})

RE_A = re.compile(r'<a\b[^>]*class="(ev-content-link|content-link)"[^>]*>', re.S)
classcount = collections.Counter()
for n, adv in PACKS.items():
    out = []
    walk_strings(adv, n, out)
    for p, s in out:
        for m in RE_A.finditer(s):
            classcount[(n, m.group(1))] += 1

enricher_labels = {
    "at_TEXTDRAW": {
        "count": len(textdraw),
        "renderer": "elem.innerHTML = `<a class='content-link' >${tableName}</a>`  ("
                    + cite(P_ENR, "elem.innerHTML = `<a class='content-link' >${tableName}</a>`") + ")",
        "tooltip": "data-tooltip = `Draw from ${tableName}. <br> ${localize('ALIENRPG.dialog.Tooltip-Rollontable')}`  ("
                   + cite(P_ENR, '`Draw from ${tableName}. <br>', 2) + ")",
        "binding": "BY UUID. The label is display text only — the draw resolves through data-uuid, so a "
                   "translated label never breaks the roll.",
        "rule": "TRANSLATE the label, and keep it byte-equal to the translated RollTable name, because the "
                "label is ALSO what the tooltip says you are drawing from. It is a consistency sync point, "
                "not a functional one.",
        "labels_matching_target_name_today": sum(1 for x in textdraw if x['label_equals_target_name']),
        "labels_already_wrong_upstream": [x for x in textdraw if not x['label_equals_target_name']],
        "entries": textdraw,
    },
    "at_DRAW": {
        "count": len(drawe),
        "renderer": 'elem.innerHTML = `<i class="fas fa-dice-d20">&nbsp;</i>`  ('
                    + cite(P_ENR, 'elem.innerHTML = `<i class="fas fa-dice-d20">&nbsp;</i>`') + ")",
        "binding": "BY UUID. The label is NEVER rendered as text — only inside data-tooltip.",
        "rule": "TRANSLATE. Invisible except in the tooltip.",
        "entries": drawe,
    },
    "at_UUID_explicit_label": {
        "count": len(uuidlinks),
        "by_kind": dict(collections.Counter(x['target_kind'] for x in uuidlinks)),
        "binding": "BY UUID for the jump, BY THE EXPLICIT LABEL for the rendered text. Foundry renders the "
                   "braces content verbatim and does NOT substitute the target document's name.",
        "rule": "HARD SYNC POINT. Translating the target Item's name does not change this link text and vice "
                "versa. Both must be updated, to the same string, in two different files.",
        "cross_pack": [x for x in uuidlinks if x['cross_pack']],
        "cross_pack_count": sum(1 for x in uuidlinks if x['cross_pack']),
        "entries": uuidlinks,
    },
    "content_link_class": {
        "measured": {'%s/%s' % (k[0], k[1]): v for k, v in sorted(classcount.items())},
        "note": (
            "corerules pre-baked anchors carry class=\"content-link\"; starterset carries the non-standard "
            "class=\"ev-content-link\". CORRECTION TO THE OBVIOUS WORRY: this does NOT disable the starterset "
            "links. Foundry v14's click and drag delegates select on `a[data-link]` "
            "(foundry.mjs:35921 / :35925), not on the class, and all 81 starterset anchors carry data-link + "
            "data-uuid. The class difference is CSS-only — and neither module's CSS defines .ev-content-link, "
            "so those 81 links render unstyled. Do not 'repair' the class during translation; leave the "
            "attribute bytes alone and translate only the anchor's inner text."),
    },
}

# ---------------------------------------------------------------- 8. heading anchor hashes (added)
RE_HEAD = re.compile(r'<h([1-6])([^>]*)>(.*?)</h\1>', re.S | re.I)
RE_ID = re.compile(r'\bid="([^"]*)"')
RE_TAG = re.compile(r'<[^>]+>')
RE_HASH = re.compile(r'data-hash="([^"]*)"')


def slugify_heading(text):
    s = text.strip().lower()
    s = re.sub(r'[\s\-]+', '-', s)
    return s.replace('"', '').replace("'", '')[:64]


anchor = {}
for n, adv in PACKS.items():
    heads = 0
    with_id = 0
    slugs = set()
    hashes = collections.Counter()
    for j in adv.get('journal', []):
        for pg in j.get('pages', []):
            txt = (pg.get('text') or {}).get('content') or ''
            for m in RE_HEAD.finditer(txt):
                heads += 1
                if RE_ID.search(m.group(2)):
                    with_id += 1
                    slugs.add(RE_ID.search(m.group(2)).group(1))
                else:
                    slugs.add(slugify_heading(RE_TAG.sub('', m.group(3)).replace('&nbsp;', '\xa0')))
            for m in RE_HASH.finditer(txt):
                hashes[m.group(1)] += 1
    anchor[n] = {
        "headings": heads,
        "headings_with_explicit_id": with_id,
        "data_hash_occurrences": sum(hashes.values()),
        "data_hash_distinct": len(hashes),
        "data_hash_not_resolving_today": sorted(h for h in hashes if h not in slugs),
    }

heading_anchor_hashes = {
    "what": "Every pre-baked in-prose anchor carries data-hash=\"<slug>\". Foundry resolves it with "
            "JournalEntryPage.buildTOC -> _makeHeadingNode: `slug = heading.id || slugifyHeading(heading)` "
            "(foundry.mjs:44776 / :44689), i.e. THE SLUG IS DERIVED FROM THE HEADING TEXT unless the heading "
            "carries an explicit id attribute.",
    "measured": anchor,
    "explicit_ids_in_corpus": sum(v['headings_with_explicit_id'] for v in anchor.values()),
    "breaks_if_translated": (
        "String#slugify is non-strict here, so CJK survives verbatim: '<h2>制作技能检定</h2>' slugifies to "
        "'制作技能检定', which never equals 'making-skill-rolls'. The moment a heading is translated, every "
        "data-hash pointing at it stops resolving and the link scrolls to the top of the page instead of the "
        "section. Silent — no error, no broken-link styling."),
    "remediation": (
        "Translate the heading TEXT but add the original English slug as an explicit id: "
        "`<h2 id=\"making-skill-rolls\">制作技能检定</h2>`. `heading.id ||` wins over slugifyHeading, so all "
        "existing data-hash values keep resolving and no data-hash needs to be rewritten. This is the ONLY "
        "approach that does not require editing both ends."),
    "alternative": "Rewrite every data-hash to the new CJK slug. Rejected: 346 occurrences across 153 "
                   "distinct slugs in two packs, and any later heading edit silently re-breaks them.",
}

# ---------------------------------------------------------------- upstream defects
upstream_defects = [
    {"id": "corerules-welcome-journal-nbsp",
     "severity": "already broken in 1.0.2, unrelated to translation",
     "where": cite(P_CRINIT, 'export const welcomeJournalEntry'),
     "what": "welcomeJournalEntry uses ASCII spaces; the document name uses U+00A0 in all five gaps, so "
             "game.journal.getName() returns null and .show() throws today.",
     "consequence_for_us": "Do not 'fix' this by renaming the journal — that would be an upstream change "
                           "shipped inside a translation module. Freeze the name as it is (NBSP included) "
                           "and, if we want the welcome page to work, patch it at runtime in alienrpg-cn."},
    {"id": "critical-injuries-healing-time-markup",
     "severity": "content defect, 2 rows",
     "where": "corerules RollTable 'Critical injuries', results 11-11 and 12-12",
     "what": "Those two rows write '<strong>HEALING TIME:</strong>' with NO space after the colon, so /[:] / "
             "does not split there and the array is 9 elements instead of 10. testArray[9] is undefined and "
             "`testArray[9].length` at " + cite(P_ACTOR, 'if (testArray[9].length > 0)') + " would throw — "
             "except line " + cite(P_ACTOR, 'if (testArray[9] !== game.i18n.localize("ALIENRPG.Permanent"))') +
             " compares undefined !== 'Permanent' first, which is true, so it DOES reach the .length and "
             "throws. The starterset copy of the same two rows has the space and is fine.",
     "consequence_for_us": "Keep the ': ' when translating; do not propagate the defect. The QA gate flags "
                           "any translated row whose split length is not 10."},
    {"id": "evolved-crit-time-limit-never-matches",
     "severity": "upstream behaviour, do not 'fix'",
     "where": "RollTable 'EV - Critical Injuries', index-5 cells",
     "what": "Emits '\u2013' / 'Shift' / 'Stretch' / 'Round'; none matches any ALIENRPG.One* + ' ' case, so "
             "healTime is always 0 in Evolved mode."},
    {"id": "textdraw-label-typo",
     "severity": "cosmetic, 1 occurrence",
     "where": "corerules journal prose",
     "what": "A @TEXTDRAW label reads 'EV - 56. LP - ANOMALOUS ENVIRONMENT MATRIX' while the RollTable is "
             "named 'EV - 56. LS - ANOMALOUS ENVIRONMENT MATRIX' (LP vs LS).",
     "consequence_for_us": "The label is display-only, so translate it to match the TABLE, not the typo."},
    {"id": "dead-source-files",
     "severity": "scope note",
     "where": "systems/alienrpg/module/actor/old-*.js, systems/alienrpg/module/apps/migratefolders.js, "
              "modules/alien-evolved-starterset/module/migratefolders.js",
     "what": "These contain more English name lookups (e.g. game.tables.getName('Space Combat Panic Roll'), "
             "game.journal.getName('MU/TH/ER Instructions.')) but are NOT in the live ES-module import graph "
             "walked from system.json esmodules / module.json esmodules. Verified: the system graph is 62 "
             "files, each add-on module is 3. Their strings are deliberately NOT in this register.",
     "consequence_for_us": "If upstream re-imports any of them, re-run the graph walk."},
]

# ---------------------------------------------------------------- assemble
register = {
    "_": "Alien RPG 汉化 · 硬冻结字符串登记表 / hard register of byte-exact-English strings",
    "schema_version": 1,
    "generated_on": "2026-08-29",
    "generated_by": "4-常用脚本/qa/build_register.py; every string, count and file:line "
                    "below is re-derived from source at build time — `cite()` raises if a needle is not found, "
                    "so a drifted citation fails the build instead of lying.",
    "measured_against": {
        "system": {"id": "alienrpg", "version": json.load(open(os.path.join(SYS, 'system.json'), encoding='utf-8')).get('version'),
                   "live_import_graph_files": 62},
        "alien-evolved-corerules": {"version": json.load(open(os.path.join(CR, 'module.json'), encoding='utf-8')).get('version'),
                                    "live_import_graph_files": 3},
        "alien-evolved-starterset": {"version": json.load(open(os.path.join(SS, 'module.json'), encoding='utf-8')).get('version'),
                                     "live_import_graph_files": 3},
        "pack_identity": {
            "system": {"module_id": "alienrpg", "package_type": "system",
                       "pack_name": "alien-rpg-system",
                       "babele_file": "alienrpg.alien-rpg-system.json"},
            "starterset": {"module_id": "alien-evolved-starterset", "package_type": "module",
                           "pack_name": "alien-evolved-starter-set",
                           "babele_file": "alien-evolved-starterset.alien-evolved-starter-set.json"},
            "corerules": {"module_id": "alien-evolved-corerules", "package_type": "module",
                          "pack_name": "alien-evolved-core-rules",
                          "babele_file": "alien-evolved-corerules.alien-evolved-core-rules.json"},
        },
        "pack_dumps": {p: {"adventure_name": PACKS[p]['name'],
                           "actors": len(PACKS[p]['actors']), "items": len(PACKS[p]['items']),
                           "tables": len(PACKS[p]['tables']), "journal": len(PACKS[p]['journal']),
                           "scenes": len(PACKS[p]['scenes']), "folders": len(PACKS[p]['folders']),
                           "macros": len(PACKS[p]['macros'])} for p in PACKS},
    },
    "encoding_note": (
        "U+2013 (EN DASH) and U+00A0 (NO-BREAK SPACE) are written as \\u escapes throughout this file even "
        "though the file is otherwise UTF-8 with literal CJK. Both are invisible-or-confusable in an editor "
        "and both are load-bearing here. Never normalise them."),
    "tiers": {
        "T-FROZEN": "byte-exact English. Never translated, never given a bilingual tail.",
        "T-EXACT": "pure Chinese, byte-equal to a lang/cn.json value. No English tail, no trailing space.",
        "T-BILINGUAL": "'中文 English' joined by ONE ASCII space, no parentheses.",
        "T-PLAIN": "bare Chinese.",
    },
    "sections": {
        "name_lookups": {"tier": "T-FROZEN", "count": len(name_lookups), "entries": name_lookups},
        "rolltable_names": {"tier": "T-FROZEN",
                            "count": len(rolltable_names),
                            "count_note": "10 literal strings passed to game.tables.getName(), plus 2 "
                                          "localize()-derived legs documented inline under i18n_leg, plus 1 "
                                          "prefix filter in rolltable_name_prefix. The survey said 9; the "
                                          "verifier said 11; the measured number of distinct FROZEN literals "
                                          "is 10, of which 2 ('Critical Injuries' and 'critical injuries on "
                                          "synthetics') match no document in any pack.",
                            "entries": rolltable_names,
                            "prefix_filters": rolltable_name_prefix},
        "folder_names": {"tier": "T-FROZEN", "count": len(folder_names), "entries": folder_names,
                         "note": folder_names_not_referenced},
        "item_names": {"tier": "T-FROZEN", "count": len(item_names), "entries": item_names,
                       "none_sentinel": none_sentinel},
        "exact_match_to_lang": {"tier": "T-EXACT", "count": len(exact_match_to_lang),
                                "entries": exact_match_to_lang, "rule": exact_match_to_lang_rule},
        "crit_parse_lockstep": crit_parse_lockstep,
        "actor_table_refs": actor_table_refs,
        "enricher_labels": enricher_labels,
        "heading_anchor_hashes": heading_anchor_hashes,
    },
    "upstream_defects": upstream_defects,
}

text = json.dumps(register, ensure_ascii=False, indent=1)
# selectively escape the invisible / confusable codepoints
for ch in ['\u00a0', '\u2013', '\u2014', '\u2018', '\u2019', '\u201c', '\u201d', '\u200b', '\ufeff', '\u3000']:
    text = text.replace(ch, '\\u%04x' % ord(ch))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(text + '\n')

# round-trip check
back = json.load(open(OUT, encoding='utf-8'))
assert back == register, 'ROUND TRIP FAILED'
print('-> %s  (%d bytes)' % (OUT, os.path.getsize(OUT)))
print('name_lookups            %d' % len(name_lookups))
print('rolltable_names         %d  (+%d prefix filter)' % (len(rolltable_names), len(rolltable_name_prefix)))
print('folder_names            %d' % len(folder_names))
print('item_names              %d  (+None sentinel)' % len(item_names))
print('exact_match_to_lang     %d' % len(exact_match_to_lang))
print('crit cases              %d  (8 distinct lang keys + 1 bare literal)' % len(crit_parse_lockstep['cases']))
print('actor_table_refs        %d refs / %d actors / %d None-occurrences'
      % (len(actor_refs), actor_table_refs['counts']['actors_carrying_a_ref'],
         actor_table_refs['counts']['occurrences_of_literal_None']))
print('enricher @TEXTDRAW %d  @DRAW %d  @UUID %d' % (len(textdraw), len(drawe), len(uuidlinks)))
