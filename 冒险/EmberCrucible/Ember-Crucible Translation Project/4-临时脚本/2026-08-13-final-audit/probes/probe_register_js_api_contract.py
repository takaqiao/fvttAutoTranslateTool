#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probe_register_js_api_contract.py  (READ-ONLY)

Re-verifies, against the *installed* upstream sources, every upstream fact that the
register.js findings of round 13 rest on.  Nothing is written outside stdout.

Upstream roots (adjust if your install differs):
  FOUNDRY = C:/Program Files/Foundry Virtual Tabletop/resources/app     (v14 build 365)
  DATA    = C:/Users/Taka/AppData/Local/FoundryVTT/Data

Each check prints PASS/FAIL plus the file:line evidence so a reviewer can re-read it.
"PASS" means "the upstream fact the finding claims is true", i.e. the finding stands.

False-positive modes of this probe:
  - It is a *text* probe.  A refactor that keeps the same strings but changes the
    semantics would still report PASS.  The line numbers are printed so the reader
    can eyeball the real code.
  - Checks 1/2 rely on ordering of two line numbers inside one file; if Foundry ever
    splits setupGame() across files this becomes meaningless (it will FAIL loudly,
    not silently pass).
"""

import io
import os
import re
import sys

FOUNDRY = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"

results = []


def read(path):
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def find(path, needle, regex=False):
    """Return list of (lineno, text) matching."""
    out = []
    try:
        lines = read(path)
    except OSError as exc:
        return [(-1, "CANNOT READ %s: %s" % (path, exc))]
    for i, line in enumerate(lines, 1):
        hit = re.search(needle, line) if regex else (needle in line)
        if hit:
            out.append((i, line.strip()))
    return out


def check(name, ok, evidence):
    results.append((name, bool(ok), evidence))


# --------------------------------------------------------------------------
# 1. initializeDocuments() runs BEFORE Hooks.callAll("setup")
#    -> anything constructed during world-document preparation (every
#       CrucibleAction, via CrucibleActionField.initialize) already exists when
#       a module's `setup` hook fires.
# --------------------------------------------------------------------------
game_mjs = os.path.join(FOUNDRY, "client", "game.mjs")
init_docs = find(game_mjs, r"^\s*this\.initializeDocuments\(\);", regex=True)
setup_hook = find(game_mjs, r'Hooks\.callAll\("setup"\)', regex=True)
ok = bool(init_docs and setup_hook and init_docs[0][0] < setup_hook[0][0])
check(
    "setup-too-late | initializeDocuments() precedes Hooks.callAll('setup')",
    ok,
    "%s: initializeDocuments@%s  setup@%s"
    % (game_mjs, init_docs and init_docs[0][0], setup_hook and setup_hook[0][0]),
)

# world documents are actually *prepared* in that same call
prep = find(game_mjs, "_safePrepareData()")
check(
    "setup-too-late | world documents are prepared inside initializeDocuments()",
    bool(prep),
    "%s:%s" % (game_mjs, prep and prep[0][0]),
)

# --------------------------------------------------------------------------
# 2. package objects (game.world) have no getFlag/setFlag
# --------------------------------------------------------------------------
pkg_hits = []
for root in (
    os.path.join(FOUNDRY, "client", "packages"),
    os.path.join(FOUNDRY, "common", "packages"),
):
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".mjs"):
                continue
            p = os.path.join(dirpath, fn)
            pkg_hits += [(p, n, t) for n, t in find(p, r"(get|set)Flag", regex=True)]
check(
    "world-flag-noop | no getFlag/setFlag anywhere in client/common packages",
    not pkg_hits,
    "scanned client/packages + common/packages; hits=%d" % len(pkg_hits),
)
gw = find(game_mjs, "new foundry.packages.World(")
check(
    "world-flag-noop | game.world is a foundry.packages.World instance",
    bool(gw),
    "%s:%s %s" % (game_mjs, gw and gw[0][0], gw and gw[0][1]),
)

# --------------------------------------------------------------------------
# 3. preCreate<Type> hook fires AFTER the document was built from a deepClone
#    -> mutating the `data` argument cannot affect what gets created.
# --------------------------------------------------------------------------
cb = os.path.join(FOUNDRY, "client", "data", "client-backend.mjs")
ctor = find(cb, "new documentClass(foundry.utils.deepClone(createData)")
hook = find(cb, "preCreate${type}")
final = find(cb, "operation.data = documents")
ok = bool(ctor and hook and final and ctor[0][0] < hook[0][0] < final[0][0])
check(
    "precreate-noop | doc constructed before preCreate hook; documents win",
    ok,
    "%s: ctor@%s hook@%s operation.data=documents@%s"
    % (cb, ctor and ctor[0][0], hook and hook[0][0], final and final[0][0]),
)

# 3b. preUpdate<Type> DOES take effect: changes are re-cleaned after the hook
hook_u = find(cb, "preUpdate${type}")
reclean = find(cb, "clean: true,        // We need to clean again because data may have changed in preUpdate")
ok = bool(hook_u and reclean and hook_u[0][0] < reclean[0][0])
check(
    "preupdate-live | mutations to `changes` in preUpdate are re-cleaned and committed",
    ok,
    "%s: hook@%s re-clean@%s" % (cb, hook_u and hook_u[0][0], reclean and reclean[0][0]),
)

# 3c. dot-notation keys are expanded *inside* cleanData, before the hook
f_mjs = os.path.join(FOUNDRY, "common", "data", "fields.mjs")
exp = find(f_mjs, "SchemaField.expandObject(data, options, _state);")
check(
    "preupdate-live | SchemaField.clean expands dotted keys in place",
    bool(exp),
    "%s:%s" % (f_mjs, exp and [n for n, _ in exp]),
)

# --------------------------------------------------------------------------
# 4. StringField coerces a non-string via String(value) -> "[object Object]"
# --------------------------------------------------------------------------
cast = find(f_mjs, "return String(value);")
html = find(f_mjs, "class HTMLField extends StringField")
check(
    "desc-coercion | StringField._cast === String(value); HTMLField extends it",
    bool(cast and html),
    "%s: _cast@%s HTMLField@%s" % (f_mjs, cast and [n for n, _ in cast], html and html[0][0]),
)

# --------------------------------------------------------------------------
# 5. crucible item types: which ones have system.description as a bare string?
# --------------------------------------------------------------------------
models = os.path.join(DATA, "systems", "crucible", "module", "models")
string_desc, schema_desc = [], []
for fn in sorted(os.listdir(models)):
    if not fn.startswith("item-") or not fn.endswith(".mjs"):
        continue
    p = os.path.join(models, fn)
    if find(p, r"description:\s*new fields\.HTMLField", regex=True):
        string_desc.append(fn)
    if find(p, r"description:\s*new fields\.SchemaField", regex=True):
        schema_desc.append(fn)
check(
    "desc-coercion | crucible item types whose system.description is a bare HTMLField",
    bool(string_desc),
    "string=%s  |  {public,private}=%s" % (string_desc, schema_desc),
)

# --------------------------------------------------------------------------
# 6. Adventure.importContent exists, calls cls.updateDocuments, and its
#    toUpdate payload carries full source data (including items/effects).
# --------------------------------------------------------------------------
adv = os.path.join(FOUNDRY, "client", "documents", "adventure.mjs")
imp = find(adv, "async importContent({toCreate, toUpdate, documentCount}={})")
upd = find(adv, "await cls.updateDocuments(updateData, options)")
src = find(adv, "const adventureData = this.toObject();")
check(
    "import-strip | Adventure#importContent -> cls.updateDocuments(full source data)",
    bool(imp and upd and src),
    "%s: importContent@%s updateDocuments@%s toObject@%s"
    % (adv, imp and imp[0][0], upd and upd[0][0], src and src[0][0]),
)

# --------------------------------------------------------------------------
# 7. crucible action hooks are snapshotted per CrucibleAction at construction
# --------------------------------------------------------------------------
act = os.path.join(DATA, "systems", "crucible", "module", "models", "action.mjs")
snap = find(act, "crucible.api.hooks.action[actionId]")
frozen = find(act, "return Object.freeze(hooks);")
check(
    "setup-too-late | CrucibleAction snapshots crucible.api.hooks.action[id] at construction",
    bool(snap and frozen),
    "%s: lookup@%s freeze@%s" % (act, snap and snap[0][0], frozen and frozen[0][0]),
)

# --------------------------------------------------------------------------
print("=" * 100)
for name, ok, ev in results:
    print("%-6s %s\n       %s" % ("PASS" if ok else "FAIL", name, ev))
print("=" * 100)
print("%d/%d checks PASS" % (sum(1 for _n, o, _e in results if o), len(results)))
sys.exit(0)
