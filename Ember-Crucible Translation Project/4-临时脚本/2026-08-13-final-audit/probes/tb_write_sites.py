# -*- coding: utf-8 -*-
"""
tb_write_sites.py  --  Step 2/3 of the "type-blind / scope-blind write" criterion.

CRITERION (abstracted from the confirmed instance at register.js:294-296):
  A defect of this class is any RUNTIME site in the two translation modules that
  MUTATES data belonging to somebody else (a document, an update payload, a
  shared CONFIG object) where BOTH hold:

    (A) the only guard is a JS SHAPE test -- `typeof x === 'string'`,
        `Array.isArray(x)`, `!x`, `typeof x === 'object'` -- or no guard at all;
        the enclosing function never consults a DISCRIMINATOR
        (`.type`, `documentName`, `documentType`, pack id, subtype key); and

    (B) the target path is POLYMORPHIC upstream -- the same leaf is declared
        with >=2 different DataField classes across document subtypes
        (see tb_schema_map.py) -- OR the mutation removes/replaces a whole
        sub-tree of somebody else's payload (scope overreach).

  Reported per site with: the guard, the target path, the polymorphism verdict,
  and whether the site is reachable from a Foundry write path.

FALSE POSITIVES this probe knowingly produces (project rule 4):
  * It is regex + brace-matching, not a JS parser. A guard implemented in a
    HELPER called by the site (rather than inline) is not seen -> false
    positive. Every hit must be read by hand.
  * "polymorphic" is decided by LEAF NAME across all three upstream trees, so
    unrelated same-named fields inflate it -> false positive.
  * Pure in-memory display mutations (DOM text nodes, CONFIG labels) are
    reported too, because scope overreach applies to them as well; severity
    must then be judged by whether the mutated object is ever persisted.
  * It cannot see writes performed by Babele on the module's behalf; the
    DOCUMENT_MAPPINGS paths are listed separately for manual pairing.

Read-only. Writes only tb_write_sites.json next to itself.
"""
import json
import os
import re

BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
TARGETS = [
    os.path.join(BASE, "1-Ember汉化插件", "register.js"),
    os.path.join(BASE, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
    os.path.join(BASE, "1-Ember汉化插件", "babele-mappings.js"),
    os.path.join(BASE, "2-Crucible汉化插件", "babele-register.js"),
    os.path.join(BASE, "2-Crucible汉化插件", "babele-mappings.js"),
]

# things that write somebody else's data
WRITE_PATTERNS = [
    ("doc.update", re.compile(r"\.update\s*\(")),
    ("doc.updateEmbeddedDocuments", re.compile(r"\.updateEmbeddedDocuments\s*\(")),
    ("doc.createEmbeddedDocuments", re.compile(r"\.createEmbeddedDocuments\s*\(")),
    ("doc.setFlag", re.compile(r"\.setFlag\s*\(")),
    ("setProperty", re.compile(r"\bsetProperty\s*\(")),
    ("mergeObject(inplace-default)", re.compile(r"\bmergeObject\s*\((?![^()]*inplace:\s*false)")),
    ("delete-key", re.compile(r"\bdelete\s+[A-Za-z_$][\w$]*\.")),
    ("assign-to-param-prop", re.compile(r"^\s*(?:if\s*\(.*\)\s*)?[a-z][\w$]*(?:\.[\w$]+)+\s*=(?!=)")),
    ("assign-index", re.compile(r"^\s*[a-z][\w$]*\[[^\]]+\]\s*=(?!=)")),
    ("class-monkeypatch", re.compile(r"^\s*[A-Za-z_$][\w$.?]*\.(updateDocuments|createDocuments|prepare|enricher)\s*=(?!=)")),
    ("i18n-global", re.compile(r"game\.i18n\.translations\.\w+\s*=(?!=)")),
]

SHAPE_GUARD = re.compile(
    r"typeof\s+[\w$.\[\]']+\s*[=!]==\s*['\"](?:string|object|number|function)['\"]"
    r"|Array\.isArray\s*\("
)
DISCRIMINATOR = re.compile(
    r"\.type\b|documentName|documentType|\.documentClass\b|metadata\.id|pack(?:Id|Name)\b"
    r"|SYSTEM\.ITEM|CONFIG\.Item\.typeLabels|\btypes\b"
)

PATH_LIT = re.compile(r"['\"]((?:system|prototypeToken|text|image|effects|items)(?:\.[\w$]+)*)['\"]")


def enclosing_function(lines, idx):
    """Walk back to the nearest `function name(` / `const name = (`/`=>` header."""
    for j in range(idx, -1, -1):
        m = re.match(r"\s*(?:export\s+)?(?:async\s+)?function\s+([\w$]+)", lines[j])
        if m:
            return m.group(1), j + 1
        m = re.match(r"\s*(?:const|let)\s+([\w$]+)\s*=\s*(?:async\s*)?\(", lines[j])
        if m:
            return m.group(1), j + 1
        m = re.match(r"\s*Hooks\.(?:on|once)\s*\(\s*['\"]([\w.]+)['\"]", lines[j])
        if m:
            return "Hooks:" + m.group(1), j + 1
    return "<top-level>", 1


def function_body(lines, start_line):
    """Crude: from the header line, take until the brace depth returns to 0."""
    depth = 0
    out = []
    started = False
    for k in range(start_line - 1, len(lines)):
        out.append(lines[k])
        depth += lines[k].count("{") - lines[k].count("}")
        if "{" in lines[k]:
            started = True
        if started and depth <= 0:
            break
    return "\n".join(out)


def main():
    poly_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_schema_map.json")
    poly = set(json.load(open(poly_path, encoding="utf-8"))["polymorphic"]) if os.path.exists(poly_path) else set()

    rows = []
    for path in TARGETS:
        lines = open(path, encoding="utf-8").read().splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(("*", "//", "/*")):
                continue
            for label, pat in WRITE_PATTERNS:
                if not pat.search(line):
                    continue
                fname, fstart = enclosing_function(lines, i)
                body = function_body(lines, fstart)
                paths = sorted(set(PATH_LIT.findall(body)))
                leaves = sorted({p.split(".")[-1] for p in paths})
                rows.append({
                    "file": os.path.relpath(path, BASE).replace("\\", "/"),
                    "line": i + 1,
                    "kind": label,
                    "code": line.strip()[:150],
                    "fn": fname,
                    "shape_guard": bool(SHAPE_GUARD.search(body)),
                    "discriminator": bool(DISCRIMINATOR.search(body)),
                    "paths": paths,
                    "polymorphic_leaves": [l for l in leaves if l in poly],
                })
                break

    # (A) and (B): shape-guarded-or-unguarded + no discriminator, and either a
    # polymorphic target leaf or a whole-subtree removal.
    flagged = [
        r for r in rows
        if not r["discriminator"]
        and (r["polymorphic_leaves"] or r["kind"] in ("delete-key", "class-monkeypatch"))
    ]

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "tb_write_sites.json"), "w", encoding="utf-8") as fh:
        json.dump({"all": rows, "flagged": flagged}, fh, ensure_ascii=False, indent=1)

    print("write sites found: %d   flagged by criterion: %d" % (len(rows), len(flagged)))
    print("-" * 100)
    for r in flagged:
        print("%-42s :%-4d %-28s fn=%-32s poly=%s"
              % (r["file"].split("/")[-1], r["line"], r["kind"], r["fn"],
                 ",".join(r["polymorphic_leaves"]) or "-"))
        print("      %s" % r["code"])
    print("-" * 100)
    print("NOT flagged (has a discriminator, or target is monomorphic):")
    for r in rows:
        if r not in flagged:
            print("  %-42s :%-4d %-26s fn=%s" % (r["file"].split("/")[-1], r["line"], r["kind"], r["fn"]))


if __name__ == "__main__":
    main()
