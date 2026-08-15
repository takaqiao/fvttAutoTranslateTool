# -*- coding: utf-8 -*-
"""
tb_schema_map.py  --  Step 1 of the "type-blind write" criterion.

Build leafKey -> {DataField class -> [sites]} from the UPSTREAM schemas
(crucible 0.10.1 + ember 0.6.0 + Foundry v14 common), so we can tell which
document paths are POLYMORPHIC (same leaf declared with >=2 different field
classes across document subtypes).

Read-only. Writes nothing outside this probes/ directory.

False-positive modes (stated up front, per project rule 4):
  * Regex, not a JS parser: it keys on the literal `<ident>: new fields.XField`
    / `new foundry.data.fields.XField`, so a schema built by a helper function
    or spread is invisible -> UNDER-reports polymorphism.
  * It groups by LEAF NAME only, so two unrelated `name:` fields in different
    models look "polymorphic" if their field classes differ. Callers must
    confirm the two declarations really are the same document path.
  * Both the compiled bundle and the loose module sources of crucible are
    scanned, so each field is usually seen twice; sites are de-duplicated by
    (class, file, line) but the duplicate FILES are kept on purpose as
    corroboration.
"""
import json
import os
import re
import sys

ROOTS = [
    (r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible", "crucible"),
    (r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember", "ember"),
    (r"C:\Program Files\Foundry Virtual Tabletop\resources\app\common", "foundry"),
]

DECL = re.compile(
    r"(?P<key>[A-Za-z_$][\w$]*)\s*:\s*new\s+(?:foundry\.data\.)?fields\.(?P<cls>\w+Field)\b"
)


def walk(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith((".mjs", ".js")):
                yield os.path.join(dirpath, fn)


def main():
    table = {}
    nfiles = 0
    nlines = 0
    for root, tag in ROOTS:
        if not os.path.isdir(root):
            print("MISSING ROOT: %s" % root, file=sys.stderr)
            continue
        for path in walk(root):
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            nfiles += 1
            lines = text.splitlines()
            nlines += len(lines)
            for i, line in enumerate(lines, 1):
                for m in DECL.finditer(line):
                    key, cls = m.group("key"), m.group("cls")
                    table.setdefault(key, {}).setdefault(cls, []).append(
                        "%s:%s:%d" % (tag, os.path.basename(path), i)
                    )

    poly = {k: v for k, v in table.items() if len(v) >= 2}
    out = {
        "scanned_files": nfiles,
        "scanned_lines": nlines,
        "distinct_leaf_keys": len(table),
        "polymorphic_leaf_keys": len(poly),
        "polymorphic": {
            k: {cls: sites[:6] for cls, sites in sorted(v.items())}
            for k, v in sorted(poly.items())
        },
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "tb_schema_map.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("files=%d lines=%d leaves=%d polymorphic=%d"
          % (nfiles, nlines, len(table), len(poly)))
    for k in sorted(poly):
        print("  %-24s %s" % (k, ", ".join(sorted(poly[k]))))


if __name__ == "__main__":
    main()
