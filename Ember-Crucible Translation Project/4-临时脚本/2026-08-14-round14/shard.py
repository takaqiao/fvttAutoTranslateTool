"""Split 2026-08-14-fix/REMAINING.json into deterministic shards for re-verification.

Findings are grouped first by the primary file they point at (so one agent owns one
target file and can judge "already fixed?" from one reading), then packed round-robin
into N shards of roughly equal evidence size.
"""
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 4-临时脚本
SRC = os.path.join(ROOT, "2026-08-14-fix", "REMAINING.json")
OUT = os.path.join(ROOT, "2026-08-14-round14", "shards")

TARGETS = [
    ("ember-hardcoded-cn.mjs", "hc-mjs"),
    ("register.js", "ember-register"),
    ("babele-register.js", "cru-register"),
    ("babele-mappings.js", "runtime-map"),
    ("mappings.mjs", "mappings"),
    ("runtime-converters.js", "runtime-conv"),
    ("module.json", "manifest"),
    ("README", "manifest"),
    ("release-body-template", "manifest"),
    (".css", "css"),
    ("lang/cn.json", "lang"),
    ("lang\\cn.json", "lang"),
    ("compendium", "compendium"),
    ("3-常用脚本", "tooling"),
    ("4-临时脚本", "tooling"),
    ("PROJECT.md", "docs"),
]


def bucket(f):
    blob = " ".join(str(f.get(k, "")) for k in ("where", "sig", "title"))
    for needle, name in TARGETS:
        if needle in blob:
            return name
    return "other"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    findings = json.load(open(SRC, encoding="utf-8"))
    for i, f in enumerate(findings):
        f["idx"] = i
        f["bucket"] = bucket(f)

    by = defaultdict(list)
    for f in findings:
        by[f["bucket"]].append(f)

    # Largest buckets first so they spread across shards evenly.
    ordered = []
    for name in sorted(by, key=lambda k: -len(by[k])):
        ordered.extend(sorted(by[name], key=lambda f: f["idx"]))

    shards = [[] for _ in range(n)]
    sizes = [0] * n
    for f in ordered:
        j = sizes.index(min(sizes))
        shards[j].append(f)
        sizes[j] += len(json.dumps(f, ensure_ascii=False))

    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for j, sh in enumerate(shards, 1):
        p = os.path.join(OUT, f"shard{j:02d}.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(sh, fh, ensure_ascii=False, indent=1)
        manifest.append({
            "shard": f"shard{j:02d}",
            "path": p,
            "count": len(sh),
            "bytes": sizes[j - 1],
            "buckets": sorted({f["bucket"] for f in sh}),
        })
        print(f"shard{j:02d}  n={len(sh):3d}  {sizes[j-1]/1024:6.1f} KB  {sorted({f['bucket'] for f in sh})}")

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print("total", sum(len(s) for s in shards))


if __name__ == "__main__":
    main()
