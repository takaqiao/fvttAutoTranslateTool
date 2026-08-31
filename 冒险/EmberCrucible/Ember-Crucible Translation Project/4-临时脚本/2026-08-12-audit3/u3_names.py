# -*- coding: utf-8 -*-
"""U3: resolve each finding's @UUID target to the CN `name` of that document,
read LIVE from compendium/cn (the cached reports/name_index.json predates the
phase-29 unifications and is stale).

Target ids -> EN name via the LevelDB dump (reports/ember_ids.json), then every
place in compendium/en where a document carries that name gets paired with the
CN value at the same structural path.  Printing every distinct CN spelling with
its path makes name-field disagreement visible instead of averaged away.
"""
import json, os, re, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
R = os.path.join(P, r"4-临时脚本\2026-08-12-fix\reports")
SC = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]
SKIP = {"_id", "path", "_variants", "_when"}


def walk(en, cn, path, out):
    if isinstance(en, dict):
        if isinstance(en.get("name"), str):
            out.append((".".join(path), en["name"],
                        cn.get("name") if isinstance(cn, dict) else None))
        for k, v in en.items():
            if k in SKIP:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            walk(v, cn[i] if isinstance(cn, list) and i < len(cn) else None,
                 path + [str(i)], out)


rows = []
for repo in REPOS:
    end = os.path.join(P, repo, "compendium", "en")
    for fn in sorted(os.listdir(end)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        en = json.load(open(os.path.join(end, fn), encoding="utf-8"))
        cp = os.path.join(P, repo, "compendium", "cn", fn)
        cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
        acc = []
        walk(en, cn, [], acc)
        for p, e, c in acc:
            rows.append((repo, fn, p, e, c))

by_en = defaultdict(list)
for repo, fn, p, e, c in rows:
    by_en[e].append((repo, fn, p, c))

json.dump({e: [{"repo": r, "pack": f, "path": p, "cn": c} for r, f, p, c in v]
           for e, v in by_en.items()},
          open(SC + "/u3_name_live.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("EN document names indexed:", len(by_en), " name nodes:", len(rows))

if len(sys.argv) > 1:
    for q in sys.argv[1:]:
        print(f"\n### {q!r}")
        v = by_en.get(q)
        if not v:
            print("   (no EN document with this name)")
            continue
        c = Counter(x[3] for x in v)
        for cnv, n in c.most_common():
            print(f"   {n:>3}  {cnv!r}")
        for r, f, p, cnv in v[:8]:
            print(f"        {f} :: {p} -> {cnv!r}")
