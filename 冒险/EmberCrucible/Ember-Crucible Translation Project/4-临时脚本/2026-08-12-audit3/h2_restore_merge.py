"""Rebuild the batches an EARLIER H2 run produced (recovered from
findings/_h2b_edits.json) and merge them with this run's batches.

Why: this session already contained an H2 pass (23:30-23:56) whose batch files
for 7 packs were overwritten when the second H2 pass wrote the same filenames.
Its edit list survived in `_h2b_edits.json`, so the lost 47 leaves are
recoverable exactly.
"""
import json
import os
from collections import defaultdict

SCRATCH = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"
EDITS = os.path.join(SCRATCH, "findings", "_h2b_edits.json")
BATCH = os.path.join(SCRATCH, "batches")
CN = {
    "crucible": r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium\cn",
    "ember": r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium\cn",
}

prev = defaultdict(dict)          # (repo, pack) -> {path: new}
for group in json.load(open(EDITS, encoding="utf-8")).values():
    for e in group:
        prev[(e["repo"], e["pack"])][e["batch_path"]] = e["new"]

conflicts = []
for (repo, pack), edits in sorted(prev.items()):
    fn = os.path.join(BATCH, f"H2__{repo}__{pack[:-5]}.json")
    mine = {}
    if os.path.exists(fn):
        mine = json.load(open(fn, encoding="utf-8-sig"))
    merged = dict(edits)
    for k, v in mine.items():
        if k in merged and merged[k] != v:
            conflicts.append((repo, pack, k, merged[k], v))
        merged[k] = v          # this run's value wins on a genuine collision
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f"{os.path.basename(fn)}: prev {len(edits)} + mine {len(mine)} -> {len(merged)}")

print("\nconflicting paths:", len(conflicts))
for c in conflicts:
    print("  ", c)
