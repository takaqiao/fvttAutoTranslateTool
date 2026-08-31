"""Drop from the H2 batches every leaf whose value already equals the current
compendium/cn value.

Why this matters and is not cosmetic: `merge_batches.py --scan` treats a path as
CLAIMED by whatever batch lists it. A no-op leaf claims a path without changing
anything, so it can out-vote another unit's genuine fix at merge time. The
earlier H2 pass in this session computed its `old` values against a stale
snapshot, so 81 of its 87 leaves are no-ops against today's cn.

Also drops two Heater Shield description rewrites: they are pure re-wording with
no English evidence, and the two proposed values disagree with each other for
the SAME English source (熨斗底板 vs 熨斗底座), which would make the library less
consistent than it is now.
"""
import json
import os

B = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches"
CN = {
    "crucible": r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium\cn",
    "ember": r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium\cn",
}
DROP = {("crucible.equipment.json", "Heater Shield.description.public"),
        ("crucible.playtest.json",
         "Playtest 1 - The Ring of Valor.actors.Duurath.items.Heater Shield.description.public")}


def get_at(node, path):
    rest = path
    while rest:
        if isinstance(node, dict):
            c = [k for k in node if rest == k or rest.startswith(k + ".")]
            if c:
                k = max(c, key=len)
                node = node[k]
                rest = rest[len(k) + 1:]
                continue
        return None
    return node


for f in sorted(os.listdir(B)):
    if not f.startswith("H2__"):
        continue
    repo = f[len("H2__"):].split("__")[0]
    pack = f.split("__")[-1]
    data = json.load(open(os.path.join(CN[repo], pack), encoding="utf-8-sig"))["entries"]
    b = json.load(open(os.path.join(B, f), encoding="utf-8-sig"))
    kept = {k: v for k, v in b.items()
            if get_at(data, k) != v and (pack, k) not in DROP}
    path = os.path.join(B, f)
    if not kept:
        os.remove(path)
        print(f"removed {f} (all {len(b)} leaves were no-ops)")
        continue
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False, indent=1)
    print(f"{f}: {len(b)} -> {len(kept)}")
