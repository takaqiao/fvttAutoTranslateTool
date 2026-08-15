# -*- coding: utf-8 -*-
"""G3 second pass: build the batch files for the confirmed renamed-name residue.

Every edit is a pin-point replacement inside one leaf; nothing else in the leaf
is touched, so the markup gate cannot be tripped by construction.  The script
refuses to emit an edit whose `old` string is not present exactly once, which is
what keeps a silently-drifted base from producing a wrong batch.
"""
import json, os, sys, collections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件"
OUT = ("C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/"
       "e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/batches")

A = "ember.adventure.json"
C = "ember.crucible-adventure.json"

# (pack, leaf path without the `entries.` prefix, old, new, note)
EDITS = [
    # --- 1. @UUID label still carries the pre-rename scene name -------------
    (A, "Ember Early Access.journals.Gamemaster's Guide.pages.Patch 0.2.0.text",
     "@UUID[Scene.emberArcturelTra]{阿克图瑞尔上层 - 贸易道}",
     "@UUID[Scene.emberArcturelTra]{阿克图瑞尔贸易道}",
     "EN label is {Arcturel Tradeway}; the twin pack already says 阿克图瑞尔贸易道"),

    # --- 2. 「上层」inserted where the English has no Upper ------------------
    (A, "Ember Early Access.journals.Arcturel Tradeway.pages.Gameplay Details.text",
     "<p>在阿克图瑞尔上层，贸易道与底腹区各层之间的距离大多数地方为20英尺",
     "<p>阿克图瑞尔的贸易道层与底腹区层之间的距离大多数地方为20英尺",
     "EN: The distance between the Tradeway and Underbelly levels of Arcturel"),
    (A, "Ember Early Access.journals.Arcturel Tradeway.pages.Gameplay Details.text",
     "<p>从阿克图瑞尔上层跳下或坠落的角色",
     "<p>从阿克图瑞尔跳下或坠落的角色",
     "EN: Characters that jump or fall off of Arcturel"),
    (C, "Ember Early Access.journals.Arcturel Tradeway.pages.Gameplay Details.text",
     "<p>在阿克图瑞尔上层，贸易道与底腹区各层之间的距离大多数地方为20英尺",
     "<p>阿克图瑞尔的贸易道层与底腹区层之间的距离大多数地方为20英尺", ""),
    (C, "Ember Early Access.journals.Arcturel Tradeway.pages.Gameplay Details.text",
     "<p>从阿克图瑞尔上层跳下或坠落的角色",
     "<p>从阿克图瑞尔跳下或坠落的角色", ""),

    # --- 3. 「阿克图瑞尔下层」where the English names the renamed area -------
    (A, "Ember Early Access.journals.Glitter in the Dark.pages.A Peculiar Encampment.text",
     "队伍就可以返回阿克图瑞尔下层，触发",
     "队伍就可以返回阿克图瑞尔矿渊，触发",
     "EN: return to the Arcturel Dives; twin pack already says 阿克图瑞尔矿渊"),
    (A, "Ember Early Access.journals.Glitter in the Dark.pages.Checkmate for Chessmen.text",
     "队伍可以返回阿克图瑞尔上层的银光束总部",
     "队伍可以返回银光束总部",
     "EN: return to the Silver Beam Headquarters; twin pack already matches"),
    (A, "Ember Early Access.journals.Glitter in the Dark.pages.Overview.text",
     "的矿工在阿克图瑞尔下层@UUID[JournalEntry.emberArctusGazet.JournalEntryPage.Byemag1k6p66PynZ]{矿渊}",
     "的矿工在@UUID[JournalEntry.emberArctusGazet.JournalEntryPage.Byemag1k6p66PynZ]{矿渊}",
     "EN: during routine operations in @UUID[...]{The Dives} — no locator"),
    (C, "Ember Early Access.journals.Glitter in the Dark.pages.Overview.text",
     "的矿工在阿克图瑞尔下层@UUID[JournalEntry.emberArctusGazet.JournalEntryPage.Byemag1k6p66PynZ]{矿渊}",
     "的矿工在@UUID[JournalEntry.emberArctusGazet.JournalEntryPage.Byemag1k6p66PynZ]{矿渊}", ""),

    # --- 4. transliteration drift 阿克图雷尔 (4) vs 阿克图瑞尔 (290) ---------
    (A, "Ember Early Access.journals.An Old Friend.pages.Traveling with Lyla.text",
     "阿克图雷尔调查", "阿克图瑞尔调查",
     "same leaf uses 阿克图瑞尔 in the very next sentence"),
    (C, "Ember Early Access.journals.An Old Friend.pages.Traveling with Lyla.text",
     "阿克图雷尔调查", "阿克图瑞尔调查", ""),
    (C, "Ember Early Access.actors.The Device.items.Powered Effect.description",
     "下层阿克图雷尔矿井效果", "下层阿克图瑞尔矿井效果", ""),
    (C, "Ember Early Access.actors.The Device.items.Powered Effect.actions.poweredEffect.description",
     "下层阿克图雷尔矿井效果", "下层阿克图瑞尔矿井效果", ""),
]


def get_at(root, path):
    """resolve a dotted path, preferring the longest key that matches."""
    node = root
    rest = path
    while rest:
        if not isinstance(node, dict):
            return None
        best = None
        for k in node:
            if rest == k or rest.startswith(k + "."):
                if best is None or len(k) > len(best):
                    best = k
        if best is None:
            return None
        node = node[best]
        rest = rest[len(best) + 1:]
    return node


def main():
    os.makedirs(OUT, exist_ok=True)
    packs = {}
    for pack in {e[0] for e in EDITS}:
        packs[pack] = json.load(open(f"{REPO}/compendium/cn/{pack}",
                                    encoding="utf-8-sig"))["entries"]
    cur = {}
    bad = 0
    for pack, path, old, new, note in EDITS:
        val = cur.get((pack, path))
        if val is None:
            val = get_at(packs[pack], path)
            if not isinstance(val, str):
                print(f"!! unresolved {pack} :: {path}")
                bad += 1
                continue
        n = val.count(old)
        if n != 1:
            print(f"!! old-string count {n} (expected 1) {pack} :: {path}\n   {old!r}")
            bad += 1
            continue
        cur[(pack, path)] = val.replace(old, new)
    if bad:
        print(f"ABORT: {bad} bad edits")
        return 1
    out = collections.defaultdict(dict)
    for (pack, path), val in cur.items():
        out[pack][path] = val
    for pack, d in out.items():
        fn = f"{OUT}/G3__ember__{pack[:-5]}.json"
        json.dump(d, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"-> {fn}   leaves={len(d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
