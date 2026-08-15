# -*- coding: utf-8 -*-
"""Diff `leaf_paths.txt` (everything in the LevelDB packs) against the field
paths `mappings.mjs` actually reads, and keep only the leftovers whose sample
values look like human-readable text.  Read-only.
"""
import re, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "leaf_paths.txt")

# ---- doc-relative paths the extractor consumes (mappings.mjs, effectiveMappings)
COVERED = [
    # names on every document type Babele walks
    r"^name$",
    r"(?:^|\.)(?:items|actors|effects|pages|results|regions|behaviors|categories|"
    r"folders|levels|tokens|sounds|journal|scenes|macros|playlists|tables|drawings|notes)"
    r"\[\](?:\.[a-zA-Z]+\[\])*\.name$",
    # description on doc types that have it
    r"(?:^|\.)description$",
    r"(?:^|\.)system\.description(?:\.(?:public|private))?$",
    r"(?:^|\.)system\.adjective$",
    # crucible actions
    r"(?:^|\.)system\.actions\[\]\.(?:name|description|condition)$",
    r"(?:^|\.)system\.actions\[\]\.effects\[\]\.name$",
    # actor details
    r"(?:^|\.)system\.details\.(?:biography|ancestry|background|archetype|taxonomy)"
    r"\.(?:name|description|public|private|appearance)$",
    r"(?:^|\.)prototypeToken\.name$",
    # journal pages
    r"(?:^|\.)text\.content$",
    r"(?:^|\.)image\.caption$",
    r"(?:^|\.)system\.(?:overview|exposition|summary|pronunciation|subtitle|height|lifespan|origin)$",
    r"(?:^|\.)system\.outcomes\[\]\.(?:label|summary)$",
    r"(?:^|\.)system\.content\.(?:overview|gamemaster)$",
    r"(?:^|\.)system\.banner\.caption$",
    # scene
    r"(?:^|\.)navName$",
    r"(?:^|\.)(?:notes|drawings)\[\]\.text$",
    # adventure
    r"^caption$",
]
COV = [re.compile(p) for p in COVERED]

# ---- shapes that are never prose
NOISE = re.compile(
    r"_id$|_stats|\bid$|\.id$|\bimg$|\.img$|\.src$|\.path$|\.thumb$|texture|"
    r"\.color|tint|\bfolder$|\bsort$|\btype$|\.type$|Version$|systemId|"
    r"compendiumSource|\borigin$|\.key$|\.mode$|\.uuid$|\.actor$|\.value$|"
    r"formula|\.units$|\.attribute$|flags\.|\.command$|\.source$|\.script$|"
    r"htmlFields|gmOnlyFields|filePathFields|\.statuses\[\]$|\.tags\[\]$|"
    r"\.properties\[\]$|\.nodes\[\]$|\.skills\[\]$|\.knowledge\[\]$|\.hexes\[\]$|"
    r"\.locations?\[?\]?$|\.creatures\[\]$|\.levels\[\]$|\.identifier$|"
    r"\.item$|\.messageId$|\.pageId$|\.entryId$|\.author$|fontFamily|"
    r"\.category$|\.eventId$|\.coefficients$|highlightMode|\.placement$|"
    r"\.material$|\.destinations\[\]$|\.events\[\]$|\.behaviors\[\]$|"
    r"\.movementActions\[\]$|\.ability$|\.expiry$|\.phase$|\.affixType$|"
    r"\.itemTypes\[\]$|\.enchantment$|\.quality$|\.grants\[\]$|\.pool\[\]$|"
    r"\.weather$|\.style$|\.initialLevel$|\.visionMode$|\.actorId$|\.level$|"
    r"\.rank$|\.status$|\.damageType$|\.onSave$|\.calculation$|\.affects\.|"
    r"\.activation\.type$|\.duration\.|\.range\.|\.target\.|\.save\.|\.book$|"
    r"\.rules$|\.favorites\[\]$|sheetClass|\.journal$|\.textColor$|\.build$|"
    r"\.stance$|partId|templateId|\.repeat$|\.easing$|\.walls$|\.darkness$")


def covered(p):
    return any(rx.search(p) for rx in COV)


def main():
    blocks, cur = [], None
    for line in open(SRC, encoding="utf-8"):
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\S+) :: (.*)$", line.rstrip("\n"))
        if m:
            cur = {"n": int(m.group(1)), "chars": int(m.group(2)),
                   "bucket": m.group(3), "path": m.group(4), "packs": "", "s": []}
            blocks.append(cur)
        elif cur is not None and line.strip().startswith("packs="):
            cur["packs"] = line.strip()[6:]
        elif cur is not None and line.strip().startswith("·"):
            cur["s"].append(line.strip()[2:])
    print(f"# {len(blocks)} distinct bucket::path combos in the packs")
    left = [b for b in blocks
            if not covered(b["path"]) and not NOISE.search(b["path"])]
    print(f"# {len(left)} left after removing mapped paths and non-prose shapes\n")
    for b in sorted(left, key=lambda x: -x["chars"]):
        print(f'{b["n"]:6d} {b["chars"]:8d}  {b["bucket"]} :: {b["path"]}')
        print(f'          packs={b["packs"]}')
        for s in b["s"][:2]:
            print(f"          · {s}")


main()
