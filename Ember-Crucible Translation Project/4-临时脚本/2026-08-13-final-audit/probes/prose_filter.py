# -*- coding: utf-8 -*-
"""Keep only the unmapped field classes whose samples look like human prose."""
import re, os
HERE = os.path.dirname(os.path.abspath(__file__))
blocks, cur = [], None
for line in open(os.path.join(HERE, 'unmapped.txt'), encoding='utf-8'):
    m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\S+) :: (.*)$", line.rstrip("\n"))
    if m:
        cur = {"n": int(m.group(1)), "chars": int(m.group(2)), "bucket": m.group(3),
               "path": m.group(4), "packs": "", "s": []}
        blocks.append(cur)
    elif cur and line.strip().startswith("packs="):
        cur["packs"] = line.strip()[6:]
    elif cur and line.strip().startswith("·"):
        cur["s"].append(line.strip()[2:])

FILEISH = re.compile(r"\.(webp|png|ogg|mp3|jpg|jpeg|json|mjs|svg|m4a|wav)")
TWOWORD = re.compile(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}")
HTMLISH = re.compile(r"<[a-z]")


def prose(b):
    for s in b["s"]:
        t = s.strip('"')
        if len(t) < 4:
            continue
        if FILEISH.search(t):
            continue
        if t.startswith("Compendium.") or t.startswith("Actor.") or t.startswith("Item."):
            continue
        if HTMLISH.search(t):
            return True
        if TWOWORD.search(t):
            return True
    return False


ID16 = re.compile(r"(?<=\.)[A-Za-z0-9]{16}(?=\.|$)")
IN_SCOPE = ("ember.crucible-", "crucible.")   # dnd5e side is out of scope

import sys, collections
SCOPED = "--scoped" in sys.argv
agg = collections.OrderedDict()
for b in blocks:
    b["path"] = ID16.sub("*", b["path"])
    if SCOPED and not any(p.strip().startswith(IN_SCOPE) for p in b["packs"].split(",")):
        continue
    key = (b["bucket"], b["path"])
    a = agg.get(key)
    if not a:
        agg[key] = dict(b)
    else:
        a["n"] += b["n"]; a["chars"] += b["chars"]
        a["packs"] = ",".join(sorted(set(a["packs"].split(",")) | set(b["packs"].split(","))))
        a["s"] = (a["s"] + b["s"])[:3]
blocks = list(agg.values())

sel = [b for b in blocks if prose(b)]
print(f"# {len(sel)} prose-looking unmapped classes (of {len(blocks)} unmapped)")
for b in sorted(sel, key=lambda x: -x["chars"]):
    print(f'{b["n"]:6d} {b["chars"]:8d}  {b["bucket"]} :: {b["path"]}')
    print(f'          packs={b["packs"]}')
    for s in b["s"][:2]:
        print(f'          . {s}')
