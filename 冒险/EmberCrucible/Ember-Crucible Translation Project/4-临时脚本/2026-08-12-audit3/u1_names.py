# -*- coding: utf-8 -*-
"""U1: resolve every @UUID target in a finding range to its LIVE cn `name`.

table.txt in the scratchpad was built from an older base and is stale for at
least one family (Rune-Marked *). The name field is the strongest evidence in
the decision ladder, so it has to be read from the file apply_translations will
actually write into.
"""
import json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
IDS = os.path.join(P, "4-临时脚本", "2026-08-12-fix", "reports", "ember_ids.json")
SWAP = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/uuid_swap.json"
PACKS = ["ember.adventure.json", "ember.crucible-adventure.json"]

ids = json.load(open(IDS, encoding="utf-8"))
cns = {}
ens = {}
for pk in PACKS:
    cns[pk] = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "cn", pk), encoding="utf-8"))["entries"]
    ens[pk] = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "en", pk), encoding="utf-8"))["entries"]


def walk_names(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "name" and isinstance(v, str):
                out.append((".".join(path), v))
            elif isinstance(v, (dict, list)):
                walk_names(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, (dict, list)):
                walk_names(v, path + [str(i)], out)


NAMEIDX = {}
for pk in PACKS:
    out = []
    walk_names(cns[pk], [], out)
    for p, v in out:
        NAMEIDX.setdefault(pk, {}).setdefault(p, v)


def cn_name_for(target):
    """Return (en_name, Counter of live CN names, sample paths)."""
    parts = target.split(".")
    got = Counter()
    where = []
    if parts[0] == "Compendium":
        return None, got, where
    en_name = None
    if parts[0] == "JournalEntry" and len(parts) == 4:
        jid, pid = parts[1], parts[3]
        jname = ids.get(jid, {}).get("name")
        pname = ids.get(pid, {}).get("name")
        en_name = pname
        for pk in PACKS:
            for root in cns[pk]:
                try:
                    v = cns[pk][root]["journals"][jname]["pages"][pname]["name"]
                except Exception:
                    continue
                got[v] += 1
                where.append(f"{pk}:{root}.journals.{jname}.pages.{pname}.name")
    elif parts[0] == "JournalEntry" and len(parts) == 2:
        jname = ids.get(parts[1], {}).get("name")
        en_name = jname
        for pk in PACKS:
            for root in cns[pk]:
                try:
                    v = cns[pk][root]["journals"][jname]["name"]
                except Exception:
                    continue
                got[v] += 1
                where.append(f"{pk}:{root}.journals.{jname}.name")
    else:
        kind = {"Actor": "actors", "Item": "items", "RollTable": "tables"}.get(parts[0])
        nm = ids.get(parts[1], {}).get("name")
        en_name = nm
        if kind:
            for pk in PACKS:
                for root in cns[pk]:
                    try:
                        v = cns[pk][root][kind][nm]["name"]
                    except Exception:
                        continue
                    got[v] += 1
                    where.append(f"{pk}:{root}.{kind}.{nm}.name")
        if not got and nm:
            # embedded item / effect: scan every `name` path whose last key matches
            for pk in PACKS:
                for p, v in NAMEIDX[pk].items():
                    if p.endswith("." + nm):
                        got[v] += 1
                        where.append(f"{pk}:{p}.name")
    return en_name, got, where


def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    findings = json.load(open(SWAP, encoding="utf-8"))["findings"]
    seen = {}
    for i in range(lo, hi):
        t = findings[i]["target"]
        seen.setdefault(t, []).append(i)
    for t, idxs in seen.items():
        en_name, got, where = cn_name_for(t)
        print(f"{t}  findings={idxs}")
        print(f"    EN name : {en_name!r}   ({ids.get(t.split('.')[-1],{}).get('type')})")
        print(f"    CN name : {got.most_common()}")
        for w in where[:3]:
            print(f"            @ {w}")


if __name__ == "__main__":
    main()
