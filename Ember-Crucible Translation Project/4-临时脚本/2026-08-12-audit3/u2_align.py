# -*- coding: utf-8 -*-
"""Positionally align @UUID occurrences of one target inside a leaf (EN vs CN).

The finding's `en_label` is the *first* EN label for that target in the leaf, which is
wrong whenever the author used more than one English label for the same target
(History/Age of Rediscovery uses both {Moiran} and {Blood Barons}). This prints the
ordered EN label list and ordered CN label list so the true English label at each
position is visible.
"""
import argparse, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SW = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/uuid_swap.json"
MARK = re.compile(r'(@[A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?')

def resolve(base, p):
    segs = p.split("."); cur = base; i = 0
    while i < len(segs):
        if isinstance(cur, list):
            cur = cur[int(segs[i])]; i += 1; continue
        for j in range(len(segs), i, -1):
            k = ".".join(segs[i:j])
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]; i = j; break
        else:
            return None
    return cur

def leaf(repo, pack, path, side):
    d = json.load(open(os.path.join(ROOT, repo, "compendium", side, pack), encoding="utf-8"))
    return resolve(d, path)

def labels_for(s, key):
    out = []
    for m in MARK.finditer(s or ""):
        tgt = m.group(2).split()[0] if m.group(2) else ""
        tgt = tgt.split("#")[0]
        if tgt.split(".")[-1] == key:
            out.append(m.group(3))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", required=True)
    a = ap.parse_args()
    F = json.load(open(SW, encoding="utf-8"))["findings"]
    for i in [int(x) for x in a.idx.split(",")]:
        x = F[i]
        en = leaf(x["repo"], x["pack"], x["path"], "en")
        cn = leaf(x["repo"], x["pack"], x["path"], "cn")
        el = labels_for(en, x["key"]); cl = labels_for(cn, x["key"])
        print(f"[{i}] {x['pack']} :: {x['path']}")
        print(f"    key={x['key']} finding_en={x['en_label']!r} finding_cn={x['cn_label']!r} maj={x['majority']['label']}({x['majority']['support']}/{x['majority']['total']})")
        print(f"    EN labels ({len(el)}): {el}")
        print(f"    CN labels ({len(cl)}): {cl}")
        same = len(el) == len(cl)
        if same:
            for a_, b_ in zip(el, cl):
                mark = "  <<<" if b_ == x["cn_label"] else ""
                print(f"       {a_!r:40} -> {b_!r}{mark}")
        else:
            print("       *** COUNT MISMATCH ***")
main()
