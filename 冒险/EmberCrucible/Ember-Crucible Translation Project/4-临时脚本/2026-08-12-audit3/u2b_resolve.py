# -*- coding: utf-8 -*-
"""U2 (findings 76..150) evidence table.

For each finding print, in decision-ladder order:
  1. target id -> EN document name (from the LevelDB id dump: hard fact)
  2. that document's CN `name` field in compendium/cn (strongest basis)
  3. the EN label actually used at this position (positional align, not the
     finding's first-label heuristic)
  4. the majority CN label across the library
"""
import argparse, json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SCR = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"
IDS = json.load(open(ROOT + r"/4-临时脚本/2026-08-12-fix/reports/ember_ids.json", encoding="utf-8"))
NAMELIVE = json.load(open(SCR + "/u3_name_live.json", encoding="utf-8"))
SW = json.load(open(SCR + "/uuid_swap.json", encoding="utf-8"))["findings"]
MARK = re.compile(r'(@[A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?')

_cache = {}


def load(repo, pack, side):
    k = (repo, pack, side)
    if k not in _cache:
        p = os.path.join(ROOT, repo, "compendium", side, pack)
        _cache[k] = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
    return _cache[k]


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


def labels_for(s, key):
    out = []
    for m in MARK.finditer(s or ""):
        tgt = (m.group(2) or "").split()[0].split("#")[0]
        if tgt.split(".")[-1] == key:
            out.append(m.group(3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", required=True, help="a,b or a-b")
    ap.add_argument("--ctx", type=int, default=0, help="chars of context around the hit")
    a = ap.parse_args()
    if "-" in a.idx:
        lo, hi = a.idx.split("-"); idxs = list(range(int(lo), int(hi)))
    else:
        idxs = [int(x) for x in a.idx.split(",")]
    for i in idxs:
        x = SW[i]
        key = x["key"]
        info = IDS.get(key)
        enname = info["name"] if info else None
        print("=" * 100)
        print(f"[{i}] {x['repo'][:1]} {x['pack']} :: {x['path']}")
        print(f"     target={x['target']}")
        print(f"     EN doc name = {enname!r}  type={info.get('type') if info else None}")
        if enname and enname in NAMELIVE:
            seen = set()
            for h in NAMELIVE[enname]:
                t = (h["cn"], h["pack"])
                if t in seen:
                    continue
                seen.add(t)
                print(f"       CN name = {h['cn']!r}   [{h['pack']}] {h['path'][:95]}")
        elif enname:
            print("       (EN name not found as a document key in cn packs)")
        en = resolve(load(x["repo"], x["pack"], "en"), x["path"])
        cn = resolve(load(x["repo"], x["pack"], "cn"), x["path"])
        el, cl = labels_for(en, key), labels_for(cn, key)
        print(f"     EN labels here ({len(el)}): {el}")
        print(f"     CN labels here ({len(cl)}): {cl}")
        if len(el) == len(cl):
            for A, B in zip(el, cl):
                mk = "  <<< finding" if B == x["cn_label"] else ""
                print(f"        {A!r:44} -> {B!r}{mk}")
        else:
            print("        *** COUNT MISMATCH ***")
        m = x["majority"]
        print(f"     majority CN label = {m['label']!r} {m['support']}/{m['total']}   own={x['own_share']['count']}")
        if a.ctx:
            j = (cn or "").find(x["cn_label"])
            if j >= 0:
                print("     CN ctx: ..." + (cn[max(0, j - a.ctx):j + a.ctx]).replace("\n", " ") + "...")
            k2 = (en or "").find(x["en_label"] or "\x00")
            if k2 >= 0:
                print("     EN ctx: ..." + (en[max(0, k2 - a.ctx):k2 + a.ctx]).replace("\n", " ") + "...")


main()
