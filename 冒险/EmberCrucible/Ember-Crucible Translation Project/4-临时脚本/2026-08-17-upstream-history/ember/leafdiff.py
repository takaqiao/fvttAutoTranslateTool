# -*- coding: utf-8 -*-
"""Leaf-level differ for Babele-shaped English extraction JSON.

Leaf = any string value reachable from the pack root. Path = dotted route with
list indices as [i]. Used to separate three buckets: added paths / removed
paths / same path different value.
"""
import io, json, os, sys

def leaves(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves(v, f"{prefix}[{i}]", out)
    elif isinstance(obj, str):
        out[prefix] = obj
    return out

def load(p):
    return json.load(io.open(p, encoding="utf-8"))

def pack_leaves(d):
    """Exclude nothing; caller decides. Returns dict path->str."""
    return leaves(d)

SCOPE = None  # None = all leaves; "entries" = only paths under entries.

def count_dir(d, skip=("_source.json", "_repaired.json", "README.md")):
    res = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json") or f in skip:
            continue
        lv = pack_leaves(load(os.path.join(d, f)))
        if SCOPE == "entries":
            lv = {k: v for k, v in lv.items() if k.startswith("entries.")}
        res[f] = lv
    return res

def diff(a, b):
    ka, kb = set(a), set(b)
    added = sorted(kb - ka)
    removed = sorted(ka - kb)
    changed = sorted(k for k in (ka & kb) if a[k] != b[k])
    return added, removed, changed

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--entries" in args:
        SCOPE = "entries"
        args.remove("--entries")
    sys.argv = [sys.argv[0]] + args
    mode = sys.argv[1]
    if mode == "count":
        d = sys.argv[2]
        tot = 0
        for f, lv in count_dir(d).items():
            print(f"{len(lv):>7}  {f}")
            tot += len(lv)
        print(f"{tot:>7}  TOTAL")
    elif mode == "diff":
        A, B = sys.argv[2], sys.argv[3]
        da, db = count_dir(A), count_dir(B)
        allf = sorted(set(da) | set(db))
        tA = tR = tC = 0
        for f in allf:
            a, b = da.get(f, {}), db.get(f, {})
            ad, rm, ch = diff(a, b)
            tA += len(ad); tR += len(rm); tC += len(ch)
            mark = ""
            if f not in da: mark = " [pack only in B]"
            if f not in db: mark = " [pack only in A]"
            print(f"{f}{mark}: +{len(ad)} -{len(rm)} ~{len(ch)}   (A={len(a)} B={len(b)})")
        print(f"TOTAL: +{tA} -{tR} ~{tC}")
