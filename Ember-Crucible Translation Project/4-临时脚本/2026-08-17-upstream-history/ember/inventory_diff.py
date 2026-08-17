# -*- coding: utf-8 -*-
"""Structural inventory diff between two extracted ember English baselines.

Reports, per adventure pack, the add/remove/keep sets for each document
collection (journals, scenes, actors, items, tables, macros, playlists,
folders) and, for journals, the per-journal page add/remove sets.
For item-style packs, the entry-name add/remove sets.
"""
import io, json, os, sys

def load(p):
    return json.load(io.open(p, encoding="utf-8"))

def names(d):
    return set(d.keys()) if isinstance(d, dict) else set()

def adv_collections(pack):
    e = pack.get("entries", {})
    if not e:
        return None, {}
    k = list(e.keys())[0]
    adv = e[k]
    KEYS = ("folders", "journals", "scenes", "macros", "playlists", "tables", "items", "actors")
    # An Adventure document carries embedded collections; an Item/ActiveEffect
    # entry does not. Require at least one non-empty collection, else treat the
    # pack as a flat entry list.
    if not any(isinstance(adv.get(c), dict) and adv.get(c) for c in KEYS):
        return None, {}
    cols = {c: (adv.get(c) if isinstance(adv.get(c), dict) else {}) for c in KEYS}
    return k, cols

def journal_pages(journals):
    out = {}
    for jn, j in journals.items():
        pages = j.get("pages", {})
        out[jn] = set(pages.keys()) if isinstance(pages, dict) else set()
    return out

def report(A, B, files):
    for f in files:
        pa = os.path.join(A, f); pb = os.path.join(B, f)
        ea = load(pa) if os.path.exists(pa) else {"entries": {}}
        eb = load(pb) if os.path.exists(pb) else {"entries": {}}
        print(f"\n{'='*70}\n## {f}\n{'='*70}")
        if not os.path.exists(pa):
            print("  [pack did not exist in A]")
        na, ca = adv_collections(ea)
        nb, cb = adv_collections(eb)
        if ca and cb:
            if na != nb:
                print(f"  adventure entry renamed: {na!r} -> {nb!r}")
            for c in ca:
                sa, sb = names(ca[c]), names(cb[c])
                add, rem = sorted(sb - sa), sorted(sa - sb)
                print(f"\n  [{c}] {len(sa)} -> {len(sb)}   +{len(add)} -{len(rem)}")
                if add: print("    ADDED  : " + " | ".join(add))
                if rem: print("    REMOVED: " + " | ".join(rem))
            # journal pages
            ja, jb = journal_pages(ca["journals"]), journal_pages(cb["journals"])
            ta = sum(len(v) for v in ja.values()); tb = sum(len(v) for v in jb.values())
            print(f"\n  [journal pages] {ta} -> {tb}")
            for jn in sorted(set(ja) | set(jb)):
                pa_, pb_ = ja.get(jn, set()), jb.get(jn, set())
                add, rem = sorted(pb_ - pa_), sorted(pa_ - pb_)
                if add or rem:
                    print(f"    - {jn}: {len(pa_)} -> {len(pb_)}  +{len(add)} -{len(rem)}")
                    if add: print("        + " + " | ".join(add))
                    if rem: print("        - " + " | ".join(rem))
        else:
            sa, sb = names(ea.get("entries", {})), names(eb.get("entries", {}))
            add, rem = sorted(sb - sa), sorted(sa - sb)
            print(f"  entries {len(sa)} -> {len(sb)}   +{len(add)} -{len(rem)}")
            if add: print("    ADDED  : " + " | ".join(add))
            if rem: print("    REMOVED: " + " | ".join(rem))
            fa, fb = names(ea.get("folders", {})), names(eb.get("folders", {}))
            fadd, frem = sorted(fb - fa), sorted(fa - fb)
            if fadd or frem:
                print(f"  folders {len(fa)} -> {len(fb)}  +{fadd} -{frem}")

if __name__ == "__main__":
    A, B = sys.argv[1], sys.argv[2]
    files = sorted(set(x for x in os.listdir(A) + os.listdir(B)
                       if x.endswith(".json") and not x.startswith("_")))
    report(A, B, files)
