# -*- coding: utf-8 -*-
"""Independent cross-check of `extract/extract_en.mjs`.

Written from Babele 2.9.1's own source semantics, NOT transliterated from the
extractor, and fed from `6-工作区/raw-dumps/*.json` rather than from LevelDB —
so the two sides share no code and no input file.

Three quantities are compared per pack:
  1. document counts per type      (raw dump walk vs extractor `_source.json._meta`)
  2. emitted string leaves         (pre key-allocation; pure function of the mapping)
  3. final string leaves           (post key-allocation; what lands in the JSON)

Usage:
  python verify_leaves.py
"""
from __future__ import annotations
import io
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project"
DUMPS = os.path.join(P, "6-工作区", "raw-dumps")
BABELE_JS = (r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\babele"
             r"\script\mapping\default-mappings.js")

PACKS = [
    # dump file,        repo dir,           output json
    ("system.json", "1-系统汉化插件", "alienrpg.alien-rpg-system.json"),
    ("starterset.json", "2-新手包汉化插件",
     "alien-evolved-starterset.alien-evolved-starter-set.json"),
    ("corerules.json", "3-核心书汉化插件",
     "alien-evolved-corerules.alien-evolved-core-rules.json"),
]


def load_babele_defaults():
    """`default-mappings.js` is one `export const X = <JSON literal>;` — no code."""
    src = io.open(BABELE_JS, encoding="utf-8").read()
    m = re.search(r"export\s+const\s+defaultMappings\s*=\s*(\{.*\})\s*;?\s*$", src, re.S)
    if not m:
        raise SystemExit("could not slice the object literal out of default-mappings.js")
    return json.loads(m.group(1))


MAP = load_babele_defaults()

# ---------------------------------------------------------------- helpers


def gp(obj, path):
    cur = obj
    for k in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


def to_array(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        if isinstance(v.get("contents"), list):
            return v["contents"]
        return list(v.values())
    return []


def nes(v):
    return isinstance(v, str) and v.strip() != ""


# --------------------------------------- Babele mapping-block.js semantics


def normalized(defn):
    base = {k: v for k, v in (defn or {}).items() if k != "_variants"}
    variants = []
    for v in (defn or {}).get("_variants", []) or []:
        body = {k: val for k, val in (v or {}).items() if k != "_when"}
        if body:
            variants.append((v.get("_when"), body))
    return base, variants


def matches(cond, data):
    """mapping-block.js:176-206"""
    if not cond:
        return False
    if isinstance(cond.get("all"), list):
        return all(matches(c, data) for c in cond["all"])
    if isinstance(cond.get("any"), list):
        return any(matches(c, data) for c in cond["any"])
    p = cond.get("path")
    if not isinstance(p, str) or not p:
        return False
    # NB: Babele uses foundry.utils.getProperty, whose "missing" is undefined.
    # In Python a missing key and a JSON null are both None, so `exists` is
    # evaluated against key PRESENCE, which is the JS `typeof !== 'undefined'`.
    parts = p.split(".")
    cur, present = data, True
    for k in parts:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            present = False
            cur = None
            break
    checks = []
    if "equals" in cond:
        checks.append(cur == cond["equals"])
    if isinstance(cond.get("in"), list):
        checks.append(cur in cond["in"])
    if "exists" in cond:
        checks.append(present == bool(cond["exists"]))
    return len(checks) > 0 and all(checks)


def active_fields(defn, doc):
    """mapping-block.js:126-140 — base UNION matching variants, later wins."""
    base, variants = normalized(defn)
    eff = {}
    for k, v in base.items():
        if k.startswith("_"):
            continue
        eff.pop(k, None)
        eff[k] = v
    for when, body in variants:
        if matches(when, doc):
            for k, v in body.items():
                if k.startswith("_"):
                    continue
                eff.pop(k, None)
                eff[k] = v
    return eff


# --------------------------------------------------- key allocation


def identity_candidates(doc, tokens):
    out = []
    for t in tokens:
        if t == "range":
            r = doc.get("range")
            if isinstance(r, list) and len(r) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in r):
                c = f"{r[0]}-{r[1]}"
            else:
                c = None
        elif t == "sourceId":
            raw = (doc.get("flags", {}) or {}).get("core", {}).get("sourceId") \
                or (doc.get("_stats", {}) or {}).get("compendiumSource")
            c = raw.split(".")[-1] if nes(raw) else None
        else:
            v = doc.get(t)
            c = v if nes(v) else None
        if c and c not in out:
            out.append(c)
    return out


def merge_entries(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, str) or isinstance(b, str):
        return a if a == b else "\x00CONFLICT"
    if isinstance(a, list) or isinstance(b, list):
        return a if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) else "\x00CONFLICT"
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
            continue
        m = merge_entries(out[k], v)
        if m == "\x00CONFLICT":
            return "\x00CONFLICT"
        out[k] = m
    return out


class Alloc:
    def __init__(self):
        self.used = {}
        self.merged = 0

    def key(self, doc, tokens, entry, prefix):
        for c in identity_candidates(doc, tokens):
            if c not in self.used:
                self.used[c] = entry
                return c, None
            m = merge_entries(self.used[c], entry)
            if m != "\x00CONFLICT":
                self.used[c] = m
                self.merged += 1
                return c, m
        i = len(self.used)
        fb = f"{prefix}-{i}"
        while fb in self.used:
            i += 1
            fb = f"{prefix}-{i}"
        self.used[fb] = entry
        return fb, None


DEFAULT_TOKENS = ["name", "_id", "id"]

# ------------------------------------------------------------- extraction

stats = Counter()      # per-run counters, reset per pack


def extract(doc, dtype):
    defn = MAP.get(dtype)
    if not defn or not isinstance(doc, dict):
        return None
    stats[f"doc:{dtype}"] += 1
    out = {}
    for field, spec in active_fields(defn, doc).items():
        if isinstance(spec, str):
            v = gp(doc, spec)
            if nes(v):
                out[field] = v
                stats["emitted"] += 1
            continue
        if not isinstance(spec, dict):
            continue
        value = gp(doc, spec.get("path", field))
        conv = spec.get("converter")

        if conv == "document":
            child_type = spec.get("documentType")
            tokens = (MAP.get(child_type, {}).get("_identity", {}) or {}).get("export", DEFAULT_TOKENS)
            alloc = Alloc()
            m = {}
            for child in to_array(value):
                e = extract(child, child_type)
                if not e:
                    continue
                k, merged = alloc.key(child, tokens, e, "embedded")
                m[k] = merged if merged is not None else e
            stats["merged"] += alloc.merged
            if m:
                out[field] = m
        elif conv == "nameCollection":
            m = {}
            for it in to_array(value):
                n = it.get("name") if isinstance(it, dict) else None
                if nes(n) and n not in m:
                    m[n] = n
                    stats["emitted"] += 1
            if m:
                out[field] = m
        elif conv == "textCollection":
            m = {}
            for it in to_array(value):
                t = it.get("text") if isinstance(it, dict) else None
                if nes(t) and t not in m:
                    m[t] = t
                    stats["emitted"] += 1
            if m:
                out[field] = m
        elif conv == "structured":
            m = {}
            for it in to_array(value):
                if not isinstance(it, dict):
                    continue
                k = it.get(spec.get("key", "id"))
                if not nes(k):
                    continue
                sub = {}
                for sf, sp in (spec.get("mapping") or {}).items():
                    sv = gp(it, sp)
                    if nes(sv):
                        sub[sf] = sv
                        stats["emitted"] += 1
                if sub and k not in m:
                    m[k] = sub
            if m:
                out[field] = m
        else:
            # 'name', 'referencedDocumentField', anything unknown -> plain read
            if nes(value):
                out[field] = value
                stats["emitted"] += 1
    return out or None


def leaves(node, out):
    if isinstance(node, dict):
        for v in node.values():
            leaves(v, out)
    elif isinstance(node, list):
        for v in node:
            leaves(v, out)
    elif isinstance(node, str) and node.strip():
        out[0] += 1


def main():
    print(f"babele default-mappings: {len(MAP)} document types  <- {BABELE_JS}\n")
    grand = Counter()
    fails = 0
    for dump, repo, outname in PACKS:
        stats.clear()
        raw = json.load(io.open(os.path.join(DUMPS, dump), encoding="utf-8-sig"))
        assert len(raw) == 1, f"{dump}: expected one Adventure document, got {len(raw)}"
        key, adv = next(iter(raw.items()))
        alloc = Alloc()
        entry = extract(adv, "Adventure")
        k, _ = alloc.key(adv, DEFAULT_TOKENS, entry, "entry")
        mine = {"entries": {k: entry}}

        n_mine = [0]
        leaves(mine["entries"], n_mine)

        got_path = os.path.join(P, repo, "compendium", "en", outname)
        got = json.load(io.open(got_path, encoding="utf-8"))
        n_got = [0]
        leaves(got["entries"], n_got)

        src = json.load(io.open(os.path.join(P, repo, "compendium", "en", "_source.json"),
                                encoding="utf-8"))
        ok = (n_mine[0] == n_got[0])
        fails += 0 if ok else 1
        print(f"=== {dump}  ({src['packageId']} v{src['packageVersion']})")
        print(f"    raw key            : {key}")
        print(f"    docs walked        : "
              + ", ".join(f"{t[4:]}={n}" for t, n in sorted(stats.items()) if t.startswith("doc:")))
        print(f"    emitted leaves     : {stats['emitted']}   (pre key-allocation)")
        print(f"    collapsed on key   : {stats['merged'] + alloc.merged} documents")
        print(f"    independent leaves : {n_mine[0]}")
        print(f"    extractor leaves   : {n_got[0]}")
        print(f"    MATCH              : {'YES' if ok else 'NO  <<<<<<'}")
        print()
        grand["mine"] += n_mine[0]
        grand["got"] += n_got[0]
        for t, n in stats.items():
            if t.startswith("doc:"):
                grand[t] += n

    print("=== totals across all three packs")
    for t, n in sorted(grand.items()):
        if t.startswith("doc:"):
            print(f"    {t[4:]:<18} {n}")
    print(f"    independent leaves : {grand['mine']}")
    print(f"    extractor leaves   : {grand['got']}")
    print(f"    MATCH              : {'YES' if grand['mine'] == grand['got'] else 'NO'}")
    return 1 if fails or grand["mine"] != grand["got"] else 0


if __name__ == "__main__":
    sys.exit(main())
