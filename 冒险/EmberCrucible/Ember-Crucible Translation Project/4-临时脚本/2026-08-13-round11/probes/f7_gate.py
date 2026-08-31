# -*- coding: utf-8 -*-
"""F7 probe: English-gated CN distribution across BOTH repos, with hardcoded
absolute paths so no non-ASCII argument ever crosses the shell boundary.

Usage:
    python f7_gate.py <query-json>

query-json = [{"label":..., "en":"<regex>", "cn":["写法1","写法2"], "ic":true?}, ...]
"""
import json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = [
    os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6"),
    os.path.join(ROOT, "2-Crucible\u6c49\u5316\u63d2\u4ef6"),
]
SKIP_KEYS = {"_id", "path", "_variants", "_when"}


def walk(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            if k in SKIP_KEYS:
                continue
            walk(v, cn.get(k) if isinstance(cn, dict) else None, path + [str(k)], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            sub = cn[i] if isinstance(cn, list) and i < len(cn) else None
            walk(v, sub, path + [str(i)], out)
    elif isinstance(en, str):
        out.append((".".join(path), en, cn if isinstance(cn, str) else ""))


_ROWS = None


def rows():
    global _ROWS
    if _ROWS is not None:
        return _ROWS
    r = []
    for repo in REPOS:
        en_dir = os.path.join(repo, "compendium", "en")
        cn_dir = os.path.join(repo, "compendium", "cn")
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = json.load(open(os.path.join(en_dir, fn), encoding="utf-8"))
            cp = os.path.join(cn_dir, fn)
            cn = json.load(open(cp, encoding="utf-8")) if os.path.isfile(cp) else {}
            sub = []
            walk(en.get("entries", {}), cn.get("entries", {}), ["entries"], sub)
            for p, e, c in sub:
                r.append((os.path.basename(repo), fn, p, e, c))
    _ROWS = r
    return r


def clip(s, n=200):
    s = s.replace("\n", " ")
    return s[:n] + ("\u2026" if len(s) > n else "")


def gate(q, show=4):
    rx = re.compile(q["en"], re.I if q.get("ic") else 0)
    terms = q["cn"]
    R = rows()
    en_match = [r for r in R if rx.search(r[3])]
    print("=" * 78)
    print("## %s   en=%r" % (q.get("label", ""), q["en"]))
    print("   scanned=%d  EN-matching leaves=%d" % (len(R), len(en_match)))
    for t in terms:
        gated = [r for r in R if t in (r[4] or "") and rx.search(r[3])]
        cn_only = [r for r in R if t in (r[4] or "") and not rx.search(r[3])]
        print("   CN %-14s gated_hit=%-5d cn_only=%d" % (repr(t), len(gated), len(cn_only)))
        for r in gated[:show]:
            print("       [hit ] %s::%s" % (r[1], r[2]))
        for r in cn_only[:show]:
            print("       [ONLY] %s::%s" % (r[1], r[2]))
            print("          EN: %s" % clip(r[3], 140))
            print("          CN: %s" % clip(r[4], 140))
    en_only = [r for r in en_match if r[4] and not any(t in r[4] for t in terms)]
    print("   EN matches but CN uses none of the listed renderings: %d" % len(en_only))
    for r in en_only[: show * 2]:
        print("       [en_only] %s::%s" % (r[1], r[2]))
        print("          EN: %s" % clip(r[3], 140))
        print("          CN: %s" % clip(r[4], 140))
    no_cn = [r for r in en_match if not r[4]]
    print("   EN matches, no CN leaf at all: %d" % len(no_cn))


if __name__ == "__main__":
    qs = json.load(open(sys.argv[1], encoding="utf-8"))
    show = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    for q in qs:
        gate(q, show)
