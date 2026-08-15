#!/usr/bin/env python3
"""Real-library term counts for the round17 landing (before/after).

Counts leaves (not raw file bytes) across BOTH plugin repos: every
`compendium/cn/*.json` plus `lang/cn.json`. Reports, per term, the number of
occurrences and the number of distinct leaves containing it.
"""
import json, os, sys, collections

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]
TERMS = ["毛毛雨", "细雨", "狂风暴雨", "风暴之月", "狂澜之月", "降雨"]


def leaves(node, path=""):
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, f"{path}.{i}")


occ = collections.Counter()
lv = collections.Counter()
per_file = collections.defaultdict(collections.Counter)
for repo in REPOS:
    files = []
    d = os.path.join(P, repo, "compendium", "cn")
    if os.path.isdir(d):
        files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".json")]
    lp = os.path.join(P, repo, "lang", "cn.json")
    if os.path.exists(lp):
        files.append(lp)
    for f in files:
        with open(f, encoding="utf-8-sig") as fh:
            data = json.load(fh)
        for _, s in leaves(data):
            for t in TERMS:
                c = s.count(t)
                if c:
                    occ[t] += c
                    lv[t] += 1
                    per_file[os.path.relpath(f, P)][t] += c

print("TERM        occurrences  leaves")
for t in TERMS:
    print(f"  {t:8s} {occ[t]:>10}  {lv[t]:>6}")
print("\nper-file (nonzero):")
for f in sorted(per_file):
    print(f"  {f}: " + ", ".join(f"{t}={n}" for t, n in sorted(per_file[f].items())))
