# -*- coding: utf-8 -*-
"""Dump surrounding context for a regex hit across a repo side, and report whether the
hit sits inside a <sub data-system="..."> swap block."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_inventory import walk_json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPOS = {"ember": "1-Ember汉化插件", "crucible": "2-Crucible汉化插件"}

SUB = re.compile(r'<sub data-system="(\w+)"[^>]*>|</sub>')


def system_context(s, pos):
    """Return the data-system of the innermost <sub> containing pos, or None."""
    stack = []
    for m in SUB.finditer(s):
        if m.start() > pos:
            break
        if m.group(1):
            stack.append(m.group(1))
        elif stack:
            stack.pop()
    return stack[-1] if stack else None


def main():
    pat = re.compile(sys.argv[1])
    repo = sys.argv[2] if len(sys.argv) > 2 else "ember"
    side = sys.argv[3] if len(sys.argv) > 3 else "en"
    filt = sys.argv[4] if len(sys.argv) > 4 else None
    d = os.path.join(ROOT, REPOS[repo], "compendium", side)
    n = 0
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn == "_source.json":
            continue
        if filt and filt not in fn:
            continue
        obj = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        sink = []
        walk_json(obj, [], sink)
        for jp, s in sink:
            for m in pat.finditer(s):
                n += 1
                sysctx = system_context(s, m.start())
                a = max(0, m.start() - 120)
                print("### %s | %s\n    sub-system=%s\n    ...%s..." %
                      (fn, jp, sysctx, s[a:m.end() + 120].replace("\n", " ")))
    print("TOTAL", n)


if __name__ == "__main__":
    main()
