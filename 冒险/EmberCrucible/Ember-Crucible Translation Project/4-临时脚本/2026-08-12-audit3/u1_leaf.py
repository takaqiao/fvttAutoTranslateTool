# -*- coding: utf-8 -*-
"""U1: dump the @UUID sequence of one leaf, EN vs CN, in document order."""
import argparse, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
MARK = re.compile(r'@UUID\[([^\]]*)\](?:\{([^{}]*)\})?')


def getp(obj, dotted):
    cur = obj
    for seg in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(seg)
        elif isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            return None
        if cur is None:
            return None
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="1-Ember汉化插件")
    ap.add_argument("--pack", default="ember.adventure.json")
    ap.add_argument("--path", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--grep", default=None)
    a = ap.parse_args()
    en = json.load(open(os.path.join(P, a.repo, "compendium", "en", a.pack), encoding="utf-8"))
    cn = json.load(open(os.path.join(P, a.repo, "compendium", "cn", a.pack), encoding="utf-8"))
    e, c = getp(en, a.path), getp(cn, a.path)
    print(f"### {a.path}")
    for tag, s in (("EN", e), ("CN", c)):
        print(f"--- {tag} @UUID sequence ({len(MARK.findall(s or ''))})")
        for i, m in enumerate(MARK.finditer(s or "")):
            print(f"  {i:3d} {m.group(1)}  {{{m.group(2)}}}")
    if a.full:
        print("--- EN full\n" + (e or ""))
        print("--- CN full\n" + (c or ""))
    if a.grep:
        for tag, s in (("EN", e), ("CN", c)):
            for m in re.finditer(a.grep, s or ""):
                lo, hi = max(0, m.start() - 220), min(len(s), m.end() + 220)
                print(f"[{tag}] ...{s[lo:hi]}...")


if __name__ == "__main__":
    main()
