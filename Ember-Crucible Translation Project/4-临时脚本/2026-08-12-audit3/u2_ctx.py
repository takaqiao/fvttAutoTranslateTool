# -*- coding: utf-8 -*-
"""Print EN/CN leaf pair (with a window around the target link) for uuid_swap findings."""
import argparse, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SW = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/uuid_swap.json"

def get(d, path):
    cur = d
    for seg in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(seg)]
        else:
            if seg not in cur: return None
            cur = cur[seg]
    return cur

def leaf(repo, pack, path):
    # path uses dot segments; keys may contain dots -> resolve greedily
    def resolve(base, p):
        segs = p.split(".")
        cur = base; i = 0
        while i < len(segs):
            if isinstance(cur, list):
                cur = cur[int(segs[i])]; i += 1; continue
            # try longest match
            for j in range(len(segs), i, -1):
                k = ".".join(segs[i:j])
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]; i = j; break
            else:
                return None
        return cur
    out = {}
    for side in ("en", "cn"):
        fp = os.path.join(ROOT, repo, "compendium", side, pack)
        d = json.load(open(fp, encoding="utf-8"))
        out[side] = resolve(d, path)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", required=True, help="comma list of finding indexes")
    ap.add_argument("--win", type=int, default=260)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    F = json.load(open(SW, encoding="utf-8"))["findings"]
    for i in [int(x) for x in a.idx.split(",")]:
        x = F[i]
        L = leaf(x["repo"], x["pack"], x["path"])
        print("=" * 100)
        print(f"[{i}] {x['repo']} / {x['pack']}")
        print("path:", x["path"])
        print("target:", x["target"], "| en_label:", x["en_label"], "| cn_label:", repr(x["cn_label"]),
              "| maj:", x["majority"]["label"], x["majority"]["support"], "/", x["majority"]["total"])
        for side in ("en", "cn"):
            s = L[side] or ""
            print("-" * 40, side.upper(), f"len={len(s)}")
            if a.full:
                print(s)
            else:
                key = x["key"]
                hits = [m.start() for m in re.finditer(re.escape(key), s)]
                if not hits and side == "cn":
                    lbl = x["cn_label"]
                    hits = [m.start() for m in re.finditer(re.escape(lbl), s)] if lbl else []
                if not hits:
                    print(s[:a.win * 2])
                for h in hits:
                    print("   ...", s[max(0, h - a.win):h + a.win], "...")
main()
