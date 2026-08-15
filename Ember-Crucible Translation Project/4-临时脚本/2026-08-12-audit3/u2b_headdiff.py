# -*- coding: utf-8 -*-
"""For findings i..j: CN labels of the finding's target key at git HEAD vs in the working tree.

Tells apart "the audit-3 pass already changed this" from "this was never touched".
"""
import argparse, json, os, re, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SCR = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3"
SW = json.load(open(SCR + "/uuid_swap.json", encoding="utf-8"))["findings"]
MARK = re.compile(r'(@[A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?')
_c = {}


def head_pack(repo, pack):
    k = (repo, pack)
    if k not in _c:
        out = subprocess.run(["git", "-C", os.path.join(ROOT, repo), "show",
                              f"HEAD:compendium/cn/{pack}"], capture_output=True)
        _c[k] = json.loads(out.stdout.decode("utf-8")) if out.returncode == 0 else {}
    return _c[k]


def cur_pack(repo, pack):
    k = (repo, pack, "cur")
    if k not in _c:
        _c[k] = json.load(open(os.path.join(ROOT, repo, "compendium", "cn", pack), encoding="utf-8"))
    return _c[k]


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


def labels(s, key):
    return [m.group(3) for m in MARK.finditer(s or "")
            if (m.group(2) or "").split()[0].split("#")[0].split(".")[-1] == key] if s else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", required=True)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.idx.split("-"))
    for i in range(lo, hi):
        x = SW[i]
        h = resolve(head_pack(x["repo"], x["pack"]), x["path"])
        c = resolve(cur_pack(x["repo"], x["pack"]), x["path"])
        hl, cl = labels(h, x["key"]), labels(c, x["key"])
        same = "SAME" if hl == cl else "CHANGED"
        print(f"[{i}] {same:8} key={x['key']}  HEAD={hl}  ->  NOW={cl}")


main()
