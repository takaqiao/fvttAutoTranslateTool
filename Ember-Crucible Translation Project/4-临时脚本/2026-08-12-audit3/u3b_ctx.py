# -*- coding: utf-8 -*-
"""打印某条 finding 所在叶子的英中片段（围绕该 @UUID 前后各 N 字）。"""
import argparse, importlib.util, json, os, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
spec = importlib.util.spec_from_file_location(
    "sw", os.path.join(P, "3-常用脚本", "qa", "scan_uuid_swap.py"))
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)
SWAP = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/uuid_swap.json"


def lv(repo, side, pack):
    p = os.path.join(P, repo, "compendium", side, pack)
    d = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}
    o = {}
    sw.leaf_strings(d, [], o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", type=int, action="append", required=True)
    ap.add_argument("--w", type=int, default=220)
    a = ap.parse_args()
    fs = json.load(open(SWAP, encoding="utf-8"))["findings"]
    for i in a.i:
        f = fs[i]
        cn = lv(f["repo"], "cn", f["pack"]).get(f["path"], "")
        en = lv(f["repo"], "en", f["pack"]).get(f["path"], "")
        print(f"\n########## {i}  {f['pack']} :: {f['path']}")
        key = f["key"]
        for tag, s in (("EN", en), ("CN", cn)):
            for m in re.finditer(re.escape(key), s):
                st = max(0, m.start() - a.w)
                print(f"  [{tag}] …{s[st:m.end()+a.w]}…")


main()
