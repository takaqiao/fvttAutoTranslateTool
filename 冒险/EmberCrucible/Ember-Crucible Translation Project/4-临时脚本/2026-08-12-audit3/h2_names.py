"""H2-B: dump every short NAME-ish leaf as EN | ZH | FR so the three can be read
side by side. Join key is the English string.

Usage: python h2_names.py [--max-len 46] [--filter regex]
"""
import argparse
import json
import os
import re

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\2-Crucible汉化插件\compendium"
FR_ROOT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\crucible-fr\compendium"
NAMEISH = re.compile(r"(^|\.)(name|adjective|tokenName|label|public|private)$")


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def pairs(en_node, tr_node, path, out):
    if isinstance(en_node, dict):
        for k, v in en_node.items():
            pairs(v, tr_node.get(k) if isinstance(tr_node, dict) else None,
                  path + [k], out)
    elif isinstance(en_node, list):
        for i, v in enumerate(en_node):
            pairs(v, tr_node[i] if isinstance(tr_node, list) and i < len(tr_node) else None,
                  path + [str(i)], out)
    elif isinstance(en_node, str) and en_node.strip():
        out.append((".".join(path), en_node, tr_node if isinstance(tr_node, str) else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=46)
    ap.add_argument("--all-paths", action="store_true")
    args = ap.parse_args()

    zh_map, fr_map, path_map, pack_map = {}, {}, {}, {}
    for fn in sorted(os.listdir(os.path.join(ROOT, "en"))):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        ours, theirs = [], []
        pairs(load(os.path.join(ROOT, "en", fn)).get("entries", {}),
              load(os.path.join(ROOT, "cn", fn)).get("entries", {}), [], ours)
        fen = os.path.join(FR_ROOT, "en", fn)
        ffr = os.path.join(FR_ROOT, "fr", fn)
        if os.path.exists(fen) and os.path.exists(ffr):
            pairs(load(fen).get("entries", {}), load(ffr).get("entries", {}), [], theirs)
        for p, e, t in ours:
            if len(e) > args.max_len:
                continue
            if not args.all_paths and not (NAMEISH.search(p) or p.count(".") == 0):
                continue
            if t:
                zh_map.setdefault(e, t)
                path_map.setdefault(e, p)
                pack_map.setdefault(e, fn)
        for p, e, t in theirs:
            if len(e) > args.max_len:
                continue
            if t:
                fr_map.setdefault(e, t)

    rows = [(e, zh_map[e], fr_map.get(e), path_map[e], pack_map[e])
            for e in sorted(zh_map) if e in fr_map]
    print(f"# {len(rows)} name-ish EN/ZH/FR triples")
    for e, z, f_, p, pk in rows:
        print(f"{e}\t{z}\t{f_}\t{pk}\t{p}")


if __name__ == "__main__":
    main()
