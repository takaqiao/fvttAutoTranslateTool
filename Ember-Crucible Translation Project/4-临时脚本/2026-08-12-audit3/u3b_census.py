# -*- coding: utf-8 -*-
"""某个 @UUID 目标 id 在全库的 (英文标签 → 中文标签) 逐叶清单；顺带打出该目标的 name 字段。

用法： python u3b_census.py --key <id> [--key <id> ...] [--ctx 60]
"""
import argparse, importlib.util, json, os, re, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SCAN = os.path.join(P, "3-常用脚本", "qa", "scan_uuid_swap.py")
IDS = os.path.join(P, "4-临时脚本", "2026-08-12-fix", "reports", "ember_ids.json")
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]

spec = importlib.util.spec_from_file_location("swapmod", SCAN)
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)


def leaves(doc):
    out = {}
    sw.leaf_strings(doc, [], out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", action="append", required=True)
    a = ap.parse_args()
    keys = set(a.key)
    ids = json.load(open(IDS, encoding="utf-8"))
    for k in a.key:
        print(f"### {k}  docEN={ (ids.get(k) or {}).get('name') }  type={(ids.get(k) or {}).get('type')}")

    rows = defaultdict(list)
    for rname in REPOS:
        en_dir = os.path.join(P, rname, "compendium", "en")
        cn_dir = os.path.join(P, rname, "compendium", "cn")
        for fn in sorted(os.listdir(cn_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            cn = json.load(open(os.path.join(cn_dir, fn), encoding="utf-8"))
            enp = os.path.join(en_dir, fn)
            en = json.load(open(enp, encoding="utf-8")) if os.path.isfile(enp) else {}
            cl, el = leaves(cn), leaves(en)
            for path, s in cl.items():
                if "@" not in s:
                    continue
                links = sw.links_in(s)
                if not any(L["key"] in keys for L in links):
                    continue
                es = el.get(path, "")
                elinks = sw.links_in(es) if es else []
                for key in keys:
                    cs_ = [L for L in links if L["key"] == key]
                    es_ = [L for L in elinks if L["key"] == key]
                    for i, L in enumerate(cs_):
                        e = es_[i]["label"] if i < len(es_) else None
                        rows[key].append((rname, fn, path, e, L["label"]))

    for key in a.key:
        print(f"\n===== {key} =====")
        agg = defaultdict(list)
        for r in rows[key]:
            agg[(r[3], r[4])].append(r)
        for (e, c), rs in sorted(agg.items(), key=lambda kv: -len(kv[1])):
            print(f"  EN={e!r:40} CN={c!r:30} x{len(rs):3}   e.g. {rs[0][1]}::{rs[0][2][:70]}")


main()
