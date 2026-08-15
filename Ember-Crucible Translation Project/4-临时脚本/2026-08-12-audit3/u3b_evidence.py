# -*- coding: utf-8 -*-
"""U3(151-226) 证据台：给每条 uuid_swap UNCERTAIN 配齐「本处真实英文标签 + 目标文档 name」。

scan_uuid_swap 的 `en_label` 取的是**该叶内该目标的第一个**英文标签；一叶里同一目标
出现多次（尤其带 `#锚点`）时它不是本处的。这里按「同目标出现序」在英文叶里取对应那一个。

用法：
  python u3b_evidence.py --lo 151 --hi 226 --out <json>
"""
import argparse, importlib.util, json, os, re, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
SCAN = os.path.join(P, "3-常用脚本", "qa", "scan_uuid_swap.py")
SWAP = r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/uuid_swap.json"
IDS = os.path.join(P, "4-临时脚本", "2026-08-12-fix", "reports", "ember_ids.json")
REPOS = {"1-Ember汉化插件": os.path.join(P, "1-Ember汉化插件"),
         "2-Crucible汉化插件": os.path.join(P, "2-Crucible汉化插件")}
CJK = re.compile(r"[一-鿿]")

spec = importlib.util.spec_from_file_location("swapmod", SCAN)
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)


def load_pack(repo, side, pack):
    p = os.path.join(REPOS[repo], "compendium", side, pack)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def leaves(doc):
    out = {}
    sw.leaf_strings(doc, [], out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=151)
    ap.add_argument("--hi", type=int, default=226)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    swap = json.load(open(SWAP, encoding="utf-8"))
    ids = json.load(open(IDS, encoding="utf-8"))
    fs = swap["findings"][a.lo:a.hi]

    cache = {}
    def leafmap(repo, side, pack):
        k = (repo, side, pack)
        if k not in cache:
            d = load_pack(repo, side, pack)
            cache[k] = leaves(d) if d is not None else {}
        return cache[k]

    # 全库：目标 id -> 中文 name 候选（从 name/label 叶按 EN name 反查）
    # 先建 EN name -> CN name 计数
    name_idx = defaultdict(lambda: defaultdict(int))
    for repo in REPOS:
        en_dir = os.path.join(REPOS[repo], "compendium", "en")
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith(".json") or fn.startswith("_"):
                continue
            en = load_pack(repo, "en", fn) or {}
            cn = load_pack(repo, "cn", fn) or {}
            enl, cnl = leaves(en), leaves(cn)
            for path, ev in enl.items():
                seg = path.split(".")[-1]
                if seg not in ("name", "label"):
                    continue
                cv = cnl.get(path)
                if not cv or not CJK.search(cv):
                    continue
                name_idx[ev][cv] += 1

    rows = []
    for i, f in enumerate(fs, a.lo):
        repo, pack, path = f["repo"], f["pack"], f["path"]
        cnl = leafmap(repo, "cn", pack)
        enl = leafmap(repo, "en", pack)
        cs, es = cnl.get(path), enl.get(path)
        cl = sw.links_in(cs) if cs else []
        el = sw.links_in(es) if es else []
        idx = f["i"]
        key = f["key"]
        # 该处是本叶中该 key 的第几次出现
        ord_ = sum(1 for L in cl[:idx] if L["key"] == key)
        en_same = [L for L in el if L["key"] == key]
        here_en = en_same[ord_]["label"] if ord_ < len(en_same) else None
        here_en_target = en_same[ord_]["target"] if ord_ < len(en_same) else None
        cn_same = [L for L in cl if L["key"] == key]
        doc = ids.get(key) or {}
        docen = doc.get("name")
        cn_names = dict(sorted(name_idx.get(docen, {}).items(), key=lambda kv: -kv[1])) if docen else {}
        rows.append({
            "i": i, "repo": repo, "pack": pack, "path": path,
            "batch_path": f["batch_path"], "target": f["target"], "key": key,
            "cn_label": f["cn_label"], "scanner_en": f["en_label"],
            "here_en": here_en, "here_en_target": here_en_target,
            "cn_target_here": cl[idx]["target"] if idx < len(cl) else None,
            "en_labels_all": [L["label"] for L in en_same],
            "cn_labels_all": [L["label"] for L in cn_same],
            "majority": f["majority"], "own_share": f["own_share"],
            "doc_en_name": docen, "doc_type": doc.get("type"), "doc_via": doc.get("via"),
            "doc_cn_names": list(cn_names.items())[:5],
        })
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in rows:
        print(r["i"], "|", r["key"], "| docEN=", r["doc_en_name"], "| docCN=",
              (r["doc_cn_names"][0][0] if r["doc_cn_names"] else None),
              "| hereEN=", r["here_en"], "| cn=", r["cn_label"],
              "| maj=", r["majority"]["label"], r["majority"]["support"], "/", r["majority"]["total"])


main()
