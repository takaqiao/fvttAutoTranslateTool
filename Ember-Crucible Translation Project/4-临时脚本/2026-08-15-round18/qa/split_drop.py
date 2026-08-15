# -*- coding: utf-8 -*-
"""Y1-B：把 scan_dropped_terms 的**整叶计数**降到**块级计数**。

照 round16 `probes/split_dives.py` 的判据写：**按 HTML 标签切块、再逐块对齐**。
标签本身是机械、两侧逐字节相同，所以 `TAG.split()` 的块数两侧应当相等；不等的会被
报出来（`shape`），不会静默跳过。

对每条告警的每个词做三件事：
  1. 旧英文 / 新英文 / 中文各自切块；
  2. 用 `scan_dropped_terms` 同款 `strip_machinery` 把 `@UUID{标签}`、裸 `@UUID`（idmap）、
     `@Condition[…]`、`[[/…]]` 展开成「玩家读到的词」——**块内展开，不跨块**；
  3. 逐块比 `新英文里该词干出现 n_en` 与 `中文里该译名出现 n_cn`。
     只有 `n_cn > n_en` 的块才是嫌疑块；整叶层面的差额若全部落在
     `n_cn <= n_en` 的块上，说明差额来自「中文在别处正当使用该词」，是假阳性。

反空转：每次运行打印「扫了多少叶 / 多少块 / 多少词×块对」。
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "3-常用脚本", "qa")))
from scan_dropped_terms import (ID_MARKUP, SEM_MARKUP, WORD, stem, load_idmap,  # noqa: E402
                                load_json, leaves)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TAG = re.compile(r"<[^>]+>")
MASKP = [re.compile(r"@[A-Za-z]+\[[^\]]*\]\s*\{[^}]*\}"), ]


def expand(s, idmap):
    """块内把 enricher 展开成玩家读到的词（与 scan_dropped_terms.strip_machinery 同款，
    但块里已经没有 HTML 标签了）。"""
    def _uuid(m):
        label = m.group(2)
        if label:
            return " " + label + " "
        tid = (m.group(1) or "").split("#")[0].strip().split(".")[-1]
        return " " + (idmap or {}).get(tid, "") + " "
    s = ID_MARKUP.sub(_uuid, s)
    s = SEM_MARKUP.sub(lambda m: " " + re.sub(r"[._:\-/#|]+", " ",
                                              next(g for g in m.groups() if g is not None)) + " ", s)
    return s


def blocks(s):
    return TAG.split(s)


def count_en(block, st, idmap):
    ws = WORD.findall(expand(block, idmap))
    return sum(1 for w in ws if stem(w) == st), [w for w in ws if stem(w) == st]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="scan_dropped_terms 的 --out json")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--bindings", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="只看路径含此子串的叶")
    a = ap.parse_args()

    idmap = load_idmap(a.bindings)
    rep = json.load(open(a.report, encoding="utf-8"))
    bl_packs = rep["meta"]["baseline_packs"]

    cache = {}

    def pack_leaves(kind, pack):
        key = (kind, pack)
        if key in cache:
            return cache[key]
        if kind == "old":
            p = bl_packs[pack]
        else:
            p = os.path.join(a.repo, "compendium", kind, pack)
        d = {}
        if os.path.exists(p):
            leaves(load_json(p).get("entries", {}), [], d)
        cache[key] = d
        return d

    st = collections.Counter()
    rows = []
    for f in rep["findings"]:
        if a.only and a.only not in f["path"]:
            continue
        pack, path = f["pack"], f["path"]
        old_en = pack_leaves("old", pack).get(path, "")
        new_en = pack_leaves("en", pack).get(path, "")
        cn = pack_leaves("cn", pack).get(path, "")
        st["leaf"] += 1
        ob, nb, cb = blocks(old_en), blocks(new_en), blocks(cn)
        st["block_new"] += len(nb)
        shape = None
        if len(nb) != len(cb):
            shape = f"新EN块{len(nb)} vs CN块{len(cb)}"
            st["shape_new_vs_cn"] += 1
        if len(ob) != len(cb):
            st["shape_old_vs_cn"] += 1
        for h in f["dropped"]:
            term_st = stem(h["en"])
            cnterm = h["cn_term"]
            st["term_leaf"] += 1
            per = []
            if shape is None:
                for i, (nblk, cblk) in enumerate(zip(nb, cb)):
                    ne, surf = count_en(nblk, term_st, idmap)
                    nc = cblk.count(cnterm)
                    st["block_pair"] += 1
                    if ne or nc:
                        per.append({"blk": i, "en": ne, "cn": nc,
                                    "surf": sorted(set(surf)),
                                    "en_txt": re.sub(r"\s+", " ", nblk)[:230],
                                    "cn_txt": re.sub(r"\s+", " ", cblk)[:230]})
            sus = [p for p in per if p["cn"] > p["en"]]
            ok = [p for p in per if p["cn"] <= p["en"]]
            if shape is None:
                st["susblk"] += len(sus)
                if not sus:
                    st["term_clean"] += 1
            rows.append({
                "pack": pack, "path": path, "en": h["en"], "cn_term": cnterm,
                "leaf_counts": f"{h['en_old_n']}->{h['en_new_n']} / CN {h['cn_count']}",
                "shape": shape,
                "n_blk_new": len(nb), "n_blk_cn": len(cb), "n_blk_old": len(ob),
                "sus_blocks": sus, "ok_blocks_n": len(ok),
                "blk_en_total": sum(p["en"] for p in per),
                "blk_cn_total": sum(p["cn"] for p in per),
            })

    json.dump({"stats": dict(st), "rows": rows},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"扫了 {st['leaf']} 叶 · 新EN块 {st['block_new']} · 词×叶 {st['term_leaf']} · "
          f"逐块比对 {st['block_pair']} 块对 · 嫌疑块 {st['susblk']} · "
          f"整词无嫌疑块 {st['term_clean']} · 标签结构不同(新EN vs CN) {st['shape_new_vs_cn']} 叶"
          f" · (旧EN vs CN) {st['shape_old_vs_cn']} 叶")
    print(f"  -> {a.out}")
    for r in rows:
        if r["shape"]:
            print(f"  ⚠SHAPE {r['path'][-60:]} [{r['en']}] {r['shape']}")
        elif r["sus_blocks"]:
            print(f"  SUS {len(r['sus_blocks']):2d} {r['path'][-58:]} "
                  f"[{r['en']}→{r['cn_term']}] 叶 {r['leaf_counts']}")


if __name__ == "__main__":
    main()
