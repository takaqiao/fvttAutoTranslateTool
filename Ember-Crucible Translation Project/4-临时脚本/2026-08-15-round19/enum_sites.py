# -*- coding: utf-8 -*-
"""枚举「阿克图瑞尔 / 阿克图里安」在 CN compendium 全库的每一处出现，
并按 assert_resolutions.split_blocks 的涂空规则标出哪些落在增强器内。

输出 sites.json：[{repo, pack, path, cn_off, term, in_enricher, enr_kind, en_label, cn_label, pair_idx}, …]
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
QA = P + "/3-常用脚本/qa"
sys.path.insert(0, QA)
import assert_resolutions as A          # noqa: E402

TERM = re.compile(r"阿克图瑞尔|阿克图里安")
# 与 A._ENRICHER 同源；单独留一份是为了能报出「命中的是哪一类增强器」
ENR = [("at", re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?")),
       ("br", re.compile(r"\[\[[^\]]*\]\](?:\{[^{}]*\})?"))]
LBL = re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{([^{}]*)\})?")


def spans(s):
    out = []
    for kind, rx in ENR:
        for m in rx.finditer(s):
            out.append((m.start(), m.end(), kind, m.group()))
    return out


def main():
    repos = {"ember": os.path.join(P, "1-Ember汉化插件"),
             "crucible": os.path.join(P, "2-Crucible汉化插件")}
    ctx = A.Ctx(repos, {})
    sites = []
    n_leaf = n_leaf_hit = 0
    for repo in repos:
        for pack, path, ev, cv in ctx.pairs[repo]:
            n_leaf += 1
            ms = list(TERM.finditer(cv))
            if not ms:
                continue
            n_leaf_hit += 1
            sp = spans(cv)
            en_enr = [m.group(1) for m in LBL.finditer(ev)]
            cn_enr = [(m.start(), m.end(), m.group(1)) for m in LBL.finditer(cv)]
            for m in ms:
                hit = next((x for x in sp if x[0] <= m.start() < x[1]), None)
                pair_idx = en_label = cn_label = None
                if hit:
                    for i, (a, b, lab) in enumerate(cn_enr):
                        if a <= m.start() < b:
                            pair_idx = i
                            cn_label = lab
                            en_label = en_enr[i] if i < len(en_enr) else None
                            break
                sites.append({"repo": repo, "pack": pack, "path": path,
                              "cn_off": m.start(), "term": m.group(),
                              "in_enricher": bool(hit), "enr_kind": hit[2] if hit else None,
                              "enr_text": hit[3] if hit else None,
                              "pair_idx": pair_idx, "en_label": en_label, "cn_label": cn_label,
                              "n_en_enr": len(en_enr), "n_cn_enr": len(cn_enr)})
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(sites, open(os.path.join(here, "sites.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ins = [s for s in sites if s["in_enricher"]]
    print(f"叶总数 {n_leaf} · 含该词的叶 {n_leaf_hit}")
    print(f"该词出现总数 {len(sites)} · 落在增强器内 {len(ins)}"
          f"（{len(ins) * 100.0 / max(1, len(sites)):.1f}%）")
    from collections import Counter
    print("  增强器类别：", Counter(s["enr_kind"] for s in ins))
    print("  其中带中文标签且能配到 EN 同序号的：",
          sum(1 for s in ins if s["en_label"] is not None))
    print("  EN 同序号是裸增强器（无标签）：",
          sum(1 for s in ins if s["pair_idx"] is not None and s["en_label"] is None
              and s["pair_idx"] < s["n_en_enr"]))
    print("  两侧增强器数不等的叶里的：",
          sum(1 for s in ins if s["n_en_enr"] != s["n_cn_enr"]))
    print("  落在方括号内（不是标签）的：",
          sum(1 for s in ins if s["cn_label"] is None or s["term"] not in (s["cn_label"] or "")))


main()
