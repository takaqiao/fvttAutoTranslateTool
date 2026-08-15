# -*- coding: utf-8 -*-
"""增强器形态普查：全库有多少增强器、多少带标签、多少带可见文本参数、两侧数量是否配得上。"""
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
sys.path.insert(0, P + "/3-常用脚本/qa")
import assert_resolutions as A          # noqa: E402

AT = re.compile(r"@([A-Za-z]+)\[([^\]]*)\](?:\{([^{}]*)\})?")
BR = re.compile(r"\[\[/?([A-Za-z][\w.-]*)([^\]]*)\]\](?:\{([^{}]*)\})?")
PARAM = re.compile(r'\b([A-Za-z][\w-]*)\s*=\s*"([^"]*)"')
CJK = re.compile(r"[一-鿿]")


def main():
    repos = {"ember": os.path.join(P, "1-Ember汉化插件"),
             "crucible": os.path.join(P, "2-Crucible汉化插件")}
    ctx = A.Ctx(repos, {})
    n_leaf = n_leaf_enr = 0
    n_en = n_cn = 0
    lbl_en = lbl_cn = 0
    mismatch = Counter()
    verbs = Counter()
    params = Counter()
    param_cjk = Counter()
    n_pairs = 0
    n_pairs_both_label = 0
    for repo in repos:
        for pack, path, ev, cv in ctx.pairs[repo]:
            n_leaf += 1
            e = list(AT.finditer(ev))
            c = list(AT.finditer(cv))
            if not e and not c:
                continue
            n_leaf_enr += 1
            n_en += len(e)
            n_cn += len(c)
            lbl_en += sum(1 for m in e if m.group(3) is not None)
            lbl_cn += sum(1 for m in c if m.group(3) is not None)
            for m in c:
                verbs[m.group(1)] += 1
                for pm in PARAM.finditer(m.group(2)):
                    params[(m.group(1), pm.group(1))] += 1
                    if CJK.search(pm.group(2)):
                        param_cjk[(m.group(1), pm.group(1))] += 1
            if len(e) != len(c):
                mismatch[(repo, len(e), len(c))] += 1
                continue
            n_pairs += len(e)
            n_pairs_both_label += sum(1 for a, b in zip(e, c)
                                      if a.group(3) is not None and b.group(3) is not None)
    print(f"叶 {n_leaf} · 含 @增强器的叶 {n_leaf_enr}")
    print(f"@增强器：EN {n_en} / CN {n_cn}；带标签 EN {lbl_en} / CN {lbl_cn}")
    print(f"两侧数量相等的叶里可配对 {n_pairs} 对，其中两侧都有标签 {n_pairs_both_label} 对")
    print(f"两侧数量不等的叶 {sum(mismatch.values())}：{dict(list(mismatch.items())[:20])}")
    print("动词 top:", verbs.most_common(15))
    print("带引号参数 top:", params.most_common(15))
    print("参数值含中文的:", param_cjk.most_common(15))


main()
