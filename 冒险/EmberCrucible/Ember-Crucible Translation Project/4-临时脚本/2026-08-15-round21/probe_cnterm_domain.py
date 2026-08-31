# -*- coding: utf-8 -*-
"""把 `scan_renamed_terms --mode cn-term` 的「0 条」拆开看：
候选专名多少个 → 其中「当前英文里没了」的多少个 → 有中文写法可追的多少个。

不复写任何正则/判据，**直接 import 真模块的函数**（硬约束 6：探针不许自己造判据）。

    python probe_cnterm_domain.py <--old 基准目录> [--old-cn 目录]
"""
from __future__ import annotations
import argparse
import importlib.util
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
SCAN = os.path.join(P, "3-常用脚本", "qa", "scan_renamed_terms.py")

spec = importlib.util.spec_from_file_location("srt", SCAN)
srt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--old-cn")
    ap.add_argument("--repo", default=os.path.join(P, "2-Crucible汉化插件"))
    a = ap.parse_args()

    old_en = srt.load_pack_leaves(a.old)
    cur_en = srt.load_pack_leaves(os.path.join(a.repo, "compendium", "en"))
    cur_cn = srt.load_pack_leaves(os.path.join(a.repo, "compendium", "cn"))
    print(f"读到：旧英文 {len(old_en)} 包 / {sum(len(v) for v in old_en.values())} 叶")
    print(f"      当前英文 {len(cur_en)} 包 / {sum(len(v) for v in cur_en.values())} 叶")
    print(f"      当前中文 {len(cur_cn)} 包 / {sum(len(v) for v in cur_cn.values())} 叶")
    assert old_en and cur_en and cur_cn, "有一侧读到 0 包 —— 探针空转"

    cur = srt.Corpus(cur_en)

    # ---- 候选：旧英文里的专名（name / tokenName / @UUID 标签），照抄真模块的过滤
    cands = {}

    def add(name, src):
        import re
        n = re.sub(r"\s+", " ", name).strip()
        if not n or len(n) < 4 or srt.CJK_RX.search(n):
            return
        pw = srt.phrase_words(n)
        if not pw or all(w in srt.WHITELIST_WORDS or w in srt.CONNECTORS for w in pw):
            return
        cands.setdefault(n, set()).add(src)

    for pack, rows in old_en.items():
        for p, s in rows:
            if p.endswith(".name") or p.endswith(".tokenName") or p == "name":
                add(s, "name")
            elif not srt.SKIP_PATH_RX.search(p):
                for m in srt.RX_UUID_LABEL.finditer(s):
                    add(m.group(1), "uuid-label")

    print(f"候选旧专名 {len(cands)} 个")
    assert cands, "候选 0 个 —— 探针空转"

    gone, alive, variant = [], 0, []
    for n in cands:
        if cur.has_phrase(n):
            alive += 1
            continue
        alt = srt.variant_alive(cur, n)
        if alt:
            variant.append((n, alt))
            continue
        gone.append(n)
    print(f"  ├ 当前英文里仍在的 {alive}")
    print(f"  ├ 只是换了拼写变体的 {len(variant)}  例: {variant[:5]}")
    print(f"  └ **当前英文里没了的 {len(gone)}**  例: {sorted(gone)[:15]}")

    if not gone:
        print("\n⇒ 候选集在这一步就空了：探测器 B 后面的中文追踪**一条都不会跑**，"
              "它报的 0 是「无从查起」，不是「查过了没问题」。")
        return 0

    old_cn = srt.load_pack_leaves(a.old_cn) if a.old_cn else None
    old_map = srt.build_name_map(old_en, cur_cn)
    if old_cn:
        for en, cns in srt.build_name_map(old_en, old_cn).items():
            old_map[en].update(cns)
    with_cn = [n for n in gone if old_map.get(n)]
    print(f"\n其中「有中文写法可追」的 {len(with_cn)}：{sorted(with_cn)[:20]}")
    print("⇒ 这些才是探测器 B 真正会去中文库里搜的东西；"
          "若最终 findings 仍为 0，那是「搜过了没残留」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
