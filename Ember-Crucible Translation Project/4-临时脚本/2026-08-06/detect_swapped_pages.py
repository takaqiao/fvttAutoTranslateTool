#!/usr/bin/env python3
"""Find translations sitting at the WRONG live path.

When upstream reshuffles page keys, a Chinese page can end up describing a
different room than the English at the same path. Coverage says 100% (there is
Chinese there), the todo list stays empty, and nothing else notices -- but the
player reads the wrong text. Mythspire Observatory's `Ancient Lift` is one.

Detection: `@UUID`/`[[/…]]` markup is copied verbatim into a translation, so it
fingerprints the source page. If a Chinese page's fingerprint matches some OTHER
English page of the same journal better than the English at its own path, the
pair is probably swapped.

2026-08-06 逐条读正文后的结论（4 个候选里只有 1 个是真错位）：

  Spellbreaker Tower / Storage      **真错位** —— 该卷有两间储藏室，开头 readaloud 几乎一模一样。
                                    中文实际是 `jyEjb9CXfSzRRZCf`（水/酒/灯油那间）的译文；
                                    真正的 `Storage`（床单/囚服那间）从未被翻译，却因为这里有中文
                                    而一直不进待译清单。
  Lightless Halls / Stone Bowl      不是错位 —— 中文对得上本页，只是英文被上游改短了，中文留着旧段落
  Aedir Signalpost / Lookout Post   同上
  Lightless Halls / Void Bridge     不是错位 —— 属第 8c 项（中文缺了英文的整块）

  另有一处指纹没抓到（标记太少被 min-markup 滤掉），是第 2 批译者读正文发现的：
  Mythspire Observatory / Ancient Lift —— 中文写的是 `CecLJBaIh4oKCvR8` 那间方厅升降梯。

试过给它加自动判据（相似度阈值、译文/英文长度比），两个方向都会判错：
按「英文本页要有足够标记」过滤会把唯一的真错位滤掉；按长度比 <0.9 判错位则正好把真错位
（1.07）判成假、把假的（0.61）判成真。**指纹只负责把 19605 条缩到 4 条，定性只能靠读正文。**

  python detect_swapped_pages.py [--pack <pack.json>]
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
CJK = re.compile(r'[一-鿿]')
MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]')


def fp(s: str) -> Counter:
    return Counter(MARKUP.findall(s))


def sim(a: Counter, b: Counter) -> float:
    u = sum((a | b).values())
    return sum((a & b).values()) / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default='ember.crucible-adventure.json')
    ap.add_argument('--min-markup', type=int, default=4,
                    help='少于这么多标记的页面指纹不可靠，跳过')
    a = ap.parse_args()

    en = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "en", a.pack), encoding="utf-8"))
    cn = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "cn", a.pack), encoding="utf-8"))
    EJ = en["entries"]["Ember Early Access"]["journals"]
    CJ = cn["entries"]["Ember Early Access"]["journals"]

    hits = []
    for jn, ej in EJ.items():
        en_pages = {pn: (p.get("text") or "") for pn, p in (ej.get("pages") or {}).items()}
        cn_pages = (CJ.get(jn, {}).get("pages") or {})
        for pn, cp in cn_pages.items():
            ct = cp.get("text") or ""
            if pn not in en_pages or not CJK.search(ct):
                continue
            f = fp(ct)
            if sum(f.values()) < a.min_markup:
                continue
            own = sim(f, fp(en_pages[pn]))
            best, best_s = None, 0.0
            for opn, ot in en_pages.items():
                if opn == pn:
                    continue
                s = sim(f, fp(ot))
                if s > best_s:
                    best, best_s = opn, s
            if best_s > own and best_s >= 0.5:
                ratio = len(ct) / max(len(en_pages[pn]), 1)
                hits.append((jn, pn, own, best, best_s, len(ct), ratio))

    hits.sort(key=lambda r: -(r[4] - r[2]))
    print(f"{'journal':<24}{'CN 所在页':<24}{'与本页':>7}{'更像':>24}{'相似':>7}{'CN字数':>8}{'CN/EN':>7}")
    for jn, pn, own, best, bs, n, ratio in hits:
        print(f"{jn[:23]:<24}{pn[:23]:<24}{own:>7.2f}  {best[:22]:<22}{bs:>7.2f}{n:>8}{ratio:>7.2f}")
    print(f"\n候选 {len(hits)} 处。**指纹只能缩小范围，不能定性** —— 试过用相似度阈值和"
          "译文/英文长度比自动分类，两个方向都会判错（见文件头的实测结论），必须逐条读正文。")


main()
