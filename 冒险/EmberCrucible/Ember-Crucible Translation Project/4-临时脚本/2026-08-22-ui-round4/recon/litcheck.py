# -*- coding: utf-8 -*-
"""611 个键在**当前安装的上游语料**里逐个查字面量在不在。

为什么先查：面板 D 档按「字面量存在性」记 miss，而那道闸的 `max.missDistinct` 现在是 4。
先查清楚有没有查不到的，免得加完表才发现顶破天花板 —— 那时再改就是「为了绿而调阈值」。
口径与面板同款：纯 ASCII 键加词边界（裸 includes 在几 MB 源码里必然巧合命中）。
"""
import io, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')
EMBER = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
texts = []
for root, _dirs, files in os.walk(EMBER):
    if os.sep + "packs" in root:
        continue
    for f in files:
        if f.rsplit(".", 1)[-1].lower() in ("mjs", "js", "hbs", "html", "json", "css"):
            p = os.path.join(root, f)
            try:
                texts.append(io.open(p, encoding="utf-8").read())
            except Exception:
                pass
blob = "\n".join(texts)
print("语料", len(texts), "份 /", len(blob), "字符")

rows = [l.rstrip("\n").split("\t") for l in io.open("review.tsv", encoding="utf-8")]
miss = []
for fam, seg, cn in rows:
    if re.search(r"(?<![A-Za-z0-9_])" + re.escape(seg) + r"(?![A-Za-z0-9_])", blob):
        continue
    miss.append((fam, seg, cn))
print("611 个键里，上游语料查不到字面量的：", len(miss))
for r in miss[:30]:
    print("   ", r)
