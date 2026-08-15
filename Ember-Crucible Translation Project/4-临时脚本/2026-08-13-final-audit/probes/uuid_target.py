# -*- coding: utf-8 -*-
"""只看 @Xxx[...] 里**真正的目标部分**（第一个空格之前）与 #锚点，查非 ASCII 混入。

之前 S4 的假阳性模式：把 `label="中文" readaloud="中文"` 这些参数值也算进目标了。
Foundry 的 _parseEmbedConfig 用空白分词，第一个 token 才是 uuid，
@UUID / @Actor / @Item 这类则整串就是目标（不带参数）。

  U1  目标（第一个 token）含非 ASCII
  U2  #锚点部分含非 ASCII（锚点要与注入的 id 逐字相等，中文字符必然对不上）
  U3  目标里含全角空格 U+3000 / 不间断空格 U+00A0（肉眼看不出，分词必错）
  U4  中英目标不一致（uuid_swap 判据已覆盖，这里只作交叉验证计数）
"""
import json, re, sys, collections
from pathlib import Path

REF = re.compile(r"(@[A-Za-z]+|&(?:amp;)?[A-Za-z]+)\[([^\[\]]*)\]")
INVIS = {"　": "全角空格", " ": "不间断空格", "​": "零宽空格",
         "﻿": "BOM", "⁠": "词连接符"}


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


counts = collections.Counter()
rows = []
n = 0
for repo in sys.argv[1:]:
    repo = Path(repo)
    for f in sorted((repo / "compendium" / "en").glob("*.json")):
        cnp = repo / "compendium" / "cn" / f.name
        if not cnp.exists():
            continue
        en = dict(leaves(json.loads(f.read_text(encoding="utf-8-sig"))))
        cn = dict(leaves(json.loads(cnp.read_text(encoding="utf-8-sig"))))
        for p, s in cn.items():
            e = en.get(p, "")
            cts = [m.group(2).split(" ")[0] for m in REF.finditer(s)]
            ets = [m.group(2).split(" ")[0] for m in REF.finditer(e)]
            n += len(cts)
            for t in cts:
                base, _, anch = t.partition("#")
                if any(ord(c) > 0x7F for c in base):
                    counts["U1"] += 1
                    rows.append(("U1", repo.name, f.name, p, t[:120]))
                if anch and any(ord(c) > 0x7F for c in anch):
                    counts["U2"] += 1
                    rows.append(("U2", repo.name, f.name, p, t[:120]))
                bad = [INVIS[c] for c in t if c in INVIS]
                if bad:
                    counts["U3"] += 1
                    rows.append(("U3", repo.name, f.name, p, f"{bad} in {t[:100]!r}"))
            if collections.Counter(cts) != collections.Counter(ets):
                counts["U4"] += 1
                d = {k: (collections.Counter(cts).get(k, 0), collections.Counter(ets).get(k, 0))
                     for k in set(cts) | set(ets)
                     if collections.Counter(cts).get(k, 0) != collections.Counter(ets).get(k, 0)}
                rows.append(("U4", repo.name, f.name, p, str(d)[:250]))

print(f"扫描 @Xxx[...] 目标: {n}")
print(counts)
seen = collections.Counter()
for code, rn, pack, p, det in rows:
    seen[code] += 1
    if seen[code] > 20:
        continue
    print(f"[{code}] {rn} {pack} | {p}\n     {det}")
