# -*- coding: utf-8 -*-
"""按 code 抽样打印 en/cn 对照片段，供人工核实。"""
import json, sys, re, collections
from pathlib import Path

f = Path(sys.argv[1])
code = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
grep = sys.argv[4] if len(sys.argv) > 4 else None
d = [x for x in json.loads(f.read_text(encoding="utf-8")) if x["code"] == code]
if grep:
    d = [x for x in d if re.search(grep, x["detail"] + x["path"])]
print(f"{len(d)} 条 {code}" + (f" (grep={grep})" if grep else ""))
for x in d[:n]:
    print("-" * 100)
    print(x["pack"], "|", x["path"])
    print("DETAIL:", x["detail"])
    print("EN:", x["en"][:600])
    print("CN:", x["cn"][:600])
