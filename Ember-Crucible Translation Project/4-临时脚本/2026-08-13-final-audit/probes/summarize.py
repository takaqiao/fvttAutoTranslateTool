# -*- coding: utf-8 -*-
import json, collections, sys
from pathlib import Path
p = Path(sys.argv[1])
codes = sys.argv[2].split(",") if len(sys.argv) > 2 else None
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
d = json.loads(p.read_text(encoding="utf-8"))
print(collections.Counter(x["code"] for x in d))
for code in (codes or sorted({x["code"] for x in d})):
    sub = [x for x in d if x["code"] == code]
    print("=" * 30, code, len(sub))
    dc = collections.Counter(x["detail"] for x in sub)
    for k, n in dc.most_common(top):
        print(f"  {n:4d}  {k}")
    print("  -- packs:", collections.Counter(x["pack"] for x in sub).most_common())
