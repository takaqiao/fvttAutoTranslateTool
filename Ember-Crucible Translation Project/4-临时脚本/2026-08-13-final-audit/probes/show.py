# -*- coding: utf-8 -*-
"""打印某叶的 en/cn 全文（可选只打印标题行）。"""
import json, re, sys
from pathlib import Path


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


repo, pack, path = sys.argv[1], sys.argv[2], sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else "head"
for side in ("en", "cn"):
    d = dict(leaves(json.loads((Path(repo) / "compendium" / side / pack)
                               .read_text(encoding="utf-8-sig"))))
    s = d.get(path)
    print("=" * 100)
    print(side, "len", len(s) if s else None)
    if s is None:
        continue
    if mode == "head":
        for m in re.finditer(r"<h[1-6][^>]*>.*?</h[1-6]>", s):
            print("  ", m.group(0)[:200])
    elif mode == "full":
        print(s)
    else:
        for m in re.finditer(mode, s):
            a = max(0, m.start() - 200)
            print("  ...", s[a:m.end() + 200].replace("\n", " "))
