# -*- coding: utf-8 -*-
"""逐个比对 en/cn 的 [ ] 片段，找出中文侧丢掉的那个 ]。"""
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
d = {}
for side in ("en", "cn"):
    d[side] = dict(leaves(json.loads((Path(repo) / "compendium" / side / pack)
                                     .read_text(encoding="utf-8-sig"))))[path]

for side in ("en", "cn"):
    s = d[side]
    print("=" * 100, side)
    # 提取所有 @Xxx[...] / [[...]] 片段
    frags = []
    i = 0
    while i < len(s):
        if s[i] == "[":
            j = s.find("]", i)
            frags.append((i, s[i:j + 1] if j > 0 else s[i:i + 80]))
            i = (j + 1) if j > 0 else i + 1
        else:
            i += 1
    print(len(frags), "个 [ 起始片段")

# 找出中文侧 depth 变负/未归零的位置
s = d["cn"]
depth = 0
opens = []
for i, ch in enumerate(s):
    if ch == "[":
        opens.append(i)
        depth += 1
    elif ch == "]":
        if opens:
            opens.pop()
        depth -= 1
print("\n未闭合的 [ 位置:", opens)
for i in opens:
    print("  CN ...", s[max(0, i - 250):i + 250].replace("\n", " "), "...")

# 英文侧对应位置附近
print("\n英文侧同段落:")
en = d["en"]
for i in opens:
    key = re.sub(r"[^A-Za-z0-9@.#\[\]]", "", s[i:i + 60])[:40]
    print("  key:", key)
    m = re.search(re.escape(s[i:i + 45]).replace(r"\ ", r"\s*"), en)
    print("  EN 命中:", (en[max(0, m.start() - 250):m.start() + 250] if m else "（未直接命中，下面按 UUID 找）"))
    ids = re.findall(r"[A-Za-z0-9]{16}", s[i:i + 80])
    for u in ids:
        for mm in re.finditer(re.escape(u), en):
            print("   EN@uuid ...", en[max(0, mm.start() - 300):mm.start() + 220].replace("\n", " "), "...")
