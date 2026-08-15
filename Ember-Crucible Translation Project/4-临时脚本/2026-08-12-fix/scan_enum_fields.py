#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性探针：在英文基准里找「看起来像枚举 id / 数字 / 文档 id」却被 mapping 当正文抽出来的叶子。

判据（保守，宁可多报）：
  - 值是数字，或
  - 值是单个 token（无空格）且形如 lowerCamel / 全小写 / 16 位 Foundry id

按「叶子键名」聚合，输出 total / suspicious / 样例。用于回答审计 1.3 的延伸问题：
mapping 里还有没有别的字段也是枚举 id 却被当正文翻译了。

只读，不写任何文件。
"""
from __future__ import annotations
import collections
import glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPOS = ("1-Ember汉化插件", "2-Crucible汉化插件")

CODE = re.compile(r"^[a-z][a-zA-Z0-9_-]*$")
ID16 = re.compile(r"^[A-Za-z0-9]{16}$")


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    else:
        out.append((path[:], node))


def suspicious(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s or " " in s or len(s) > 24:
        return False
    return bool(CODE.fullmatch(s) or ID16.fullmatch(s))


def main() -> None:
    agg = collections.defaultdict(lambda: [0, 0, collections.Counter(), collections.Counter()])
    for repo in REPOS:
        for f in sorted(glob.glob(os.path.join(ROOT, repo, "compendium", "en", "*.json"))):
            if os.path.basename(f) == "_source.json":
                continue
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            out = []
            walk(d, [], out)
            for p, v in out:
                if not p:
                    continue
                key = p[-1]
                if key.isdigit() and len(p) >= 2:
                    key = p[-2] + "[]"
                rec = agg[key]
                rec[0] += 1
                if suspicious(v):
                    rec[1] += 1
                    rec[2][v if isinstance(v, str) else "<NUM>"] += 1
                    rec[3][os.path.basename(f)] += 1

    rows = [(k, t, s, c, files) for k, (t, s, c, files) in agg.items() if s]
    rows.sort(key=lambda r: -r[2])
    print(f"{'leaf key':30s} {'total':>7s} {'susp':>7s} {'ratio':>6s}  samples")
    for k, t, s, c, files in rows:
        print(f"{k:30s} {t:7d} {s:7d} {s / t:6.2f}  {dict(c.most_common(8))}")
        print(f"{'':30s} packs: {dict(files.most_common(4))}")


if __name__ == "__main__":
    main()
