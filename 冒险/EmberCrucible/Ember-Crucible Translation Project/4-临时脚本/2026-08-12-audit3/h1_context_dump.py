#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1: 为每个英文键打印它在 ember 模块里出现的位置与上下文，用来判断渲染路径。

只读。
"""
import json
import sys
from pathlib import Path

MOD = Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember")
EXTS = {".mjs", ".js", ".hbs", ".html"}


def main():
    keys = sys.argv[1:]
    files = []
    for p in sorted(MOD.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            files.append((str(p.relative_to(MOD)), p.read_text(encoding="utf-8", errors="replace").splitlines()))

    for k in keys:
        print(f"\n########## {k!r}")
        n = 0
        for fn, lines in files:
            for i, line in enumerate(lines):
                if k in line:
                    n += 1
                    ctx = line.strip()
                    if len(ctx) > 260:
                        j = ctx.find(k)
                        ctx = "…" + ctx[max(0, j - 110):j + 150] + "…"
                    print(f"  {fn}:{i+1}: {ctx}")
                    if n >= 6:
                        break
            if n >= 6:
                break
        if not n:
            print("  (MISS)")


if __name__ == "__main__":
    main()
