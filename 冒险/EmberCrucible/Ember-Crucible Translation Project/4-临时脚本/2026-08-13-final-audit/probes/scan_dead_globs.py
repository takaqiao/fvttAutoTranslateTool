#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scan_dead_globs.py —— 「失效的守卫」类判据的第二个切面：**取材口径落空**（只读）

种子缺陷的本质是「判据写了，但它对不上目标，且失败是静默的」。
QA 扫描器里同一类问题长这样：某个 glob / 硬编码目录 / 后缀过滤今天已经匹配不到
任何文件，于是扫描器读了 0 个文件、报 0 条缺陷、退出码 0 —— 看上去「全绿」。

本脚本静态抽出 3-常用脚本 下所有脚本里的路径字面量与 glob 模式，逐个在磁盘上试解析，
列出**解析不到任何文件**的那些。

假阳性模式：
  1. 模式里含变量插值（f-string / .format / + 拼接）的，脚本只能取到常量片段，
     会误报「落空」。这类一律标 partial，需人工看。
  2. 输出路径（写文件用的）本来就可以不存在。脚本按变量名/上下文猜，猜不准要人工排。
  3. 有些脚本要求命令行传 --repo，默认常量只是回退值。
"""
import glob as globmod
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT_DIRS = [os.path.join(ROOT, "3-常用脚本")]

RE_STR = re.compile(r"""(?:'([^'\n]{3,120})'|"([^"\n]{3,120})")""")
LOOKS_PATHY = re.compile(r"[\\/]|\*|\.json$|\.mjs$|汉化插件|compendium|lang")


def main():
    rows = []
    for d in SCRIPT_DIRS:
        for dirpath, _dn, fn in os.walk(d):
            if "__pycache__" in dirpath:
                continue
            for n in fn:
                if not n.endswith((".py", ".mjs", ".js")):
                    continue
                p = os.path.join(dirpath, n)
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    src = f.read()
                for i, line in enumerate(src.split("\n"), 1):
                    s = line.strip()
                    if s.startswith("#") or s.startswith("//") or s.startswith("*"):
                        continue
                    for m in RE_STR.finditer(line):
                        lit = m.group(1) or m.group(2)
                        if not lit or not LOOKS_PATHY.search(lit):
                            continue
                        if lit.startswith(("http", "Compendium.", "@UUID", "<", "{")):
                            continue
                        rows.append((os.path.relpath(p, ROOT), i, lit))

    # 试解析
    dead, alive, skipped = [], 0, 0
    seen = set()
    for rel, ln, lit in rows:
        if lit in seen:
            continue
        seen.add(lit)
        if "{" in lit or "%" in lit:
            skipped += 1
            continue
        cands = [lit, os.path.join(ROOT, lit)]
        hit = False
        for c in cands:
            try:
                if os.path.exists(c) or globmod.glob(c, recursive=True):
                    hit = True
                    break
            except Exception:
                pass
        if hit:
            alive += 1
        else:
            dead.append((rel, ln, lit))

    print("路径/glob 字面量 %d 个（去重后 %d），解析得到 %d，含插值跳过 %d，**解析落空 %d**\n"
          % (len(rows), len(seen), alive, skipped, len(dead)))
    for rel, ln, lit in sorted(dead):
        print("  %-52s :%-4d %s" % (rel, ln, lit))


if __name__ == "__main__":
    main()
