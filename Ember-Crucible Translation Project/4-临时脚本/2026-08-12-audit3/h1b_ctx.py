#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1-b：给定英文串，打印**出现处的英中对照片段**（不是整叶），用来判定库里到底怎么译的。

用法： python h1b_ctx.py "Begin Event" [窗口字符数]
只读。
"""
import json
import os
import re
import sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]


def main():
    needle = sys.argv[1]
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    rx = re.compile(r"(?<![A-Za-z])" + re.escape(needle) + r"(?![A-Za-z])")
    n = 0
    for repo in REPOS:
        d = os.path.join(P, repo, "compendium")
        for f in sorted(os.listdir(os.path.join(d, "en"))):
            if not f.endswith(".json"):
                continue
            try:
                en = json.load(open(os.path.join(d, "en", f), encoding="utf-8"))
                cn = json.load(open(os.path.join(d, "cn", f), encoding="utf-8"))
            except Exception:
                continue

            def walk(e, c, path=""):
                nonlocal n
                if isinstance(e, dict):
                    for k, v in e.items():
                        walk(v, c.get(k) if isinstance(c, dict) else None, f"{path}.{k}")
                elif isinstance(e, list):
                    for i, v in enumerate(e):
                        walk(v, c[i] if isinstance(c, list) and i < len(c) else None, f"{path}[{i}]")
                elif isinstance(e, str):
                    for m in rx.finditer(e):
                        n += 1
                        a = max(0, m.start() - win)
                        print(f"--- {repo}/{f}{path}")
                        print("EN: ..." + e[a:m.end() + win].replace("\n", " ") + "...")
                        if isinstance(c, str):
                            print("CN: " + c[:1400].replace("\n", " "))
                        else:
                            print("CN: <无>")
                        break
            walk(en, cn)
    print(f"# 共 {n} 处", file=sys.stderr)


if __name__ == "__main__":
    main()
