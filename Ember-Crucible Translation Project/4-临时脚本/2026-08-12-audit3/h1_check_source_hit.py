#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1: 拿 ember 模块的真实文件核对 ember-hardcoded-cn.mjs 里每个英文键能不能匹配上。

判据：补丁是在渲染出的 DOM 文本节点 / tooltip 属性上做**整串（trim 后）相等**替换，
所以英文串必须在模块源码（js 字面量、hbs 模板、lang/en.json 的值）里出现。
找不到 = 这个键永远不生效（阻断级）。

只读。
"""
import json
import re
import sys
from pathlib import Path

MOD = Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember")
KEYS = Path(sys.argv[1] if len(sys.argv) > 1 else "h1_keys.json")


def load_corpus():
    files = {}
    for p in sorted(MOD.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".mjs", ".js", ".hbs", ".html", ".json", ".css"}:
            continue
        try:
            files[str(p.relative_to(MOD))] = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"!! 读不了 {p}: {e}", file=sys.stderr)
    return files


def main():
    raw = KEYS.read_text(encoding="utf-8")
    # 去掉脚本尾部的 stderr 注释（若混进来）
    data = json.loads(raw[:raw.rindex("}") + 1])
    corpus = load_corpus()

    rows = []
    for table, d in data.items():
        if table == "PREFIXED":
            for e in d:
                rows.append((table, e["en"] + ": ", e["cn"]))
            continue
        if table == "PATTERNS":
            continue
        for en, cn in d.items():
            rows.append((table, en, cn))

    for table, en, cn in rows:
        hits = []
        for fn, txt in corpus.items():
            n = txt.count(en)
            if n:
                hits.append(f"{fn}x{n}")
        status = "HIT" if hits else "MISS"
        print(f"{status}\t{table}\t{en!r}\t{cn}\t{';'.join(hits[:4])}")


if __name__ == "__main__":
    main()
