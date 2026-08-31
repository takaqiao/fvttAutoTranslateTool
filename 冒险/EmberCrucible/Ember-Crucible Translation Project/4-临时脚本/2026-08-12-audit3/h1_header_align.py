#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1: 按 HTML 标题标签逐个对齐英中，统计「英文小节标题 -> 中文写法」的真实分布。

比滑动窗口的词对齐可靠：同一叶子里第 i 个 <h2>/<h4>/<strong> 必然对第 i 个。
只在英中标题个数相等的叶子上统计，个数不等的整叶跳过（宁可漏不可错）。
只读。
"""
import collections
import json
import re
import sys
from pathlib import Path

P = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project")
TAG = re.compile(r"<(h[1-6])\b[^>]*>(.*?)</\1>", re.S)
STRIP = re.compile(r"<[^>]+>")


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(o, str):
        yield path, o


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "1-Ember汉化插件"
    wanted = set(sys.argv[2:]) or None
    cnd = P / repo / "compendium" / "cn"
    end = P / repo / "compendium" / "en"
    counts = collections.defaultdict(collections.Counter)
    for cf in sorted(cnd.glob("*.json")):
        ef = end / cf.name
        if not ef.exists():
            continue
        cn = dict(walk(json.loads(cf.read_text(encoding="utf-8"))))
        en = dict(walk(json.loads(ef.read_text(encoding="utf-8"))))
        for path, etext in en.items():
            ctext = cn.get(path)
            if not ctext or "<" not in etext:
                continue
            eh = [STRIP.sub("", m.group(2)).strip() for m in TAG.finditer(etext)]
            ch = [STRIP.sub("", m.group(2)).strip() for m in TAG.finditer(ctext)]
            if len(eh) != len(ch) or not eh:
                continue
            for a, b in zip(eh, ch):
                if wanted and a not in wanted:
                    continue
                counts[a][b] += 1
    for k in sorted(counts):
        print(f"{k!r}: " + ", ".join(f"{v}x{n}" for v, n in counts[k].most_common(6)))


if __name__ == "__main__":
    main()
