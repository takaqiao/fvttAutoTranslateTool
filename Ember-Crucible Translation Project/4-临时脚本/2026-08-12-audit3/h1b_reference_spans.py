#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1-b：`<span class="reference">…</span>` 里留着英文、而 `.mjs` 已把同一个串翻成中文的地方。

后果：GM 指南写「点击 <span>Begin Event</span> 按钮」，而按钮上写的是「开始事件」——
读者按指南找不到那个按钮。这是 .mjs 通道与 compendium 通道的**同屏冲突**，
既有的 scan_cross_channel B 段只按 lang 键比对，看不见这一类。

只读。
"""
import json
import os
import re
import sys
from collections import Counter

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]
SPAN = re.compile(r'<span class="reference">([^<]{1,60})</span>')
ASCII_EN = re.compile(r"^[\x20-\x7E]+$")


def load_keys(path):
    raw = open(path, encoding="utf-8").read()
    data = json.loads(raw[:raw.rindex("}") + 1])
    covered = {}
    for t, d in data.items():
        if t == "PATTERNS":
            continue
        items = [(e["en"], e["cn"]) for e in d] if t == "PREFIXED" else list(d.items())
        for en, cn in items:
            covered[en] = (t, cn)
    return covered


def main():
    covered = load_keys(sys.argv[1])
    all_en = Counter()
    hits = []
    for repo in REPOS:
        d = os.path.join(P, repo, "compendium", "cn")
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            # 必须先解析 JSON：文件里引号是转义的，直接正则原文一个都匹配不到
            try:
                doc = json.load(open(os.path.join(d, f), encoding="utf-8"))
            except Exception:
                continue
            buf = []

            def walk(o):
                if isinstance(o, dict):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
                elif isinstance(o, str):
                    buf.append(o)
            walk(doc)
            txt = "\n".join(buf)
            for m in SPAN.finditer(txt):
                s = m.group(1).strip()
                if not ASCII_EN.match(s):
                    continue
                all_en[s] += 1
                if s in covered:
                    hits.append((repo, f, s, covered[s]))

    print("### A. `.mjs` 已翻、compendium/cn 的 reference span 仍是英文")
    c = Counter((h[2], h[3]) for h in hits)
    for (s, (t, cn)), n in c.most_common():
        print(f"{n}\t{s}\t-> .mjs[{t}]={cn}")
    print(f"# 命中 {sum(c.values())} 处 / {len(c)} 个串")

    print("\n### B. reference span 里全部英文串（普查，看规模）")
    for s, n in all_en.most_common(60):
        print(f"{n}\t{s}")
    print(f"# 英文 reference span 共 {sum(all_en.values())} 处 / {len(all_en)} 个串")


if __name__ == "__main__":
    main()
