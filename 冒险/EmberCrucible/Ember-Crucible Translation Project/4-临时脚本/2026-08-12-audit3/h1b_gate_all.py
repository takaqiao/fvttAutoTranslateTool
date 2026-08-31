#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1-b：一次性把 .mjs 的每个英文键在**两个仓库**的 compendium 上跑英文闸。

单进程加载全部 en/cn 叶对，比逐键起 term_gate.py 子进程快两个数量级。
三桶与 term_gate 一致：
  gated_hit  英文命中该词且中文用了 .mjs 的译法
  en_only    英文命中但中文用了别的写法（要人看：可能 .mjs 与库里不一致）
  cn_only    中文有该译法但英文是别的词（多半不是残留）
只读。
"""
import json
import os
import re
import sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
REPOS = ["1-Ember汉化插件", "2-Crucible汉化插件"]


def load_pairs():
    pairs = []
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
                if isinstance(e, dict):
                    for k, v in e.items():
                        walk(v, c.get(k) if isinstance(c, dict) else None, f"{path}.{k}")
                elif isinstance(e, list):
                    for i, v in enumerate(e):
                        walk(v, c[i] if isinstance(c, list) and i < len(c) else None, f"{path}[{i}]")
                elif isinstance(e, str):
                    pairs.append((f"{repo}/{f}{path}", e, c if isinstance(c, str) else ""))
            walk(en, cn)
    return pairs


def main():
    raw = open(sys.argv[1], encoding="utf-8").read()
    data = json.loads(raw[:raw.rindex("}") + 1])
    pairs = load_pairs()
    print(f"# 叶对 {len(pairs)}", file=sys.stderr)

    rows = []
    for t, d in data.items():
        if t == "PATTERNS":
            continue
        items = [(e["en"], e["cn"]) for e in d] if t == "PREFIXED" else list(d.items())
        rows += [(t, en, cn) for en, cn in items]

    print("表\t英文键\t.mjs中文\ten_rows\tgated_hit\ten_only\tcn_only\t示例(en_only)")
    for t, en, cn in rows:
        rx = re.compile(r"(?<![A-Za-z])" + re.escape(en) + r"(?![A-Za-z])")
        g = eo = co = enr = 0
        sample = ""
        for path, e, c in pairs:
            hit_en = bool(rx.search(e))
            hit_cn = cn in c
            if hit_en:
                enr += 1
                if hit_cn:
                    g += 1
                else:
                    eo += 1
                    if not sample:
                        sample = path
            elif hit_cn:
                co += 1
        flag = "  <<<" if eo else ""
        print(f"{t}\t{en}\t{cn}\t{enr}\t{g}\t{eo}\t{co}\t{sample[:120]}{flag}")


if __name__ == "__main__":
    main()
