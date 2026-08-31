#!/usr/bin/env python3
"""Cut item 8c into agent-sized units.

第 8c 项是「已译页里缺失的整块内容」：上游把页面加长了，中文还停在旧版本，
覆盖率却因为路径上有中文而算它 100%，待译清单永远不会列它。

与普通翻译单元的不同之处：每条要同时给出**英文原文**和**现有中文**，
agent 的任务是把缺掉的区块补进现有中文里，而不是从零重译 —— 已有的部分要原样保留，
否则会把别人校对过的译文洗掉。

验收有个天然抓手：`apply_translations.py` 的标记闸拿英文比对标记多重集，
补全之前这些条目**必然**被拒（缺的区块里带着 @UUID 与标签），补对了才会过。
所以「0 拒绝」在这批的含义比平时更强 —— 它直接证明区块补齐了。

  python prep_8c.py <每单元字符数> [--min <单条最小估算字符>]
"""
from __future__ import annotations
import json
import os
import re
import sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
ROOT = os.environ.get("EMBER_PARALLEL_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "parallel")
PACK = "ember.crucible-adventure.json"
REPORT = os.path.join(P, "5-其他内容", "reports", "ember", "missing_blocks.json")


def get_at(node, parts):
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p)
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    min_chars = 200
    if "--min" in sys.argv:
        min_chars = int(sys.argv[sys.argv.index("--min") + 1])

    items = json.load(open(REPORT, encoding="utf-8"))["items"]
    en = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "en", PACK), encoding="utf-8"))["entries"]
    cn = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "cn", PACK), encoding="utf-8"))["entries"]

    rows = []
    for it in items:
        if it["est_chars"] < min_chars:
            continue
        parts = it["path"].split(".")
        e, c = get_at(en, parts), get_at(cn, parts)
        if not isinstance(e, str) or not isinstance(c, str):
            continue
        rows.append({"path": it["path"], "en": e, "current_cn": c,
                     "missing_blocks": it["missing_blocks"], "en_blocks": it["en_blocks"],
                     "chars": it["est_chars"]})
    rows.sort(key=lambda r: -r["chars"])

    chunks, cur, acc = [], [], 0
    for r in rows:
        cur.append(r); acc += r["chars"]
        if acc >= chunk:
            chunks.append(cur); cur, acc = [], 0
    if cur:
        chunks.append(cur)

    manifest = []
    for i, ch in enumerate(chunks, 1):
        d = os.path.join(ROOT, f"fill-{i}")
        os.makedirs(d, exist_ok=True)
        json.dump({"journal": f"补缺块 第 {i}/{len(chunks)} 批", "items": ch},
                  open(os.path.join(d, "todo.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        json.dump({}, open(os.path.join(d, "already_translated.json"), "w", encoding="utf-8"))
        c = sum(x["chars"] for x in ch)
        manifest.append({"unit": f"fill-{i}", "dir": d, "items": len(ch), "chars": c})
        print(f"fill-{i:<22}{len(ch):>5} 条 {c:>8} 估算字符")

    out = os.path.join(ROOT, "manifest.fill.json")
    json.dump(manifest, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n合计 {sum(m['chars'] for m in manifest)} 估算字符 / {len(rows)} 条")
    print("->", out)


main()
