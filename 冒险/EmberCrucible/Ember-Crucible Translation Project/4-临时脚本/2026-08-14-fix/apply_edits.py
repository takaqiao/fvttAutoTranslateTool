#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把修复轮各分片返回的编辑清单串行套用到源码上。

为什么不让 agent 直接改：同一个文件（如 ember-hardcoded-cn.mjs）会被多个 agent 同时处理，
各自基于同一份原文产出编辑，谁后写谁覆盖。所以 agent 只交 (old, new) 对，由本脚本统一套用。

判据：每个 old 在目标文件里必须**逐字节唯一**（count == 1）。命中 0 次说明 agent 抄错或
文件已被别的编辑改动过；命中 >1 次说明上下文不够，都必须停下来人工看，不能猜。
"""
import argparse
import collections
import io
import json
import os
import sys

BS = chr(92)  # 反斜杠，避免在字面量里转义


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="workflow 结果 JSON（含各分片的 edits）")
    ap.add_argument("--write", action="store_true", help="不加则只试算")
    args = ap.parse_args()

    rows = json.load(io.open(args.batch, encoding="utf-8"))
    edits = [dict(slug=v.get("slug"), **e) for v in rows for e in (v.get("edits") or [])]
    for e in edits:
        e["file"] = e["file"].replace(BS, "/").lstrip("./")

    by_file = collections.OrderedDict()
    for e in edits:
        by_file.setdefault(e["file"], []).append(e)

    print("待套用 %d 处，覆盖 %d 个文件%s" % (len(edits), len(by_file),
                                          "" if args.write else "（试算，未写盘）"))
    ok = 0
    failures = []
    for path, group in by_file.items():
        if not os.path.exists(path):
            failures.append((path, "文件不存在", len(group)))
            continue
        text = io.open(path, encoding="utf-8").read()
        original = text
        applied = 0
        for e in group:
            hits = text.count(e["old"])
            if hits != 1:
                failures.append((path, "old 命中 %d 次（需 1）：%r" % (hits, e["old"][:70]), 1))
                continue
            text = text.replace(e["old"], e["new"], 1)
            applied += 1
            ok += 1
        if args.write and text != original:
            io.open(path, "w", encoding="utf-8").write(text)
        print("  %-58s %d/%d" % (path, applied, len(group)))

    print("\n成功 %d / 失败 %d" % (ok, len(failures)))
    for path, why, _ in failures:
        print("  ✗ %s: %s" % (path, why))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
