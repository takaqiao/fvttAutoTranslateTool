#!/usr/bin/env python3
"""查中文句子内部的裸空格（HTML 会把它渲染成可见空隙）。

  python scan_cjk_spacing.py <dir-or-file> [...]

为什么必须只在**文本节点内部**查
--------------------------------
把标签删掉再查 `[汉字] +[汉字]` 会**大量误报**：

    <strong>伤势: </strong>脚踝扭伤 <br /><strong>致命: </strong>No

剥完标签变成 `伤势: 脚踝扭伤 致命: No`，于是「扭伤 致命」看着像句中空格——
可那个空格在英文原文里就有，而且它在 `<br />` **之前**，浏览器根本不会把两个汉字排到一起。
实测：错误查法在 starterset 报 76 处，正确查法报的是另一个数。

⚠ 这是本项目第二次栽在「归一化不对称」上（第一次是 Map Pins 的 60.8%，见 PROJECT.md §8
2026-08-29 那条撤回）。**通则：做文本比对/扫描时，标签要么换成空格、要么按节点切开，
永远不要直接删掉。** 删标签会把跨标签的相邻字符黏成一个词。

真正要抓的是译者在**同一个文本节点内**、两个汉字之间留下的空格——通常来自
把英文的换行或词间空格照抄了过来。HTML 会把连续空白折成一个空格并**渲染出来**，
读者看到的就是句子中间裂开一道缝。
"""
import argparse
import io
import json
import os
import re
import sys

CJK_GAP = re.compile(r"[一-鿿][ \t]+[一-鿿]")
TAG = re.compile(r"<[^>]+>")


def text_nodes(html):
    """按标签切开，只返回文本节点。不跨标签边界。"""
    return TAG.split(html)


def scan_text(s):
    out = []
    for node in text_nodes(s):
        for m in CJK_GAP.finditer(node):
            lo = max(0, m.start() - 18)
            out.append(node[lo:m.end() + 18].replace("\n", "\\n"))
    return out


def walk_json(o, path=""):
    """JSON 里所有字符串叶子，逐个当 HTML 片段扫。"""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk_json(v, "%s.%s" % (path, k) if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk_json(v, "%s[%d]" % (path, i))
    elif isinstance(o, str):
        for hit in scan_text(o):
            yield path, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--show", type=int, default=5, help="每个文件打印几条样例")
    a = ap.parse_args()

    files = []
    for t in a.targets:
        if os.path.isdir(t):
            for root, _dirs, names in os.walk(t):
                for n in names:
                    if n.endswith((".html", ".json")):
                        files.append(os.path.join(root, n))
        else:
            files.append(t)

    total, bad = 0, 0
    for f in sorted(files):
        try:
            s = io.open(f, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        if f.endswith(".json"):
            try:
                hits = [h for _p, h in walk_json(json.loads(s))]
            except json.JSONDecodeError:
                print("  !! 解析失败 %s" % f, file=sys.stderr)
                continue
        else:
            hits = scan_text(s)
        if hits:
            bad += 1
            total += len(hits)
            print("  %-30s %d" % (os.path.basename(f), len(hits)))
            for h in hits[:a.show]:
                print("     …%s…" % h)

    print("\n文本节点内部的 CJK 裸空格：%d 处 / %d 个文件" % (total, bad))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
