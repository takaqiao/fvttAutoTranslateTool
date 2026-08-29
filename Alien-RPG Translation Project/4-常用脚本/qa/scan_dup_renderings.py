#!/usr/bin/env python3
"""闸：同一段英文在同一个包里不许出现两种中文。

  python scan_dup_renderings.py [--min-len 40] [--show 6] [<repo> ...]

这条闸在防什么
--------------
合集里同一件物品常常存在两份：一份在 Adventure 的 `items/` 下，一份内嵌在
某个 actor 身上。并行翻译时它们会落到不同的切片、被**当成两段新内容各翻一遍**，
于是同一段英文得到两种措辞。玩家在物品栏看到一种说法，点开角色卡看到另一种。

PROJECT.md §8 已经记过这个成因（Phase 4 复核实测 69 对同英文双中文）。
本闸把它变成可复现的检查，而不是靠人再发现一次。

为什么有 --min-len
------------------
短串（"Yes" / "Fire" / 单个词的标签）在不同语境下**本来就该**译得不一样，
一刀切会淹没在假阳性里。默认只看 40 字符以上的整句/整段。

⚠ 比对必须走「英文基准 → 同一路径的中文」，不能只看中文自身。
   compendium/en 是抽取出来的基准，路径与 cn 一一对应。
"""
import argparse
import collections
import glob
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, "%s[%d]" % (path, i))
    elif isinstance(o, str):
        yield path, o


def plain(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*", help="不给就扫全部 [123]-*")
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    pats = a.repos or ["1-系统汉化插件", "2-新手包汉化插件", "3-核心书汉化插件"]
    checked = bad = 0
    for repo in pats:
        for cnp in sorted(glob.glob(os.path.join(ROOT, repo, "compendium", "cn", "*.json"))):
            enp = cnp.replace(os.sep + "cn" + os.sep, os.sep + "en" + os.sep)
            if not os.path.exists(enp):
                continue
            en = dict(leaves(json.load(io.open(enp, encoding="utf-8"))))
            cn = dict(leaves(json.load(io.open(cnp, encoding="utf-8"))))
            groups = collections.defaultdict(dict)
            for p, e in en.items():
                c = cn.get(p)
                if not c or c == e or len(e) < a.min_len:
                    continue
                groups[e][p] = c
            checked += len(groups)
            dup = {e: v for e, v in groups.items() if len(set(v.values())) > 1}
            if dup:
                print("  ✗ %s：%d 组" % (os.path.basename(cnp), len(dup)))
                for e, locs in list(dup.items())[:a.show]:
                    print("      EN: %s…" % plain(e)[:78])
                    for c in sorted(set(locs.values())):
                        print("        → %s" % plain(c)[:76])
                    print("        位置: %s" % list(locs)[:2])
                bad += len(dup)

    print("checked=%d  violations=%d" % (checked, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
