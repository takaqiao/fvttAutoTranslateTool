#!/usr/bin/env python3
"""闸：译文的 HTML 结构必须与英文基准一致。

  python scan_html_fidelity.py [--show 8]

查两件事
--------
1. **标签多重集**逐字段与 en 基准相同（漏标签 / 多标签 / 配错）。
2. **不许多出 `id=` 属性**。译文管线曾经在 14 个字段上凭空加出
   `<h2 id="no-stunts-entered">`，而英文基准是光秃秃的 `<h2>`。
   这不只是脏——`character-sheet.mjs:1125` 拿
   `chatData.startsWith("<h2>No Stunts Entered</h2>")` 做字符串比较，
   多一个属性这条判断就废了。

⚠⚠ 路径配对必须用 os.sep，并且**断言配出来的路径真的不一样**
--------------------------------------------------------------
Windows 上 `glob` 返回的是反斜杠，`cnp.replace("/cn/","/en/")` 会**静默不生效**，
于是拿中文文件跟自己比，一切全绿。EC 项目栽过一次（PROJECT.md §3.7），
本项目在做这条闸的过程中**又栽了一次**。下面那行 assert 就是解药：
配对路径若没变化，立刻炸掉，而不是给出一个漂亮的 0。
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
TAG = re.compile(r"</?\w+")
HAS_ID = re.compile(r"<(?:h[1-6]|p|div|span|ol|ul|li|em|strong)\s+id=\"")


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, "%s[%d]" % (path, i))
    elif isinstance(o, str):
        yield path, o


def pair(cn_path):
    """cn 路径 → en 路径。必须真的换掉一段，否则是在拿文件跟自己比。"""
    seg_cn = os.sep + "cn" + os.sep
    seg_en = os.sep + "en" + os.sep
    en_path = cn_path.replace(seg_cn, seg_en)
    assert en_path != cn_path, (
        "路径配对失效：%r 里找不到 %r。"
        "多半是 glob 返回了另一种分隔符——别用写死的 '/cn/'。" % (cn_path, seg_cn))
    return en_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()

    checked = bad = 0
    pattern = os.path.join(ROOT, "[123]-*", "compendium", "cn", "*.json")
    for cnp in sorted(glob.glob(pattern)):
        enp = pair(cnp)
        if not os.path.exists(enp):
            print("  · %s：没有 en 基准，跳过" % os.path.basename(cnp))
            continue
        en = dict(leaves(json.load(io.open(enp, encoding="utf-8"))))
        cn = dict(leaves(json.load(io.open(cnp, encoding="utf-8"))))
        shown = 0
        for p, c in cn.items():
            e = en.get(p)
            if e is None or "<" not in e:
                continue
            checked += 1
            probs = []
            te, tc = collections.Counter(TAG.findall(e)), collections.Counter(TAG.findall(c))
            if te != tc:
                probs.append("标签多重集不同 英%s 中%s" % (dict(te - tc), dict(tc - te)))
            if len(HAS_ID.findall(c)) > len(HAS_ID.findall(e)):
                probs.append("多出 id= 属性")
            if probs:
                bad += 1
                if shown < a.show:
                    print("  ✗ %s\n      %s" % (p, "；".join(probs)))
                    shown += 1

    print("checked=%d  violations=%d" % (checked, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
