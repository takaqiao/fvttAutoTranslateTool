# -*- coding: utf-8 -*-
"""
scan_config_label_ownership.py —— 判据 P 的「越界改写别人家的值」形态

ember-hardcoded-cn.mjs:patchCrucibleConfig 遍历 crucible.CONFIG.languages /
.knowledge 的**每一条**，只要英文 label 在自己的表里就改写，**不判断这条是
Ember 新增的还是 crucible 自带的**。crucible 自带那些条目的译名正主是
2-Crucible汉化插件/lang/cn.json 的 KNOWLEDGE.* / LANGUAGES.*。

于是两份译名必须逐条一致，否则 ember_cn 会在 ready 时把 crucible-cn 的译名
覆盖成自己表里的那个（同一个世界里两处显示不同的词）。

本脚本比对：mjs 里的 KNOWLEDGE / LANGUAGES 表 ×（crucible-cn en.json 反查英文
→ cn.json 取译名）。只读。
"""
import json
import re

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
MJS = ROOT + r"\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
CN = ROOT + r"\2-Crucible汉化插件\lang\cn.json"
EN = ROOT + r"\2-Crucible汉化插件\lang\en.json"

PAIR = re.compile(r'"([^"]*)"\s*:\s*"([^"]*)"')


def table(src, name):
    m = re.search(r"const " + name + r" = \{(.*?)\n\};", src, re.S)
    return dict(PAIR.findall(m.group(1)))


def flat(o, p="", out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            flat(v, p + "." + k if p else k, out)
    else:
        out[p] = o
    return out


def main():
    src = open(MJS, encoding="utf-8").read()
    tables = {"KNOWLEDGE": table(src, "KNOWLEDGE"), "LANGUAGES": table(src, "LANGUAGES")}
    fc = flat(json.load(open(CN, encoding="utf-8")))
    fe = flat(json.load(open(EN, encoding="utf-8")))

    for ns, tbl in tables.items():
        keys = {k: v for k, v in fc.items() if k.startswith(ns + ".")}
        owned = 0
        drift = []
        for key, cnv in keys.items():
            enw = fe.get(key)
            if enw in tbl:
                owned += 1
                if tbl[enw] != cnv:
                    drift.append((key, enw, tbl[enw], cnv))
        print(f"{ns}: mjs 表 {len(tbl)} 条 / crucible-cn lang {len(keys)} 条 / "
              f"两边都有（= ember_cn 会覆盖 crucible-cn 的）{owned} 条 / 漂移 {len(drift)} 条")
        for d in drift:
            print(f"   {d[0]:34s} EN={d[1]:16s} mjs={d[2]}  lang={d[3]}")


if __name__ == "__main__":
    main()
