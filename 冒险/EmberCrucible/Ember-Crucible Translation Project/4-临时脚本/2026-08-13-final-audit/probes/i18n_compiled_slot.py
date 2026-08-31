# -*- coding: utf-8 -*-
"""
probe: i18n_compiled_slot —— 对 **实际被加载的构建产物** 扫同一类缺陷（只读）

为什么要单独扫 compiled：
    crucible 的 system.json esmodules 只有 `crucible-compiled.mjs`（48309 行）。
    `module/` 是源码目录，但主入口（Hooks.once("init"/"i18nInit"/"setup"…)、
    preLocalizeConfig()、CONFIG 装配）**不在 module/ 里**，只存在于 compiled。
    所以只扫 module/ 会漏掉入口层的注入点。

判据同 i18n_slot_gap.py：往「已证明会过 i18n 的属性槽」写裸英文字面量，
且该串不在 上游 en.json / 本项目 cn.json / core en.json 里。

只读。
"""
import io
import json
import os
import re
import sys

FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
CORE = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_compiled_slot.json")


def flat(o, p=""):
    out = set()
    if isinstance(o, dict):
        for k, v in o.items():
            q = f"{p}.{k}" if p else k
            out.add(q)
            out |= flat(v, q)
    return out


EN = flat(json.load(io.open(os.path.join(FVTT, "systems", "crucible", "lang", "en.json"), encoding="utf-8")))
CN = flat(json.load(io.open(os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"), encoding="utf-8")))
COREK = flat(json.load(io.open(os.path.join(CORE, "public", "lang", "en.json"), encoding="utf-8")))
KNOWN = EN | CN | COREK | {"Sort", "sort"}

KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$-]+)+$")
HUMAN = re.compile("^[A-Z][A-Za-z0-9'\u2019-]*(?:[ /-][A-Za-z0-9'\u2019-]+)*[?!.]?$")
PROPS = ["label", "tooltip", "title", "hint", "adjective", "abbreviation", "shortLabel",
         "group", "placeholder"]
RX = re.compile(r'(?<![\w$.])(' + "|".join(PROPS) + r')\s*:\s*"((?:[^"\\]|\\.)*)"')
# 直接进 localize 家族的字面量
CALL = re.compile(r'(?:game\.i18n\.(?:localize|format|has)|\b_loc|\b_lformat)\s*\(\s*"((?:[^"\\]|\\.)*)"')
# 从 JS 直接写 data-tooltip
DSET = re.compile(r'(?:dataset\.tooltip|setAttribute\(\s*"data-tooltip"\s*,)\s*=?\s*"((?:[^"\\]|\\.)*)"')


def main():
    path = os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs")
    s = io.open(path, encoding="utf-8").read()
    res = {"slot_writes": {}, "direct_localize": {}, "dataset_tooltip": {}}

    def keep(t):
        t = t.strip()
        if not t or len(t) < 3:
            return None
        if t in KNOWN or KEYISH.match(t) or t.startswith("fa-"):
            return None
        if not HUMAN.match(t):
            return None
        return t

    for m in RX.finditer(s):
        t = keep(m.group(2))
        if t:
            res["slot_writes"].setdefault(f"{m.group(1)}|{t}", []).append(s[:m.start()].count("\n") + 1)
    for m in CALL.finditer(s):
        t = keep(m.group(1))
        if t:
            res["direct_localize"].setdefault(t, []).append(s[:m.start()].count("\n") + 1)
    for m in DSET.finditer(s):
        t = keep(m.group(1))
        if t:
            res["dataset_tooltip"].setdefault(t, []).append(s[:m.start()].count("\n") + 1)

    for sec in res:
        print("=" * 90)
        print(sec, len(res[sec]))
        for k, v in sorted(res[sec].items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"   {k[:70]:<72} x{len(v)} first@{v[0]}")
    json.dump(res, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
