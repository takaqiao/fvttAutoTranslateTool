# -*- coding: utf-8 -*-
"""
probe: i18n_literal_gap  —— 「上游注入面枚举不全」这一类的第三个形态（只读）
=========================================================================

已确认实例的抽象形状：
    本模块（或本项目的汉化模型）对**上游的某个面**做了闭集枚举，
    上游实际写出的面更大，差集永远英文且**没有任何信号**。

已报过的两个面：
    ① crucible.CONFIG.{languages,knowledge}  ←→ ember 还写了 crucible.CONFIG 别的键
    ② crucible.CONFIG.*                      ←→ ember 还写了 core CONFIG.*

本探针查**第三个面：i18n 通道本身**。

    本项目的 lang/cn.json 是**按上游 lang/en.json 的键清单**逐键翻的
    （既有判据「lang 四项且拍平三数相等」正是拿 en.json 当全集来对的）。
    但上游代码并不只把 en.json 里声明过的键喂给 i18n ——
    它还把**裸英文字面量**直接喂给 `game.i18n.localize()` / `{{localize x}}`。
    Foundry 的 localize 查不到键就原样返回，所以这些串渲染成英文，
    而且因为「en.json 里没有这个键」，任何按 en.json 对表的 lang 判据都**永远看不见它**。

    关键：这些串**本来就在 i18n 通道上**，加一条顶层键就能译
    （本项目自己已经用过这招：2-Crucible汉化插件/babele-register.js 的
     `game.i18n.translations.Sort = 'Sort'` / `.sort = '排序'`）。

判据（三步，全机械）：
    A. 从上游 JS 抓所有直接传给 localize 家族函数的**字符串字面量**
       （game.i18n.localize / game.i18n.format / _loc / _lformat / 以及
        `ui.notifications.x("...", {localize:true})`）；
    B. 从上游 .hbs 抓 `{{localize "字面量"}}` / `{{#localize}}`；
    C. 抓「JS 写 `label: "裸英文"` → 模板 `{{localize xxx.label}}`」这条**间接**路径：
       先在模板里找出所有 `{{localize <非字面量>}}` 的字段名（如 boon.label），
       再在 JS 里找同名字段被赋裸英文的位置。
    然后对每个串问：它在上游 en.json 的**拍平键集**里吗？在本项目 cn.json 里吗？
    两个都不在 = 永远英文。

假阳性模式（必须逐条人工核实，本脚本不做）：
    - 有些串虽然进了 localize，但那段代码是死路 / 仅开发者可达；
    - `{{localize x}}` 的 x 在别处可能被赋的是真键（同一字段两种来源）；
    - 有些「裸英文」其实是刻意保留的记号（DC / AC / 缩写）；
    - C 步按**字段名**匹配，跨文件同名字段会误连，需回源确认渲染点。

只读，不写库。
"""
import io
import json
import os
import re
import sys

FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_literal_gap.json")

TARGETS = {
    "crucible": {
        "src": [os.path.join(FVTT, "systems", "crucible", "module")],
        "tpl": [os.path.join(FVTT, "systems", "crucible", "templates")],
        "en": os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
        "cn": os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"),
        "cnjs": [os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js")],
    },
    "ember": {
        "src": [os.path.join(FVTT, "modules", "ember", "scripts")],
        "tpl": [os.path.join(FVTT, "modules", "ember", "templates")],
        "en": os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
        "cn": os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json"),
        "cnjs": [os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
                 os.path.join(ROOT, "1-Ember汉化插件", "register.js")],
    },
}


def flatten(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.add(p)
            out |= flatten(v, p)
    return out


def walk(dirs, exts):
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for base, _dn, fns in os.walk(d):
            for fn in fns:
                if os.path.splitext(fn)[1] in exts:
                    yield os.path.join(base, fn)


# localize 家族：直接吃字面量的调用点
CALL = re.compile(
    r"(?:game\.i18n\.(?:localize|format|has)|\bi18n\.(?:localize|format)"
    r"|\b_loc|\b_lformat|\blocalize)\s*\(\s*([\"'`])((?:[^\"'`\\]|\\.)*)\1"
)
# ui.notifications.x("...", {localize: true})
NOTIF_LOC = re.compile(
    r"ui\.notifications\.\w+\(\s*([\"'`])((?:[^\"'`\\]|\\.)*)\1\s*,\s*\{[^}]*localize\s*:\s*true"
)
# 模板里的字面量
HBS_LIT = re.compile(r"\{\{\s*localize\s+([\"'])((?:[^\"'\\]|\\.)*)\1")
# 模板里的变量（间接路径）
HBS_VAR = re.compile(r"\{\{\s*localize\s+([A-Za-z_$][\w$.]*)\s*[}\s]")

KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$\[\]-]+)+$")   # 形如 A.B.C 的键
BARE_EN = re.compile(r"^[A-Z][A-Za-z0-9'’\- ]*$")


def scan(name, cfg):
    en_keys = flatten(json.load(io.open(cfg["en"], encoding="utf-8")))
    cn_keys = flatten(json.load(io.open(cfg["cn"], encoding="utf-8")))
    # 汉化侧 JS 里 game.i18n.translations.X = 的直接赋值也算覆盖
    for p in cfg["cnjs"]:
        if os.path.exists(p):
            s = io.open(p, encoding="utf-8").read()
            for m in re.finditer(r"game\.i18n\.translations(?:\.([\w$]+)|\[[\"']([^\"']+)[\"']\])", s):
                cn_keys.add(m.group(1) or m.group(2))

    direct = {}     # text -> [(file, line, kind)]
    for path in walk(cfg["src"], {".mjs", ".js"}):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for rx, kind in ((CALL, "js-localize"), (NOTIF_LOC, "js-notification")):
            for m in rx.finditer(s):
                txt = m.group(2)
                ln = s[:m.start()].count("\n") + 1
                direct.setdefault(txt, []).append((rel, ln, kind))
    for path in walk(cfg["tpl"], {".hbs", ".html"}):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for m in HBS_LIT.finditer(s):
            txt = m.group(2)
            ln = s[:m.start()].count("\n") + 1
            direct.setdefault(txt, []).append((rel, ln, "hbs-literal"))

    # 差集：进了 i18n 通道，但既不在 en.json 也不在 cn.json
    gaps = []
    for txt, sites in sorted(direct.items()):
        t = txt.strip()
        if not t:
            continue
        if t in en_keys or t in cn_keys:
            continue
        if KEYISH.match(t):
            continue                      # 形如 A.B.C：上游漏声明的键，另一类问题
        if not re.search(r"[A-Za-z]{2,}", t):
            continue
        gaps.append({"text": txt, "sites": sites})

    # 间接路径：模板里 {{localize 变量}} 的字段名
    varfields = {}
    for path in walk(cfg["tpl"], {".hbs", ".html"}):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for m in HBS_VAR.finditer(s):
            expr = m.group(1)
            ln = s[:m.start()].count("\n") + 1
            varfields.setdefault(expr, []).append((rel, ln))

    return {"en_keys": len(en_keys), "cn_keys": len(cn_keys),
            "direct_literals": len(direct), "gaps": gaps,
            "hbs_localize_vars": {k: v for k, v in sorted(varfields.items())}}


def main():
    result = {}
    for name, cfg in TARGETS.items():
        result[name] = scan(name, cfg)
        r = result[name]
        print("=" * 78)
        print(f"[{name}]  en.json 拍平键 {r['en_keys']} / cn.json 拍平键 {r['cn_keys']}")
        print(f"  直接进 i18n 的字面量共 {r['direct_literals']} 个；其中**无键**（永远渲染成原文）{len(r['gaps'])} 个")
        for g in r["gaps"]:
            head = g["sites"][0]
            print(f"    {g['text'][:70]!r:<74} {head[0]}:{head[1]} ({head[2]}) x{len(g['sites'])}")
        print(f"  模板里 {{{{localize <变量>}}}} 的字段共 {len(r['hbs_localize_vars'])} 个：")
        for k, v in r["hbs_localize_vars"].items():
            print(f"    {k:<34} {len(v)} 处  e.g. {v[0][0]}:{v[0][1]}")
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
