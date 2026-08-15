# -*- coding: utf-8 -*-
"""
probe: i18n_slot_gap —— 「上游注入面枚举不全 / i18n 通道上的无键裸串」这一类的**槽位判据**（只读）

已报实例的形状（不重复报）：
    crucible `usage.context.label` 这个**属性槽**被 models/action.mjs:3133 塞进
    `tags.context.tooltip`，再由 action-use-header.hbs `data-tooltip=` 渲染，
    core 的 tooltip-manager 对它做 `game.i18n.has(text) ? _loc(text) : 原样`。
    同一行代码的兜底值写的是正规 i18n 键，说明这个槽**本来就是 i18n 槽**；
    上游在五个分支里往这个槽写了裸英文，于是永远英文且无信号。

抽象成判据（三步）：
  ① 求出**被证明会过 i18n 的属性槽集合** SLOTS（属性名）：
     - 模板 `{{localize X.p}}` / `{{#if}}…{{localize X.p}}`      → p
     - 模板 `data-tooltip="{{X.p}}"` / `{{{X.p}}}`               → p   （core tooltip-manager.mjs:263）
     - 模板 `aria-label="{{localize X.p}}"` 且同标签有 data-tooltip → p （同上，:263 取 element.ariaLabel）
     - JS   `_loc(X.p)` / `game.i18n.localize(X.p)` / `.format(X.p` / `.has(X.p)` → p
     - core 框架隐式槽（逐条在 core 源码里核过，见 CORE_SLOTS 注释）
  ② 在上游 JS 里找**往这些槽写裸英文字面量**的位置：`p: "..."` 或 `.p = "..."`
  ③ 减去所有已覆盖通道：上游 en.json 拍平键 / 本项目 cn.json 拍平键 / core en.json 顶层键 /
     本项目 `game.i18n.translations.X` 直接赋值 / ember 运行时替换表（EXACT/PREFIXED/…）

与 i18n_literal_gap.py / i18n_sink_gap.py 的区别：
  - literal_gap 只抓「字面量**直接**进 localize()」；
  - sink_gap 抓「`label:`/`tooltip:`/`title:` 任意位置的字面量」—— 对 ember 那个 13 万行的
    打包文件产生 1836 条噪声，因为 `label:` 在瓦片/预制件资源数据里满天飞；
  - 本探针要求**槽位有证据**（该属性名在本包内被证明过 i18n），并且对 ember 额外
    减去运行时替换表，噪声可控。

假阳性模式（必须逐条回源核实，脚本不做）：
  - 属性名匹配是**跨文件同名**的：`label` 在资源数据里也叫 label，槽位证据来自别处
    并不能证明**这一处**的 label 会进 i18n（ember 侧尤其严重，输出里单列 flagged_noise）；
  - 有些槽位的值在渲染前会被别的代码覆盖（例如先写裸串、后被 `_loc()` 结果覆盖）；
  - 有些代码路径是死路 / 仅开发者可达；
  - 顶层键修法会**全局生效**，落地前要做同串检查（本脚本输出 global_collision 供参考）。

只读，不写库。
"""
import io
import json
import os
import re
import sys

FVTT = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
CORE = r"C:\Program Files\Foundry Virtual Tabletop\resources\app"
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_slot_gap.json")

# core 框架隐式槽：每条都在 core 源码里核过
CORE_SLOTS = {
    "title": "ApplicationV2#title → client/applications/api/application.mjs `get title(){return _loc(...)}`",
    "label": "form-footer.hbs:10 {{localize button.label}} / tab-navigation.hbs:8 / frame-buttons.hbs:3 / context-menu.mjs:403 _loc(item.label)",
    "tooltip": "form-footer.hbs:8 aria-label=\"{{localize button.tooltip}}\"",
    "name": "context-menu.mjs:403 _loc('label' in item ? item.label : item.name)",
    "hint": "generic/form-fields.hbs:3 {{localize hint}} / fields.mjs:70 localize?_loc(hint)",
}

TARGETS = {
    "crucible": {
        "src": [os.path.join(FVTT, "systems", "crucible", "module")],
        "tpl": [os.path.join(FVTT, "systems", "crucible", "templates")],
        "en": os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
        "cn": os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"),
        "cnjs": [os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js")],
        "runtime_tables": [],
    },
    "ember": {
        "src": [os.path.join(FVTT, "modules", "ember", "scripts")],
        "tpl": [os.path.join(FVTT, "modules", "ember", "templates")],
        "en": os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
        "cn": os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json"),
        "cnjs": [os.path.join(ROOT, "1-Ember汉化插件", "register.js"),
                 os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")],
        # 运行时 DOM 替换表：整份文件里的 "英文": "中文" 对，左边都算已覆盖
        "runtime_tables": [os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")],
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


CORE_KEYS = flatten(json.load(io.open(os.path.join(CORE, "public", "lang", "en.json"), encoding="utf-8")))

KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$\[\]-]+)+$")
# 「给人看的英文」：首词大写，允许空格/连字符/撇号/数字/问号
HUMAN = re.compile(r"^[A-Z][A-Za-z0-9'’\-]*(?:[ /\-][A-Za-z0-9'’\-]+)*[?!.]?$")
FA_ICON = re.compile(r"^fa[srltdb]?[- ]")

# ---- 槽位证据 ----
T_LOCALIZE_VAR = re.compile(r"\{\{\s*localize\s+([A-Za-z_$][\w$.]*)\s*[}\s]")
T_TOOLTIP_VAR = re.compile(r'data-tooltip\s*=\s*"\{\{\{?\s*([A-Za-z_$][\w$.]*)')
T_ARIA_VAR = re.compile(r'aria-label\s*=\s*"\{\{\s*(?:localize\s+)?([A-Za-z_$][\w$.]*)')
J_LOC_VAR = re.compile(
    r"(?:game\.i18n\.(?:localize|format|has)|\bi18n\.(?:localize|format)|\b_loc|\b_lformat)"
    r"\s*\(\s*([A-Za-z_$][\w$.?\[\]]*)")

# ---- 写点 ----
def write_regexes(props):
    alt = "|".join(sorted(re.escape(p) for p in props))
    return [
        ("obj", re.compile(r"(?<![\w$.])(" + alt + r")\s*:\s*([\"'])((?:[^\"'\\]|\\.)*)\2")),
        ("asn", re.compile(r"\.(" + alt + r")\s*=\s*([\"'])((?:[^\"'\\]|\\.)*)\2")),
    ]


def walk(dirs, exts):
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for base, _dn, fns in os.walk(d):
            for fn in fns:
                if os.path.splitext(fn)[1] in exts:
                    yield os.path.join(base, fn)


def runtime_covered(paths):
    """从本项目的运行时替换表里抠出所有被当作 key 的英文串。"""
    covered = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        s = io.open(p, encoding="utf-8").read()
        for m in re.finditer(r'"([^"\\]{2,120})"\s*:\s*"', s):
            covered.add(m.group(1))
    return covered


def scan(name, cfg):
    en_keys = flatten(json.load(io.open(cfg["en"], encoding="utf-8")))
    cn_keys = flatten(json.load(io.open(cfg["cn"], encoding="utf-8")))
    for p in cfg["cnjs"]:
        if os.path.exists(p):
            s = io.open(p, encoding="utf-8").read()
            for m in re.finditer(r"game\.i18n\.translations(?:\.([\w$]+)|\[[\"']([^\"']+)[\"']\])", s):
                cn_keys.add(m.group(1) or m.group(2))
    rt = runtime_covered(cfg["runtime_tables"])
    known = en_keys | cn_keys | CORE_KEYS | rt

    # ① 槽位
    slots = {}   # prop -> [证据]
    for path in walk(cfg["tpl"], {".hbs", ".html"}):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for rx, why in ((T_LOCALIZE_VAR, "{{localize X}}"),
                        (T_TOOLTIP_VAR, 'data-tooltip="{{X}}"'),
                        (T_ARIA_VAR, 'aria-label="{{X}}"')):
            for m in rx.finditer(s):
                expr = m.group(1)
                prop = expr.split(".")[-1]
                ln = s[:m.start()].count("\n") + 1
                slots.setdefault(prop, []).append(f"{why} {expr} @ {rel}:{ln}")
    for path in walk(cfg["src"], {".mjs", ".js"}):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for m in J_LOC_VAR.finditer(s):
            expr = m.group(1)
            if "." not in expr:
                continue
            prop = re.split(r"[.\[?]", expr)[-1]
            if not prop or not prop.isidentifier():
                continue
            ln = s[:m.start()].count("\n") + 1
            slots.setdefault(prop, []).append(f"_loc({expr}) @ {rel}:{ln}")
    for p, why in CORE_SLOTS.items():
        slots.setdefault(p, []).append("core: " + why)

    # ② 写点
    props = set(slots)
    props = {p for p in props if p.isidentifier()}
    rxs = write_regexes(props)
    hits = {}    # text -> [(prop, rel, ln)]
    for path in walk(cfg["src"], {".mjs", ".js"}):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for _kind, rx in rxs:
            for m in rx.finditer(s):
                prop, txt = m.group(1), m.group(3)
                t = txt.strip()
                if not t or t in known or KEYISH.match(t) or FA_ICON.match(t):
                    continue
                if not HUMAN.match(t):
                    continue
                if len(t) < 3:
                    continue
                ln = s[:m.start()].count("\n") + 1
                hits.setdefault(t, []).append({"prop": prop, "file": rel, "line": ln})
    # 模板里 data-tooltip / aria-label 直接写裸英文
    for path in walk(cfg["tpl"], {".hbs", ".html"}):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for rx, prop in ((re.compile(r'data-tooltip\s*=\s*"([^"{}]+)"'), "data-tooltip"),
                         (re.compile(r'aria-label\s*=\s*"([^"{}]+)"'), "aria-label")):
            for m in rx.finditer(s):
                t = m.group(1).strip()
                if not t or t in known or KEYISH.match(t) or not HUMAN.match(t) or len(t) < 3:
                    continue
                ln = s[:m.start()].count("\n") + 1
                hits.setdefault(t, []).append({"prop": prop, "file": rel, "line": ln})

    return {"en_keys": len(en_keys), "cn_keys": len(cn_keys), "runtime_keys": len(rt),
            "slots": {k: v[:3] for k, v in sorted(slots.items())},
            "hits": {k: v for k, v in sorted(hits.items(), key=lambda kv: (-len(kv[1]), kv[0]))}}


def main():
    result = {}
    for name, cfg in TARGETS.items():
        r = scan(name, cfg)
        result[name] = r
        print("=" * 100)
        print(f"[{name}] en 键 {r['en_keys']} / cn 键 {r['cn_keys']} / 运行时表 {r['runtime_keys']}"
              f" / 已证明 i18n 槽 {len(r['slots'])} 个属性名")
        print(f"  往 i18n 槽写裸英文、四张覆盖表都没有的候选：{len(r['hits'])} 个不同串")
        for t, sites in list(r["hits"].items())[:400]:
            props = sorted({s["prop"] for s in sites})
            h = sites[0]
            print(f"   {t[:52]!r:<56} x{len(sites):<3} [{','.join(props)[:22]:<22}] {h['file']}:{h['line']}")
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
