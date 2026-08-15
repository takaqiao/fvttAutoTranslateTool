# -*- coding: utf-8 -*-
"""
probe: i18n_js_tooltip_sink_gap —— 「上游注入面枚举不全」在 **JS 侧 data-tooltip** 上的形态（只读）

已有 i18n_sink_gap.py 的 S4 只匹配 HTML 属性写法 `data-tooltip="..."`：
    ("S4 data-tooltip", re.compile(r'data-tooltip\\s*=\\s*"([^"{}]+)"'), 1)
上游更常用的是 **JS 侧属性赋值**，正则里根本没有 `data-tooltip` 这个字面：
    el.dataset.tooltip = "Add Hex";
    el.setAttribute("data-tooltip", "Add Hex");
两种写法产生完全相同的 DOM，却一个被扫到、一个扫不到。

核心侧 sink（v14.348，已读源码确认）
-----------------------------------
  client/helpers/interaction/tooltip-manager.mjs:262-264
      text = element.dataset.tooltip || element.ariaLabel;
      if ( game.i18n.has(text) ) this.tooltip.innerHTML = _loc(text);
      else this.tooltip.innerHTML = foundry.utils.cleanHTML(text);
  → data-tooltip 的值**先过 game.i18n.has()**，命中就走 localize。
    也就是说：在 cn.json 里加一条以英文原串为键的顶层条目，这些 tooltip 立刻变中文；
    不加，就永远是英文，且 en.json 里没有这个键 → 任何按 en.json 对表的判据都看不见。

判据
----
  在上游 JS 里抓 `X.dataset.tooltip = <字面量>` 和 `setAttribute("data-tooltip", <字面量>)`，
  剔除点号键、剔除已在 en/cn/core 键集里的，剩下的即为无键裸英文。
  同时把**模板 .hbs 里的同名属性**一并列出（标记为 already-covered），
  以便和 i18n_sink_gap 的结果对齐、避免重复上报。

假阳性模式
----------
  - 模板字符串（含 ${}）的值是运行期拼的，不在本判据内（单列 dynamic）
  - `dataset.tooltipText` / `dataset.tooltipHtml` **不过 i18n**（同文件 :257-259 / :250），
    属于另一类（完全不在通道上），本脚本单列不计入
  - 有些赋值点在 GM-only / 开发者路径上，需回源判可达性

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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_js_tooltip_sink_gap.json")

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

JS_SINKS = [
    ("dataset.tooltip", re.compile(r"\.dataset\.tooltip\s*=\s*([\"'`])((?:[^\"'`\\]|\\.)*)\1")),
    ("setAttribute", re.compile(r"setAttribute\(\s*[\"']data-tooltip[\"']\s*,\s*([\"'`])((?:[^\"'`\\]|\\.)*)\1")),
]
JS_OFFCHANNEL = [
    ("dataset.tooltipText", re.compile(r"\.dataset\.tooltipText\s*=\s*([\"'`])((?:[^\"'`\\]|\\.)*)\1")),
    ("dataset.tooltipHtml", re.compile(r"\.dataset\.tooltipHtml\s*=\s*([\"'`])((?:[^\"'`\\]|\\.)*)\1")),
]
HBS_SINK = re.compile(r'data-tooltip\s*=\s*"([^"{}]+)"')
KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$\[\]-]+)+$")


def flatten(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.add(p)
            out |= flatten(v, p)
    return out


CORE_KEYS = flatten(json.load(io.open(os.path.join(CORE, "public", "lang", "en.json"), encoding="utf-8")))


def walk(dirs, exts):
    for d in dirs:
        if os.path.isfile(d):
            yield d
        elif os.path.isdir(d):
            for base, _dn, fns in os.walk(d):
                for fn in fns:
                    if os.path.splitext(fn)[1] in exts:
                        yield os.path.join(base, fn)


def scan(cfg):
    known = flatten(json.load(io.open(cfg["en"], encoding="utf-8")))
    known |= flatten(json.load(io.open(cfg["cn"], encoding="utf-8")))
    known |= CORE_KEYS
    for p in cfg["cnjs"]:
        if os.path.exists(p):
            s = io.open(p, encoding="utf-8", errors="replace").read()
            for m in re.finditer(r"game\.i18n\.translations(?:\.([\w$]+)|\[[\"']([^\"']+)[\"']\])", s):
                known.add(m.group(1) or m.group(2))
            for m in re.finditer(r"[\"']((?:[^\"'\\]|\\.){4,})[\"']\s*:", s):
                known.add(m.group(1))

    res = {"js_gap": [], "js_dynamic": [], "js_offchannel": [], "hbs_already_covered": []}
    for path in walk(cfg["src"], {".mjs", ".js"}):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for tag, rx in JS_SINKS:
            for m in rx.finditer(s):
                t = m.group(2)
                ln = s[:m.start()].count("\n") + 1
                if "${" in t:
                    res["js_dynamic"].append({"how": tag, "text": t, "file": rel, "line": ln})
                    continue
                if KEYISH.match(t.strip()) or t.strip() in known:
                    continue
                if not re.search(r"[A-Za-z]{3,}", t):
                    continue
                res["js_gap"].append({"how": tag, "text": t, "file": rel, "line": ln})
        for tag, rx in JS_OFFCHANNEL:
            for m in rx.finditer(s):
                res["js_offchannel"].append({"how": tag, "text": m.group(2), "file": rel,
                                             "line": s[:m.start()].count("\n") + 1})
    for path in walk(cfg["tpl"], {".hbs", ".html"}):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for m in HBS_SINK.finditer(s):
            t = m.group(1).strip()
            if KEYISH.match(t) or t in known or not re.search(r"[A-Za-z]{3,}", t):
                continue
            res["hbs_already_covered"].append({"text": t, "file": rel,
                                               "line": s[:m.start()].count("\n") + 1})
    return res


def main():
    out = {}
    for name, cfg in TARGETS.items():
        r = scan(cfg)
        out[name] = r
        print("=" * 100)
        print(f"[{name}] JS 侧 data-tooltip 无键裸英文（本探针新增覆盖）：{len(r['js_gap'])} 处")
        for x in r["js_gap"]:
            print(f"  {x['file']}:{x['line']:<7} {x['how']:<15} {x['text']!r}")
        print(f"  -- 动态拼串（不判）：{len(r['js_dynamic'])}；"
              f"tooltipText/Html 不过 i18n（另一类）：{len(r['js_offchannel'])}；"
              f"hbs 属性写法（i18n_sink_gap 已覆盖）：{len(r['hbs_already_covered'])}")
        for x in r["hbs_already_covered"][:20]:
            print(f"     [已覆盖] {x['file']}:{x['line']} {x['text']!r}")
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
