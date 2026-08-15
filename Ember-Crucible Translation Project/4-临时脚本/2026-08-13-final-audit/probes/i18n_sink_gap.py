# -*- coding: utf-8 -*-
"""
probe: i18n_sink_gap —— 「上游注入面枚举不全」第三形态的**完整判据**（只读）

i18n_literal_gap.py 只抓了「JS 里直接写 localize("字面量")」这一种。
但 Foundry v14 里**大量字段是被框架隐式 localize 的**，上游只要写个裸英文
字面量，它就会走一遭 i18n 再渲染出来。已在 core 源码逐个确认的隐式 sink：

  S1 ApplicationV2#title          client/applications/api/application.mjs:319-321
                                  `get title() { return _loc(this.options.window.title); }`
                                  → 任何 `window: {title: "裸英文"}` 都过 i18n
  S2 DialogV2 按钮                client/applications/api/dialog.mjs:249 `span.innerText = _loc(label)`
                                  :240 `button.setAttribute("aria-label", _loc(tooltip))`
  S3 通用页脚模板                 templates/generic/form-footer.hbs
                                  `{{localize button.label}}` / `{{localize button.tooltip}}`
  S4 data-tooltip                 client/helpers/interaction/tooltip-manager.mjs:216
                                  「automatically localized」
  S5 模板里的 {{localize x}}      x 为变量时，其值在 JS 里被赋裸英文

结论形状：这些串**已经在 i18n 通道上**，只是上游没给它们注册键，
于是 en.json 里没有、cn.json 也就没有 —— 而本项目的 lang 判据是**拿 en.json 当全集**对的，
所以永远看不见。修法极廉价：在 cn.json 里加一条**以英文原串为键**的顶层条目
（本项目自己已经用过这招：2-Crucible汉化插件/babele-register.js 里
 `game.i18n.translations.Sort = 'Sort'` / `.sort = '排序'`）。

假阳性模式：
  - 有些 window.title 是 DEFAULT_OPTIONS 里的开发者用窗口 / 仅 debug 可达；
  - 有些 label 键名同形但并不进这些 sink（例如纯数据字段叫 label）；
  - 顶层键会**全局生效**：英文串若同时是别处一个不该改的字符串，会误伤
    （所以下一轮落地前要对每个候选做一次「全库同串」检查）；
  - 本脚本按**语法邻近**判定 sink，不做数据流分析。

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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_sink_gap.json")

TARGETS = {
    "crucible": {
        "src": [os.path.join(FVTT, "systems", "crucible", "module")],
        "tpl": [os.path.join(FVTT, "systems", "crucible", "templates")],
        "en": os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
        "cn": os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"),
    },
    "ember": {
        "src": [os.path.join(FVTT, "modules", "ember", "scripts")],
        "tpl": [os.path.join(FVTT, "modules", "ember", "templates")],
        "en": os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
        "cn": os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json"),
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

KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$-]+)+$")
# 只看「像给人看的英文」：首字母大写，词间空格/撇号/连字符，允许 ? ! : 结尾
HUMAN = re.compile(r"^[A-Z][A-Za-z0-9'’\-]*(?:[ /][A-Za-z0-9'’\-]+)*[?!.]?$")

SINKS = [
    # (名字, 正则, 取值组)
    ("S1 window.title", re.compile(r"window\s*:\s*\{[^{}]*?\btitle\s*:\s*([\"'])((?:[^\"'\\]|\\.)*)\1", re.S), 2),
    ("S1 window.title(multiline)", re.compile(r"window\s*:\s*\{(?:[^{}]|\{[^{}]*\})*?\btitle\s*:\s*([\"'`])((?:[^\"'`\\]|\\.)*)\1", re.S), 2),
    ("S2/S3 button.label", re.compile(r"\blabel\s*:\s*([\"'])((?:[^\"'\\]|\\.)*)\1", re.S), 2),
    ("S2 button.tooltip", re.compile(r"\btooltip\s*:\s*([\"'])((?:[^\"'\\]|\\.)*)\1", re.S), 2),
    ("S4 data-tooltip", re.compile(r'data-tooltip\s*=\s*"([^"{}]+)"'), 1),
]


def scan(name, cfg):
    en = flatten(json.load(io.open(cfg["en"], encoding="utf-8")))
    cn = flatten(json.load(io.open(cfg["cn"], encoding="utf-8")))
    known = en | cn | CORE_KEYS
    found = {}

    def add(sink, text, rel, ln):
        t = text.strip()
        if not t or KEYISH.match(t) or t in known:
            return
        if not HUMAN.match(t):
            return
        if re.match(r"^(fa-|fa\b)", t):
            return
        found.setdefault(t, []).append((sink, rel, ln))

    files = []
    for d in cfg["src"]:
        for base, _dn, fns in os.walk(d):
            for fn in fns:
                if fn.endswith((".mjs", ".js")):
                    files.append(os.path.join(base, fn))
    for d in cfg["tpl"]:
        for base, _dn, fns in os.walk(d):
            for fn in fns:
                if fn.endswith((".hbs", ".html")):
                    files.append(os.path.join(base, fn))

    for path in files:
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for sink, rx, gi in SINKS:
            for m in rx.finditer(s):
                add(sink, m.group(gi), rel, s[:m.start()].count("\n") + 1)
    return found


def main():
    result = {}
    for name, cfg in TARGETS.items():
        f = scan(name, cfg)
        result[name] = {k: v for k, v in sorted(f.items())}
        print("=" * 96)
        print(f"[{name}]  裸英文进 i18n sink、en/cn/core 三张表都无键的候选：{len(f)} 个不同串")
        for k, v in sorted(f.items(), key=lambda kv: -len(kv[1])):
            sinks = sorted({s for s, _r, _l in v})
            head = v[0]
            print(f"  {k[:58]!r:<62} x{len(v):<3} {','.join(x.split()[0] for x in sinks):<12} {head[1]}:{head[2]}")
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
