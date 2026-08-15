# -*- coding: utf-8 -*-
r"""
probe: i18n_notification_sink_gap —— 「上游注入面枚举不全：i18n 通道上的无键裸串」
                                     在 **ui.notifications 第一实参** 上的形态（只读）

为什么这是同一类
----------------
已确认实例（@Spell 的 data-tooltip）的形状是：
    上游把一个**没有键的裸英文串**写进了一个 core 会替你跑 localize 的位置；
    en.json 里没有这个键 → 所有「拿 en.json 当全集」的 lang 判据看不见；
    babele 够不到（不是文档字段）；
    但**加一条以英文原串为键的顶层条目就能翻**。

本探针查的位置是 `ui.notifications.info/warn/error/success(message, ...)` 的第一个实参。

core 侧证据（v14.348，已读源码逐行确认）
---------------------------------------
  client/applications/ui/notifications.mjs:108-121
      notify(message, type="info", {localize=false, ..., format}={}) {
        const error = message instanceof Error ? message : null;
        message = String(message);
        if ( format ) { ... if ( game.i18n.has(message) ) clean = false; }
        else if ( localize ) { if ( game.i18n.has(message) ) clean = false; }
        message = _loc(message, format);        // <<< 无条件，不看 localize 选项
  → v14 里 `localize:true` **只决定要不要 cleanHTML**，不决定要不要本地化。
    `_loc()` 是无条件跑的。因此**每一条通知文本都已经站在 i18n 通道上**。
  client/helpers/localization.mjs:435-447  localize() 用 getProperty(this.translations, stringId)
  common/utils/helpers.mjs  getProperty 首行 `if ( key in object ) return object[key];`
      → **带空格、带句点的整句英文可以直接当顶层扁平键**，点号不会被切成路径。
  同文件 :161-166  info/warn/error/success 全部转调 notify()。

判据
----
  在上游**实际加载的** JS（crucible: crucible-compiled.mjs；ember: scripts/ember.mjs）里
  抓 `ui.notifications.X(<第一实参>)`，按实参形态分三桶：
    static   —— 纯字符串字面量，且不在 en/cn/core 键集里 → **本类缺陷候选**（加键即可修）
    dynamic  —— 模板串含 ${}，运行期拼出来 → 同样在通道上，但加键救不了（单列，不计入）
    indirect —— 传的是变量/表达式（err、err.message、warning…）→ 需回源追字面量来源
  再对 `throw new Error("字面量")` 做一次反向追踪：凡是被 `ui.notifications.*(err.message)`
  之类捞起来的，字面量同样落在同一条无条件 localize 通道上。

假阳性模式（本探针自己会错的地方）
--------------------------------
  - 形如 `A.B.C` 的正规 i18n 键会被 KEYISH 剔除；剔不干净的（例如 "EMBER.X" 带引号拼接）需人工看；
  - 某些调用点只在 dnd5e 分支 / GM 专用 / 开发者路径上可达 —— 本脚本不做可达性分析，需回源；
  - `String(message)` 对 Error 对象取 `"Error: xxx"`，所以 `ui.notifications.error(err)`
    这一支即便加键也对不上（单列在 indirect，不作为候选）；
  - 顶层扁平键是**全局**的：英文串若在别处也出现（例如恰好等于某个 UI 文案），会一起被改。
  - 本脚本按语法邻近取第一实参，遇到嵌套括号用配平切分，极端写法可能切错（会打印原文供核对）。

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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_notification_sink_gap.json")

TARGETS = {
    "crucible": {
        # system.json esmodules == ["crucible-compiled.mjs"] —— 这才是真正加载的那份
        "src": [os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs")],
        "en": os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
        "cn": os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"),
        "tpl": [os.path.join(FVTT, "systems", "crucible", "templates")],
        "cnjs": [os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js")],
    },
    "ember": {
        "src": [os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs")],
        "en": os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
        "cn": os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json"),
        "tpl": [os.path.join(FVTT, "modules", "ember", "templates")],
        "cnjs": [os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
                 os.path.join(ROOT, "1-Ember汉化插件", "register.js")],
    },
}

CALL = re.compile(r"ui\.notifications\.(info|warn|error|success|notify)\s*\(")
# 第二个 sink：**无值的** data-tooltip + aria-label 裸串
#   core v14 tooltip-manager.mjs:139  `if ( (tooltip===undefined) || ((dataset.tooltip==="") && !element.ariaLabel) ) return;`
#        → data-tooltip 写成无值属性（dataset.tooltip===""）且有 aria-label 时**放行**
#   core v14 tooltip-manager.mjs:261-264  `text = element.dataset.tooltip || element.ariaLabel;`
#        → "" 是 falsy，取 ariaLabel，再 `game.i18n.has(text) ? _loc(text) : cleanHTML(text)`
#   已有 i18n_sink_gap.py 的 S4 正则是 `data-tooltip\s*=\s*"([^"{}]+)"`，**要求 data-tooltip 带值**，
#   对 `data-tooltip aria-label="Make Active"` 一个字符都对不上 → 结构性看不见。
ARIA_SINK = re.compile(r"data-tooltip(?![-=\w])[^>]{0,200}?aria-label=\"([^\"{}]{2,})\""
                       r"|aria-label=\"([^\"{}]{2,})\"[^>]{0,200}?data-tooltip(?![-=\w])", re.S)
THROW = re.compile(r"throw new (?:\w*Error)\(\s*([\"'`])((?:[^\"'`\\]|\\.)*)\1\s*\)")
KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(\.[\w$-]+)+$")


def flatten(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.add(p)
            out |= flatten(v, p)
    return out


CORE_KEYS = flatten(json.load(io.open(os.path.join(CORE, "public", "lang", "en.json"), encoding="utf-8")))


def first_arg(s, open_paren):
    """从 '(' 开始，配平括号/引号，切出第一个实参的源码文本。"""
    depth = 0
    i = open_paren
    start = open_paren + 1
    quote = None
    while i < len(s):
        c = s[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'`":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return s[start:i].strip()
        elif c == "," and depth == 1:
            return s[start:i].strip()
        i += 1
    return s[start:start + 200]


STR_LIT = re.compile(r"^([\"'`])((?:[^\"'`\\]|\\.)*)\1$", re.S)


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

    res = {"static_gap": [], "static_keyed": [], "dynamic": [], "indirect": [], "throw_literals": [],
           "aria_tooltip_gap": []}
    for path in cfg.get("tpl", []) + cfg["src"]:
        for base, _dn, fns in ([(path, [], [os.path.basename(path)])] if os.path.isfile(path)
                               else os.walk(path)):
            for fn in fns:
                fp = path if os.path.isfile(path) else os.path.join(base, fn)
                if os.path.splitext(fp)[1] not in {".hbs", ".html", ".mjs", ".js"}:
                    continue
                s = io.open(fp, encoding="utf-8", errors="replace").read()
                for m in ARIA_SINK.finditer(s):
                    t = (m.group(1) or m.group(2)).strip()
                    if KEYISH.match(t) or t in known or "{{" in t:
                        continue
                    res["aria_tooltip_gap"].append({"text": t, "file": os.path.relpath(fp, FVTT),
                                                    "line": s[:m.start()].count("\n") + 1})
    for path in cfg["src"]:
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for m in CALL.finditer(s):
            arg = first_arg(s, m.end() - 1)
            ln = s[:m.start()].count("\n") + 1
            rec = {"how": m.group(1), "arg": arg[:300], "file": rel, "line": ln}
            lit = STR_LIT.match(arg)
            if lit:
                text = lit.group(2)
                if "${" in text:
                    res["dynamic"].append(rec | {"text": text})
                elif KEYISH.match(text.strip()) or text.strip() in known:
                    res["static_keyed"].append(rec | {"text": text})
                else:
                    res["static_gap"].append(rec | {"text": text})
            elif arg.startswith("`"):
                res["dynamic"].append(rec)
            else:
                res["indirect"].append(rec)
        for m in THROW.finditer(s):
            t = m.group(2)
            if "${" in t or KEYISH.match(t.strip()) or t.strip() in known:
                continue
            res["throw_literals"].append({"text": t, "file": rel, "line": s[:m.start()].count("\n") + 1})
    return res


def main():
    out = {}
    for name, cfg in TARGETS.items():
        r = scan(cfg)
        out[name] = r
        print("=" * 110)
        print(f"[{name}] ui.notifications 第一实参 = **无键裸英文静态串**（本类缺陷候选）：{len(r['static_gap'])} 处")
        for x in r["static_gap"]:
            print(f"  {x['file']}:{x['line']:<7} .{x['how']:<8} {x['text'][:110]!r}")
        print(f"  -- 已是正规键/已有键：{len(r['static_keyed'])}；"
              f"模板串动态拼（在通道上但加键救不了）：{len(r['dynamic'])}；"
              f"传变量/表达式（需回源）：{len(r['indirect'])}；"
              f"throw new Error 裸串（经 err.message 转手同样落本通道）：{len(r['throw_literals'])}")
        print(f"  [第二 sink] 无值 data-tooltip + aria-label 裸串（同一 tooltip i18n 通道）：{len(r['aria_tooltip_gap'])} 处")
        for x in r["aria_tooltip_gap"]:
            print(f"     {x['file']}:{x['line']:<7} {x['text']!r}")
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
