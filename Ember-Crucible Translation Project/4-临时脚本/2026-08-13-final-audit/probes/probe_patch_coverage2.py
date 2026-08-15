# -*- coding: utf-8 -*-
"""
probe_patch_coverage2.py — 「补丁覆盖面不足」判据 v2（只读，不写库）

v1 的假阳性在这里被剔除：
  * data-tooltip="EMBER.XXX" 之类是 i18n 键（TooltipManager 只在 game.i18n.has(text) 时
    才 localize，见 Foundry v14 client/helpers/interaction/tooltip-manager.mjs:261-264），
    走 lang 通道，不属于本判据；
  * {{formField ...}} 跨行片段被 TAG.split 切出来的碎片；
  * 纯枚举值属性（data-tooltip-direction=LEFT）。

并补上 v1 没做的那一维：**谁渲染这个模板**，以及补丁的应用闸放不放行它。
闸的规则（ember-hardcoded-cn.mjs:453）：className 含 ember 或 constructor.name 以 Ember 开头
才 translateNode；否则只在 DialogV2 时取一次 .window-title 就 return。
"""
import json
import os
import re

EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
CN = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"

src = open(CN, encoding="utf-8").read()
exact_block = src[src.index("const EXACT = {"): src.index("/** 掷骰结果档位")]
EXACT_KEYS = set(re.findall(r'^\s*"([^"]+)":', exact_block, re.M))
PREFIXES = set(re.findall(r'\{\s*en:\s*"([^"]+)"', src))
ATTRS = set(re.findall(r'"([^"]+)"', re.search(r'for \(const attr of \[([^\]]+)\]\)', src).group(1)))

# ---- 模板 -> 渲染它的类 ----
# 逐文件（不能把三个 .mjs 拼起来，偏移会串味 —— v2 初版就栽在这里，
# tab-attunement.hbs 被归给了 DetectionModeThermalVision）。
# 取模板路径出现处往上最近的 `class Xxx extends`；若那一处是 `cls.PARTS.x = {...}`
# 形式（把模板塞进**别的**类），最近类名就不是真正的宿主，标 `注入:` 交人工判。
JS_FILES = ["ember.mjs", "crucible-async.mjs", "dnd5e-async.mjs"]
SOURCES = {f: open(os.path.join(EMBER, "scripts", f), encoding="utf-8").read() for f in JS_FILES}
CLASS_RE = re.compile(r"^class ([A-Za-z0-9_$]+) extends", re.M)
CLASS_AT = {f: [(m.start(), m.group(1)) for m in CLASS_RE.finditer(s)] for f, s in SOURCES.items()}


def owner_of(tpl):
    owners = set()
    for f, s in SOURCES.items():
        for m in re.finditer(re.escape(tpl), s):
            i = m.start()
            ctx = s[max(0, i - 400):i]
            prev = [c for p, c in CLASS_AT[f] if p < i]
            host = prev[-1] if prev else "?"
            if re.search(r"cls\.PARTS\.\w+\s*=", ctx):
                host = "注入到外部类(" + host + ")"
            owners.add(host)
    return owners


def gate(owners):
    """补丁的应用闸是否放行（constructor.name 以 Ember 开头即放行）"""
    if not owners:
        return "未定位"
    return "放行" if all(o.startswith("Ember") for o in owners) else "拦截:" + ",".join(sorted(owners))


TAG = re.compile(r"<[^>]+>")
HB = re.compile(r"\{\{[^}]*\}\}")
ATTR_RE = re.compile(r'\b(aria-label|placeholder|title|alt|data-tooltip|data-tooltip-text)\s*=\s*"([^"]*)"')
ENGLISH = re.compile(r"[A-Za-z]{3,}")
I18NKEY = re.compile(r"^[A-Z][A-Z0-9_]*(\.[A-Za-z0-9_]+)+$")

rows = []
for dirpath, _d, files in os.walk(os.path.join(EMBER, "templates")):
    for f in sorted(files):
        if not f.endswith(".hbs"):
            continue
        full = os.path.join(dirpath, f)
        rel = os.path.relpath(full, EMBER).replace("\\", "/")
        tpl = "modules/ember/" + rel
        text = open(full, encoding="utf-8").read()
        owners = owner_of(tpl)
        g = gate(owners)

        def add(kind, raw):
            v = HB.sub("", raw).strip().rstrip(":").strip()
            if len(v) < 3 or not ENGLISH.search(v):
                return
            if I18NKEY.match(v):
                return                                   # i18n 键，走 lang 通道
            if re.match(r"^(value|placeholder|localize|input|stacked)=", v):
                return                                   # handlebars 续行碎片
            if v.upper() == v and len(v) <= 6:
                return
            rows.append({"tpl": rel, "owner": "/".join(sorted(owners)) or "?", "gate": g,
                         "kind": kind, "text": v, "inEXACT": v in EXACT_KEYS,
                         "prefixCovered": any(v.startswith(p + ":") for p in PREFIXES)})

        for attr, val in ATTR_RE.findall(text):
            if "{{" in val and HB.sub("", val).strip() in ("", ":"):
                continue
            add("attr:" + attr, val)
        for chunk in TAG.split(text):
            for line in chunk.splitlines():
                add("text", line)

bad = [r for r in rows if not r["inEXACT"] and not r["prefixCovered"]]
print("模板总命中(去 i18n 键/碎片后):", len(rows), " 其中补丁翻不了:", len(bad))
print()
for g in sorted({r["gate"] for r in bad}):
    sub = [r for r in bad if r["gate"] == g]
    print(f"### 闸={g}  ({len(sub)} 条)")
    for r in sub:
        print(f"   {r['tpl']:<52} [{r['kind']}] {r['text']}")
    print()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_coverage_v2.json")
json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("raw ->", out)
