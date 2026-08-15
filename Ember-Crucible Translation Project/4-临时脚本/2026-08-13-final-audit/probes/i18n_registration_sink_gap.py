# -*- coding: utf-8 -*-
"""
probe: i18n_registration_sink_gap —— 「上游注入面枚举不全」在**注册型 API** 上的形态（只读）

与已有探针的分工
----------------
  i18n_literal_gap.py  抓的是 `game.i18n.localize("字面量")` / `{{localize "字面量"}}`
                       以及「模板 {{localize x.label}} ←→ JS label: '裸英文'」
  i18n_sink_gap.py     抓的是 window.title / label: / tooltip: / data-tooltip
  upstream_surface_gap.py 抓的是 ember 往 CONFIG.* 注入的 label

三者都**没有扫 `name:` / `hint:` / `buttonText:` / `blank:` / choices 值**，
而 Foundry v14 里有一整族**注册型 API**：调用方交给核心一个配置对象，
核心在渲染时对其中某几个字段跑 `game.i18n.localize()`。
这些字段名恰好是 `name` / `hint`，不是 `label`，所以整族被漏掉了。

核心侧 sink 的逐条出处（v14.348，均已读源码确认）
-----------------------------------------------
  R1 世界设置项      client/applications/settings/config.mjs:126-127
                     `data.field.label ||= _loc(setting.name ?? "")`
                     `data.field.hint  ||= _loc(setting.hint ?? "")`
  R2 设置子菜单      templates/settings/config-category.hbs
                     `{{localize entry.label}}` / `{{localize entry.buttonText}}` /
                     `{{localize entry.hint}}`，其值来自 config.mjs:63-66
                     （label←menu.name / buttonText←menu.label / hint←menu.hint）
  R3 键位绑定        client/applications/sidebar/apps/controls-config.mjs:154,158
                     `label: _loc(action.name)` / `_loc(action.hint)`
  R4 选择框选项      client/applications/forms/fields.mjs:313 `if (config.localize) label = _loc(label)`
                     :331 group、:338 groupName、:351 blank
                     设置页固定传 localize=true（config-category.hbs 里
                     `{{formGroup entry.field ... localize=true}}`）
  R5 右键菜单项      client/applications/ux/context-menu.mjs:403
                     `_loc("label" in item ? item.label : item.name)`
  R6 状态效果名      client/documents/active-effect.mjs:131 `effectData.name = _loc(effectData.name)`

判据
----
  在上游 JS 里定位这些注册调用/配置对象，取出上述字段位置上的**字符串字面量**，
  剔除：① 形如 `A.B.C` 的点号键（那是另一类：键漏声明，已由 i18n_undeclared_key 覆盖）
        ② 已在 crucible/ember en.json、本项目 cn.json、core en.json 拍平键集里的
  剩下的就是「已经在 i18n 通道上、但谁也没给它注册键」的裸英文。

假阳性模式
----------
  - `config: false` 的设置项不会出现在设置页（name/hint 写了也没人看）→ 本脚本单独标注
  - `restricted: true` 的只有 GM 看得到（不是假阳性，但受众更小）
  - choices 值若同时是点号键则不算
  - 括号配平用的是朴素计数，字符串里的花括号可能算错（已对命中逐条回源核对）
  - 本脚本不做数据流分析，只做语法邻近

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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i18n_registration_sink_gap.json")

TARGETS = {
    "crucible": {
        "src": [os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs"),
                os.path.join(FVTT, "systems", "crucible", "module")],
        "en": os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
        "cn": os.path.join(ROOT, "2-Crucible汉化插件", "lang", "cn.json"),
        "cnjs": [os.path.join(ROOT, "2-Crucible汉化插件", "babele-register.js")],
    },
    "ember": {
        "src": [os.path.join(FVTT, "modules", "ember", "scripts")],
        "en": os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
        "cn": os.path.join(ROOT, "1-Ember汉化插件", "lang", "cn.json"),
        "cnjs": [os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs"),
                 os.path.join(ROOT, "1-Ember汉化插件", "register.js")],
    },
}

REG_CALL = re.compile(
    r"\bgame\.(settings\.registerMenu|settings\.register|keybindings\.register)\s*\(\s*"
    r"[\"']([^\"']+)[\"']\s*,\s*[\"']([^\"']+)[\"']\s*,\s*\{")

FIELD = re.compile(r"(?m)^\s*(name|hint|label)\s*:\s*([\"'])((?:[^\"'\\]|\\.)*)\2")
CHOICE = re.compile(r"[\"']?([\w\-]+)[\"']?\s*:\s*([\"'])((?:[^\"'\\]|\\.)*)\2")
CONFIG_FLAG = re.compile(r"(?m)^\s*config\s*:\s*(true|false)")
BLANK = re.compile(r"\bblank\s*:\s*([\"'])((?:[^\"'\\]|\\.)*)\1")

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


def brace_block(s, open_idx):
    """从 '{' 位置起返回配平的块（朴素计数，忽略字符串内的花括号会造成偏差）"""
    depth = 0
    i = open_idx
    n = len(s)
    while i < n:
        c = s[i]
        if c in "\"'`":
            q = c
            i += 1
            while i < n and s[i] != q:
                if s[i] == "\\":
                    i += 1
                i += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx:i + 1]
        i += 1
    return s[open_idx:open_idx + 4000]


def files_of(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        elif os.path.isdir(p):
            for base, _dn, fns in os.walk(p):
                for fn in fns:
                    if fn.endswith((".mjs", ".js")):
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
            # 运行时替换表里若已出现该英文原串，也算覆盖
            for m in re.finditer(r"[\"']((?:[^\"'\\]|\\.){4,})[\"']\s*:", s):
                known.add(m.group(1))

    hits = []
    for path in files_of(cfg["src"]):
        s = io.open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, FVTT)
        for m in REG_CALL.finditer(s):
            api, ns, key = m.group(1), m.group(2), m.group(3)
            blk = brace_block(s, m.end() - 1)
            ln0 = s[:m.start()].count("\n") + 1
            cfgflag = CONFIG_FLAG.search(blk)
            visible = True
            if api == "settings.register":
                visible = bool(cfgflag and cfgflag.group(1) == "true")
            for fm in FIELD.finditer(blk):
                field, text = fm.group(1), fm.group(3)
                if api == "settings.register" and field == "label":
                    continue  # register() 没有 label 语义
                t = text.strip()
                if not t or KEYISH.match(t) or t in known:
                    continue
                if not re.search(r"[A-Za-z]{3,}", t):
                    continue
                hits.append({
                    "sink": {"settings.register": "R1", "settings.registerMenu": "R2",
                             "keybindings.register": "R3"}[api],
                    "api": api, "namespace": ns, "key": key, "field": field,
                    "text": t, "file": rel, "line": ln0 + blk[:fm.start()].count("\n"),
                    "user_visible": visible,
                })
            # choices 值（R4）
            for cm in re.finditer(r"\bchoices\s*:\s*\{", blk):
                sub = brace_block(blk, cm.end() - 1)
                for om in CHOICE.finditer(sub):
                    t = om.group(3).strip()
                    if not t or KEYISH.match(t) or t in known:
                        continue
                    if not re.search(r"[A-Za-z]{3,}", t):
                        continue
                    hits.append({"sink": "R4", "api": api, "namespace": ns, "key": key,
                                 "field": f"choices.{om.group(1)}", "text": t, "file": rel,
                                 "line": ln0 + blk[:cm.start()].count("\n"), "user_visible": visible})
            for bm in BLANK.finditer(blk):
                t = bm.group(2).strip()
                if t and not KEYISH.match(t) and t not in known and re.search(r"[A-Za-z]{3,}", t):
                    hits.append({"sink": "R4-blank", "api": api, "namespace": ns, "key": key,
                                 "field": "blank", "text": t, "file": rel,
                                 "line": ln0 + blk[:bm.start()].count("\n"), "user_visible": visible})
            # 设置项的 type 里 ForeignDocumentField/StringField 的 choices 回调常量（"": "-- None -- "）
            for dm in re.finditer(r"\{\s*[\"']{2}\s*:\s*([\"'])((?:[^\"'\\]|\\.)*)\1\s*\}", blk):
                t = dm.group(2).strip()
                if t and not KEYISH.match(t) and t not in known and re.search(r"[A-Za-z]{3,}", t):
                    hits.append({"sink": "R4-choicefn", "api": api, "namespace": ns, "key": key,
                                 "field": "choices['']", "text": dm.group(2), "file": rel,
                                 "line": ln0 + blk[:dm.start()].count("\n"), "user_visible": visible})
    return hits


def main():
    out = {}
    for name, cfg in TARGETS.items():
        h = scan(cfg)
        # 同一串在 compiled 与 module/ 两份源码里会重复，按 (sink,key,field,text) 去重保留首个
        seen = {}
        for x in h:
            k = (x["sink"], x["namespace"], x["key"], x["field"], x["text"])
            seen.setdefault(k, x)
        h = list(seen.values())
        out[name] = h
        print("=" * 100)
        print(f"[{name}] 注册型 API 上无键裸英文：{len(h)} 条")
        for x in sorted(h, key=lambda y: (not y["user_visible"], y["key"], y["field"])):
            vis = "可见" if x["user_visible"] else "config:false"
            print(f"  {x['sink']:<11} {vis:<12} {x['namespace']}.{x['key']}.{x['field']:<16} "
                  f"{x['file']}:{x['line']}")
            print(f"      {x['text'][:120]!r}")
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
