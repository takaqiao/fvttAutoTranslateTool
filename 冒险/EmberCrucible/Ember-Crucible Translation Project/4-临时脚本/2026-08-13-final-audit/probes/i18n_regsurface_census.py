# -*- coding: utf-8 -*-
"""
probe: i18n_regsurface_census —— 把「注册型 API 的 name/hint 裸串」这一类**推广到全部
上游注入面**（只读）

前一条已确认实例只覆盖了三个注册 API：
    game.settings.register / registerMenu / game.keybindings.register
本探针问的是：**Foundry v14 里还有多少「调用方交一个配置对象、核心对其中某字段跑
localize()」的注入面？ember / crucible 在这些面上写了几处裸英文？**

核心侧 sink 逐条回源（v14.348，均已读源码/模板确认）
----------------------------------------------------
  N1 文档表单类型标签  applications/apps/document-sheet-config.mjs:442
                       `else if ( label ) label = _loc(label)`
                       ← DocumentSheetConfig.registerSheet(cls, pkg, sheet, {label})
  N2 场景控件工具      templates/ui/scene-controls-tools.hbs:5  `aria-label="{{localize tool.title}}"`
                       templates/ui/scene-controls-layers.hbs:6 `aria-label="{{localize control.title}}"`
                       ← Hooks "getSceneControlButtons" 里塞的 {name,title,icon,...}
  N3 应用窗口头部控件  applications/api/application.mjs:910 `span.innerText = _loc(control.label)`
                       templates/app-window.html:6 `{{localize this.label}}`
                       ← Hooks "getHeaderControls<X>" 里 controls.push({label,...})
  N4 右键菜单项        applications/ux/context-menu.mjs:403
                       `_loc("label" in item ? item.label : item.name)`
  N5 DataField 三件套  helpers/localization.mjs:195-197 `this.label/hint/placeholder ||= _loc(...)`
                       applications/forms/fields.mjs (formGroup) 亦对 label/hint 取 localize
  N6 文档类型标签/提示 documents/abstract/client-document.mjs:821,834,857
                       `_loc(this.metadata.label)` / `_loc(config.typeHints?.[type])`
                       ← CONFIG.<Doc>.typeLabels / typeHints
  N7 令牌移动动作      crucible-compiled 36819 等处 `_loc(cfg.label)`；核心 token-hud.mjs:175
                       ← CONFIG.Token.movement.actions[id].label
  N8 导览 Tour         nue/tour.mjs:160,168,466,467 `_loc(this.config.title/description/step.title/step.content)`
  N9 ProseMirror 下拉  common/prosemirror/dropdown.mjs:228 `_loc(item.title)`
  N10 骰子求值方法     applications/dice/roll-resolver.mjs:143 `tooltip: _loc(config.label)`
                       ← CONFIG.Dice.fulfillment.methods[].label
  N11 合集包标题       applications/sidebar/apps/compendium.mjs:57 `_loc(this.collection.title)`
                       ← 清单 packs[].label（**babele 已接管，本脚本仅计数不报**）

判据
----
  1. 在上游 JS 里按「兄弟键签名」识别上述配置对象（不是只看字段名，避免把
     普通业务对象的 name/label 也算进来）
  2. 取该字段位置上的字符串字面量，剔除
       ① 点号形 i18n key（那是另一类，i18n_undeclared_key 覆盖）
       ② 已在 core / crucible / ember en.json 或本项目 cn.json 拍平键集里的
       ③ 已被 ember-hardcoded-cn.mjs / register.js 运行时替换表按原串覆盖的
  3. 剩下的即「已在 i18n 通道上、却没有任何键、汉化侧也够不到」的裸英文

假阳性模式（必须逐条回源核对）
------------------------------
  - 兄弟键签名是启发式：一个恰好同时有 icon/title/onChange 的普通对象会误命中
  - 只做语法邻近，不做数据流；`title: SOME_CONST` 这种变量引用一律看不见
  - 某些注入点在 dev-only / 依赖第三方模块（CDT）才执行 —— 脚本单列 `gated` 字段
  - 括号配平用朴素扫描（会跳过字符串内的花括号），深层嵌套可能取错块
  - 值同时是英文文本又是别处 key 的极少数情况会被 ② 误滤（宁漏不滥）

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
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "i18n_regsurface_census.json")

SRC = {
    "ember": [os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs"),
              os.path.join(FVTT, "modules", "ember", "scripts", "crucible-async.mjs"),
              os.path.join(FVTT, "modules", "ember", "scripts", "dnd5e-async.mjs")],
    "crucible": [os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs")],
}
EN = {
    "ember": os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
    "crucible": os.path.join(FVTT, "systems", "crucible", "lang", "en.json"),
}
CN = {
    "ember": os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6", "lang", "cn.json"),
    "crucible": os.path.join(ROOT, "2-Crucible\u6c49\u5316\u63d2\u4ef6", "lang", "cn.json"),
}
CNJS = {
    "ember": [os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6", "scripts", "ember-hardcoded-cn.mjs"),
              os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6", "register.js")],
    "crucible": [os.path.join(ROOT, "2-Crucible\u6c49\u5316\u63d2\u4ef6", "babele-register.js")],
}

KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[\w$\[\]-]+)+$")
HAS_WORD = re.compile(r"[A-Za-z]{3}")
STR = r"([\"'])((?:[^\"'\\]|\\.){1,400})\1"


def flatten(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = "%s.%s" % (prefix, k) if prefix else k
            out.add(p)
            out |= flatten(v, p)
    return out


CORE_KEYS = flatten(json.load(io.open(os.path.join(CORE, "public", "lang", "en.json"), encoding="utf-8")))


def brace_block(s, open_idx, limit=20000):
    depth = 0
    i = open_idx
    n = min(len(s), open_idx + limit)
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
    return s[open_idx:n]


def enclosing_object(s, idx, back=4000):
    """从 idx 向前找到最近的未配平 '{'，返回 (start, block)"""
    depth = 0
    i = idx
    lo = max(0, idx - back)
    while i > lo:
        c = s[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                return i, brace_block(s, i)
            depth -= 1
        i -= 1
    return None, ""


SHALLOW_KEY = re.compile(r"(?m)(?:^|[{,])\s*(?:\[[^\]]+\]|[\"']?([A-Za-z_$][\w$]*)[\"']?)\s*:")


def shallow_keys(block):
    """块内**第一层**键名（朴素：按深度过滤）"""
    keys = set()
    depth = 0
    i = 0
    n = len(block)
    while i < n:
        c = block[i]
        if c in "\"'`":
            q = c
            i += 1
            while i < n and block[i] != q:
                if block[i] == "\\":
                    i += 1
                i += 1
        elif c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        elif depth == 1 and (c.isalpha() or c in "_$"):
            m = re.match(r"([A-Za-z_$][\w$]*)\s*:", block[i:])
            if m and (i == 0 or block[i - 1] in "{,\n\r\t "):
                keys.add(m.group(1))
                i += m.end() - 1
        i += 1
    return keys


# --- sink 定义：字段名 + 必须同时出现的兄弟键（任一组满足即可） -------------------
SINKS = [
    ("N2 scene-control tool", "title", [{"icon"}]),
    ("N3 header control", "label", [{"icon", "action"}, {"icon", "onClick"}]),
    ("N4 context-menu entry", "name", [{"icon", "callback"}]),
    ("N4 context-menu entry", "label", [{"icon", "callback"}]),
    ("N5 DataField", "hint", [{"required"}, {"initial"}, {"blank"}, {"nullable"}, {"choices"}]),
    ("N5 DataField", "placeholder", [{"required"}, {"initial"}, {"blank"}, {"nullable"}]),
    ("N7 movement action", "label", [{"costMultiplier"}, {"speedMultiplier"}, {"teleport"}]),
    ("N9 prosemirror item", "title", [{"action", "html"}]),
    ("N10 dice method", "label", [{"interactive"}, {"handler"}]),
]

FIELD_RE = {f: re.compile(r"\b%s\s*:\s*%s" % (f, STR))
            for f in {"title", "label", "name", "hint", "placeholder"}}


def scan_sinks(text, rel, known):
    hits = []
    for sink, field, sigs in SINKS:
        for m in FIELD_RE[field].finditer(text):
            val = m.group(2)
            if KEYISH.match(val) or not HAS_WORD.search(val):
                continue
            if val in known:
                continue
            start, blk = enclosing_object(text, m.start())
            if start is None:
                continue
            ks = shallow_keys(blk)
            if not any(sig <= ks for sig in sigs):
                continue
            hits.append({"sink": sink, "field": field, "text": val, "file": rel,
                         "line": text[:m.start()].count("\n") + 1,
                         "siblings": sorted(ks)[:14]})
    return hits


REGSHEET = re.compile(r"registerSheet\s*\(")
TYPELABELS = re.compile(r"(typeLabels|typeHints)\s*(?:\[[^\]]*\]\s*=|=)\s*")


def scan_extra(text, rel, known):
    hits = []
    # N1 registerSheet(..., {label: "..."})
    for m in REGSHEET.finditer(text):
        seg = text[m.end():m.end() + 900]
        b = seg.find("{")
        if b < 0:
            continue
        blk = brace_block(seg, b)
        lm = re.search(r"\blabel\s*:\s*%s" % STR, blk)
        if not lm:
            continue
        val = lm.group(2)
        if KEYISH.match(val) or val in known or not HAS_WORD.search(val):
            continue
        hits.append({"sink": "N1 sheet label", "field": "label", "text": val, "file": rel,
                     "line": text[:m.start()].count("\n") + 1, "siblings": sorted(shallow_keys(blk))[:14]})
    # N6 typeLabels / typeHints tables
    for m in TYPELABELS.finditer(text):
        seg = text[m.end():m.end() + 1500]
        if not seg.lstrip().startswith("{"):
            continue
        blk = brace_block(seg, seg.find("{"))
        for om in re.finditer(r"[\"']?([\w-]+)[\"']?\s*:\s*%s" % STR, blk):
            val = om.group(3)
            if KEYISH.match(val) or val in known or not HAS_WORD.search(val):
                continue
            hits.append({"sink": "N6 typeLabels", "field": m.group(1) + "." + om.group(1),
                         "text": val, "file": rel,
                         "line": text[:m.start()].count("\n") + 1, "siblings": []})
    # N8 tours
    for m in re.finditer(r"(game\.tours\.register|new\s+Tour\s*\(|Tour\.fromJSON)", text):
        hits.append({"sink": "N8 tour", "field": "-", "text": "<tour registration present>",
                     "file": rel, "line": text[:m.start()].count("\n") + 1, "siblings": []})
    return hits


def main():
    result = {}
    for pkg, files in SRC.items():
        known = set()
        known |= CORE_KEYS
        for p in (EN[pkg], CN[pkg]):
            if os.path.exists(p):
                known |= flatten(json.load(io.open(p, encoding="utf-8")))
        for p in CNJS[pkg]:
            if os.path.exists(p):
                s = io.open(p, encoding="utf-8", errors="replace").read()
                for m in re.finditer(r"[\"']((?:[^\"'\\]|\\.){3,})[\"']\s*:", s):
                    known.add(m.group(1))
        hits = []
        for f in files:
            if not os.path.exists(f):
                continue
            t = io.open(f, encoding="utf-8", errors="replace").read()
            rel = os.path.relpath(f, FVTT)
            hits += scan_sinks(t, rel, known)
            hits += scan_extra(t, rel, known)
        dedup = {}
        for h in hits:
            dedup.setdefault((h["sink"], h["field"], h["text"]), h)
        result[pkg] = sorted(dedup.values(), key=lambda x: (x["sink"], x["file"], x["line"]))
        print("=" * 100)
        print("[%s] %d 条候选" % (pkg, len(result[pkg])))
        for h in result[pkg]:
            print("  %-24s %-12s %s:%s" % (h["sink"], h["field"], h["file"], h["line"]))
            print("      %r" % h["text"][:140])
            if h["siblings"]:
                print("      siblings=%s" % (",".join(h["siblings"])))
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
