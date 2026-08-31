# -*- coding: utf-8 -*-
"""
probe: i18n_sinkfield_enumeration —— 「上游注入面枚举不全」的**元判据**（只读）

前面几条同类探针（i18n_sink_gap / i18n_registration_sink_gap / i18n_literal_gap）都
**先手工挑几个字段名当 sink**，再去上游找裸英文：
    i18n_sink_gap                → window.title / label / tooltip / data-tooltip
    i18n_registration_sink_gap   → settings/keybindings 调用块内的 name / hint / choices / blank
于是「字段名清单」本身成了新的枚举盲区：**没被列进清单的字段名，永远查不到。**

本探针换个方向：**从核心与上游的模板/源码里反推出「i18n 通道字段名」的全集**，
再拿这个全集去扫，专门看**前面清单里没有的那些字段名**能捞出什么。

sink 字段全集怎么来（可复现）
-----------------------------
  A. 核心模板 `resources/app/templates/**/*.hbs|html` 里所有 `{{localize <变量路径>}}`，
     取路径最后一段做字段名  → 48×label, 6×hint, 5×tooltip, 2×title, legend,
                               buttonText, heading, content, method, levelLabel,
                               paragraph, reference, restoreLabel, loading, name …
  B. 核心 JS `client/**` `common/**` 里所有 `_loc(<成员表达式>)`，同样取最后一段
     → label(54), title(7), labelPlural(7), tooltip(6), name(6), typeLabel(5),
       hint(4), placeholder(2), prefix(2), units, blank, group, groupName,
       denomination, description, disabledLabel, content, text, message, add, remove …
  C. crucible / ember **自己的模板**里的 `{{localize <变量>}}`（模块自建 sink）
     → resistanceLabel, actionTooltip, outcomeKey, outcome, pace.label, step.label …

  三者并集减去「前面已经扫过的 label / tooltip」= 本探针的重点字段。

判据
----
  在 ember.mjs / crucible-compiled.mjs 里找 `<重点字段>: "<裸英文>"`，
  剔除点号形 key、剔除已在 core/crucible/ember en.json 及本项目 cn.json 拍平键集里的，
  按「兄弟键签名」给出该对象最可能属于哪种注入面，供人工回源。

假阳性模式（很大，必须逐条回源）
--------------------------------
  - `name:` / `description:` / `content:` / `text:` 在冒险数据里海量出现，
    绝大多数根本不进 i18n 通道（它们是文档数据，走 babele）。脚本对这几个字段
    默认**只统计不列举**（--all 才全列），避免用噪声淹没真信号。
  - `units:` 是核心里**唯一无条件本地化**的字段（forms/fields.mjs:57
    `_loc(units)` 不看 localize 开关），但 ActiveEffect 的 `duration.units`
    同名不同物，需逐条区分。
  - createFormGroup 的 label/hint **只有传了 localize=true 才本地化**
    （forms/fields.mjs:54,70）。因此 DataField 里硬写的英文 label/hint
    在绝大多数渲染点上**根本不在 i18n 通道**（ember 174 处 formGroup 只有 2 处
    带 localize，crucible 206 处只有 8 处）—— 这类应判为「硬编码英文」而不是本类。
  - 只做语法邻近，不做数据流。

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
OUT = os.path.join(HERE, "i18n_sinkfield_enumeration.json")

ALREADY_SCANNED = {"label", "tooltip"}          # i18n_sink_gap 已覆盖
NOISY = {"name", "description", "content", "text", "message", "msg", "value", "prefix",
         "loading", "reference", "paragraph", "add", "remove", "w", "i18n", "localization",
         "locPath", "baseNameKey"}               # 默认只统计

KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[\w$\[\]-]+)+$")
HAS_WORD = re.compile(r"[A-Za-z]{3}")
LOC_TPL = re.compile(r"\{\{[#]?\s*localize\s+([A-Za-z_$][A-Za-z0-9_$.]*)")
LOC_JS = re.compile(r"(?:_loc|game\.i18n\.localize)\(\s*([A-Za-z_$][A-Za-z0-9_$.]*)\s*[),]")


def walk(root, exts):
    for base, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(exts):
                yield os.path.join(base, fn)


def collect_sink_fields():
    fields = {}

    def add(f, where):
        if not f:
            return
        fields.setdefault(f, set()).add(where)

    for p in walk(os.path.join(CORE, "templates"), (".hbs", ".html")):
        s = io.open(p, encoding="utf-8", errors="replace").read()
        for m in LOC_TPL.finditer(s):
            add(m.group(1).split(".")[-1], "core-tpl")
    for d in ("client", "common"):
        for p in walk(os.path.join(CORE, d), (".mjs", ".js")):
            s = io.open(p, encoding="utf-8", errors="replace").read()
            for m in LOC_JS.finditer(s):
                path = m.group(1)
                if "." not in path:
                    continue
                add(path.split(".")[-1], "core-js")
    for d in (os.path.join(FVTT, "systems", "crucible", "templates"),
              os.path.join(FVTT, "modules", "ember", "templates")):
        if not os.path.isdir(d):
            continue
        for p in walk(d, (".hbs", ".html")):
            s = io.open(p, encoding="utf-8", errors="replace").read()
            for m in LOC_TPL.finditer(s):
                add(m.group(1).split(".")[-1], "pkg-tpl")
    return fields


def flatten(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = "%s.%s" % (prefix, k) if prefix else k
            out.add(p)
            out |= flatten(v, p)
    return out


def known_keys(pkg):
    ks = flatten(json.load(io.open(os.path.join(CORE, "public", "lang", "en.json"), encoding="utf-8")))
    en = {"ember": os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
          "crucible": os.path.join(FVTT, "systems", "crucible", "lang", "en.json")}[pkg]
    cn = {"ember": os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6", "lang", "cn.json"),
          "crucible": os.path.join(ROOT, "2-Crucible\u6c49\u5316\u63d2\u4ef6", "lang", "cn.json")}[pkg]
    for p in (en, cn):
        if os.path.exists(p):
            ks |= flatten(json.load(io.open(p, encoding="utf-8")))
    return ks


SRC = {
    "ember": [os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs"),
              os.path.join(FVTT, "modules", "ember", "scripts", "crucible-async.mjs"),
              os.path.join(FVTT, "modules", "ember", "scripts", "dnd5e-async.mjs")],
    "crucible": [os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs")],
}


def main():
    show_all = "--all" in sys.argv
    fields = collect_sink_fields()
    focus = sorted(f for f in fields if f not in ALREADY_SCANNED)
    print("i18n 通道字段名全集：%d 个" % len(fields))
    print("  已被既有探针扫过：%s" % ", ".join(sorted(ALREADY_SCANNED)))
    print("  本探针重点（%d 个）：%s" % (len(focus), ", ".join(focus)))
    print()

    out = {}
    for pkg, files in SRC.items():
        known = known_keys(pkg)
        per = {}
        for f in files:
            if not os.path.exists(f):
                continue
            s = io.open(f, encoding="utf-8", errors="replace").read()
            rel = os.path.relpath(f, FVTT)
            for fld in focus:
                rx = re.compile(r"\b%s\s*:\s*([\"'])((?:[^\"'\\]|\\.){1,300})\1" % re.escape(fld))
                for m in rx.finditer(s):
                    v = m.group(2)
                    if KEYISH.match(v) or not HAS_WORD.search(v) or v in known:
                        continue
                    per.setdefault(fld, []).append({
                        "text": v, "file": rel, "line": s[:m.start()].count("\n") + 1})
        out[pkg] = per
        print("=" * 100)
        print("[%s]" % pkg)
        for fld in sorted(per, key=lambda k: -len(per[k])):
            rows = per[fld]
            uniq = sorted({r["text"] for r in rows})
            tag = "（噪声字段，只统计）" if (fld in NOISY and not show_all) else ""
            print("  %-16s 命中 %-5d 去重 %-5d %s" % (fld, len(rows), len(uniq), tag))
            if fld in NOISY and not show_all:
                continue
            for r in rows[:40]:
                print("        %s:%s  %r" % (r["file"], r["line"], r["text"][:120]))
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
