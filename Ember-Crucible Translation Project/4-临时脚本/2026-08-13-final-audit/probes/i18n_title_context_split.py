# -*- coding: utf-8 -*-
"""
probe: i18n_title_context_split —— 把 ember 里 64 个裸英文 `title:` 按**所属注入面**分家（只读）

为什么要分家
------------
`title:` 这个字段名同时落在**两个完全不同的 sink** 上：
  (a) `window: {title: "..."}`  → ApplicationV2#title，applications/api/application.mjs:319-321
      —— 这一支已被 i18n_sink_gap 的 S1 覆盖（但它的 HUMAN 正则不允许串内出现
         `:` `&` 等符号，所以 'Ember: Create Weather' / 'Q&A Block' 这类会漏）
  (b) **非 window 的 title**    → 场景控件工具/图层：templates/ui/scene-controls-tools.hbs:5
      `aria-label="{{localize tool.title}}"`；scene-controls-layers.hbs:6 同理
      —— 这一支**没有任何探针扫过**，因为所有探针都只在 `window:{}` 里找 title

本脚本对每个裸英文 `title:` 回溯它的直接父对象，判断是不是 window 块，
并顺带报告 (a) 支里**会被 i18n_sink_gap 的 HUMAN 正则漏掉**的那些串。

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
OUT = os.path.join(HERE, "i18n_title_context_split.json")

SRC = {
    "ember": [os.path.join(FVTT, "modules", "ember", "scripts", "ember.mjs"),
              os.path.join(FVTT, "modules", "ember", "scripts", "crucible-async.mjs"),
              os.path.join(FVTT, "modules", "ember", "scripts", "dnd5e-async.mjs")],
    "crucible": [os.path.join(FVTT, "systems", "crucible", "crucible-compiled.mjs")],
}
EN = {"ember": os.path.join(FVTT, "modules", "ember", "lang", "en.json"),
      "crucible": os.path.join(FVTT, "systems", "crucible", "lang", "en.json")}
CN = {"ember": os.path.join(ROOT, "1-Ember\u6c49\u5316\u63d2\u4ef6", "lang", "cn.json"),
      "crucible": os.path.join(ROOT, "2-Crucible\u6c49\u5316\u63d2\u4ef6", "lang", "cn.json")}

KEYISH = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[\w$\[\]-]+)+$")
# i18n_sink_gap 用的 HUMAN 正则，原样复制，用来判断哪些串它会滤掉
SINKGAP_HUMAN = re.compile(r"^[A-Z][A-Za-z0-9'\u2019\-]*(?:[ /][A-Za-z0-9'\u2019\-]+)*[?!.]?$")
TITLE = re.compile(r"\btitle\s*:\s*([\"'])((?:[^\"'\\]|\\.){1,300})\1")


def flatten(obj, prefix=""):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = "%s.%s" % (prefix, k) if prefix else k
            out.add(p)
            out |= flatten(v, p)
    return out


CORE_KEYS = flatten(json.load(io.open(os.path.join(CORE, "public", "lang", "en.json"), encoding="utf-8")))


def enclosing_open(s, idx, back=6000):
    depth = 0
    i = idx
    lo = max(0, idx - back)
    while i > lo:
        c = s[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                return i
            depth -= 1
        i -= 1
    return None


def main():
    result = {}
    for pkg, files in SRC.items():
        known = set(CORE_KEYS)
        for p in (EN[pkg], CN[pkg]):
            if os.path.exists(p):
                known |= flatten(json.load(io.open(p, encoding="utf-8")))
        rows = []
        for f in files:
            if not os.path.exists(f):
                continue
            s = io.open(f, encoding="utf-8", errors="replace").read()
            rel = os.path.relpath(f, FVTT)
            for m in TITLE.finditer(s):
                v = m.group(2)
                if KEYISH.match(v) or v in known or not re.search(r"[A-Za-z]{3}", v):
                    continue
                op = enclosing_open(s, m.start())
                prefix = s[max(0, (op or m.start()) - 60):(op or m.start())]
                is_window = bool(re.search(r"\bwindow\s*:\s*$", prefix.rstrip()))
                # 场景控件工具：兄弟里有 icon 且父路径写到 .tools
                blk_head = s[max(0, (op or 0) - 220):(op or 0)]
                is_tool = bool(re.search(r"\.tools\b|tools\s*,?\s*$|controls\.\w+\.tools", blk_head))
                rows.append({
                    "text": v, "file": rel, "line": s[:m.start()].count("\n") + 1,
                    "ctx": "window.title" if is_window else ("scene-control tool" if is_tool else "OTHER"),
                    "ctx_prefix": re.sub(r"\s+", " ", prefix)[-60:],
                    "sinkgap_would_catch": bool(SINKGAP_HUMAN.match(v)),
                })
        result[pkg] = rows
        print("=" * 100)
        print("[%s] 裸英文 title: 共 %d 处" % (pkg, len(rows)))
        for ctx in ("scene-control tool", "OTHER", "window.title"):
            sub = [r for r in rows if r["ctx"] == ctx]
            print("  --- %s : %d ---" % (ctx, len(sub)))
            for r in sub:
                flag = "" if r["sinkgap_would_catch"] else "  <== i18n_sink_gap 的 HUMAN 会滤掉"
                print("      %s:%s  %r%s" % (r["file"], r["line"], r["text"][:90], flag))
                if ctx != "window.title":
                    print("            prefix=%r" % r["ctx_prefix"])
    json.dump(result, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n->", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
