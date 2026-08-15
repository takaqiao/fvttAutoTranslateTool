#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探针：闸/选择器失配（gate/selector mismatch）—— 把「enricher 词表闸漏掉 crucible 的
talent/Spell」抽象成一条可机械化判据，在全库扩查。

判据（三步）：
  1) 列出汉化插件里所有「闸」：一个用来决定「哪些上游对象要被处理」的过滤条件
     （正则词表 / 前缀表 / 键名白名单 / DOM 选择器 / 钩子名）。
  2) 列出上游（ember 0.6.0 + crucible 0.10.1）实际存在的对象全集。
  3) 求差集：上游有、闸放不进来、且该对象确实输出英文字面量 → 候选缺陷。

本脚本只读，不写库。产出 JSON 到 findings/gate_scope_vs_targets.json。

已知假阳性模式（人工复核时要逐条排掉）：
  - 模板里的英文可能是 {{localize}} 的键名或变量名，不是显示文本；
  - 有些英文字面量只在 GM/开发模式下出现；
  - 有些「非 ember 名字的应用」根本不含 ember 注入的内容，只是恰好被扫到；
  - 上游若已用 _loc()，即使闸放不进来，也能由 crucible-cn / ember lang 覆盖，不算缺陷。
"""
import json
import os
import re
import sys

DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
EMBER = os.path.join(DATA, "modules", "ember")
CRUCIBLE = os.path.join(DATA, "systems", "crucible")
PROJ = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
HARDCODED = os.path.join(PROJ, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
OUT = os.path.join(PROJ, "4-临时脚本", "2026-08-13-final-audit", "findings",
                   "gate_scope_vs_targets.json")


def read(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------- 1. 闸
def parse_gates(src):
    """从 ember-hardcoded-cn.mjs 里抽出各道闸的字面定义。"""
    gates = {}
    m = re.search(r"if \(!(/[^/]+/i)\.test\(src\)\) continue;", src)
    gates["enricher_wordlist"] = m.group(1) if m else None
    m = re.search(r"if \(!/ember/i\.test\(cls\) && !/\^Ember/\.test\(id\)\)", src)
    gates["render_app"] = "/ember/i on className OR /^Ember/ on constructor.name" if m else None
    m = re.search(r"for \(const \[key, table\] of \[\[\"languages\", LANGUAGES\], \[\"knowledge\", KNOWLEDGE\]\]\)", src)
    gates["crucible_config_groups"] = ["languages", "knowledge"] if m else None
    m = re.search(r"for \(const \[key, table\] of \[\[\"months\", CALENDAR_MONTHS\], \[\"days\", CALENDAR_DAYS\]\]\)", src)
    gates["calendar_keys"] = ["months", "days"] if m else None
    m = re.search(r'for \(const attr of \[([^\]]+)\]\)', src)
    gates["attributes"] = re.findall(r'"([^"]+)"', m.group(1)) if m else None
    gates["render_hooks"] = re.findall(r'Hooks\.on\("(render[A-Za-z]*)"', src)
    gates["prefixes"] = re.findall(r'\{ en: "([^"]+)"', src)
    exact = re.search(r"const EXACT = \{(.*?)\n\};", src, re.S)
    gates["exact_keys"] = re.findall(r'\n  "([^"]+)":', exact.group(1)) if exact else []
    return gates


# --------------------------------------------- 2. 上游：模板里的英文字面量
TEXT_EN = re.compile(r">([^<>{}]*[A-Za-z]{3}[^<>{}]*)<")
ATTR_EN = re.compile(r'\b(aria-label|data-tooltip|title|placeholder|alt|label)="([^"{}]*[A-Za-z]{3}[^"{}]*)"')
CN = re.compile(r"[\u4e00-\u9fff]")


def scan_templates():
    """扫 ember 全部 .hbs：抓写死的英文显示文本与英文属性值。"""
    hits = []
    root = os.path.join(EMBER, "templates")
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith((".hbs", ".html")):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, EMBER).replace("\\", "/")
            src = read(path)
            for i, line in enumerate(src.splitlines(), 1):
                for m in TEXT_EN.finditer(line):
                    txt = m.group(1).strip()
                    if not txt or CN.search(txt):
                        continue
                    if len(txt) < 3 or not re.search(r"[A-Za-z]{3}", txt):
                        continue
                    hits.append({"template": rel, "line": i, "kind": "text", "value": txt})
                for m in ATTR_EN.finditer(line):
                    val = m.group(2).strip()
                    if not val or CN.search(val):
                        continue
                    if "." in val and " " not in val:      # 多半是 i18n 键或路径
                        continue
                    hits.append({"template": rel, "line": i, "kind": f"attr:{m.group(1)}", "value": val})
    return hits


# ------------------------------------ 3. 模板 -> 宿主应用类 -> 闸判定
CLASS_RE = re.compile(r"^\s*class ([A-Za-z0-9_$]+) extends", re.M)


def template_owner(js_sources, tpl):
    """找出注册该模板的应用类：取模板路径出现处之前最近的 class 声明。
    形如 `cls.PARTS.x = {...}` 的注入无法靠位置判定，另行标记。"""
    owners = []
    needle = tpl
    for name, src in js_sources.items():
        for m in re.finditer(re.escape(needle), src):
            pos = m.start()
            head = src[max(0, pos - 4000):pos]
            inject = re.findall(r"([A-Za-z0-9_$.]+)\.PARTS\.\w+\s*=", head[-800:])
            cls = None
            for cm in CLASS_RE.finditer(src[:pos]):
                cls = cm.group(1)
            owners.append({"file": name, "class": cls, "inject_target": inject[-1] if inject else None,
                           "line": src[:pos].count("\n") + 1})
    return owners


def gate_pass(cls_name, classes_tokens):
    if cls_name and re.match(r"^Ember", cls_name):
        return True
    return any("ember" in t.lower() for t in classes_tokens)


def main():
    hardcoded = read(HARDCODED)
    gates = parse_gates(hardcoded)

    js = {
        "scripts/ember.mjs": read(os.path.join(EMBER, "scripts", "ember.mjs")),
        "scripts/crucible-async.mjs": read(os.path.join(EMBER, "scripts", "crucible-async.mjs")),
        "scripts/dnd5e-async.mjs": read(os.path.join(EMBER, "scripts", "dnd5e-async.mjs")),
    }

    tpl_hits = scan_templates()
    by_tpl = {}
    for h in tpl_hits:
        by_tpl.setdefault(h["template"], []).append(h)

    report = []
    for tpl, hits in sorted(by_tpl.items()):
        owners = template_owner(js, "modules/ember/" + tpl)
        report.append({
            "template": tpl,
            "n_english": len(hits),
            "owners": owners,
            "literals": hits[:20],
        })

    out = {
        "gates": gates,
        "n_templates_with_english": len(report),
        "n_english_literals": len(tpl_hits),
        "templates": report,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"gates: {json.dumps(gates, ensure_ascii=False)[:400]}")
    print(f"templates with english: {len(report)}, literals: {len(tpl_hits)}")
    for r in report:
        cls = ",".join(sorted({str(o['class']) for o in r['owners']})) or "?"
        inj = ",".join(sorted({str(o['inject_target']) for o in r['owners'] if o['inject_target']}))
        print(f"  {r['template']:60s} n={r['n_english']:3d} owner={cls} inject={inj}")


if __name__ == "__main__":
    sys.exit(main())
