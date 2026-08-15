#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探针 2：上游「英文字面量发射点」 vs 汉化插件的闸/词表。

把已确认实例（crucibleTalent 发 `Talent: ${name}`，而 PREFIXED 里没有 Talent 条）
抽象为：**上游任何往 DOM 写英文字面量的地方，都要能被 translateText 的某一条规则接住**。

判据：
  发射点 = 形如 `X.innerHTML/innerText/textContent = \`...\`` 或
           `dataset.tooltip* = "..."` / `setAttribute("aria-label"|"title", "...")`
           且右值含英文字面量。
  接住   = 该字面量整串 ∈ EXACT，或其前缀 ∈ PREFIXED，或整串匹配 PATTERNS 之一。
差集即候选。

假阳性模式：
  - 右值可能是 i18n 键（含点号、全大写段）或 CSS 类名 —— 已过滤但不完全；
  - 有些发射点只在 dnd5e 系统分支跑（本项目只管 crucible 侧）；
  - 有些元素根本不显示（隐藏节点 / 开发模式）。
只读，不写库。
"""
import json
import os
import re
import sys

DATA = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data"
PROJ = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
HARDCODED = os.path.join(PROJ, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")
SOURCES = {
    "ember/scripts/ember.mjs": os.path.join(DATA, "modules", "ember", "scripts", "ember.mjs"),
    "ember/scripts/crucible-async.mjs": os.path.join(DATA, "modules", "ember", "scripts", "crucible-async.mjs"),
    "crucible/crucible-compiled.mjs": os.path.join(DATA, "systems", "crucible", "crucible-compiled.mjs"),
}
OUT = os.path.join(PROJ, "4-临时脚本", "2026-08-13-final-audit", "findings", "emitter_vs_tables.json")


def read(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def parse_tables(src):
    exact = re.search(r"const EXACT = \{(.*?)\n\};", src, re.S).group(1)
    exact_keys = set(re.findall(r'\n  "([^"]+)":', exact))
    prefixes = re.findall(r'\{ en: "([^"]+)", cn:', src)
    patterns = re.findall(r"\{ re: (/\^[^,]+/), cn:", src)
    return exact_keys, prefixes, patterns


# 发射点：`… = \`Something: ${…}\`` 或 `… = "Some English"` 挂在显示属性上
EMIT_TPL = re.compile(
    r"(innerHTML|innerText|textContent)\s*(?:\+)?=\s*`([^`]{0,120})`")
EMIT_ATTR = re.compile(
    r"""(?:dataset\.(?:tooltip|tooltipText|tooltipHtml)|setAttribute\(\s*["'](?:aria-label|title|data-tooltip)["']\s*,)\s*=?\s*["`]([^"`]{2,120})["`]""")
PREFIX_FORM = re.compile(r"^([A-Z][A-Za-z][A-Za-z ]{1,24}):\s*\$\{")
HAS_EN = re.compile(r"[A-Za-z]{3}")
LOOKS_KEY = re.compile(r"^[A-Z][A-Z0-9_.]+$|^[A-Za-z]+\.[A-Za-z.]+$")


def main():
    hard = read(HARDCODED)
    exact_keys, prefixes, patterns = parse_tables(hard)

    found = []
    for name, path in SOURCES.items():
        src = read(path)
        for m in EMIT_TPL.finditer(src):
            val = m.group(2).strip()
            if not HAS_EN.search(val) or "${" not in val and val in exact_keys:
                continue
            line = src[:m.start()].count("\n") + 1
            pm = PREFIX_FORM.match(val)
            if pm:
                prefix = pm.group(1)
                found.append({
                    "file": name, "line": line, "kind": "prefix-emitter",
                    "literal": val, "prefix": prefix,
                    "covered": prefix in prefixes,
                })
            elif "${" not in val and not LOOKS_KEY.match(val):
                found.append({
                    "file": name, "line": line, "kind": "exact-emitter",
                    "literal": val, "covered": val in exact_keys,
                })
        for m in EMIT_ATTR.finditer(src):
            val = m.group(1).strip()
            if not HAS_EN.search(val) or LOOKS_KEY.match(val) or "${" in val:
                continue
            line = src[:m.start()].count("\n") + 1
            found.append({
                "file": name, "line": line, "kind": "attr-emitter",
                "literal": val, "covered": val in exact_keys,
            })

    misses = [f for f in found if not f["covered"]]
    out = {
        "tables": {"n_exact": len(exact_keys), "prefixes": prefixes, "patterns": patterns},
        "n_emitters": len(found),
        "n_uncovered": len(misses),
        "uncovered": misses,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print(f"PREFIXED = {prefixes}")
    print(f"emitters={len(found)}  uncovered={len(misses)}")
    for f in misses:
        print(f"  [{f['kind']:14s}] {f['file']}:{f['line']}  {f['literal'][:80]!r}")


if __name__ == "__main__":
    sys.exit(main())
