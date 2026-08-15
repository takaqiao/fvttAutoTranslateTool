#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1-b 反方向：从 ember 模块里**收割**玩家可见的硬编码英文串，看 .mjs 覆盖了没有。

三个来源：
  1. hbs/html 模板里 `>文本<` 的纯文本（排除 {{...}} 表达式与纯符号）
  2. js 里的 `窗口标题 / innerText= / innerHTML= / label:` 英文字面量
  3. js 里 DialogV2 的 window.title / content 里的 <p> 文案（只取标题）

输出：未被 .mjs 任何一张表覆盖的候选（按来源分组）。只读。
"""
import json
import re
import sys
from pathlib import Path

MOD = Path(r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember")
KEYS = Path(sys.argv[1])

WORD = re.compile(r"[A-Za-z]")
HB = re.compile(r"\{\{.*?\}\}", re.S)


def load_keys():
    raw = KEYS.read_text(encoding="utf-8")
    data = json.loads(raw[:raw.rindex("}") + 1])
    covered = set()
    for t, d in data.items():
        if t == "PREFIXED":
            for e in d:
                covered.add(e["en"])
            continue
        if t == "PATTERNS":
            continue
        covered.update(d.keys())
    return covered, data


def main():
    covered, _ = load_keys()

    # ---- 1. 模板可见文本 ----
    tpl_hits = {}
    for p in sorted(MOD.rglob("*")):
        if p.suffix.lower() not in {".hbs", ".html"}:
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p.relative_to(MOD)).replace("\\", "/")
        for m in re.finditer(r">([^<>]{2,80})<", txt):
            s = m.group(1)
            if "{{" in s or "}}" in s:
                continue
            s = s.strip()
            if not s or not WORD.search(s):
                continue
            if s.startswith("//") or s.startswith("/*"):
                continue
            tpl_hits.setdefault(s, []).append(rel)
        # 属性里的可见文本
        for m in re.finditer(r'(?:placeholder|title|aria-label|data-tooltip)\s*=\s*"([^"{}]{2,80})"', txt):
            s = m.group(1).strip()
            if s and WORD.search(s):
                tpl_hits.setdefault(s, []).append(rel + "@attr")

    # ---- 2. js 里的可见字面量 ----
    js_hits = {}
    pats = [
        (r'title:\s*"([^"]{2,80})"', "title"),
        (r"title:\s*`([^`$]{2,80})`", "title"),
        (r'innerText\s*=\s*"([^"]{2,80})"', "innerText"),
        (r'innerHTML\s*=\s*"([^"]{2,80})"', "innerHTML"),
        (r'label:\s*"([^"]{2,80})"', "label"),
        (r'header:\s*"([^"]{2,80})"', "header"),
        (r'legend:\s*"([^"]{2,80})"', "legend"),
        (r'ok:\s*\{[^}]*label:\s*"([^"]{2,80})"', "ok"),
        (r'hint:\s*"([^"]{2,80})"', "hint"),
        (r'name:\s*"([^"]{2,80})"', "name"),
    ]
    for p in sorted(MOD.rglob("*.mjs")):
        txt = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p.relative_to(MOD)).replace("\\", "/")
        for rx, kind in pats:
            for m in re.finditer(rx, txt):
                s = m.group(1).strip()
                if not s or not WORD.search(s):
                    continue
                if s.startswith(("EMBER.", "CRUCIBLE.", "DND5E.", "fa-", "modules/", "http")):
                    continue
                if re.fullmatch(r"[a-z][A-Za-z0-9]*", s):     # 标识符样式，非文案
                    continue
                js_hits.setdefault(s, []).append(f"{rel}[{kind}]")

    def dump(title, hits):
        print(f"\n########## {title}（未被 .mjs 覆盖） ##########")
        for s in sorted(hits):
            if s in covered:
                continue
            print(f"{s}\t{len(hits[s])}\t{sorted(set(hits[s]))[:3]}")

    dump("模板可见文本", tpl_hits)
    dump("JS 可见字面量", js_hits)


if __name__ == "__main__":
    main()
