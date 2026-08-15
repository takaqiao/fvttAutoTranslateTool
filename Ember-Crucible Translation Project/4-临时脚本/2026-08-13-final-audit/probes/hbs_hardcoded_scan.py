# -*- coding: utf-8 -*-
"""
hbs_hardcoded_scan.py —— 探针 D：Ember 模板里的裸英文（i18n 够不到的输出面）

同一类判据的第二个落点：ember 的 .hbs 模板里如果**直接写死英文文本**
（不是 {{localize}}、不是变量插值），那么 Foundry i18n 与 babele 都够不到，
只剩 ember-hardcoded-cn.mjs 的 translateNode + EXACT 这一条路。
凡是 EXACT / PREFIXED / PATTERNS 命不中的，就是渲染出来仍是英文的输出面。

假阳性模式：
  - 注释、class 名、data-* 值、图标名 —— 已过滤
  - 只在 dnd5e 世界才渲染的模板（文件名带 dnd5e）—— 单独标出
  - 模板可能根本不被 crucible 分支使用 —— 需回源码确认

只读，不写库。
"""
import os, re, sys

TPL = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\templates"
PLUGIN = (r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
          r"\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")

HANDLEBARS = re.compile(r"\{\{[^}]*\}\}")
TAG = re.compile(r"<[^>]+>")
COMMENT = re.compile(r"<!--.*?-->", re.S)
WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]{1,}")

# 标签属性里也要看的可见文本
ATTRS = re.compile(r'\b(?:data-tooltip|data-tooltip-text|aria-label|title|placeholder|label)="([^"{]+)"')


def plugin_keys():
    s = open(PLUGIN, encoding="utf-8").read()
    return set(re.findall(r'"([^"]+)":\s*"', s))


def main():
    keys = plugin_keys()
    nfiles = 0
    hits = []
    for root, _dirs, files in os.walk(TPL):
        for fn in files:
            if not fn.endswith(".hbs"):
                continue
            nfiles += 1
            p = os.path.join(root, fn)
            raw = open(p, encoding="utf-8").read()
            body = COMMENT.sub("", raw)
            # 1) 属性里的裸英文
            for m in ATTRS.finditer(body):
                v = m.group(1).strip()
                if WORD.search(v) and v not in keys and not v.startswith("EMBER."):
                    hits.append((os.path.relpath(p, TPL), "ATTR", v))
            # 2) 标签之间的文本节点
            txt = TAG.sub("\n", HANDLEBARS.sub("\x00", body))
            for line in txt.split("\n"):
                s = line.replace("\x00", "").strip()
                if len(s) < 3 or not WORD.search(s):
                    continue
                if s in keys:
                    continue
                hits.append((os.path.relpath(p, TPL), "TEXT", s))
    print(f"# scanned .hbs files = {nfiles}; raw hits = {len(hits)}")
    seen = set()
    for f, kind, s in hits:
        k = (f, s)
        if k in seen:
            continue
        seen.add(k)
        print(f"{kind:5s} {f:58s} {s[:120]!r}")


if __name__ == "__main__":
    main()
