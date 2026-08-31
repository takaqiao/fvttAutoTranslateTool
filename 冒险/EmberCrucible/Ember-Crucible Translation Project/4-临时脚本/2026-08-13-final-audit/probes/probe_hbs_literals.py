# -*- coding: utf-8 -*-
"""
探针 2：ember 0.6.0 的 .hbs 模板里，**裸英文字面量**（既不是 {{localize}}、也不是插值）
有哪些？它们只能靠 ember-hardcoded-cn.mjs 的 EXACT 表在运行时替换。
逐条比对 EXACT / PREFIXED / PATTERNS 是否覆盖。

只读。
假阳性：
  - 单个字母、纯符号、CSS 值会被误当文本 → 已过滤长度<3 与无字母项。
  - `{{#if}}` 之类块内文本仍是真实可见文本，保留。
  - 属性里的 aria-label/data-tooltip 单独抓一遍（模块也翻这两个属性）。
"""
import os, re, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TPL = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\templates"
MOD = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
src = open(MOD, encoding="utf-8").read()

def table(name):
    m = re.search(r"const %s = \{(.*?)\n\};" % name, src, re.S)
    return set(re.findall(r'"([^"]+)":\s*"', m.group(1))) if m else set()

COVERED = set()
for t in ["ATTUNEMENTS", "LANGUAGES", "KNOWLEDGE", "MOODS", "EXACT", "RESULTS",
          "CALENDAR_MONTHS", "CALENDAR_DAYS", "CALENDAR_DAY_ABBR"]:
    COVERED |= table(t)

TAG = re.compile(r"<[^>]*>", re.S)
HANDLEBARS = re.compile(r"\{\{[^}]*\}\}", re.S)
ATTR = re.compile(r'(aria-label|data-tooltip|data-tooltip-text|title|placeholder|alt)\s*=\s*"([^"]*)"')

rows_text, rows_attr = [], []
nfiles = 0
for dp, dn, fn in os.walk(TPL):
    for f in fn:
        if not f.endswith(".hbs"):
            continue
        nfiles += 1
        p = os.path.join(dp, f)
        s = open(p, encoding="utf-8").read()
        rel = os.path.relpath(p, TPL)

        # 属性
        for m in ATTR.finditer(s):
            v = m.group(2).strip()
            if not v or "{{" in v:
                continue
            if not re.search(r"[A-Za-z]{3}", v):
                continue
            rows_attr.append((rel, m.group(1), v, v in COVERED))

        # 文本节点
        body = ATTR.sub("", TAG.sub("\n", s))
        body = HANDLEBARS.sub("\n", body)
        for line in body.split("\n"):
            t = line.strip()
            if len(t) < 3 or not re.search(r"[A-Za-z]{3}", t):
                continue
            if re.fullmatch(r"[\w\-./]+", t) and "." in t:  # 路径
                continue
            rows_text.append((rel, t, t in COVERED))

print(f"扫描 {nfiles} 个 .hbs")
print("\n=== 裸英文文本节点（未被替换表覆盖）===")
seen = set()
for rel, t, cov in rows_text:
    if cov or t in seen:
        continue
    seen.add(t)
    print(f"  {rel:55s} | {t}")
print("\n=== 裸英文属性（未被替换表覆盖）===")
seen = set()
for rel, a, v, cov in rows_attr:
    if cov or (a, v) in seen:
        continue
    seen.add((a, v))
    print(f"  {rel:55s} | {a} = {v}")
