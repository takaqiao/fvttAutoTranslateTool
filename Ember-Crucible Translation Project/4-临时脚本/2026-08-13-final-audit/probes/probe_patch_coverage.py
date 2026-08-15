# -*- coding: utf-8 -*-
"""
probe_patch_coverage.py  —  「补丁覆盖面不足」类判据（只读，不写库）

判据抽象
--------
运行时汉化补丁（ember-hardcoded-cn.mjs）由三部分组成：
  (1) 一道**闸**决定哪些渲染表面会被遍历（patchRenderedApplications 的 ember/DialogV2 闸、
      patchEnrichers 的 pattern 正则闸、patchCrucibleConfig 的 ["languages","knowledge"] 闸、
      patchCalendarNames 的 ["months","days"] 闸）；
  (2) 一个**遍历器** translateNode（文本节点 + 5 个属性名）；
  (3) 一张**查表** EXACT / PREFIXED / PATTERNS。
只要上游在同一个表面族里发射的字符串，闸没放行、遍历器没走到、或查表没有键，
它就永远是英文。已确认实例：DialogV2 分支只取 .window-title 就 return（闸+遍历器）。

本探针把这条抽象拆成四个可机械化的子判据：
  A. 闸-模板：ember 的 .hbs 模板里的可见英文（文本节点 / aria-label / placeholder / title /
     alt / data-tooltip），按「渲染它的应用能否过闸」+「串是否在 EXACT」两维分类。
  B. 闸-配置：ember 写进 crucible.CONFIG 的分组 vs 补丁遍历的分组。
  C. 表-前缀：上游用 `Xxx: ${name}` 形式拼的增强器标签 vs PREFIXED 表里的前缀。
  D. 遍历器-属性：translateNode 认的属性名 vs 模板里实际出现的文案类属性名。

已知假阳性模式（人工复核时必须逐条排除）
  * 模板里 `{{localize '...'}}` / `{{#if}}` 之类是 i18n，不算硬编码；
  * `data-tooltip-direction="LEFT"`、`alt="{{...}}"` 这类是枚举值/插值，不是文案；
  * 编辑态（GM 编辑 journal page 的表单）与游玩态权重不同，需人工判断是否值得报；
  * ember.mjs 里的字符串字面量有大量是 CSS 类名 / dataset key / 日志文本，
    本探针只取 hbs 模板，不去猜 mjs 里的字面量（噪声太大），mjs 侧靠人工定点核。
"""
import json
import os
import re
import sys

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EMBER = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
CN = os.path.join(ROOT, "1-Ember汉化插件", "scripts", "ember-hardcoded-cn.mjs")

src = open(CN, encoding="utf-8").read()

# ---- 从补丁源码里解析 EXACT 的键（只取 "xxx": "yyy" 的左半边） ----
exact_block = src[src.index("const EXACT = {"): src.index("/** 掷骰结果档位")]
EXACT_KEYS = set(re.findall(r'"((?:[^"\\]|\\.)*)"\s*:', exact_block))
PREFIXES = set(re.findall(r'\{\s*en:\s*"([^"]+)"', src))
ATTR_LIST = re.search(r'for \(const attr of \[([^\]]+)\]\)', src).group(1)
ATTRS = set(re.findall(r'"([^"]+)"', ATTR_LIST))

print("=== 补丁自述 ===")
print("EXACT 键数:", len(EXACT_KEYS))
print("PREFIXED 前缀:", sorted(PREFIXES))
print("translateNode 认的属性:", sorted(ATTRS))

# ---- 子判据 C：上游 `Xxx: ${...}` 形式的增强器标签 ----
mjs = open(os.path.join(EMBER, "scripts", "ember.mjs"), encoding="utf-8").read()
lab = sorted(set(re.findall(r'innerHTML\s*=\s*`([A-Z][A-Za-z ]+): \$\{', mjs)))
print("\n=== C. 上游增强器拼的英文前缀 vs PREFIXED ===")
for p in lab:
    print(f"  {p:<12} {'OK' if p in PREFIXES else '*** 缺（前缀永远是英文）***'}")

# ---- 子判据 B：ember 写进 crucible.CONFIG 的分组 ----
groups = sorted(set(re.findall(r'crucible\.CONFIG\.(\w+)[\.\[ ]', mjs)))
patched = set(re.findall(r'\[\["(\w+)", [A-Z]+\], \["(\w+)", [A-Z]+\]\]', src)[0]) if re.search(
    r'\[\["(\w+)", [A-Z]+\], \["(\w+)", [A-Z]+\]\]', src) else set()
print("\n=== B. crucible.CONFIG 分组 ===")
print("  ember 触碰的分组:", groups)
print("  补丁遍历的分组:", sorted(patched))

# ---- 子判据 A + D：模板扫描 ----
TAG = re.compile(r"<[^>]+>")
HB = re.compile(r"\{\{[^}]*\}\}")
ATTR_RE = re.compile(r'\b(aria-label|placeholder|title|alt|data-tooltip|data-tooltip-text|value|label)\s*=\s*"([^"]*)"')
ENGLISH = re.compile(r"[A-Za-z]{2,}")

rows = []
for dirpath, _dirs, files in os.walk(os.path.join(EMBER, "templates")):
    for f in sorted(files):
        if not f.endswith(".hbs"):
            continue
        rel = os.path.relpath(os.path.join(dirpath, f), EMBER).replace("\\", "/")
        text = open(os.path.join(dirpath, f), encoding="utf-8").read()

        # 文案属性
        for attr, val in ATTR_RE.findall(text):
            v = HB.sub("", val).strip()
            if not v or not ENGLISH.search(v):
                continue
            if v.upper() == v and len(v) <= 6:      # LEFT / UP 之类枚举值
                continue
            rows.append((rel, "attr:" + attr, v))

        # 文本节点
        for chunk in TAG.split(text):
            for line in chunk.splitlines():
                s = HB.sub("", line).strip()
                if len(s) < 3 or not ENGLISH.search(s):
                    continue
                if re.fullmatch(r"[\s\.\|,:;\-–—/&()\[\]0-9]*", s):
                    continue
                rows.append((rel, "text", s))

print("\n=== A/D. 模板里可见的硬编码英文（{{localize}} 已剔除） ===")
print("命中条数:", len(rows))
miss_attr = sorted({a.split(":", 1)[1] for _f, a, _v in rows if a.startswith("attr:")} - ATTRS - {"value", "label"})
print("D. 模板用到但 translateNode 不认的文案属性:", miss_attr)

by_file = {}
for rel, kind, val in rows:
    by_file.setdefault(rel, []).append((kind, val))
print()
for rel in sorted(by_file):
    hits = by_file[rel]
    uncovered = [(k, v) for k, v in hits if v not in EXACT_KEYS]
    if not uncovered:
        continue
    print(f"-- {rel}  ({len(uncovered)}/{len(hits)} 不在 EXACT)")
    for k, v in uncovered:
        print(f"     [{k}] {v}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_coverage_raw.json")
json.dump([{"file": r, "kind": k, "text": v, "inEXACT": v in EXACT_KEYS} for r, k, v in rows],
          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nraw ->", out)
