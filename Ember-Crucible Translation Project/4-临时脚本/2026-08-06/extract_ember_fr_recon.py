# -*- coding: utf-8 -*-
"""从 ember-fr（法语社区包）里抽出对我们有用的**英文侧**清单。

法语译文本身对中文项目没用；有用的是三份英文清单：
  1. 他们整理的 Ember 专名表（键是英文专名）
  2. 他们自动提取的名词/标题词表（3889 条英文键）
  3. **babele 够不到、只能靠 monkey-patch 才能翻的硬编码字符串** —— 这份最有价值，
     否则只能靠一遍遍开世界才能发现

用法：python extract_ember_fr_recon.py <ember-fr 目录> <outils 目录> <输出目录>
"""
import json
import re
import sys
from pathlib import Path

fr_dir, outils_dir, out_dir = (Path(p) for p in sys.argv[1:4])
out_dir.mkdir(parents=True, exist_ok=True)

terms = json.loads((fr_dir / "glossary/en/glossaire-ember.json").read_text(encoding="utf-8"))
(out_dir / "ember-terms-en.json").write_text(
    json.dumps(sorted(terms), ensure_ascii=False, indent=1), encoding="utf-8")

auto = json.loads((outils_dir / "glossaires/glossaire-auto.json").read_text(encoding="utf-8"))
(out_dir / "ember-auto-terms-en.json").write_text(
    json.dumps(sorted(auto), ensure_ascii=False, indent=1), encoding="utf-8")

src = (fr_dir / "scripts/enrichers-ember-fr.mjs").read_text(encoding="utf-8")
SECTION = re.compile(r'^\s{2}(\w+):\s*\[')
EN_ENTRY = re.compile(r'\{\s*en:\s*"((?:[^"\\]|\\.)*)"')
RE_ENTRY = re.compile(r'\{\s*re:\s*(/.+?/[a-z]*)\s*,')

section, buckets = None, {}
for line in src.splitlines():
    m = SECTION.match(line)
    if m:
        section = m.group(1)
        buckets.setdefault(section, [])
    if not section:
        continue
    hit = EN_ENTRY.search(line)
    if hit:
        buckets[section].append(hit.group(1))
        continue
    hit = RE_ENTRY.search(line)
    if hit:
        buckets[section].append("<正则> " + hit.group(1))

buckets = {k: v for k, v in buckets.items() if v}
(out_dir / "ember-hardcoded-strings-en.json").write_text(
    json.dumps(buckets, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"Ember 专名 {len(terms)} | 自动词表 {len(auto)} | 硬编码字符串 {sum(len(v) for v in buckets.values())} 条，分 {len(buckets)} 类")
for k, v in buckets.items():
    print(f"   {k:<24} {len(v)}")
