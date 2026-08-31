# -*- coding: utf-8 -*-
"""诊断: 看看 JSON 里到底有没有 中文(English) 这种 pattern"""
import json, re, os

p = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\第一本\fvtt-JournalEntry-chapter-1-c0yHRsNbVDGXKaIu.json"
with open(p, encoding="utf-8") as f:
    raw = f.read()

print(f"File size: {len(raw)} chars")

# 抽前 500 字符看 sample
print("\n--- sample ---")
print(raw[:500])

# 找所有 "汉字(英文" 模式片段
# 用更宽松的正则
patt = re.compile(r"[一-鿿·•\-]{2,8}\s*\(\s*[A-Z][a-zA-Z\s\'\-\d]{2,40}\s*\)")
hits = patt.findall(raw)
print(f"\nZH(EN) hits: {len(hits)}")
for h in hits[:20]:
    print(f"  {h}")

# 英文(汉字)
patt2 = re.compile(r"[A-Z][a-zA-Z\s\'\-\d]{2,40}\s*\(\s*[一-鿿·•\-]{1,12}\s*\)")
hits2 = patt2.findall(raw)
print(f"\nEN(ZH) hits: {len(hits2)}")
for h in hits2[:20]:
    print(f"  {h}")

# 看 HTML 字段, 因为 JSON 里 < > 都被 escape 成 < 之类
# 试试直接看一些常用术语
for term in ["Yoh", "Souran", "Hwanggot", "Lighthouse", "Sun Wing", "Aldanar", "Phoenix"]:
    n = raw.count(term)
    print(f"  raw count '{term}' = {n}")
