# -*- coding: utf-8 -*-
"""数一数 Ember 音景/编排的英文 label —— 它们出现在核心 PlaylistDirectory 侧栏里
（#ember-mood 的三个 <select>），宿主根元素不含 "ember"，DOM 闸整片拒绝。只读。"""
import re

P = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
t = open(P, encoding="utf-8", errors="replace").read()

# 顶层音景对象：  var X = {\n  id: "...",\n  label: "...",\n  type: "music"|"environment",
soundscapes = []
for m in re.finditer(r'(?:var|const)\s+(\w+)\s*=\s*\{\s*\n\s*id:\s*"([^"]+)",\s*\n\s*label:\s*"([^"]+)",\s*\n\s*type:\s*"(music|environment|effects)"', t):
    soundscapes.append((m.group(2), m.group(3), m.group(4), m.start()))

print("soundscape 定义数:", len(soundscapes))
from collections import Counter
print(Counter(s[2] for s in soundscapes))

# arrangements: 每个音景后面的 arrangements: { id: {label: "..."} }
arr_labels = []
for idx, (sid, slabel, stype, pos) in enumerate(soundscapes):
    end = soundscapes[idx + 1][3] if idx + 1 < len(soundscapes) else min(pos + 40000, len(t))
    body = t[pos:end]
    am = re.search(r"\n  arrangements:\s*\{", body)
    if not am:
        continue
    seg = body[am.end(): am.end() + 6000]
    for lm in re.finditer(r'label:\s*"([^"]+)"', seg):
        arr_labels.append((sid, stype, lm.group(1)))

print("arrangement label 数:", len(arr_labels))
mus = [a for a in arr_labels if a[1] in ("music", "environment")]
print("其中 music/environment（会进侧栏下拉）:", len(mus))
print("音景 label（music/environment）:", len([s for s in soundscapes if s[2] in ("music", "environment")]))
for s in soundscapes[:40]:
    print("  ", s[2], "|", s[0], "|", s[1])
print("--- arrangement 样例 ---")
for a in mus[:25]:
    print("  ", a)
