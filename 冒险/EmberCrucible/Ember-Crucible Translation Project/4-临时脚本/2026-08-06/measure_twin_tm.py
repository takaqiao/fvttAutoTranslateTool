#!/usr/bin/env python3
"""How much of the dnd5e twin pack (`ember.adventure`) can be filled from the
crucible pack by value-level translation memory?

Stage 0 found the journal prose byte-identical between the two campaigns. This
measures it against today's data: for every untranslated leaf of the twin, is
there a leaf anywhere in `ember.crucible-adventure` whose ENGLISH is identical
and whose Chinese exists? Those need no translator at all, only a script.
"""
from __future__ import annotations
import json
import os
import re

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
CN_DIR = os.path.join(P, "1-Ember汉化插件", "compendium", "cn")
EN_DIR = os.path.join(P, "1-Ember汉化插件", "compendium", "en")
CJK = re.compile(r'[一-鿿]')
TAG = re.compile(r'<[^>]+>')


def leaves(en, cn, path, out):
    if isinstance(en, dict):
        for k, v in en.items():
            leaves(v, cn.get(k) if isinstance(cn, dict) else None, path + [k], out)
    elif isinstance(en, list):
        for i, v in enumerate(en):
            leaves(v, cn[i] if isinstance(cn, list) and i < len(cn) else None, path + [str(i)], out)
    elif isinstance(en, str) and en.strip():
        out.append(('.'.join(path), en, cn if isinstance(cn, str) else None))


def load(d, f):
    return json.load(open(os.path.join(d, f), encoding='utf-8'))


src_en = load(EN_DIR, 'ember.crucible-adventure.json')
src_cn = load(CN_DIR, 'ember.crucible-adventure.json')
src = []
leaves(src_en.get('entries', {}), src_cn.get('entries', {}), [], src)
memory = {}
for _, e, c in src:
    if c and CJK.search(c):
        memory.setdefault(e.strip(), c)
print(f'翻译记忆条目（crucible 版已译的不同英文原文）：{len(memory)}')

tw_en = load(EN_DIR, 'ember.adventure.json')
tw_cn = load(CN_DIR, 'ember.adventure.json') if os.path.exists(os.path.join(CN_DIR, 'ember.adventure.json')) else {}
tw = []
leaves(tw_en.get('entries', {}), tw_cn.get('entries', {}), [], tw)

todo = [(p, e) for p, e, c in tw if not (c and CJK.search(c))]
hit = [(p, e) for p, e in todo if e.strip() in memory]
miss = [(p, e) for p, e in todo if e.strip() not in memory]


def chars(rows):
    return sum(len(TAG.sub(' ', e)) for _, e in rows)


print(f'\n孪生包待译        {len(todo):>7} 条 {chars(todo):>10} 字符')
print(f'  TM 可直接填      {len(hit):>7} 条 {chars(hit):>10} 字符  ({chars(hit)/max(chars(todo),1):.0%})')
print(f'  需要真翻译       {len(miss):>7} 条 {chars(miss):>10} 字符')

# 关键区分：TM 打不中的部分里，有多少只是「crucible 版同一句也还没译」——
# 那些会随着 crucible 侧推进自动被 TM 覆盖，不是孪生包自己的工作量。
present = {e.strip() for _, e, _ in src}
later = [(p, e) for p, e in miss if e.strip() in present]
own = [(p, e) for p, e in miss if e.strip() not in present]

from collections import Counter
b, bc = Counter(), Counter()
for p, e in miss:
    parts = p.split('.')
    k = 'journals' if 'journals' in parts else (parts[1] if len(parts) > 1 else parts[0])
    b[k] += 1
    bc[k] += len(TAG.sub(' ', e))
print(f'\n  其中 crucible 版有同一句英文、只是也还没译（推进 crucible 侧后 TM 自动覆盖）：'
      f'{len(later):>6} 条 {chars(later):>9} 字符')
print(f'  其中孪生包**独有**、必须单独翻译：'
      f'                            {len(own):>6} 条 {chars(own):>9} 字符')

print(f'\n  孪生包独有部分按桶：')
ob, obc = Counter(), Counter()
for p, e in own:
    parts = p.split('.')
    k = 'journals' if 'journals' in parts else (parts[1] if len(parts) > 1 else parts[0])
    ob[k] += 1
    obc[k] += len(TAG.sub(' ', e))
for k, c in obc.most_common():
    print(f'    {k:<24}{ob[k]:>7} 条 {c:>10} 字符')
