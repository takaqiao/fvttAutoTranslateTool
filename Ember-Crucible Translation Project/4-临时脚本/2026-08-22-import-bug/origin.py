# -*- coding: utf-8 -*-
"""查这两条 item 的「字符串描述」是怎么来的：英文基线里有没有 description，cn 里有没有。"""
import io, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
NAMES = ["Potion of Climbing", "Growing Thorns"]

def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d: return None
        d = d[p]
    return d

print("=== 英文基线（抽取器抽到了什么）===")
for base in sorted(os.listdir('5-其他内容/english-baseline')):
    p = os.path.join('5-其他内容/english-baseline', base, 'ember.crucible-adventure.json')
    if not os.path.exists(p): continue
    try: d = json.load(io.open(p, encoding='utf-8'))
    except Exception as e: print(f"  {base}: 读不动 {e}"); continue
    items = get(d, 'entries', 'Ember Early Access', 'items') or {}
    for n in NAMES:
        it = items.get(n)
        print(f"  {base:<34}{n:<22}{'(不在)' if it is None else '键=' + ','.join(it.keys())}")

print()
print("=== 我们的 cn（写了什么）===")
d = json.load(io.open('1-Ember汉化插件/compendium/cn/ember.crucible-adventure.json', encoding='utf-8'))
items = get(d, 'entries', 'Ember Early Access', 'items') or {}
for n in NAMES:
    it = items.get(n)
    if it is None: print(f"  {n}: (不在)"); continue
    desc = it.get('description')
    shape = '(无)' if desc is None else ('string' if isinstance(desc, str) else 'object{' + ','.join(desc.keys()) + '}')
    print(f"  {n:<22}键={','.join(it.keys())}   description 形状={shape}")
