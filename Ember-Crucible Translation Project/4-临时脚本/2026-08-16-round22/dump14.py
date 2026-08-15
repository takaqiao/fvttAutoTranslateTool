#!/usr/bin/env python3
"""把 14 条差集叶的 旧英文/新英文/中文 三方全文落盘，供逐条人读。
判据不复写：直接 import scan_en_drift 的 load_json / leaves / norm。
"""
import json, os, sys, importlib.util

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location(
    'sed', os.path.join(P, '3-常用脚本', 'qa', 'scan_en_drift.py'))
sed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sed)

REPO = os.path.join(P, '2-Crucible汉化插件')
BASE = os.path.join(P, '5-其他内容', 'english-baseline', 'crucible-cn-0.8.9.1-shipped-en')

TARGETS = json.load(open(sys.argv[1], encoding='utf-8'))

cache = {}
def get(pack):
    if pack in cache:
        return cache[pack]
    o, n, c = {}, {}, {}
    sed.leaves(sed.load_json(os.path.join(BASE, pack)).get('entries', {}), [], o)
    sed.leaves(sed.load_json(os.path.join(REPO, 'compendium', 'en', pack)).get('entries', {}), [], n)
    sed.leaves(sed.load_json(os.path.join(REPO, 'compendium', 'cn', pack)).get('entries', {}), [], c)
    cache[pack] = (o, n, c)
    return cache[pack]

out = []
missing = 0
for pack, path in TARGETS:
    o, n, c = get(pack)
    if path not in o or path not in n:
        missing += 1
    out.append({'pack': pack, 'path': path,
                'old_en': o.get(path), 'new_en': n.get(path), 'cn': c.get(path),
                'en_changed': sed.norm(o.get(path, '')) != sed.norm(n.get(path, ''))})

print(f'读入目标 {len(TARGETS)} 条 · 三方全取到 {len(TARGETS)-missing} 条 · '
      f'其中英文确实变过 {sum(1 for r in out if r["en_changed"])} 条 · '
      f'中文非空 {sum(1 for r in out if r["cn"])} 条')
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_way_14.json')
json.dump(out, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('->', dst)
