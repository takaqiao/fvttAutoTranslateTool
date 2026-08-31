# -*- coding: utf-8 -*-
"""130 个遭遇 token 覆盖名里，有多少能在现有 compendium/cn 的 name 译文里直接找到对应中文。"""
import json, io, os, sys, re
sys.stdout.reconfigure(encoding='utf-8')
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

NAMES = json.load(io.open(os.path.join(os.path.dirname(__file__), 's3_encounter_names.json'), encoding='utf-8')) \
    if os.path.exists(os.path.join(os.path.dirname(__file__), 's3_encounter_names.json')) else None

def build_tm():
    tm = {}
    for repo in ['1-Ember汉化插件', '2-Crucible汉化插件']:
        en_dir = os.path.join(PROJ, repo, 'compendium', 'en')
        cn_dir = os.path.join(PROJ, repo, 'compendium', 'cn')
        for fn in sorted(os.listdir(en_dir)):
            if not fn.endswith('.json') or fn == '_source.json': continue
            cp = os.path.join(cn_dir, fn)
            if not os.path.exists(cp): continue
            en = json.load(io.open(os.path.join(en_dir, fn), encoding='utf-8'))
            cn = json.load(io.open(cp, encoding='utf-8'))
            def walk(a, b):
                if isinstance(a, dict) and isinstance(b, dict):
                    for k in ('name', 'tokenName'):
                        if isinstance(a.get(k), str) and isinstance(b.get(k), str):
                            tm.setdefault(a[k], set()).add(b[k])
                    for k, v in a.items():
                        if k in b: walk(v, b[k])
            walk(en, cn)
    return tm

tm = build_tm()
names = NAMES or []
if not names:
    print('（先用 s3_encounter_detail.mjs 的输出手工填 s3_encounter_names.json，此处跳过）')
    sys.exit(0)
hit = [n for n in names if n in tm]
print(f'{len(hit)}/{len(names)} 个 token 覆盖名在现有 name/tokenName 译文里有直接对应')
multi = [n for n in hit if len(tm[n]) > 1]
print(f'  其中 {len(multi)} 个有多种中文写法，需人裁：', multi[:10])
for n in hit[:15]:
    print('   ', n, '->', sorted(tm[n]))
print('  未命中（要新译）:', [n for n in names if n not in tm][:40])
