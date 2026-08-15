#!/usr/bin/env python3
"""按关键词取上下文（EN 段 + CN 段），不依赖块号。"""
import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPO = os.path.join(ROOT, '1-Ember汉化插件')

def load(p):
    with open(p, encoding='utf-8-sig') as f: return json.load(f)

def walk(o, pre=''):
    if isinstance(o, str): yield pre, o
    elif isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, f'{pre}.{k}' if pre else k)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from walk(v, f'{pre}.{i}')

PARA = re.compile(r'(?=<p\b|<li\b|<h[1-6]\b|<blockquote\b)', re.I)

CASES = [
    ('E1', 'Ember Early Access.actors.Sadri Zhalimorne.biography.private', ['阿克图里安高原'], ['Plateau']),
    ('E2', 'Ember Early Access.actors.Constructed Companion.biography.private', ['名匠'], ['Arcturelian']),
    ('E2ref', 'Ember Early Access.actors.Woven Construct.biography.private', ['阿克图里安工匠'], ['Arcturelian']),
    ('E3', 'Ember Early Access.journals.Unfinished Business.pages.Shine On.text', ['级魂印'], ['Rank 1 Soulmark', 'Rank']),
    ('E4', 'Ember Early Access.journals.Unfinished Business.pages.The Old Flame.text', ['Rank 1', 'Rank 2'], ['Rank 1', 'Rank 2']),
]

for pack in ['ember.adventure.json', 'ember.crucible-adventure.json']:
    en = dict(walk(load(os.path.join(REPO, 'compendium', 'en', pack)).get('entries', {})))
    cn = dict(walk(load(os.path.join(REPO, 'compendium', 'cn', pack)).get('entries', {})))
    print('#' * 78)
    print(f'PACK {pack}  EN叶={len(en)} CN叶={len(cn)}')
    for tag, path, cnkw, enkw in CASES:
        if path not in cn:
            continue
        print('=' * 78)
        print(f'{tag} {path}')
        cps = [p for p in PARA.split(cn[path]) if p.strip()]
        eps = [p for p in PARA.split(en.get(path, '')) if p.strip()]
        print(f'  CN段={len(cps)} EN段={len(eps)} 段数相同={len(cps)==len(eps)}')
        for i, p in enumerate(cps):
            if any(k in p for k in cnkw):
                print(f'  --CN段{i}: {p}')
                if len(cps) == len(eps):
                    print(f'  --EN段{i}: {eps[i]}')
        for i, p in enumerate(eps):
            if any(k in p for k in enkw) and not any(k in cps[i] if len(cps)==len(eps) else '' for k in cnkw):
                print(f'  ~~EN段{i}: {p}')
                if len(cps) == len(eps):
                    print(f'  ~~CN段{i}: {cps[i]}')
