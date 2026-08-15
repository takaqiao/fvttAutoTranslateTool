#!/usr/bin/env python3
"""落盘前/后的库真实状态计数（不是影子库投影）。用法: python count_state.py <label>"""
import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = ['1-Ember汉化插件', '2-Crucible汉化插件']
label = sys.argv[1] if len(sys.argv) > 1 else 'state'

def load(p):
    with open(p, encoding='utf-8-sig') as f: return json.load(f)

def walk(o):
    if isinstance(o, str): yield o
    elif isinstance(o, dict):
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)

CN_KW = ['贾胡德', '刺客', '如果队伍成功', '阿克图里安高原', '阿克图斯高原',
         '阿克图里安', '阿克图瑞尔', '名匠瓦索洛缪', '阿克图里安工匠瓦索洛缪',
         '1 级魂印', '级魂印', '阶位 1 魂印', '魂印阶位', '魂缚阶位', '阶位']
CN_RX = {'CN 残留 Rank N': re.compile(r'\bRank\s*\d'), 'CN 阶位N': re.compile(r'阶位\s*\d')}
EN_RX = {'EN Rank N': re.compile(r'\bRank\s*\d'), 'EN Jahud': re.compile(r'\bJahud\w*'),
         'EN assassin(小写)': re.compile(r'\bassassins?\b'), 'EN Arcturelian': re.compile(r'\bArcturelian\b')}

cn = {k: 0 for k in CN_KW}
cnr = {k: 0 for k in CN_RX}
enr = {k: 0 for k in EN_RX}
cl = el = 0
for repo in REPOS:
    d = os.path.join(ROOT, repo, 'compendium', 'cn')
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.json'): continue
        for s in walk(load(os.path.join(d, fn))):
            cl += 1
            for k in CN_KW: cn[k] += s.count(k)
            for k, rx in CN_RX.items(): cnr[k] += len(rx.findall(s))
    d = os.path.join(ROOT, repo, 'compendium', 'en')
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.json') or fn == '_source.json': continue
        for s in walk(load(os.path.join(d, fn))):
            el += 1
            for k, rx in EN_RX.items(): enr[k] += len(rx.findall(s))

print(f'### {label}  中文叶数={cl}  英文叶数={el}')
for k in CN_KW: print(f'  CN {k}: {cn[k]}')
for k, v in cnr.items(): print(f'  {k}: {v}')
for k, v in enr.items(): print(f'  {k}: {v}')
