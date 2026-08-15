#!/usr/bin/env python3
"""E3/E4 定案所需的英文闸 + 中文对照计数。反空转：报叶数。"""
import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = ['1-Ember汉化插件', '2-Crucible汉化插件']

def load(p):
    with open(p, encoding='utf-8-sig') as f: return json.load(f)

def walk(o, pre=''):
    if isinstance(o, str): yield pre, o
    elif isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, f'{pre}.{k}' if pre else k)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from walk(v, f'{pre}.{i}')

# ---- 英文闸（大小写敏感，逐条单独判）----
EN_GATES = {
    'Rank <数字>（大写 R）':  re.compile(r'\bRank\s*\d'),
    'rank <数字>（小写 r）':  re.compile(r'\brank\s*\d'),
    'Rank N Soulmark':        re.compile(r'\bRank\s*\d\s*Soulmark'),
    'Rank N Soulbound':       re.compile(r'\bRank\s*\d\s*Soulbound'),
    'Soulmark Rank':          re.compile(r'\bSoulmarks?\s+[Rr]ank'),
    'Soulbound Rank':         re.compile(r'\bSoulbound\s+[Rr]ank'),
    'attunement rank(小写)':  re.compile(r'\battunement\s+rank'),
    'Level <数字>':           re.compile(r'\bLevel\s*\d'),
}
CN_GATES = {
    '级魂印': None, '阶位魂印': None, '魂印阶位': None,
    '级魂缚': None, '阶位魂缚': None, '魂缚阶位': None,
    '阶位 ': None, '阶位': None, '等级': None, '层级': None,
}

en_counts = {k: 0 for k in EN_GATES}
cn_counts = {k: 0 for k in CN_GATES}
en_leaves = cn_leaves = 0
samples = []
cn_hits = []

for repo in REPOS:
    for side, counts in (('en', en_counts), ('cn', cn_counts)):
        d = os.path.join(ROOT, repo, 'compendium', side)
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json') or fn == '_source.json': continue
            for p, s in walk(load(os.path.join(d, fn))):
                if side == 'en':
                    en_leaves += 1
                    for k, rx in EN_GATES.items():
                        n = len(rx.findall(s))
                        if n:
                            counts[k] += n
                            if k in ('Rank N Soulmark', 'Rank N Soulbound', 'Soulmark Rank', 'Soulbound Rank'):
                                for m in rx.finditer(s):
                                    samples.append((repo, fn, p, k, s[max(0, m.start()-70):m.end()+70]))
                else:
                    cn_leaves += 1
                    for k in CN_GATES:
                        n = s.count(k)
                        if n:
                            counts[k] += n
                            if k in ('级魂印', '魂印阶位', '级魂缚', '魂缚阶位', '阶位魂印', '阶位魂缚'):
                                for m in re.finditer(re.escape(k), s):
                                    cn_hits.append((repo, fn, p, k, s[max(0, m.start()-50):m.end()+50]))

print(f'英文叶数={en_leaves}  中文叶数={cn_leaves}')
print('--- 英文闸 ---')
for k, v in en_counts.items(): print(f'  {k}: {v}')
print('--- 中文闸 ---')
for k, v in cn_counts.items(): print(f'  {k}: {v}')
print('--- 英文样本（Rank N Soulmark/Soulbound + Soulmark/Soulbound Rank） ---')
for r in samples: print(f'  [{r[3]}] {r[0]}/{r[1]} :: {r[2]}\n      …{r[4]}…')
print('--- 中文样本 ---')
for r in cn_hits: print(f'  [{r[3]}] {r[0]}/{r[1]} :: {r[2]}\n      …{r[4]}…')

# emberSoulbound00 的条目名
print('--- emberSoulbound00 条目名 ---')
for repo in REPOS:
    for side in ('en', 'cn'):
        d = os.path.join(ROOT, repo, 'compendium', side)
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json') or fn == '_source.json': continue
            o = load(os.path.join(d, fn))
            for p, s in walk(o):
                if 'Soulbound' in p or 'oulmark' in p or 'Soulbound' in s and len(s) < 40:
                    if re.search(r'[Ss]oul', p) or re.search(r'^[^<]{0,30}$', s) and 'Soul' in s:
                        pass
        # 直接找 entries 里键名含 Soulbound 的
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.json') or fn == '_source.json': continue
            o = load(os.path.join(d, fn))
            for k in (o.get('entries') or {}):
                if 'Soulbound' in k or 'Soulmark' in k:
                    v = o['entries'][k]
                    nm = v.get('name') if isinstance(v, dict) else v
                    print(f'  {side} {repo}/{fn} entries[{k!r}] name={nm!r}')
