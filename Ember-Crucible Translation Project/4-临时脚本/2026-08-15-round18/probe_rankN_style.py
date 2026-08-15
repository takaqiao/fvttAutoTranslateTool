#!/usr/bin/env python3
"""`Rank N` 全库的中文对位写法普查：EN 段 -> CN 同位段，统计中文用了哪种构词。"""
import json, os, re, sys, collections
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project'
REPOS = ['1-Ember汉化插件', '2-Crucible汉化插件']
PARA = re.compile(r'(?=<p\b|<li\b|<h[1-6]\b|<blockquote\b|<td\b)', re.I)
RANKN = re.compile(r'\bRank\s*\d')

def load(p):
    with open(p, encoding='utf-8-sig') as f: return json.load(f)

def walk(o, pre=''):
    if isinstance(o, str): yield pre, o
    elif isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, f'{pre}.{k}' if pre else k)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from walk(v, f'{pre}.{i}')

FORMS = collections.Counter()
rows = []
en_leaves = 0
aligned = 0
unaligned = 0
for repo in REPOS:
    end = os.path.join(ROOT, repo, 'compendium', 'en')
    cnd = os.path.join(ROOT, repo, 'compendium', 'cn')
    for fn in sorted(os.listdir(end)):
        if not fn.endswith('.json') or fn == '_source.json': continue
        en = dict(walk(load(os.path.join(end, fn)).get('entries', {})))
        cnp = os.path.join(cnd, fn)
        cn = dict(walk(load(cnp).get('entries', {}))) if os.path.exists(cnp) else {}
        for p, s in en.items():
            en_leaves += 1
            if not RANKN.search(s): continue
            c = cn.get(p)
            if c is None:
                unaligned += 1
                rows.append((repo, fn, p, '<无中文>', s[:0]))
                continue
            eps = [x for x in PARA.split(s) if x.strip()]
            cps = [x for x in PARA.split(c) if x.strip()]
            if len(eps) != len(cps):
                unaligned += 1
                continue
            for i, e in enumerate(eps):
                for m in RANKN.finditer(e):
                    aligned += 1
                    seg = cps[i]
                    # 抽中文里 Rank 对应的构词
                    for pat, name in [(r'阶位\s*\d', '阶位N'), (r'\d\s*级', 'N级'), (r'\bRank\s*\d', '英文残留 Rank N'),
                                      (r'\d\s*阶', 'N阶'), (r'第\s*\d\s*阶位', '第N阶位')]:
                        if re.search(pat, seg):
                            FORMS[name] += 1
                    rows.append((repo, fn, p, e.strip()[:160], seg.strip()[:160]))

print(f'英文叶数={en_leaves}  对齐命中 Rank N 处={aligned}  未对齐叶={unaligned}')
print('中文构词分布:', dict(FORMS))
print('-' * 78)
seen = set()
for r in rows:
    key = (r[0], r[2], r[3])
    if key in seen: continue
    seen.add(key)
    print(f'{r[0]}/{r[1]} :: {r[2]}')
    print(f'  EN: {r[3]}')
    print(f'  CN: {r[4]}')
