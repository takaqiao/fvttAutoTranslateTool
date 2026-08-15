#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""细化：把「角色约定型误伤」从「语义歧义型误伤」里分出来。只读。"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EM, CR = os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")
CJK = re.compile(r'[\u4e00-\u9fff]')
WRAPS = [("", ""), ("(", ")"), ("（", "）"), ("[", "]"), ("【", "】")]
SEPS = " \t\r\n　-—–~·:：,，、;；/|(（[【"


def strip_tail(cn, en):
    s, e = (cn or "").strip(), (en or "").strip()
    if not e or s == e:
        return s
    for lb, rb in WRAPS:
        pat = lb + e + rb
        if len(pat) < len(s) and s.endswith(pat):
            h = s[:-len(pat)].rstrip(SEPS)
            if h and CJK.search(h):
                return h
    return s


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return json.load(f)


def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f'{p}.{k}' if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f'{p}.{i}' if p else str(i))
    elif isinstance(o, str) and o.strip():
        yield p, o


CONTAINERS = ('items', 'actions', 'effects', 'pages', 'journals', 'actors',
              'results', 'folders', 'tables', 'scenes', 'macros', 'regions',
              'levels', 'tokens', 'outcomes', 'categories', 'notes')


def subrole(path):
    segs = [s for s in path.split('.') if not s.isdigit()]
    last = segs[-1] if segs else path
    if last != 'name':
        for s in reversed(segs[:-1]):
            if s in CONTAINERS:
                return f'{s}.<key>'
        return last
    for s in reversed(segs[:-1]):
        if s in CONTAINERS:
            return f'{s}.name'
    return 'top.name'


STRUCT = {'entries', 'actors', 'items', 'actions', 'effects', 'pages', 'journals',
          'results', 'folders', 'biography', 'description', 'name', 'text',
          'label', 'public', 'private', 'appearance', 'condition', 'tokenName'}


def shape_of(path):
    return '.'.join(p for p in path.split('.') if p in STRUCT or p.isdigit())


def pairs(repo):
    en_dir, cn_dir = os.path.join(repo, 'compendium', 'en'), os.path.join(repo, 'compendium', 'cn')
    for fn in sorted(os.listdir(en_dir)):
        if not fn.endswith('.json') or fn == '_source.json':
            continue
        cnp = os.path.join(cn_dir, fn)
        if os.path.exists(cnp):
            yield fn, load(os.path.join(en_dir, fn)), load(cnp)


tm_shape, tm_plain, allleaf = defaultdict(Counter), defaultdict(Counter), []
shape_roles = defaultdict(Counter)
for repo in (EM, CR):
    for fn, en, cn in pairs(repo):
        enf, cnf = dict(walk(en)), dict(walk(cn))
        for path, src in enf.items():
            tgt = cnf.get(path)
            if tgt and CJK.search(tgt):
                tm_shape[(shape_of(path), src)][tgt] += 1
                tm_plain[src][tgt] += 1
                allleaf.append((repo, fn, path, src, tgt))
                shape_roles[shape_of(path)][subrole(path)] += 1

print('fill_missing.shape_of 的桶里，角色最杂的几个：')
for sh, rc in sorted(shape_roles.items(), key=lambda kv: -len(kv[1]))[:8]:
    print(f'  shape={sh!r:24} 叶 {sum(rc.values()):6d}  角色 {len(rc):2d} 种: '
          f'{[f"{r}×{n}" for r, n in rc.most_common(9)]}')

conv, sem, other = [], [], []
for repo, fn, path, src, tgt in allleaf:
    sk = (shape_of(path), src)
    c = Counter(tm_shape[sk]); c[tgt] -= 1
    if c[tgt] <= 0: del c[tgt]
    fell_back = not c
    cands = c
    if fell_back:
        cands = Counter(tm_plain[src]); cands[tgt] -= 1
        if cands[tgt] <= 0: del cands[tgt]
    if not cands:
        continue
    best = cands.most_common(1)[0][0]
    if best == tgt:
        continue
    row = (fn, path, subrole(path), src, tgt, best, fell_back)
    if strip_tail(best, src) == strip_tail(tgt, src):
        conv.append(row)          # 中文头相同，只差双语尾巴 = 纯角色约定误伤
    else:
        sem.append(row)

print(f'\n留一法误填合计 {len(conv)+len(sem)}')
print(f'  ① 纯角色约定误伤（中文头逐字相同，只差双语尾巴）: {len(conv)}')
print('     按角色:', Counter(r[2] for r in conv).most_common())
print(f'  ② 中文头也不同（语义歧义/上游同名等）          : {len(sem)}')
print('     按角色:', Counter(r[2] for r in sem).most_common(10))
print(f'  其中经「纯英文键」回退产生的: {sum(1 for r in conv+sem if r[6])}')
print('\n  ① 的样例：')
for r in conv[:12]:
    print(f'   {r[2]:<16} {r[1][-78:]}\n       EN={r[3][:44]!r} 实际={r[4][:34]!r} TM会填={r[5][:34]!r} 回退={r[6]}')

# 站点 A 补充：lib 里取值为「裸中文」且来源角色是裸约定角色的条目
seen = Counter(); vote_roles = defaultdict(lambda: defaultdict(Counter))
for repo in (EM, CR):
    for fn, en, cn in pairs(repo):
        enf, cnf = dict(walk(en.get('entries', {}))), dict(walk(cn.get('entries', {})))
        for path, e in enf.items():
            if not path.endswith('.name'):
                continue
            c = cnf.get(path)
            if c and CJK.search(c):
                seen[(e.strip(), c)] += 1
                vote_roles[e.strip()][c][subrole(path)] += 1
lib = {}
for (e, c), n in seen.items():
    if e not in lib or n > lib[e][1]:
        lib[e] = (c, n)
BARE = {'regions.name', 'results.name', 'levels.name', 'categories.name',
        'tokens.name', 'outcomes.name'}
poison = [(e, v) for e, (v, n) in lib.items()
          if strip_tail(v, e) == v.strip() and set(vote_roles[e][v]) <= BARE]
print(f'\n\nA 站点：lib 里「取值裸中文 且 全部票源都是裸约定角色」的条目 {len(poison)}')
print('   ', poison[:25])
